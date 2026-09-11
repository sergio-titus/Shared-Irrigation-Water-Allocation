# ============================================================
# WEATHER DATA AUDIT
# MULTI-CLIMATE WEATHER DOWNLOAD + AUDIT
#
# PURPOSE
# -------
# 1. Download one consistent weather source for three climates.
# 2. Build AquaCrop-compatible daily weather tables.
# 3. Calculate daily FAO-56 reference ET0.
# 4. Audit missing data and climate contrasts.
#
# IMPORTANT
# ---------
# NO irrigation controller is evaluated here.
# NO PPO training is performed here.
# NO Equal/Priority results are generated here.
#
# Weather source:
# NASA POWER Daily API, Agroclimatology community.
#
# Candidate climates:
#   Tunis   = Mediterranean
#   Niamey  = hot semi-arid
#   Cotonou = humid tropical/coastal
# ============================================================

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("results") / "multiclimate_extension" / "weather"
RAW_DIR = OUTPUT_DIR / "raw"
PROCESSED_DIR = OUTPUT_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# NASA POWER daily meteorology is available from 1981 onward.
# We deliberately download a much longer period than we will
# eventually use so that the train/test split can be frozen
# AFTER weather QC, but BEFORE any controller evaluation.
START_YEAR = 1981
END_YEAR = 2025


SITES = {
    "Tunis": {
        "country": "Tunisia",
        "latitude": 36.8065,
        "longitude": 10.1815,
        "climate_role": "Mediterranean",
    },
    "Niamey": {
        "country": "Niger",
        "latitude": 13.5116,
        "longitude": 2.1254,
        "climate_role": "Hot semi-arid",
    },
    "Cotonou": {
        "country": "Benin",
        "latitude": 6.3703,
        "longitude": 2.3912,
        "climate_role": "Humid tropical/coastal",
    },
}


# Variables needed for AquaCrop weather + FAO-56 ET0.
#
# T2M_MAX           daily maximum air temperature, degC
# T2M_MIN           daily minimum air temperature, degC
# T2M               daily mean air temperature, degC
# T2MDEW            daily mean dew-point temperature, degC
# PRECTOTCORR       corrected precipitation, mm/day
# ALLSKY_SFC_SW_DWN surface shortwave radiation, MJ/m2/day
# WS2M              wind speed at 2 m, m/s
# PS                surface pressure, kPa
POWER_PARAMETERS = [
    "T2M_MAX",
    "T2M_MIN",
    "T2M",
    "T2MDEW",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
    "PS",
]


POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)


# NASA POWER normally uses -999 for missing values.
MISSING_SENTINELS = [-999, -999.0, -9999, -9999.0]


# ============================================================
# 2. NASA POWER DOWNLOAD
# ============================================================

def download_power_json(site_name, site):
    """
    Download NASA POWER daily data for one point.
    """

    start_date = f"{START_YEAR}0101"
    end_date = f"{END_YEAR}1231"

    params = {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "AG",
        "longitude": site["longitude"],
        "latitude": site["latitude"],
        "start": start_date,
        "end": end_date,
        "format": "JSON",
        "time-standard": "LST",
    }

    print()
    print("=" * 78)
    print(f"DOWNLOADING: {site_name}")
    print(
        f"Coordinates : "
        f"{site['latitude']:.4f}, {site['longitude']:.4f}"
    )
    print(f"Climate role: {site['climate_role']}")
    print("=" * 78)

    response = requests.get(
        POWER_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()
    content = response.json()

    raw_file = RAW_DIR / f"{site_name.lower()}_power_raw.json"

    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

    print(f"Saved raw response: {raw_file}")

    return content


# ============================================================
# 3. JSON -> DATAFRAME
# ============================================================

def power_json_to_dataframe(content):
    """
    Convert NASA POWER parameter dictionary into one daily DataFrame.
    """

    parameter_block = content["properties"]["parameter"]

    frames = []

    for parameter_name in POWER_PARAMETERS:

        if parameter_name not in parameter_block:
            raise KeyError(
                f"NASA POWER response does not contain "
                f"{parameter_name}"
            )

        series = pd.Series(
            parameter_block[parameter_name],
            name=parameter_name,
            dtype="float64",
        )

        series.index = pd.to_datetime(
            series.index,
            format="%Y%m%d",
        )

        frames.append(series)

    df = pd.concat(frames, axis=1)
    df.index.name = "Date"

    df = df.sort_index()

    for sentinel in MISSING_SENTINELS:
        df = df.replace(sentinel, np.nan)

    return df


# ============================================================
# 4. FAO-56 REFERENCE ET0
# ============================================================

def saturation_vapour_pressure(temp_c):
    """
    Saturation vapour pressure, kPa.
    FAO-56 equation.
    """
    return 0.6108 * np.exp(
        (17.27 * temp_c) / (temp_c + 237.3)
    )


def estimate_elevation_from_pressure(pressure_kpa):
    """
    Approximate elevation (m) from mean atmospheric pressure.

    Used only for the clear-sky radiation calculation.
    """

    p = float(np.nanmean(pressure_kpa))

    if not np.isfinite(p) or p <= 0:
        return 0.0

    z = 44330.0 * (
        1.0 - (p / 101.325) ** (1.0 / 5.255)
    )

    return max(-100.0, min(z, 5000.0))


def extraterrestrial_radiation(latitude_deg, dates):
    """
    Daily extraterrestrial radiation Ra in MJ m^-2 day^-1.
    FAO-56 equations 21-25.
    """

    lat_rad = math.radians(latitude_deg)

    julian_day = dates.dayofyear.to_numpy(dtype=float)

    dr = 1.0 + 0.033 * np.cos(
        2.0 * np.pi * julian_day / 365.0
    )

    solar_declination = (
        0.409
        * np.sin(
            2.0 * np.pi * julian_day / 365.0 - 1.39
        )
    )

    x = (
        -np.tan(lat_rad)
        * np.tan(solar_declination)
    )

    # Numerical protection for high-latitude values.
    x = np.clip(x, -1.0, 1.0)

    sunset_hour_angle = np.arccos(x)

    solar_constant = 0.0820  # MJ m^-2 min^-1

    ra = (
        (24.0 * 60.0 / np.pi)
        * solar_constant
        * dr
        * (
            sunset_hour_angle
            * np.sin(lat_rad)
            * np.sin(solar_declination)
            +
            np.cos(lat_rad)
            * np.cos(solar_declination)
            * np.sin(sunset_hour_angle)
        )
    )

    return ra


def calculate_fao56_et0(df, latitude_deg):
    """
    Calculate daily FAO-56 Penman-Monteith reference ET0.

    Inputs
    ------
    Tmin/Tmax/Tmean : degC
    dew point       : degC
    Rs              : MJ m^-2 day^-1
    wind            : m/s at 2 m
    pressure        : kPa

    Output
    ------
    ReferenceET : mm/day
    """

    result = df.copy()

    tmax = result["T2M_MAX"].to_numpy(float)
    tmin = result["T2M_MIN"].to_numpy(float)
    tmean = result["T2M"].to_numpy(float)
    tdew = result["T2MDEW"].to_numpy(float)

    rs = result["ALLSKY_SFC_SW_DWN"].to_numpy(float)
    u2 = result["WS2M"].to_numpy(float)
    pressure = result["PS"].to_numpy(float)

    # --------------------------------------------------------
    # Saturation vapour pressure
    # --------------------------------------------------------

    es_tmax = saturation_vapour_pressure(tmax)
    es_tmin = saturation_vapour_pressure(tmin)

    es = (es_tmax + es_tmin) / 2.0

    # Actual vapour pressure from dew-point temperature.
    ea = saturation_vapour_pressure(tdew)

    # --------------------------------------------------------
    # Vapour pressure curve slope
    # --------------------------------------------------------

    es_tmean = saturation_vapour_pressure(tmean)

    delta = (
        4098.0
        * es_tmean
        / ((tmean + 237.3) ** 2)
    )

    # --------------------------------------------------------
    # Psychrometric constant
    # --------------------------------------------------------

    gamma = 0.000665 * pressure

    # --------------------------------------------------------
    # Radiation
    # --------------------------------------------------------

    ra = extraterrestrial_radiation(
        latitude_deg,
        result.index,
    )

    elevation_m = estimate_elevation_from_pressure(
        pressure
    )

    # Clear-sky solar radiation.
    rso = (
        0.75 + 2e-5 * elevation_m
    ) * ra

    # Net shortwave radiation.
    albedo = 0.23

    rns = (1.0 - albedo) * rs

    # Net longwave radiation.
    sigma = 4.903e-9

    tmax_k = tmax + 273.16
    tmin_k = tmin + 273.16

    # Protect against invalid negative vapour pressure.
    ea_safe = np.maximum(ea, 0.0)

    # Rs/Rso should not become unphysically huge.
    rs_rso = np.divide(
        rs,
        rso,
        out=np.zeros_like(rs),
        where=rso > 1e-9,
    )

    rs_rso = np.clip(rs_rso, 0.0, 1.0)

    rnl = (
        sigma
        * (
            (tmax_k ** 4 + tmin_k ** 4)
            / 2.0
        )
        * (
            0.34
            - 0.14 * np.sqrt(ea_safe)
        )
        * (
            1.35 * rs_rso
            - 0.35
        )
    )

    rn = rns - rnl

    # Daily soil heat flux is assumed zero.
    g = 0.0

    # --------------------------------------------------------
    # FAO-56 Penman-Monteith
    # --------------------------------------------------------

    numerator = (
        0.408
        * delta
        * (rn - g)
        +
        gamma
        * (
            900.0
            / (tmean + 273.0)
        )
        * u2
        * (es - ea)
    )

    denominator = (
        delta
        +
        gamma
        * (
            1.0
            +
            0.34 * u2
        )
    )

    et0 = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-12,
    )

    # Negative daily reference ET is not physically meaningful.
    et0 = np.maximum(et0, 0.0)

    result["ReferenceET"] = et0
    result["Ra"] = ra
    result["EstimatedElevation_m"] = elevation_m

    return result


# ============================================================
# 5. AQUACROP-COMPATIBLE WEATHER TABLE
# ============================================================

def build_aquacrop_weather(df):
    """
    Build AquaCrop-style daily weather DataFrame.

    Expected columns:
    MinTemp
    MaxTemp
    Precipitation
    ReferenceET
    Date
    """

    aq = pd.DataFrame(
        {
            "MinTemp": df["T2M_MIN"],
            "MaxTemp": df["T2M_MAX"],
            "Precipitation": df["PRECTOTCORR"],
            "ReferenceET": df["ReferenceET"],
        },
        index=df.index,
    )

    aq = aq.reset_index()

    aq = aq[
        [
            "MinTemp",
            "MaxTemp",
            "Precipitation",
            "ReferenceET",
            "Date",
        ]
    ]

    return aq


# ============================================================
# 6. QUALITY CONTROL
# ============================================================

def audit_weather(site_name, site, df):
    """
    Audit temporal completeness and physical plausibility.
    """

    expected_dates = pd.date_range(
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-12-31",
        freq="D",
    )

    audit_df = df.reindex(expected_dates)

    audit = {
        "Site": site_name,
        "Country": site["country"],
        "Climate role": site["climate_role"],
        "Latitude": site["latitude"],
        "Longitude": site["longitude"],
        "Start date": df.index.min(),
        "End date": df.index.max(),
        "Expected days": len(expected_dates),
        "Available rows": len(df),
        "Missing calendar rows": int(
            len(expected_dates.difference(df.index))
        ),
    }

    required = [
        "T2M_MAX",
        "T2M_MIN",
        "T2M",
        "T2MDEW",
        "PRECTOTCORR",
        "ALLSKY_SFC_SW_DWN",
        "WS2M",
        "PS",
        "ReferenceET",
    ]

    for col in required:
        audit[f"Missing {col}"] = int(
            audit_df[col].isna().sum()
        )

    # Basic physical checks.
    audit["Tmax<Tmin days"] = int(
        (
            audit_df["T2M_MAX"]
            <
            audit_df["T2M_MIN"]
        ).sum()
    )

    audit["Negative rainfall days"] = int(
        (
            audit_df["PRECTOTCORR"] < 0
        ).sum()
    )

    audit["Negative ET0 days"] = int(
        (
            audit_df["ReferenceET"] < 0
        ).sum()
    )

    return audit


# ============================================================
# 7. YEAR-LEVEL CLIMATE CHARACTERIZATION
# ============================================================

def build_annual_summary(site_name, site, df):
    """
    Summarize each complete weather year.
    """

    tmp = df.copy()

    tmp["Year"] = tmp.index.year

    rows = []

    for year, group in tmp.groupby("Year"):

        expected_days = (
            366
            if pd.Timestamp(year=year, month=12, day=31).is_leap_year
            else 365
        )

        required_cols = [
            "T2M_MAX",
            "T2M_MIN",
            "T2M",
            "PRECTOTCORR",
            "ReferenceET",
        ]

        complete = (
            len(group) == expected_days
            and not group[required_cols].isna().any().any()
        )

        rows.append(
            {
                "Site": site_name,
                "Country": site["country"],
                "Climate": site["climate_role"],
                "Year": int(year),
                "Days": len(group),
                "CompleteYear": bool(complete),
                "AnnualRain_mm":
                    group["PRECTOTCORR"].sum(),
                "AnnualET0_mm":
                    group["ReferenceET"].sum(),
                "MeanTemp_C":
                    group["T2M"].mean(),
                "MeanTmax_C":
                    group["T2M_MAX"].mean(),
                "MeanTmin_C":
                    group["T2M_MIN"].mean(),
                "MeanSolar_MJ_m2_day":
                    group[
                        "ALLSKY_SFC_SW_DWN"
                    ].mean(),
                "MeanWind_m_s":
                    group["WS2M"].mean(),
                "Rain_ET0_ratio":
                    (
                        group["PRECTOTCORR"].sum()
                        /
                        group["ReferenceET"].sum()
                        if group["ReferenceET"].sum() > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. MAY-OCTOBER SUMMARY
# ============================================================

def build_growing_season_summary(site_name, site, df):
    """
    Characterize the May 1 - Oct 31 window currently used
    by the irrigation simulation.

    This is diagnostic only.
    """

    tmp = df[
        (
            (df.index.month >= 5)
            &
            (df.index.month <= 10)
        )
    ].copy()

    tmp["Year"] = tmp.index.year

    rows = []

    for year, group in tmp.groupby("Year"):

        rows.append(
            {
                "Site": site_name,
                "Country": site["country"],
                "Climate": site["climate_role"],
                "Year": int(year),
                "SeasonRain_mm":
                    group["PRECTOTCORR"].sum(),
                "SeasonET0_mm":
                    group["ReferenceET"].sum(),
                "SeasonMeanTemp_C":
                    group["T2M"].mean(),
                "SeasonRain_ET0_ratio":
                    (
                        group["PRECTOTCORR"].sum()
                        /
                        group["ReferenceET"].sum()
                        if group["ReferenceET"].sum() > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print()
    print("#" * 78)
    print("WEATHER DATA AUDIT")
    print("MULTI-CLIMATE WEATHER DATASET AUDIT")
    print("#" * 78)

    all_audits = []
    all_annual = []
    all_seasonal = []

    for idx, (site_name, site) in enumerate(SITES.items()):

        content = download_power_json(
            site_name,
            site,
        )

        df = power_json_to_dataframe(content)

        df = calculate_fao56_et0(
            df,
            latitude_deg=site["latitude"],
        )

        # ----------------------------------------------------
        # Save processed full meteorological table
        # ----------------------------------------------------

        processed_file = (
            PROCESSED_DIR
            / f"{site_name.lower()}_weather_full.csv"
        )

        df.reset_index().to_csv(
            processed_file,
            index=False,
        )

        # ----------------------------------------------------
        # Save AquaCrop-compatible table
        # ----------------------------------------------------

        aq = build_aquacrop_weather(df)

        aq_file = (
            PROCESSED_DIR
            / f"{site_name.lower()}_aquacrop_weather.csv"
        )

        aq.to_csv(
            aq_file,
            index=False,
        )

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        audit = audit_weather(
            site_name,
            site,
            df,
        )

        all_audits.append(audit)

        annual = build_annual_summary(
            site_name,
            site,
            df,
        )

        all_annual.append(annual)

        seasonal = build_growing_season_summary(
            site_name,
            site,
            df,
        )

        all_seasonal.append(seasonal)

        print()
        print(f"{site_name}:")
        print(f"  rows       : {len(df):,}")
        print(
            f"  period     : "
            f"{df.index.min().date()} -> "
            f"{df.index.max().date()}"
        )
        print(
            f"  rain/year  : "
            f"{annual['AnnualRain_mm'].mean():.1f} mm"
        )
        print(
            f"  ET0/year   : "
            f"{annual['AnnualET0_mm'].mean():.1f} mm"
        )
        print(
            f"  mean temp  : "
            f"{annual['MeanTemp_C'].mean():.2f} °C"
        )
        print(
            f"  May-Oct rain/ET0 ratio: "
            f"{seasonal['SeasonRain_ET0_ratio'].mean():.3f}"
        )

        print(f"  AquaCrop file: {aq_file}")

        # Be polite to the public API.
        if idx < len(SITES) - 1:
            time.sleep(2)

    # ========================================================
    # Combined outputs
    # ========================================================

    audit_table = pd.DataFrame(all_audits)

    annual_table = pd.concat(
        all_annual,
        ignore_index=True,
    )

    seasonal_table = pd.concat(
        all_seasonal,
        ignore_index=True,
    )

    audit_file = OUTPUT_DIR / "weather_quality_audit.csv"
    annual_file = OUTPUT_DIR / "annual_climate_summary.csv"
    seasonal_file = (
        OUTPUT_DIR
        / "may_october_climate_summary.csv"
    )

    audit_table.to_csv(audit_file, index=False)
    annual_table.to_csv(annual_file, index=False)
    seasonal_table.to_csv(seasonal_file, index=False)

    # ========================================================
    # Climate-level summary
    # ========================================================

    complete_annual = annual_table[
        annual_table["CompleteYear"]
    ].copy()

    climate_summary = (
        complete_annual
        .groupby(
            ["Site", "Country", "Climate"],
            as_index=False,
        )
        .agg(
            CompleteYears=("Year", "count"),
            MeanAnnualRain_mm=(
                "AnnualRain_mm",
                "mean",
            ),
            SDAnnualRain_mm=(
                "AnnualRain_mm",
                "std",
            ),
            MeanAnnualET0_mm=(
                "AnnualET0_mm",
                "mean",
            ),
            MeanTemperature_C=(
                "MeanTemp_C",
                "mean",
            ),
            MeanRain_ET0_ratio=(
                "Rain_ET0_ratio",
                "mean",
            ),
        )
    )

    climate_file = (
        OUTPUT_DIR
        / "climate_contrast_summary.csv"
    )

    climate_summary.to_csv(
        climate_file,
        index=False,
    )

    # ========================================================
    # Identify years complete in ALL THREE climates
    # ========================================================

    complete_year_sets = []

    for site_name in SITES:

        years = set(
            annual_table.loc[
                (
                    annual_table["Site"] == site_name
                )
                &
                annual_table["CompleteYear"],
                "Year",
            ].astype(int)
        )

        complete_year_sets.append(years)

    common_complete_years = sorted(
        set.intersection(*complete_year_sets)
    )

    common_file = (
        OUTPUT_DIR
        / "common_complete_years.json"
    )

    with open(common_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sites": list(SITES.keys()),
                "common_complete_years":
                    common_complete_years,
                "n_common_complete_years":
                    len(common_complete_years),
            },
            f,
            indent=2,
        )

    # ========================================================
    # Terminal report
    # ========================================================

    print()
    print("=" * 78)
    print("WEATHER QUALITY AUDIT")
    print("=" * 78)

    display_cols = [
        "Site",
        "Climate role",
        "Expected days",
        "Available rows",
        "Missing calendar rows",
        "Missing T2M_MAX",
        "Missing T2M_MIN",
        "Missing PRECTOTCORR",
        "Missing ReferenceET",
        "Tmax<Tmin days",
    ]

    print(
        audit_table[display_cols]
        .to_string(index=False)
    )

    print()
    print("=" * 78)
    print("CLIMATE CONTRAST SUMMARY")
    print("=" * 78)

    print(
        climate_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print()
    print("=" * 78)
    print("COMMON COMPLETE YEARS")
    print("=" * 78)

    print(
        f"Number of years complete in all climates: "
        f"{len(common_complete_years)}"
    )

    print(common_complete_years)

    print()
    print("=" * 78)
    print("WEATHER DATA AUDIT COMPLETE")
    print("=" * 78)

    print()
    print("Saved:")
    print(f"  {audit_file}")
    print(f"  {annual_file}")
    print(f"  {seasonal_file}")
    print(f"  {climate_file}")
    print(f"  {common_file}")

    print()
    print("IMPORTANT:")
    print(
        "No controller performance has been evaluated."
    )
    print(
        "No PPO training has been performed."
    )
    print(
        "Train/validation/test years have NOT yet been selected."
    )
    print(
        "The next step is to inspect this weather audit and "
        "freeze the multi-climate experimental split."
    )


if __name__ == "__main__":
    main()