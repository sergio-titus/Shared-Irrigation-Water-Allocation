from pathlib import Path
import json, time, hashlib, gc
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import multiclimate_optimization_core as core

ROOT = Path('results') / 'multiclimate_extension'
SPEC_FILE = ROOT / 'training_protocol' / 'multiclimate_ppo_training_spec_frozen.json'
SELECTION_FILE = ROOT / 'horizon_reconfirmation' / 'ppo_horizon_selection_horizon_selection.json'
OUT = ROOT / 'final_training'
MODELS = OUT / 'models'
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = OUT / 'ppo_final_training_final_training_summary.csv'
INTEGRITY_FILE = OUT / 'ppo_final_training_final_training_integrity.json'
PROGRESS_FILE = OUT / 'ppo_final_training_completed_final_seeds.json'


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


spec = load_json(SPEC_FILE)
selection = load_json(SELECTION_FILE)

DEV_YEARS = [int(y) for y in spec['development_years']]
TEST_YEARS = [int(y) for y in spec['protected_final_test_years']]
CLIMATES = list(spec['climates'])
FINAL_SEEDS = [int(x) for x in spec['final_seeds']]
HP = dict(spec['ppo_hyperparameters'])
REWARD = dict(spec['reward_configuration'])
TRAINING_SCARCITY = float(spec['training_scarcity'])
HORIZON = int(selection['selected_horizon'])

# Strict protocol checks
assert DEV_YEARS == list(range(1984, 2019))
assert TEST_YEARS == list(range(2019, 2026))
assert CLIMATES == ['Tunis', 'Niamey', 'Cotonou']
assert FINAL_SEEDS == [1103, 2207, 3319, 4421, 5527, 6637, 7741, 8849, 9967, 11071]
assert HORIZON == 102400
assert selection['final_test_used'] is False
assert selection['ppo_reward_used_for_selection'] is False
assert abs(TRAINING_SCARCITY - 0.40) < 1e-12
assert not (set(DEV_YEARS) & set(TEST_YEARS))
assert not any(y >= 2019 for y in DEV_YEARS)
assert HORIZON % int(HP['n_steps']) == 0

# Freeze reward into core
core.configure_reward(
    daily_stress_weight=REWARD['daily_stress_weight'],
    terminal_yield_weight=REWARD['terminal_yield_weight'],
    terminal_jain_weight=REWARD['terminal_jain_weight'],
    terminal_wp_weight=REWARD['terminal_wp_weight'],
    terminal_loss_weight=REWARD['terminal_loss_weight'],
    terminal_worst_weight=REWARD['terminal_worst_weight'],
)
if abs(float(core.SCARCITY_FRACTION) - TRAINING_SCARCITY) > 1e-12:
    raise RuntimeError('Core scarcity does not match frozen 53E training scarcity.')


def build_model(seed):
    env = core.MultiClimateSharedWaterEnv(
        years=DEV_YEARS,
        climates=CLIMATES,
        seed=int(seed),
    )
    env = Monitor(env)
    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=float(HP['learning_rate']),
        n_steps=int(HP['n_steps']),
        batch_size=int(HP['batch_size']),
        n_epochs=int(HP['n_epochs']),
        gamma=float(HP['gamma']),
        gae_lambda=float(HP['gae_lambda']),
        clip_range=float(HP['clip_range']),
        ent_coef=float(HP['ent_coef']),
        vf_coef=float(HP['vf_coef']),
        max_grad_norm=float(HP['max_grad_norm']),
        policy_kwargs={'net_arch': list(HP['net_arch'])},
        verbose=0,
        seed=int(seed),
        device='auto',
    )
    return model, env


def load_completed():
    if not PROGRESS_FILE.exists():
        return set()
    d = load_json(PROGRESS_FILE)
    return {int(x) for x in d.get('completed_final_seeds', [])}


def save_completed(done):
    PROGRESS_FILE.write_text(json.dumps({
        'completed_final_seeds': sorted(int(x) for x in done),
        'selected_horizon': HORIZON,
        'final_test_opened': False,
        'seed_selection': False,
    }, indent=2), encoding='utf-8')


def load_summary():
    if not SUMMARY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SUMMARY_FILE)
    if not df.empty:
        df['Seed'] = df['Seed'].astype(int)
    return df


def main():
    start_all = time.perf_counter()
    print('\n' + '#' * 78)
    print('FINAL PPO ENSEMBLE TRAINING')
    print('FINAL MULTI-CLIMATE PPO TRAINING')
    print('#' * 78)
    print('\nDevelopment years       : 1984-2018')
    print('Protected final test    : 2019-2025')
    print(f'Climates                : {CLIMATES}')
    print('Training scarcity       : 40%')
    print(f'Selected horizon        : {HORIZON:,}')
    print(f'Final seeds             : {FINAL_SEEDS}')
    print('Hyperparameters retuned : NO')
    print('Reward retuned          : NO')
    print('Seed/model selection    : NO')
    print('Final test used         : NO')

    completed = load_completed()
    summary = load_summary()

    for i, seed in enumerate(FINAL_SEEDS, start=1):
        print('\n' + '=' * 78)
        print(f'FINAL SEED {i}/{len(FINAL_SEEDS)}: {seed}')
        print('=' * 78)

        base = MODELS / f'multiclimate_ppo_seed_{seed}'
        zip_path = Path(str(base) + '.zip')
        existing = summary[summary['Seed'] == seed] if not summary.empty else pd.DataFrame()

        if seed in completed and zip_path.exists() and not existing.empty:
            print('Status                  : already complete; SKIPPED')
            continue

        if not summary.empty:
            summary = summary[summary['Seed'] != seed].copy()

        np.random.seed(seed)
        torch.manual_seed(seed)
        model, env = build_model(seed)

        print('Training...')
        t0 = time.perf_counter()
        model.learn(total_timesteps=HORIZON, reset_num_timesteps=False, progress_bar=False)
        runtime_min = (time.perf_counter() - t0) / 60.0
        actual = int(model.num_timesteps)
        if actual != HORIZON:
            raise RuntimeError(f'Seed {seed}: expected {HORIZON} steps, got {actual}.')

        model.save(str(base))
        if not zip_path.exists():
            raise RuntimeError(f'Model not saved: {zip_path}')

        size_mb = zip_path.stat().st_size / (1024.0 * 1024.0)
        row = pd.DataFrame([{
            'Seed': seed,
            'Selected_horizon': HORIZON,
            'Actual_timesteps': actual,
            'Training_runtime_min': runtime_min,
            'Model_size_MB': size_mb,
            'Model_SHA256': sha256_file(zip_path),
            'Training_year_start': DEV_YEARS[0],
            'Training_year_end': DEV_YEARS[-1],
            'Number_of_training_years': len(DEV_YEARS),
            'Number_of_climates': len(CLIMATES),
            'Training_climate_years': len(DEV_YEARS) * len(CLIMATES),
            'Training_scarcity_pct': TRAINING_SCARCITY * 100.0,
            'Final_test_used': False,
            'Seed_selected': False,
        }])
        summary = pd.concat([summary, row], ignore_index=True).sort_values('Seed').reset_index(drop=True)
        summary.to_csv(SUMMARY_FILE, index=False)
        completed.add(seed)
        save_completed(completed)

        print(f'Actual timesteps        : {actual:,}')
        print(f'Runtime                 : {runtime_min:.2f} min')
        print(f'Model size              : {size_mb:.3f} MB')
        print(f'Model saved             : {zip_path}')
        print('Final test used         : NO')

        env.close()
        del model, env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = pd.read_csv(SUMMARY_FILE)
    summary['Seed'] = summary['Seed'].astype(int)
    if len(summary) != len(FINAL_SEEDS):
        raise RuntimeError(f'Expected 10 final models, found {len(summary)}.')
    if set(summary['Seed']) != set(FINAL_SEEDS):
        raise RuntimeError('Final seed set differs from frozen manifest.')
    if not (summary['Actual_timesteps'].astype(int) == HORIZON).all():
        raise RuntimeError('One or more models has the wrong horizon.')
    if summary['Final_test_used'].astype(bool).any():
        raise RuntimeError('Final-test leakage flag detected.')

    for seed in FINAL_SEEDS:
        p = Path(str(MODELS / f'multiclimate_ppo_seed_{seed}') + '.zip')
        if not p.exists():
            raise RuntimeError(f'Missing final model: {p}')

    total_min = float(summary['Training_runtime_min'].sum())
    mean_min = float(summary['Training_runtime_min'].mean())
    sd_min = float(summary['Training_runtime_min'].std(ddof=1))
    mean_size = float(summary['Model_size_MB'].mean())

    integrity = {
        'milestone': '53G',
        'status': 'COMPLETE',
        'selected_horizon': HORIZON,
        'development_years': DEV_YEARS,
        'protected_final_test_years': TEST_YEARS,
        'climates': CLIMATES,
        'training_climate_years': len(DEV_YEARS) * len(CLIMATES),
        'training_scarcity': TRAINING_SCARCITY,
        'final_seeds': FINAL_SEEDS,
        'number_of_final_models': len(summary),
        'all_final_seeds_retained': True,
        'model_selection_among_final_seeds': False,
        'seed_selection': False,
        'reward_retuned': False,
        'hyperparameters_retuned': False,
        'horizon_changed_after_53F': False,
        'equal_evaluated': False,
        'priority_evaluated': False,
        'final_test_used': False,
        'final_test_opened': False,
        'training_spec_sha256': sha256_file(SPEC_FILE),
        'horizon_selection_sha256': sha256_file(SELECTION_FILE),
        'total_training_runtime_min': total_min,
        'mean_training_runtime_min': mean_min,
        'sd_training_runtime_min': sd_min,
        'mean_model_size_MB': mean_size,
    }
    INTEGRITY_FILE.write_text(json.dumps(integrity, indent=2), encoding='utf-8')

    print('\n' + '#' * 78)
    print('FINAL PPO ENSEMBLE TRAINING COMPLETE')
    print('#' * 78)
    print('\nFINAL TRAINING SUMMARY\n')
    cols = ['Seed', 'Selected_horizon', 'Actual_timesteps', 'Training_runtime_min', 'Model_size_MB', 'Final_test_used']
    print(summary[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print('\n' + '=' * 78)
    print('AGGREGATE')
    print('=' * 78)
    print(f'Final models retained      : {len(summary)}')
    print(f'Selected horizon/model     : {HORIZON:,}')
    print(f'Total final-training steps : {HORIZON * len(FINAL_SEEDS):,}')
    print(f'Mean runtime/model         : {mean_min:.2f} min')
    print(f'SD runtime/model           : {sd_min:.2f} min')
    print(f'Cumulative training time   : {total_min / 60.0:.2f} h')
    print(f'Mean model size            : {mean_size:.3f} MB')

    print('\n' + '=' * 78)
    print('FINAL-TEST INTEGRITY')
    print('=' * 78)
    print('Development climate-years : 105')
    print('2019-2025 used             : NO')
    print('Equal evaluated            : NO')
    print('Priority evaluated         : NO')
    print('Model selection            : NO')
    print('Seed selection             : NO')
    print('All 10 retained            : YES')
    print('Final test remains unopened: YES')
    print(f'Current-script elapsed     : {(time.perf_counter() - start_all) / 60.0:.2f} min')

    print('\nSaved:')
    print(f'  {SUMMARY_FILE}')
    print(f'  {INTEGRITY_FILE}')
    print(f'  {PROGRESS_FILE}')
    print(f'  {MODELS}')
    print('\nNEXT: generate the protected 2019-2025 reference library only after these final PPO models are frozen, then evaluate Equal, Priority and PPO on the common final test.')


if __name__ == '__main__':
    main()
