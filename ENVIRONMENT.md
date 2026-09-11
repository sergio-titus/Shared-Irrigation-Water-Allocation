# Computational environment

The Stable-Baselines3 metadata embedded in each final PPO checkpoint records the following training environment:

- Operating system: Windows 10 (`Windows-10-10.0.26200-SP0`)
- Python: 3.11.16
- Stable-Baselines3: 2.9.0
- PyTorch: 2.14.0+cpu
- GPU enabled: False
- NumPy: 2.4.6
- Cloudpickle: 3.1.2
- Gymnasium: 1.3.0

These values were extracted directly from `system_info.txt` inside the final PPO checkpoint archives.

The exact historical versions of AquaCrop-OSPy, pandas, SciPy, matplotlib, requests, and psutil were not stored in the PPO checkpoint metadata available for this repository. They are therefore listed without invented version pins in `requirements.txt`.

For archival reproducibility, the repository includes the final model checkpoints and the CSV/JSON outputs used in the manuscript. A future tagged release can additionally include a fully exported environment lock file if the original environment remains available.
