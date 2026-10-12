# Simulated active-touch material classification

This workflow keeps the lexicographically first path for each SHA-256 group in `metadata/path_mapping.csv`, leaving 35 unique simulated signals. It predicts the six merged material labels from a 272-feature spectral vector per unique signal.

```powershell
python run.py prepare
python run.py train
python run.py plot
```

Artifacts are written to `manifests/`, `derived/`, `models/`, and `results/`.
