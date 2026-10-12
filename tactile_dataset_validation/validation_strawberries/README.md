# Strawberry acquisition-day classification

This workflow predicts the acquisition day (1, 2, 3, or 4), not fruit identity. Each CSV becomes one 272-feature spectral vector. Fruit IDs are grouped before the fixed 18/6/6 split, keeping all days and sweeps for a fruit in the same partition.

```powershell
python run.py prepare
python run.py train
python run.py plot
```

Artifacts are written to `manifests/`, `derived/`, `models/`, and `results/`.
