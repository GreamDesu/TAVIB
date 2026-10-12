# Slippage motion-state classification

This workflow labels 400-sample windows as no motion, slippage, or sliding from encoder velocities, but uses only the six accelerometer channels as input. Windows require at least 360 samples with the same target label. Each retained window has 30 RMS and spectral-band features.

```powershell
python run.py prepare
python run.py train
python run.py plot
```

Artifacts are written to `manifests/`, `derived/`, `models/`, and `results/`.
