# Experimental active-touch material classification

This directory implements experimental active-touch material classification. The model predicts FLEX, PLA, Silicone12, Silicone18, Silicone40, or TPU from one 272-feature spectral vector per CSV. PLA100 and all PLA infill folders are mapped to PLA only after the original material/infill-folder and force grouping has determined the split.

Run from this directory with the base Python environment that includes scikit-learn:

```powershell
python run.py prepare
python run.py train
python run.py plot
```

Artifacts are written directly to `manifests/`, `derived/`, `models/`, and `results/`.
