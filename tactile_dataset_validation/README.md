# TAVIB classification validation

This directory contains the four fixed classification experiments used for the
manuscript's Technical Validation section. See the [repository README](../README.md)
for installation, dataset placement, tasks, and interpretation of the results.

| Directory | Prediction target |
| --- | --- |
| `validation_experimental/` | Six material classes |
| `validation_simulation/` | Six material classes after deduplicating signals |
| `validation_strawberries/` | Acquisition day (1-4), grouped by fruit |
| `validation_slippage/` | No motion, slippage, sliding |

From the repository root, refit the models using the bundled features:

```sh
python tactile_dataset_validation/run_all.py --from-features
```

Omit `--from-features` to also extract features from the raw CSV files.
`dataset_config.json` specifies the raw dataset directory; relative paths are
resolved from this directory. The default is `../TAVIB_dataset`.

Each task includes:

- `config.json`: description of the fixed model, labels, and split rule.
- `manifests/`: recording identities, split assignments, source hashes, and
  feature-extraction metadata; slippage also includes retained window indices.
- `derived/features.npz`: numeric features, labels, and recording identifiers.
- `models/`: final estimators fitted on training plus validation data.
- `results/`: predictions, metrics, confusion matrices, and runtime metadata.

`overall/summary.py` creates the combined result table and figure.
`overall/audit.py` checks the bundled features, splits, and required artifacts.
Both also run automatically at the end of `run_all.py`.

The `config.json` files document the experiment; model settings are fixed in
`pipeline.py`. To change the raw data location, edit only `dataset_config.json`.
