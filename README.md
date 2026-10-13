# TAVIB: Tactile sensing time series data for robotic object interaction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23319688.svg)](https://doi.org/10.5281/zenodo.23319688)

Loading examples and technical-validation code accompanying the data descriptor
*Tactile sensing time series data for robotic object interaction*.

This repository contains the four classification workflows, fixed split manifests,
extracted features, trained models, predictions, metrics, and confusion matrices.
The raw recording files are distributed separately.

**Code release:** [v1.0.0](https://github.com/GreamDesu/TAVIB/releases/tag/v1.0.0). The DOI badge above identifies the code archive.

**Dataset DOI:** [10.5281/zenodo.23319677](https://doi.org/10.5281/zenodo.23319677).
Download `TAVIB_dataset.zip` from this Zenodo record.

## Contents

| Path | Contents |
| --- | --- |
| [`examples/load_recordings.py`](examples/load_recordings.py) | NumPy loading example for one recording from each subset |
| [`tactile_dataset_validation/`](tactile_dataset_validation/) | Feature extraction, training, evaluation, and saved artifacts |
| [`tactile_dataset_validation/overall/results/`](tactile_dataset_validation/overall/results/) | Combined results, figures, and artifact audit |
| [`docs/DATASET_README.md`](docs/DATASET_README.md) | Dataset layout, column schemas, units, and duplicate-file notes |
| [`tests/`](tests/) | Loading and command-line checks |
| [`docs/REPRODUCTION_CHECK.md`](docs/REPRODUCTION_CHECK.md) | Recorded checks of the release and reproduced results |
| [`CITATION.cff`](CITATION.cff) | Citation metadata for the code |

## Installation

Use Python 3.13, the version used to check this release. No GPU is required.
Run the commands below from the repository root.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv/bin/activate` on Linux/macOS. Then install the pinned dependencies:

```sh
python -m pip install -r requirements.txt
```

## Reproduce the classification results without the raw recordings

The bundled feature arrays and split manifests are sufficient to refit all four
models and regenerate their predictions, metrics, and confusion matrices:

```sh
python tactile_dataset_validation/run_all.py --from-features
```

The script fits on training data for validation, then refits on training plus
validation data for the final test. It also regenerates the combined result table
and checks the saved artifacts. Output files are regenerated in place.

To inspect the bundled artifacts without training:

```sh
python tactile_dataset_validation/overall/audit.py
```

This audit checks feature shapes, split rules, and required artifact files; it
does not check raw recording bytes or independently recalculate predictions.

## Load the raw recordings and reproduce feature extraction

Download and extract the separate dataset so that the layout is:

```text
repository/
  examples/
  tactile_dataset_validation/
  TAVIB_dataset/
    data/
    metadata/
    checksums/
    README.md
    LICENSE
```

The raw dataset directory is excluded by `.gitignore`. Use the dataset DOI above
to download the recordings.

If the dataset is elsewhere, edit
[`tactile_dataset_validation/dataset_config.json`](tactile_dataset_validation/dataset_config.json).
Relative paths are resolved from `tactile_dataset_validation/`, not the shell's
working directory. The default is:

```json
{"dataset_root": "../TAVIB_dataset"}
```

Load one CSV from each subset:

```sh
python examples/load_recordings.py
```

The example preserves all rows and the original column order, skips one header
row for active-touch files, and treats slippage files as headerless. It checks
column counts and finite values and constructs slippage timestamps from sample
indices at 8 kHz. An optional `--dataset-root PATH` overrides the configuration
for this example only. Loading the example itself requires NumPy only.

Reproduce the full workflow, including features extracted from the raw CSVs:

```sh
python tactile_dataset_validation/run_all.py
```

## Tasks and reference test results

| Subset | Prediction target | Model | Test set | Macro-F1 | Accuracy |
| --- | --- | --- | --- | ---: | ---: |
| Experimental active touch | Six material classes | Random forest | 350 recordings | 1.0000 | 1.0000 |
| Simulated active touch | Six material classes | Random forest | 6 unique signals | 1.0000 | 1.0000 |
| Strawberries | Acquisition day, 1-4 | Random forest | 1,200 recordings from 6 fruits | 0.9958 | 0.9958 |
| Slippage | No motion, slippage, sliding | Histogram gradient boosting | 15,454 windows from 12 recordings | 0.8374 | 0.8607 |

Full-precision scores and majority-class baselines are in
[`classification_results_table.csv`](tactile_dataset_validation/overall/results/classification_results_table.csv).

- **Experimental:** 30/10/10 recordings per material/infill-force group for
  training/validation/test. PLA infill variants share the PLA target class.
- **Simulation:** the 1,750 files contain 35 unique signals. After deduplication,
  1-3 N supplies 23 training signals, 4 N supplies 6 validation signals, and
  5 N supplies 6 test signals.
- **Strawberries:** 18/6/6 fruit identifiers for training/validation/test; all
  days and sweeps for one fruit remain together. Acquisition day is not an
  independently measured ripeness label.
- **Slippage:** 17/6/12 complete recordings after removing the duplicated
  aluminum/0-degree medium-speed copy. Material-orientation combinations appear
  in both training and test sets. Encoder velocities define labels; only the
  six accelerometer channels provide model inputs. Metrics are calculated for
  each recording and averaged equally.

The fixed random seed is 42. Active-touch classification uses 272 spectral and
RMS features per recording. Contact-motion classification uses 30 features per
400-sample window, retaining windows with at least 360 samples sharing one valid
label. Model settings are documented in each task's `config.json` and implemented
in `pipeline.py`; the JSON files describe the fixed experiment, rather than
configure a hyperparameter search.

For slippage, multiple prediction rows share a recording ID. Within each split
and method, their order follows the retained windows in `manifests/windows.csv`;
the recording ID alone is not a unique window identifier.

These results demonstrate classification within the stated splits. Perfect
material scores do not establish generalization to unseen objects or arbitrary
conditions. The dataset can support research on tactile encoders for future VLA
systems; it contains no synchronized vision-language-action demonstrations, and
these experiments do not measure VLA policy performance.

## Figures and individual tasks

The saved figures are in each task's `results/figures/` directory. Regenerate the
combined table and four-panel figure from saved metrics without fitting models:

```sh
python tactile_dataset_validation/overall/summary.py
```

Generate the alternative, single-row vector PDF:

```sh
python tactile_dataset_validation/overall/plot_paper_figure.py
```

The PDF is written to `tactile_dataset_validation/overall/results/confusion_matrices_row.pdf`.

For a single task, use `prepare`, `train`, or `plot`, for example:

```sh
python tactile_dataset_validation/validation_slippage/run.py train
```

Only `prepare` requires the raw dataset. `train` uses the bundled features and
`plot` uses the saved confusion matrices. Original software versions are recorded
in each task's `results/run_metadata.json`.

## Checks

```sh
python -m unittest discover -s tests -v
```

## Citation and license

Citation metadata for code version 1.0.0 is provided in [`CITATION.cff`](CITATION.cff).
Cite the code using DOI [10.5281/zenodo.23319688](https://doi.org/10.5281/zenodo.23319688),
and cite the dataset using DOI [10.5281/zenodo.23319677](https://doi.org/10.5281/zenodo.23319677).
Cite the associated data descriptor when published.

The active-touch data were previously deposited in the
[original OSF dataset](https://doi.org/10.17605/OSF.IO/G4AE8), associated with
[Sandykbayeva et al. (2026)](https://doi.org/10.1038/s41598-026-54766-9).
The slippage and sliding recordings originate from
[Massalim et al. (2020)](https://doi.org/10.3390/s20154121).
See [dataset provenance and formats](docs/DATASET_README.md) for details.

The code is released under the [MIT License](LICENSE). The separately distributed
raw dataset uses [CC0 1.0](docs/DATASET_LICENSE.txt).

The acquisition firmware and simulation-generation software from the source
studies are not included; this repository provides CSV loading and the technical
validation performed for the data descriptor.
