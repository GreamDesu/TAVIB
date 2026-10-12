# Classification experiment summary

The directory contains the combined summaries for the four classification tasks. Run from `tactile_dataset_validation/`:

```powershell
python run_all.py --from-features
python overall/summary.py
python overall/audit.py
```

The four-row test table is `results/classification_results_table.csv`, the four-panel row-normalized test confusion-matrix figure is `results/four_panel_test_confusion_matrices.png` and `.pdf`, and the audit is `results/classification_audit.json`.

`run_all.py` already runs the summary and artifact audit. Run the two standalone
scripts only when regenerating summaries or inspecting existing artifacts.
Omit `--from-features` to also extract features from the raw recordings.

`python overall/plot_paper_figure.py` creates the alternative single-row vector
figure at `results/confusion_matrices_row.pdf`.
