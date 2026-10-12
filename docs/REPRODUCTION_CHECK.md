# Release verification

Checked on 12 October 2026 in a separate local copy of this repository, using
Windows and Python 3.13.15. The manuscript and original validation directory were
not modified during packaging or verification.

The environment used NumPy 2.1.3, scikit-learn 1.6.1, joblib 1.4.2, and
Matplotlib 3.10.0, matching the pinned direct dependencies. SciPy 1.18.1 and
threadpoolctl 3.7.0 were installed as dependencies. `python -m pip check` passed.

| Check | Outcome |
| --- | --- |
| Loading and command-line tests | All four tests passed, including help commands launched from an unrelated working directory |
| Loading one raw recording per subset | Passed: 4,096 experimental, 4,000 simulated, 9,600 strawberry, and 1,151,120 slippage rows |
| Audit of bundled artifacts without the raw dataset | Passed for all four tasks |
| Refitting all models with `run_all.py --from-features` | Metrics, prediction tables, and test confusion matrices matched the bundled reference values exactly |
| Predictions from the bundled `.joblib` models | Test predictions matched the saved predictions exactly for all four tasks |
| Re-extracting features from the existing raw dataset | Feature arrays, labels, recording IDs, recording/window manifests, splits, and preparation metadata matched the bundled artifacts exactly |
| Summary and figure generation | Succeeded within the copied repository; the single-row PDF was also visually checked |
| Relative file links and paths | README file links resolve; no local user paths or old project-directory names remain in the package |

Recreated feature shapes were `(1750, 272)` for experimental active touch,
`(35, 272)` for simulation, `(6000, 272)` for strawberries, and `(52473, 30)`
for slippage. The checks used the existing recordings only.

The saved reference artifacts retain their original run metadata. Reproduction
was performed in a separate copy, so it did not replace the bundled models or
reference results with newly trained versions.

The main README records the reserved dataset DOI; the dataset is not yet public.
Uploading to GitHub and execution on Linux/macOS were not part of this check.
