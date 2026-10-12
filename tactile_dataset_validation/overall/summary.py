"""Write the four-row classification table and four-panel test confusion matrix."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VALIDATION = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
TASKS = [
    ("experimental", "validation_experimental", "random_forest", "Experimental active touch"),
    ("simulation", "validation_simulation", "random_forest", "Simulated active touch"),
    ("strawberries", "validation_strawberries", "random_forest", "Strawberries"),
    ("slippage", "validation_slippage", "histogram_gradient_boosting", "Slippage"),
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def matrix(path: Path) -> tuple[list[str], np.ndarray]:
    values = rows(path)
    labels = [row["true_label"] for row in values]
    result = np.asarray([[float(row[f"predicted_{label}"]) for label in labels] for row in values])
    return labels, result


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    matrices = []
    for task, directory, method, title in TASKS:
        metric_rows = rows(VALIDATION / directory / "results" / "metrics.csv")
        model = next(row for row in metric_rows if row["split"] == "test" and row["method"] == method)
        baseline = next(row for row in metric_rows if row["split"] == "test" and row["method"] == "majority_class")
        extra = json.loads((VALIDATION / directory / "manifests" / "prepare_metadata.json").read_text(encoding="utf-8"))
        summary.append({
            "subset": task,
            "prediction_target": "material" if task in ("experimental", "simulation") else "acquisition_day" if task == "strawberries" else "motion_state",
            "model": method,
            "test_records_or_signals": extra["records_by_split"]["test"],
            "test_windows": extra["windows_by_split"]["test"] if extra["windows_by_split"] else "not_applicable",
            "test_macro_f1": model["macro_f1"],
            "test_accuracy": model["accuracy"],
            "test_balanced_accuracy": model["balanced_accuracy"],
            "majority_macro_f1": baseline["macro_f1"],
            "majority_accuracy": baseline["accuracy"],
            "majority_balanced_accuracy": baseline["balanced_accuracy"],
        })
        labels, values = matrix(VALIDATION / directory / "results" / f"test_{method}_confusion_matrix.csv")
        matrices.append((title, labels, values))
    write(RESULTS / "classification_results_table.csv", summary)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for axis, (title, labels, values) in zip(axes.flat, matrices):
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                          color="white" if values[i, j] >= 0.55 else "black")
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title(title)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Row-normalized proportion", shrink=0.8)
    figure = RESULTS / "four_panel_test_confusion_matrices"
    fig.savefig(figure.with_suffix(".png"), dpi=300)
    fig.savefig(figure.with_suffix(".pdf"))
    plt.close(fig)
    (RESULTS / "summary_metadata.json").write_text(json.dumps({"tasks": [task[0] for task in TASKS], "matrix": "row-normalized test confusion matrix"}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
