"""Verify classification splits and required result artifacts."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VALIDATION = Path(__file__).resolve().parent.parent
DATASET_CONFIG_PATH = VALIDATION / "dataset_config.json"


TASKS = {
    "experimental": ("validation_experimental", "random_forest"),
    "simulation": ("validation_simulation", "random_forest"),
    "strawberries": ("validation_strawberries", "random_forest"),
    "slippage": ("validation_slippage", "histogram_gradient_boosting"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    checks: list[dict] = []
    for task, (directory, method) in TASKS.items():
        root = VALIDATION / directory
        manifests = root / "manifests"
        results = root / "results"
        record_rows = rows(manifests / "records.csv")
        payload = np.load(root / "derived" / "features.npz", allow_pickle=False)
        assert len(payload["X"]) == len(payload["y"]) == len(payload["record_ids"])
        assert (root / "models" / f"{task}_final_model.joblib").is_file()
        assert (results / "predictions.csv").is_file()
        metric_rows = rows(results / "metrics.csv")
        assert {(row["split"], row["method"]) for row in metric_rows} == {
            ("validation", method), ("validation", "majority_class"),
            ("test", method), ("test", "majority_class"),
        }
        if task == "experimental":
            groups = defaultdict(list)
            for row in record_rows:
                groups[(row["source_material"], row["force_n"])].append(row)
            assert len(record_rows) == 1750 and len(groups) == 35
            assert all(Counter(row["split"] for row in group) == {"train": 30, "validation": 10, "test": 10} for group in groups.values())
            assert payload["X"].shape == (1750, 272)
        elif task == "simulation":
            assert Counter(row["split"] for row in record_rows) == {"train": 23, "validation": 6, "test": 6}
            assert len({row["sha256"] for row in record_rows}) == 35
            assert payload["X"].shape == (35, 272)
        elif task == "strawberries":
            fruit_splits = defaultdict(set)
            for row in record_rows:
                fruit_splits[row["sample_id"]].add(row["split"])
            assert len(record_rows) == 6000 and len(fruit_splits) == 30
            assert all(len(values) == 1 for values in fruit_splits.values())
            assert Counter(row["split"] for row in record_rows) == {"train": 3600, "validation": 1200, "test": 1200}
            assert payload["X"].shape == (6000, 272)
        else:
            assert Counter(row["split"] for row in record_rows) == {"train": 17, "validation": 6, "test": 12}
            assert "data/slippage/aluminum/Orientation_0/speed_medium/data.csv" not in {row["recording_id"] for row in record_rows}
            conditions = defaultdict(set)
            for row in record_rows:
                conditions[(row["source_material"], row["orientation_deg"])].add(row["split"])
            assert all({"train", "test"}.issubset(values) for values in conditions.values())
            windows = rows(manifests / "windows.csv")
            assert len(windows) == len(payload["X"])
            assert all(int(row["end_index_exclusive"]) - int(row["start_index"]) == 400 for row in windows)
            assert {row["target_label"] for row in windows}.issubset({"0", "1", "2"})
        checks.append({"task": task, "status": "PASS", "feature_shape": list(payload["X"].shape), "record_count": len(record_rows)})
    output = {"status": "PASS", "checks": checks, "dataset_config": DATASET_CONFIG_PATH.name,
              "source_file_index": "metadata/file_index.csv"}
    destination = Path(__file__).resolve().parent / "results"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "classification_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
