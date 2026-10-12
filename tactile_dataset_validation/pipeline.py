"""Reproducible tactile classification workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


VALIDATION_ROOT = Path(__file__).resolve().parent
DATASET_CONFIG_PATH = VALIDATION_ROOT / "dataset_config.json"


def load_dataset_root() -> Path:
    if not DATASET_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset configuration: {DATASET_CONFIG_PATH.name}")
    config = json.loads(DATASET_CONFIG_PATH.read_text(encoding="utf-8"))
    configured_path = config.get("dataset_root")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("dataset_config.json must contain a non-empty dataset_root string")
    candidate = Path(configured_path).expanduser()
    root = (candidate if candidate.is_absolute() else VALIDATION_ROOT / candidate).resolve()
    return root


DATASET_ROOT = load_dataset_root()

ACTIVE_CLASSES = ["FLEX", "PLA", "Silicone12", "Silicone18", "Silicone40", "TPU"]
SLIPPAGE_CLASSES = ["no_motion", "slippage", "sliding"]
TASK_DIRS = {
    "experimental": VALIDATION_ROOT / "validation_experimental",
    "simulation": VALIDATION_ROOT / "validation_simulation",
    "strawberries": VALIDATION_ROOT / "validation_strawberries",
    "slippage": VALIDATION_ROOT / "validation_slippage",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    directory: Path
    classes: list[str]
    estimator_name: str


def task_spec(name: str) -> TaskSpec:
    if name not in TASK_DIRS:
        raise ValueError(f"Unknown task: {name}")
    if name == "slippage":
        return TaskSpec(name, TASK_DIRS[name], SLIPPAGE_CLASSES, "histogram_gradient_boosting")
    if name == "strawberries":
        return TaskSpec(name, TASK_DIRS[name], ["1", "2", "3", "4"], "random_forest")
    return TaskSpec(name, TASK_DIRS[name], ACTIVE_CLASSES, "random_forest")


def ensure_dirs(spec: TaskSpec) -> None:
    for relative in ("manifests", "derived", "models", "results", "results/figures"):
        (spec.directory / relative).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_file_index() -> list[dict[str, str]]:
    return read_csv_rows(DATASET_ROOT / "metadata" / "file_index.csv")


def load_checksums() -> dict[str, str]:
    return {row["path"]: row["sha256"] for row in read_csv_rows(DATASET_ROOT / "checksums" / "sha256_manifest.csv")}


def numeric_trial(row: dict[str, str]) -> int:
    return int(row["trial"])


def material_label(material: str) -> str:
    label = "PLA" if material.startswith("PLA") else material
    if label not in ACTIVE_CLASSES:
        raise ValueError(f"Unexpected active-touch material: {material}")
    return label


def active_features(path: Path, receiver_column: int, transmitter_column: int, trim_terminal_zero: bool) -> tuple[np.ndarray, int, int]:
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 4 or not np.isfinite(data).all():
        raise ValueError(f"Invalid active-touch CSV: {path}")
    receiver = data[:, receiver_column]
    transmitter = data[:, transmitter_column]
    trimmed = 0
    if trim_terminal_zero:
        valid_length = len(receiver)
        while valid_length and receiver[valid_length - 1] == 0 and transmitter[valid_length - 1] == 0:
            valid_length -= 1
        trimmed = len(receiver) - valid_length
        receiver, transmitter = receiver[:valid_length], transmitter[:valid_length]
    if len(receiver) < 1024:
        raise ValueError(f"Fewer than one 1024-sample segment after processing: {path}")

    def spectrum(signal: np.ndarray) -> tuple[np.ndarray, float, int]:
        starts = np.arange(0, len(signal) - 1024 + 1, 512)
        segments = np.stack([signal[start:start + 1024] for start in starts])
        segments = segments - segments.mean(axis=1, keepdims=True)
        segments *= np.hanning(1025)[:-1]
        power = np.abs(np.fft.rfft(segments, axis=1)) ** 2
        mean_power = power.mean(axis=0)
        frequencies = np.fft.rfftfreq(1024, d=1 / 8000)
        selected = mean_power[(frequencies >= 100) & (frequencies <= 800)]
        if len(selected) != 90:
            raise ValueError(f"Expected 90 spectrum bins; got {len(selected)}")
        demeaned = signal - signal.mean()
        rms = np.sqrt(np.mean(demeaned ** 2))
        return np.log10(np.maximum(selected, 1e-20)), float(np.log10(max(rms, 1e-12))), len(starts)

    receiver_spectrum, receiver_rms, count = spectrum(receiver)
    transmitter_spectrum, transmitter_rms, other_count = spectrum(transmitter)
    if count != other_count:
        raise AssertionError("Channel segment counts differ")
    feature = np.concatenate([
        receiver_spectrum,
        transmitter_spectrum,
        receiver_spectrum - transmitter_spectrum,
        np.asarray([receiver_rms, transmitter_rms]),
    ])
    if feature.shape != (272,):
        raise AssertionError(f"Expected 272 features, got {feature.shape}")
    return feature.astype(np.float64), trimmed, count


def experimental_rows(index_rows: list[dict[str, str]], checksums: dict[str, str]) -> list[dict]:
    candidates = [row for row in index_rows if row["subset"] == "experimental"]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[(row["material"], int(row["force_n"]))].append(row)
    rng = np.random.default_rng(42)
    output: list[dict] = []
    for key in sorted(grouped, key=lambda value: (value[0], value[1])):
        group = sorted(grouped[key], key=numeric_trial)
        if len(group) != 50:
            raise ValueError(f"Expected 50 records in experimental group {key}; got {len(group)}")
        shuffled = list(group)
        rng.shuffle(shuffled)
        for split, items in (("train", shuffled[:30]), ("validation", shuffled[30:40]), ("test", shuffled[40:])):
            for row in items:
                path = row["path"]
                output.append({
                    "recording_id": path, "path": path, "subset": "experimental", "split": split,
                    "source_material": row["material"], "target_label": material_label(row["material"]),
                    "force_n": int(row["force_n"]), "day": "", "sample_id": "",
                    "orientation_deg": "", "motor_speed_label": "", "trial": int(row["trial"]),
                    "sha256": checksums[path],
                })
    counts = Counter(row["split"] for row in output)
    if counts != {"train": 1050, "validation": 350, "test": 350}:
        raise ValueError(f"Unexpected experimental split counts: {counts}")
    return sorted(output, key=lambda row: row["recording_id"])


def simulation_rows(index_rows: list[dict[str, str]], checksums: dict[str, str]) -> list[dict]:
    mappings = read_csv_rows(DATASET_ROOT / "metadata" / "path_mapping.csv")
    simulations = {row["path"]: row for row in index_rows if row["subset"] == "simulation"}
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in mappings:
        if row["new_path"] in simulations:
            by_hash[row["sha256"]].append(row["new_path"])
    selected = sorted(min(paths) for paths in by_hash.values())
    if len(selected) != 35:
        raise ValueError(f"Expected 35 unique simulated signals; got {len(selected)}")
    output: list[dict] = []
    for path in selected:
        row = simulations[path]
        force = int(row["force_n"])
        split = "train" if force in (1, 2, 3) else "validation" if force == 4 else "test"
        output.append({
            "recording_id": path, "path": path, "subset": "simulation", "split": split,
            "source_material": row["material"], "target_label": material_label(row["material"]),
            "force_n": force, "day": "", "sample_id": "", "orientation_deg": "",
            "motor_speed_label": "", "trial": int(row["trial"]), "sha256": checksums[path],
        })
    counts = Counter(row["split"] for row in output)
    if counts != {"train": 23, "validation": 6, "test": 6}:
        raise ValueError(f"Unexpected simulation split counts: {counts}")
    return sorted(output, key=lambda row: row["recording_id"])


def strawberry_rows(index_rows: list[dict[str, str]], checksums: dict[str, str]) -> list[dict]:
    candidates = [row for row in index_rows if row["subset"] == "strawberries"]
    sample_ids = sorted({row["sample_id"] for row in candidates}, key=lambda value: int(value[1:]))
    if len(sample_ids) != 30:
        raise ValueError(f"Expected 30 strawberry IDs; got {len(sample_ids)}")
    rng = np.random.default_rng(42)
    shuffled = list(sample_ids)
    rng.shuffle(shuffled)
    allocation = {sample_id: "train" for sample_id in shuffled[:18]}
    allocation.update({sample_id: "validation" for sample_id in shuffled[18:24]})
    allocation.update({sample_id: "test" for sample_id in shuffled[24:]})
    output: list[dict] = []
    for row in candidates:
        path = row["path"]
        output.append({
            "recording_id": path, "path": path, "subset": "strawberries", "split": allocation[row["sample_id"]],
            "source_material": "", "target_label": row["day"], "force_n": "", "day": int(row["day"]),
            "sample_id": row["sample_id"], "orientation_deg": "", "motor_speed_label": "",
            "trial": int(row["trial"]), "sha256": checksums[path],
        })
    counts = Counter(row["split"] for row in output)
    if counts != {"train": 3600, "validation": 1200, "test": 1200}:
        raise ValueError(f"Unexpected strawberry split counts: {counts}")
    return sorted(output, key=lambda row: row["recording_id"])


def slippage_file_rows(index_rows: list[dict[str, str]], checksums: dict[str, str]) -> list[dict]:
    duplicate = "data/slippage/aluminum/Orientation_0/speed_medium/data.csv"
    candidates = [row for row in index_rows if row["subset"] == "slippage" and row["path"] != duplicate]
    if len(candidates) != 35:
        raise ValueError(f"Expected 35 unique slippage recordings; got {len(candidates)}")
    by_condition: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_condition[(row["material"], int(row["orientation_deg"]))].append(row)
    rng = np.random.default_rng(42)
    splits: dict[str, str] = {}
    leftovers: dict[str, list[dict[str, str]]] = defaultdict(list)
    for condition in sorted(by_condition, key=lambda value: (value[0], value[1])):
        group = sorted(by_condition[condition], key=lambda row: row["path"])
        rng.shuffle(group)
        if len(group) < 2:
            raise ValueError(f"Need train and test records in {condition}")
        splits[group[0]["path"]] = "train"
        splits[group[1]["path"]] = "test"
        leftovers[condition[0]].extend(group[2:])
    for material in sorted(leftovers):
        group = list(leftovers[material])
        rng.shuffle(group)
        if len(group) < 2:
            raise ValueError(f"Need two validation records for {material}")
        for row in group[:2]:
            splits[row["path"]] = "validation"
        for row in group[2:]:
            splits[row["path"]] = "train"
    output = []
    for row in candidates:
        path = row["path"]
        output.append({
            "recording_id": path, "path": path, "subset": "slippage", "split": splits[path],
            "source_material": row["material"], "target_label": "", "force_n": "", "day": "", "sample_id": "",
            "orientation_deg": int(row["orientation_deg"]), "motor_speed_label": row["motor_speed_label"],
            "trial": "", "sha256": checksums[path],
        })
    counts = Counter(row["split"] for row in output)
    if counts != {"train": 17, "validation": 6, "test": 12}:
        raise ValueError(f"Unexpected slippage split counts: {counts}")
    # Every material-angle pair must appear in both training and testing.
    for condition, group in by_condition.items():
        group_splits = {splits[row["path"]] for row in group}
        if not {"train", "test"}.issubset(group_splits):
            raise ValueError(f"Split violates condition coverage: {condition}, {group_splits}")
    return sorted(output, key=lambda row: row["recording_id"])


def slippage_features_and_windows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 8 or not np.isfinite(data).all():
        raise ValueError(f"Invalid slippage CSV: {path}")
    window_length = 400
    window_count = len(data) // window_length
    if window_count == 0:
        raise ValueError(f"No complete 400-sample windows: {path}")
    usable = data[:window_count * window_length]
    accelerometers = usable[:, :6].reshape(window_count, window_length, 6).transpose(0, 2, 1)
    velocities = np.abs(usable[:, 6:8].reshape(window_count, window_length, 2))
    labels = np.full((window_count, window_length), -1, dtype=np.int8)
    slip, slide = velocities[:, :, 0], velocities[:, :, 1]
    labels[(slip < 0.05) & (slide < 0.05)] = 0
    labels[(slip >= 0.05) & (slide < 0.05)] = 1
    labels[(slip < 0.05) & (slide >= 0.05)] = 2
    class_counts = np.stack([(labels == value).sum(axis=1) for value in range(3)], axis=1)
    chosen = np.argmax(class_counts, axis=1)
    count = class_counts[np.arange(window_count), chosen]
    keep = count >= 360
    starts = np.arange(window_count, dtype=np.int64) * window_length
    if not np.any(keep):
        return np.empty((0, 30), dtype=np.float64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8)

    frequencies = np.fft.rfftfreq(window_length, d=1 / 8000)
    bands = [(20, 125, False), (125, 500, False), (500, 1000, False), (1000, 1500, True)]
    total_mask = (frequencies >= 20) & (frequencies <= 1500)
    band_masks = [
        (frequencies >= lower) & ((frequencies <= upper) if inclusive_upper else (frequencies < upper))
        for lower, upper, inclusive_upper in bands
    ]
    retained_indices = np.flatnonzero(keep)
    features: list[np.ndarray] = []
    hann = np.hanning(401)[:-1]
    for begin in range(0, len(retained_indices), 256):
        batch = accelerometers[retained_indices[begin:begin + 256]]
        demeaned = batch - batch.mean(axis=2, keepdims=True)
        rms = np.sqrt(np.mean(demeaned ** 2, axis=2))
        log_rms = np.log10(np.maximum(rms, 1e-12))
        power = np.abs(np.fft.rfft(demeaned * hann, axis=2)) ** 2
        total = power[:, :, total_mask].sum(axis=2)
        fractions = [
            np.divide(power[:, :, mask].sum(axis=2), total, out=np.zeros_like(total), where=total > 0)
            for mask in band_masks
        ]
        feature = np.concatenate([log_rms, *fractions], axis=1)
        if feature.shape[1] != 30:
            raise AssertionError(f"Expected 30 features, got {feature.shape}")
        features.append(feature)
    return np.concatenate(features), starts[keep], chosen[keep]


def prepare_task(name: str) -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            "Raw dataset not found. Download TAVIB_dataset and set "
            "dataset_root in dataset_config.json, or use run_all.py --from-features."
        )
    spec = task_spec(name)
    ensure_dirs(spec)
    index_rows, checksums = load_file_index(), load_checksums()
    if name == "experimental":
        rows = experimental_rows(index_rows, checksums)
        columns, trim = (1, 3), True
    elif name == "simulation":
        rows = simulation_rows(index_rows, checksums)
        columns, trim = (3, 1), False
    elif name == "strawberries":
        rows = strawberry_rows(index_rows, checksums)
        columns, trim = (2, 3), False
    else:
        rows = slippage_file_rows(index_rows, checksums)
        columns, trim = None, False

    manifest_dir = spec.directory / "manifests"
    derived_dir = spec.directory / "derived"
    if name != "slippage":
        features, labels, record_ids, feature_rows = [], [], [], []
        for number, row in enumerate(rows, start=1):
            path = DATASET_ROOT / row["path"]
            feature, trimmed, segment_count = active_features(path, columns[0], columns[1], trim)
            features.append(feature)
            labels.append(spec.classes.index(row["target_label"]))
            record_ids.append(row["recording_id"])
            feature_rows.append({**row, "feature_index": len(features) - 1, "feature_count": 272,
                                 "segment_count": segment_count, "trimmed_terminal_zero_rows": trimmed})
            if number % 100 == 0 or number == len(rows):
                print(f"Prepared {name} feature {number}/{len(rows)}", flush=True)
        x = np.stack(features)
        y = np.asarray(labels, dtype=np.int64)
        np.savez_compressed(derived_dir / "features.npz", X=x, y=y, record_ids=np.asarray(record_ids, dtype=str))
        write_csv(manifest_dir / "records.csv", feature_rows)
        output_counts = {split: int(sum(row["split"] == split for row in feature_rows)) for split in ("train", "validation", "test")}
        window_count = None
    else:
        all_features, all_labels, all_ids, window_rows = [], [], [], []
        record_rows = []
        for number, row in enumerate(rows, start=1):
            path = DATASET_ROOT / row["path"]
            feature, starts, labels = slippage_features_and_windows(path)
            record_rows.append({**row, "retained_window_count": len(feature), "feature_count": 30})
            for index, (start, label) in enumerate(zip(starts.tolist(), labels.tolist())):
                window_rows.append({**row, "window_feature_index": len(all_features) + index,
                                    "window_index": int(start // 400), "start_index": int(start),
                                    "end_index_exclusive": int(start + 400), "target_label": int(label)})
            all_features.extend(feature)
            all_labels.extend(labels.tolist())
            all_ids.extend([row["recording_id"]] * len(feature))
            print(f"Prepared slippage recording {number}/{len(rows)}: {len(feature)} retained windows", flush=True)
        x = np.asarray(all_features, dtype=np.float64)
        y = np.asarray(all_labels, dtype=np.int64)
        np.savez_compressed(derived_dir / "features.npz", X=x, y=y, record_ids=np.asarray(all_ids, dtype=str))
        write_csv(manifest_dir / "records.csv", record_rows)
        write_csv(manifest_dir / "windows.csv", window_rows)
        output_counts = {split: int(sum(row["split"] == split for row in record_rows)) for split in ("train", "validation", "test")}
        window_count = {split: int(sum(row["split"] == split for row in window_rows)) for split in ("train", "validation", "test")}

    write_csv(manifest_dir / "splits.csv", rows)
    metadata = {
        "task": name, "dataset_config": DATASET_CONFIG_PATH.name, "workflow": "classification",
        "classes": spec.classes, "estimator": spec.estimator_name, "records_by_split": output_counts,
        "windows_by_split": window_count, "feature_shape": list(x.shape), "feature_file": "derived/features.npz",
        "file_index_sha256": sha256_file(DATASET_ROOT / "metadata" / "file_index.csv"),
        "path_mapping_sha256": sha256_file(DATASET_ROOT / "metadata" / "path_mapping.csv"),
        "release_manifest_sha256": sha256_file(DATASET_ROOT / "checksums" / "sha256_manifest.csv"),
    }
    (manifest_dir / "prepare_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def load_prepared(spec: TaskSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str]]]:
    payload = np.load(spec.directory / "derived" / "features.npz", allow_pickle=False)
    record_rows = read_csv_rows(spec.directory / "manifests" / "records.csv")
    split_by_id = {row["recording_id"]: row["split"] for row in record_rows}
    record_ids = payload["record_ids"].astype(str)
    if any(record_id not in split_by_id for record_id in record_ids):
        raise ValueError("Feature recording missing from record manifest")
    return payload["X"], payload["y"], record_ids, record_rows


def make_model(spec: TaskSpec):
    if spec.name == "slippage":
        return HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
            learning_rate=0.1, l2_regularization=1.0, class_weight="balanced",
            early_stopping=False, random_state=42,
        )
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features="sqrt",
        min_samples_leaf=1 if spec.name == "simulation" else 5,
        bootstrap=True, class_weight="balanced", random_state=42, n_jobs=-1,
    )


def labels_for_ids(record_ids: np.ndarray, record_rows: list[dict[str, str]]) -> np.ndarray:
    split_by_id = {row["recording_id"]: row["split"] for row in record_rows}
    return np.asarray([split_by_id[record_id] for record_id in record_ids])


def majority_class(labels: np.ndarray, class_count: int) -> int:
    counts = np.bincount(labels, minlength=class_count)
    return int(np.flatnonzero(counts == counts.max())[0])


def active_metric_rows(y_true: np.ndarray, y_pred: np.ndarray, spec: TaskSpec, method: str, split: str) -> tuple[dict, np.ndarray]:
    labels = np.arange(len(spec.classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    result = {
        "task": spec.name, "split": split, "method": method, "aggregation_unit": "one_prediction_per_csv",
        "n_predictions": int(len(y_true)), "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(np.mean(recall)),
    }
    result.update({f"recall_{name}": float(value) for name, value in zip(spec.classes, recall)})
    return result, normalized


def slippage_metric_rows(y_true: np.ndarray, y_pred: np.ndarray, record_ids: np.ndarray, spec: TaskSpec, method: str, split: str) -> tuple[dict, np.ndarray, list[dict]]:
    labels = np.arange(3)
    per_record = []
    matrices = []
    for record_id in sorted(set(record_ids.tolist())):
        selected = record_ids == record_id
        truth, pred = y_true[selected], y_pred[selected]
        matrix = confusion_matrix(truth, pred, labels=labels)
        normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
        recall = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
        row = {
            "task": spec.name, "split": split, "method": method, "recording_id": record_id,
            "window_count": int(selected.sum()), "macro_f1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(truth, pred)), "balanced_accuracy": float(np.mean(recall)),
            **{f"recall_{name}": float(value) for name, value in zip(spec.classes, recall)},
        }
        per_record.append(row)
        matrices.append(normalized)
    averaged = {field: float(np.mean([row[field] for row in per_record])) for field in ("macro_f1", "accuracy", "balanced_accuracy")}
    averaged.update({f"recall_{name}": float(np.mean([row[f"recall_{name}"] for row in per_record])) for name in spec.classes})
    result = {
        "task": spec.name, "split": split, "method": method, "aggregation_unit": "equal_weight_per_test_recording",
        "n_predictions": int(len(y_true)), "n_recordings": len(per_record), **averaged,
    }
    return result, np.mean(matrices, axis=0), per_record


def prediction_rows(record_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, spec: TaskSpec, split: str, method: str) -> list[dict]:
    return [
        {"task": spec.name, "split": split, "method": method, "recording_id": record_id,
         "true_label_index": int(actual), "true_label": spec.classes[int(actual)],
         "predicted_label_index": int(predicted), "predicted_label": spec.classes[int(predicted)]}
        for record_id, actual, predicted in zip(record_ids.tolist(), y_true.tolist(), y_pred.tolist())
    ]


def write_confusion(path: Path, matrix: np.ndarray, classes: list[str]) -> None:
    rows = []
    for i, label in enumerate(classes):
        rows.append({"true_label": label, **{f"predicted_{name}": float(matrix[i, j]) for j, name in enumerate(classes)}})
    write_csv(path, rows)


def train_and_evaluate(name: str) -> None:
    spec = task_spec(name)
    ensure_dirs(spec)
    x, y, record_ids, record_rows = load_prepared(spec)
    assignment = labels_for_ids(record_ids, record_rows)
    train = assignment == "train"
    validation = assignment == "validation"
    test = assignment == "test"
    if not all((np.any(train), np.any(validation), np.any(test))):
        raise ValueError("Missing a data partition")
    model = make_model(spec)
    model.fit(x[train], y[train])
    validation_predictions = model.predict(x[validation])
    validation_majority = np.full(validation.sum(), majority_class(y[train], len(spec.classes)), dtype=int)
    model.fit(x[train | validation], y[train | validation])
    final_path = spec.directory / "models" / f"{name}_final_model.joblib"
    joblib.dump(model, final_path)
    test_predictions = model.predict(x[test])
    final_majority = np.full(test.sum(), majority_class(y[train | validation], len(spec.classes)), dtype=int)
    result_dir = spec.directory / "results"
    all_metrics: list[dict] = []
    all_predictions: list[dict] = []
    for split, mask, model_pred, baseline_pred in (
        ("validation", validation, validation_predictions, validation_majority),
        ("test", test, test_predictions, final_majority),
    ):
        if name == "slippage":
            model_metrics, model_matrix, model_per_record = slippage_metric_rows(y[mask], model_pred, record_ids[mask], spec, spec.estimator_name, split)
            baseline_metrics, baseline_matrix, baseline_per_record = slippage_metric_rows(y[mask], baseline_pred, record_ids[mask], spec, "majority_class", split)
            write_csv(result_dir / f"{split}_per_recording_metrics.csv", model_per_record + baseline_per_record)
        else:
            model_metrics, model_matrix = active_metric_rows(y[mask], model_pred, spec, spec.estimator_name, split)
            baseline_metrics, baseline_matrix = active_metric_rows(y[mask], baseline_pred, spec, "majority_class", split)
        all_metrics.extend([model_metrics, baseline_metrics])
        all_predictions.extend(prediction_rows(record_ids[mask], y[mask], model_pred, spec, split, spec.estimator_name))
        all_predictions.extend(prediction_rows(record_ids[mask], y[mask], baseline_pred, spec, split, "majority_class"))
        write_confusion(result_dir / f"{split}_{spec.estimator_name}_confusion_matrix.csv", model_matrix, spec.classes)
        write_confusion(result_dir / f"{split}_majority_class_confusion_matrix.csv", baseline_matrix, spec.classes)
    write_csv(result_dir / "metrics.csv", all_metrics)
    write_csv(result_dir / "predictions.csv", all_predictions)
    runtime = {
        "task": name, "model": repr(model), "final_model": final_path.relative_to(spec.directory).as_posix(),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "matplotlib": plt.matplotlib.__version__, "random_seed": 42,
        "fit_for_validation": "training partition", "fit_for_test": "training plus validation partitions",
        "feature_shape": list(x.shape), "records": len(record_rows),
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2), flush=True)


def plot_task(name: str) -> None:
    spec = task_spec(name)
    result_dir = spec.directory / "results"
    matrix_rows = read_csv_rows(result_dir / f"test_{spec.estimator_name}_confusion_matrix.csv")
    matrix = np.asarray([[float(row[f"predicted_{label}"]) for label in spec.classes] for row in matrix_rows])
    fig, axis = plt.subplots(figsize=(6.2, 5.1), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(spec.classes)):
        for j in range(len(spec.classes)):
            value = matrix[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value >= 0.55 else "black")
    axis.set_xticks(range(len(spec.classes)), spec.classes, rotation=30, ha="right")
    axis.set_yticks(range(len(spec.classes)), spec.classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    title = "Test: row-normalized confusion matrix"
    if name == "slippage":
        title += "\n(equal-weight average across recordings)"
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label="Proportion")
    out = result_dir / "figures" / f"{name}_test_confusion_matrix"
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=list(TASK_DIRS))
    parser.add_argument("action", choices=["prepare", "train", "plot"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_task(args.task)
    elif args.action == "train":
        train_and_evaluate(args.task)
    else:
        plot_task(args.task)


if __name__ == "__main__":
    cli()
