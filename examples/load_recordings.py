"""Load one released CSV from each subset, preserving its columns and samples."""
import argparse
import json
from pathlib import Path

import numpy as np


EXAMPLES = {
    "experimental": ("active_touch/experimental/3N/TPU/cubeTPU_3N_trial1.csv", 1, 4),
    "simulation": ("active_touch/simulation/5N/Silicone18/8000_27.csv", 1, 4),
    "strawberry": ("active_touch/strawberries/day1/s1/stm_sweep_1.csv", 1, 4),
    "slippage": ("slippage/aluminum/Orientation_0/speed_low/data.csv", 0, 8),
}


def load_examples(dataset_root: Path) -> dict[str, np.ndarray]:
    recordings = {}
    for subset, (relative_path, header_rows, columns) in EXAMPLES.items():
        path = dataset_root / "data" / relative_path
        data = np.loadtxt(path, delimiter=",", skiprows=header_rows, ndmin=2)
        if data.shape[0] == 0 or data.shape[1] != columns or not np.isfinite(data).all():
            raise ValueError(f"Expected nonempty finite data with {columns} columns: {path}")
        recordings[subset] = data
    return recordings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Override dataset_config.json.")
    args = parser.parse_args()
    root = args.dataset_root
    if root is None:
        validation = Path(__file__).resolve().parents[1] / "tactile_dataset_validation"
        config = json.loads((validation / "dataset_config.json").read_text(encoding="utf-8"))
        root = validation / Path(config["dataset_root"]).expanduser()
    root = root.expanduser().resolve()
    if not root.is_dir():
        parser.error("Dataset directory not found; set --dataset-root or dataset_config.json.")
    recordings = load_examples(root)
    time_slippage_s = np.arange(len(recordings["slippage"])) / 8000.0
    for name, data in recordings.items():
        print(f"{name}: {data.shape[0]:,} rows, {data.shape[1]} columns")
    print(f"slippage timestamps: {time_slippage_s[0]:.6f} to {time_slippage_s[-1]:.6f} s")
