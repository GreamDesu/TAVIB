"""Reproduce the four classification experiments and their combined results."""
import argparse
import json
from pathlib import Path

from pipeline import TASK_DIRS, plot_task, prepare_task, train_and_evaluate
from overall.summary import main as summarize
from overall.audit import main as audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-features", action="store_true",
        help="Use the bundled features and splits instead of reading the raw dataset.",
    )
    args = parser.parse_args()
    for task in TASK_DIRS:
        if not args.from_features:
            prepare_task(task)
        train_and_evaluate(task)
        plot_task(task)
    summarize()
    audit()
    output = Path(__file__).resolve().parent / "reproducibility.json"
    output.write_text(json.dumps({
        "command": "python tactile_dataset_validation/run_all.py" +
                   (" --from-features" if args.from_features else ""),
        "postprocess": [
            "python tactile_dataset_validation/overall/summary.py",
            "python tactile_dataset_validation/overall/audit.py"
        ],
        "tasks": list(TASK_DIRS),
        "random_seed": 42,
        "requirements": "tactile_dataset_validation/requirements.txt"
    }, indent=2) + "\n", encoding="utf-8")
