from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import cli

if __name__ == "__main__":
    sys.argv[1:1] = ["simulation"]
    cli()
