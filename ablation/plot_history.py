from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wslgmr.plotting import plot_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one native experiment history without importing Torch."
    )
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.history.read_text(encoding="utf-8"))
    plot_history(rows, args.output)


if __name__ == "__main__":
    main()
