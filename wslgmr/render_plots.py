from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotting import plot_branched_history, plot_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render training curves in a Matplotlib-only process."
    )
    parser.add_argument("kind", choices=("history", "branches"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if args.kind == "history":
        plot_history(value, args.output)
    else:
        plot_branched_history(value, args.output)


if __name__ == "__main__":
    main()
