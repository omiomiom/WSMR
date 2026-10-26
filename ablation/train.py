from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wslgmr import load_config_file, resolve_config
from wslgmr import engine as core_engine

from ablation.engine import train_one_epoch


def _plot_history_isolated(rows, output_path: Path) -> None:
    """Render the current history in a Matplotlib-only subprocess."""

    del rows
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(PROJECT_ROOT / "ablation/plot_history.py"),
            str(output_path.with_name("history.json")),
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated ablation.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", nargs="?", const="lastest", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training and ablation randomness for an ablation repeat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the result directory for an isolated ablation repeat.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = resolve_config(
        load_config_file(config_path), project_root=PROJECT_ROOT
    )
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
        config["wake_sleep"]["pairing_seed"] = int(args.seed)
        config["ablation"]["protocol"] = (
            "cifar100_wrn28_10_"
            f"seed{int(args.seed)}_100epochs_fixed_eval_noise42"
        )
    if args.output_dir is not None:
        config["paths"]["output_dir"] = str(args.output_dir)
    variant = str(config.get("ablation", {}).get("variant", "")).lower()
    if not variant:
        raise ValueError("Ablation configs must define ablation.variant.")
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    # Use the ablation epoch loop with the shared checkpoint and evaluation runner.
    core_engine.train_one_epoch = train_one_epoch
    # Render curves in a subprocess to isolate the PyTorch and Matplotlib
    # OpenMP runtimes on Windows.
    core_engine.plot_history = _plot_history_isolated
    core_engine.run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
