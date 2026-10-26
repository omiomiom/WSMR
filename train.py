from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from wslgmr import load_config_file, resolve_config
from wslgmr.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train from one Python configuration file."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/config_cifar100_gi_i20.py",
        help="Config file containing a CONFIG dictionary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the training seed without creating another config file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the result directory for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the configuration without training.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="lastest",
        default=None,
        metavar="EPOCH",
        help=(
            "Resume from lastest.pt, or from checkpoints/epoch_XXX.pt "
            "when an epoch number is supplied."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = resolve_config(
        load_config_file(config_path),
        project_root=project_root,
    )
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
        name = re.sub(r"_seed\d+$", "", config["experiment"]["name"])
        config["experiment"]["name"] = f"{name}_seed{args.seed}"
        output = Path(config["paths"]["output_dir"])
        stem = re.sub(r"_seed\d+$", "", output.name)
        config["paths"]["output_dir"] = str(
            output.with_name(f"{stem}_seed{args.seed}")
        )
    if args.output_dir is not None:
        output = Path(args.output_dir).expanduser()
        if not output.is_absolute():
            output = project_root / output
        config["paths"]["output_dir"] = str(output.resolve())
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
