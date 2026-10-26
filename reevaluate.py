from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from wslgmr import load_config_file, resolve_config
from wslgmr.datasets import build_dataloaders
from wslgmr.evaluation import (
    calibrated_batchnorm,
    evaluate,
    representative_calibration_loader,
)
from wslgmr.models import build_model
from wslgmr.utils import select_device, set_seed, write_json


def _load_payload(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved checkpoints with the current validation protocol."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path or name under the configured output directory; repeatable.",
    )
    parser.add_argument(
        "--output",
        default="reevaluation_current.json",
        help="JSON filename written under the configured output directory.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = resolve_config(load_config_file(config_path), project_root=project_root)
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_tokens = args.checkpoint or ["lastest.pt", "best.pt"]
    checkpoint_paths = []
    for token in checkpoint_tokens:
        path = Path(token)
        if not path.is_absolute():
            path = output_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint_paths.append(path)

    device = select_device(config["experiment"]["device"])
    set_seed(
        int(config["experiment"]["seed"]),
        bool(config["experiment"]["deterministic"]),
    )
    data = build_dataloaders(config)
    model = build_model(
        config,
        input_shape=data.input_shape,
        mean=data.mean,
        std=data.std,
        num_classes=data.num_classes,
    ).to(device)

    calibration_batches = int(config["evaluation"]["bn_calibration_batches"])
    evaluations = []
    for checkpoint_path in checkpoint_paths:
        payload = _load_payload(checkpoint_path, device)
        model.load_state_dict(payload["model"])
        calibration_loader = representative_calibration_loader(
            data.calibration_loader,
            batches=calibration_batches,
            seed=int(config["experiment"]["seed"]) + 1,
        )
        started = time.perf_counter()
        with calibrated_batchnorm(
            model,
            calibration_loader,
            device,
            calibration_batches,
        ):
            metrics = evaluate(
                model,
                data.eval_loader,
                device,
                noise_type=str(config["evaluation"]["noise_type"]),
                noise_sigma=float(config["evaluation"]["noise_sigma"]),
                noise_clip=bool(config["evaluation"]["noise_clip"]),
                noise_seed=int(config["evaluation"]["noise_seed"]),
                foreground_flip_probability=float(
                    config["evaluation"]["foreground_flip_probability"]
                ),
                background_flip_probability=float(
                    config["evaluation"]["background_flip_probability"]
                ),
                binary_threshold=float(config["evaluation"]["binary_threshold"]),
            )
        evaluations.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "epoch": int(payload.get("epoch", -1)),
                "evaluation_seconds": time.perf_counter() - started,
                **metrics,
            }
        )
        print(
            f"{checkpoint_path.name} epoch={evaluations[-1]['epoch']} "
            f"clean@1={100.0 * metrics['clean_top1_accuracy']:.2f}% "
            f"noise@1={100.0 * metrics['noisy_top1_accuracy']:.2f}%"
        )

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": str(config_path),
        "experiment": config["experiment"],
        "dataset": config["dataset"],
        "model": config["model"],
        "evaluation_config": config["evaluation"],
        "batchnorm_calibration": {
            "sampling": "uniform_without_replacement",
            "seed": int(config["experiment"]["seed"]) + 1,
            "batches": calibration_batches,
            "momentum": None,
            "temporary_statistics": True,
        },
        "evaluations": evaluations,
    }
    destination = output_dir / args.output
    write_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
