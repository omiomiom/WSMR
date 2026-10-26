from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import traceback
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wslgmr import load_config_file, resolve_config
from wslgmr.adversarial import (
    indices_sha256,
    materialize_subset,
    select_evaluation_indices,
)
from wslgmr.datasets import build_dataloaders
from wslgmr.evaluation import calibrated_batchnorm, representative_calibration_loader
from wslgmr.models import build_model
from wslgmr.utils import select_device, set_seed, write_json


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config: str


@dataclass(frozen=True)
class FamilySpec:
    name: str
    models: tuple[ModelSpec, ...]
    epsilons_pixels: tuple[float, ...]
    sample_count: int
    attack_batch_size: int
    dataset_root: str | None = None


FAMILIES = (
    FamilySpec(
        name="mnist_mlp",
        models=(
            ModelSpec("baseline", "Baseline", "configs/config_mnist_baseline.py"),
            ModelSpec("gmr", "GMR", "configs/config_mnist_exact_g0.py"),
            ModelSpec("sleep1", "Sleep T=1", "configs/config_mnist_gi_i1.py"),
            ModelSpec("sleep20", "Sleep T=20", "configs/config_mnist_gi_i20.py"),
        ),
        # The original LMDN MNIST figures sweep epsilon through 0.4 in
        # [0, 1] intensity units.  Store the same grid in the script's
        # common pixel-value convention so the attack implementation and
        # serialized schema stay identical across datasets.
        epsilons_pixels=(0.0, 12.75, 25.5, 51.0, 76.5, 102.0),
        sample_count=0,
        attack_batch_size=256,
    ),
    FamilySpec(
        name="cifar100_wrn28_10",
        models=(
            ModelSpec("baseline", "Baseline", "configs/config_cifar100_baseline.py"),
            ModelSpec("gmr", "GMR", "configs/config_cifar100_exact_g0.py"),
            ModelSpec("sleep1", "Sleep T=1", "configs/config_cifar100_gi_i1.py"),
            ModelSpec("sleep20", "Sleep T=20", "configs/config_cifar100_gi_i20.py"),
        ),
        epsilons_pixels=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
        sample_count=0,
        attack_batch_size=64,
    ),
    FamilySpec(
        name="imagenet1k_resnet18",
        models=(
            ModelSpec("baseline", "Baseline", "configs/config_imagenet1k_resnet18_baseline.py"),
            ModelSpec("gmr", "GMR", "configs/config_imagenet1k_resnet18_exact_g0.py"),
            ModelSpec("sleep1", "Sleep T=1", "configs/config_imagenet1k_resnet18_gi_i1.py"),
            ModelSpec("sleep20", "Sleep T=20", "configs/config_imagenet1k_resnet18_gi_i20.py"),
        ),
        epsilons_pixels=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
        sample_count=1000,
        attack_batch_size=16,
        dataset_root=None,
    ),
    FamilySpec(
        name="imagenet1k_resnet50",
        models=(
            ModelSpec("baseline", "Baseline", "configs/config_imagenet1k_baseline.py"),
            ModelSpec("sleep20", "Sleep T=20", "configs/config_imagenet1k_gi_i20.py"),
        ),
        epsilons_pixels=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
        sample_count=1000,
        attack_batch_size=8,
        dataset_root=None,
    ),
)

DEFAULT_FAMILIES = (
    "mnist_mlp",
    "cifar100_wrn28_10",
    "imagenet1k_resnet18",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def classic_ifgsm_steps(epsilon_pixels: float) -> int:
    """Kurakin-style BIM/IFGSM iteration heuristic in pixel units."""
    if epsilon_pixels <= 0:
        return 0
    return max(1, int(math.ceil(min(epsilon_pixels + 4.0, 1.25 * epsilon_pixels))))


def attack_batch(
    model: nn.Module,
    images: Tensor,
    targets: Tensor,
    *,
    epsilon: float,
    attack: str,
) -> Tensor:
    if epsilon <= 0:
        return images.detach().clone()
    if attack == "fgsm":
        steps = 1
        step_size = epsilon
    elif attack == "ifgsm":
        steps = classic_ifgsm_steps(epsilon * 255.0)
        step_size = 1.0 / 255.0
    else:
        raise ValueError(f"Unsupported attack: {attack}")

    adversarial = images.detach().clone()
    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), targets, reduction="sum")
        gradient = torch.autograd.grad(loss, adversarial)[0]
        with torch.no_grad():
            adversarial = adversarial + step_size * gradient.sign()
            adversarial = torch.maximum(
                torch.minimum(adversarial, images + epsilon),
                images - epsilon,
            ).clamp_(0.0, 1.0)
    return adversarial.detach()


def evaluate_adversarial_accuracy(
    source_model: nn.Module,
    target_model: nn.Module,
    images: Tensor,
    targets: Tensor,
    *,
    device: torch.device,
    batch_size: int,
    epsilon: float,
    attack: str,
) -> dict:
    """Report test accuracy on generated examples, matching Elsayed et al."""
    correct = 0
    started = time.perf_counter()
    for start in range(0, targets.numel(), batch_size):
        end = min(start + batch_size, targets.numel())
        x = images[start:end].to(device, non_blocking=True)
        y = targets[start:end].to(device, non_blocking=True)
        adversarial = attack_batch(source_model, x, y, epsilon=epsilon, attack=attack)
        with torch.no_grad():
            correct += int(target_model(adversarial).argmax(dim=1).eq(y).sum())
        if (start // batch_size + 1) % 50 == 0 or end == targets.numel():
            print(f"  evaluated {end}/{targets.numel()} images | accuracy={correct / end:.4%}", flush=True)
    count = int(targets.numel())
    return {
        "sample_count": count,
        "correct": correct,
        "accuracy": correct / count,
        "epsilon": float(epsilon),
        "epsilon_pixels": float(epsilon * 255.0),
        "steps": 1 if attack == "fgsm" else classic_ifgsm_steps(epsilon * 255.0),
        "step_size": float(epsilon if attack == "fgsm" else 1.0 / 255.0),
        "wall_seconds": time.perf_counter() - started,
    }


def load_config(
    project_root: Path,
    relative_path: str,
    dataset_root: str | None,
) -> tuple[Path, dict]:
    path = project_root / relative_path
    raw = load_config_file(path)
    if dataset_root is not None:
        raw["paths"]["dataset_root"] = dataset_root
    return path, resolve_config(raw, project_root=project_root)


def load_model_for_config(config: dict, data, device: torch.device) -> tuple[nn.Module, Path, dict]:
    checkpoint = Path(config["paths"]["output_dir"]) / "lastest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    saved_config_path = checkpoint.parent / "resolved_config.json"
    saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
    if saved_config["model"] != config["model"]:
        raise ValueError(f"Checkpoint model/config mismatch: {checkpoint}")
    model = build_model(
        config,
        input_shape=data.input_shape,
        mean=data.mean,
        std=data.std,
        num_classes=data.num_classes,
    ).to(device)
    payload = load_checkpoint(checkpoint, device)
    if int(payload.get("epoch", -1)) != int(config["training"]["epochs"]):
        raise ValueError(f"Expected the final scheduled checkpoint: {checkpoint}")
    model.load_state_dict(payload["model"])
    model.eval()
    return model, checkpoint, payload


def new_result(
    spec: FamilySpec,
    indices: list[int],
    configs: dict[str, tuple[Path, dict]],
    checkpoints: dict[str, tuple[Path, dict]],
) -> dict:
    keys = [model.key for model in spec.models]
    transfer_keys = [key for key in keys if key != "baseline"]
    baseline_config = configs["baseline"][1]
    return {
        "schema_version": 2,
        "state": "running",
        "created_at_utc": utc_now(),
        "family": spec.name,
        "protocol": {
            "reported_metric": "test accuracy on adversarial examples",
            "attacks": [
                "white-box FGSM",
                "white-box IFGSM",
                "baseline-source black-box FGSM",
                "baseline-source black-box IFGSM",
            ],
            "purpose": "Attack and reporting protocol matched to Elsayed et al. (NeurIPS 2018).",
            "pixel_domain": [0.0, 1.0],
            "norm": "Linf",
            "loss": "cross_entropy",
            "targeted": False,
            "random_start": False,
            "ifgsm_step_size": 1.0 / 255.0,
            "ifgsm_iterations": "ceil(min(epsilon_pixels + 4, 1.25 * epsilon_pixels))",
            "note": (
                "Dataset, architecture and training loss differ from the original paper; "
                "this matches its reported attack metrics but is not a faithful reproduction "
                "of its training setup."
            ),
        },
        "subset": {
            "sample_count": len(indices),
            "seed": 4242,
            "indices_sha256": indices_sha256(indices),
            "indices": indices,
            "selection": "class_balanced" if baseline_config["dataset"] == "imagenet1k" else "full_test_set",
        },
        "attack_batch_size": spec.attack_batch_size,
        "models": {
            model.key: {
                "label": model.label,
                "config": str(configs[model.key][0]),
                "checkpoint": str(checkpoints[model.key][0]),
                "checkpoint_sha256": sha256_file(checkpoints[model.key][0]),
                "checkpoint_epoch": int(checkpoints[model.key][1].get("epoch", -1)),
            }
            for model in spec.models
        },
        "batchnorm_calibration": {
            "sampling": "uniform_without_replacement",
            "seed": int(baseline_config["experiment"]["seed"]) + 1,
            "batches": int(baseline_config["evaluation"]["bn_calibration_batches"]),
            "temporary_statistics": True,
            "frozen_during_attacks": True,
        },
        "results": {
            "whitebox_fgsm": {key: [] for key in keys},
            "whitebox_ifgsm": {key: [] for key in keys},
            "blackbox_fgsm": {key: [] for key in transfer_keys},
            "blackbox_ifgsm": {key: [] for key in transfer_keys},
        },
    }


def evaluate_family(project_root: Path, spec: FamilySpec, output_root: Path) -> dict:
    configs = {
        model.key: load_config(project_root, model.config, spec.dataset_root)
        for model in spec.models
    }
    baseline_config = configs["baseline"][1]
    device = select_device(baseline_config["experiment"]["device"])
    set_seed(
        int(baseline_config["experiment"]["seed"]),
        bool(baseline_config["experiment"]["deterministic"]),
    )
    data = build_dataloaders(baseline_config)
    requested = spec.sample_count if spec.sample_count > 0 else None
    indices = select_evaluation_indices(
        data.eval_loader.dataset,
        sample_count=requested,
        num_classes=data.num_classes,
        seed=4242,
        stratified=baseline_config["dataset"] == "imagenet1k",
    )
    print(f"[{spec.name}] loading {len(indices)} evaluation images; subset SHA256={indices_sha256(indices)}", flush=True)
    images, targets = materialize_subset(
        data.eval_loader.dataset,
        indices,
        batch_size=int(baseline_config["loader"]["eval_batch_size"]),
        num_workers=int(baseline_config["loader"]["num_workers"]),
        pin_memory=bool(baseline_config["loader"]["pin_memory"]),
    )
    print(f"[{spec.name}] subset loaded; per-class counts: min={int(torch.bincount(targets).min())}, max={int(torch.bincount(targets).max())}", flush=True)

    models: dict[str, nn.Module] = {}
    checkpoints: dict[str, tuple[Path, dict]] = {}
    for model_spec in spec.models:
        model, checkpoint, payload = load_model_for_config(
            configs[model_spec.key][1], data, device
        )
        models[model_spec.key] = model
        checkpoints[model_spec.key] = (checkpoint, payload)

    result_path = output_root / f"{spec.name}.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result.get("schema_version", 0)) != 2:
            raise ValueError(f"Unsupported existing result schema: {result_path}; use a different --output-dir.")
        if result["subset"]["sample_count"] != len(indices) or result["subset"]["indices_sha256"] != indices_sha256(indices):
            raise ValueError(f"Evaluation subset changed: {result_path}; use a different --output-dir.")
        for key, (checkpoint, payload) in checkpoints.items():
            if result["models"][key]["checkpoint_sha256"] != sha256_file(checkpoint):
                raise ValueError(f"Checkpoint changed for {key}: use a different --output-dir.")
        for section in result["results"].values():
            for points in section.values():
                if any(point["sample_count"] != len(indices) for point in points):
                    raise ValueError(f"Mixed sample counts in {result_path}")
    else:
        result = new_result(spec, indices, configs, checkpoints)
        write_json(result_path, result)

    calibration_batches = int(baseline_config["evaluation"]["bn_calibration_batches"])
    calibration_seed = int(baseline_config["experiment"]["seed"]) + 1
    with ExitStack() as stack:
        for model in models.values():
            loader = representative_calibration_loader(
                data.calibration_loader,
                batches=calibration_batches,
                seed=calibration_seed,
            )
            stack.enter_context(
                calibrated_batchnorm(model, loader, device, calibration_batches)
            )

        jobs: list[tuple[str, str, nn.Module, nn.Module, str]] = []
        for attack in ("fgsm", "ifgsm"):
            for key, model in models.items():
                jobs.append((f"whitebox_{attack}", key, model, model, attack))
            for key, model in models.items():
                if key != "baseline":
                    jobs.append(
                        (f"blackbox_{attack}", key, models["baseline"], model, attack)
                    )

        for section, model_name, source, target, attack in jobs:
            completed_eps = {
                round(float(item["epsilon_pixels"]), 8)
                for item in result["results"][section][model_name]
            }
            for epsilon_pixels in spec.epsilons_pixels:
                if round(float(epsilon_pixels), 8) in completed_eps:
                    continue
                print(
                    f"[{spec.name}] {section}/{model_name} "
                    f"epsilon={epsilon_pixels}/255",
                    flush=True,
                )
                metrics = evaluate_adversarial_accuracy(
                    source,
                    target,
                    images,
                    targets,
                    device=device,
                    batch_size=spec.attack_batch_size,
                    epsilon=epsilon_pixels / 255.0,
                    attack=attack,
                )
                result["results"][section][model_name].append(metrics)
                result["updated_at_utc"] = utc_now()
                write_json(result_path, result)

    result["state"] = "completed"
    result["updated_at_utc"] = utc_now()
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FGSM/IFGSM accuracy curves reported by Elsayed et al."
    )
    parser.add_argument(
        "--family",
        action="append",
        choices=tuple(spec.name for spec in FAMILIES),
    )
    parser.add_argument("--sample-count", type=int, help="Override the ImageNet class-balanced subset size only.")
    parser.add_argument("--output-dir", type=Path, help="Separate output directory for a new evaluation protocol.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent
    output_root = args.output_dir or (project_root / "Results" / "lmdn_attack_protocol")
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    requested = set(args.family or DEFAULT_FAMILIES)
    selected = [spec for spec in FAMILIES if spec.name in requested]
    if args.sample_count is not None:
        if args.sample_count < 1000 or args.sample_count % 1000:
            parser.error("ImageNet --sample-count must be a positive multiple of 1000.")
        selected = [replace(spec, sample_count=args.sample_count) if spec.name.startswith("imagenet1k_") else spec for spec in selected]
    try:
        write_json(
            status_path,
            {
                "state": "running",
                "families": [spec.name for spec in selected],
                "updated_at_utc": utc_now(),
            },
        )
        for position, spec in enumerate(selected, start=1):
            print(f"[{position}/{len(selected)}] START {spec.name}", flush=True)
            evaluate_family(project_root, spec, output_root)
            print(f"[{position}/{len(selected)}] END {spec.name}", flush=True)
        write_json(
            status_path,
            {
                "state": "completed",
                "families": [spec.name for spec in selected],
                "updated_at_utc": utc_now(),
            },
        )
    except Exception as error:
        write_json(
            status_path,
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "updated_at_utc": utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
