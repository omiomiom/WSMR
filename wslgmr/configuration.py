from __future__ import annotations

import copy
import runpy
from pathlib import Path


SUPPORTED_DATASETS = {"mnist", "cifar100", "imagenet1k"}
SUPPORTED_MODELS = {
    "mlp",
    "wrn28_2",
    "wrn28_10",
    "wrn40_10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
}


def load_config_file(path: Path) -> dict:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")
    namespace = runpy.run_path(str(resolved_path))
    config = namespace.get("CONFIG")
    if not isinstance(config, dict):
        raise TypeError(
            f"{resolved_path} must define a dictionary named CONFIG."
        )
    return copy.deepcopy(config)


def resolve_config(config: dict, *, project_root: Path) -> dict:
    resolved = copy.deepcopy(config)

    dataset = str(resolved["dataset"]).lower()
    model = str(resolved["model"]).lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            "dataset must be mnist, cifar100, or imagenet1k."
        )
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    if dataset == "mnist" and model != "mlp":
        raise ValueError("MNIST requires model='mlp'.")
    if dataset == "cifar100" and not model.startswith("wrn"):
        raise ValueError("CIFAR-100 requires a WRN model.")
    if dataset == "imagenet1k" and not model.startswith("resnet"):
        raise ValueError("ImageNet-1k requires a ResNet model.")
    resolved["dataset"] = dataset
    resolved["model"] = model

    method = str(resolved["experiment"]["method"]).lower()
    if method not in {"baseline", "exact_g0", "replace_gi"}:
        raise ValueError(
            "experiment.method must be baseline, exact_g0, or replace_gi."
        )
    resolved["experiment"]["method"] = method

    device = str(resolved["experiment"]["device"]).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("experiment.device must be auto, cpu, or cuda.")
    resolved["experiment"]["device"] = device

    if method == "replace_gi":
        hidden_metric = str(
            resolved["regularizer"].get(
                "hidden_gradient_metric", "euclidean"
            )
        ).lower()
        if hidden_metric not in {"euclidean", "activation_rms"}:
            raise ValueError(
                "regularizer.hidden_gradient_metric must be euclidean "
                "or activation_rms."
            )
        resolved["regularizer"]["hidden_gradient_metric"] = hidden_metric
        if float(
            resolved["regularizer"].get("hidden_metric_epsilon", 1.0e-6)
        ) <= 0.0:
            raise ValueError(
                "regularizer.hidden_metric_epsilon must be positive."
            )
        interval = int(resolved["wake_sleep"]["sleep_interval"])
        if interval < 1:
            raise ValueError("wake_sleep.sleep_interval must be positive.")
    if int(resolved["loader"]["batch_size"]) < 1:
        raise ValueError("loader.batch_size must be positive.")
    if int(resolved["loader"]["eval_batch_size"]) < 1:
        raise ValueError("loader.eval_batch_size must be positive.")
    if int(resolved["training"]["epochs"]) < 1:
        raise ValueError("training.epochs must be positive.")
    if int(resolved["training"]["checkpoint_every"]) < 1:
        raise ValueError("training.checkpoint_every must be positive.")
    if int(resolved["evaluation"]["every_epochs"]) < 1:
        raise ValueError("evaluation.every_epochs must be positive.")
    if int(resolved["evaluation"]["bn_calibration_batches"]) < 0:
        raise ValueError(
            "evaluation.bn_calibration_batches cannot be negative."
        )
    noise_type = str(resolved["evaluation"]["noise_type"]).lower()
    if noise_type not in {"gaussian", "binary_flip"}:
        raise ValueError(
            "evaluation.noise_type must be gaussian or binary_flip."
        )
    resolved["evaluation"]["noise_type"] = noise_type
    for key in (
        "foreground_flip_probability",
        "background_flip_probability",
        "binary_threshold",
    ):
        value = float(resolved["evaluation"][key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"evaluation.{key} must be in [0, 1].")
    paths = resolved["paths"]
    for key in ("dataset_root", "output_dir"):
        path = Path(paths[key]).expanduser()
        if not path.is_absolute():
            path = (project_root / path).resolve()
        paths[key] = str(path)
    resolved["_project_root"] = str(project_root.resolve())
    return resolved
