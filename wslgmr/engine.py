from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .datasets import DataBundle, build_dataloaders
from .evaluation import (
    calibrated_batchnorm,
    evaluate,
    representative_calibration_loader,
)
from .models import build_model
from .models.common import NormalizedModel
from .regularization.gmr import (
    choose_competitors,
    compute_exact_gmr_loss,
    compute_gmr_quantities,
    compute_margins,
)
from .regularization.local_adjoint import (
    LocalAdjointCapture,
)
from .regularization.wake_sleep import (
    WakeSleepAccumulator,
    build_layer_gmr_conditions,
)
from .run_history import (
    begin_resume_history,
    read_history,
    remove_future_checkpoints,
    update_resume_history,
    write_resume_artifacts,
)
from .utils import (
    select_device,
    set_seed,
    synchronize,
    write_csv,
    write_json,
)


def _plot_history_isolated(history_path: Path, output_path: Path) -> None:
    """Render curves in a subprocess.

    Isolate the PyTorch and Matplotlib OpenMP runtimes to avoid conflicts
    when plotting during training on Windows.
    """

    subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("render_plots.py")),
            "history",
            str(history_path),
            str(output_path),
        ],
        check=True,
    )


def _regularizer_weight(config: dict) -> float:
    # Scale the GMR loss by half the configured regularization weight.
    return 0.5 * float(config["regularizer"]["weight"])


def _build_optimizer(
    model: nn.Module,
    config: dict,
) -> torch.optim.Optimizer:
    training = config["training"]
    name = str(training["optimizer"]).lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(training["learning_rate"]),
            momentum=float(training["momentum"]),
            weight_decay=float(training["weight_decay"]),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
    raise ValueError("training.optimizer must be sgd or adamw.")


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: dict,
    epoch: int,
) -> float:
    # Use a constant learning rate when no schedule is configured.
    schedule = config["training"].get("lr_schedule") or {}
    learning_rate = float(config["training"]["learning_rate"])
    for start_epoch, value in sorted(
        ((int(key), float(value)) for key, value in schedule.items())
    ):
        if epoch >= start_epoch:
            learning_rate = value
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _ensure_finite_gradients(model: nn.Module) -> None:
    named_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if not named_gradients:
        return

    # Reduce finite checks on the device before synchronizing with the CPU.
    # Inspect individual parameters only to identify a non-finite gradient.
    finite_flags = torch.stack(
        [torch.isfinite(gradient).all() for _, gradient in named_gradients]
    )
    if bool(finite_flags.all()):
        return
    for name, gradient in named_gradients:
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Non-finite gradient in {name}.")


def _sleep_and_record(
    accumulator: WakeSleepAccumulator,
    optimizer: torch.optim.Optimizer,
    weight: float,
    events: list[dict[str, float]],
) -> None:
    events.append(
        accumulator.sleep(
            optimizer=optimizer,
            regularizer_weight=weight,
        )
    )


def _mean_event(events: list[dict[str, float]], key: str) -> float:
    values = [float(event[key]) for event in events if key in event]
    return sum(values) / len(values) if values else float("nan")


def train_one_epoch(
    *,
    model: NormalizedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    epoch: int,
) -> dict[str, float]:
    method = config["experiment"]["method"]
    weight = (
        0.0 if method == "baseline" else _regularizer_weight(config)
    )
    accumulator = (
        WakeSleepAccumulator(
            config["wake_sleep"],
            epoch,
        )
        if method == "replace_gi"
        else None
    )
    interval = (
        int(config["wake_sleep"]["sleep_interval"])
        if method == "replace_gi"
        else 0
    )
    sleep_events: list[dict[str, float]] = []
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()

    sample_count = 0
    correct_count = torch.zeros((), device=device, dtype=torch.int64)
    ce_sum = torch.zeros((), device=device, dtype=torch.float64)
    regularizer_sum = torch.zeros((), device=device, dtype=torch.float64)
    layer_diagnostic_sums = torch.zeros(
        3, device=device, dtype=torch.float64
    )
    layer_diagnostic_batches = 0
    progress = tqdm(
        loader,
        desc=f"{method} {epoch:03d}/{config['training']['epochs']}",
        dynamic_ncols=True,
        leave=False,
    )
    update_interval = max(
        1, int(config["training"]["progress_update_interval"])
    )

    for batch_index, (images, targets) in enumerate(progress, start=1):
        batch_size = int(targets.size(0))
        if (
            accumulator is not None
            and accumulator.pending
            and not accumulator.accepts_batch_size(batch_size)
        ):
            _sleep_and_record(accumulator, optimizer, weight, sleep_events)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        regularizer_value = torch.zeros((), device=device)

        if method == "baseline":
            logits = model(images)
            ce_loss = F.cross_entropy(logits, targets)
            ce_loss.backward()

        elif method == "exact_g0":
            images = images.detach().requires_grad_(True)
            logits = model(images)
            ce_loss = F.cross_entropy(logits, targets)
            competitors = choose_competitors(logits, targets)
            margins = compute_margins(logits, targets, competitors)
            input_gradients = torch.autograd.grad(
                margins.sum(),
                images,
                create_graph=True,
                retain_graph=True,
            )[0]
            regularizer_value = compute_exact_gmr_loss(
                margins,
                input_gradients,
                config["regularizer"],
            )
            (ce_loss + weight * regularizer_value).backward()

        elif method == "replace_gi":
            images = images.detach().requires_grad_(True)
            with LocalAdjointCapture(model) as capture:
                logits = model(images)
            ce_loss = F.cross_entropy(logits, targets)
            competitors = choose_competitors(logits, targets)
            margins = compute_margins(logits, targets, competitors)
            records, input_gradients = capture.compute(
                margins,
                images,
            )
            quantities = compute_gmr_quantities(
                margins,
                input_gradients,
                config["regularizer"],
            )
            regularizer_value = quantities["regularizer"]
            if accumulator is None:
                raise RuntimeError("replace_gi accumulator was not created.")
            conditions = build_layer_gmr_conditions(
                records,
                quantities["kappa"],
                model,
                config["regularizer"],
            )
            with torch.no_grad():
                activation = quantities["activation"].float()
                rho = quantities["rho"].float()
                layer_diagnostic_sums.add_(
                    torch.stack(
                        (
                            activation.mean(),
                            activation.gt(0.5).float().mean(),
                            rho.mean(),
                        )
                    ).to(dtype=torch.float64)
                )
                layer_diagnostic_batches += 1
            accumulator.update_records(
                records,
                conditions.kappas,
                conditions.metrics,
            )
            records.clear()
            surrogate = (-quantities["activation"].detach() * margins).mean()
            (ce_loss + weight * surrogate).backward()
        else:
            raise AssertionError(f"Unhandled method: {method}")

        _ensure_finite_gradients(model)
        clip_norm = config["training"]["gradient_clip_norm"]
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_norm)
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if (
            accumulator is not None
            and accumulator.batch_count >= interval
        ):
            _sleep_and_record(accumulator, optimizer, weight, sleep_events)

        with torch.no_grad():
            batch_correct = logits.argmax(1).eq(targets).sum()
            ce_value_tensor = ce_loss.detach()
            reg_value_tensor = regularizer_value.detach()
            correct_count.add_(batch_correct)
            ce_sum.add_(ce_value_tensor, alpha=batch_size)
            regularizer_sum.add_(reg_value_tensor, alpha=batch_size)
        sample_count += batch_size
        if batch_index % update_interval == 0:
            # Transfer scalar metrics to the CPU only when updating the progress display.
            correct = int(batch_correct)
            ce_value = float(ce_value_tensor)
            reg_value = float(reg_value_tensor)
            progress.set_postfix(
                ce=f"{ce_value:.3f}",
                reg=f"{reg_value:.3f}",
                acc=f"{correct / batch_size:.3f}",
            )

    if accumulator is not None and accumulator.pending:
        _sleep_and_record(accumulator, optimizer, weight, sleep_events)

    synchronize(device)
    seconds = time.perf_counter() - started
    correct_count_value = int(correct_count)
    ce_sum_value = float(ce_sum)
    regularizer_sum_value = float(regularizer_sum)
    if layer_diagnostic_batches > 0:
        layer_diagnostic_values = (
            layer_diagnostic_sums / layer_diagnostic_batches
        ).tolist()
    else:
        layer_diagnostic_values = [float("nan")] * 3
    metrics = {
        "train_accuracy": correct_count_value / sample_count,
        "train_ce": ce_sum_value / sample_count,
        "train_regularizer": regularizer_sum_value / sample_count,
        "train_seconds": seconds,
        "sleep_updates": float(len(sleep_events)),
        "sleep_mean_batches": _mean_event(
            sleep_events, "sleep_batches"
        ),
        "sleep_mean_update_norm": _mean_event(
            sleep_events, "sleep_update_norm"
        ),
        "diagnostic_gmr_activation_mean": layer_diagnostic_values[0],
        "diagnostic_gmr_active_fraction": layer_diagnostic_values[1],
        "diagnostic_gmr_rho_mean": layer_diagnostic_values[2],
    }
    if device.type == "cuda":
        metrics["gpu_peak_allocated_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        metrics["gpu_peak_reserved_gb"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )
    return metrics


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    data: DataBundle,
    branch_id: int,
) -> dict:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "history": history,
        "branch_id": int(branch_id),
        "train_generator_state": data.train_generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _cpu_rng_state(state: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8:
        raise TypeError(f"{name} must be a torch.ByteTensor.")
    # torch.load(..., map_location=device) also moves serialized RNG tensors
    # to CUDA. CPU generators and torch.set_rng_state require CPU tensors.
    return state.detach().cpu().contiguous()


def _load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: DataBundle,
    device: torch.device,
) -> tuple[int, list[dict], int]:
    try:
        payload = torch.load(
            path, map_location=device, weights_only=False
        )
    except TypeError:
        payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    data.train_generator.set_state(
        _cpu_rng_state(
            payload["train_generator_state"],
            "train_generator_state",
        )
    )
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(
        _cpu_rng_state(payload["torch_rng_state"], "torch_rng_state")
    )
    if (
        device.type == "cuda"
        and payload.get("cuda_rng_state") is not None
    ):
        torch.cuda.set_rng_state_all(
            [
                _cpu_rng_state(state, f"cuda_rng_state[{index}]")
                for index, state in enumerate(payload["cuda_rng_state"])
            ]
        )
    return (
        int(payload["epoch"]),
        list(payload.get("history", [])),
        int(payload.get("branch_id", 0)),
    )


def _resolve_resume_path(
    output_dir: Path,
    resume: str | None,
) -> tuple[Path | None, int | None]:
    if resume is None:
        return None, None
    token = str(resume).strip().lower()
    if token in {"lastest", "latest"}:
        path = output_dir / "lastest.pt"
        requested_epoch = None
    else:
        try:
            requested_epoch = int(token)
        except ValueError as error:
            raise ValueError(
                "--resume must be used alone or followed by a positive epoch."
            ) from error
        if requested_epoch < 1:
            raise ValueError("--resume epoch must be positive.")
        path = (
            output_dir
            / "checkpoints"
            / f"epoch_{requested_epoch:03d}.pt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path, requested_epoch


def _config_differences(
    stored,
    current,
    *,
    prefix: str = "",
) -> list[str]:
    if isinstance(stored, dict) and isinstance(current, dict):
        differences: list[str] = []
        keys = sorted(set(stored) | set(current))
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in stored:
                differences.append(
                    f"{path}: <missing> -> {current[key]!r}"
                )
            elif key not in current:
                differences.append(
                    f"{path}: {stored[key]!r} -> <missing>"
                )
            else:
                differences.extend(
                    _config_differences(
                        stored[key],
                        current[key],
                        prefix=path,
                    )
                )
        return differences
    if isinstance(stored, list) and isinstance(current, list):
        if len(stored) != len(current):
            return [
                f"{prefix}.length: {len(stored)} -> {len(current)}"
            ]
        differences = []
        for index, (stored_value, current_value) in enumerate(
            zip(stored, current)
        ):
            differences.extend(
                _config_differences(
                    stored_value,
                    current_value,
                    prefix=f"{prefix}[{index}]",
                )
            )
        return differences
    if stored != current or type(stored) is not type(current):
        return [f"{prefix}: {stored!r} -> {current!r}"]
    return []


def _config_for_resume_comparison(config: dict) -> dict:
    """Return config fields used to check whether a run can be resumed."""
    comparable = json.loads(json.dumps(config))
    experiment = comparable.get("experiment")
    if isinstance(experiment, dict):
        # A saved resume flag is not part of the training configuration.
        # Resuming is selected through the command line.
        experiment.pop("resume", None)
    return comparable


def _report_resume_config_changes(
    output_dir: Path,
    current: dict,
) -> list[str]:
    path = output_dir / "resolved_config.json"
    if not path.is_file():
        print(
            "resume config comparison unavailable: previous "
            f"resolved config not found at {path}"
        )
        return []
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(
            "resume config comparison unavailable: cannot read "
            f"{path}: {error}"
        )
        return []
    differences = _config_differences(
        _config_for_resume_comparison(stored),
        _config_for_resume_comparison(current),
    )
    if differences:
        print(
            "resume config changes since previous run "
            f"({len(differences)}):"
        )
        for difference in differences:
            print(f"  - {difference}")
    else:
        print("resume config changes since previous run: none")
    return differences


def _record_config_history(
    output_dir: Path,
    current: dict,
    *,
    kind: str,
    changes: list[str],
    resume_path: Path | None = None,
    resume_epoch: int | None = None,
    branch_id: int | None = None,
) -> None:
    history_path = output_dir / "config_history.json"
    state: dict = {"version": 1, "events": []}
    if history_path.is_file():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            if (
                not isinstance(loaded, dict)
                or not isinstance(loaded.get("events"), list)
            ):
                raise ValueError("expected an object containing an events list")
            state = loaded
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(
                "warning: existing config history is unreadable; "
                f"starting a new history: {error}"
            )

    events = state["events"]
    previous_path = output_dir / "resolved_config.json"
    if not events and previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous is not None and (
            kind == "resume"
            or _config_for_resume_comparison(previous)
            != _config_for_resume_comparison(current)
        ):
            events.append(
                {
                    "event_id": 1,
                    "kind": "previous_run_imported",
                    "recorded_at_utc": None,
                    "resume_checkpoint": None,
                    "resume_epoch": None,
                    "branch_id": None,
                    "changes_from_previous": [],
                    "config": previous,
                }
            )

    checkpoint_text = None
    if resume_path is not None:
        try:
            checkpoint_text = str(resume_path.relative_to(output_dir))
        except ValueError:
            checkpoint_text = str(resume_path)
    event = {
        "event_id": len(events) + 1,
        "kind": kind,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_checkpoint": checkpoint_text,
        "resume_epoch": (
            int(resume_epoch) if resume_epoch is not None else None
        ),
        "branch_id": int(branch_id) if branch_id is not None else None,
        "changes_from_previous": list(changes),
        "config": current,
    }
    events.append(event)
    write_json(history_path, state)
    print(
        f"config history event {event['event_id']} recorded: "
        f"{history_path}"
    )


def _last_recorded_metric(
    history: list[dict],
    metric_name: str,
) -> float:
    for row in reversed(history):
        if metric_name in row:
            return float(row[metric_name])
    return -math.inf


def run_training(
    config: dict,
    *,
    resume: str | None = None,
) -> list[dict]:
    device = select_device(config["experiment"]["device"])
    set_seed(
        int(config["experiment"]["seed"]),
        bool(config["experiment"]["deterministic"]),
    )
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path, requested_epoch = _resolve_resume_path(
        output_dir,
        resume,
    )
    resume_config_changes: list[str] = []
    if resume_path is not None:
        resume_config_changes = _report_resume_config_changes(
            output_dir,
            config,
        )
    else:
        _record_config_history(
            output_dir,
            config,
            kind="fresh_start",
            changes=[],
            branch_id=0,
        )
        write_json(output_dir / "resolved_config.json", config)
    previous_active_history = read_history(
        output_dir / "history.json"
    )

    data = build_dataloaders(config)
    model = build_model(
        config,
        input_shape=data.input_shape,
        mean=data.mean,
        std=data.std,
        num_classes=data.num_classes,
    ).to(device)
    optimizer = _build_optimizer(model, config)
    checkpoint_path = output_dir / "lastest.pt"
    start_epoch = 0
    history: list[dict] = []
    branch_id = 0
    resume_state: dict | None = None
    if resume_path is not None:
        start_epoch, history, checkpoint_branch_id = _load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            data=data,
            device=device,
        )
        if requested_epoch is not None and start_epoch != requested_epoch:
            raise RuntimeError(
                "Requested resume epoch does not match checkpoint content: "
                f"requested {requested_epoch}, found {start_epoch}."
            )
        if not previous_active_history:
            previous_active_history = history
        resume_state, branch_id = begin_resume_history(
            output_dir=output_dir,
            previous_active_history=previous_active_history,
            checkpoint_history=history,
            checkpoint_branch_id=checkpoint_branch_id,
            resume_epoch=start_epoch,
            checkpoint_path=resume_path,
        )
        _record_config_history(
            output_dir,
            config,
            kind="resume",
            changes=resume_config_changes,
            resume_path=resume_path,
            resume_epoch=start_epoch,
            branch_id=branch_id,
        )
        # Save the current configuration for subsequent resume checks.
        write_json(output_dir / "resolved_config.json", config)
        removed = remove_future_checkpoints(
            output_dir,
            resume_epoch=start_epoch,
        )
        if removed:
            print(
                "removed future periodic checkpoints: "
                + ", ".join(path.name for path in removed)
            )
        write_json(output_dir / "history.json", history)
        write_csv(output_dir / "history.csv", history)
        _plot_history_isolated(
            output_dir / "history.json",
            output_dir / "curves.png",
        )
        write_resume_artifacts(resume_state, output_dir)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"dataset: {config['dataset']}")
    print(f"model: {config['model']} ({parameter_count:,} parameters)")
    print(f"method: {config['experiment']['method']}")
    print(f"output: {output_dir}")
    metric_name = "clean_top1_accuracy"
    best_metric = max(
        (float(row.get(metric_name, -math.inf)) for row in history),
        default=-math.inf,
    )
    best_path = output_dir / "best.pt"
    if resume_path is not None:
        resume_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=start_epoch,
            history=history,
            data=data,
            branch_id=branch_id,
        )
        _save_checkpoint(checkpoint_path, resume_payload)
    if requested_epoch is not None:
        # A numbered resume rolls the single best-weight file back to the
        # selected branch point. Future evaluations overwrite it normally.
        _save_checkpoint(resume_path, resume_payload)
        _save_checkpoint(best_path, resume_payload)
        best_metric = _last_recorded_metric(history, metric_name)
    elif resume_path is not None and not best_path.exists():
        _save_checkpoint(best_path, resume_payload)
        best_metric = _last_recorded_metric(history, metric_name)

    total_epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch + 1, total_epochs + 1):
        learning_rate = _set_learning_rate(
            optimizer, config, epoch
        )
        train_metrics = train_one_epoch(
            model=model,
            loader=data.train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            epoch=epoch,
        )
        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            **train_metrics,
        }
        evaluate_every = int(config["evaluation"]["every_epochs"])
        if epoch % evaluate_every == 0 or epoch == total_epochs:
            evaluation_started = time.perf_counter()
            calibration_batches = int(
                config["evaluation"]["bn_calibration_batches"]
            )
            calibration_loader = representative_calibration_loader(
                data.calibration_loader,
                batches=calibration_batches,
                seed=int(config["experiment"]["seed"]) + 1,
            )
            with calibrated_batchnorm(
                model,
                calibration_loader,
                device,
                calibration_batches,
            ):
                row.update(
                    evaluate(
                        model,
                        data.eval_loader,
                        device,
                        noise_type=str(
                            config["evaluation"]["noise_type"]
                        ),
                        noise_sigma=float(
                            config["evaluation"]["noise_sigma"]
                        ),
                        noise_clip=bool(
                            config["evaluation"]["noise_clip"]
                        ),
                        noise_seed=int(
                            config["evaluation"]["noise_seed"]
                        ),
                        foreground_flip_probability=float(
                            config["evaluation"][
                                "foreground_flip_probability"
                            ]
                        ),
                        background_flip_probability=float(
                            config["evaluation"][
                                "background_flip_probability"
                            ]
                        ),
                        binary_threshold=float(
                            config["evaluation"]["binary_threshold"]
                        ),
                    )
                )
            row["evaluation_seconds"] = (
                time.perf_counter() - evaluation_started
            )
        history.append(row)

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            data=data,
            branch_id=branch_id,
        )
        checkpoint_every = int(config["training"]["checkpoint_every"])
        # The rolling checkpoint is always current, independently of the
        # periodic archival interval.
        _save_checkpoint(checkpoint_path, payload)
        if epoch % checkpoint_every == 0:
            _save_checkpoint(
                output_dir
                / "checkpoints"
                / f"epoch_{epoch:03d}.pt",
                payload,
            )
        metric = float(row.get(metric_name, -math.inf))
        if metric > best_metric:
            best_metric = metric
            _save_checkpoint(best_path, payload)

        write_json(output_dir / "history.json", history)
        write_csv(output_dir / "history.csv", history)
        _plot_history_isolated(
            output_dir / "history.json",
            output_dir / "curves.png",
        )
        if resume_state is not None:
            update_resume_history(
                resume_state,
                active_history=history,
            )
            write_resume_artifacts(resume_state, output_dir)
        write_json(
            output_dir / "status.json",
            {
                "state": (
                    "completed" if epoch == total_epochs else "running"
                ),
                "completed_epoch": epoch,
                "target_epochs": total_epochs,
                "best_metric": best_metric,
                "best_metric_name": metric_name,
                "active_branch_id": branch_id,
                "resume_count": (
                    len(resume_state["resume_events"])
                    if resume_state is not None
                    else 0
                ),
            },
        )
        print(
            f"epoch {epoch:03d}/{total_epochs} | "
            f"lr={row['learning_rate']:.6g} | "
            f"train={100 * row['train_accuracy']:.2f}% | "
            f"clean@1="
            f"{100 * row.get('clean_top1_accuracy', float('nan')):.2f}% | "
            f"clean@5="
            f"{100 * row.get('clean_top5_accuracy', float('nan')):.2f}% | "
            f"noise@1="
            f"{100 * row.get('noisy_top1_accuracy', float('nan')):.2f}% | "
            f"noise@5="
            f"{100 * row.get('noisy_top5_accuracy', float('nan')):.2f}% | "
            f"train_time={row['train_seconds']:.1f}s"
        )
    return history
