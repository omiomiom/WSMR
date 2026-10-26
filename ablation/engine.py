from __future__ import annotations

import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wslgmr import engine as core_engine
from wslgmr.models.common import NormalizedModel
from wslgmr.regularization.gmr import (
    choose_competitors,
    compute_gmr_quantities,
    compute_margins,
)
from wslgmr.regularization.local_adjoint import LocalAdjointCapture
from wslgmr.regularization.wake_sleep import (
    WakeSleepAccumulator,
    build_layer_gmr_conditions,
)
from wslgmr.utils import synchronize

from ablation.accumulators import (
    ExactDelayedAccumulator,
)


SUPPORTED_VARIANTS = {
    "first_order_only",
    "local_sensitivity_only",
    "exact_delayed",
    "full",
}


def _make_accumulator(config: dict, epoch: int):
    variant = str(config["ablation"]["variant"]).lower()
    if variant == "first_order_only":
        return None
    if variant == "exact_delayed":
        return ExactDelayedAccumulator(config["wake_sleep"], epoch)
    return WakeSleepAccumulator(config["wake_sleep"], epoch)


def train_one_epoch(
    *,
    model: NormalizedModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: dict,
    epoch: int,
) -> dict[str, float]:
    variant = str(config["ablation"]["variant"]).lower()
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported ablation variant: {variant}")
    weight = core_engine._regularizer_weight(config)
    accumulator = _make_accumulator(config, epoch)
    interval = int(config["wake_sleep"]["sleep_interval"])
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
    diagnostic_sums = torch.zeros(3, device=device, dtype=torch.float64)
    diagnostic_batches = 0
    progress = tqdm(
        loader,
        desc=f"ablation:{variant} {epoch:03d}/{config['training']['epochs']}",
        dynamic_ncols=True,
        leave=False,
    )
    update_interval = max(
        1, int(config["training"]["progress_update_interval"])
    )

    for batch_index, (images, targets) in enumerate(progress, start=1):
        batch_size = int(targets.size(0))
        accepts_batch_size = (
            accumulator is None
            or not accumulator.pending
            or accumulator.accepts_batch_size(batch_size)
        )
        if accumulator is not None and not accepts_batch_size:
            core_engine._sleep_and_record(
                accumulator, optimizer, weight, sleep_events
            )

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        images = images.detach().requires_grad_(True)

        if variant == "first_order_only":
            logits = model(images)
            ce_loss = F.cross_entropy(logits, targets)
            competitors = choose_competitors(logits, targets)
            margins = compute_margins(logits, targets, competitors)
            input_gradients = torch.autograd.grad(
                margins.sum(),
                images,
                create_graph=False,
                retain_graph=True,
            )[0]
            quantities = compute_gmr_quantities(
                margins, input_gradients, config["regularizer"]
            )
        else:
            with LocalAdjointCapture(model) as capture:
                logits = model(images)
            ce_loss = F.cross_entropy(logits, targets)
            competitors = choose_competitors(logits, targets)
            margins = compute_margins(logits, targets, competitors)
            records, input_gradients = capture.compute(margins, images)
            quantities = compute_gmr_quantities(
                margins, input_gradients, config["regularizer"]
            )
        if variant != "first_order_only":
            conditions = build_layer_gmr_conditions(
                records,
                quantities["kappa"],
                model,
                config["regularizer"],
            )
            if accumulator is None:
                raise RuntimeError("This ablation requires an accumulator.")
            accumulator.update_records(
                records, conditions.kappas, conditions.metrics
            )
            records.clear()

        regularizer_value = quantities["regularizer"]
        with torch.no_grad():
            activation = quantities["activation"].float()
            rho = quantities["rho"].float()
            diagnostic_sums.add_(
                torch.stack(
                    (
                        activation.mean(),
                        activation.gt(0.5).float().mean(),
                        rho.mean(),
                    )
                ).to(dtype=torch.float64)
            )
            diagnostic_batches += 1

        if variant == "local_sensitivity_only":
            ce_loss.backward()
        else:
            surrogate = (-quantities["activation"].detach() * margins).mean()
            (ce_loss + weight * surrogate).backward()

        core_engine._ensure_finite_gradients(model)
        clip_norm = config["training"]["gradient_clip_norm"]
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(clip_norm)
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if accumulator is not None and accumulator.batch_count >= interval:
            core_engine._sleep_and_record(
                accumulator, optimizer, weight, sleep_events
            )

        with torch.no_grad():
            batch_correct = logits.argmax(1).eq(targets).sum()
            ce_value = ce_loss.detach()
            reg_value = regularizer_value.detach()
            correct_count.add_(batch_correct)
            ce_sum.add_(ce_value, alpha=batch_size)
            regularizer_sum.add_(reg_value, alpha=batch_size)
        sample_count += batch_size
        if batch_index % update_interval == 0:
            progress.set_postfix(
                ce=f"{float(ce_value):.3f}",
                reg=f"{float(reg_value):.3f}",
                acc=f"{int(batch_correct) / batch_size:.3f}",
            )

    if accumulator is not None and accumulator.pending:
        core_engine._sleep_and_record(
            accumulator, optimizer, weight, sleep_events
        )

    synchronize(device)
    seconds = time.perf_counter() - started
    diagnostics = (
        diagnostic_sums / diagnostic_batches
        if diagnostic_batches
        else torch.full_like(diagnostic_sums, float("nan"))
    ).tolist()
    metrics = {
        "train_accuracy": int(correct_count) / sample_count,
        "train_ce": float(ce_sum) / sample_count,
        "train_regularizer": float(regularizer_sum) / sample_count,
        "train_seconds": seconds,
        "sleep_updates": float(len(sleep_events)),
        "sleep_mean_batches": core_engine._mean_event(
            sleep_events, "sleep_batches"
        ),
        "sleep_mean_update_norm": core_engine._mean_event(
            sleep_events, "sleep_update_norm"
        ),
        "diagnostic_gmr_activation_mean": diagnostics[0],
        "diagnostic_gmr_active_fraction": diagnostics[1],
        "diagnostic_gmr_rho_mean": diagnostics[2],
    }
    if device.type == "cuda":
        metrics["gpu_peak_allocated_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        metrics["gpu_peak_reserved_gb"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )
    return metrics
