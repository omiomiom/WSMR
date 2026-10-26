from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


def representative_calibration_loader(
    source_loader: DataLoader,
    *,
    batches: int,
    seed: int,
) -> DataLoader:
    """Build a reproducible uniform calibration sample.

    Sampling uses an independent generator, so calibration neither consumes
    the training sampler state nor inherits a class-sorted dataset order.
    Rebuilding this loader for every evaluation also reproduces the same
    stochastic training transforms without persistent worker state.
    """
    if batches < 0:
        raise ValueError("batches cannot be negative")
    if source_loader.batch_size is None:
        raise ValueError("calibration requires a fixed batch size")
    batch_size = int(source_loader.batch_size)
    sample_count = min(
        len(source_loader.dataset),
        int(batches) * batch_size,
    )
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(int(seed))
    indices = torch.randperm(
        len(source_loader.dataset),
        generator=sampler_generator,
    )[:sample_count].tolist()
    worker_generator = torch.Generator(device="cpu")
    worker_generator.manual_seed(int(seed))
    return DataLoader(
        Subset(source_loader.dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=source_loader.num_workers,
        pin_memory=source_loader.pin_memory,
        persistent_workers=False,
        worker_init_fn=source_loader.worker_init_fn,
        generator=worker_generator,
    )


def _topk_correct(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int]:
    top5 = logits.topk(k=min(5, logits.size(1)), dim=1).indices
    matches = top5.eq(targets.view(-1, 1))
    top1_correct = int(matches[:, :1].any(dim=1).sum())
    top5_correct = int(matches.any(dim=1).sum())
    return top1_correct, top5_correct


@contextmanager
def calibrated_batchnorm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    batches: int,
):
    batchnorms = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    snapshots = [
        (
            module,
            module.running_mean.detach().clone()
            if module.running_mean is not None
            else None,
            module.running_var.detach().clone()
            if module.running_var is not None
            else None,
            module.num_batches_tracked.detach().clone()
            if module.num_batches_tracked is not None
            else None,
            module.training,
            module.momentum,
        )
        for module in batchnorms
    ]
    model_was_training = model.training
    try:
        model.eval()
        if batches > 0 and batchnorms:
            for module in batchnorms:
                # ``None`` selects cumulative averaging: after k equal-sized
                # batches every batch has weight 1/k, and no reset-to-one
                # variance term survives the first calibration batch.
                module.momentum = None
                module.reset_running_stats()
                module.train()
            with torch.no_grad():
                for index, (images, _) in enumerate(loader):
                    if index >= batches:
                        break
                    model(images.to(device, non_blocking=True))
        model.eval()
        yield
    finally:
        for module, mean, variance, count, was_training, momentum in snapshots:
            if mean is not None:
                module.running_mean.copy_(mean)
            if variance is not None:
                module.running_var.copy_(variance)
            if count is not None:
                module.num_batches_tracked.copy_(count)
            module.momentum = momentum
            module.train(was_training)
        model.train(model_was_training)


@torch.no_grad()
def apply_evaluation_noise(
    images: Tensor,
    *,
    generator: torch.Generator,
    noise_type: str,
    noise_sigma: float,
    noise_clip: bool,
    foreground_flip_probability: float,
    background_flip_probability: float,
    binary_threshold: float,
) -> Tensor:
    if noise_type == "gaussian":
        noise = torch.randn(
            images.shape,
            generator=generator,
            device=images.device,
            dtype=images.dtype,
        )
        noisy_images = images + float(noise_sigma) * noise
        if noise_clip:
            noisy_images.clamp_(0.0, 1.0)
        return noisy_images
    if noise_type == "binary_flip":
        foreground = images >= float(binary_threshold)
        probabilities = torch.where(
            foreground,
            torch.as_tensor(
                foreground_flip_probability,
                device=images.device,
                dtype=images.dtype,
            ),
            torch.as_tensor(
                background_flip_probability,
                device=images.device,
                dtype=images.dtype,
            ),
        )
        draws = torch.rand(
            images.shape,
            generator=generator,
            device=images.device,
            dtype=images.dtype,
        )
        return torch.where(draws < probabilities, 1.0 - images, images)
    raise ValueError(f"Unsupported evaluation noise type: {noise_type}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    noise_type: str,
    noise_sigma: float,
    noise_clip: bool,
    noise_seed: int,
    foreground_flip_probability: float,
    background_flip_probability: float,
    binary_threshold: float,
) -> dict[str, float]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    clean_loss = 0.0
    noisy_loss = 0.0
    clean_top1_correct = 0
    clean_top5_correct = 0
    noisy_top1_correct = 0
    noisy_top5_correct = 0
    sample_count = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        clean_logits = model(images)
        noisy_images = apply_evaluation_noise(
            images,
            generator=generator,
            noise_type=noise_type,
            noise_sigma=noise_sigma,
            noise_clip=noise_clip,
            foreground_flip_probability=foreground_flip_probability,
            background_flip_probability=background_flip_probability,
            binary_threshold=binary_threshold,
        )
        noisy_logits = model(noisy_images)
        batch_size = int(targets.size(0))
        clean_loss += float(
            F.cross_entropy(clean_logits, targets, reduction="sum")
        )
        noisy_loss += float(
            F.cross_entropy(noisy_logits, targets, reduction="sum")
        )
        clean_top1, clean_top5 = _topk_correct(clean_logits, targets)
        noisy_top1, noisy_top5 = _topk_correct(noisy_logits, targets)
        clean_top1_correct += clean_top1
        clean_top5_correct += clean_top5
        noisy_top1_correct += noisy_top1
        noisy_top5_correct += noisy_top5
        sample_count += batch_size
    return {
        "clean_loss": clean_loss / sample_count,
        "clean_top1_accuracy": clean_top1_correct / sample_count,
        "clean_top5_accuracy": clean_top5_correct / sample_count,
        "noisy_loss": noisy_loss / sample_count,
        "noisy_top1_accuracy": noisy_top1_correct / sample_count,
        "noisy_top5_accuracy": noisy_top5_correct / sample_count,
    }
