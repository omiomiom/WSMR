from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


def select_evaluation_indices(
    dataset: Dataset,
    *,
    sample_count: int | None,
    num_classes: int,
    seed: int,
    stratified: bool,
) -> list[int]:
    """Select a deterministic full or class-balanced evaluation subset."""
    dataset_size = len(dataset)
    if sample_count is None or sample_count >= dataset_size:
        return list(range(dataset_size))
    if sample_count < 1:
        raise ValueError("sample_count must be positive or None")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if not stratified:
        return torch.randperm(dataset_size, generator=generator)[
            :sample_count
        ].tolist()

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError(
            "Stratified selection requires dataset.targets to be available."
        )
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    missing = sorted(set(range(num_classes)) - set(by_class))
    if missing:
        raise RuntimeError(f"Evaluation dataset is missing classes: {missing[:10]}")

    base = sample_count // num_classes
    remainder = sample_count % num_classes
    selected: list[int] = []
    for class_index in range(num_classes):
        class_indices = by_class[class_index]
        take = base + int(class_index < remainder)
        if take > len(class_indices):
            raise ValueError(
                f"Class {class_index} contains {len(class_indices)} samples, "
                f"but {take} were requested."
            )
        permutation = torch.randperm(
            len(class_indices), generator=generator
        )[:take]
        selected.extend(class_indices[int(offset)] for offset in permutation)
    return selected


def indices_sha256(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def materialize_subset(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[Tensor, Tensor]:
    """Load a deterministic subset into CPU memory for repeatable attacks."""
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )
    images: list[Tensor] = []
    targets: list[Tensor] = []
    for batch_images, batch_targets in loader:
        images.append(batch_images.cpu())
        targets.append(batch_targets.cpu())
    return torch.cat(images, dim=0), torch.cat(targets, dim=0)

