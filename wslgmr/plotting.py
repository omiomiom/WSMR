from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    panels = [
        ("train_accuracy", "Train Top-1 accuracy"),
        ("clean_top1_accuracy", "Clean Top-1 accuracy"),
        ("noisy_top1_accuracy", "Gaussian-noise Top-1 accuracy"),
        ("clean_top5_accuracy", "Clean Top-5 accuracy"),
        ("noisy_top5_accuracy", "Gaussian-noise Top-5 accuracy"),
        ("train_seconds", "Train seconds"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    for axis, (key, title) in zip(axes.flat, panels):
        values = [row.get(key, float("nan")) for row in rows]
        axis.plot(epochs, values, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_branched_history(state: dict, output_path: Path) -> None:
    panels = [
        ("train_accuracy", "Train Top-1 accuracy"),
        ("clean_top1_accuracy", "Clean Top-1 accuracy"),
        ("noisy_top1_accuracy", "Gaussian-noise Top-1 accuracy"),
        ("clean_top5_accuracy", "Clean Top-5 accuracy"),
        ("noisy_top5_accuracy", "Gaussian-noise Top-5 accuracy"),
        ("train_seconds", "Train seconds"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    active_id = int(state["active_branch_id"])
    color_map = plt.get_cmap("tab10")
    for branch in state["branches"]:
        branch_id = int(branch["branch_id"])
        parent_id = branch["parent_branch_id"]
        resume_epoch = branch["resume_from_epoch"]
        rows = [
            row
            for row in branch["history"]
            if (
                parent_id is None
                or int(row["epoch"]) >= int(resume_epoch)
            )
        ]
        if not rows:
            continue
        epochs = [int(row["epoch"]) for row in rows]
        is_active = branch_id == active_id
        for axis, (key, title) in zip(axes.flat, panels):
            values = [row.get(key, float("nan")) for row in rows]
            axis.plot(
                epochs,
                values,
                color=color_map(branch_id % 10),
                linewidth=2.4 if is_active else 1.4,
                alpha=1.0 if is_active else 0.72,
                label=f"branch {branch_id}",
            )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
    axes.flat[0].legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
