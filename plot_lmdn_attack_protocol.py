from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import patheffects
from matplotlib import pyplot as plt


COLORS = {
    "baseline": "#66717D",
    "gmr": "#3B6F8F",
    "sleep1": "#E09B36",
    "sleep20": "#C94F4F",
}

LINESTYLES = {
    "baseline": "--",
    "gmr": "--",
    "sleep1": "-",
    "sleep20": "-",
}

TITLES = {
    "mnist_mlp": "MNIST - MLP",
    "cifar100_wrn28_10": "CIFAR-100 - WideResNet-28-10",
    "imagenet1k_resnet18": "ImageNet-1K Adversarial Evaluation (ResNet-18)",
    "imagenet1k_resnet50": "ImageNet-1K Adversarial Evaluation (ResNet-50)",
}


def add_curve(axis, values: list[dict], key: str, label: str) -> None:
    x = [float(item["epsilon_pixels"]) for item in values]
    y = [100.0 * float(item["accuracy"]) for item in values]
    color = COLORS[key]
    (line,) = axis.plot(
        x,
        y,
        color=color,
        linestyle=LINESTYLES[key],
        linewidth=2.4,
        marker="o",
        markersize=4.2,
        label=label,
        zorder=3,
    )
    line.set_path_effects(
        [
            patheffects.SimpleLineShadow(
                offset=(1.2, -1.2), alpha=0.24, rho=0.9
            ),
            patheffects.Normal(),
        ]
    )


def plot(result_path: Path) -> Path:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharey=True)
    panels = (
        ("whitebox_fgsm", "White-box FGSM"),
        ("whitebox_ifgsm", "White-box IFGSM"),
        ("blackbox_fgsm", "Baseline-source black-box FGSM"),
        ("blackbox_ifgsm", "Baseline-source black-box IFGSM"),
    )
    for axis, (section, title) in zip(axes.flat, panels):
        for key, values in result["results"][section].items():
            add_curve(axis, values, key, result["models"][key]["label"])
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(r"$\epsilon$ (pixel values / 255)")
        axis.grid(axis="y", linestyle="--", color="#A8A8A8", alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="best")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(0.0, 100.0)
    fig.suptitle(
        TITLES.get(result["family"], result["family"].replace("_", " ")),
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    output = result_path.with_suffix(".png")
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completed FGSM/IFGSM evaluation results.")
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "Results" / "lmdn_attack_protocol")
    args = parser.parse_args()
    root = args.input_dir
    for result_path in sorted(root.glob("*.json")):
        if result_path.name == "status.json":
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result.get("schema_version", 0)) != 2:
            continue
        if result.get("state") != "completed":
            raise ValueError(f"Cannot plot incomplete results: {result_path}")
        output = plot(result_path)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
