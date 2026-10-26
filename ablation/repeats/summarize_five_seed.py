"""Audit, summarize, and plot the paper's CIFAR-100 five-seed results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "Results"
OUTPUT_ROOT = RESULT_ROOT / "ablation" / "repeats"
SEEDS = (42, 43, 44, 45, 46)
T95_DF4 = 2.776445105
REQUIRED_METRICS = (
    "clean_top1_accuracy",
    "noisy_top1_accuracy",
    "train_seconds",
    "evaluation_seconds",
    "gpu_peak_allocated_gb",
    "gpu_peak_reserved_gb",
)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    directory_pattern: str
    color: str
    linestyle: str = "-"
    group: str = "ablation"


VARIANTS = (
    Variant("baseline", "Baseline", "CIFAR100_WRN28-10_baseline_seed{seed}", "#66717D", "--", "formal"),
    Variant("exact_gmr", "Exact GMR", "CIFAR100_WRN28-10_exact_gmr_beta5_z1p5_weight0p05_seed{seed}", "#3B6F8F", "--", "formal"),
    Variant("formal_t1", "Wake-Sleep (T=1)", "CIFAR100_WRN28-10_wake_sleep_t1_beta5_z1p5_weight0p05_seed{seed}", "#E09B36", "-", "formal"),
    Variant("formal_t20", "Wake-Sleep (T=20)", "CIFAR100_WRN28-10_wake_sleep_t20_beta5_z1p5_weight0p05_seed{seed}", "#C94F4F", "-", "formal"),
    Variant("exact_delayed", "Exact delayed (T=20)", "ablation/repeats/exact_delayed/CIFAR100_WRN28-10_exact_delayed_seed{seed}", "#3A7D78", "--"),
    Variant("sleep5", "Wake-Sleep (T=5)", "ablation/repeats/sleep5/CIFAR100_WRN28-10_sleep5_seed{seed}", "#4C78A8", "-"),
    Variant("sleep40", "Wake-Sleep (T=40)", "ablation/repeats/sleep40/CIFAR100_WRN28-10_sleep40_seed{seed}", "#6E6396", "-"),
    Variant("first_order_only", "First-order term only", "ablation/repeats/first_order_only/CIFAR100_WRN28-10_first_order_only_seed{seed}", "#4C78A8", "--"),
    Variant("local_sensitivity_only", "Local-sensitivity term only", "ablation/repeats/local_sensitivity_only/CIFAR100_WRN28-10_local_sensitivity_only_seed{seed}", "#6E6396", "--"),
)

SINGLE_RUN_FIGURES = {
    "mnist_accuracy_curves.png": {
        "title": "MNIST - MLP",
        "noise_title": "Asymmetric pixel flips: foreground 0.45 / background 0.15",
        "runs": {
            "baseline": "MNIST_MLP_baseline_seed42",
            "exact_gmr": "MNIST_MLP_exact_gmr_beta5_z1p5_weight0p05_seed42",
            "formal_t1": "MNIST_MLP_wake_sleep_t1_beta5_z1p5_weight0p05_seed42",
            "formal_t20": "MNIST_MLP_wake_sleep_t20_beta5_z1p5_weight0p05_seed42",
        },
    },
    "imagenet18_accuracy_curves.png": {
        "title": "ImageNet-1K - ResNet-18",
        "noise_title": "Gaussian noise (sigma=0.15)",
        "runs": {
            "baseline": "ImageNet1K_ResNet18_baseline",
            "exact_gmr": "ImageNet1K_ResNet18_exact_g0_beta5_z1p5_weight0p05",
            "formal_t1": "ImageNet1K_ResNet18_wake_sleep_sleep1_beta5_z1p5_weight0p05",
            "formal_t20": "ImageNet1K_ResNet18_wake_sleep_sleep20_beta5_z1p5_weight0p05",
        },
    },
    "imagenet50_accuracy_curves.png": {
        "title": "ImageNet-1K - ResNet-50",
        "noise_title": "Gaussian noise (sigma=0.15)",
        "runs": {
            "baseline": "ImageNet1K_ResNet50_baseline",
            "formal_t20": "ImageNet1K_ResNet50_wake_sleep_sleep20_beta5_z1p5_weight0p05",
        },
    },
}

TODO_VARIANTS: dict[str, dict] = {}


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _source_dir(spec: Variant, seed: int) -> Path:
    return RESULT_ROOT / spec.directory_pattern.format(seed=seed)


def _read_run(spec: Variant, seed: int) -> tuple[dict, list[dict]]:
    directory = _source_dir(spec, seed)
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed" or int(status.get("completed_epoch", 0)) != 100:
        raise RuntimeError(f"Run is not completed through epoch 100: {directory}")
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    epochs = [int(row["epoch"]) for row in history]
    if len(history) != 100 or epochs != list(range(1, 101)):
        raise RuntimeError(f"History is not a unique continuous 100-epoch run: {directory}")
    for row in history:
        for metric in REQUIRED_METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {metric} at epoch {row['epoch']}: {directory}")

    best_clean = max(history, key=lambda row: float(row["clean_top1_accuracy"]))
    best_noisy = max(history, key=lambda row: float(row["noisy_top1_accuracy"]))
    run = {
        "variant": spec.key,
        "label": spec.label,
        "seed": seed,
        "final_clean_top1": float(history[-1]["clean_top1_accuracy"]),
        "best_clean_top1": float(best_clean["clean_top1_accuracy"]),
        "best_clean_epoch": int(best_clean["epoch"]),
        "final_noisy_top1": float(history[-1]["noisy_top1_accuracy"]),
        "best_noisy_top1": float(best_noisy["noisy_top1_accuracy"]),
        "best_noisy_epoch": int(best_noisy["epoch"]),
        "fastest_train_seconds": min(float(row["train_seconds"]) for row in history),
        "total_train_hours": sum(float(row["train_seconds"]) for row in history) / 3600.0,
        "peak_allocated_gb": max(float(row["gpu_peak_allocated_gb"]) for row in history),
        "peak_reserved_gb": max(float(row["gpu_peak_reserved_gb"]) for row in history),
        "directory": str(directory),
    }
    return run, history


def _aggregate(spec: Variant, rows: list[dict]) -> dict:
    metrics = (
        "final_clean_top1", "best_clean_top1", "best_clean_epoch",
        "final_noisy_top1", "best_noisy_top1", "best_noisy_epoch",
        "fastest_train_seconds", "total_train_hours",
        "peak_allocated_gb", "peak_reserved_gb",
    )
    result = {"variant": spec.key, "label": spec.label, "runs": len(rows)}
    for metric in metrics:
        mean, std = _mean_std([float(row[metric]) for row in rows])
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
    return result


def _paired_comparisons(by_variant: dict[str, list[dict]]) -> list[dict]:
    reference = {row["seed"]: row for row in by_variant["formal_t20"]}
    metrics = (
        "final_clean_top1", "best_clean_top1", "final_noisy_top1", "best_noisy_top1",
        "fastest_train_seconds", "total_train_hours", "peak_allocated_gb", "peak_reserved_gb",
    )
    output = []
    for variant, rows in by_variant.items():
        if variant == "formal_t20":
            continue
        candidate = {row["seed"]: row for row in rows}
        for metric in metrics:
            differences = [float(candidate[seed][metric]) - float(reference[seed][metric]) for seed in SEEDS]
            mean, std = _mean_std(differences)
            half = T95_DF4 * std / math.sqrt(len(differences))
            output.append({
                "variant": variant,
                "reference": "formal_t20",
                "metric": metric,
                "mean_difference": mean,
                "std_difference": std,
                "ci95_lower": mean - half,
                "ci95_upper": mean + half,
            })
    return output


def _cell(row: dict, metric: str, scale: float = 1.0, digits: int = 2) -> str:
    return f"{scale * row[metric + '_mean']:.{digits}f} ± {scale * row[metric + '_std']:.{digits}f}"


def _write_table_rows(lines: list[str], keys: tuple[str, ...], by_key: dict[str, dict]) -> None:
    for key in keys:
        row = by_key[key]
        lines.append(
            f"| {row['label']} | {_cell(row, 'best_clean_top1', 100)} | "
            f"{_cell(row, 'best_noisy_top1', 100)} |"
        )


def _write_markdown(aggregates: list[dict], paired: list[dict], path: Path) -> None:
    by_key = {row["variant"]: row for row in aggregates}
    header = [
        "| method | best clean / % | best noisy / % |",
        "|---|---:|---:|",
    ]
    lines = [
        "# CIFAR-100 five-seed results and ablations", "",
        "All reported values are mean ± sample standard deviation over seeds 42--46. "
        "Best clean and noisy values are selected independently within each seed before aggregation. "
        "Only completed, continuous 100-epoch histories from the current direct-aggregation implementation are included.",
        "", "## Formal methods", "", *header,
    ]
    _write_table_rows(lines, ("baseline", "exact_gmr", "formal_t1", "formal_t20"), by_key)
    lines.extend(["", "## Completed ablations", "", *header])
    _write_table_rows(lines, ("exact_delayed", "sleep5", "sleep40", "first_order_only", "local_sensitivity_only"), by_key)
    lines.extend([
        "", "## Paired differences relative to formal T=20", "",
        "Positive accuracy differences favor the candidate. Values are percentage points.", "",
        "| variant | metric | mean difference | 95% t CI |", "|---|---|---:|---:|",
    ])
    selected_metrics = {"best_clean_top1", "best_noisy_top1"}
    selected_variants = {"exact_delayed", "sleep5", "sleep40", "first_order_only", "local_sensitivity_only"}
    for row in paired:
        if row["variant"] not in selected_variants or row["metric"] not in selected_metrics:
            continue
        scale = 100.0 if "top1" in row["metric"] else 1.0
        lines.append(
            f"| {row['variant']} | {row['metric']} | {scale * row['mean_difference']:.2f} | "
            f"[{scale * row['ci95_lower']:.2f}, {scale * row['ci95_upper']:.2f}] |"
        )
    lines.extend([
        "", "## Efficiency measurements", "",
        "Timing and memory fields in the JSON/CSV describe the audited runs. "
        "Measure methods on the same device and protocol for efficiency comparisons.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _style_axis(axis) -> None:
    axis.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Top-1 accuracy")


def _plot_line(axis, epochs: list[int], values: list[float], spec: Variant) -> None:
    line, = axis.plot(
        epochs,
        values,
        label=spec.label,
        color=spec.color,
        linestyle=spec.linestyle,
        linewidth=2.15,
        zorder=3,
    )
    line.set_path_effects([
        path_effects.SimpleLineShadow(offset=(0.9, -0.9), shadow_color="#1F2937", alpha=0.20, rho=0.9),
        path_effects.Normal(),
    ])


def _plot_group(keys: tuple[str, ...], histories: dict[str, list[list[dict]]], output_path: Path, title: str) -> None:
    specs = {spec.key: spec for spec in VARIANTS}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for key in keys:
        spec = specs[key]
        runs = histories[key]
        epochs = [int(row["epoch"]) for row in runs[0]]
        for axis, metric in zip(axes, ("clean_top1_accuracy", "noisy_top1_accuracy")):
            series = [[float(row[metric]) for row in run] for run in runs]
            means = [statistics.fmean(values) for values in zip(*series)]
            _plot_line(axis, epochs, means, spec)
    axes[0].set_title("Clean validation")
    axes[1].set_title("Gaussian noise (sigma=0.15)")
    for axis in axes:
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_single_run_group(config: dict, output_path: Path) -> None:
    specs = {spec.key: spec for spec in VARIANTS}
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for key, directory_name in config["runs"].items():
        spec = specs[key]
        history_path = RESULT_ROOT / directory_name / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        epochs = [int(row["epoch"]) for row in history]
        for axis, metric in zip(axes, ("clean_top1_accuracy", "noisy_top1_accuracy")):
            values = [float(row[metric]) for row in history]
            _plot_line(axis, epochs, values, spec)
    axes[0].set_title("Clean validation")
    axes[1].set_title(config["noise_title"])
    for axis in axes:
        _style_axis(axis)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.suptitle(config["title"], fontsize=13, fontweight="bold")
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-figures-dir",
        type=Path,
        default=None,
        help="Optionally copy generated figures to a manuscript figure directory.",
    )
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    by_variant: dict[str, list[dict]] = {}
    histories: dict[str, list[list[dict]]] = {}
    audits: dict[str, dict] = {}
    for spec in VARIANTS:
        rows, runs = [], []
        for seed in SEEDS:
            row, history = _read_run(spec, seed)
            rows.append(row)
            runs.append(history)
        by_variant[spec.key] = rows
        histories[spec.key] = runs
        all_rows.extend(rows)
        audits[spec.key] = {
            "state": "completed", "seeds": list(SEEDS), "epochs_per_seed": 100,
            "continuous_unique_epochs": True, "finite_required_metrics": True,
        }

    aggregates = [_aggregate(spec, by_variant[spec.key]) for spec in VARIANTS]
    paired = _paired_comparisons(by_variant)
    payload = {
        "seeds": list(SEEDS),
        "selection_rule": "select each seed's best epoch independently, then aggregate",
        "runs": all_rows,
        "aggregates": aggregates,
        "paired_comparisons": paired,
        "audits": audits,
        "todo": TODO_VARIANTS,
    }
    (OUTPUT_ROOT / "five_seed_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT_ROOT / "five_seed_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    _write_markdown(aggregates, paired, OUTPUT_ROOT / "CIFAR100_FIVE_SEED_ABLATIONS.md")

    figures = {
        "cifar100_accuracy_curves.png": (("baseline", "exact_gmr", "formal_t1", "formal_t20"), "CIFAR-100 formal methods (five seeds)"),
        "cifar100_ablation_period_curves.png": (("formal_t1", "sleep5", "formal_t20", "sleep40"), "CIFAR-100 sleep-period ablation (five seeds)"),
        "cifar100_ablation_exact_delayed_curves.png": (("formal_t20", "exact_delayed"), "CIFAR-100 delayed-update ablation (five seeds)"),
        "cifar100_ablation_gradient_branches.png": (("formal_t20", "first_order_only", "local_sensitivity_only"), "CIFAR-100 gradient-branch ablation (five seeds)"),
    }
    for filename, (keys, title) in figures.items():
        _plot_group(keys, histories, figure_root / filename, title)
    for filename, config in SINGLE_RUN_FIGURES.items():
        _plot_single_run_group(config, figure_root / filename)
    if args.paper_figures_dir:
        args.paper_figures_dir.mkdir(parents=True, exist_ok=True)
        for filename in (*figures, *SINGLE_RUN_FIGURES):
            shutil.copy2(figure_root / filename, args.paper_figures_dir / filename)
    manifest = {
        "figures": [*figures, *SINGLE_RUN_FIGURES],
        "curves": "repeated-run figures show per-epoch means only; sample standard deviations are reported in the corresponding tables",
        "style": "global method-consistent palette with subtle visual line shadows; Baseline is gray dashed; formal Wake-Sleep T=1/T=20 are warm solid lines; the T=5/T=40 period supplements are contrasting cool solid lines; Exact GMR and the remaining controlled ablations are cool dashed lines",
        "todo_excluded": list(TODO_VARIANTS),
    }
    (OUTPUT_ROOT / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(VARIANTS)} variants and wrote {len(all_rows)} run rows.")


if __name__ == "__main__":
    main()
