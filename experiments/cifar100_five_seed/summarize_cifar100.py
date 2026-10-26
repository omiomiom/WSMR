from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "Results"
OUTPUT_ROOT = RESULT_ROOT / "cifar100_five_seed_summary"

RESULT_METHOD_DIRECTORIES = {
    "baseline": "CIFAR100_WRN28-10_baseline_seed{seed}",
    "exact_gmr": "CIFAR100_WRN28-10_exact_gmr_beta5_z1p5_weight0p05_seed{seed}",
    "sleep1": "CIFAR100_WRN28-10_wake_sleep_t1_beta5_z1p5_weight0p05_seed{seed}",
    "sleep20": "CIFAR100_WRN28-10_wake_sleep_t20_beta5_z1p5_weight0p05_seed{seed}",
}
DIRECTORIES = {
    (method, seed): RESULT_ROOT / RESULT_METHOD_DIRECTORIES[method].format(seed=seed)
    for method in ("baseline", "exact_gmr", "sleep1", "sleep20")
    for seed in (42, 43, 44, 45, 46)
}

METRICS = (
    "final_clean_top1",
    "best_clean_top1",
    "final_noisy_top1",
    "best_noisy_top1",
    "fastest_train_seconds",
    "total_train_hours",
    "peak_allocated_gb",
)

PAIRS = (
    ("exact_gmr", "baseline"),
    ("sleep1", "baseline"),
    ("sleep20", "baseline"),
    ("sleep1", "exact_gmr"),
    ("sleep20", "exact_gmr"),
    ("sleep20", "sleep1"),
)

# Two-sided 95% Student-t critical value for five paired runs (df = 4).
T_95_DF4 = 2.776445105


def read_run(method: str, seed: int, directory: Path) -> dict:
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise RuntimeError(f"Incomplete repeat: {directory}")
    return {
        "method": method,
        "seed": seed,
        "final_clean_top1": history[-1]["clean_top1_accuracy"],
        "best_clean_top1": max(row["clean_top1_accuracy"] for row in history),
        "final_noisy_top1": history[-1]["noisy_top1_accuracy"],
        "best_noisy_top1": max(row["noisy_top1_accuracy"] for row in history),
        "fastest_train_seconds": min(row["train_seconds"] for row in history),
        "total_train_hours": sum(row["train_seconds"] for row in history) / 3600.0,
        "peak_allocated_gb": max(
            row.get("gpu_peak_allocated_gb", float("nan")) for row in history
        ),
        "directory": str(directory),
    }


def main() -> None:
    rows = [
        read_run(method, seed, directory)
        for (method, seed), directory in DIRECTORIES.items()
    ]
    aggregates = []
    for method in ("baseline", "exact_gmr", "sleep1", "sleep20"):
        selected = [row for row in rows if row["method"] == method]
        aggregate = {"method": method, "runs": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std"] = statistics.stdev(values)
        aggregates.append(aggregate)

    runs_by_key = {(row["method"], row["seed"]): row for row in rows}
    paired_comparisons = []
    for method, reference in PAIRS:
        for metric in METRICS:
            differences = [
                float(runs_by_key[(method, seed)][metric])
                - float(runs_by_key[(reference, seed)][metric])
                for seed in (42, 43, 44, 45, 46)
            ]
            mean_difference = statistics.fmean(differences)
            std_difference = statistics.stdev(differences)
            half_width = T_95_DF4 * std_difference / math.sqrt(len(differences))
            paired_comparisons.append(
                {
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "runs": len(differences),
                    "mean_difference": mean_difference,
                    "std_difference": std_difference,
                    "ci95_lower": mean_difference - half_width,
                    "ci95_upper": mean_difference + half_width,
                }
            )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": [42, 43, 44, 45, 46],
        "runs": rows,
        "aggregates": aggregates,
        "paired_comparisons": paired_comparisons,
    }
    (OUTPUT_ROOT / "cifar100_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0])
    with (OUTPUT_ROOT / "cifar100_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CIFAR-100 five-seed summary",
        "",
        "Seeds: 42, 43, 44, 45, 46. Values are mean ± sample standard deviation.",
        "",
        "| method | final clean | best clean | final noisy | best noisy | fastest epoch / s | total / h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        def cell(metric: str, scale: float = 1.0) -> str:
            return (
                f"{scale * row[metric + '_mean']:.2f} ± "
                f"{scale * row[metric + '_std']:.2f}"
            )
        lines.append(
            f"| {row['method']} | {cell('final_clean_top1', 100)} | "
            f"{cell('best_clean_top1', 100)} | {cell('final_noisy_top1', 100)} | "
            f"{cell('best_noisy_top1', 100)} | {cell('fastest_train_seconds')} | "
            f"{cell('total_train_hours')} |"
        )
    lines.extend(
        [
            "",
            "Paired differences use the same five seeds. Intervals are two-sided "
            "95% Student-t confidence intervals with 4 degrees of freedom.",
            "",
            "| method − reference | metric | mean difference | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    reported_metrics = {
        "best_clean_top1": ("best clean / pp", 100.0),
        "best_noisy_top1": ("best noisy / pp", 100.0),
        "fastest_train_seconds": ("fastest epoch / s", 1.0),
    }
    for row in paired_comparisons:
        if row["metric"] not in reported_metrics:
            continue
        label, scale = reported_metrics[row["metric"]]
        lines.append(
            f"| {row['method']} − {row['reference']} | {label} | "
            f"{scale * row['mean_difference']:.2f} | "
            f"[{scale * row['ci95_lower']:.2f}, {scale * row['ci95_upper']:.2f}] |"
        )
    (OUTPUT_ROOT / "CIFAR100_REPEATS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
