"""Run the CIFAR-100 tau=1,2,4 comparison serially with WSMR T=20.

Run from project root: python -B -u experiments/run_cifar100_tau_sweep.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wslgmr import load_config_file, resolve_config

OUTPUT = ROOT / "Results/cifar100_tau_sweep"
LOGS = ROOT / "Results/_run_logs"
CONFIGS = {0: "config_cifar100_gi_i20.py", **{tau: f"config_cifar100_gi_i20_tau{tau}.py" for tau in (1, 2, 4)}}


def config(tau):
    return resolve_config(load_config_file(ROOT / "configs" / CONFIGS[tau]), project_root=ROOT)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit(tau):
    current = config(tau)
    directory = Path(current["paths"]["output_dir"])
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed", directory
    saved = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    for field in ("dataset", "model", "loader", "training", "regularizer", "wake_sleep", "evaluation", "model_options", "dataset_options"):
        assert saved[field] == current[field], (directory, field)
    assert saved["experiment"]["seed"] == current["experiment"]["seed"] == 42
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in history] == list(range(1, 101)), directory
    fields = ("train_ce", "train_regularizer", "train_seconds", "clean_loss", "noisy_loss", "clean_top1_accuracy", "noisy_top1_accuracy", "gpu_peak_allocated_gb")
    assert all(math.isfinite(float(row[field])) for row in history for field in fields), directory
    return history


def summarize(completed):
    histories = {tau: audit(tau) for tau in [0] + completed}
    rows = []
    for tau, history in histories.items():
        best_noisy = max(history, key=lambda r: r["noisy_top1_accuracy"])
        row = {"tau": tau, "epochs": 100, "directory": config(tau)["paths"]["output_dir"],
               "final_clean_percent": 100 * history[-1]["clean_top1_accuracy"],
               "final_noisy_percent": 100 * history[-1]["noisy_top1_accuracy"],
               "best_noisy_epoch": best_noisy["epoch"],
               "best_noisy_percent": 100 * best_noisy["noisy_top1_accuracy"],
               "clean_at_best_noisy_percent": 100 * best_noisy["clean_top1_accuracy"]}
        for metric, label in (("clean_top1_accuracy", "clean"), ("noisy_top1_accuracy", "noisy")):
            row[f"mean_curve_{label}_percent"] = 100 * statistics.fmean(r[metric] for r in history)
            row[f"mean_curve_{label}_delta_vs_tau0_pp"] = 100 * statistics.fmean(a[metric] - b[metric] for a, b in zip(history, histories[0]))
        rows.append(row)
    write_json(OUTPUT / "summary.json", {"runs": rows, "completed_tau_values": list(histories), "single_seed_screening": True,
        "note": "All settings use seed42, T=20 and 100 epochs; only tau changes. Curve comparisons cover all epochs, not selected peaks."})
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), sharey=False, constrained_layout=True)
    colors = {0: "#C94F4F", 1: "#4C78A8", 2: "#3A7D78", 4: "#6E6396"}
    for tau, history in histories.items():
        for axis, field in zip(axes, ("clean_top1_accuracy", "noisy_top1_accuracy")):
            axis.plot([r["epoch"] for r in history], [r[field] for r in history], label=f"tau={tau}",
                      color=colors[tau], linestyle="-" if tau == 0 else "--", linewidth=2)
    for axis, heading in zip(axes, ("Clean Accuracy", "Gaussian Noise Accuracy")):
        axis.set_title(heading)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Top-1 accuracy")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.legend(frameon=False, loc="lower right")
        axis.grid(axis="y", linestyle="--", alpha=0.2)
    fig.suptitle("CIFAR-100 Target Margin", weight="bold")
    fig.savefig(OUTPUT / "tau_comparison.png", dpi=240)
    plt.close(fig)


def main():
    reference = config(0)
    for tau in (1, 2, 4):
        candidate = config(tau)
        for key in ("loader", "training", "wake_sleep", "evaluation", "model_options", "dataset_options"):
            assert candidate[key] == reference[key], key
        reg = dict(candidate["regularizer"])
        reg["target_margin"] = reference["regularizer"]["target_margin"]
        assert reg == reference["regularizer"]
    audit(0)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "completed": [], "current_tau": None, "order": [1, 2, 4], "reference_tau": 0}
    path = OUTPUT / "suite_status.json"
    write_json(path, state)
    try:
        for tau in (1, 2, 4):
            state["current_tau"] = tau
            write_json(path, state)
            directory = Path(config(tau)["paths"]["output_dir"])
            status = directory / "status.json"
            if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("state") == "completed":
                audit(tau)
                print(f"[reuse] tau={tau}", flush=True)
            else:
                print(f"[start] tau={tau} | seed42, T=20, lambda=0.05, 100 epochs", flush=True)
                command = [sys.executable, "-B", "-u", str(ROOT / "train.py"), str(ROOT / "configs" / CONFIGS[tau])]
                if (directory / "lastest.pt").exists():
                    command.append("--resume")
                with (LOGS / f"cifar100_tau{tau}.stdout.log").open("a", encoding="utf-8") as out, (LOGS / f"cifar100_tau{tau}.stderr.log").open("a", encoding="utf-8") as err:
                    subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
                audit(tau)
            state["completed"].append(tau)
            summarize(state["completed"])
            print(f"[done] tau={tau}; summary and comparison plot updated", flush=True)
            write_json(path, state)
        state["state"] = "completed"
        state["current_tau"] = None
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(path, state)


if __name__ == "__main__":
    main()
