"""Run five-seed MNIST experiments serially.

Run all four paper methods: python -B -u experiments/run_mnist_five_seed.py
Run one method: add --config configs/config_mnist_gi_i20.py
Only training seed, experiment name and output directory vary across repeats.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIGS = {
    "baseline": "config_mnist_baseline.py",
    "exact_gmr": "config_mnist_exact_g0.py",
    "formal_t1": "config_mnist_gi_i1.py",
    "formal_t20": "config_mnist_gi_i20.py",
}
SUMMARY = ROOT / "Results/mnist_five_seed_summary"
LOGS = ROOT / "Results/_run_logs"
CORE = ("train_accuracy", "train_ce", "train_regularizer", "train_seconds",
        "clean_loss", "clean_top1_accuracy", "noisy_loss", "noisy_top1_accuracy",
        "gpu_peak_allocated_gb", "gpu_peak_reserved_gb")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def configuration(filename, seed):
    from wslgmr import load_config_file, resolve_config
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    config = resolve_config(load_config_file(path), project_root=ROOT)
    if config["dataset"] != "mnist":
        raise ValueError("This launcher accepts MNIST configurations only.")
    config["experiment"]["seed"] = seed
    for key in ("name",):
        config["experiment"][key] = re.sub(r"_seed\d+$", "", config["experiment"][key]) + f"_seed{seed}"
    output = Path(config["paths"]["output_dir"])
    config["paths"]["output_dir"] = str(output.with_name(re.sub(r"_seed\d+$", "", output.name) + f"_seed{seed}"))
    return config


def audit(config):
    directory = Path(config["paths"]["output_dir"])
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise ValueError(f"Incomplete: {directory}")
    saved = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    for key in ("dataset", "model", "loader", "training", "regularizer", "wake_sleep", "evaluation", "model_options", "dataset_options"):
        if saved[key] != config[key]:
            raise ValueError(f"Configuration mismatch ({key}): {directory}")
    if saved["experiment"]["seed"] != config["experiment"]["seed"] or saved["experiment"]["method"] != config["experiment"]["method"]:
        raise ValueError(f"Seed/method mismatch: {directory}")
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    expected = list(range(1, int(config["training"]["epochs"]) + 1))
    if [row["epoch"] for row in history] != expected:
        raise ValueError(f"Non-continuous epochs: {directory}")
    for row in history:
        if not all(math.isfinite(float(row[k])) for k in CORE):
            raise ValueError(f"Nonfinite core metric: {directory}, epoch {row['epoch']}")
    return history


def summarize(runs):
    rows = []
    methods = {}
    for key, filename, seed in runs:
        config = configuration(filename, seed)
        history = audit(config)
        row = {"method": key, "seed": seed, "directory": config["paths"]["output_dir"],
               "epoch_count": len(history), "final_clean": history[-1]["clean_top1_accuracy"],
               "final_noisy": history[-1]["noisy_top1_accuracy"],
               "best_clean": max(r["clean_top1_accuracy"] for r in history),
               "best_noisy": max(r["noisy_top1_accuracy"] for r in history)}
        rows.append(row)
    for key in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == key]
        aggregate = {"seeds": [row["seed"] for row in selected], "runs": selected}
        for metric in ("final_clean", "final_noisy", "best_clean", "best_noisy"):
            values = [row[metric] for row in selected]
            aggregate[metric] = {"mean": statistics.fmean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0}
        methods[key] = aggregate
    write_json(SUMMARY / "five_seed_summary.json", {"methods": methods, "selection": "final epoch for paper tables; best values diagnostic only", "core_metrics_finite": True})
    with (SUMMARY / "five_seed_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional original MNIST config; otherwise all four paper methods.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        from wslgmr.engine import run_training
        config = configuration(args.config, args.worker)
        result = Path(config["paths"]["output_dir"])
        run_training(config, resume="lastest" if (result / "lastest.pt").exists() else None)
        audit(config)
        return
    configs = CONFIGS if args.config is None else {Path(args.config).stem: args.config}
    runs = [(key, name if Path(name).parent != Path(".") else f"configs/{name}", seed)
            for seed in args.seeds for key, name in configs.items()]
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "current": None, "completed": [], "total": len(runs), "pid": os.getpid()}
    status_path = SUMMARY / "suite_status.json"
    write_json(status_path, state)
    try:
        for key, filename, seed in runs:
            label = f"mnist_{key}_seed{seed}"
            config = configuration(filename, seed)
            result = Path(config["paths"]["output_dir"])
            state["current"] = label
            write_json(status_path, state)
            completed = (result / "status.json").is_file() and json.loads((result / "status.json").read_text(encoding="utf-8")).get("state") == "completed"
            if completed:
                audit(config)
                print(f"[reuse] {label}: completed and audited", flush=True)
            else:
                print(f"[start] {label} ({len(state['completed']) + 1}/{len(runs)})", flush=True)
                command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--config", filename, "--worker", str(seed)]
                with (LOGS / f"{label}.stdout.log").open("a", encoding="utf-8") as out, (LOGS / f"{label}.stderr.log").open("a", encoding="utf-8") as err:
                    subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
                audit(config)
                print(f"[done] {label}: all 20 epochs audited", flush=True)
            state["completed"].append(label)
            write_json(status_path, state)
        summarize(runs)
        state["state"] = "completed"
        state["current"] = None
    except Exception as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, state)


if __name__ == "__main__":
    main()
