from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "Results"
SUMMARY_ROOT = RESULT_ROOT / "cifar100_five_seed_summary"
LOG_ROOT = PROJECT_ROOT / "Results/_run_logs"
STATUS_PATH = SUMMARY_ROOT / "suite_status.json"

METHODS = {
    "baseline": {
        "config": "config_cifar100_baseline.py",
        "directory": "CIFAR100_WRN28-10_baseline_seed{seed}",
    },
    "exact_gmr": {
        "config": "config_cifar100_exact_g0.py",
        "directory": "CIFAR100_WRN28-10_exact_gmr_beta5_z1p5_weight0p05_seed{seed}",
    },
    "sleep1": {
        "config": "config_cifar100_gi_i1.py",
        "directory": "CIFAR100_WRN28-10_wake_sleep_t1_beta5_z1p5_weight0p05_seed{seed}",
    },
    "sleep20": {
        "config": "config_cifar100_gi_i20.py",
        "directory": "CIFAR100_WRN28-10_wake_sleep_t20_beta5_z1p5_weight0p05_seed{seed}",
    },
}

RUNS = []
for seed in (42, 43, 44, 45, 46):
    for method, spec in METHODS.items():
        config_name = spec["config"]
        RUNS.append(
            (
                f"seed{seed}_{method}",
                config_name,
                spec["directory"].format(seed=seed),
                seed,
            )
        )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def is_completed(directory: Path) -> bool:
    status_path = directory / "status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    return status.get("state") == "completed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper's CIFAR-100 four-method, five-seed suite."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        for label, config_name, result_name, seed in RUNS:
            print(
                f"{label}: {sys.executable} -B -u train.py "
                f"configs/{config_name} --seed {seed} "
                f"--output-dir Results/{result_name}"
            )
        return

    state = {
        "state": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "current": None,
        "completed": [],
        "failed": [],
        "seeds": [42, 43, 44, 45, 46],
        "note": "All paper runs use the canonical Results/<experiment> layout.",
    }
    write_json(STATUS_PATH, state)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    for label, config_name, result_name, seed in RUNS:
        result_dir = RESULT_ROOT / result_name
        if is_completed(result_dir):
            state["completed"].append(label)
            write_json(STATUS_PATH, state)
            continue
        state["current"] = label
        write_json(STATUS_PATH, state)
        command = [
            sys.executable,
            "-B",
            "-u",
            str(PROJECT_ROOT / "train.py"),
            str(
                PROJECT_ROOT
                / "configs"
                / config_name
            ),
            "--seed",
            str(seed),
            "--output-dir",
            str(result_dir),
        ]
        if (result_dir / "lastest.pt").is_file():
            command.append("--resume")
        environment = dict(os.environ)
        environment.setdefault("PYTHONUNBUFFERED", "1")
        with (LOG_ROOT / f"repeat_{label}.stdout.log").open(
            "a", encoding="utf-8"
        ) as stdout, (LOG_ROOT / f"repeat_{label}.stderr.log").open(
            "a", encoding="utf-8"
        ) as stderr:
            return_code = subprocess.call(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
        if return_code != 0 or not is_completed(result_dir):
            state["state"] = "failed"
            state["failed"].append(
                {"experiment": label, "return_code": return_code}
            )
            write_json(STATUS_PATH, state)
            raise SystemExit(return_code or 1)
        state["completed"].append(label)
        write_json(STATUS_PATH, state)

    state["current"] = "summary"
    write_json(STATUS_PATH, state)
    return_code = subprocess.call(
        [
            sys.executable,
            "-B",
            str(
                PROJECT_ROOT
                / "experiments/cifar100_five_seed/summarize_cifar100.py"
            ),
        ],
        cwd=PROJECT_ROOT,
    )
    state["current"] = None
    state["state"] = "completed" if return_code == 0 else "failed"
    state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if return_code != 0:
        state["failed"].append(
            {"experiment": "summary", "return_code": return_code}
        )
    write_json(STATUS_PATH, state)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
