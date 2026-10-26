"""Run one CIFAR-100 ablation serially over five seeds.

Use ``ablation/train.py`` for training, skip completed seeds, and resume
interrupted seeds from their checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wslgmr.utils import write_json
from tqdm.auto import tqdm


SEEDS = (42, 43, 44, 45, 46)
RESULT_ROOT = PROJECT_ROOT / "Results/ablation/repeats"
LOG_ROOT = PROJECT_ROOT / "Results/_run_logs"


@dataclass(frozen=True)
class VariantSpec:
    config_name: str
    canonical_seed42: Path | None = None


SPECS = {
    "local_sensitivity_only": VariantSpec(
        "config_local_sensitivity_only.py",
    ),
    "first_order_only": VariantSpec(
        "config_first_order_only.py",
    ),
    "exact_delayed": VariantSpec("config_exact_delayed.py"),
    "sleep5": VariantSpec("config_interval5.py"),
    "sleep40": VariantSpec("config_interval40.py"),
}


def _is_complete(directory: Path) -> bool:
    status = directory / "status.json"
    if not status.is_file():
        return False
    try:
        import json

        return json.loads(status.read_text(encoding="utf-8")).get("state") == "completed"
    except (OSError, ValueError):
        return False


def _target_dir(variant: str, seed: int) -> Path:
    return RESULT_ROOT / variant / f"CIFAR100_WRN28-10_{variant}_seed{seed}"


def _available_dir(variant: str, seed: int) -> Path | None:
    spec = SPECS[variant]
    if seed == 42 and spec.canonical_seed42 is not None:
        if _is_complete(spec.canonical_seed42):
            return spec.canonical_seed42
    target = _target_dir(variant, seed)
    return target if _is_complete(target) else None


def _epoch_snapshot(output_dir: Path) -> tuple[int, dict] | None:
    status_path = output_dir / "status.json"
    history_path = output_dir / "history.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        epoch = int(status.get("completed_epoch", 0))
        if epoch <= 0 or not history_path.is_file():
            return epoch, {}
        history = json.loads(history_path.read_text(encoding="utf-8"))
        return epoch, history[-1] if history else {}
    except (OSError, ValueError, TypeError):
        return None


def _run_with_progress(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    output_dir: Path,
    label: str,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        snapshot = _epoch_snapshot(output_dir)
        completed_epoch = snapshot[0] if snapshot is not None else 0
        bar = tqdm(
            total=100,
            initial=min(completed_epoch, 100),
            desc=label,
            unit="epoch",
            dynamic_ncols=True,
        )
        last_epoch = completed_epoch
        try:
            while process.poll() is None:
                snapshot = _epoch_snapshot(output_dir)
                if snapshot is not None:
                    epoch, row = snapshot
                    if epoch > last_epoch:
                        bar.update(min(epoch, 100) - last_epoch)
                        last_epoch = min(epoch, 100)
                        print(
                            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                            f"{label}: epoch {epoch}/100, "
                            f"clean={100.0 * float(row.get('clean_top1_accuracy', float('nan'))):.2f}%, "
                            f"noisy={100.0 * float(row.get('noisy_top1_accuracy', float('nan'))):.2f}%, "
                            f"train={float(row.get('train_seconds', float('nan'))):.1f}s",
                            flush=True,
                        )
                time.sleep(2.0)
            return_code = process.wait()
            snapshot = _epoch_snapshot(output_dir)
            if snapshot is not None and snapshot[0] > last_epoch:
                bar.update(min(snapshot[0], 100) - last_epoch)
            return return_code
        finally:
            bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one isolated CIFAR-100 ablation over five seeds."
    )
    parser.add_argument("variant", choices=tuple(SPECS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    invalid = set(seeds).difference(SEEDS)
    if invalid:
        raise ValueError(f"Only the registered seeds are supported: {sorted(invalid)}")

    variant = str(args.variant)
    spec = SPECS[variant]
    variant_root = RESULT_ROOT / variant
    status_path = variant_root / "suite_status.json"
    state = {
        "state": "running",
        "variant": variant,
        "seeds": list(seeds),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "current": None,
        "completed": [],
        "reused": [],
        "failed": [],
    }
    write_json(status_path, state)

    for seed in seeds:
        available = _available_dir(variant, seed)
        if available is not None:
            print(
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"{variant} seed{seed}: reuse completed result at {available}",
                flush=True,
            )
            state["completed"].append(seed)
            state["reused"].append({"seed": seed, "directory": str(available)})
            write_json(status_path, state)
            continue

        output_dir = _target_dir(variant, seed)
        print(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"{variant} seed{seed}: start; detailed child logs are "
            f"{LOG_ROOT / f'ablation_{variant}_seed{seed}.stdout.log'}",
            flush=True,
        )
        state["current"] = seed
        write_json(status_path, state)
        command = [
            sys.executable,
            "-B",
            "-u",
            str(PROJECT_ROOT / "ablation/train.py"),
            str(PROJECT_ROOT / "ablation/configs" / spec.config_name),
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
        ]
        if (output_dir / "lastest.pt").is_file():
            command.append("--resume")
        if args.dry_run:
            print(" ".join(command))
            state["current"] = None
            continue
        return_code = _run_with_progress(
            command,
            LOG_ROOT / f"ablation_{variant}_seed{seed}.stdout.log",
            LOG_ROOT / f"ablation_{variant}_seed{seed}.stderr.log",
            output_dir,
            f"{variant} seed{seed}",
        )
        if return_code != 0 or not _is_complete(output_dir):
            state["state"] = "failed"
            state["failed"].append({"seed": seed, "return_code": return_code})
            write_json(status_path, state)
            raise SystemExit(return_code or 1)
        state["completed"].append(seed)
        write_json(status_path, state)
        print(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"{variant} seed{seed}: completed",
            flush=True,
        )

    state["current"] = None
    if not args.dry_run:
        state["state"] = "completed"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(status_path, state)


if __name__ == "__main__":
    main()
