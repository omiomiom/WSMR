"""Run the core CIFAR-100 ablations serially over five seeds each.

Each variant runs through ``run_five_seed.py``. Completed results are skipped,
and interrupted runs resume from their checkpoints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wslgmr.utils import write_json


VARIANTS = (
    "exact_delayed",
    "sleep5",
    "sleep40",
    "first_order_only",
    "local_sensitivity_only",
)
STATUS_PATH = PROJECT_ROOT / "Results/ablation/repeats/core_suite_status.json"
RUNNER = PROJECT_ROOT / "ablation/repeats/run_five_seed.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ordered five-seed core ablation suite."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = {
        "state": "running",
        "variants": list(VARIANTS),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "current": None,
        "completed": [],
        "failed": [],
    }
    write_json(STATUS_PATH, state)

    for variant in VARIANTS:
        state["current"] = variant
        write_json(STATUS_PATH, state)
        command = [sys.executable, "-B", "-u", str(RUNNER), variant]
        print(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"core ablation suite: start {variant}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        return_code = subprocess.run(command, cwd=PROJECT_ROOT).returncode
        if return_code:
            state["state"] = "failed"
            state["failed"].append(
                {"variant": variant, "return_code": return_code}
            )
            write_json(STATUS_PATH, state)
            raise SystemExit(return_code)
        state["completed"].append(variant)
        write_json(STATUS_PATH, state)
        print(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"core ablation suite: completed {variant}",
            flush=True,
        )

    state["current"] = None
    if not args.dry_run:
        state["state"] = "completed"
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(STATUS_PATH, state)


if __name__ == "__main__":
    main()
