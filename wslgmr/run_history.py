from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from .utils import write_csv, write_json


RESUME_HISTORY_JSON = "history_with_resumes.json"
RESUME_HISTORY_CSV = "history_with_resumes.csv"
RESUME_CURVES = "curves_with_resumes.png"


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError(f"History file must contain a list: {path}")
    return value


def _find_branch(state: dict, branch_id: int) -> dict | None:
    for branch in state["branches"]:
        if int(branch["branch_id"]) == int(branch_id):
            return branch
    return None


def begin_resume_history(
    *,
    output_dir: Path,
    previous_active_history: list[dict],
    checkpoint_history: list[dict],
    checkpoint_branch_id: int,
    resume_epoch: int,
    checkpoint_path: Path,
) -> tuple[dict, int]:
    state_path = output_dir / RESUME_HISTORY_JSON
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        active = _find_branch(state, int(state["active_branch_id"]))
        if active is None:
            raise RuntimeError("Resume history has an invalid active branch.")
        active["history"] = previous_active_history
    else:
        state = {
            "version": 1,
            "active_branch_id": 0,
            "branches": [
                {
                    "branch_id": 0,
                    "parent_branch_id": None,
                    "resume_from_epoch": None,
                    "created_at": None,
                    "history": previous_active_history,
                }
            ],
            "resume_events": [],
        }

    source = _find_branch(state, checkpoint_branch_id)
    if source is None:
        # Legacy checkpoints did not record a branch id. In that case the
        # current active branch is the only safe parent interpretation.
        checkpoint_branch_id = int(state["active_branch_id"])
        source = _find_branch(state, checkpoint_branch_id)
    if source is None:
        raise RuntimeError("Cannot identify the checkpoint's source branch.")

    new_branch_id = max(
        int(branch["branch_id"]) for branch in state["branches"]
    ) + 1
    timestamp = datetime.now(timezone.utc).isoformat()
    previous_active_id = int(state["active_branch_id"])
    previous_end_epoch = (
        int(previous_active_history[-1]["epoch"])
        if previous_active_history
        else 0
    )
    state["branches"].append(
        {
            "branch_id": new_branch_id,
            "parent_branch_id": checkpoint_branch_id,
            "resume_from_epoch": int(resume_epoch),
            "created_at": timestamp,
            "history": checkpoint_history,
        }
    )
    state["resume_events"].append(
        {
            "event_id": len(state["resume_events"]) + 1,
            "created_at": timestamp,
            "previous_active_branch_id": previous_active_id,
            "source_branch_id": checkpoint_branch_id,
            "new_branch_id": new_branch_id,
            "resume_epoch": int(resume_epoch),
            "previous_active_end_epoch": previous_end_epoch,
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        }
    )
    state["active_branch_id"] = new_branch_id
    return state, new_branch_id


def update_resume_history(
    state: dict,
    *,
    active_history: list[dict],
) -> None:
    branch = _find_branch(state, int(state["active_branch_id"]))
    if branch is None:
        raise RuntimeError("Resume history has an invalid active branch.")
    branch["history"] = active_history


def _flatten_branch_rows(state: dict) -> list[dict]:
    rows: list[dict] = []
    active_id = int(state["active_branch_id"])
    for branch in state["branches"]:
        branch_id = int(branch["branch_id"])
        parent_id = branch["parent_branch_id"]
        resume_epoch = branch["resume_from_epoch"]
        for row in branch["history"]:
            # Shared history before the fork already belongs to the parent.
            if (
                parent_id is not None
                and int(row["epoch"]) < int(resume_epoch)
            ):
                continue
            rows.append(
                {
                    "branch_id": branch_id,
                    "parent_branch_id": parent_id,
                    "resume_from_epoch": resume_epoch,
                    "is_active_branch": branch_id == active_id,
                    **row,
                }
            )
    return rows


def write_resume_artifacts(state: dict, output_dir: Path) -> None:
    write_json(output_dir / RESUME_HISTORY_JSON, state)
    write_csv(
        output_dir / RESUME_HISTORY_CSV,
        _flatten_branch_rows(state),
    )
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("render_plots.py")),
            "branches",
            str(output_dir / RESUME_HISTORY_JSON),
            str(output_dir / RESUME_CURVES),
        ],
        check=True,
    )


def remove_future_checkpoints(
    output_dir: Path,
    *,
    resume_epoch: int,
) -> list[Path]:
    removed: list[Path] = []
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        return removed
    pattern = re.compile(r"epoch_(\d+)\.pt")
    for path in checkpoint_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if (
            path.is_file()
            and match is not None
            and int(match.group(1)) > int(resume_epoch)
        ):
            path.unlink()
            removed.append(path)
    return removed
