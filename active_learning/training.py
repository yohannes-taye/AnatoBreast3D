from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from active_learning.annotation_store import dataset_summary


RUNS_DIR = Path("active_learning") / "runs"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def latest_status(project_root: Path) -> dict[str, Any]:
    status_path = project_root / RUNS_DIR / "latest_finetune.json"
    summary = dataset_summary(project_root)
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
    else:
        status = {"state": "idle"}
    status["dataset"] = summary
    return status


def launch_finetune(project_root: Path, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    summary = dataset_summary(project_root)
    min_samples = int(payload.get("min_samples", 1))
    if summary["samples"] < min_samples:
        return {
            "ok": False,
            "state": "not_enough_samples",
            "message": f"Need at least {min_samples} samples, found {summary['samples']}.",
            "dataset": summary,
        }

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "finetune.log"
    work_dir = project_root / "active_learning" / "checkpoints" / run_id
    init_from = str(payload.get("init_from") or "checkpoint/breastseg.pth")

    command = [
        sys.executable,
        "tools/train_breastseg.py",
        "--config",
        "active_learning/configs/breastseg_active.py",
        "--data-root",
        "active_learning/dataset",
        "--init-from",
        init_from,
        "--work-dir",
        str(work_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    status = {
        "ok": True,
        "state": "running",
        "run_id": run_id,
        "pid": process.pid,
        "command": command,
        "log_path": str(log_path),
        "work_dir": str(work_dir),
        "dataset": summary,
        "started_at": time.time(),
    }
    _write_json(project_root / RUNS_DIR / "latest_finetune.json", status)
    return status
