"""Tiện ích dùng chung cho chuỗi thực nghiệm Full Scope (W&B + ánh xạ kết quả)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_MANIFEST_PATH = RESULTS_DIR / "fullscope_wandb_runs.json"

DEFAULT_ENTITY = "lgkai-hcmus"
DEFAULT_PROJECT = "STUDD"

SEED_TEACHER = 42
SEED_STUDENT = 142
SEED_RNG = 42


def normalize_wandb_url(url: str) -> str:
    """Chuẩn hóa URL W&B về domain canonical `https://wandb.ai/`."""
    return (
        str(url)
        .replace("https://forge.coreweave.com/wandb/", "https://wandb.ai/")
        .replace("http://forge.coreweave.com/wandb/", "https://wandb.ai/")
    )


def update_runs_manifest(key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ghi nhận Run URL và thông số chính vào `results/fullscope_wandb_runs.json`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RUNS_MANIFEST_PATH.exists():
        data = json.loads(RUNS_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        data = {"entity": DEFAULT_ENTITY, "project": DEFAULT_PROJECT, "runs": {}}
    data.setdefault("runs", {})[key] = payload
    RUNS_MANIFEST_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return data


def log_fullscope_run(
    *,
    run_key: str,
    run_name: str,
    job_type: str,
    config: Dict[str, Any],
    summary_metrics: Dict[str, Any],
    tables: Optional[Dict[str, pd.DataFrame]] = None,
    images: Optional[Dict[str, Path]] = None,
    tags: Optional[List[str]] = None,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    enabled: bool = True,
) -> Dict[str, Any]:
    """Log một run Full Scope lên W&B và lưu URL canonical vào manifest JSON."""
    if not enabled:
        payload = {
            "run_name": run_name,
            "run_id": None,
            "url": None,
            "job_type": job_type,
            "summary": summary_metrics,
        }
        update_runs_manifest(run_key, payload)
        return payload

    import wandb

    os.environ.setdefault("WANDB_SILENT", "true")
    with wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        group="fullscope-reproduction",
        job_type=job_type,
        tags=tags or ["full-scope", "reproduction", job_type],
        config=config,
    ) as run:
        log_payload: Dict[str, Any] = {}
        if tables:
            for name, df in tables.items():
                log_payload[name] = wandb.Table(dataframe=df)
        if images:
            for name, img_path in images.items():
                if Path(img_path).exists():
                    log_payload[name] = wandb.Image(str(img_path))
        if log_payload:
            run.log(log_payload)
        run.summary.update(summary_metrics)
        run_id = str(run.id)
        raw_url = getattr(run, "url", None) or f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
        canonical_url = normalize_wandb_url(raw_url)

    payload = {
        "run_name": run_name,
        "run_id": run_id,
        "url": canonical_url,
        "job_type": job_type,
        "config": config,
        "summary": summary_metrics,
    }
    update_runs_manifest(run_key, payload)
    print(f"[W&B] Logged '{run_name}' -> {canonical_url}")
    return payload
