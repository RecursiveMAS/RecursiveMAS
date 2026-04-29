from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import snapshot_download



def snapshot_repo(repo_id: str) -> Path:
    resolved = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
    )
    return Path(resolved).resolve()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_inner_adapter(repo_dir: Path, task: Optional[str]) -> Path:
    manifest = repo_dir / "innerlink_config.json"
    if manifest.is_file():
        data = _load_json(manifest)
        tasks = data.get("tasks", {})
        if task and task in tasks:
            adapter_name = tasks[task].get("adapter.pt", "adapter.pt")
            return (repo_dir / adapter_name).resolve()
    if task:
        candidate = repo_dir / f"adapter({task}).pt"
        if candidate.is_file():
            return candidate.resolve()
    candidate = repo_dir / "adapter.pt"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"No adapter weights found under {repo_dir}")


def resolve_outer_paths(outer_dir: Path, *, task: Optional[str]) -> Dict[str, Path]:
    manifest = outer_dir / "outerlink_config.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing outerlink manifest: {manifest}")
    data = _load_json(manifest)
    if "tasks" in data:
        if task is None:
            raise ValueError(f"Task must be set for task-scoped outer manifest: {outer_dir}")
        adapters = data["tasks"][task].get("adapters", [])
    else:
        adapters = data.get("adapters", [])
    out: Dict[str, Path] = {}
    for item in adapters:
        key = str(item["legacy_key"])
        out[key] = (outer_dir / item["filename"]).resolve()
    if not out:
        raise FileNotFoundError(f"No outer adapters found under {outer_dir}")
    return out


def infer_adapter_task_for_dataset(dataset: str) -> str:
    key = str(dataset or "").strip().lower()
    if key in {"mbppplus", "evalplus/mbppplus"}:
        return "code"
    return "math"


def resolve_medqa_dataset_arg(dataset: str, release_code_root: Path) -> str:
    key = str(dataset or "").strip().lower()
    if key not in {"medqa", "local/medqa"}:
        return dataset
    default_path = (release_code_root / "dataset" / "medqa.json").resolve()
    if default_path.is_file():
        return str(default_path)
    raise FileNotFoundError(
        f"MedQA json not found at {default_path}"
    )


def task_for_inner_repo(dataset: str) -> str:
    return infer_adapter_task_for_dataset(dataset)
