from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from hf_resolver import (
    resolve_inner_adapter,
    resolve_outer_paths,
    snapshot_repo,
    task_for_inner_repo,
)
from inference_utils import inference_mas as base
from load_from_repo import STYLE_SPECS


INNER_ADAPTER_TYPE_FALLBACK = "ln_res_adapter"
OUTER_ADAPTER_TYPE_FALLBACK = "outer_ln_res_adapter"


@dataclass
class LoadedAgent:
    role: str
    repo_id: str
    repo_path: Path
    model: Any
    tokenizer: Any
    inner_adapter_path: Path
    inner_adapter: Any
    hidden_size: int


@dataclass
class ResolvedMASPaths:
    style: str
    family: str
    dataset: str
    repo_ids: Dict[str, str]
    repo_paths: Dict[str, Path]
    inner_adapter_paths: Dict[str, Path]
    outer_adapter_paths: Dict[str, Path]


@dataclass
class LoadedMASSystem:
    style: str
    family: str
    dataset: str
    agents: Dict[str, LoadedAgent]
    outer_adapters: Dict[str, Any]
    paths: ResolvedMASPaths


_AGENT_LAYOUTS: Dict[str, Dict[str, str]] = {
    "sequential": {
        "planner": "planner",
        "critic": "critic",
        "solver": "solver",
    },
    "mixture": {
        "math": "math",
        "code": "code",
        "science": "science",
        "summarizer": "summarizer",
    },
    "distillation": {
        "expert": "expert",
        "learner": "learner",
    },
    "deliberation": {
        "reflector": "reflector",
        "toolcaller": "toolcaller",
    },
}

_OUTER_LAYOUTS: Dict[str, Dict[str, tuple[str, str]]] = {
    "sequential": {
        "outer_12": ("planner", "critic"),
        "outer_23": ("critic", "solver"),
        "outer_31": ("solver", "planner"),
    },
    "mixture": {
        "outer_1s": ("math", "summarizer"),
        "outer_2s": ("code", "summarizer"),
        "outer_3s": ("science", "summarizer"),
        "outer_s1": ("summarizer", "math"),
        "outer_s2": ("summarizer", "code"),
        "outer_s3": ("summarizer", "science"),
    },
    "distillation": {
        "outer_el": ("expert", "learner"),
        "outer_le": ("learner", "expert"),
    },
    "deliberation": {
        "outer_rt": ("reflector", "toolcaller"),
        "outer_tr": ("toolcaller", "reflector"),
    },
}


def _materialize_repo(repo_id: str) -> Path:
    return snapshot_repo(repo_id)


def _normalize_dtype(value: str | torch.dtype):
    if isinstance(value, str):
        resolved = base.resolve_dtype(value)
        if resolved is None:
            raise ValueError(f"Unsupported dtype value: {value}")
        return resolved
    return value


def resolve_mas_paths(style: str, dataset: str = "math500") -> ResolvedMASPaths:
    if style not in STYLE_SPECS:
        raise ValueError(f"Unsupported style: {style}")

    spec = STYLE_SPECS[style]
    family = str(spec["family"])
    repo_ids = dict(spec["repos"])
    repo_paths: Dict[str, Path] = {}
    inner_adapter_paths: Dict[str, Path] = {}
    outer_adapter_paths: Dict[str, Path] = {}
    task = task_for_inner_repo(dataset)

    for repo_key, repo_id in repo_ids.items():
        repo_paths[repo_key] = _materialize_repo(str(repo_id))

    if family == "sequential":
        for repo_key in ("planner", "critic", "solver"):
            inner_adapter_paths[repo_key] = resolve_inner_adapter(repo_paths[repo_key], task)
        outer_adapter_paths.update(resolve_outer_paths(repo_paths["outer"], task=task))
    elif family == "mixture":
        for repo_key in ("math", "code", "science", "summarizer"):
            inner_adapter_paths[repo_key] = resolve_inner_adapter(repo_paths[repo_key], None)
        outer_adapter_paths.update(resolve_outer_paths(repo_paths["outer"], task=None))
    elif family == "distillation":
        for repo_key in ("expert", "learner"):
            inner_adapter_paths[repo_key] = resolve_inner_adapter(repo_paths[repo_key], task)
        outer_adapter_paths.update(resolve_outer_paths(repo_paths["outer"], task=task))
    elif family == "deliberation":
        for repo_key in ("reflector", "toolcaller"):
            inner_adapter_paths[repo_key] = resolve_inner_adapter(repo_paths[repo_key], None)
        outer_adapter_paths.update(resolve_outer_paths(repo_paths["outer"], task=None))
    else:
        raise ValueError(f"Unsupported style family: {family}")

    return ResolvedMASPaths(
        style=style,
        family=family,
        dataset=dataset,
        repo_ids={k: str(v) for k, v in repo_ids.items()},
        repo_paths=repo_paths,
        inner_adapter_paths=inner_adapter_paths,
        outer_adapter_paths=outer_adapter_paths,
    )


def load_mas_system(
    style: str,
    dataset: str = "math500",
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "auto",
    outer_dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
) -> LoadedMASSystem:
    paths = resolve_mas_paths(style=style, dataset=dataset)
    family = paths.family
    device_obj = torch.device(device)
    model_dtype = _normalize_dtype(dtype)
    outer_module_dtype = _normalize_dtype(outer_dtype)

    agents: Dict[str, LoadedAgent] = {}
    role_to_repo_key = _AGENT_LAYOUTS[family]
    for role, repo_key in role_to_repo_key.items():
        repo_id = paths.repo_ids[repo_key]
        repo_path = paths.repo_paths[repo_key]
        inner_adapter_path = paths.inner_adapter_paths[repo_key]
        model, tokenizer = base.load_agent_model_and_tokenizer(
            model_name_or_path=str(repo_path),
            device=device_obj,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            agent_name=role,
        )
        hidden_size = int(model.get_input_embeddings().weight.size(-1))
        inner_adapter = base.load_inner_adapter_module(
            adapter_path=str(inner_adapter_path),
            hidden_size=hidden_size,
            device=device_obj,
            dtype=model_dtype,
            fallback_adapter_type=INNER_ADAPTER_TYPE_FALLBACK,
        )
        agents[role] = LoadedAgent(
            role=role,
            repo_id=repo_id,
            repo_path=repo_path,
            model=model,
            tokenizer=tokenizer,
            inner_adapter_path=inner_adapter_path,
            inner_adapter=inner_adapter,
            hidden_size=hidden_size,
        )

    outer_adapters: Dict[str, Any] = {}
    for outer_key, (src_role, dst_role) in _OUTER_LAYOUTS[family].items():
        outer_path = paths.outer_adapter_paths[outer_key]
        out_dim = base.infer_outer_adapter_out_dim_from_file(str(outer_path))
        expected_target_dim = agents[dst_role].hidden_size
        if out_dim != expected_target_dim:
            raise RuntimeError(
                f"{outer_key} output dim mismatch: file={out_dim}, "
                f"target_role={dst_role}, target_hidden={expected_target_dim}"
            )
        outer_adapters[outer_key] = base.load_outer_adapter_module(
            adapter_path=str(outer_path),
            in_dim=agents[src_role].hidden_size,
            out_dim=out_dim,
            adapter_type=OUTER_ADAPTER_TYPE_FALLBACK,
            device=device_obj,
            dtype=outer_module_dtype,
        )

    return LoadedMASSystem(
        style=style,
        family=family,
        dataset=dataset,
        agents=agents,
        outer_adapters=outer_adapters,
        paths=paths,
    )


def unload_mas_system(system: LoadedMASSystem) -> None:
    modules = []
    for agent in system.agents.values():
        modules.extend([agent.model, agent.tokenizer, agent.inner_adapter])
    modules.extend(system.outer_adapters.values())
    base.release_resources(*modules)


def summarize_mas_paths(paths: ResolvedMASPaths) -> Dict[str, Dict[str, str]]:
    return {
        "repo_paths": {k: str(v) for k, v in paths.repo_paths.items()},
        "inner_adapter_paths": {k: str(v) for k, v in paths.inner_adapter_paths.items()},
        "outer_adapter_paths": {k: str(v) for k, v in paths.outer_adapter_paths.items()},
    }

