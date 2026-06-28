import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download



INNER_ADAPTER_TYPE = "ln_res_adapter"
OUTER_ADAPTER_TYPE = "outer_ln_res_adapter"


def infer_inner_adapter_type_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fallback: str = INNER_ADAPTER_TYPE,
) -> str:
    keys = set(state_dict.keys())
    expected = {
        "proj1.weight",
        "proj1.bias",
        "proj2.weight",
        "proj2.bias",
        "pre_ln.weight",
        "pre_ln.bias",
        "post_ln.weight",
        "post_ln.bias",
    }
    if not expected.issubset(keys):
        raise ValueError(
            "Release code only supports ln_res_adapter inner weights; "
            f"got keys={sorted(keys)}"
        )
    return normalize_inner_adapter_type(fallback)


def infer_outer_adapter_type_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fallback: str = OUTER_ADAPTER_TYPE,
) -> str:
    keys = set(state_dict.keys())
    required_prefixes = ("ln_source.", "ln_target.", "residual_proj.")
    if not all(any(k.startswith(prefix) for k in keys) for prefix in required_prefixes):
        raise ValueError(
            "Release code only supports outer_ln_res_adapter outer weights; "
            f"got keys={sorted(keys)}"
        )
    return normalize_outer_adapter_type(fallback)


def _maybe_build_plain_model_view(model_dir: str) -> str:
    path = Path(model_dir)
    if not path.is_dir():
        return model_dir

    # Released repos colocate adapter files with full-model weights. Recent
    # transformers versions may treat any directory containing adapter_config.json
    # as a PEFT adapter repo and fail before loading the full model. Build a
    # lightweight view that exposes only the base-model/tokenizer files.
    if not (path / "adapter_config.json").is_file():
        return str(path)
    if not ((path / "config.json").is_file() and any(path.glob("model*.safetensors"))):
        return str(path)

    view_dir = path / "_plain_model_view"
    if view_dir.is_dir():
        return str(view_dir)

    view_dir.mkdir(parents=True, exist_ok=True)
    excluded_names = {
        "adapter_config.json",
        "innerlink_config.json",
        "README.md",
    }
    excluded_suffixes = (".pt",)
    for item in path.iterdir():
        if item.name == view_dir.name:
            continue
        if item.name in excluded_names:
            continue
        if item.name.startswith("adapter("):
            continue
        if item.suffix in excluded_suffixes:
            continue
        target = view_dir / item.name
        if target.exists():
            continue
        try:
            target.symlink_to(item.resolve())
        except OSError:
            if item.is_file():
                target.write_bytes(item.read_bytes())
            elif item.is_dir():
                target.symlink_to(item.resolve(), target_is_directory=True)
    return str(view_dir)


def resolve_local_pretrained_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return _maybe_build_plain_model_view(model_name_or_path)
    try:
        resolved = snapshot_download(model_name_or_path, local_files_only=True)
        return _maybe_build_plain_model_view(resolved)
    except Exception:
        return model_name_or_path


def normalize_inner_adapter_type(adapter_type: str) -> str:
    if adapter_type != INNER_ADAPTER_TYPE:
        raise ValueError(f"Unsupported adapter_type: {adapter_type}")
    return INNER_ADAPTER_TYPE


def normalize_outer_adapter_type(adapter_type: str) -> str:
    if adapter_type != OUTER_ADAPTER_TYPE:
        raise ValueError(f"Unsupported outer adapter_type: {adapter_type}")
    return OUTER_ADAPTER_TYPE


class Adapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_type: str) -> None:
        super().__init__()
        adapter_type = normalize_inner_adapter_type(adapter_type)
        self.adapter_type = adapter_type
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = x + out
        return self.post_ln(out)


class CrossModelAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adapter_type: str) -> None:
        super().__init__()
        adapter_type = normalize_outer_adapter_type(adapter_type)
        self.adapter_type = adapter_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        hidden_dim = out_dim * 2
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.ln_source = nn.LayerNorm(in_dim)
        self.ln_target = nn.LayerNorm(out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x)
        out = self.proj2(self.act(self.proj1(h)))
        out = out + self.residual_proj(x)
        return self.ln_target(out)

