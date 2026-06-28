import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM


INNER_ADAPTER_ALIASES = {
    "1layer": "linear_adapter",
    "1layer_res": "linear_res_adapter",
    "2layer": "adapter",
    "2layer_res": "res_adapter",
    "2layer_ln_res": "ln_res_adapter",
}
INNER_ADAPTER_TYPES = {
    "linear_adapter",
    "linear_res_adapter",
    "adapter",
    "res_adapter",
    "ln_res_adapter",
}

OUTER_ADAPTER_ALIASES = {
    "1layer": "outer_linear_adapter",
    "1layer_res": "outer_linear_res_adapter",
    "2layer": "outer_adapter",
    "2layer_res": "outer_res_adapter",
    "2layer_ln_res": "outer_ln_res_adapter",
}
OUTER_ADAPTER_TYPES = {
    "outer_linear_adapter",
    "outer_linear_res_adapter",
    "outer_adapter",
    "outer_res_adapter",
    "outer_ln_adapter",
    "outer_ln_res_adapter",
}


def resolve_local_pretrained_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    try:
        return snapshot_download(model_name_or_path, local_files_only=True)
    except Exception:
        return model_name_or_path


def normalize_inner_adapter_type(adapter_type: str) -> str:
    adapter_type = INNER_ADAPTER_ALIASES.get(adapter_type, adapter_type)
    if adapter_type not in INNER_ADAPTER_TYPES:
        raise ValueError(f"Unsupported adapter_type: {adapter_type}")
    return adapter_type


def normalize_outer_adapter_type(adapter_type: str) -> str:
    adapter_type = OUTER_ADAPTER_ALIASES.get(adapter_type, adapter_type)
    if adapter_type not in OUTER_ADAPTER_TYPES:
        raise ValueError(f"Unsupported outer adapter_type: {adapter_type}")
    return adapter_type


class Adapter(nn.Module):
    def __init__(self, hidden_size: int, adapter_type: str) -> None:
        super().__init__()
        adapter_type = normalize_inner_adapter_type(adapter_type)
        self.adapter_type = adapter_type
        self.is_linear = adapter_type in {"linear_adapter", "linear_res_adapter"}
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU() if not self.is_linear else None
        self.proj2 = nn.Linear(hidden_size, hidden_size) if not self.is_linear else None
        self.use_residual = adapter_type in {"linear_res_adapter", "res_adapter", "ln_res_adapter"}
        self.pre_ln = nn.LayerNorm(hidden_size) if adapter_type == "ln_res_adapter" else None
        self.post_ln = nn.LayerNorm(hidden_size) if adapter_type == "ln_res_adapter" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_ln(x) if self.pre_ln is not None else x
        if self.is_linear:
            out = self.proj1(h)
        else:
            out = self.proj2(self.act(self.proj1(h)))
        if self.use_residual:
            out = x + out
        if self.post_ln is not None:
            out = self.post_ln(out)
        return out


class CrossModelAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, adapter_type: str) -> None:
        super().__init__()
        adapter_type = normalize_outer_adapter_type(adapter_type)
        self.adapter_type = adapter_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.is_linear = adapter_type in {"outer_linear_adapter", "outer_linear_res_adapter"}
        self.use_ln = adapter_type in {"outer_ln_adapter", "outer_ln_res_adapter"}
        self.use_residual = adapter_type in {"outer_linear_res_adapter", "outer_res_adapter", "outer_ln_res_adapter"}

        hidden_dim = out_dim * 2 if self.use_ln else out_dim
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU() if not self.is_linear else None
        self.proj2 = nn.Linear(hidden_dim, out_dim) if not self.is_linear else None

        self.ln_source = nn.LayerNorm(in_dim) if self.use_ln else None
        self.ln_target = nn.LayerNorm(out_dim) if self.use_ln else None
        self.residual_proj = nn.Linear(in_dim, out_dim) if self.use_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_source(x) if self.ln_source is not None else x
        if self.is_linear:
            out = self.proj1(h)
        else:
            out = self.proj2(self.act(self.proj1(h)))
        if self.residual_proj is not None:
            out = out + self.residual_proj(x)
        if self.ln_target is not None:
            out = self.ln_target(out)
        return out


class LatentReasoningModel(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        adapter_type: str = "adapter",
        torch_dtype: Optional[torch.dtype] = None,
        adapter_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()
        resolved_path = resolve_local_pretrained_path(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        hidden_size = self._infer_hidden_size()
        self.adapter = Adapter(hidden_size, adapter_type)

        base_param = next(self.model.parameters(), None)
        base_dtype = base_param.dtype if base_param is not None else torch.float32
        self.adapter_dtype = adapter_dtype if adapter_dtype is not None else base_dtype
        self.adapter.to(dtype=self.adapter_dtype)

    def _infer_hidden_size(self) -> int:
        embeddings = self.model.get_input_embeddings()
        return embeddings.weight.shape[-1]

    def train(self, mode: bool = True) -> "LatentReasoningModel":
        super().train(mode)
        # Keep the base LLM frozen in eval mode; only adapter is trainable.
        self.model.eval()
        return self

    def save_adapter(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.adapter.state_dict(), os.path.join(output_dir, "adapter.pt"))
        config = {"adapter_type": self.adapter.adapter_type}
        with open(os.path.join(output_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom

    def _run_adapter(
        self,
        hidden_states: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        x = hidden_states
        if x.dtype != self.adapter_dtype:
            x = x.to(self.adapter_dtype)
        out = self.adapter(x)
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out

    def _compute_adapter_losses(
        self,
        hidden_states: torch.Tensor,
        target_embeds: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        preds = self._run_adapter(hidden_states)
        preds_float = preds.float()
        targets_float = target_embeds.float()

        cosine = 1.0 - F.cosine_similarity(preds_float, targets_float, dim=-1)
        mse = (preds_float - targets_float).pow(2).mean(dim=-1)

        return {
            "cosine": self._masked_mean(cosine, mask),
            "mse": self._masked_mean(mse, mask),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        latent_mask: Optional[torch.Tensor] = None,
        adapter_cos_weight: float = 1.0,
        adapter_mse_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            input_embeds = self.model.get_input_embeddings()(input_ids)
            base_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = base_outputs.hidden_states[-1]

        if loss_mask is None:
            loss_mask = attention_mask
        if latent_mask is None:
            latent_mask = loss_mask
        latent_mask = latent_mask.to(attention_mask.device)

        adapter_cosine_loss = torch.zeros((), device=input_ids.device)
        adapter_mse_loss = torch.zeros((), device=input_ids.device)
        total_loss = torch.zeros((), device=input_ids.device)

        if adapter_cos_weight > 0 or adapter_mse_weight > 0:
            realign_pair_mask = latent_mask[:, 1:] * attention_mask[:, :-1]
            losses = self._compute_adapter_losses(
                hidden_states[:, :-1, :],
                input_embeds[:, 1:, :],
                realign_pair_mask,
            )
            adapter_cosine_loss = losses["cosine"]
            adapter_mse_loss = losses["mse"]
            total_loss = (
                adapter_cos_weight * adapter_cosine_loss
                + adapter_mse_weight * adapter_mse_loss
            )

        return {
            "loss": total_loss,
            "adapter_cosine_loss": adapter_cosine_loss.detach(),
            "adapter_mse_loss": adapter_mse_loss.detach(),
            "adapter_cosine_weighted": (adapter_cos_weight * adapter_cosine_loss).detach(),
            "adapter_mse_weighted": (adapter_mse_weight * adapter_mse_loss).detach(),
        }
