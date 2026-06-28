import argparse
from collections.abc import Mapping
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from mas_prompt import (
    PLANNER_SLOT,
    REFINED_SLOT,
    SYSTEM_PROMPT,
    build_code_planner_prompt,
    build_code_refiner_prompt_with_slot,
    build_code_solver_prompt_with_slots,
    build_math_planner_prompt,
    build_math_refiner_prompt_with_slot,
    build_math_solver_prompt_with_slots,
)
from model import (
    Adapter,
    CrossModelAdapter,
    INNER_ADAPTER_TYPES,
    INNER_ADAPTER_ALIASES,
    OUTER_ADAPTER_TYPES,
    OUTER_ADAPTER_ALIASES,
    resolve_local_pretrained_path,
)

_CHAT_TEMPLATE_IDS_FALLBACK_WARNED = False


def resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return None


@dataclass
class StagePack:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    assistant_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent1_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent2_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent3_model_name_or_path", type=str, required=True)

    parser.add_argument("--agent1_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent2_inner_aligner_path", type=str, required=True)
    parser.add_argument(
        "--inner_adapter_type_fallback",
        type=str,
        default="res_adapter",
        choices=sorted(INNER_ADAPTER_TYPES | set(INNER_ADAPTER_ALIASES)),
    )

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_json_field", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mas_shape", type=str, default="chain", choices=["chain"], help=argparse.SUPPRESS)
    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--solver_pre_question", type=int, default=0)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument(
        "--max_latent_tokens",
        type=int,
        default=80,
        help="If >0, truncate transferred latent sequence length to this value.",
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--outer_lr", type=float, required=True)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)

    parser.add_argument(
        "--outer_adapter_type",
        type=str,
        default="outer_res_adapter",
        choices=sorted(OUTER_ADAPTER_TYPES | set(OUTER_ADAPTER_ALIASES)),
    )
    parser.add_argument(
        "--outer_12_type",
        type=str,
        default=None,
        choices=[None, *sorted(OUTER_ADAPTER_TYPES | set(OUTER_ADAPTER_ALIASES))],
    )
    parser.add_argument(
        "--outer_23_type",
        type=str,
        default=None,
        choices=[None, *sorted(OUTER_ADAPTER_TYPES | set(OUTER_ADAPTER_ALIASES))],
    )

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--outer_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=0)
    return parser.parse_args()


def apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
):
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as inner_exc:
            err_text = str(inner_exc)
            if "Conversation roles must alternate user/assistant/user/assistant/." not in err_text:
                raise
    except Exception as exc:
        err_text = str(exc)
        if "Conversation roles must alternate user/assistant/user/assistant/." not in err_text:
            raise

    normalized = list(messages)
    if (
        len(normalized) >= 2
        and isinstance(normalized[0], dict)
        and isinstance(normalized[1], dict)
        and normalized[0].get("role") == "system"
        and normalized[1].get("role") == "user"
    ):
        merged_user = dict(normalized[1])
        merged_user["content"] = (
            f"{normalized[0].get('content', '')}\n\n{normalized[1].get('content', '')}".strip()
        )
        normalized = [merged_user] + normalized[2:]
        return tokenizer.apply_chat_template(normalized, **kwargs)

    raise


def _normalize_template_text(tokenizer, value) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, Mapping):
        if "input_ids" in value:
            return _normalize_template_text(tokenizer, value["input_ids"])
        raise ValueError("chat template output mapping missing input_ids for text normalization")

    if hasattr(value, "tolist"):
        return _normalize_template_text(tokenizer, value.tolist())

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if not value:
            return ""
        # Batched [1, seq] or [1, text]
        if isinstance(value[0], list):
            return _normalize_template_text(tokenizer, value[0])
        if isinstance(value[0], str):
            return "".join(value)
        return tokenizer.decode([int(x) for x in value], skip_special_tokens=False)

    raise ValueError(f"Unsupported chat template output type for text: {type(value)}")


def _normalize_template_ids(tokenizer, value, max_length: Optional[int] = None) -> List[int]:
    global _CHAT_TEMPLATE_IDS_FALLBACK_WARNED

    if isinstance(value, str):
        if not _CHAT_TEMPLATE_IDS_FALLBACK_WARNED:
            print(
                "[warn] apply_chat_template(tokenize=True) returned str in train_outer; "
                "falling back to tokenizer(...) to get input_ids."
            )
            _CHAT_TEMPLATE_IDS_FALLBACK_WARNED = True
        fallback_max_len = max_length
        if fallback_max_len is None:
            fallback_max_len = getattr(tokenizer, "model_max_length", 32768)
            if fallback_max_len is None or fallback_max_len > 1_000_000:
                fallback_max_len = 32768
        return tokenizer(
            value,
            truncation=True,
            max_length=int(fallback_max_len),
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat template output mapping missing input_ids")
        return _normalize_template_ids(tokenizer, value["input_ids"], max_length=max_length)

    if hasattr(value, "tolist"):
        return _normalize_template_ids(tokenizer, value.tolist(), max_length=max_length)

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        if not value:
            return []
        # Batched output shape [1, seq_len]
        if isinstance(value[0], list):
            return _normalize_template_ids(tokenizer, value[0], max_length=max_length)
        return [int(x) for x in value]

    raise ValueError(f"Unsupported chat template output type for ids: {type(value)}")


def render_chat_text(
    tokenizer,
    user_prompt: str,
    assistant_text: Optional[str],
    enable_thinking: bool,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_text is None:
        rendered = apply_chat_template(
            tokenizer,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return _normalize_template_text(tokenizer, rendered)

    messages.append({"role": "assistant", "content": assistant_text})
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return _normalize_template_text(tokenizer, rendered)


def render_chat_ids(
    tokenizer,
    user_prompt: str,
    assistant_text: Optional[str],
    enable_thinking: bool,
    max_length: Optional[int] = None,
) -> List[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_text is None:
        rendered = apply_chat_template(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return _normalize_template_ids(tokenizer, rendered, max_length=max_length)

    messages.append({"role": "assistant", "content": assistant_text})
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    return _normalize_template_ids(tokenizer, rendered, max_length=max_length)


def split_rendered_text_by_slot(rendered_text: str, slot_text: str) -> Tuple[str, str]:
    pos = rendered_text.find(slot_text)
    if pos < 0:
        raise RuntimeError(f"Failed to locate slot marker {slot_text!r} in rendered chat text.")
    return rendered_text[:pos], rendered_text[pos + len(slot_text) :]


def text_to_ids(tokenizer, text: str) -> List[int]:
    if not text:
        return []
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def ids_to_embeds(
    embedding_layer,
    token_ids: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden_size = embedding_layer.weight.size(-1)
    if not token_ids:
        return torch.empty((0, hidden_size), dtype=dtype, device=device)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    embeds = embedding_layer(token_tensor)[0]
    if embeds.dtype != dtype:
        embeds = embeds.to(dtype)
    return embeds


def resolve_inner_adapter_files(adapter_path: str) -> Tuple[str, str]:
    if os.path.isdir(adapter_path):
        state_path = os.path.join(adapter_path, "adapter.pt")
        config_path = os.path.join(adapter_path, "adapter_config.json")
    else:
        state_path = adapter_path
        config_path = os.path.join(os.path.dirname(adapter_path), "adapter_config.json")
    return state_path, config_path


def load_inner_adapter(
    adapter_path: str,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    fallback_adapter_type: str,
) -> Adapter:
    state_path, config_path = resolve_inner_adapter_files(adapter_path)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Inner aligner weights not found: {state_path}")

    adapter_type = fallback_adapter_type
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        adapter_type = cfg.get("adapter_type", adapter_type)

    adapter = Adapter(hidden_size=hidden_size, adapter_type=adapter_type)
    adapter.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)
    adapter.to(device=device, dtype=dtype)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False
    return adapter


def run_inner_adapter_preserve_input_grad(adapter: Adapter, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    # No torch.no_grad: gradient must reach the upstream outer links (inner params are already frozen).
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else x.dtype
    y = adapter(x.to(adapter_dtype))
    if y.dtype != out_dtype:
        y = y.to(out_dtype)
    return y


def write_outerlink_manifest(output_dir: str, paradigm: str, adapters: List[Dict[str, object]]) -> None:
    """Write outerlink_config.json mapping each outer link's legacy_key -> filename."""
    manifest = {
        "format_version": 1,
        "paradigm": paradigm,
        "legacy_config_filename": "outer_adapter_config.json",
        "adapters": adapters,
    }
    with open(os.path.join(output_dir, "outerlink_config.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def run_outer_adapter(adapter: CrossModelAdapter, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else x.dtype
    y = adapter(x.to(adapter_dtype))
    if y.dtype != out_dtype:
        y = y.to(out_dtype)
    return y


def load_model_and_tokenizer(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    agent_name: str,
    gradient_checkpointing: bool = False,
):
    resolved_path = resolve_local_pretrained_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(f"{agent_name} tokenizer has no apply_chat_template")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(f"{agent_name} tokenizer has no pad/eos token")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as exc:
            print(f"[warn] failed to enable gradient checkpointing for {agent_name}: {exc}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def load_outer_training_dataset(
    dataset_name: str,
    dataset_split: str,
    dataset_json_field: Optional[str],
) -> Dataset:
    if os.path.isfile(dataset_name):
        suffix = os.path.splitext(dataset_name)[1].lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported local dataset file type: {dataset_name}")
        kwargs = {"data_files": dataset_name}
        if suffix == ".json" and dataset_json_field is not None:
            kwargs["field"] = dataset_json_field
        return load_dataset("json", **kwargs, split="train")

    return load_dataset(dataset_name, split=dataset_split)


def build_stage_with_slot(
    tokenizer,
    embedding_layer,
    user_prompt_with_slot: str,
    assistant_text: str,
    slot_text: str,
    slot_embeds: torch.Tensor,
    enable_thinking: bool,
    device: torch.device,
    embed_dtype: torch.dtype,
    max_length: int,
) -> StagePack:
    prompt_rendered = render_chat_text(
        tokenizer,
        user_prompt_with_slot,
        assistant_text=None,
        enable_thinking=enable_thinking,
    )
    full_rendered = render_chat_text(
        tokenizer,
        user_prompt_with_slot,
        assistant_text=assistant_text,
        enable_thinking=enable_thinking,
    )

    prompt_prefix_text, prompt_suffix_text = split_rendered_text_by_slot(prompt_rendered, slot_text)
    full_prefix_text, full_suffix_text = split_rendered_text_by_slot(full_rendered, slot_text)

    prompt_prefix_ids = text_to_ids(tokenizer, prompt_prefix_text)
    prompt_suffix_ids = text_to_ids(tokenizer, prompt_suffix_text)
    full_prefix_ids = text_to_ids(tokenizer, full_prefix_text)
    full_suffix_ids = text_to_ids(tokenizer, full_suffix_text)

    slot_len = int(slot_embeds.size(0))
    prompt_len = len(prompt_prefix_ids) + slot_len + len(prompt_suffix_ids)

    token_ids = full_prefix_ids + ([-100] * slot_len) + full_suffix_ids
    truncate_left = 0
    if len(token_ids) > max_length:
        truncate_left = len(token_ids) - max_length
        token_ids = token_ids[truncate_left:]
        prompt_len = max(prompt_len - truncate_left, 0)

    labels = torch.full((len(token_ids),), -100, dtype=torch.long, device=device)
    for idx in range(prompt_len, len(token_ids)):
        tid = token_ids[idx]
        if tid >= 0:
            labels[idx] = tid
    assistant_mask = labels.ne(-100)

    prefix_embeds = ids_to_embeds(
        embedding_layer,
        full_prefix_ids,
        device=device,
        dtype=embed_dtype,
    )
    suffix_embeds = ids_to_embeds(
        embedding_layer,
        full_suffix_ids,
        device=device,
        dtype=embed_dtype,
    )

    if slot_embeds.dtype != embed_dtype:
        slot_embeds = slot_embeds.to(embed_dtype)

    seq_embeds = torch.cat([prefix_embeds, slot_embeds, suffix_embeds], dim=0)
    if truncate_left > 0:
        seq_embeds = seq_embeds[truncate_left:]
    attention_mask = torch.ones((seq_embeds.size(0),), dtype=torch.long, device=device)

    return StagePack(
        inputs_embeds=seq_embeds.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        labels=labels.unsqueeze(0),
        assistant_mask=assistant_mask,
    )


def build_planner_teacher_forced_inputs(
    tokenizer,
    question: str,
    plan: str,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
    mas_task: str = "math",
    task_type: str = "complete",
    fn_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if str(mas_task).strip().lower() == "code":
        user_prompt = build_code_planner_prompt(question, task_type=task_type, fn_name=fn_name)
    else:
        user_prompt = build_math_planner_prompt(question)
    prompt_ids = render_chat_ids(
        tokenizer,
        user_prompt,
        assistant_text=None,
        enable_thinking=enable_thinking,
        max_length=max_length,
    )
    full_ids = render_chat_ids(
        tokenizer,
        user_prompt,
        assistant_text=plan,
        enable_thinking=enable_thinking,
        max_length=max_length,
    )

    assistant_token_count = max(len(full_ids) - len(prompt_ids), 0)
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]
        assistant_kept = min(assistant_token_count, len(full_ids))
        prompt_len = len(full_ids) - assistant_kept
    else:
        prompt_len = min(len(prompt_ids), len(full_ids))

    assistant_mask = torch.zeros((len(full_ids),), dtype=torch.bool, device=device)
    if prompt_len < len(full_ids):
        assistant_mask[prompt_len:] = True

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, assistant_mask


def trim_latent(latent: torch.Tensor, max_latent_tokens: int) -> torch.Tensor:
    if max_latent_tokens > 0 and latent.size(0) > max_latent_tokens:
        return latent[:max_latent_tokens]
    return latent


def compute_solver_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def save_outer_checkpoint(
    save_dir: str,
    step: Optional[int],
    outer_12: CrossModelAdapter,
    outer_23: CrossModelAdapter,
    args: argparse.Namespace,
) -> None:
    output_dir = os.path.join(save_dir, f"checkpoint-{step}") if step is not None else save_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.save(outer_12.state_dict(), os.path.join(output_dir, "outer_12.pt"))
    torch.save(outer_23.state_dict(), os.path.join(output_dir, "outer_23.pt"))

    outer_cfg = {
        "outer_12_type": outer_12.adapter_type,
        "outer_23_type": outer_23.adapter_type,
        "outer_12_in_dim": outer_12.in_dim,
        "outer_12_out_dim": outer_12.out_dim,
        "outer_23_in_dim": outer_23.in_dim,
        "outer_23_out_dim": outer_23.out_dim,
        "mas_shape": args.mas_shape,
        "agent1_model_name_or_path": args.agent1_model_name_or_path,
        "agent2_model_name_or_path": args.agent2_model_name_or_path,
        "agent3_model_name_or_path": args.agent3_model_name_or_path,
        "agent1_inner_aligner_path": args.agent1_inner_aligner_path,
        "agent2_inner_aligner_path": args.agent2_inner_aligner_path,
        "enable_thinking": args.enable_thinking,
    }
    with open(os.path.join(output_dir, "outer_adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(outer_cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    if args.mas_shape != "chain":
        raise ValueError("Only mas_shape=chain is supported.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")

    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back model dtype to float32.")
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        outer_dtype = torch.float32

    torch.manual_seed(args.seed)

    planner_model, planner_tok = load_model_and_tokenizer(
        args.agent1_model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="planner",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    refiner_model, refiner_tok = load_model_and_tokenizer(
        args.agent2_model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="refiner",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    solver_model, solver_tok = load_model_and_tokenizer(
        args.agent3_model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="solver",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )

    planner_embed_layer = planner_model.get_input_embeddings()
    refiner_embed_layer = refiner_model.get_input_embeddings()
    solver_embed_layer = solver_model.get_input_embeddings()

    planner_hidden = planner_embed_layer.weight.size(-1)
    refiner_hidden = refiner_embed_layer.weight.size(-1)
    solver_hidden = solver_embed_layer.weight.size(-1)

    inner_1 = load_inner_adapter(
        args.agent1_inner_aligner_path,
        hidden_size=planner_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=args.inner_adapter_type_fallback,
    )
    inner_2 = load_inner_adapter(
        args.agent2_inner_aligner_path,
        hidden_size=refiner_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=args.inner_adapter_type_fallback,
    )

    outer_12_type = args.outer_12_type or args.outer_adapter_type
    outer_23_type = args.outer_23_type or args.outer_adapter_type

    outer_12 = CrossModelAdapter(planner_hidden, refiner_hidden, outer_12_type).to(device=device, dtype=outer_dtype)
    outer_23 = CrossModelAdapter(refiner_hidden, solver_hidden, outer_23_type).to(device=device, dtype=outer_dtype)
    outer_12.train()
    outer_23.train()

    params = list(outer_12.parameters()) + list(outer_23.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.outer_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = load_outer_training_dataset(
        args.dataset_name,
        args.dataset_split,
        args.dataset_json_field,
    )
    needed_cols = {"question", "plan", "refined_plan", "answer"}
    missing = needed_cols.difference(set(dataset.column_names))
    if missing:
        raise ValueError(f"Dataset missing required fields: {sorted(missing)}")
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    rows = [
        {
            "question": sample.get("question", ""),
            "plan": sample.get("plan", ""),
            "refined_plan": sample.get("refined_plan", ""),
            "answer": sample.get("answer", ""),
            "type": sample.get("type", "complete"),
            "fn_name": sample.get("fn_name", None),
        }
        for sample in dataset
    ]

    dataloader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Increase dataset size or reduce batch_size.")

    steps_per_epoch = len(dataloader)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * steps_per_epoch

    if args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=0.5,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )

    solver_args = argparse.Namespace(solver_pre_question=args.solver_pre_question)
    enable_thinking = bool(args.enable_thinking)

    os.makedirs(args.save_dir, exist_ok=True)


    global_step = 0
    total_loss_acc = 0.0
    total_count = 0
    skipped_count = 0
    start_time = time.time()

    while global_step < max_train_steps:
        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            valid_count = 0
            batch_loss_sum = 0.0

            optimizer.zero_grad(set_to_none=True)

            for sample in batch:
                q = str(sample["question"]).strip()
                p = str(sample["plan"]).strip()
                rp = str(sample["refined_plan"]).strip()
                ans = str(sample["answer"]).strip()
                task_type = str(sample.get("type", "complete")).strip().lower() or "complete"
                fn_name = sample.get("fn_name", None)
                if not q or not p or not rp or not ans:
                    skipped_count += 1
                    continue

                try:
                    # Stage 1: planner teacher forcing
                    planner_input_ids, planner_attention_mask, planner_assist_mask = build_planner_teacher_forced_inputs(
                        planner_tok,
                        question=q,
                        plan=p,
                        enable_thinking=enable_thinking,
                        device=device,
                        max_length=args.max_length,
                        mas_task=args.mas_task,
                        task_type=task_type,
                        fn_name=fn_name,
                    )
                    with torch.no_grad():
                        planner_out = planner_model(
                            input_ids=planner_input_ids,
                            attention_mask=planner_attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                        )
                    planner_hidden_seq = planner_out.hidden_states[-1][0]
                    planner_tokens_hidden = planner_hidden_seq[planner_assist_mask]
                    if planner_tokens_hidden.size(0) == 0:
                        skipped_count += 1
                        continue

                    planner_inner = run_inner_adapter_preserve_input_grad(inner_1, planner_tokens_hidden, out_dtype=model_dtype)
                    planner_to_refiner = run_outer_adapter(
                        outer_12,
                        planner_inner,
                        out_dtype=refiner_embed_layer.weight.dtype,
                    )
                    planner_to_refiner = trim_latent(planner_to_refiner, args.max_latent_tokens)

                    # Stage 2: refiner teacher forcing with planner slot injection
                    if args.mas_task == "code":
                        refiner_user = build_code_refiner_prompt_with_slot(
                            q,
                            task_type=task_type,
                            fn_name=fn_name,
                        )
                    else:
                        refiner_user = build_math_refiner_prompt_with_slot(q)
                    refiner_pack = build_stage_with_slot(
                        tokenizer=refiner_tok,
                        embedding_layer=refiner_embed_layer,
                        user_prompt_with_slot=refiner_user,
                        assistant_text=rp,
                        slot_text=PLANNER_SLOT,
                        slot_embeds=planner_to_refiner,
                        enable_thinking=enable_thinking,
                        device=device,
                        embed_dtype=refiner_embed_layer.weight.dtype,
                        max_length=args.max_length,
                    )

                    refiner_out = refiner_model(
                        inputs_embeds=refiner_pack.inputs_embeds,
                        attention_mask=refiner_pack.attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    refiner_hidden_seq = refiner_out.hidden_states[-1][0]
                    refiner_tokens_hidden = refiner_hidden_seq[refiner_pack.assistant_mask]
                    if refiner_tokens_hidden.size(0) == 0:
                        skipped_count += 1
                        continue

                    refiner_inner = run_inner_adapter_preserve_input_grad(inner_2, refiner_tokens_hidden, out_dtype=model_dtype)
                    refiner_to_solver = run_outer_adapter(
                        outer_23,
                        refiner_inner,
                        out_dtype=solver_embed_layer.weight.dtype,
                    )
                    refiner_to_solver = trim_latent(refiner_to_solver, args.max_latent_tokens)

                    # Stage 3: solver teacher forcing + final answer CE loss
                    if args.mas_task == "code":
                        solver_user = build_code_solver_prompt_with_slots(
                            q,
                            task_type=task_type,
                            args=solver_args,
                            mas_shape=args.mas_shape,
                            fn_name=fn_name,
                        )
                    else:
                        solver_user = build_math_solver_prompt_with_slots(
                            q,
                            args=solver_args,
                            mas_shape=args.mas_shape,
                        )
                    solver_pack = build_stage_with_slot(
                        tokenizer=solver_tok,
                        embedding_layer=solver_embed_layer,
                        user_prompt_with_slot=solver_user,
                        assistant_text=ans,
                        slot_text=REFINED_SLOT,
                        slot_embeds=refiner_to_solver,
                        enable_thinking=enable_thinking,
                        device=device,
                        embed_dtype=solver_embed_layer.weight.dtype,
                        max_length=args.max_length,
                    )

                    solver_out = solver_model(
                        inputs_embeds=solver_pack.inputs_embeds,
                        attention_mask=solver_pack.attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )

                    loss = compute_solver_ce_loss(solver_out.logits, solver_pack.labels)
                    if torch.isnan(loss) or torch.isinf(loss):
                        skipped_count += 1
                        continue

                    (loss / max(args.batch_size, 1)).backward()
                    valid_count += 1
                    batch_loss_sum += float(loss.item())
                except RuntimeError as exc:
                    # Skip pathological samples that exceed max_length.
                    exc_msg = str(exc).lower()
                    if "sequence_too_long" in exc_msg:
                        skipped_count += 1
                        continue
                    raise

            if valid_count == 0:
                continue

            if valid_count != args.batch_size:
                grad_scale = args.batch_size / valid_count
                for param in params:
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_batch = batch_loss_sum / valid_count
            total_loss_acc += float(loss_batch)
            total_count += 1

            if global_step % args.log_every == 0:
                avg_loss = total_loss_acc / max(total_count, 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"step={global_step} loss={avg_loss:.4f}", flush=True)
                total_loss_acc = 0.0
                total_count = 0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_outer_checkpoint(args.save_dir, global_step, outer_12, outer_23, args)


            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    save_outer_checkpoint(args.save_dir, None, outer_12, outer_23, args)

    elapsed = time.time() - start_time


# Shared library for the per-style trainers; not a runnable entry point.
