import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from mas_prompt import (
    DELIBERATION_FEEDBACK_SLOT,
    DELIBERATION_REFLECTOR_SLOT,
    build_deliberation_reflector_prompt,
    build_deliberation_reflector_prompt_with_feedback_slot,
    build_deliberation_toolcaller_prompt_with_slot,
    get_system_prompt,
)
from model import Adapter, CrossModelAdapter
from .common import (
    StagePack,
    _normalize_template_ids,
    _normalize_template_text,
    apply_chat_template,
    compute_solver_ce_loss,
    ids_to_embeds,
    load_inner_adapter,
    load_model_and_tokenizer,
    load_outer_training_dataset,
    resolve_dtype,
    run_outer_adapter,
    split_rendered_text_by_slot,
    text_to_ids,
    trim_latent,
    write_outerlink_manifest,
)
from .sequential import (
    ALLOWED_OUTER_TYPES,
    activate_gc_runtime,
    build_optional_token_type_ids,
    normalize_outer_type,
)

DELIB_REFLECTOR_FIELD_CANDIDATES = ("deliberation_reflector", "answer")
DELIB_TOOLCALLER_FIELD_CANDIDATES = ("deliberation_toolcaller", "answer")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflector_model_name_or_path", type=str, required=True)
    parser.add_argument("--toolcaller_model_name_or_path", type=str, required=True)

    parser.add_argument("--reflector_inner_aligner_path", type=str, required=True)
    parser.add_argument("--toolcaller_inner_aligner_path", type=str, required=True)
    parser.add_argument(
        "--inner_adapter_type_fallback",
        type=str,
        default="res_adapter",
        choices=[
            "1layer",
            "1layer_res",
            "2layer",
            "2layer_res",
            "2layer_ln_res",
            "adapter",
            "linear_adapter",
            "linear_res_adapter",
            "ln_res_adapter",
            "res_adapter",
        ],
    )

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_json_field", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mas_shape", type=str, default="deliberation", choices=["deliberation"])
    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code", "choice"])
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_latent_tokens", type=int, default=80)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--outer_lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--num_recursive_rounds", type=int, default=3)
    parser.add_argument("--supervise_final_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--non_last_loss_weight", type=float, default=0.1)

    parser.add_argument("--outer_adapter_type", type=str, default="outer_ln_res_adapter", choices=sorted(ALLOWED_OUTER_TYPES))
    parser.add_argument("--outer_rt_type", type=str, default=None)
    parser.add_argument("--outer_tr_type", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--outer_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--log_grad_norm", type=int, default=1, choices=[0, 1])
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=0)
    return parser.parse_args(argv)


def first_present_text(sample: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = sample.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def validate_delib_dataset_columns(column_names: Sequence[str]) -> None:
    columns = set(column_names)
    missing_groups = []
    if "question" not in columns:
        missing_groups.append("question")
    if not columns.intersection(DELIB_REFLECTOR_FIELD_CANDIDATES):
        missing_groups.append(f"reflector: one of {list(DELIB_REFLECTOR_FIELD_CANDIDATES)}")
    if not columns.intersection(DELIB_TOOLCALLER_FIELD_CANDIDATES):
        missing_groups.append(f"toolcaller: one of {list(DELIB_TOOLCALLER_FIELD_CANDIDATES)}")
    if missing_groups:
        raise ValueError(f"Dataset missing deliberation fields: {missing_groups}")


def render_chat_text_with_system(
    tokenizer,
    user_prompt: str,
    assistant_text: Optional[str],
    enable_thinking: bool,
    system_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
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


def render_chat_ids_with_system(
    tokenizer,
    user_prompt: str,
    assistant_text: Optional[str],
    enable_thinking: bool,
    system_prompt: str,
    max_length: Optional[int] = None,
) -> List[int]:
    messages = [
        {"role": "system", "content": system_prompt},
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


def build_stage_with_slot_and_system(
    tokenizer,
    embedding_layer,
    user_prompt_with_slot: str,
    assistant_text: str,
    slot_text: str,
    slot_embeds: torch.Tensor,
    enable_thinking: bool,
    system_prompt: str,
    device: torch.device,
    embed_dtype: torch.dtype,
    max_length: int,
) -> StagePack:
    prompt_rendered = render_chat_text_with_system(
        tokenizer,
        user_prompt_with_slot,
        assistant_text=None,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    full_rendered = render_chat_text_with_system(
        tokenizer,
        user_prompt_with_slot,
        assistant_text=assistant_text,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
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


def build_reflector_teacher_forced_inputs(
    tokenizer,
    question: str,
    reflector_text: str,
    enable_thinking: bool,
    system_prompt: str,
    device: torch.device,
    max_length: int,
    mas_task: str,
    task_type: str,
    fn_name: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    user_prompt = build_deliberation_reflector_prompt(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
    prompt_ids = render_chat_ids_with_system(
        tokenizer,
        user_prompt,
        assistant_text=None,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
        max_length=max_length,
    )
    full_ids = render_chat_ids_with_system(
        tokenizer,
        user_prompt,
        assistant_text=reflector_text,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
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


def run_model_hidden(model, input_ids=None, attention_mask=None, inputs_embeds=None, output_hidden_states=True):
    token_type_ids = build_optional_token_type_ids(
        model,
        attention_mask=attention_mask,
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
    )
    kwargs = {}
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        output_hidden_states=output_hidden_states,
        use_cache=False,
        return_dict=True,
        **kwargs,
    )


def run_inner_adapter_preserve_input_grad(adapter: Adapter, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    # No torch.no_grad: gradient must reach the upstream outer links (inner params are already frozen).
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else x.dtype
    y = adapter(x.to(adapter_dtype))
    if y.dtype != out_dtype:
        y = y.to(out_dtype)
    return y


def grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().float().norm().item())
        total += value * value
    return total ** 0.5


def save_delib_outer_checkpoint(
    save_dir: str,
    step: Optional[int],
    outer_rt: CrossModelAdapter,
    outer_tr: CrossModelAdapter,
    args: argparse.Namespace,
) -> None:
    output_dir = os.path.join(save_dir, f"checkpoint-{step}") if step is not None else save_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.save(outer_rt.state_dict(), os.path.join(output_dir, "outer_rt.pt"))
    torch.save(outer_tr.state_dict(), os.path.join(output_dir, "outer_tr.pt"))

    cfg = {
        "mas_shape": "deliberation",
        "mas_task": args.mas_task,
        "num_recursive_rounds": args.num_recursive_rounds,
        "supervise_final_only": args.supervise_final_only,
        "non_last_loss_weight": args.non_last_loss_weight,
        "reflector_model_name_or_path": args.reflector_model_name_or_path,
        "toolcaller_model_name_or_path": args.toolcaller_model_name_or_path,
        "reflector_inner_aligner_path": args.reflector_inner_aligner_path,
        "toolcaller_inner_aligner_path": args.toolcaller_inner_aligner_path,
        "outer_rt_type": outer_rt.adapter_type,
        "outer_rt_in_dim": outer_rt.in_dim,
        "outer_rt_out_dim": outer_rt.out_dim,
        "outer_tr_type": outer_tr.adapter_type,
        "outer_tr_in_dim": outer_tr.in_dim,
        "outer_tr_out_dim": outer_tr.out_dim,
    }
    with open(os.path.join(output_dir, "outer_adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    write_outerlink_manifest(output_dir, "deliberation", [
        {"legacy_key": "outer_rt", "filename": "outer_rt.pt", "adapter_type": outer_rt.adapter_type, "in_dim": outer_rt.in_dim, "out_dim": outer_rt.out_dim},
        {"legacy_key": "outer_tr", "filename": "outer_tr.pt", "adapter_type": outer_tr.adapter_type, "in_dim": outer_tr.in_dim, "out_dim": outer_tr.out_dim},
    ])


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.num_recursive_rounds <= 0:
        raise ValueError("--num_recursive_rounds must be positive.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")

    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        outer_dtype = torch.float32

    torch.manual_seed(args.seed)
    enable_thinking = bool(args.enable_thinking)
    system_prompt = get_system_prompt("deliberation")

    reflector_model, reflector_tok = load_model_and_tokenizer(
        args.reflector_model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="deliberation_reflector",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    toolcaller_model, toolcaller_tok = load_model_and_tokenizer(
        args.toolcaller_model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
        agent_name="deliberation_toolcaller",
        gradient_checkpointing=bool(args.gradient_checkpointing),
    )
    if bool(args.gradient_checkpointing):
        activate_gc_runtime(reflector_model, "deliberation_reflector")
        activate_gc_runtime(toolcaller_model, "deliberation_toolcaller")

    reflector_embed = reflector_model.get_input_embeddings()
    toolcaller_embed = toolcaller_model.get_input_embeddings()
    reflector_hidden = reflector_embed.weight.size(-1)
    toolcaller_hidden = toolcaller_embed.weight.size(-1)

    reflector_inner = load_inner_adapter(
        args.reflector_inner_aligner_path,
        hidden_size=reflector_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=args.inner_adapter_type_fallback,
    )
    toolcaller_inner = load_inner_adapter(
        args.toolcaller_inner_aligner_path,
        hidden_size=toolcaller_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=args.inner_adapter_type_fallback,
    )

    outer_rt = CrossModelAdapter(
        reflector_hidden,
        toolcaller_hidden,
        normalize_outer_type(args.outer_rt_type, args.outer_adapter_type),
    ).to(device=device, dtype=outer_dtype)
    outer_tr = CrossModelAdapter(
        toolcaller_hidden,
        reflector_hidden,
        normalize_outer_type(args.outer_tr_type, args.outer_adapter_type),
    ).to(device=device, dtype=outer_dtype)
    outer_rt.train()
    outer_tr.train()

    params = list(outer_rt.parameters()) + list(outer_tr.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.outer_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = load_outer_training_dataset(args.dataset_name, args.dataset_split, args.dataset_json_field)
    validate_delib_dataset_columns(dataset.column_names)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    rows = []
    for sample in dataset:
        rows.append(
            {
                "question": str(sample.get("question", "")).strip(),
                "reflector_text": first_present_text(sample, DELIB_REFLECTOR_FIELD_CANDIDATES),
                "toolcaller_text": first_present_text(sample, DELIB_TOOLCALLER_FIELD_CANDIDATES),
                "type": str(sample.get("type", "complete")).strip().lower() or "complete",
                "task_family": str(sample.get("task_family", args.mas_task)).strip().lower()
                or str(args.mas_task).strip().lower(),
                "fn_name": sample.get("fn_name", None),
            }
        )

    dataloader = DataLoader(rows, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: x)
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Increase dataset size or reduce batch_size.")

    max_train_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * len(dataloader)
    if args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=max_train_steps,
            num_cycles=0.5,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    skipped_count = 0
    log_loss = 0.0
    log_r_first = 0.0
    log_r_last = 0.0
    log_grad_rt = 0.0
    log_grad_tr = 0.0
    log_count = 0
    start_time = time.time()

    while global_step < max_train_steps:
        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            optimizer.zero_grad(set_to_none=True)
            valid_count = 0
            batch_loss_sum = 0.0
            sample_r_first_losses: List[float] = []
            sample_r_last_losses: List[float] = []

            for sample in batch:
                question = str(sample["question"]).strip()
                reflector_text = str(sample["reflector_text"]).strip()
                toolcaller_text = str(sample["toolcaller_text"]).strip()
                task_type = str(sample.get("type", "complete")).strip().lower() or "complete"
                task_family = str(sample.get("task_family", args.mas_task)).strip().lower() or str(args.mas_task).strip().lower()
                fn_name = sample.get("fn_name", None)
                if not question or not reflector_text or not toolcaller_text:
                    skipped_count += 1
                    continue

                try:
                    feedback_to_reflector: Optional[torch.Tensor] = None
                    round_losses: List[torch.Tensor] = []

                    for round_idx in range(args.num_recursive_rounds):
                        if round_idx == 0:
                            input_ids, attention_mask, assistant_mask = build_reflector_teacher_forced_inputs(
                                reflector_tok,
                                question=question,
                                reflector_text=reflector_text,
                                enable_thinking=enable_thinking,
                                system_prompt=system_prompt,
                                device=device,
                                max_length=args.max_length,
                                mas_task=task_family,
                                task_type=task_type,
                                fn_name=fn_name,
                            )
                            with torch.no_grad():
                                reflector_out = run_model_hidden(
                                    reflector_model,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    output_hidden_states=True,
                                )
                            reflector_hidden_states = reflector_out.hidden_states[-1][0][assistant_mask]
                        else:
                            if feedback_to_reflector is None or feedback_to_reflector.size(0) == 0:
                                round_losses = []
                                break
                            reflector_user_with_slot = build_deliberation_reflector_prompt_with_feedback_slot(
                                question,
                                mas_task=task_family,
                                task_type=task_type,
                                fn_name=fn_name,
                            )
                            reflector_pack = build_stage_with_slot_and_system(
                                tokenizer=reflector_tok,
                                embedding_layer=reflector_embed,
                                user_prompt_with_slot=reflector_user_with_slot,
                                assistant_text=reflector_text,
                                slot_text=DELIBERATION_FEEDBACK_SLOT,
                                slot_embeds=feedback_to_reflector,
                                enable_thinking=enable_thinking,
                                system_prompt=system_prompt,
                                device=device,
                                embed_dtype=reflector_embed.weight.dtype,
                                max_length=args.max_length,
                            )
                            reflector_out = run_model_hidden(
                                reflector_model,
                                attention_mask=reflector_pack.attention_mask,
                                inputs_embeds=reflector_pack.inputs_embeds,
                                output_hidden_states=True,
                            )
                            reflector_hidden_states = reflector_out.hidden_states[-1][0][reflector_pack.assistant_mask]

                        if reflector_hidden_states.size(0) == 0:
                            round_losses = []
                            break

                        reflector_inner_hidden = run_inner_adapter_preserve_input_grad(
                            reflector_inner,
                            reflector_hidden_states,
                            out_dtype=model_dtype,
                        )
                        reflector_to_toolcaller = trim_latent(
                            run_outer_adapter(
                                outer_rt,
                                reflector_inner_hidden,
                                out_dtype=toolcaller_embed.weight.dtype,
                            ),
                            args.max_latent_tokens,
                        )
                        if reflector_to_toolcaller.size(0) == 0:
                            round_losses = []
                            break

                        toolcaller_user_with_slot = build_deliberation_toolcaller_prompt_with_slot(
                            question,
                            mas_task=task_family,
                            task_type=task_type,
                            fn_name=fn_name,
                        )
                        toolcaller_pack = build_stage_with_slot_and_system(
                            tokenizer=toolcaller_tok,
                            embedding_layer=toolcaller_embed,
                            user_prompt_with_slot=toolcaller_user_with_slot,
                            assistant_text=toolcaller_text,
                            slot_text=DELIBERATION_REFLECTOR_SLOT,
                            slot_embeds=reflector_to_toolcaller,
                            enable_thinking=enable_thinking,
                            system_prompt=system_prompt,
                            device=device,
                            embed_dtype=toolcaller_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        need_feedback = round_idx < args.num_recursive_rounds - 1
                        toolcaller_out = run_model_hidden(
                            toolcaller_model,
                            attention_mask=toolcaller_pack.attention_mask,
                            inputs_embeds=toolcaller_pack.inputs_embeds,
                            output_hidden_states=need_feedback,
                        )
                        loss_round = compute_solver_ce_loss(toolcaller_out.logits, toolcaller_pack.labels)
                        if torch.isnan(loss_round) or torch.isinf(loss_round):
                            round_losses = []
                            break
                        round_losses.append(loss_round)

                        if need_feedback:
                            toolcaller_hidden_states = toolcaller_out.hidden_states[-1][0][toolcaller_pack.assistant_mask]
                            if toolcaller_hidden_states.size(0) == 0:
                                round_losses = []
                                break
                            toolcaller_inner_hidden = run_inner_adapter_preserve_input_grad(
                                toolcaller_inner,
                                toolcaller_hidden_states,
                                out_dtype=model_dtype,
                            )
                            feedback_to_reflector = trim_latent(
                                run_outer_adapter(
                                    outer_tr,
                                    toolcaller_inner_hidden,
                                    out_dtype=reflector_embed.weight.dtype,
                                ),
                                args.max_latent_tokens,
                            )

                    if not round_losses:
                        skipped_count += 1
                        continue

                    if args.supervise_final_only:
                        loss = round_losses[-1]
                    elif len(round_losses) > 1:
                        loss = round_losses[-1] + args.non_last_loss_weight * torch.stack(round_losses[:-1]).mean()
                    else:
                        loss = round_losses[-1]

                    (loss / max(args.batch_size, 1)).backward()
                    valid_count += 1
                    batch_loss_sum += float(loss.item())
                    sample_r_first_losses.append(float(round_losses[0].item()))
                    sample_r_last_losses.append(float(round_losses[-1].item()))
                except RuntimeError as exc:
                    if "sequence_too_long" in str(exc).lower():
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

            grad_rt_value = grad_norm(outer_rt)
            grad_tr_value = grad_norm(outer_tr)

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_batch = batch_loss_sum / valid_count
            log_loss += float(loss_batch)
            log_r_first += sum(sample_r_first_losses) / max(len(sample_r_first_losses), 1)
            log_r_last += sum(sample_r_last_losses) / max(len(sample_r_last_losses), 1)
            log_grad_rt += grad_rt_value
            log_grad_tr += grad_tr_value
            log_count += 1

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                grad_text = ""
                if args.log_grad_norm:
                    grad_text = (
                        f" grad_rt={log_grad_rt / max(log_count, 1):.3e}"
                        f" grad_tr={log_grad_tr / max(log_count, 1):.3e}"
                    )
                print(f"step={global_step} loss={log_loss / max(log_count, 1):.4f}", flush=True)
                log_loss = 0.0
                log_r_first = 0.0
                log_r_last = 0.0
                log_grad_rt = 0.0
                log_grad_tr = 0.0
                log_count = 0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_delib_outer_checkpoint(args.save_dir, global_step, outer_rt, outer_tr, args)

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    save_delib_outer_checkpoint(args.save_dir, None, outer_rt, outer_tr, args)
    elapsed = time.time() - start_time


if __name__ == "__main__":
    main()
