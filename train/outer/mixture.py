import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from mas_prompt import (
    HIE_CODE_EXPERT_SLOT,
    HIE_FEEDBACK_SLOT,
    HIE_MATH_EXPERT_SLOT,
    HIE_SCIENCE_EXPERT_SLOT,
    build_hie_expert_prompt,
    build_hie_expert_prompt_with_feedback_slot,
    build_hie_summarizer_prompt_with_slots,
)
from model import CrossModelAdapter
from .common import (
    StagePack,
    compute_solver_ce_loss,
    ids_to_embeds,
    load_inner_adapter,
    load_model_and_tokenizer,
    load_outer_training_dataset,
    render_chat_ids,
    render_chat_text,
    run_inner_adapter_preserve_input_grad,
    run_outer_adapter,
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

HIE_EXPERT_ROLES = ("hie_math_expert", "hie_code_expert", "hie_science_expert")
HIE_EXPERT_FIELD_CANDIDATES = {
    "hie_math_expert": ("hie_math_expert", "math_expert", "expert_math"),
    "hie_code_expert": ("hie_code_expert", "code_expert", "expert_code"),
    "hie_science_expert": ("hie_science_expert", "science_expert", "expert_science"),
}
HIE_SUMMARIZER_TARGET_CANDIDATES = ("answer", "hie_summarizer", "summary")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent1_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent2_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent3_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent4_model_name_or_path", type=str, required=True)

    parser.add_argument("--agent1_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent2_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent3_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent4_inner_aligner_path", type=str, required=True)
    parser.add_argument(
        "--inner_adapter_type_fallback",
        type=str,
        default="res_adapter",
        choices=["1layer", "1layer_res", "2layer", "2layer_res", "2layer_ln_res", "adapter", "linear_adapter", "linear_res_adapter", "ln_res_adapter", "res_adapter"],
    )

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_json_field", type=str, default="data")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mas_shape", type=str, default="hie", choices=["hie"])
    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code", "choice"])
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gradient_checkpointing", type=int, default=1, choices=[0, 1])

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_latent_tokens", type=int, default=80)

    parser.add_argument("--batch_size", type=int, default=1)
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
    parser.add_argument("--outer_1s_type", type=str, default=None)
    parser.add_argument("--outer_2s_type", type=str, default=None)
    parser.add_argument("--outer_3s_type", type=str, default=None)
    parser.add_argument("--outer_s1_type", type=str, default=None)
    parser.add_argument("--outer_s2_type", type=str, default=None)
    parser.add_argument("--outer_s3_type", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--outer_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=0)
    return parser.parse_args(argv)


def resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return None


def split_text_by_ordered_slots(text: str, slot_texts: Sequence[str]) -> List[str]:
    parts: List[str] = []
    cursor = 0
    for slot_text in slot_texts:
        pos = text.find(slot_text, cursor)
        if pos < 0:
            raise RuntimeError(f"Failed to locate slot marker {slot_text!r} in rendered chat text.")
        parts.append(text[cursor:pos])
        cursor = pos + len(slot_text)
    parts.append(text[cursor:])
    return parts


def build_stage_with_hie_slots(
    tokenizer,
    embedding_layer,
    user_prompt_with_slots: str,
    assistant_text: str,
    slot_texts: Sequence[str],
    slot_embeds: Sequence[torch.Tensor],
    enable_thinking: bool,
    device: torch.device,
    embed_dtype: torch.dtype,
    max_length: int,
) -> StagePack:
    if len(slot_texts) != len(slot_embeds):
        raise ValueError("slot_texts and slot_embeds must have the same length.")

    prompt_rendered = render_chat_text(
        tokenizer,
        user_prompt_with_slots,
        assistant_text=None,
        enable_thinking=enable_thinking,
    )
    full_rendered = render_chat_text(
        tokenizer,
        user_prompt_with_slots,
        assistant_text=assistant_text,
        enable_thinking=enable_thinking,
    )

    prompt_parts = split_text_by_ordered_slots(prompt_rendered, slot_texts)
    full_parts = split_text_by_ordered_slots(full_rendered, slot_texts)
    prompt_part_ids = [text_to_ids(tokenizer, part) for part in prompt_parts]
    full_part_ids = [text_to_ids(tokenizer, part) for part in full_parts]
    slot_lengths = [int(x.size(0)) for x in slot_embeds]

    prompt_len = sum(len(ids) for ids in prompt_part_ids) + sum(slot_lengths)

    token_ids: List[int] = []
    embed_chunks: List[torch.Tensor] = []
    for idx, part_ids in enumerate(full_part_ids):
        token_ids.extend(part_ids)
        embed_chunks.append(ids_to_embeds(embedding_layer, part_ids, device=device, dtype=embed_dtype))
        if idx < len(slot_embeds):
            slot = slot_embeds[idx].to(embed_dtype) if slot_embeds[idx].dtype != embed_dtype else slot_embeds[idx]
            token_ids.extend([-100] * int(slot.size(0)))
            embed_chunks.append(slot)

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

    seq_embeds = torch.cat(embed_chunks, dim=0)
    if truncate_left > 0:
        seq_embeds = seq_embeds[truncate_left:]
    attention_mask = torch.ones((seq_embeds.size(0),), dtype=torch.long, device=device)

    return StagePack(
        inputs_embeds=seq_embeds.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        labels=labels.unsqueeze(0),
        assistant_mask=assistant_mask,
    )


def build_hie_teacher_forced_inputs(
    tokenizer,
    question: str,
    assistant_text: str,
    hie_role: str,
    enable_thinking: bool,
    device: torch.device,
    max_length: int,
    mas_task: str,
    task_type: str,
    fn_name: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    user_prompt = build_hie_expert_prompt(
        question,
        hie_role,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )
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
        assistant_text=assistant_text,
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


def first_present_text(sample: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = sample.get(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def validate_hie_dataset_columns(column_names: Sequence[str]) -> None:
    columns = set(column_names)
    missing_groups = []
    for role, keys in HIE_EXPERT_FIELD_CANDIDATES.items():
        if not columns.intersection(keys):
            missing_groups.append(f"{role}: one of {list(keys)}")
    if not columns.intersection(HIE_SUMMARIZER_TARGET_CANDIDATES):
        missing_groups.append(f"hie_summarizer target: one of {list(HIE_SUMMARIZER_TARGET_CANDIDATES)}")
    if "question" not in columns:
        missing_groups.append("question")
    if missing_groups:
        raise ValueError(f"Dataset missing hierarchical fields: {missing_groups}")


def save_hie_outer_checkpoint(
    save_dir: str,
    step: Optional[int],
    outers: Dict[str, CrossModelAdapter],
    args: argparse.Namespace,
) -> None:
    output_dir = os.path.join(save_dir, f"checkpoint-{step}") if step is not None else save_dir
    os.makedirs(output_dir, exist_ok=True)

    for name, adapter in outers.items():
        torch.save(adapter.state_dict(), os.path.join(output_dir, f"{name}.pt"))

    cfg = {
        "mas_shape": "hie",
        "mas_task": args.mas_task,
        "num_recursive_rounds": args.num_recursive_rounds,
        "supervise_final_only": args.supervise_final_only,
        "non_last_loss_weight": args.non_last_loss_weight,
        "agent1_model_name_or_path": args.agent1_model_name_or_path,
        "agent2_model_name_or_path": args.agent2_model_name_or_path,
        "agent3_model_name_or_path": args.agent3_model_name_or_path,
        "agent4_model_name_or_path": args.agent4_model_name_or_path,
        "agent1_inner_aligner_path": args.agent1_inner_aligner_path,
        "agent2_inner_aligner_path": args.agent2_inner_aligner_path,
        "agent3_inner_aligner_path": args.agent3_inner_aligner_path,
        "agent4_inner_aligner_path": args.agent4_inner_aligner_path,
    }
    for name, adapter in outers.items():
        cfg[f"{name}_type"] = adapter.adapter_type
        cfg[f"{name}_in_dim"] = adapter.in_dim
        cfg[f"{name}_out_dim"] = adapter.out_dim
    with open(os.path.join(output_dir, "outer_adapter_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    write_outerlink_manifest(output_dir, "mixture", [
        {"legacy_key": name, "filename": f"{name}.pt", "adapter_type": adapter.adapter_type,
         "in_dim": adapter.in_dim, "out_dim": adapter.out_dim}
        for name, adapter in outers.items()
    ])


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


def main(argv=None) -> None:
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

    models = []
    tokenizers = []
    embeds = []
    model_specs = [
        (args.agent1_model_name_or_path, "hie_math_expert"),
        (args.agent2_model_name_or_path, "hie_code_expert"),
        (args.agent3_model_name_or_path, "hie_science_expert"),
        (args.agent4_model_name_or_path, "hie_summarizer"),
    ]
    for model_name, agent_name in model_specs:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            device=device,
            dtype=model_dtype,
            trust_remote_code=args.trust_remote_code,
            agent_name=agent_name,
            gradient_checkpointing=bool(args.gradient_checkpointing),
        )
        if bool(args.gradient_checkpointing):
            activate_gc_runtime(model, agent_name)
        models.append(model)
        tokenizers.append(tokenizer)
        embeds.append(model.get_input_embeddings())

    expert_models = models[:3]
    summarizer_model = models[3]
    expert_toks = tokenizers[:3]
    summarizer_tok = tokenizers[3]
    expert_embeds = embeds[:3]
    summarizer_embed = embeds[3]
    hidden_sizes = [embed.weight.size(-1) for embed in embeds]

    inner_paths = [
        args.agent1_inner_aligner_path,
        args.agent2_inner_aligner_path,
        args.agent3_inner_aligner_path,
        args.agent4_inner_aligner_path,
    ]
    inners = [
        load_inner_adapter(
            path,
            hidden_size=hidden_size,
            device=device,
            dtype=model_dtype,
            fallback_adapter_type=args.inner_adapter_type_fallback,
        )
        for path, hidden_size in zip(inner_paths, hidden_sizes)
    ]

    outer_types = {
        "outer_1s": normalize_outer_type(args.outer_1s_type, args.outer_adapter_type),
        "outer_2s": normalize_outer_type(args.outer_2s_type, args.outer_adapter_type),
        "outer_3s": normalize_outer_type(args.outer_3s_type, args.outer_adapter_type),
        "outer_s1": normalize_outer_type(args.outer_s1_type, args.outer_adapter_type),
        "outer_s2": normalize_outer_type(args.outer_s2_type, args.outer_adapter_type),
        "outer_s3": normalize_outer_type(args.outer_s3_type, args.outer_adapter_type),
    }
    outers = {
        "outer_1s": CrossModelAdapter(hidden_sizes[0], hidden_sizes[3], outer_types["outer_1s"]).to(
            device=device, dtype=outer_dtype
        ),
        "outer_2s": CrossModelAdapter(hidden_sizes[1], hidden_sizes[3], outer_types["outer_2s"]).to(
            device=device, dtype=outer_dtype
        ),
        "outer_3s": CrossModelAdapter(hidden_sizes[2], hidden_sizes[3], outer_types["outer_3s"]).to(
            device=device, dtype=outer_dtype
        ),
        "outer_s1": CrossModelAdapter(hidden_sizes[3], hidden_sizes[0], outer_types["outer_s1"]).to(
            device=device, dtype=outer_dtype
        ),
        "outer_s2": CrossModelAdapter(hidden_sizes[3], hidden_sizes[1], outer_types["outer_s2"]).to(
            device=device, dtype=outer_dtype
        ),
        "outer_s3": CrossModelAdapter(hidden_sizes[3], hidden_sizes[2], outer_types["outer_s3"]).to(
            device=device, dtype=outer_dtype
        ),
    }
    for adapter in outers.values():
        adapter.train()

    params = [param for adapter in outers.values() for param in adapter.parameters()]
    optimizer = torch.optim.AdamW(params, lr=args.outer_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = load_outer_training_dataset(args.dataset_name, args.dataset_split, args.dataset_json_field)
    validate_hie_dataset_columns(dataset.column_names)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    rows = []
    for sample in dataset:
        row = {
            "question": str(sample.get("question", "")).strip(),
            "answer": first_present_text(sample, HIE_SUMMARIZER_TARGET_CANDIDATES),
            "type": str(sample.get("type", "complete")).strip().lower() or "complete",
            "task_family": str(sample.get("task_family", args.mas_task)).strip().lower() or str(args.mas_task).strip().lower(),
            "fn_name": sample.get("fn_name", None),
        }
        for role, keys in HIE_EXPERT_FIELD_CANDIDATES.items():
            row[role] = first_present_text(sample, keys)
        rows.append(row)

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
                answer = str(sample["answer"]).strip()
                task_type = str(sample.get("type", "complete")).strip().lower() or "complete"
                task_family = str(sample.get("task_family", args.mas_task)).strip().lower() or str(args.mas_task).strip().lower()
                fn_name = sample.get("fn_name", None)
                expert_texts = [str(sample[role]).strip() for role in HIE_EXPERT_ROLES]
                if not question or not answer or any(not text for text in expert_texts):
                    skipped_count += 1
                    continue

                try:
                    feedback_to_experts: List[Optional[torch.Tensor]] = [None, None, None]
                    round_losses: List[torch.Tensor] = []

                    for round_idx in range(args.num_recursive_rounds):
                        expert_to_summary: List[torch.Tensor] = []
                        for expert_idx, role in enumerate(HIE_EXPERT_ROLES):
                            if round_idx == 0:
                                input_ids, attention_mask, assistant_mask = build_hie_teacher_forced_inputs(
                                    expert_toks[expert_idx],
                                    question=question,
                                    assistant_text=expert_texts[expert_idx],
                                    hie_role=role,
                                    enable_thinking=enable_thinking,
                                    device=device,
                                    max_length=args.max_length,
                                    mas_task=task_family,
                                    task_type=task_type,
                                    fn_name=fn_name,
                                )
                                with torch.no_grad():
                                    expert_out = run_model_hidden(
                                        expert_models[expert_idx],
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        output_hidden_states=True,
                                    )
                                expert_hidden = expert_out.hidden_states[-1][0][assistant_mask]
                            else:
                                feedback = feedback_to_experts[expert_idx]
                                if feedback is None or feedback.size(0) == 0:
                                    round_losses = []
                                    break
                                user_with_slot = build_hie_expert_prompt_with_feedback_slot(
                                    question,
                                    role,
                                    mas_task=task_family,
                                    task_type=task_type,
                                    fn_name=fn_name,
                                )
                                pack = build_stage_with_hie_slots(
                                    tokenizer=expert_toks[expert_idx],
                                    embedding_layer=expert_embeds[expert_idx],
                                    user_prompt_with_slots=user_with_slot,
                                    assistant_text=expert_texts[expert_idx],
                                    slot_texts=[HIE_FEEDBACK_SLOT],
                                    slot_embeds=[feedback],
                                    enable_thinking=enable_thinking,
                                    device=device,
                                    embed_dtype=expert_embeds[expert_idx].weight.dtype,
                                    max_length=args.max_length,
                                )
                                expert_out = run_model_hidden(
                                    expert_models[expert_idx],
                                    attention_mask=pack.attention_mask,
                                    inputs_embeds=pack.inputs_embeds,
                                    output_hidden_states=True,
                                )
                                expert_hidden = expert_out.hidden_states[-1][0][pack.assistant_mask]

                            if expert_hidden.size(0) == 0:
                                round_losses = []
                                break
                            expert_inner = run_inner_adapter_preserve_input_grad(inners[expert_idx], expert_hidden, out_dtype=model_dtype)
                            outer_name = f"outer_{expert_idx + 1}s"
                            transferred = run_outer_adapter(
                                outers[outer_name],
                                expert_inner,
                                out_dtype=summarizer_embed.weight.dtype,
                            )
                            expert_to_summary.append(trim_latent(transferred, args.max_latent_tokens))

                        if len(expert_to_summary) != 3:
                            break

                        summarizer_user = build_hie_summarizer_prompt_with_slots(
                            question,
                            mas_task=task_family,
                            task_type=task_type,
                            fn_name=fn_name,
                        )
                        summarizer_pack = build_stage_with_hie_slots(
                            tokenizer=summarizer_tok,
                            embedding_layer=summarizer_embed,
                            user_prompt_with_slots=summarizer_user,
                            assistant_text=answer,
                            slot_texts=[
                                HIE_MATH_EXPERT_SLOT,
                                HIE_CODE_EXPERT_SLOT,
                                HIE_SCIENCE_EXPERT_SLOT,
                            ],
                            slot_embeds=expert_to_summary,
                            enable_thinking=enable_thinking,
                            device=device,
                            embed_dtype=summarizer_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        need_feedback = round_idx < args.num_recursive_rounds - 1
                        summarizer_out = run_model_hidden(
                            summarizer_model,
                            attention_mask=summarizer_pack.attention_mask,
                            inputs_embeds=summarizer_pack.inputs_embeds,
                            output_hidden_states=need_feedback,
                        )
                        loss_round = compute_solver_ce_loss(summarizer_out.logits, summarizer_pack.labels)
                        if torch.isnan(loss_round) or torch.isinf(loss_round):
                            round_losses = []
                            break
                        round_losses.append(loss_round)

                        if need_feedback:
                            summarizer_hidden = summarizer_out.hidden_states[-1][0][summarizer_pack.assistant_mask]
                            if summarizer_hidden.size(0) == 0:
                                round_losses = []
                                break
                            summarizer_inner = run_inner_adapter_preserve_input_grad(inners[3], summarizer_hidden, out_dtype=model_dtype)
                            feedback_to_experts = [
                                trim_latent(
                                    run_outer_adapter(
                                        outers[f"outer_s{idx + 1}"],
                                        summarizer_inner,
                                        out_dtype=expert_embeds[idx].weight.dtype,
                                    ),
                                    args.max_latent_tokens,
                                )
                                for idx in range(3)
                            ]

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

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_batch = batch_loss_sum / valid_count
            log_loss += float(loss_batch)
            log_r_first += sum(sample_r_first_losses) / max(len(sample_r_first_losses), 1)
            log_r_last += sum(sample_r_last_losses) / max(len(sample_r_last_losses), 1)
            log_count += 1

            if global_step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"step={global_step} loss={log_loss / max(log_count, 1):.4f}", flush=True)
                log_loss = 0.0
                log_r_first = 0.0
                log_r_last = 0.0
                log_count = 0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_hie_outer_checkpoint(args.save_dir, global_step, outers, args)

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    save_hie_outer_checkpoint(args.save_dir, None, outers, args)
    elapsed = time.time() - start_time


if __name__ == "__main__":
    main()
