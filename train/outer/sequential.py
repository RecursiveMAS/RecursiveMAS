import argparse
import json
import math
import os
import time
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from mas_prompt import (
    FEEDBACK_SLOT,
    PLANNER_SLOT,
    REFINED_SLOT,
    build_code_planner_prompt_with_feedback_slot,
    build_code_refiner_prompt_with_slot,
    build_code_solver_prompt_with_slots,
    build_math_planner_prompt_with_feedback_slot,
    build_math_refiner_prompt_with_slot,
    build_math_solver_prompt_with_slots,
)
from model import CrossModelAdapter
from .common import (
    build_planner_teacher_forced_inputs,
    build_stage_with_slot,
    compute_solver_ce_loss,
    load_inner_adapter,
    load_model_and_tokenizer,
    load_outer_training_dataset,
    resolve_dtype,
    run_inner_adapter_preserve_input_grad,
    run_outer_adapter,
    trim_latent,
    write_outerlink_manifest,
)

ALLOWED_OUTER_TYPES = {
    "outer_linear_adapter",
    "outer_linear_res_adapter",
    "outer_adapter",
    "outer_res_adapter",
    "outer_ln_adapter",
    "outer_ln_res_adapter",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent1_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent2_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent3_model_name_or_path", type=str, required=True)

    parser.add_argument("--agent1_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent2_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent3_inner_aligner_path", type=str, required=True)
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

    parser.add_argument("--mas_shape", type=str, default="chain", choices=["chain"], help=argparse.SUPPRESS)
    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code"])
    parser.add_argument("--solver_pre_question", type=int, default=0)
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
    parser.add_argument(
        "--num_recursive_rounds",
        type=int,
        default=3,
        help="Number of recursive rounds (planner->refiner->solver cycles).",
    )
    parser.add_argument(
        "--supervise_final_only",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, optimize only the final-round CE loss.",
    )
    parser.add_argument(
        "--non_last_loss_weight",
        type=float,
        default=0.1,
        help="When supervise_final_only=0, add this weight * mean(non-last round losses).",
    )

    parser.add_argument(
        "--outer_adapter_type",
        type=str,
        default="outer_ln_res_adapter",
        choices=sorted(ALLOWED_OUTER_TYPES),
    )
    parser.add_argument("--outer_12_type", type=str, default=None)
    parser.add_argument("--outer_23_type", type=str, default=None)
    parser.add_argument("--outer_31_type", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--outer_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=0)
    return parser.parse_args(argv)


def normalize_outer_type(name: Optional[str], fallback: str) -> str:
    out = fallback if name is None else name
    if out not in ALLOWED_OUTER_TYPES:
        raise ValueError(f"Unsupported outer adapter type: {out}")
    return out




def activate_gc_runtime(model: torch.nn.Module, model_name: str) -> None:
    # In many HF models, gradient-checkpointing branches are gated by `self.training`.
    # Keep params frozen, switch to train-mode only to activate checkpointing,
    # and disable dropout for near-deterministic behavior.
    model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    dropout_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            dropout_count += 1



def build_optional_token_type_ids(
    model: torch.nn.Module,
    attention_mask: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower() if config is not None else ""

    if model_type != "gemma3" or not model.training:
        return None

    if input_ids is not None:
        base = input_ids
    elif attention_mask is not None:
        base = attention_mask
    elif inputs_embeds is not None:
        base = inputs_embeds[..., 0]
    else:
        return None

    return torch.zeros_like(base, dtype=torch.long)

def save_recursive_outer_checkpoint(
    save_dir: str,
    step: Optional[int],
    outer_12: CrossModelAdapter,
    outer_23: CrossModelAdapter,
    outer_31: CrossModelAdapter,
    args: argparse.Namespace,
) -> None:
    output_dir = os.path.join(save_dir, f"checkpoint-{step}") if step is not None else save_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.save(outer_12.state_dict(), os.path.join(output_dir, "outer_12.pt"))
    torch.save(outer_23.state_dict(), os.path.join(output_dir, "outer_23.pt"))
    torch.save(outer_31.state_dict(), os.path.join(output_dir, "outer_31.pt"))

    cfg = {
        "outer_12_type": outer_12.adapter_type,
        "outer_23_type": outer_23.adapter_type,
        "outer_31_type": outer_31.adapter_type,
        "outer_12_in_dim": outer_12.in_dim,
        "outer_12_out_dim": outer_12.out_dim,
        "outer_23_in_dim": outer_23.in_dim,
        "outer_23_out_dim": outer_23.out_dim,
        "outer_31_in_dim": outer_31.in_dim,
        "outer_31_out_dim": outer_31.out_dim,
        "mas_shape": args.mas_shape,
        "agent1_model_name_or_path": args.agent1_model_name_or_path,
        "agent2_model_name_or_path": args.agent2_model_name_or_path,
        "agent3_model_name_or_path": args.agent3_model_name_or_path,
        "agent1_inner_aligner_path": args.agent1_inner_aligner_path,
        "agent2_inner_aligner_path": args.agent2_inner_aligner_path,
        "agent3_inner_aligner_path": args.agent3_inner_aligner_path,
        "enable_thinking": args.enable_thinking,
        "mas_task": args.mas_task,
        "num_recursive_rounds": args.num_recursive_rounds,
        "supervise_final_only": args.supervise_final_only,
        "non_last_loss_weight": args.non_last_loss_weight,
    }
    with open(os.path.join(output_dir, "outer_adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    write_outerlink_manifest(output_dir, "sequential", [
        {"legacy_key": "outer_12", "filename": "outer_12.pt", "adapter_type": outer_12.adapter_type, "in_dim": outer_12.in_dim, "out_dim": outer_12.out_dim},
        {"legacy_key": "outer_23", "filename": "outer_23.pt", "adapter_type": outer_23.adapter_type, "in_dim": outer_23.in_dim, "out_dim": outer_23.out_dim},
        {"legacy_key": "outer_31", "filename": "outer_31.pt", "adapter_type": outer_31.adapter_type, "in_dim": outer_31.in_dim, "out_dim": outer_31.out_dim},
    ])


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mas_shape != "chain":
        raise ValueError("Only mas_shape=chain is supported.")
    if args.num_recursive_rounds <= 0:
        raise ValueError("--num_recursive_rounds must be positive.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")

    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back model dtype to float32.")
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU + fp16/bf16 is unstable. Falling back outer dtype to float32.")
        outer_dtype = torch.float32

    torch.manual_seed(args.seed)
    enable_thinking = bool(args.enable_thinking)
    solver_args = argparse.Namespace(solver_pre_question=args.solver_pre_question)

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

    if bool(args.gradient_checkpointing):
        activate_gc_runtime(planner_model, "planner")
        activate_gc_runtime(refiner_model, "refiner")
        activate_gc_runtime(solver_model, "solver")

    planner_embed = planner_model.get_input_embeddings()
    refiner_embed = refiner_model.get_input_embeddings()
    solver_embed = solver_model.get_input_embeddings()

    planner_hidden = planner_embed.weight.size(-1)
    refiner_hidden = refiner_embed.weight.size(-1)
    solver_hidden = solver_embed.weight.size(-1)

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
    inner_3 = load_inner_adapter(
        args.agent3_inner_aligner_path,
        hidden_size=solver_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=args.inner_adapter_type_fallback,
    )

    outer_12_type = normalize_outer_type(args.outer_12_type, args.outer_adapter_type)
    outer_23_type = normalize_outer_type(args.outer_23_type, args.outer_adapter_type)
    outer_31_type = normalize_outer_type(args.outer_31_type, args.outer_adapter_type)

    outer_12 = CrossModelAdapter(planner_hidden, refiner_hidden, outer_12_type).to(device=device, dtype=outer_dtype)
    outer_23 = CrossModelAdapter(refiner_hidden, solver_hidden, outer_23_type).to(device=device, dtype=outer_dtype)
    outer_31 = CrossModelAdapter(solver_hidden, planner_hidden, outer_31_type).to(device=device, dtype=outer_dtype)
    outer_12.train()
    outer_23.train()
    outer_31.train()

    params = list(outer_12.parameters()) + list(outer_23.parameters()) + list(outer_31.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.outer_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    dataset = load_outer_training_dataset(args.dataset_name, args.dataset_split, args.dataset_json_field)
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
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    start_time = time.time()
    skipped_count = 0

    log_loss = 0.0
    log_r_first = 0.0
    log_r_last = 0.0
    log_count = 0

    while global_step < max_train_steps:
        for batch in dataloader:
            if global_step >= max_train_steps:
                break

            sample_r_first_losses: List[float] = []
            sample_r_last_losses: List[float] = []
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
                    feedback_to_planner = None
                    round_losses: List[torch.Tensor] = []

                    for round_idx in range(args.num_recursive_rounds):
                        # Planner
                        if round_idx == 0:
                            planner_input_ids, planner_attention_mask, planner_assist_mask = (
                                build_planner_teacher_forced_inputs(
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
                            )
                            with torch.no_grad():
                                planner_token_type_ids = build_optional_token_type_ids(
                                    planner_model,
                                    attention_mask=planner_attention_mask,
                                    input_ids=planner_input_ids,
                                )
                                planner_kwargs = {}
                                if planner_token_type_ids is not None:
                                    planner_kwargs["token_type_ids"] = planner_token_type_ids
                                planner_out = planner_model(
                                    input_ids=planner_input_ids,
                                    attention_mask=planner_attention_mask,
                                    output_hidden_states=True,
                                    use_cache=False,
                                    return_dict=True,
                                    **planner_kwargs,
                                )
                            planner_hidden = planner_out.hidden_states[-1][0][planner_assist_mask]
                        else:
                            if feedback_to_planner is None or feedback_to_planner.size(0) == 0:
                                round_losses = []
                                break
                            if args.mas_task == "code":
                                planner_user_with_slot = build_code_planner_prompt_with_feedback_slot(
                                    q,
                                    task_type=task_type,
                                    fn_name=fn_name,
                                )
                            else:
                                planner_user_with_slot = build_math_planner_prompt_with_feedback_slot(q)

                            planner_pack = build_stage_with_slot(
                                tokenizer=planner_tok,
                                embedding_layer=planner_embed,
                                user_prompt_with_slot=planner_user_with_slot,
                                assistant_text=p,
                                slot_text=FEEDBACK_SLOT,
                                slot_embeds=feedback_to_planner,
                                enable_thinking=enable_thinking,
                                device=device,
                                embed_dtype=planner_embed.weight.dtype,
                                max_length=args.max_length,
                            )
                            planner_token_type_ids = build_optional_token_type_ids(
                                planner_model,
                                attention_mask=planner_pack.attention_mask,
                                inputs_embeds=planner_pack.inputs_embeds,
                            )
                            planner_kwargs = {}
                            if planner_token_type_ids is not None:
                                planner_kwargs["token_type_ids"] = planner_token_type_ids
                            planner_out = planner_model(
                                inputs_embeds=planner_pack.inputs_embeds,
                                attention_mask=planner_pack.attention_mask,
                                output_hidden_states=True,
                                use_cache=False,
                                return_dict=True,
                                **planner_kwargs,
                            )
                            planner_hidden = planner_out.hidden_states[-1][0][planner_pack.assistant_mask]

                        if planner_hidden.size(0) == 0:
                            round_losses = []
                            break
                        planner_inner = run_inner_adapter_preserve_input_grad(inner_1, planner_hidden, out_dtype=model_dtype)
                        planner_to_refiner = run_outer_adapter(
                            outer_12, planner_inner, out_dtype=refiner_embed.weight.dtype
                        )
                        planner_to_refiner = trim_latent(planner_to_refiner, args.max_latent_tokens)

                        # Refiner
                        if args.mas_task == "code":
                            refiner_user_with_slot = build_code_refiner_prompt_with_slot(
                                q,
                                task_type=task_type,
                                fn_name=fn_name,
                            )
                        else:
                            refiner_user_with_slot = build_math_refiner_prompt_with_slot(q)

                        refiner_pack = build_stage_with_slot(
                            tokenizer=refiner_tok,
                            embedding_layer=refiner_embed,
                            user_prompt_with_slot=refiner_user_with_slot,
                            assistant_text=rp,
                            slot_text=PLANNER_SLOT,
                            slot_embeds=planner_to_refiner,
                            enable_thinking=enable_thinking,
                            device=device,
                            embed_dtype=refiner_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        refiner_token_type_ids = build_optional_token_type_ids(
                            refiner_model,
                            attention_mask=refiner_pack.attention_mask,
                            inputs_embeds=refiner_pack.inputs_embeds,
                        )
                        refiner_kwargs = {}
                        if refiner_token_type_ids is not None:
                            refiner_kwargs["token_type_ids"] = refiner_token_type_ids
                        refiner_out = refiner_model(
                            inputs_embeds=refiner_pack.inputs_embeds,
                            attention_mask=refiner_pack.attention_mask,
                            output_hidden_states=True,
                            use_cache=False,
                            return_dict=True,
                            **refiner_kwargs,
                        )
                        refiner_hidden = refiner_out.hidden_states[-1][0][refiner_pack.assistant_mask]
                        if refiner_hidden.size(0) == 0:
                            round_losses = []
                            break
                        refiner_inner = run_inner_adapter_preserve_input_grad(inner_2, refiner_hidden, out_dtype=model_dtype)
                        refiner_to_solver = run_outer_adapter(
                            outer_23, refiner_inner, out_dtype=solver_embed.weight.dtype
                        )
                        refiner_to_solver = trim_latent(refiner_to_solver, args.max_latent_tokens)

                        # Solver
                        if args.mas_task == "code":
                            solver_user_with_slot = build_code_solver_prompt_with_slots(
                                q,
                                task_type=task_type,
                                args=solver_args,
                                mas_shape=args.mas_shape,
                                fn_name=fn_name,
                            )
                        else:
                            solver_user_with_slot = build_math_solver_prompt_with_slots(
                                q,
                                args=solver_args,
                                mas_shape=args.mas_shape,
                            )

                        solver_pack = build_stage_with_slot(
                            tokenizer=solver_tok,
                            embedding_layer=solver_embed,
                            user_prompt_with_slot=solver_user_with_slot,
                            assistant_text=ans,
                            slot_text=REFINED_SLOT,
                            slot_embeds=refiner_to_solver,
                            enable_thinking=enable_thinking,
                            device=device,
                            embed_dtype=solver_embed.weight.dtype,
                            max_length=args.max_length,
                        )
                        need_feedback = round_idx < args.num_recursive_rounds - 1
                        solver_token_type_ids = build_optional_token_type_ids(
                            solver_model,
                            attention_mask=solver_pack.attention_mask,
                            inputs_embeds=solver_pack.inputs_embeds,
                        )
                        solver_kwargs = {}
                        if solver_token_type_ids is not None:
                            solver_kwargs["token_type_ids"] = solver_token_type_ids
                        solver_out = solver_model(
                            inputs_embeds=solver_pack.inputs_embeds,
                            attention_mask=solver_pack.attention_mask,
                            output_hidden_states=need_feedback,
                            use_cache=False,
                            return_dict=True,
                            **solver_kwargs,
                        )
                        loss_round = compute_solver_ce_loss(solver_out.logits, solver_pack.labels)
                        if torch.isnan(loss_round) or torch.isinf(loss_round):
                            round_losses = []
                            break
                        round_losses.append(loss_round)

                        # Feedback for next round (solver -> planner)
                        if need_feedback:
                            solver_hidden = solver_out.hidden_states[-1][0][solver_pack.assistant_mask]
                            if solver_hidden.size(0) == 0:
                                round_losses = []
                                break
                            solver_inner = run_inner_adapter_preserve_input_grad(inner_3, solver_hidden, out_dtype=model_dtype)
                            feedback_to_planner = run_outer_adapter(
                                outer_31, solver_inner, out_dtype=planner_embed.weight.dtype
                            )
                            feedback_to_planner = trim_latent(feedback_to_planner, args.max_latent_tokens)

                    if not round_losses:
                        skipped_count += 1
                        continue

                    if args.supervise_final_only:
                        loss = round_losses[-1]
                    else:
                        if len(round_losses) > 1:
                            non_last_mean = torch.stack(round_losses[:-1]).mean()
                            loss = round_losses[-1] + args.non_last_loss_weight * non_last_mean
                        else:
                            loss = round_losses[-1]

                    (loss / max(args.batch_size, 1)).backward()
                    valid_count += 1
                    batch_loss_sum += float(loss.item())
                    sample_r_first_losses.append(float(round_losses[0].item()))
                    sample_r_last_losses.append(float(round_losses[-1].item()))
                except RuntimeError as exc:
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

            log_loss += float(loss_batch)
            log_r_first += sum(sample_r_first_losses) / max(len(sample_r_first_losses), 1)
            log_r_last += sum(sample_r_last_losses) / max(len(sample_r_last_losses), 1)
            log_count += 1

            if global_step % args.log_every == 0:
                avg_loss = log_loss / max(log_count, 1)
                avg_r_first = log_r_first / max(log_count, 1)
                avg_r_last = log_r_last / max(log_count, 1)
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"step={global_step} loss={avg_loss:.4f}", flush=True)
                log_loss = 0.0
                log_r_first = 0.0
                log_r_last = 0.0
                log_count = 0

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_recursive_outer_checkpoint(args.save_dir, global_step, outer_12, outer_23, outer_31, args)


            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    save_recursive_outer_checkpoint(args.save_dir, None, outer_12, outer_23, outer_31, args)
    elapsed = time.time() - start_time


if __name__ == "__main__":
    main()
