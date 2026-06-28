import argparse
import json
import math
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed as accel_set_seed
try:
    from transformers import AutoTokenizer, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
except Exception:
    from transformers import AutoTokenizer
    from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from data import ALL_MAS_ROLES, build_dataloader
from model import INNER_ADAPTER_TYPES, INNER_ADAPTER_ALIASES, LatentReasoningModel, resolve_local_pretrained_path

try:  # keep training output minimal: suppress datasets .map() progress bars
    import datasets as _datasets
    _datasets.disable_progress_bars()
except Exception:
    pass


def resolve_dtype(dtype_str: str) -> Optional[torch.dtype]:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return None


def build_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    adapter_lr: float,
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.adapter.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": weight_decay, "lr": adapter_lr})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0, "lr": adapter_lr})
    if not param_groups:
        raise ValueError("No trainable adapter parameters found.")

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.95))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument("--mas_role", type=str, default=None, help="Role-specific supervision data: planner/refiner/solver.")
    parser.add_argument("--dataset_json_field", type=str, default=None, help="Top-level field for local json dataset (e.g., data).")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--adapter_lr", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--adapter_dtype", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="ln_res_adapter",
        choices=sorted(INNER_ADAPTER_TYPES | set(INNER_ADAPTER_ALIASES)),
        help="Inner latent adapter type. The release inference runner only loads "
             "'ln_res_adapter' inner weights, so this is the default.",
    )
    parser.add_argument("--adapter_cos_weight", type=float, default=1.0)
    parser.add_argument("--adapter_mse_weight", type=float, default=0.0)
    parser.add_argument( "--enable_thinking", type=int, default=0, choices=[0, 1], help="0 is non-thinking mode, pass to tokenizer",)
    parser.add_argument("--solver_pre_question", type=int, default=0, choices=[0, 1])
    parser.add_argument("--mas_task", type=str, default="math", choices=["math", "code", "choice"])
    parser.add_argument("--mas_design", type=str, default="sequential", choices=["sequential", "hie", "distill", "deliberation"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mas_role is not None and args.mas_role not in ALL_MAS_ROLES:
        raise ValueError(f"--mas_role must be one of: {sorted(ALL_MAS_ROLES)}")

    mixed_precision = "no"
    if args.dtype == "float16":
        mixed_precision = "fp16"
    elif args.dtype == "bfloat16":
        mixed_precision = "bf16"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    accel_set_seed(args.seed)

    load_path = resolve_local_pretrained_path(args.load_dir or args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        text_field=args.text_field,
        enable_thinking=bool(args.enable_thinking),
        mas_role=args.mas_role,
        dataset_json_field=args.dataset_json_field,
        solver_pre_question=args.solver_pre_question,
        mas_task=args.mas_task,
        mas_design=args.mas_design,
    )

    adapter_overrides = {}
    if args.load_dir:
        config_path = os.path.join(args.load_dir, "adapter_config.json")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                adapter_overrides = json.load(f)

    model = LatentReasoningModel(
        model_name_or_path=load_path,
        adapter_type=adapter_overrides.get("adapter_type", args.adapter_type),
        torch_dtype=resolve_dtype(args.dtype),
        adapter_dtype=None if args.adapter_dtype == "auto" else resolve_dtype(args.adapter_dtype),
        trust_remote_code=args.trust_remote_code,
    )

    if args.load_dir:
        adapter_path = os.path.join(args.load_dir, "adapter.pt")
        if os.path.isfile(adapter_path):
            model.adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"), strict=True)

    optimizer = build_optimizer(
        model=model,
        weight_decay=args.weight_decay,
        adapter_lr=args.adapter_lr,
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    max_train_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * num_update_steps_per_epoch

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

    scheduler = accelerator.prepare(scheduler)
    model.train()

    def save_checkpoint(step: Optional[int] = None) -> None:
        if args.save_dir is None:
            return
        output_dir = os.path.join(args.save_dir, f"checkpoint-{step}") if step else args.save_dir
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            unwrapped.save_adapter(output_dir)
            with open(os.path.join(output_dir, "train_args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, sort_keys=True)

    global_step = 0
    start_time = time.time()
    log_stats = torch.zeros(4, device=accelerator.device)  # loss, cos_w, mse_w, samples
    step_stats = torch.zeros(4, device=accelerator.device)

    while global_step < max_train_steps:
        for _, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    loss_mask=batch.get("loss_mask"),
                    latent_mask=batch.get("latent_mask"),
                    adapter_cos_weight=args.adapter_cos_weight,
                    adapter_mse_weight=args.adapter_mse_weight,
                )

                bs = batch["input_ids"].size(0)
                step_stats[0] += outputs["loss"].item() * bs
                step_stats[1] += outputs["adapter_cosine_weighted"].item() * bs
                step_stats[2] += outputs["adapter_mse_weighted"].item() * bs
                step_stats[3] += bs

                accelerator.backward(outputs["loss"] / args.grad_accum_steps)

                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if step_stats[3] > 0:
                        log_stats += step_stats
                    step_stats.zero_()

                    if global_step % args.log_every == 0:
                        total = accelerator.reduce(log_stats, reduction="sum")
                        denom = max(total[3].item(), 1.0)
                        avg = total[:3] / denom
                        if accelerator.is_main_process:
                            print(f"step={global_step} loss={avg[0]:.4f}", flush=True)
                        log_stats.zero_()

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        save_checkpoint(global_step)

                    if args.max_steps > 0 and global_step >= args.max_steps:
                        break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint()


if __name__ == "__main__":
    main()
