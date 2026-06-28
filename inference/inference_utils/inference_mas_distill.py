import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import inference_mas as base
from .lcb_utils import (
    clean_raw_output,
    evaluate_generated_code,
    extract_python_code,
    is_code_eval_dataset,
    is_mbppplus_dataset,
)
from prompts import (
    DISTILL_EXPERT_SLOT,
    DISTILL_FEEDBACK_SLOT,
    build_distill_expert_prompt,
    build_distill_expert_prompt_with_feedback_slot,
    build_distill_learner_prompt,
    build_distill_learner_prompt_with_slot,
)


def infer_distill_task(dataset_name: str, is_code_eval: bool) -> str:
    if is_code_eval:
        return "code"
    if base.is_choice_dataset(dataset_name):
        return "choice"
    return "math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mas_shape", type=str, default="distill", choices=["distill"])
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=-1)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_recursive_rounds", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--expert_model_name_or_path", type=str, required=True)
    parser.add_argument("--learner_model_name_or_path", type=str, required=True)

    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--expert_inner_aligner_path", type=str, default="")
    parser.add_argument("--learner_inner_aligner_path", type=str, default="")
    parser.add_argument("--outer_el_path", type=str, default=None)
    parser.add_argument("--outer_le_path", type=str, default=None)

    parser.add_argument(
        "--inner_adapter_type_fallback",
        type=str,
        default="ln_res_adapter",
        choices=["ln_res_adapter"],
    )
    parser.add_argument(
        "--outer_adapter_type_fallback",
        type=str,
        default="outer_ln_res_adapter",
        choices=["outer_ln_res_adapter"],
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--min_p", type=float, default=-1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--ans", action="store_true")
    parser.add_argument("--ans_max_new_tokens", type=int, default=-1)

    parser.add_argument("--lcb_use_private_tests", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lcb_timeout_s", type=int, default=6)
    parser.add_argument("--mbppplus_timeout_s", type=int, default=10)
    parser.add_argument("--mbppplus_num_prompt_tests", type=int, default=3)
    parser.add_argument("--mbppplus_subset", type=str, default="")
    parser.add_argument("--mbppplus_cache_dir", type=str, default="")

    parser.add_argument("--dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--outer_dtype", type=str, default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    parser.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    parser.add_argument("--result_jsonl", type=str, default="")
    return parser.parse_args()


def resolve_distill_outer_paths(
    outer_el_path: Optional[str],
    outer_le_path: Optional[str],
) -> Tuple[str, str]:
    if outer_el_path is None or outer_le_path is None:
        raise ValueError("Please provide both --outer_el_path and --outer_le_path.")
    return outer_el_path, outer_le_path


def resolve_distill_outer_types(
    outer_cfg_path: Optional[str],
    fallback_type: str,
) -> Tuple[str, str]:
    el_type = fallback_type
    le_type = fallback_type
    if outer_cfg_path and os.path.isfile(outer_cfg_path):
        with open(outer_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        el_type = cfg.get("outer_el_type", el_type)
        le_type = cfg.get("outer_le_type", le_type)
    return el_type, le_type


def _outer_out_dim(path: str) -> int:
    state = torch.load(path, map_location="cpu")
    out_dim = int(state["proj2.bias"].shape[0])
    del state
    return out_dim


def build_distill_expert_prompt_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
    feedback_text: Optional[str] = None,
) -> str:
    is_code_eval = mas_task == "code"
    task_type = task_types[sample_idx] if (is_code_eval and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    if feedback_text is None:
        return build_distill_expert_prompt(
            question,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )
    return build_distill_expert_prompt_with_feedback_slot(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    ).replace(DISTILL_FEEDBACK_SLOT, feedback_text)


def build_distill_learner_prompt_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
    expert_plan_text: str,
) -> str:
    is_code_eval = mas_task == "code"
    task_type = task_types[sample_idx] if (is_code_eval and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    return build_distill_learner_prompt(
        question,
        expert_plan=expert_plan_text,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def build_distill_learner_prompt_with_slot_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
) -> str:
    is_code_eval = mas_task == "code"
    task_type = task_types[sample_idx] if (is_code_eval and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    return build_distill_learner_prompt_with_slot(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def run_distill_expert_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    inner_aligner_path: str,
    outer_path: str,
    outer_type: str,
    latent_steps: int,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    outer_dtype: torch.dtype,
    trust_remote_code: bool,
    inner_adapter_type_fallback: str,
    enable_thinking: bool,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
    feedback_latents: Optional[Sequence[torch.Tensor]] = None,
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = _outer_out_dim(outer_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="distill_expert",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    hidden_size = embed_layer.weight.size(-1)

    inner = base.load_inner_adapter_module(
        adapter_path=inner_aligner_path,
        hidden_size=hidden_size,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=inner_adapter_type_fallback,
    )
    outer = base.load_outer_adapter_module(
        adapter_path=outer_path,
        in_dim=hidden_size,
        out_dim=_outer_out_dim(outer_path),
        adapter_type=outer_type,
        device=device,
        dtype=outer_dtype,
    )

    use_feedback = feedback_latents is not None
    if use_feedback and len(feedback_latents) != len(questions):
        raise ValueError("feedback_latents size mismatch.")

    prompt_payloads = []
    for idx, question in enumerate(questions):
        if use_feedback:
            user_prompt = build_distill_expert_prompt_with_feedback_slot(
                question,
                mas_task=mas_task,
                task_type=(task_types[idx] if (mas_task == "code" and task_types is not None) else "complete"),
                fn_name=(fn_names[idx] if fn_names is not None else None),
            )
            prompt_payloads.append(
                base.split_prompt_ids_by_slots(
                    tokenizer,
                    user_prompt,
                    [DISTILL_FEEDBACK_SLOT],
                    enable_thinking,
                )
            )
        else:
            prompt_payloads.append(
                base.render_chat_prompt_ids(
                    tokenizer,
                    build_distill_expert_prompt_text(question, idx, mas_task, task_types, fn_names),
                    enable_thinking,
                )
            )

    outputs: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="distill_expert_latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            if use_feedback:
                seg_prefix, seg_suffix = prompt_payloads[idx]
                prefix_embeds = base.token_ids_to_embeds(embed_layer, seg_prefix, device=device, dtype=embed_dtype)
                suffix_embeds = base.token_ids_to_embeds(embed_layer, seg_suffix, device=device, dtype=embed_dtype)
                feedback_embed = feedback_latents[idx].to(device=device, dtype=embed_dtype)
                seq = torch.cat([prefix_embeds, feedback_embed, suffix_embeds], dim=0)
            else:
                seq = embed_layer(torch.tensor(prompt_payloads[idx], dtype=torch.long, device=device).unsqueeze(0))[0]
                if seq.dtype != embed_dtype:
                    seq = seq.to(embed_dtype)
            embed_seqs.append(seq)

        batch_embeds, attention_mask = base.pad_left_embeds(embed_seqs, device=device)
        hidden_rollout = base.autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        self_latent = base.run_inner_adapter(inner, hidden_rollout, output_dtype=embed_dtype)
        mapped = base.run_outer_adapter(outer, self_latent, output_dtype=torch.float32)
        for i in range(mapped.size(0)):
            outputs.append(mapped[i].detach().cpu())

    base.release_resources(model, tokenizer, inner, outer)
    return outputs


def run_distill_learner_text_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    expert_latents: Sequence[torch.Tensor],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_thinking: bool,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
) -> List[str]:
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="distill_learner",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype

    prompt_segments = [
        base.split_prompt_ids_by_slots(
            tokenizer,
            build_distill_learner_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names),
            [DISTILL_EXPERT_SLOT],
            enable_thinking,
        )
        for idx, question in enumerate(questions)
    ]

    gen_kwargs = base.build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    outputs: List[str] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="distill_learner_text_from_latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            prefix = base.token_ids_to_embeds(embed_layer, seg_prefix, device=device, dtype=embed_dtype)
            suffix = base.token_ids_to_embeds(embed_layer, seg_suffix, device=device, dtype=embed_dtype)
            seq = torch.cat(
                [
                    prefix,
                    expert_latents[idx].to(device=device, dtype=embed_dtype),
                    suffix,
                ],
                dim=0,
            )
            embed_seqs.append(seq)

        batch_embeds, attention_mask = base.pad_left_embeds(embed_seqs, device=device)
        with torch.no_grad():
            generated = model.generate(
                inputs_embeds=batch_embeds,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = attention_mask.size(1)
        gen_ids = sequences[:, prompt_len:] if sequences.size(1) > max_new_tokens else sequences
        batch_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend([text.strip() for text in batch_texts])

    base.release_resources(model, tokenizer)
    return outputs


def run_distill_learner_feedback_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    expert_latents: Sequence[torch.Tensor],
    inner_aligner_path: str,
    outer_path: str,
    outer_type: str,
    latent_steps: int,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    outer_dtype: torch.dtype,
    trust_remote_code: bool,
    inner_adapter_type_fallback: str,
    enable_thinking: bool,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = _outer_out_dim(outer_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="distill_learner_feedback",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    hidden_size = embed_layer.weight.size(-1)
    inner = base.load_inner_adapter_module(
        adapter_path=inner_aligner_path,
        hidden_size=hidden_size,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=inner_adapter_type_fallback,
    )
    outer = base.load_outer_adapter_module(
        adapter_path=outer_path,
        in_dim=hidden_size,
        out_dim=_outer_out_dim(outer_path),
        adapter_type=outer_type,
        device=device,
        dtype=outer_dtype,
    )

    prompt_segments = [
        base.split_prompt_ids_by_slots(
            tokenizer,
            build_distill_learner_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names),
            [DISTILL_EXPERT_SLOT],
            enable_thinking,
        )
        for idx, question in enumerate(questions)
    ]

    outputs: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="distill_learner_feedback_latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            seq = torch.cat(
                [
                    base.token_ids_to_embeds(embed_layer, seg_prefix, device=device, dtype=embed_dtype),
                    expert_latents[idx].to(device=device, dtype=embed_dtype),
                    base.token_ids_to_embeds(embed_layer, seg_suffix, device=device, dtype=embed_dtype),
                ],
                dim=0,
            )
            embed_seqs.append(seq)

        batch_embeds, attention_mask = base.pad_left_embeds(embed_seqs, device=device)
        hidden_rollout = base.autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        learner_self = base.run_inner_adapter(inner, hidden_rollout, output_dtype=embed_dtype)
        mapped = base.run_outer_adapter(outer, learner_self, output_dtype=torch.float32)
        for i in range(mapped.size(0)):
            outputs.append(mapped[i].detach().cpu())

    base.release_resources(model, tokenizer, inner, outer)
    return outputs


def main() -> None:
    args = parse_args()
    args.method = "ours_recursive"
    base._GEN_TOP_K = int(args.top_k) if int(args.top_k) >= 0 else None
    base._GEN_MIN_P = float(args.min_p) if float(args.min_p) >= 0 else None
    base._GEN_REPETITION_PENALTY = float(args.repetition_penalty)

    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive.")
    if args.num_recursive_rounds <= 0:
        raise ValueError("--num_recursive_rounds must be positive.")
    if args.presence_penalty != 0.0:
        print("[warn] --presence_penalty is ignored by HF generation in this pipeline.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = base.resolve_dtype(args.dtype)
    outer_dtype = base.resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")
    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        outer_dtype = torch.float32

    trust_remote_code = bool(args.trust_remote_code)
    enable_thinking = bool(args.enable_thinking)

    outer_el_path, outer_le_path = resolve_distill_outer_paths(
        args.outer_el_path,
        args.outer_le_path,
    )
    outer_el_type = outer_le_type = args.outer_adapter_type_fallback

    dataset_name, questions, gold_answers, sample_metadata = base.load_eval_questions_and_answers(
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
        return_metadata=True,
        lcb_use_private_tests=bool(args.lcb_use_private_tests),
        mbppplus_subset=args.mbppplus_subset,
        mbppplus_cache_dir=args.mbppplus_cache_dir,
        mbppplus_num_prompt_tests=int(args.mbppplus_num_prompt_tests),
    )

    is_code_eval = is_code_eval_dataset(dataset_name)
    code_eval_timeout_s = int(args.mbppplus_timeout_s) if is_mbppplus_dataset(dataset_name) else int(args.lcb_timeout_s)
    task_types: Optional[List[str]] = None
    fn_names: Optional[List[Optional[str]]] = None
    if is_code_eval:
        if sample_metadata is None or len(sample_metadata) != len(questions):
            raise RuntimeError("Missing code-eval sample metadata.")
        task_types = [str(meta.get("task_type", "complete")) for meta in sample_metadata]
        fn_names = [meta.get("fn_name") if isinstance(meta, dict) else None for meta in sample_metadata]

    mas_task = infer_distill_task(dataset_name, is_code_eval)
    print(
        f"Running method=ours_recursive on {len(questions)} samples "
        f"(expert={args.expert_model_name_or_path}, learner={args.learner_model_name_or_path}, mas_shape={args.mas_shape})"
    )
    if args.num_rollouts > 1:
        print(f"[rollout] num_rollouts={args.num_rollouts} (pass@{args.num_rollouts})")
        if not args.do_sample:
            print("[warn] --num_rollouts > 1 but --do_sample is disabled; outputs may be identical.")

    base_sample_seed = args.sample_seed if args.sample_seed >= 0 else args.seed

    def set_rollout_seed(rollout_idx: int) -> int:
        rollout_seed = int(base_sample_seed + rollout_idx)
        torch.manual_seed(rollout_seed)
        random.seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        return rollout_seed

    first_rollout_seed = set_rollout_seed(0)
    expert_inputs_for_log: List[str] = []
    learner_input_for_log: List[str] = []

    if args.method == "text":
        expert_prompts = [
            build_distill_expert_prompt_text(question, idx, mas_task, task_types, fn_names)
            for idx, question in enumerate(questions)
        ]
        expert_outputs, expert_inputs_for_log = base.run_text_generation_stage(
            stage_name="distill_expert",
            model_name_or_path=args.expert_model_name_or_path,
            user_prompts=expert_prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
        )
        learner_prompts = [
            build_distill_learner_prompt_text(question, idx, mas_task, task_types, fn_names, expert_outputs[idx])
            for idx, question in enumerate(questions)
        ]
        final_outputs, learner_input_for_log = base.run_text_generation_stage(
            stage_name="distill_learner",
            model_name_or_path=args.learner_model_name_or_path,
            user_prompts=learner_prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
        )
        final_expert_latents = None
    else:
        current_feedbacks: Optional[List[torch.Tensor]] = None
        final_expert_latents: Optional[List[torch.Tensor]] = None
        final_outputs: List[str] = []
        for round_idx in range(args.num_recursive_rounds):
            expert_inputs_for_log = [
                build_distill_expert_prompt_text(
                    question,
                    idx,
                    mas_task,
                    task_types,
                    fn_names,
                    feedback_text=(None if current_feedbacks is None else DISTILL_FEEDBACK_SLOT),
                )
                for idx, question in enumerate(questions)
            ]
            expert_latents = run_distill_expert_latent_stage(
                model_name_or_path=args.expert_model_name_or_path,
                questions=questions,
                inner_aligner_path=args.expert_inner_aligner_path,
                outer_path=outer_el_path,
                outer_type=outer_el_type,
                latent_steps=args.latent_steps,
                batch_size=args.batch_size,
                device=device,
                model_dtype=model_dtype,
                outer_dtype=outer_dtype,
                trust_remote_code=trust_remote_code,
                inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                enable_thinking=enable_thinking,
                mas_task=mas_task,
                task_types=task_types,
                fn_names=fn_names,
                feedback_latents=current_feedbacks,
            )
            final_expert_latents = expert_latents
            learner_input_for_log = [
                build_distill_learner_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names)
                for idx, question in enumerate(questions)
            ]
            if round_idx == args.num_recursive_rounds - 1:
                final_outputs = run_distill_learner_text_stage(
                    model_name_or_path=args.learner_model_name_or_path,
                    questions=questions,
                    expert_latents=expert_latents,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                    mas_task=mas_task,
                    task_types=task_types,
                    fn_names=fn_names,
                )
            else:
                current_feedbacks = run_distill_learner_feedback_latent_stage(
                    model_name_or_path=args.learner_model_name_or_path,
                    questions=questions,
                    expert_latents=expert_latents,
                    inner_aligner_path=args.learner_inner_aligner_path,
                    outer_path=outer_le_path,
                    outer_type=outer_le_type,
                    latent_steps=args.latent_steps,
                    batch_size=args.batch_size,
                    device=device,
                    model_dtype=model_dtype,
                    outer_dtype=outer_dtype,
                    trust_remote_code=trust_remote_code,
                    inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                    enable_thinking=enable_thinking,
                    mas_task=mas_task,
                    task_types=task_types,
                    fn_names=fn_names,
                )

    if args.ans:
        ans_retry_max_new_tokens = int(args.ans_max_new_tokens)
        if ans_retry_max_new_tokens < 0:
            ans_retry_max_new_tokens = 256 if is_code_eval else 16
        final_outputs, retried = base.run_answer_retry_stage(
            model_name_or_path=args.learner_model_name_or_path,
            outputs=final_outputs,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=ans_retry_max_new_tokens,
        )
        if retried > 0:
            retry_target = "code block" if is_code_eval else "boxed/choice answer"
            print(f"[ans] retried {retried} outputs for missing {retry_target}.")

    rollout_seeds = [first_rollout_seed]
    outputs_by_rollout: List[List[str]] = [list(final_outputs)]
    if args.num_rollouts > 1:
        if args.method == "ours_recursive" and final_expert_latents is None:
            raise RuntimeError("Missing final expert latents for multi-rollout generation.")
        for rollout_idx in range(1, args.num_rollouts):
            rollout_seed = set_rollout_seed(rollout_idx) if args.do_sample else int(base_sample_seed + rollout_idx)
            rollout_seeds.append(rollout_seed)
            if not args.do_sample:
                outputs_by_rollout.append(list(final_outputs))
                continue
            if args.method == "text":
                expert_prompts = [
                    build_distill_expert_prompt_text(question, idx, mas_task, task_types, fn_names)
                    for idx, question in enumerate(questions)
                ]
                expert_outputs, _ = base.run_text_generation_stage(
                    stage_name="distill_expert",
                    model_name_or_path=args.expert_model_name_or_path,
                    user_prompts=expert_prompts,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )
                learner_prompts = [
                    build_distill_learner_prompt_text(question, idx, mas_task, task_types, fn_names, expert_outputs[idx])
                    for idx, question in enumerate(questions)
                ]
                rollout_outputs, _ = base.run_text_generation_stage(
                    stage_name="distill_learner",
                    model_name_or_path=args.learner_model_name_or_path,
                    user_prompts=learner_prompts,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )
            else:
                rollout_outputs = run_distill_learner_text_stage(
                    model_name_or_path=args.learner_model_name_or_path,
                    questions=questions,
                    expert_latents=final_expert_latents,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                    mas_task=mas_task,
                    task_types=task_types,
                    fn_names=fn_names,
                )
            if args.ans:
                rollout_outputs, _ = base.run_answer_retry_stage(
                    model_name_or_path=args.learner_model_name_or_path,
                    outputs=rollout_outputs,
                    dataset_name=dataset_name,
                    batch_size=args.batch_size,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=ans_retry_max_new_tokens,
                )
            outputs_by_rollout.append(rollout_outputs)

    total = len(questions)
    result_jsonl_path = args.result_jsonl.strip()
    sample_records: List[Dict[str, object]] = []
    rollout_eval_math: List[List[Tuple[str, Optional[str], bool, str, str]]] = []
    rollout_eval_code: List[List[Dict[str, Any]]] = []
    rollout_correct_counts: List[int] = []

    for rollout_idx, outputs in enumerate(outputs_by_rollout):
        correct_count = 0
        if is_code_eval:
            if sample_metadata is None:
                raise RuntimeError("Missing code eval sample metadata.")
            eval_rows_code: List[Dict[str, Any]] = []
            eval_start_time = time.time()
            for i in range(total):
                cleaned_output = clean_raw_output(outputs[i])
                parsed_code = extract_python_code(cleaned_output)
                eval_result = evaluate_generated_code(
                    parsed_code,
                    sample_metadata[i].get("eval_sample", {}),
                    timeout_s=code_eval_timeout_s,
                )
                is_correct = bool(eval_result.get("all_passed", False))
                if is_correct:
                    correct_count += 1
                eval_rows_code.append(
                    {
                        "parsed_code": parsed_code,
                        "parse_ok": bool(parsed_code),
                        "correct": is_correct,
                        "eval": eval_result,
                    }
                )
                if ((i + 1) % 10 == 0) or (i + 1 == total):
                    elapsed = time.time() - eval_start_time
                    print(
                        f"[code-eval][rollout {rollout_idx + 1}/{args.num_rollouts}] "
                        f"checked {i + 1}/{total} samples, correct={correct_count}, "
                        f"elapsed={elapsed:.1f}s"
                    )
            rollout_eval_code.append(eval_rows_code)
        else:
            eval_rows_math: List[Tuple[str, Optional[str], bool, str, str]] = []
            for i in range(total):
                eval_row = base.compare_answers(gold_answers[i], outputs[i], dataset_name=dataset_name)
                if eval_row[2]:
                    correct_count += 1
                eval_rows_math.append(eval_row)
            rollout_eval_math.append(eval_rows_math)

        rollout_correct_counts.append(correct_count)
        rollout_acc = 100.0 * correct_count / total if total > 0 else 0.0
        if args.num_rollouts > 1:
            print(f"[rollout {rollout_idx + 1}/{args.num_rollouts}] accuracy={rollout_acc:.2f}% ({correct_count}/{total})")

    pass_correct_total = 0
    for i in range(total):
        if is_code_eval:
            if any(bool(rollout_eval_code[r][i]["correct"]) for r in range(args.num_rollouts)):
                pass_correct_total += 1
        else:
            if any(rollout_eval_math[r][i][2] for r in range(args.num_rollouts)):
                pass_correct_total += 1
    pass_at_k = 100.0 * pass_correct_total / total if total > 0 else 0.0

    method_name = f"{args.method}_distill"
    if result_jsonl_path:
        for i in range(total):
            if args.num_rollouts == 1:
                if is_code_eval:
                    row = rollout_eval_code[0][i]
                    sample_records.append(
                        {
                            "sample_idx": i,
                            "dataset": args.dataset,
                            "dataset_split": args.dataset_split,
                            "method": method_name,
                            "mas_shape": args.mas_shape,
                            "latent_steps": args.latent_steps,
                            "question": questions[i],
                            "gold_answer_raw": gold_answers[i],
                            "pred_code_parsed": row.get("parsed_code", ""),
                            "parse_ok": bool(row.get("parse_ok", False)),
                            "correct": bool(row.get("correct", False)),
                            "eval": row.get("eval", {}),
                            "expert_input": expert_inputs_for_log[i] if i < len(expert_inputs_for_log) else "",
                            "learner_input": learner_input_for_log[i] if i < len(learner_input_for_log) else "",
                            "raw_output": outputs_by_rollout[0][i],
                            "rollout_idx": 0,
                            "sample_seed": rollout_seeds[0],
                        }
                    )
                else:
                    gold_parsed, pred_parsed, is_correct, gold_norm, pred_norm = rollout_eval_math[0][i]
                    sample_records.append(
                        {
                            "sample_idx": i,
                            "dataset": args.dataset,
                            "dataset_split": args.dataset_split,
                            "method": method_name,
                            "mas_shape": args.mas_shape,
                            "latent_steps": args.latent_steps,
                            "question": questions[i],
                            "gold_answer_raw": gold_answers[i],
                            "gold_answer_parsed": gold_parsed,
                            "pred_answer_parsed": pred_parsed,
                            "correct": bool(is_correct),
                            "gold_norm": gold_norm,
                            "pred_norm": pred_norm,
                            "expert_input": expert_inputs_for_log[i] if i < len(expert_inputs_for_log) else "",
                            "learner_input": learner_input_for_log[i] if i < len(learner_input_for_log) else "",
                            "raw_output": outputs_by_rollout[0][i],
                            "rollout_idx": 0,
                            "sample_seed": rollout_seeds[0],
                        }
                    )
            else:
                rollout_records = []
                for r in range(args.num_rollouts):
                    if is_code_eval:
                        row = rollout_eval_code[r][i]
                        rollout_records.append(
                            {
                                "rollout_idx": r,
                                "sample_seed": rollout_seeds[r],
                                "pred_code_parsed": row.get("parsed_code", ""),
                                "parse_ok": bool(row.get("parse_ok", False)),
                                "correct": bool(row.get("correct", False)),
                                "eval": row.get("eval", {}),
                            }
                        )
                    else:
                        gold_parsed, pred_parsed, is_correct, gold_norm, pred_norm = rollout_eval_math[r][i]
                        rollout_records.append(
                            {
                                "rollout_idx": r,
                                "sample_seed": rollout_seeds[r],
                                "pred_answer_parsed": pred_parsed,
                                "correct": bool(is_correct),
                                "gold_norm": gold_norm,
                                "pred_norm": pred_norm,
                            }
                        )
                row_obj: Dict[str, Any] = {
                    "sample_idx": i,
                    "dataset": args.dataset,
                    "dataset_split": args.dataset_split,
                    "method": method_name,
                    "mas_shape": args.mas_shape,
                    "latent_steps": args.latent_steps,
                    "question": questions[i],
                    "gold_answer_raw": gold_answers[i],
                    "pass_at_k_correct": any(rec["correct"] for rec in rollout_records),
                    "rollouts": rollout_records,
                }
                if is_code_eval:
                    row_obj["pred_code_parsed"] = rollout_records[0].get("pred_code_parsed", "")
                else:
                    row_obj["gold_answer_parsed"] = rollout_eval_math[0][i][0]
                sample_records.append(row_obj)

    if args.num_rollouts == 1:
        accuracy = 100.0 * rollout_correct_counts[0] / total if total > 0 else 0.0
        print("=" * 100)
        print("Overall Accuracy")
        print(f"accuracy={accuracy:.2f}% ({rollout_correct_counts[0]}/{total})")
        print("=" * 100)
    else:
        accuracy = pass_at_k
        print("=" * 100)
        print(f"pass@{args.num_rollouts}")
        print(f"pass@{args.num_rollouts}={pass_at_k:.2f}% ({pass_correct_total}/{total})")
        print("=" * 100)

    if result_jsonl_path:
        out_dir = os.path.dirname(result_jsonl_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(result_jsonl_path, "w", encoding="utf-8") as f:
            for record in sample_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary_record = {
                "type": "summary",
                "dataset": args.dataset,
                "dataset_split": args.dataset_split,
                "method": method_name,
                "mas_shape": args.mas_shape,
                "latent_steps": args.latent_steps,
                "num_samples": total,
                "num_rollouts": args.num_rollouts,
                "sample_seed_base": base_sample_seed,
                "per_rollout_num_correct": rollout_correct_counts,
                "per_rollout_accuracy": [(100.0 * n / total if total > 0 else 0.0) for n in rollout_correct_counts],
                "num_correct": pass_correct_total if args.num_rollouts > 1 else rollout_correct_counts[0],
                "accuracy": accuracy,
                "pass_at_k": pass_at_k,
            }
            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
        print(f"[jsonl] wrote {len(sample_records)} sample records to {result_jsonl_path}")


if __name__ == "__main__":
    main()
