import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from . import inference_mas as base
from .lcb_utils import clean_raw_output, evaluate_generated_code, extract_python_code, is_code_eval_dataset, is_mbppplus_dataset
from prompts import (
    HIE_CODE_EXPERT_SLOT,
    HIE_FEEDBACK_SLOT,
    HIE_MATH_EXPERT_SLOT,
    HIE_SCIENCE_EXPERT_SLOT,
    build_hie_expert_prompt,
    build_hie_expert_prompt_with_feedback_slot,
    build_hie_summarizer_prompt_with_slots,
)


HIE_EXPERT_ROLES = ("hie_math_expert", "hie_code_expert", "hie_science_expert")
HIE_SLOT_TEXTS = (HIE_MATH_EXPERT_SLOT, HIE_CODE_EXPERT_SLOT, HIE_SCIENCE_EXPERT_SLOT)


def infer_hie_task(dataset_name: str, is_code_eval: bool) -> str:
    if is_code_eval:
        return "code"
    if base.is_choice_dataset(dataset_name):
        return "choice"
    return "math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mas_shape", type=str, default="hie", choices=["hie"])
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=-1)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_recursive_rounds", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--agent1_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent2_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent3_model_name_or_path", type=str, required=True)
    parser.add_argument("--agent4_model_name_or_path", type=str, required=True)

    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--agent1_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent2_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent3_inner_aligner_path", type=str, required=True)
    parser.add_argument("--agent4_inner_aligner_path", type=str, required=True)

    parser.add_argument("--outer_dir", type=str, default=None)
    parser.add_argument("--outer_1s_path", type=str, default=None)
    parser.add_argument("--outer_2s_path", type=str, default=None)
    parser.add_argument("--outer_3s_path", type=str, default=None)
    parser.add_argument("--outer_s1_path", type=str, default=None)
    parser.add_argument("--outer_s2_path", type=str, default=None)
    parser.add_argument("--outer_s3_path", type=str, default=None)

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


def resolve_hie_outer_paths(
    outer_dir: Optional[str],
    outer_1s_path: Optional[str],
    outer_2s_path: Optional[str],
    outer_3s_path: Optional[str],
    outer_s1_path: Optional[str],
    outer_s2_path: Optional[str],
    outer_s3_path: Optional[str],
) -> Tuple[Dict[str, str], Optional[str]]:
    names = {
        "outer_1s": outer_1s_path,
        "outer_2s": outer_2s_path,
        "outer_3s": outer_3s_path,
        "outer_s1": outer_s1_path,
        "outer_s2": outer_s2_path,
        "outer_s3": outer_s3_path,
    }
    if outer_dir:
        for name in list(names.keys()):
            if names[name] is None:
                names[name] = os.path.join(outer_dir, f"{name}.pt")

    missing = [name for name, path in names.items() if path is None]
    if missing:
        raise ValueError(
            "For hierarchical inference, provide --outer_dir or all of "
            "--outer_1s_path/--outer_2s_path/--outer_3s_path/--outer_s1_path/--outer_s2_path/--outer_s3_path. "
            f"Missing: {missing}"
        )

    cfg_path = None
    if outer_dir:
        maybe_cfg = os.path.join(outer_dir, "outer_adapter_config.json")
        if os.path.isfile(maybe_cfg):
            cfg_path = maybe_cfg
    if cfg_path is None:
        maybe_cfg = os.path.join(os.path.dirname(next(iter(names.values()))), "outer_adapter_config.json")
        if os.path.isfile(maybe_cfg):
            cfg_path = maybe_cfg
    return names, cfg_path


def resolve_hie_outer_types(outer_cfg_path: Optional[str], fallback_type: str) -> Dict[str, str]:
    names = {
        "outer_1s": fallback_type,
        "outer_2s": fallback_type,
        "outer_3s": fallback_type,
        "outer_s1": fallback_type,
        "outer_s2": fallback_type,
        "outer_s3": fallback_type,
    }
    if outer_cfg_path and os.path.isfile(outer_cfg_path):
        with open(outer_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in list(names.keys()):
            names[key] = cfg.get(f"{key}_type", names[key])
    return names


def split_prompt_ids_by_hie_slots(
    tokenizer,
    user_prompt_with_slots: str,
    enable_thinking: bool,
) -> List[List[int]]:
    return base.split_prompt_ids_by_slots(
        tokenizer,
        user_prompt_with_slots,
        list(HIE_SLOT_TEXTS),
        enable_thinking,
    )


def build_hie_expert_prompt_text(
    question: str,
    role: str,
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
        return build_hie_expert_prompt(
            question,
            role,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )
    return build_hie_expert_prompt_with_feedback_slot(
        question,
        role,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    ).replace(HIE_FEEDBACK_SLOT, feedback_text)


def build_hie_summarizer_prompt_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
) -> str:
    is_code_eval = mas_task == "code"
    task_type = task_types[sample_idx] if (is_code_eval and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    return build_hie_summarizer_prompt_with_slots(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def _outer_out_dim(path: str) -> int:
    state = torch.load(path, map_location="cpu")
    out_dim = int(state["proj2.bias"].shape[0])
    del state
    return out_dim


def run_hie_expert_latent_stage(
    stage_name: str,
    model_name_or_path: str,
    questions: Sequence[str],
    role: str,
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
    is_code_eval = mas_task == "code"
    if latent_steps == 0:
        out_dim = _outer_out_dim(outer_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name=stage_name,
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
        raise ValueError(f"{stage_name}: feedback_latents size mismatch.")

    prompt_payloads = []
    for idx, question in enumerate(questions):
            if use_feedback:
                user_prompt = build_hie_expert_prompt_with_feedback_slot(
                    question,
                    role,
                    mas_task=mas_task,
                    task_type=(task_types[idx] if (is_code_eval and task_types is not None) else "complete"),
                    fn_name=(fn_names[idx] if fn_names is not None else None),
                )
                prompt_payloads.append(
                    base.split_prompt_ids_by_slots(
                        tokenizer,
                        user_prompt,
                        [HIE_FEEDBACK_SLOT],
                        enable_thinking,
                    )
                )
            else:
                user_prompt = build_hie_expert_prompt_text(
                    question, role, idx, mas_task, task_types, fn_names, feedback_text=None
                )
                prompt_payloads.append(base.render_chat_prompt_ids(tokenizer, user_prompt, enable_thinking))

    outputs: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc=stage_name,
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


def run_hie_summarizer_text_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    expert1_latents: Sequence[torch.Tensor],
    expert2_latents: Sequence[torch.Tensor],
    expert3_latents: Sequence[torch.Tensor],
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
    is_code_eval = mas_task == "code"
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="hie_summarizer",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    hidden_size = embed_layer.weight.size(-1)

    for name, latents in (
        ("expert1", expert1_latents),
        ("expert2", expert2_latents),
        ("expert3", expert3_latents),
    ):
        if latents and latents[0].size(-1) != hidden_size:
            raise RuntimeError(f"{name} -> summarizer latent dim mismatch: {latents[0].size(-1)} vs {hidden_size}")

    prompt_segments = [
        split_prompt_ids_by_hie_slots(
            tokenizer,
            build_hie_summarizer_prompt_text(question, idx, mas_task, task_types, fn_names),
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
        desc="hie_summarizer text-from-latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_a, seg_b, seg_c, seg_d = prompt_segments[idx]
            part_a = base.token_ids_to_embeds(embed_layer, seg_a, device=device, dtype=embed_dtype)
            part_b = base.token_ids_to_embeds(embed_layer, seg_b, device=device, dtype=embed_dtype)
            part_c = base.token_ids_to_embeds(embed_layer, seg_c, device=device, dtype=embed_dtype)
            part_d = base.token_ids_to_embeds(embed_layer, seg_d, device=device, dtype=embed_dtype)
            seq = torch.cat(
                [
                    part_a,
                    expert1_latents[idx].to(device=device, dtype=embed_dtype),
                    part_b,
                    expert2_latents[idx].to(device=device, dtype=embed_dtype),
                    part_c,
                    expert3_latents[idx].to(device=device, dtype=embed_dtype),
                    part_d,
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


def run_hie_summarizer_feedback_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    expert1_latents: Sequence[torch.Tensor],
    expert2_latents: Sequence[torch.Tensor],
    expert3_latents: Sequence[torch.Tensor],
    inner_aligner_path: str,
    outer_back_paths: Dict[str, str],
    outer_back_types: Dict[str, str],
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
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    is_code_eval = mas_task == "code"
    if latent_steps == 0:
        out1 = _outer_out_dim(outer_back_paths["outer_s1"])
        out2 = _outer_out_dim(outer_back_paths["outer_s2"])
        out3 = _outer_out_dim(outer_back_paths["outer_s3"])
        return (
            [torch.empty((0, out1), dtype=torch.float32) for _ in questions],
            [torch.empty((0, out2), dtype=torch.float32) for _ in questions],
            [torch.empty((0, out3), dtype=torch.float32) for _ in questions],
        )

    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="hie_summarizer_feedback",
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
    outer_s1 = base.load_outer_adapter_module(
        adapter_path=outer_back_paths["outer_s1"],
        in_dim=hidden_size,
        out_dim=_outer_out_dim(outer_back_paths["outer_s1"]),
        adapter_type=outer_back_types["outer_s1"],
        device=device,
        dtype=outer_dtype,
    )
    outer_s2 = base.load_outer_adapter_module(
        adapter_path=outer_back_paths["outer_s2"],
        in_dim=hidden_size,
        out_dim=_outer_out_dim(outer_back_paths["outer_s2"]),
        adapter_type=outer_back_types["outer_s2"],
        device=device,
        dtype=outer_dtype,
    )
    outer_s3 = base.load_outer_adapter_module(
        adapter_path=outer_back_paths["outer_s3"],
        in_dim=hidden_size,
        out_dim=_outer_out_dim(outer_back_paths["outer_s3"]),
        adapter_type=outer_back_types["outer_s3"],
        device=device,
        dtype=outer_dtype,
    )

    prompt_segments = [
        split_prompt_ids_by_hie_slots(
            tokenizer,
            build_hie_summarizer_prompt_text(question, idx, mas_task, task_types, fn_names),
            enable_thinking,
        )
        for idx, question in enumerate(questions)
    ]

    outputs1: List[torch.Tensor] = []
    outputs2: List[torch.Tensor] = []
    outputs3: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="hie_summarizer feedback latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_a, seg_b, seg_c, seg_d = prompt_segments[idx]
            seq = torch.cat(
                [
                    base.token_ids_to_embeds(embed_layer, seg_a, device=device, dtype=embed_dtype),
                    expert1_latents[idx].to(device=device, dtype=embed_dtype),
                    base.token_ids_to_embeds(embed_layer, seg_b, device=device, dtype=embed_dtype),
                    expert2_latents[idx].to(device=device, dtype=embed_dtype),
                    base.token_ids_to_embeds(embed_layer, seg_c, device=device, dtype=embed_dtype),
                    expert3_latents[idx].to(device=device, dtype=embed_dtype),
                    base.token_ids_to_embeds(embed_layer, seg_d, device=device, dtype=embed_dtype),
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
        summarizer_self = base.run_inner_adapter(inner, hidden_rollout, output_dtype=embed_dtype)
        map1 = base.run_outer_adapter(outer_s1, summarizer_self, output_dtype=torch.float32)
        map2 = base.run_outer_adapter(outer_s2, summarizer_self, output_dtype=torch.float32)
        map3 = base.run_outer_adapter(outer_s3, summarizer_self, output_dtype=torch.float32)
        for i in range(map1.size(0)):
            outputs1.append(map1[i].detach().cpu())
            outputs2.append(map2[i].detach().cpu())
            outputs3.append(map3[i].detach().cpu())

    base.release_resources(model, tokenizer, inner, outer_s1, outer_s2, outer_s3)
    return outputs1, outputs2, outputs3


def main() -> None:
    args = parse_args()
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

    outer_paths, outer_cfg = resolve_hie_outer_paths(
        args.outer_dir,
        args.outer_1s_path,
        args.outer_2s_path,
        args.outer_3s_path,
        args.outer_s1_path,
        args.outer_s2_path,
        args.outer_s3_path,
    )
    outer_types = resolve_hie_outer_types(outer_cfg, args.outer_adapter_type_fallback)

    dataset_name, questions, gold_answers, sample_metadata = base.load_eval_questions_and_answers(
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        shuffle=bool(args.shuffle),
        seed=args.seed,
        return_metadata=True,
        lcb_use_private_tests=bool(args.lcb_use_private_tests),
        mbppplus_subset=args.mbppplus_subset,
        mbppplus_cache_dir=args.mbppplus_cache_dir,
        mbppplus_num_prompt_tests=int(args.mbppplus_num_prompt_tests),
    )

    is_code_eval = is_code_eval_dataset(dataset_name)
    mas_task = infer_hie_task(dataset_name, is_code_eval)
    code_eval_timeout_s = int(args.mbppplus_timeout_s) if is_mbppplus_dataset(dataset_name) else int(args.lcb_timeout_s)
    task_types: Optional[List[str]] = None
    fn_names: Optional[List[Optional[str]]] = None
    if is_code_eval:
        if sample_metadata is None or len(sample_metadata) != len(questions):
            raise RuntimeError("Missing code-eval sample metadata.")
        task_types = [str(meta.get("task_type", "complete")) for meta in sample_metadata]
        fn_names = [meta.get("fn_name") if isinstance(meta, dict) else None for meta in sample_metadata]

    print(
        f"Running hierarchical inference on {len(questions)} samples "
        f"(expert1={args.agent1_model_name_or_path}, expert2={args.agent2_model_name_or_path}, "
        f"expert3={args.agent3_model_name_or_path}, summarizer={args.agent4_model_name_or_path}, "
        f"mas_shape={args.mas_shape}, rounds={args.num_recursive_rounds})"
    )
    if args.num_rollouts > 1:
        print(f"[rollout] num_rollouts={args.num_rollouts} (pass@{args.num_rollouts})")
        if not args.do_sample:
            print("[warn] --num_rollouts > 1 but --do_sample is disabled; outputs may be identical.")

    base_sample_seed = args.sample_seed if args.sample_seed >= 0 else args.seed

    def set_rollout_seed(rollout_idx: int) -> int:
        rollout_seed = int(base_sample_seed + rollout_idx)
        random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        return rollout_seed

    if args.do_sample:
        first_rollout_seed = set_rollout_seed(0)
        print(f"[rollout 1/{args.num_rollouts}] sample_seed={first_rollout_seed}")
    else:
        first_rollout_seed = int(base_sample_seed)

    current_feedbacks: List[Optional[List[torch.Tensor]]] = [None, None, None]
    final_expert_latents: Optional[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]] = None
    final_outputs: List[str] = []
    expert_inputs_for_log: Dict[str, List[str]] = {}
    summarizer_input_for_log: List[str] = []

    for round_idx in range(args.num_recursive_rounds):
        rid = round_idx + 1
        print(f"[round {rid}/{args.num_recursive_rounds}]")
        expert_latents = []
        for expert_idx, role in enumerate(HIE_EXPERT_ROLES):
            model_name = getattr(args, f"agent{expert_idx + 1}_model_name_or_path")
            inner_path = getattr(args, f"agent{expert_idx + 1}_inner_aligner_path")
            out_key = f"outer_{expert_idx + 1}s"
            feedback_latents = current_feedbacks[expert_idx]

            if feedback_latents is None:
                expert_inputs_for_log[f"{role}_r{rid}"] = [
                    build_hie_expert_prompt_text(q, role, i, mas_task, task_types, fn_names)
                    for i, q in enumerate(questions)
                ]
            else:
                expert_inputs_for_log[f"{role}_r{rid}"] = [
                    build_hie_expert_prompt_with_feedback_slot(
                        q,
                        role,
                        mas_task=mas_task,
                        task_type=(task_types[i] if (is_code_eval and task_types is not None) else "complete"),
                        fn_name=(fn_names[i] if fn_names is not None else None),
                    )
                    for i, q in enumerate(questions)
                ]

            latents = run_hie_expert_latent_stage(
                stage_name=f"{role}-latent-r{rid}",
                model_name_or_path=model_name,
                questions=questions,
                role=role,
                inner_aligner_path=inner_path,
                outer_path=outer_paths[out_key],
                outer_type=outer_types[out_key],
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
                feedback_latents=feedback_latents,
            )
            expert_latents.append(latents)

        final_expert_latents = (expert_latents[0], expert_latents[1], expert_latents[2])
        summarizer_input_for_log = [
            build_hie_summarizer_prompt_text(q, i, mas_task, task_types, fn_names)
            for i, q in enumerate(questions)
        ]
        final_outputs = run_hie_summarizer_text_stage(
            model_name_or_path=args.agent4_model_name_or_path,
            questions=questions,
            expert1_latents=expert_latents[0],
            expert2_latents=expert_latents[1],
            expert3_latents=expert_latents[2],
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

        if round_idx < args.num_recursive_rounds - 1:
            fb1, fb2, fb3 = run_hie_summarizer_feedback_latent_stage(
                model_name_or_path=args.agent4_model_name_or_path,
                questions=questions,
                expert1_latents=expert_latents[0],
                expert2_latents=expert_latents[1],
                expert3_latents=expert_latents[2],
                inner_aligner_path=args.agent4_inner_aligner_path,
                outer_back_paths=outer_paths,
                outer_back_types=outer_types,
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
            current_feedbacks = [fb1, fb2, fb3]

    if args.ans:
        ans_retry_max_new_tokens = int(args.ans_max_new_tokens)
        if ans_retry_max_new_tokens < 0:
            ans_retry_max_new_tokens = 256 if is_code_eval else 16
        final_outputs, retried = base.run_answer_retry_stage(
            model_name_or_path=args.agent4_model_name_or_path,
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
        if final_expert_latents is None:
            raise RuntimeError("Missing final hierarchical latents for multi-rollout generation.")
        e1, e2, e3 = final_expert_latents
        for rollout_idx in range(1, args.num_rollouts):
            rollout_seed = set_rollout_seed(rollout_idx) if args.do_sample else int(base_sample_seed + rollout_idx)
            rollout_seeds.append(rollout_seed)
            if not args.do_sample:
                outputs_by_rollout.append(list(final_outputs))
                continue
            rollout_outputs = run_hie_summarizer_text_stage(
                model_name_or_path=args.agent4_model_name_or_path,
                questions=questions,
                expert1_latents=e1,
                expert2_latents=e2,
                expert3_latents=e3,
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
                    model_name_or_path=args.agent4_model_name_or_path,
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

    if args.num_rollouts == 1:
        print("=" * 120)
        print("Per-Sample Logs")
        print("=" * 120)
        print("[note] Hierarchical summarizer inputs show slot placeholders marking latent injection positions.")
        rendered_expert_logs = {}
        for key, prompts in expert_inputs_for_log.items():
            if not prompts:
                rendered_expert_logs[key] = []
                continue
            agent_name = (
                args.agent1_model_name_or_path if "math_expert" in key else
                args.agent2_model_name_or_path if "code_expert" in key else
                args.agent3_model_name_or_path
            )
            rendered_expert_logs[key] = base.render_inputs_for_logging(
                model_name_or_path=agent_name,
                user_prompts=prompts,
                trust_remote_code=trust_remote_code,
                agent_name=key,
                enable_thinking=enable_thinking,
            )
        summarizer_inputs_for_log = base.render_inputs_for_logging(
            model_name_or_path=args.agent4_model_name_or_path,
            user_prompts=summarizer_input_for_log,
            trust_remote_code=trust_remote_code,
            agent_name="hie_summarizer",
            enable_thinking=enable_thinking,
        )

        for i in range(total):
            if is_code_eval:
                code_eval = rollout_eval_code[0][i]
                parsed_code = str(code_eval.get("parsed_code", ""))
                eval_result = dict(code_eval.get("eval", {}))
                is_correct = bool(code_eval.get("correct", False))
                if result_jsonl_path:
                    sample_records.append(
                        {
                            "sample_idx": i,
                            "dataset": args.dataset,
                            "dataset_split": args.dataset_split,
                            "method": "ours_hie_recursive",
                            "mas_shape": args.mas_shape,
                            "latent_steps": args.latent_steps,
                            "question": questions[i],
                            "gold_answer_raw": gold_answers[i],
                            "pred_code_parsed": parsed_code,
                            "parse_ok": bool(code_eval.get("parse_ok", False)),
                            "correct": is_correct,
                            "eval": eval_result,
                            "rollout_idx": 0,
                            "sample_seed": rollout_seeds[0],
                        }
                    )
                print("=" * 120)
                print(f"Sample {i + 1}/{total}")
                print("-" * 120)
                print("1) Question:")
                print(questions[i])
                round_keys = sorted(rendered_expert_logs.keys())
                sec_idx = 2
                for key in round_keys:
                    print(f"\n{sec_idx}) {key} Input (full chat template):")
                    print(rendered_expert_logs[key][i])
                    sec_idx += 1
                print(f"\n{sec_idx}) Summarizer Input (full chat template):")
                print(summarizer_inputs_for_log[i])
                print(f"\n{sec_idx + 1}) Summarizer Output:")
                print(outputs_by_rollout[0][i])
                print(f"\n{sec_idx + 2}) Parsed python code:")
                print(parsed_code if parsed_code else "<NO_VALID_CODE_BLOCK>")
                print(f"\n{sec_idx + 3}) Code evaluation:")
                print(json.dumps(eval_result, ensure_ascii=False))
            else:
                gold_parsed, pred_parsed, is_correct, gold_norm, pred_norm = rollout_eval_math[0][i]
                if result_jsonl_path:
                    sample_records.append(
                        {
                            "sample_idx": i,
                            "dataset": args.dataset,
                            "dataset_split": args.dataset_split,
                            "method": "ours_hie_recursive",
                            "mas_shape": args.mas_shape,
                            "latent_steps": args.latent_steps,
                            "question": questions[i],
                            "gold_answer_raw": gold_answers[i],
                            "gold_answer_parsed": gold_parsed,
                            "pred_answer_parsed": pred_parsed,
                            "correct": bool(is_correct),
                            "gold_norm": gold_norm,
                            "pred_norm": pred_norm,
                            "rollout_idx": 0,
                            "sample_seed": rollout_seeds[0],
                        }
                    )
                print("=" * 120)
                print(f"Sample {i + 1}/{total}")
                print("-" * 120)
                print("1) Question:")
                print(questions[i])
                print("\n2) Gold answer:")
                print(gold_parsed)
                round_keys = sorted(rendered_expert_logs.keys())
                sec_idx = 3
                for key in round_keys:
                    print(f"\n{sec_idx}) {key} Input (full chat template):")
                    print(rendered_expert_logs[key][i])
                    sec_idx += 1
                print(f"\n{sec_idx}) Summarizer Input (full chat template):")
                print(summarizer_inputs_for_log[i])
                print(f"\n{sec_idx + 1}) Summarizer Output:")
                print(outputs_by_rollout[0][i])
                print(f"\n{sec_idx + 2}) Parsed answer:")
                print(pred_parsed if pred_parsed is not None else "<NOT_FOUND>")
                print(f"\n{sec_idx + 3}) Compare normalized pure answers:")
                print(
                    f"correct={is_correct} "
                    f"(gold_norm={gold_norm if gold_norm else '<EMPTY>'}, "
                    f"pred_norm={pred_norm if pred_norm else '<EMPTY>'})"
                )
    else:
        if result_jsonl_path:
            for i in range(total):
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
                    "method": "ours_hie_recursive",
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
        print("=" * 120)
        print("Overall Accuracy")
        print(f"accuracy={accuracy:.2f}% ({rollout_correct_counts[0]}/{total})")
        print("=" * 120)
    else:
        accuracy = pass_at_k
        print("=" * 120)
        print(f"pass@{args.num_rollouts}")
        print(f"pass@{args.num_rollouts}={pass_at_k:.2f}% ({pass_correct_total}/{total})")
        print("=" * 120)

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
                "method": "ours_hie_recursive",
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
