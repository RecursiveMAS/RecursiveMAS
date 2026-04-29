# inference_mas.py

import argparse
from collections.abc import Mapping
import gc
import json
import hashlib
import importlib.util
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Work around broken torchvision installs in text-only eval envs.
_ORIG_FIND_SPEC = importlib.util.find_spec


def _patched_find_spec(name: str, *args, **kwargs):
    if name == "torchvision" or name.startswith("torchvision."):
        return None
    return _ORIG_FIND_SPEC(name, *args, **kwargs)


if os.environ.get("MAS_FORCE_DISABLE_TORCHVISION", "1") == "1":
    importlib.util.find_spec = _patched_find_spec

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .answer_utils import (
    compare_answers,
    ensure_choice_instruction,
    extract_boxed_answer,
    extract_choice_answer,
    format_latent_info,
    is_gemma_model_name,
    is_choice_dataset,
    is_gpqa_dataset,
    is_medqa_dataset,
    medqa_gold_to_choice,
    soften_planner_format_instruction,
    strip_choice_instruction_lines,
    truncate_text_chars,
)
from prompts import (
    FEEDBACK_SLOT,
    PLANNER_SLOT,
    REFINED_SLOT,
    SYSTEM_PROMPT,
    build_code_planner_prompt,
    build_code_planner_prompt_with_feedback_slot,
    build_code_refiner_prompt,
    build_code_refiner_prompt_with_slot,
    build_code_solver_prompt,
    build_code_solver_prompt_with_slots,
    build_math_planner_prompt,
    build_math_planner_prompt_with_feedback_slot,
    build_math_refiner_prompt,
    build_math_refiner_prompt_with_slot,
    build_math_solver_prompt,
    build_math_solver_prompt_with_slots,
)
from .lcb_utils import (
    build_code_reparse_suffix,
    build_mbppplus_sample_meta,
    clean_raw_output,
    evaluate_generated_code,
    extract_python_code,
    is_code_eval_dataset,
    is_mbppplus_dataset,
    load_mbppplus_records,
)

from modeling import (
    Adapter,
    CrossModelAdapter,
    infer_inner_adapter_type_from_state_dict,
    infer_outer_adapter_type_from_state_dict,
    normalize_inner_adapter_type,
    normalize_outer_adapter_type,
    resolve_local_pretrained_path,
)

_CHAT_TEMPLATE_IDS_FALLBACK_WARNED = False
_GEN_TOP_K: Optional[int] = None
_GEN_MIN_P: Optional[float] = None
_GEN_REPETITION_PENALTY: float = 1.0

RELEASE_RECOMMENDED_SETTINGS: Dict[Tuple[str, str], Dict[str, int]] = {
    ("sequential_light", "math500"): {"seed": 42, "batch_size": 32, "latent_length": 48},
    ("sequential_light", "medqa"): {"seed": 42, "batch_size": 16, "latent_length": 32},
    ("sequential_light", "gpqa"): {"seed": 42, "batch_size": 16, "latent_length": 32},
    ("sequential_light", "mbppplus"): {"seed": 42, "batch_size": 16, "latent_length": 16},
    ("sequential_scaled", "math500"): {"seed": 42, "batch_size": 16, "latent_length": 32},
    ("sequential_scaled", "medqa"): {"seed": 42, "batch_size": 16, "latent_length": 48},
    ("sequential_scaled", "gpqa"): {"seed": 42, "batch_size": 16, "latent_length": 48},
    ("sequential_scaled", "mbppplus"): {"seed": 42, "batch_size": 16, "latent_length": 16},
}


def get_release_recommended_settings(style: str, dataset: str) -> Optional[Dict[str, int]]:
    return RELEASE_RECOMMENDED_SETTINGS.get((str(style).lower(), str(dataset).lower()))


def resolve_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "auto":
        return "auto"
    return None


def resolve_dataset(name: str) -> Tuple[str, Optional[str]]:
    key = name.strip().lower()
    if key in {"gsm8k", "openai/gsm8k"}:
        return "openai/gsm8k", "main"
    if key in {"math500", "math-500", "huggingfaceh4/math-500"}:
        return "HuggingFaceH4/MATH-500", None
    if is_medqa_dataset(key):
        return "__local_medqa__", None
    if is_gpqa_dataset(key):
        return "Idavidrein/gpqa", "gpqa_diamond"
    if is_mbppplus_dataset(key):
        return "__mbppplus__", None
    return name, None


def _stable_shuffle_indices(n: int, seed_text: str) -> List[int]:
    order = list(range(n))
    seed_hex = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    seed_int = int(seed_hex[:16], 16)
    rng = random.Random(seed_int)
    rng.shuffle(order)
    return order


def _sample_first_nonempty_text(sample: Mapping, keys: Sequence[str]) -> str:
    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _build_gpqa_question_and_choice(
    sample: Mapping,
    seed: int,
    shuffle_options: bool = True,
) -> Tuple[str, str]:
    # GPQA-diamond uses "Question" in some dumps; keep both cases for robustness.
    stem = _sample_first_nonempty_text(sample, ("Question", "question", "query", "prompt"))
    correct = _sample_first_nonempty_text(sample, ("Correct Answer", "correct_answer"))
    wrong1 = _sample_first_nonempty_text(sample, ("Incorrect Answer 1", "incorrect_answer_1"))
    wrong2 = _sample_first_nonempty_text(sample, ("Incorrect Answer 2", "incorrect_answer_2"))
    wrong3 = _sample_first_nonempty_text(sample, ("Incorrect Answer 3", "incorrect_answer_3"))

    options = [correct, wrong1, wrong2, wrong3]
    if shuffle_options:
        seed_key = stem if stem else "||".join(options)
        perm = _stable_shuffle_indices(4, f"{seed}::{seed_key}")
    else:
        perm = list(range(4))
    labels = ["A", "B", "C", "D"]
    labeled_lines = []
    gold_choice = "A"
    for pos, idx in enumerate(perm):
        label = labels[pos]
        opt = options[idx]
        if idx == 0:
            gold_choice = label
        labeled_lines.append(f"{label}. {opt}")

    question = ensure_choice_instruction(f"{stem}\n" + "\n".join(labeled_lines))
    return question, gold_choice


def _load_medqa_records(path: str) -> List[Mapping]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"medqa json must be a list. got type={type(data)}")
    return [x for x in data if isinstance(x, Mapping)]


def load_eval_questions_and_answers(
    dataset: str,
    dataset_split: str,
    num_samples: int,
    shuffle: bool,
    seed: int,
    gpqa_shuffle_options: bool = True,
    return_metadata: bool = False,
    lcb_use_private_tests: bool = False,
    mbppplus_subset: str = "",
    mbppplus_cache_dir: str = "",
    mbppplus_num_prompt_tests: int = 3,
):
    dataset_name, dataset_config = resolve_dataset(dataset)
    key = dataset.strip().lower()
    if dataset_name == dataset and os.path.isfile(dataset) and dataset.lower().endswith(".json"):
        dataset_name = "__local_medqa__"

    sample_metadata: Optional[List[Dict[str, Any]]] = None

    if dataset_name == "__mbppplus__":
        records = load_mbppplus_records(
            split=dataset_split,
            subset=(mbppplus_subset or None),
            cache_dir=(mbppplus_cache_dir or None),
        )
        if len(records) == 0:
            raise ValueError("Loaded MBPP+ records are empty.")

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(records)
        if num_samples > 0:
            records = records[: min(num_samples, len(records))]

        questions: List[str] = []
        gold_answers: List[str] = []
        sample_metadata = []
        for row in records:
            meta = build_mbppplus_sample_meta(
                row,
                max_prompt_tests=int(mbppplus_num_prompt_tests),
            )
            questions.append(str(meta["question"]))
            gold_answers.append(str(meta.get("gold_answer", "")))
            sample_metadata.append(meta)

        if return_metadata:
            return "mbppplus", questions, gold_answers, sample_metadata
        return "mbppplus", questions, gold_answers

    if dataset_name == "__local_medqa__":
        medqa_path = dataset
        if not os.path.isfile(medqa_path):
            medqa_path = "dataset/medqa.json"
        if not os.path.isfile(medqa_path):
            raise FileNotFoundError(
                f"medqa file not found. expected '{dataset}' or 'dataset/medqa.json'."
            )

        records = _load_medqa_records(medqa_path)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(records)
        if num_samples > 0:
            records = records[: min(num_samples, len(records))]

        questions = []
        gold_answers = []
        for sample in records:
            base_q = str(sample.get("query") or sample.get("question") or "").strip()
            options = sample.get("options")
            if isinstance(options, list) and options:
                has_lettered = bool(
                    re.search(r"(?mi)^\s*[A-Da-d]\s*[\.\):\-]\s+", base_q)
                )
                if not has_lettered:
                    base_q = base_q.rstrip() + "\n" + "\n".join(str(x) for x in options)

            questions.append(ensure_choice_instruction(base_q))
            gold_answers.append(medqa_gold_to_choice(sample, default_choice="A"))

        if return_metadata:
            return "medqa", questions, gold_answers, None
        return "medqa", questions, gold_answers

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if len(ds) == 0:
        raise ValueError("Loaded dataset is empty.")
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    if is_gpqa_dataset(key):
        questions = []
        gold_answers = []
        for sample in ds:
            q, ans = _build_gpqa_question_and_choice(
                sample,
                seed=seed,
                shuffle_options=gpqa_shuffle_options,
            )
            questions.append(q)
            gold_answers.append(ans)
        if return_metadata:
            return "gpqa_diamond", questions, gold_answers, None
        return "gpqa_diamond", questions, gold_answers

    question_column = None
    for candidate in ("question", "problem"):
        if candidate in ds.column_names:
            question_column = candidate
            break
    if question_column is None:
        raise ValueError(f"Dataset missing question/problem column: {ds.column_names}")
    if "answer" not in ds.column_names:
        raise ValueError(f"Dataset missing answer column: {ds.column_names}")

    questions = [sample.get(question_column, "") for sample in ds]
    gold_answers = [sample.get("answer", "") for sample in ds]
    if return_metadata:
        return dataset_name, questions, gold_answers, None
    return dataset_name, questions, gold_answers


def ensure_chat_template(tokenizer, agent_name: str) -> None:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            f"{agent_name} tokenizer does not implement apply_chat_template. "
            "MAS inference requires chat template and does not support fallback."
        )


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

    def call_template(template_messages: List[Dict[str, str]]):
        try:
            return tokenizer.apply_chat_template(template_messages, **kwargs)
        except TypeError as exc:
            # Older tokenizers may not expose/accept enable_thinking.
            if "enable_thinking" not in str(exc):
                raise
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(template_messages, **fallback_kwargs)

    try:
        return call_template(messages)
    except Exception as exc:
        if "Conversation roles must alternate" not in str(exc):
            raise
        system_parts: List[str] = []
        merged_messages: List[Dict[str, str]] = []
        inserted_system = False
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role == "user" and system_parts and not inserted_system:
                merged_messages.append(
                    {
                        "role": "user",
                        "content": "\n\n".join(system_parts + [content]),
                    }
                )
                inserted_system = True
            else:
                merged_messages.append(dict(message))
        if system_parts and not inserted_system:
            merged_messages.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})
        return call_template(merged_messages)


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
        if isinstance(value[0], list):
            # Batched output like [ [ids...] ].
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
                "[warn] apply_chat_template(tokenize=True) returned str in inference; "
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
        if isinstance(value[0], list):
            # Batched output like [ [ids...] ].
            return _normalize_template_ids(tokenizer, value[0], max_length=max_length)
        if isinstance(value[0], str):
            return _normalize_template_ids(tokenizer, "".join(value), max_length=max_length)
        return [int(x) for x in value]

    raise ValueError(f"Unsupported chat template output type for ids: {type(value)}")


def render_chat_prompt(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return _normalize_template_text(tokenizer, rendered)


def render_chat_prompt_ids(tokenizer, user_prompt: str, enable_thinking: bool) -> List[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    rendered = apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return _normalize_template_ids(tokenizer, rendered)


def split_prompt_ids_by_slots(
    tokenizer,
    user_prompt_with_slots: str,
    slot_texts: Sequence[str],
    enable_thinking: bool,
) -> List[List[int]]:
    full_text = render_chat_prompt(tokenizer, user_prompt_with_slots, enable_thinking)
    segments_text: List[str] = []
    cursor = 0
    for slot_text in slot_texts:
        pos = full_text.find(slot_text, cursor)
        if pos < 0:
            raise RuntimeError(
                f"Failed to locate slot {slot_text!r} in rendered chat text. "
                "Please verify tokenizer and prompt formatting."
            )
        segments_text.append(full_text[cursor:pos])
        cursor = pos + len(slot_text)
    segments_text.append(full_text[cursor:])

    segments: List[List[int]] = []
    for text_segment in segments_text:
        if text_segment:
            seg_ids = tokenizer(text_segment, add_special_tokens=False)["input_ids"]
            segments.append(list(seg_ids))
        else:
            segments.append([])
    return segments


def batch_iter_indices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def pad_left_ids(
    all_ids: Sequence[Sequence[int]],
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not all_ids:
        raise ValueError("all_ids is empty")
    max_len = max(len(ids) for ids in all_ids)
    if max_len == 0:
        raise ValueError("Encountered empty prompt after truncation.")
    bs = len(all_ids)
    input_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(all_ids):
        if not ids:
            raise ValueError("Found empty token sequence; cannot build valid attention mask.")
        seq = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, max_len - len(ids) :] = seq
        attention_mask[i, max_len - len(ids) :] = 1
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("Invalid padded token batch: at least one sample has all-zero attention mask.")
    return input_ids, attention_mask


def pad_left_embeds(
    embed_seqs: Sequence[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not embed_seqs:
        raise ValueError("embed_seqs is empty")
    hidden_size = embed_seqs[0].size(-1)
    dtype = embed_seqs[0].dtype
    max_len = max(seq.size(0) for seq in embed_seqs)
    if max_len == 0:
        raise ValueError("Encountered empty embedding sequence.")
    bs = len(embed_seqs)
    batch_embeds = torch.zeros((bs, max_len, hidden_size), dtype=dtype, device=device)
    attention_mask = torch.zeros((bs, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(embed_seqs):
        if seq.size(0) == 0:
            raise ValueError("Found empty embedding sequence; cannot build valid attention mask.")
        seq = seq.to(device=device, dtype=dtype)
        length = seq.size(0)
        batch_embeds[i, max_len - length :, :] = seq
        attention_mask[i, max_len - length :] = 1
    if not torch.all(attention_mask.sum(dim=1) > 0):
        raise ValueError("Invalid padded embed batch: at least one sample has all-zero attention mask.")
    return batch_embeds, attention_mask


def token_ids_to_embeds(
    embed_layer,
    token_ids: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden_size = embed_layer.weight.size(-1)
    if not token_ids:
        return torch.empty((0, hidden_size), device=device, dtype=dtype)
    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    embeds = embed_layer(token_tensor)[0]
    if embeds.dtype != dtype:
        embeds = embeds.to(dtype)
    return embeds


def build_generation_kwargs(
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, object]:
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    kwargs: Dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
        if _GEN_TOP_K is not None:
            kwargs["top_k"] = int(_GEN_TOP_K)
        if _GEN_MIN_P is not None:
            kwargs["min_p"] = float(_GEN_MIN_P)
    # Repetition penalty is supported by HF generate and can be used for both sampled/greedy decoding.
    if _GEN_REPETITION_PENALTY is not None and float(_GEN_REPETITION_PENALTY) > 0:
        kwargs["repetition_penalty"] = float(_GEN_REPETITION_PENALTY)
    return kwargs


def load_agent_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool,
    agent_name: str,
):
    resolved_path = resolve_local_pretrained_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    ensure_chat_template(tokenizer, agent_name)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(
                f"{agent_name} tokenizer has no pad token and no eos token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_agent_model_and_tokenizer(
    model_name_or_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    agent_name: str,
):
    resolved_path = resolve_local_pretrained_path(model_name_or_path)
    tokenizer = load_agent_tokenizer(
        model_name_or_path=resolved_path,
        trust_remote_code=trust_remote_code,
        agent_name=agent_name,
    )

    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        torch_dtype=(dtype if dtype != "auto" else "auto"),
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def release_resources(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_inner_adapter_files(adapter_path: str) -> Tuple[str, str]:
    if os.path.isdir(adapter_path):
        state_path = os.path.join(adapter_path, "adapter.pt")
        config_path = os.path.join(adapter_path, "adapter_config.json")
    else:
        state_path = adapter_path
        config_path = os.path.join(os.path.dirname(adapter_path), "adapter_config.json")
    return state_path, config_path


def load_inner_adapter_module(
    adapter_path: str,
    hidden_size: int,
    device: torch.device,
    dtype,
    fallback_adapter_type: str,
) -> Adapter:
    state_path, config_path = resolve_inner_adapter_files(adapter_path)
    if not os.path.isfile(state_path):
        raise FileNotFoundError(f"Inner aligner weights not found: {state_path}")

    adapter_type = normalize_inner_adapter_type(fallback_adapter_type)
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    state_dict = torch.load(state_path, map_location="cpu")
    adapter_type = infer_inner_adapter_type_from_state_dict(
        state_dict,
        fallback=cfg.get("adapter_type", adapter_type) if os.path.isfile(config_path) else adapter_type,
    )
    adapter = Adapter(hidden_size=hidden_size, adapter_type=adapter_type)
    adapter.load_state_dict(state_dict, strict=True)
    resolved_dtype = dtype
    if dtype == "auto":
        resolved_dtype = next((v.dtype for v in state_dict.values() if torch.is_tensor(v) and v.is_floating_point()), torch.float32)
    adapter.to(device=device, dtype=resolved_dtype)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False
    return adapter


def resolve_recursive_outer_paths(
    outer_12_path: Optional[str],
    outer_23_path: Optional[str],
    outer_31_path: Optional[str],
) -> Tuple[str, str, str]:
    if outer_12_path is None or outer_23_path is None or outer_31_path is None:
        raise ValueError(
            "Please provide all of --outer_12_path/--outer_23_path/--outer_31_path."
        )
    return outer_12_path, outer_23_path, outer_31_path


def resolve_outer_types(
    outer_cfg_path: Optional[str],
    fallback_type: str,
) -> Tuple[str, str]:
    o12_type = fallback_type
    o23_type = fallback_type
    if outer_cfg_path and os.path.isfile(outer_cfg_path):
        with open(outer_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        o12_type = cfg.get("outer_12_type", o12_type)
        o23_type = cfg.get("outer_23_type", o23_type)
    return o12_type, o23_type


def resolve_recursive_outer_types(
    outer_cfg_path: Optional[str],
    fallback_type: str,
) -> Tuple[str, str, str]:
    o12_type = fallback_type
    o23_type = fallback_type
    o31_type = fallback_type
    if outer_cfg_path and os.path.isfile(outer_cfg_path):
        with open(outer_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        o12_type = cfg.get("outer_12_type", o12_type)
        o23_type = cfg.get("outer_23_type", o23_type)
        o31_type = cfg.get("outer_31_type", o31_type)
    return o12_type, o23_type, o31_type


def load_outer_adapter_module(
    adapter_path: str,
    in_dim: int,
    out_dim: int,
    adapter_type: str,
    device: torch.device,
    dtype,
) -> CrossModelAdapter:
    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"Outer aligner weights not found: {adapter_path}")

    state_dict = torch.load(adapter_path, map_location="cpu")
    adapter_type = infer_outer_adapter_type_from_state_dict(
        state_dict,
        fallback=normalize_outer_adapter_type(adapter_type),
    )
    adapter = CrossModelAdapter(in_dim=in_dim, out_dim=out_dim, adapter_type=adapter_type)
    adapter.load_state_dict(state_dict, strict=True)
    resolved_dtype = dtype
    if dtype == "auto":
        resolved_dtype = next((v.dtype for v in state_dict.values() if torch.is_tensor(v) and v.is_floating_point()), torch.float32)
    adapter.to(device=device, dtype=resolved_dtype)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad = False
    return adapter


def infer_outer_adapter_out_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    for key in (
        "proj2.bias",
        "proj1.bias",
        "residual_proj.bias",
        "ln_target.weight",
    ):
        tensor = state_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    for key in ("proj2.weight", "proj1.weight", "residual_proj.weight"):
        tensor = state_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    raise KeyError(
        "Cannot infer outer adapter output dim from state_dict keys: "
        f"{sorted(state_dict.keys())}"
    )


def infer_outer_adapter_out_dim_from_file(adapter_path: str) -> int:
    state_dict = torch.load(adapter_path, map_location="cpu")
    try:
        return infer_outer_adapter_out_dim(state_dict)
    finally:
        del state_dict


def run_inner_adapter(
    adapter: Adapter,
    hidden_states: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else hidden_states.dtype
    x = hidden_states
    if x.dtype != adapter_dtype:
        x = x.to(adapter_dtype)
    out = adapter(x)
    if out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out


def run_outer_adapter(
    adapter: CrossModelAdapter,
    hidden_states: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    adapter_param = next(adapter.parameters(), None)
    adapter_dtype = adapter_param.dtype if adapter_param is not None else hidden_states.dtype
    x = hidden_states
    if x.dtype != adapter_dtype:
        x = x.to(adapter_dtype)
    out = adapter(x)
    if out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out


@torch.no_grad()
def autoregressive_latent_rollout(
    model,
    rollout_inner_adapter: Adapter,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
) -> torch.Tensor:
    if latent_steps <= 0:
        raise ValueError("latent_steps must be positive for latent rollout.")
    hidden_states: List[torch.Tensor] = []
    for _ in range(latent_steps):
        # Keep only last-token logits to avoid full vocab logits OOM on long prompts.
        forward_kwargs = {
            "inputs_embeds": input_embeds,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }
        try:
            outputs = model(logits_to_keep=1, **forward_kwargs)
        except TypeError:
            outputs = model(**forward_kwargs)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        hidden_states.append(last_hidden.unsqueeze(1))

        next_embed = run_inner_adapter(
            rollout_inner_adapter,
            last_hidden,
            output_dtype=input_embeds.dtype,
        ).unsqueeze(1)
        input_embeds = torch.cat([input_embeds, next_embed], dim=1)

        next_mask = torch.ones(
            (attention_mask.size(0), 1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    return torch.cat(hidden_states, dim=1)


def run_text_generation_stage(
    stage_name: str,
    model_name_or_path: str,
    user_prompts: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_thinking: bool,
) -> Tuple[List[str], List[str]]:
    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name=stage_name,
    )
    rendered_prompts = [render_chat_prompt(tokenizer, p, enable_thinking) for p in user_prompts]
    gen_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    outputs: List[str] = []
    total_batches = (len(rendered_prompts) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(rendered_prompts), batch_size),
        total=total_batches,
        desc=f"{stage_name} text",
    ):
        batch_prompts = rendered_prompts[start:end]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                **gen_kwargs,
            )

        prompt_len = batch_inputs["input_ids"].size(1)
        gen_ids = generated[:, prompt_len:]
        batch_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend([text.strip() for text in batch_texts])

    release_resources(model, tokenizer)
    return outputs, rendered_prompts


def run_planner_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    agent1_inner_aligner_path: str,
    outer_12_path: str,
    outer_12_type: str,
    latent_steps: int,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    outer_dtype: torch.dtype,
    trust_remote_code: bool,
    inner_adapter_type_fallback: str,
    enable_thinking: bool,
    task_types: Optional[Sequence[str]] = None,
    fn_names: Optional[Sequence[Optional[str]]] = None,
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = infer_outer_adapter_out_dim_from_file(outer_12_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="planner",
    )
    embed_layer = model.get_input_embeddings()
    planner_embed_dtype = embed_layer.weight.dtype
    planner_hidden = embed_layer.weight.size(-1)

    inner_1 = load_inner_adapter_module(
        adapter_path=agent1_inner_aligner_path,
        hidden_size=planner_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=inner_adapter_type_fallback,
    )

    probe_out_dim = infer_outer_adapter_out_dim_from_file(outer_12_path)
    outer_12 = load_outer_adapter_module(
        adapter_path=outer_12_path,
        in_dim=planner_hidden,
        out_dim=probe_out_dim,
        adapter_type=outer_12_type,
        device=device,
        dtype=outer_dtype,
    )

    prompt_ids = []
    for idx, question in enumerate(questions):
        if task_types is not None:
            fn_name = fn_names[idx] if fn_names is not None else None
            user_prompt = build_code_planner_prompt(question, task_types[idx], fn_name=fn_name)
        else:
            user_prompt = build_math_planner_prompt(question)
        prompt_ids.append(render_chat_prompt_ids(tokenizer, user_prompt, enable_thinking))

    planner_to_refiner: List[torch.Tensor] = []
    total_batches = (len(prompt_ids) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(prompt_ids), batch_size),
        total=total_batches,
        desc="planner latent",
    ):
        batch_ids = prompt_ids[start:end]
        input_ids, attention_mask = pad_left_ids(
            batch_ids,
            pad_id=tokenizer.pad_token_id,
            device=device,
        )
        input_embeds = embed_layer(input_ids)

        hidden_rollout = autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner_1,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        planner_self = run_inner_adapter(inner_1, hidden_rollout, output_dtype=planner_embed_dtype)
        lat12 = run_outer_adapter(outer_12, planner_self, output_dtype=planner_embed_dtype)

        for i in range(lat12.size(0)):
            planner_to_refiner.append(lat12[i].detach().cpu())

    release_resources(model, tokenizer, inner_1, outer_12)
    return planner_to_refiner


def run_refiner_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    planner_latents: Sequence[torch.Tensor],
    agent2_inner_aligner_path: str,
    outer_23_path: str,
    outer_23_type: str,
    latent_steps: int,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    outer_dtype: torch.dtype,
    trust_remote_code: bool,
    inner_adapter_type_fallback: str,
    enable_thinking: bool,
    task_types: Optional[Sequence[str]] = None,
    fn_names: Optional[Sequence[Optional[str]]] = None,
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = infer_outer_adapter_out_dim_from_file(outer_23_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="refiner",
    )
    embed_layer = model.get_input_embeddings()
    refiner_embed_dtype = embed_layer.weight.dtype
    refiner_hidden = embed_layer.weight.size(-1)

    if planner_latents and planner_latents[0].size(-1) != refiner_hidden:
        raise RuntimeError(
            "Planner-to-refiner latent dim does not match refiner embedding dim: "
            f"{planner_latents[0].size(-1)} vs {refiner_hidden}"
        )

    inner_2 = load_inner_adapter_module(
        adapter_path=agent2_inner_aligner_path,
        hidden_size=refiner_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=inner_adapter_type_fallback,
    )

    probe_out_dim = infer_outer_adapter_out_dim_from_file(outer_23_path)
    outer_23 = load_outer_adapter_module(
        adapter_path=outer_23_path,
        in_dim=refiner_hidden,
        out_dim=probe_out_dim,
        adapter_type=outer_23_type,
        device=device,
        dtype=outer_dtype,
    )

    prompt_segments = []
    for idx, question in enumerate(questions):
        if task_types is not None:
            fn_name = fn_names[idx] if fn_names is not None else None
            user_prompt = build_code_refiner_prompt_with_slot(
                question,
                task_types[idx],
                fn_name=fn_name,
            )
        else:
            user_prompt = build_math_refiner_prompt_with_slot(question)
        prompt_segments.append(
            split_prompt_ids_by_slots(
                tokenizer,
                user_prompt,
                [PLANNER_SLOT],
                enable_thinking,
            )
        )

    refiner_to_solver: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="refiner latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            prefix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_prefix,
                device=device,
                dtype=refiner_embed_dtype,
            )
            suffix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_suffix,
                device=device,
                dtype=refiner_embed_dtype,
            )
            planner_embed = planner_latents[idx].to(device=device, dtype=refiner_embed_dtype)
            seq = torch.cat([prefix_embeds, planner_embed, suffix_embeds], dim=0)
            embed_seqs.append(seq)

        batch_embeds, attention_mask = pad_left_embeds(embed_seqs, device=device)
        hidden_rollout = autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner_2,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        refiner_self = run_inner_adapter(inner_2, hidden_rollout, output_dtype=refiner_embed_dtype)
        mapped = run_outer_adapter(outer_23, refiner_self, output_dtype=refiner_embed_dtype)
        for i in range(mapped.size(0)):
            refiner_to_solver.append(mapped[i].detach().cpu())

    release_resources(model, tokenizer, inner_2, outer_23)
    return refiner_to_solver


def run_solver_feedback_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    refiner_latents: Sequence[torch.Tensor],
    agent3_inner_aligner_path: str,
    outer_31_path: str,
    outer_31_type: str,
    latent_steps: int,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    outer_dtype: torch.dtype,
    trust_remote_code: bool,
    inner_adapter_type_fallback: str,
    enable_thinking: bool,
    args: argparse.Namespace,
    task_types: Optional[Sequence[str]] = None,
    fn_names: Optional[Sequence[Optional[str]]] = None,
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = infer_outer_adapter_out_dim_from_file(outer_31_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="solver-feedback",
    )
    embed_layer = model.get_input_embeddings()
    solver_embed_dtype = embed_layer.weight.dtype
    solver_hidden = embed_layer.weight.size(-1)

    if refiner_latents and refiner_latents[0].size(-1) != solver_hidden:
        raise RuntimeError(
            "Refiner-to-solver latent dim does not match solver embedding dim: "
            f"{refiner_latents[0].size(-1)} vs {solver_hidden}"
        )

    inner_3 = load_inner_adapter_module(
        adapter_path=agent3_inner_aligner_path,
        hidden_size=solver_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=inner_adapter_type_fallback,
    )

    probe_out_dim = infer_outer_adapter_out_dim_from_file(outer_31_path)
    outer_31 = load_outer_adapter_module(
        adapter_path=outer_31_path,
        in_dim=solver_hidden,
        out_dim=probe_out_dim,
        adapter_type=outer_31_type,
        device=device,
        dtype=outer_dtype,
    )

    prompt_segments = []
    for idx, question in enumerate(questions):
        if task_types is not None:
            fn_name = fn_names[idx] if fn_names is not None else None
            user_prompt = build_code_solver_prompt_with_slots(
                question,
                task_types[idx],
                args=args,
                mas_shape=args.mas_shape,
                fn_name=fn_name,
            )
        else:
            user_prompt = build_math_solver_prompt_with_slots(question, args, mas_shape=args.mas_shape)
        prompt_segments.append(
            split_prompt_ids_by_slots(
                tokenizer,
                user_prompt,
                [REFINED_SLOT],
                enable_thinking,
            )
        )

    feedback_latents: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="solver feedback latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            prefix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_prefix,
                device=device,
                dtype=solver_embed_dtype,
            )
            suffix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_suffix,
                device=device,
                dtype=solver_embed_dtype,
            )
            refiner_embed = refiner_latents[idx].to(device=device, dtype=solver_embed_dtype)
            seq = torch.cat([prefix_embeds, refiner_embed, suffix_embeds], dim=0)
            embed_seqs.append(seq)

        batch_embeds, attention_mask = pad_left_embeds(embed_seqs, device=device)
        hidden_rollout = autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner_3,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        solver_self = run_inner_adapter(inner_3, hidden_rollout, output_dtype=solver_embed_dtype)
        mapped_feedback = run_outer_adapter(outer_31, solver_self, output_dtype=torch.float32)
        for i in range(mapped_feedback.size(0)):
            feedback_latents.append(mapped_feedback[i].detach().cpu())

    release_resources(model, tokenizer, inner_3, outer_31)
    return feedback_latents


def run_planner_feedback_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    feedback_latents: Sequence[torch.Tensor],
    agent1_inner_aligner_path: str,
    outer_12_path: str,
    outer_12_type: str,
    latent_steps: int,
    batch_size: int,
    device: torch.device,
    model_dtype: torch.dtype,
    outer_dtype: torch.dtype,
    trust_remote_code: bool,
    inner_adapter_type_fallback: str,
    enable_thinking: bool,
    task_types: Optional[Sequence[str]] = None,
    fn_names: Optional[Sequence[Optional[str]]] = None,
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = infer_outer_adapter_out_dim_from_file(outer_12_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="planner-feedback",
    )
    embed_layer = model.get_input_embeddings()
    planner_embed_dtype = embed_layer.weight.dtype
    planner_hidden = embed_layer.weight.size(-1)

    if feedback_latents and feedback_latents[0].size(-1) != planner_hidden:
        raise RuntimeError(
            "Solver-feedback latent dim does not match planner embedding dim: "
            f"{feedback_latents[0].size(-1)} vs {planner_hidden}"
        )

    inner_1 = load_inner_adapter_module(
        adapter_path=agent1_inner_aligner_path,
        hidden_size=planner_hidden,
        device=device,
        dtype=model_dtype,
        fallback_adapter_type=inner_adapter_type_fallback,
    )

    probe_out_dim = infer_outer_adapter_out_dim_from_file(outer_12_path)
    outer_12 = load_outer_adapter_module(
        adapter_path=outer_12_path,
        in_dim=planner_hidden,
        out_dim=probe_out_dim,
        adapter_type=outer_12_type,
        device=device,
        dtype=outer_dtype,
    )

    prompt_segments = []
    for idx, question in enumerate(questions):
        if task_types is not None:
            fn_name = fn_names[idx] if fn_names is not None else None
            user_prompt = build_code_planner_prompt_with_feedback_slot(
                question,
                task_types[idx],
                fn_name=fn_name,
            )
        else:
            user_prompt = build_math_planner_prompt_with_feedback_slot(question)
        prompt_segments.append(
            split_prompt_ids_by_slots(
                tokenizer,
                user_prompt,
                [FEEDBACK_SLOT],
                enable_thinking,
            )
        )

    planner_to_refiner: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="planner feedback latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            prefix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_prefix,
                device=device,
                dtype=planner_embed_dtype,
            )
            suffix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_suffix,
                device=device,
                dtype=planner_embed_dtype,
            )
            feedback_embed = feedback_latents[idx].to(device=device, dtype=planner_embed_dtype)
            seq = torch.cat([prefix_embeds, feedback_embed, suffix_embeds], dim=0)
            embed_seqs.append(seq)

        batch_embeds, attention_mask = pad_left_embeds(embed_seqs, device=device)
        hidden_rollout = autoregressive_latent_rollout(
            model=model,
            rollout_inner_adapter=inner_1,
            input_embeds=batch_embeds,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
        )
        planner_self = run_inner_adapter(inner_1, hidden_rollout, output_dtype=planner_embed_dtype)
        lat12 = run_outer_adapter(outer_12, planner_self, output_dtype=planner_embed_dtype)
        for i in range(lat12.size(0)):
            planner_to_refiner.append(lat12[i].detach().cpu())

    release_resources(model, tokenizer, inner_1, outer_12)
    return planner_to_refiner


def run_solver_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    refiner_latents: Sequence[torch.Tensor],
    args: argparse.Namespace,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    enable_thinking: bool,
    task_types: Optional[Sequence[str]] = None,
    fn_names: Optional[Sequence[Optional[str]]] = None,
) -> List[str]:
    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="solver",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype
    hidden_size = embed_layer.weight.size(-1)

    if refiner_latents and refiner_latents[0].size(-1) != hidden_size:
        raise RuntimeError(
            "Refiner-to-solver latent dim does not match solver embedding dim: "
            f"{refiner_latents[0].size(-1)} vs {hidden_size}"
        )

    prompt_segments = []
    for idx, question in enumerate(questions):
        if task_types is not None:
            fn_name = fn_names[idx] if fn_names is not None else None
            user_prompt = build_code_solver_prompt_with_slots(
                question,
                task_types[idx],
                args=args,
                mas_shape=args.mas_shape,
                fn_name=fn_name,
            )
        else:
            user_prompt = build_math_solver_prompt_with_slots(question, args, mas_shape=args.mas_shape)
        prompt_segments.append(
            split_prompt_ids_by_slots(
                tokenizer,
                user_prompt,
                [REFINED_SLOT],
                enable_thinking,
            )
        )

    gen_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    outputs: List[str] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="solver text-from-latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            prefix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_prefix,
                device=device,
                dtype=embed_dtype,
            )
            suffix_embeds = token_ids_to_embeds(
                embed_layer,
                seg_suffix,
                device=device,
                dtype=embed_dtype,
            )
            refiner_embed = refiner_latents[idx].to(device=device, dtype=embed_dtype)
            seq = torch.cat(
                [
                    prefix_embeds,
                    refiner_embed,
                    suffix_embeds,
                ],
                dim=0,
            )
            embed_seqs.append(seq)

        batch_embeds, attention_mask = pad_left_embeds(embed_seqs, device=device)
        with torch.no_grad():
            generated = model.generate(
                inputs_embeds=batch_embeds,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_len = attention_mask.size(1)
        # `inputs_embeds` generation return format differs across model families:
        # some return continuation-only, others return prompt+continuation.
        # Use max_new_tokens as a robust discriminator to avoid truncating outputs.
        if sequences.size(1) > max_new_tokens:
            gen_ids = sequences[:, prompt_len:]
        else:
            gen_ids = sequences
        batch_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend([text.strip() for text in batch_texts])

    release_resources(model, tokenizer)
    return outputs


def run_answer_retry_stage(
    model_name_or_path: str,
    outputs: Sequence[str],
    dataset_name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int = 16,
) -> Tuple[List[str], int]:
    is_code_eval = is_code_eval_dataset(dataset_name)
    if is_code_eval:
        pending_indices = [
            i for i, text in enumerate(outputs)
            if not extract_python_code(clean_raw_output(text))
        ]
        retry_suffix = build_code_reparse_suffix()
    elif is_choice_dataset(dataset_name):
        pending_indices = [
            i for i, text in enumerate(outputs) if extract_choice_answer(text, default=None) is None
        ]
        retry_suffix = "Final Choice: \\boxed{"
    else:
        pending_indices = [i for i, text in enumerate(outputs) if extract_boxed_answer(text) is None]
        retry_suffix = "Final Answer: \\boxed{"

    if not pending_indices:
        return list(outputs), 0

    retry_max_new_tokens = int(max_new_tokens)
    if is_code_eval and retry_max_new_tokens < 256:
        retry_max_new_tokens = 256

    model, tokenizer = load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="solver-retry",
    )
    gen_kwargs = build_generation_kwargs(
        tokenizer,
        max_new_tokens=retry_max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    updated_outputs = list(outputs)
    prompts = [
        f"{updated_outputs[idx].rstrip()}\n{retry_suffix}"
        for idx in pending_indices
    ]
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for start, end in tqdm(
        batch_iter_indices(len(prompts), batch_size),
        total=total_batches,
        desc="solver answer-retry",
    ):
        batch_prompts = prompts[start:end]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                **gen_kwargs,
            )

        prompt_len = batch_inputs["input_ids"].size(1)
        gen_ids = generated[:, prompt_len:]
        batch_suffix = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for local_i, suffix in enumerate(batch_suffix):
            global_i = pending_indices[start + local_i]
            updated_outputs[global_i] = batch_prompts[local_i] + suffix

    release_resources(model, tokenizer)
    return updated_outputs, len(pending_indices)


def render_inputs_for_logging(
    model_name_or_path: str,
    user_prompts: Sequence[str],
    trust_remote_code: bool,
    agent_name: str,
    enable_thinking: bool,
) -> List[str]:
    if not user_prompts:
        return []
    tokenizer = None
    try:
        tokenizer = load_agent_tokenizer(
            model_name_or_path=model_name_or_path,
            trust_remote_code=trust_remote_code,
            agent_name=f"{agent_name}-log",
        )
        rendered = [render_chat_prompt(tokenizer, prompt, enable_thinking) for prompt in user_prompts]
        return rendered
    except Exception as exc:
        print(
            f"[warn] failed to render full chat template for {agent_name} inputs: {exc}. "
            "Falling back to raw user prompts in logs."
        )
        return list(user_prompts)
    finally:
        if tokenizer is not None:
            release_resources(tokenizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mas_shape", type=str, default="chain", choices=["chain"])
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--choice_old_prompt", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--gpqa_no_option_shuffle", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=-1,
        help=(
            "Base sampling seed for rollout generation. "
            "If < 0, uses --seed. Rollout r uses (base + r)."
        ),
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Number of stochastic rollouts for pass@k evaluation.",
    )
    parser.add_argument(
        "--num_recursive_rounds",
        type=int,
        default=2,
        help="Number of recursive rounds for *_recursive methods.",
    )
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--agent1_model_name_or_path", type=str, default=None)
    parser.add_argument("--agent2_model_name_or_path", type=str, default=None)
    parser.add_argument("--agent3_model_name_or_path", type=str, default=None)

    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--agent1_inner_aligner_path", type=str, default=None)
    parser.add_argument("--agent2_inner_aligner_path", type=str, default=None)
    parser.add_argument("--agent3_inner_aligner_path", type=str, default=None)
    parser.add_argument("--outer_12_path", type=str, default=None)
    parser.add_argument("--outer_23_path", type=str, default=None)
    parser.add_argument("--outer_31_path", type=str, default=None)
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="If >=0, set generation top_k (sampling). -1 keeps model default.",
    )
    parser.add_argument(
        "--min_p",
        type=float,
        default=-1.0,
        help="If >=0, set generation min_p (sampling). -1 keeps model default.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="HF repetition penalty for generation.",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help=(
            "Accepted for compatibility, but not supported by HF generation in this pipeline. "
            "Non-zero values are currently ignored."
        ),
    )
    parser.add_argument(
        "--ans",
        action="store_true",
        help=(
            "Retry missing boxed answers by appending '\nFinal Answer: \\boxed{' "
            "to agent3 output and generating more tokens."
        ),
    )
    parser.add_argument(
        "--ans_max_new_tokens",
        type=int,
        default=-1,
        help="Answer-retry generation length. -1 uses 16 for math/choice and 256 for code.",
    )
    parser.add_argument(
        "--lcb_use_private_tests",
        type=int,
        default=0,
        choices=[0, 1],
        help="For LiveCodeBench dataset, include private tests during evaluation.",
    )
    parser.add_argument(
        "--lcb_timeout_s",
        type=int,
        default=6,
        help="Per-test timeout (seconds) for LiveCodeBench code evaluation.",
    )
    parser.add_argument(
        "--mbppplus_timeout_s",
        type=int,
        default=10,
        help="Per-sample timeout (seconds) for MBPP+ script-based evaluation.",
    )
    parser.add_argument(
        "--mbppplus_num_prompt_tests",
        type=int,
        default=3,
        help="How many test_list assertions to include in MBPP+ question prompt.",
    )
    parser.add_argument(
        "--mbppplus_subset",
        type=str,
        default="",
        help="Optional evalplus/mbppplus subset name. Empty uses default.",
    )
    parser.add_argument(
        "--mbppplus_cache_dir",
        type=str,
        default="",
        help="Optional HF datasets cache_dir for MBPP+ loading.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["float32", "float16", "bfloat16", "auto"],
    )
    parser.add_argument(
        "--outer_dtype",
        type=str,
        default="auto",
        choices=["float32", "float16", "bfloat16", "auto"],
    )
    parser.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--enable_thinking",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 enables thinking mode, 0 disables it for non-thinking generation.",
    )

    parser.add_argument(
        "--result_jsonl",
        type=str,
        default="",
        help="Optional path to save per-sample results as JSONL.",
    )

    parser.add_argument("--solver_pre_question", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.method = "ours_recursive"
    global _GEN_TOP_K, _GEN_MIN_P, _GEN_REPETITION_PENALTY

    if args.mas_shape != "chain":
        raise ValueError(f"Unsupported --mas_shape: {args.mas_shape}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if args.num_rollouts <= 0:
        raise ValueError("--num_rollouts must be positive.")
    if args.num_recursive_rounds <= 0:
        raise ValueError("--num_recursive_rounds must be positive.")
    if args.top_k < -1:
        raise ValueError("--top_k must be >= -1.")
    if args.min_p < -1.0:
        raise ValueError("--min_p must be >= -1.")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition_penalty must be positive.")
    _GEN_TOP_K = args.top_k if args.top_k >= 0 else None
    _GEN_MIN_P = args.min_p if args.min_p >= 0 else None
    _GEN_REPETITION_PENALTY = float(args.repetition_penalty)
    if abs(float(args.presence_penalty)) > 1e-12:
        print(
            "[warn] --presence_penalty is currently ignored in this HF pipeline "
            "(not supported by GenerationConfig)."
        )

    planner_model = args.agent1_model_name_or_path
    refiner_model = args.agent2_model_name_or_path
    solver_model = args.agent3_model_name_or_path
    if not planner_model or not refiner_model or not solver_model:
        raise ValueError(
            "Please provide all of "
            "--agent1_model_name_or_path/--agent2_model_name_or_path/--agent3_model_name_or_path."
        )

    if args.latent_steps < 0:
        raise ValueError("--latent_steps must be non-negative.")
    if not args.agent1_inner_aligner_path or not args.agent2_inner_aligner_path or not args.agent3_inner_aligner_path:
        raise ValueError(
            "Please provide --agent1_inner_aligner_path, --agent2_inner_aligner_path, "
            "and --agent3_inner_aligner_path."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = resolve_dtype(args.dtype)
    outer_dtype = resolve_dtype(args.outer_dtype)
    if model_dtype is None or outer_dtype is None:
        raise ValueError("Unsupported dtype configuration.")

    if device.type == "cpu" and model_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU selected with fp16/bf16. Falling back model dtype to float32.")
        model_dtype = torch.float32
    if device.type == "cpu" and outer_dtype in {torch.float16, torch.bfloat16}:
        print("[warn] CPU selected with fp16/bf16 outer adapter. Falling back outer dtype to float32.")
        outer_dtype = torch.float32

    trust_remote_code = bool(args.trust_remote_code)
    enable_thinking = bool(args.enable_thinking)

    outer_12_type = args.outer_adapter_type_fallback
    outer_23_type = args.outer_adapter_type_fallback
    outer_31_type = args.outer_adapter_type_fallback
    outer_12_path, outer_23_path, outer_31_path = resolve_recursive_outer_paths(
        outer_12_path=args.outer_12_path,
        outer_23_path=args.outer_23_path,
        outer_31_path=args.outer_31_path,
    )

    dataset_name, questions, gold_answers, sample_metadata = load_eval_questions_and_answers(
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
        shuffle=bool(args.shuffle),
        seed=int(args.seed),
        gpqa_shuffle_options=not bool(args.gpqa_no_option_shuffle),
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

    print(
        f"Running method=ours_recursive on {len(questions)} samples "
        f"(planner={planner_model}, refiner={refiner_model}, solver={solver_model}, mas_shape={args.mas_shape})"
    )
    if args.num_rollouts > 1:
        print(f"[rollout] num_rollouts={args.num_rollouts} (pass@{args.num_rollouts})")
        if not args.do_sample:
            print("[warn] --num_rollouts > 1 but --do_sample is disabled; outputs may be identical.")

    base_sample_seed = args.sample_seed if args.sample_seed >= 0 else args.seed

    planner_questions = list(questions)
    is_text_method = args.method in {"text", "text_recursive"}
    planner_soften_step_template = is_text_method and (not is_code_eval) and is_gemma_model_name(planner_model)
    refiner_force_plan_only = is_text_method and (not is_code_eval) and is_choice_dataset(dataset_name)
    is_gpqa_4b_text = (
        is_text_method
        and is_gpqa_dataset(dataset_name)
        and "gemma-3-4b" in planner_model.lower()
        and "llama-3.2-3b" in refiner_model.lower()
    )
    if (
        args.method in {"text", "text_recursive"}
        and is_choice_dataset(dataset_name)
        and is_gemma_model_name(planner_model)
    ):
        planner_questions = [strip_choice_instruction_lines(q) for q in questions]
        print(
            "[prompt] planner question cleanup enabled for Gemma on choice dataset: "
            "removed 'Choose the correct option' / 'Final Choice' lines in planner stage."
        )
    if planner_soften_step_template:
        print(
            "[prompt] planner format softening enabled for Gemma in text methods: "
            "removed rigid 'Step 1...Step n' template instruction."
        )
    planner_stage_max_new_tokens = args.max_new_tokens
    planner_output_char_limit = -1
    refiner_output_char_limit = -1
    if planner_soften_step_template:
        planner_stage_max_new_tokens = min(args.max_new_tokens, 200)
        planner_output_char_limit = 1000
        if planner_stage_max_new_tokens < args.max_new_tokens:
            print(
                f"[prompt] planner generation cap enabled for Gemma in text methods: "
                f"max_new_tokens {args.max_new_tokens} -> {planner_stage_max_new_tokens}."
            )
        print(
            f"[prompt] planner output char clamp enabled for Gemma in text methods: "
            f"max_chars={planner_output_char_limit}."
        )
    if is_gpqa_4b_text:
        planner_output_char_limit = 500
        refiner_output_char_limit = 500
        print(
            "[prompt] GPQA 4B text clamp enabled: planner/refiner outputs capped to 500 chars."
        )

    def postprocess_planner_outputs(outputs: List[str], stage_name: str) -> List[str]:
        if planner_output_char_limit <= 0:
            return outputs
        patched: List[str] = []
        truncated = 0
        for text in outputs:
            clipped = truncate_text_chars(text, planner_output_char_limit)
            if len(clipped) < len(text):
                truncated += 1
            patched.append(clipped)
        if truncated > 0:
            print(
                f"[prompt] {stage_name}: char-clamped {truncated}/{len(outputs)} planner outputs "
                f"to {planner_output_char_limit} chars."
            )
        return patched

    def postprocess_refiner_outputs(outputs: List[str], stage_name: str) -> List[str]:
        if refiner_output_char_limit <= 0:
            return outputs
        patched: List[str] = []
        truncated = 0
        for text in outputs:
            clipped = truncate_text_chars(text, refiner_output_char_limit)
            if len(clipped) < len(text):
                truncated += 1
            patched.append(clipped)
        if truncated > 0:
            print(
                f"[prompt] {stage_name}: char-clamped {truncated}/{len(outputs)} refiner outputs "
                f"to {refiner_output_char_limit} chars."
            )
        return patched

    def set_rollout_seed(rollout_idx: int) -> int:
        rollout_seed = int(base_sample_seed + rollout_idx)
        random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        return rollout_seed
    def build_planner_prompt_text(question: str, sample_idx: int, feedback_text: Optional[str] = None) -> str:
        if is_code_eval:
            if task_types is None:
                raise RuntimeError("Missing task_types for code planner prompt.")
            fn_name = fn_names[sample_idx] if fn_names is not None else None
            if feedback_text is None:
                return build_code_planner_prompt(question, task_types[sample_idx], fn_name=fn_name)
            return build_code_planner_prompt_with_feedback_slot(
                question,
                task_types[sample_idx],
                fn_name=fn_name,
            ).replace(FEEDBACK_SLOT, feedback_text)

        if feedback_text is None:
            prompt = build_math_planner_prompt(question)
        else:
            prompt = build_math_planner_prompt_with_feedback_slot(question).replace(FEEDBACK_SLOT, feedback_text)
        if planner_soften_step_template:
            prompt = soften_planner_format_instruction(prompt)
        return prompt

    def build_refiner_prompt_text(question: str, planner_output: str, sample_idx: int) -> str:
        if is_code_eval:
            if task_types is None:
                raise RuntimeError("Missing task_types for code refiner prompt.")
            fn_name = fn_names[sample_idx] if fn_names is not None else None
            return build_code_refiner_prompt(
                question,
                planner_output,
                task_types[sample_idx],
                fn_name=fn_name,
            )
        prompt = build_math_refiner_prompt(question, planner_output)
        if refiner_force_plan_only:
            prompt = f"{prompt}\nDo not calculate the final answer."
        return prompt

    def build_solver_prompt_text(question: str, refined_plan: str, sample_idx: int) -> str:
        if is_code_eval:
            if task_types is None:
                raise RuntimeError("Missing task_types for code solver prompt.")
            fn_name = fn_names[sample_idx] if fn_names is not None else None
            return build_code_solver_prompt(
                question,
                refined_plan,
                task_types[sample_idx],
                args=args,
                fn_name=fn_name,
            )
        return build_math_solver_prompt(question, refined_plan, args)

    agent1_inputs: List[str] = [
        build_planner_prompt_text(planner_questions[i], i)
        for i in range(len(planner_questions))
    ]
    agent1_outputs: List[str] = []
    agent2_inputs: List[str] = []
    agent2_outputs: List[str] = []
    agent3_inputs: List[str] = []
    agent3_outputs: List[str] = []
    agent1_inputs_for_log: List[str] = []
    agent2_inputs_for_log: List[str] = []
    agent3_inputs_for_log: List[str] = []
    rollout_seeds: List[int] = []
    if args.do_sample:
        first_seed = set_rollout_seed(0)
        rollout_seeds.append(first_seed)
        print(f"[rollout 1/{args.num_rollouts}] sample_seed={first_seed}")
    else:
        rollout_seeds.append(base_sample_seed)

    solver_rollout_latents: Optional[List[torch.Tensor]] = None
    text_recursive_solver_outputs_rounds_for_log: Optional[List[List[str]]] = None

    if args.method == "text":
        planner_outputs, planner_inputs_rendered = run_text_generation_stage(
            stage_name="planner",
            model_name_or_path=planner_model,
            user_prompts=agent1_inputs,
            batch_size=args.batch_size,
            max_new_tokens=planner_stage_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
        )
        planner_outputs = postprocess_planner_outputs(planner_outputs, "planner")

        refiner_prompts = [
            build_refiner_prompt_text(questions[i], planner_outputs[i], i)
            for i in range(len(questions))
        ]
        refiner_outputs, refiner_inputs_rendered = run_text_generation_stage(
            stage_name="refiner",
            model_name_or_path=refiner_model,
            user_prompts=refiner_prompts,
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
        refiner_outputs = postprocess_refiner_outputs(refiner_outputs, "refiner")

        solver_prompts = [
            build_solver_prompt_text(questions[i], refiner_outputs[i], i)
            for i in range(len(questions))
        ]
        solver_outputs, solver_inputs_rendered = run_text_generation_stage(
            stage_name="solver",
            model_name_or_path=solver_model,
            user_prompts=solver_prompts,
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
        agent1_outputs = planner_outputs
        agent2_inputs = refiner_prompts
        agent2_outputs = refiner_outputs
        agent3_inputs = solver_prompts
        agent3_outputs = solver_outputs

        agent1_inputs_for_log = planner_inputs_rendered
        agent2_inputs_for_log = refiner_inputs_rendered
        agent3_inputs_for_log = solver_inputs_rendered
    elif args.method == "text_recursive":
        recursive_rounds = int(args.num_recursive_rounds)
        if recursive_rounds <= 0:
            raise ValueError("--num_recursive_rounds must be positive for text_recursive.")

        planner_outputs_rounds: List[List[str]] = []
        refiner_outputs_rounds: List[List[str]] = []
        solver_outputs_rounds: List[List[str]] = []
        planner_inputs_rendered_rounds: List[List[str]] = []
        refiner_inputs_rendered_rounds: List[List[str]] = []
        solver_inputs_rendered_rounds: List[List[str]] = []

        refiner_prompts_last: List[str] = []
        solver_prompts_last: List[str] = []
        solver_feedback: Optional[List[str]] = None

        for rid in range(1, recursive_rounds + 1):
            if rid == 1:
                planner_prompts_r = list(agent1_inputs)
            else:
                if solver_feedback is None:
                    raise RuntimeError("Missing solver feedback for text-recursive round > 1.")
                planner_prompts_r = [
                    build_planner_prompt_text(planner_questions[i], i, solver_feedback[i])
                    for i in range(len(planner_questions))
                ]

            planner_outputs_r, planner_inputs_r_rendered = run_text_generation_stage(
                stage_name=f"planner-r{rid}",
                model_name_or_path=planner_model,
                user_prompts=planner_prompts_r,
                batch_size=args.batch_size,
                max_new_tokens=planner_stage_max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
                enable_thinking=enable_thinking,
            )
            planner_outputs_r = postprocess_planner_outputs(planner_outputs_r, f"planner-r{rid}")

            refiner_prompts_r = [
                build_refiner_prompt_text(questions[i], planner_outputs_r[i], i)
                for i in range(len(questions))
            ]
            refiner_outputs_r, refiner_inputs_r_rendered = run_text_generation_stage(
                stage_name=f"refiner-r{rid}",
                model_name_or_path=refiner_model,
                user_prompts=refiner_prompts_r,
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
            refiner_outputs_r = postprocess_refiner_outputs(refiner_outputs_r, f"refiner-r{rid}")

            solver_prompts_r = [
                build_solver_prompt_text(questions[i], refiner_outputs_r[i], i)
                for i in range(len(questions))
            ]
            solver_outputs_r, solver_inputs_r_rendered = run_text_generation_stage(
                stage_name=f"solver-r{rid}",
                model_name_or_path=solver_model,
                user_prompts=solver_prompts_r,
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

            planner_outputs_rounds.append(planner_outputs_r)
            refiner_outputs_rounds.append(refiner_outputs_r)
            solver_outputs_rounds.append(solver_outputs_r)
            planner_inputs_rendered_rounds.append(planner_inputs_r_rendered)
            refiner_inputs_rendered_rounds.append(refiner_inputs_r_rendered)
            solver_inputs_rendered_rounds.append(solver_inputs_r_rendered)

            refiner_prompts_last = refiner_prompts_r
            solver_prompts_last = solver_prompts_r
            solver_feedback = solver_outputs_r

        agent1_outputs = [
            "\n\n".join(
                f"[round{rid}]\n{planner_outputs_rounds[rid - 1][i]}"
                for rid in range(1, recursive_rounds + 1)
            )
            for i in range(len(questions))
        ]
        agent2_inputs = refiner_prompts_last
        agent2_outputs = [
            "\n\n".join(
                f"[round{rid}]\n{refiner_outputs_rounds[rid - 1][i]}"
                for rid in range(1, recursive_rounds + 1)
            )
            for i in range(len(questions))
        ]
        agent3_inputs = solver_prompts_last
        agent3_outputs = solver_outputs_rounds[-1]
        text_recursive_solver_outputs_rounds_for_log = [list(x) for x in solver_outputs_rounds]

        agent1_inputs_for_log = [
            "\n\n".join(
                f"[Round{rid} planner input]\n{planner_inputs_rendered_rounds[rid - 1][i]}"
                for rid in range(1, recursive_rounds + 1)
            )
            for i in range(len(questions))
        ]
        agent2_inputs_for_log = [
            "\n\n".join(
                f"[Round{rid} refiner input]\n{refiner_inputs_rendered_rounds[rid - 1][i]}"
                for rid in range(1, recursive_rounds + 1)
            )
            for i in range(len(questions))
        ]
        agent3_inputs_for_log = [
            "\n\n".join(
                f"[Round{rid} solver input]\n{solver_inputs_rendered_rounds[rid - 1][i]}"
                for rid in range(1, recursive_rounds + 1)
            )
            for i in range(len(questions))
        ]
    elif args.method == "ours":
        planner_to_refiner = run_planner_latent_stage(
            model_name_or_path=planner_model,
            questions=questions,
            agent1_inner_aligner_path=args.agent1_inner_aligner_path,
            outer_12_path=outer_12_path,
            outer_12_type=outer_12_type,
            latent_steps=args.latent_steps,
            batch_size=args.batch_size,
            device=device,
            model_dtype=model_dtype,
            outer_dtype=outer_dtype,
            trust_remote_code=trust_remote_code,
            inner_adapter_type_fallback=args.inner_adapter_type_fallback,
            enable_thinking=enable_thinking,
            task_types=task_types,
            fn_names=fn_names,
        )
        refiner_to_solver = run_refiner_latent_stage(
            model_name_or_path=refiner_model,
            questions=questions,
            planner_latents=planner_to_refiner,
            agent2_inner_aligner_path=args.agent2_inner_aligner_path,
            outer_23_path=outer_23_path,
            outer_23_type=outer_23_type,
            latent_steps=args.latent_steps,
            batch_size=args.batch_size,
            device=device,
            model_dtype=model_dtype,
            outer_dtype=outer_dtype,
            trust_remote_code=trust_remote_code,
            inner_adapter_type_fallback=args.inner_adapter_type_fallback,
            enable_thinking=enable_thinking,
            task_types=task_types,
            fn_names=fn_names,
        )
        solver_outputs = run_solver_latent_stage(
            model_name_or_path=solver_model,
            questions=questions,
            refiner_latents=refiner_to_solver,
            args=args,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            task_types=task_types,
            fn_names=fn_names,
        )
        planner_to_refiner_desc = [format_latent_info(x) for x in planner_to_refiner]
        refiner_to_solver_desc = [format_latent_info(x) for x in refiner_to_solver]
        solver_rollout_latents = [x for x in refiner_to_solver]

        agent1_outputs = [
            f"to_agent2={planner_to_refiner_desc[i]}"
            for i in range(len(questions))
        ]
        agent2_inputs = []
        for i, question in enumerate(questions):
            if is_code_eval:
                if task_types is None:
                    raise RuntimeError("Missing task_types for code refiner slot prompt.")
                fn_name = fn_names[i] if fn_names is not None else None
                a2_in = build_code_refiner_prompt_with_slot(
                    question,
                    task_types[i],
                    fn_name=fn_name,
                ).replace(PLANNER_SLOT, planner_to_refiner_desc[i])
            else:
                a2_in = build_math_refiner_prompt_with_slot(question).replace(
                    PLANNER_SLOT, planner_to_refiner_desc[i]
                )
            agent2_inputs.append(a2_in)
        agent2_outputs = [
            f"to_agent3={refiner_to_solver_desc[i]}"
            for i in range(len(questions))
        ]
        agent3_inputs = []
        for i, question in enumerate(questions):
            if is_code_eval:
                if task_types is None:
                    raise RuntimeError("Missing task_types for code solver slot prompt.")
                fn_name = fn_names[i] if fn_names is not None else None
                a3_in = build_code_solver_prompt_with_slots(
                    question,
                    task_types[i],
                    args=args,
                    mas_shape=args.mas_shape,
                    fn_name=fn_name,
                )
            else:
                a3_in = build_math_solver_prompt_with_slots(question, args, mas_shape=args.mas_shape)
            a3_in = a3_in.replace(REFINED_SLOT, refiner_to_solver_desc[i])
            agent3_inputs.append(a3_in)
        agent3_outputs = solver_outputs
    else:
        recursive_rounds = int(args.num_recursive_rounds)
        planner_to_refiner_rounds: List[List[torch.Tensor]] = []
        refiner_to_solver_rounds: List[List[torch.Tensor]] = []
        feedback_to_planner_rounds: List[List[torch.Tensor]] = []

        feedback_to_planner: Optional[List[torch.Tensor]] = None
        for round_idx in range(recursive_rounds):
            if round_idx == 0:
                planner_to_refiner = run_planner_latent_stage(
                    model_name_or_path=planner_model,
                    questions=questions,
                    agent1_inner_aligner_path=args.agent1_inner_aligner_path,
                    outer_12_path=outer_12_path,
                    outer_12_type=outer_12_type,
                    latent_steps=args.latent_steps,
                    batch_size=args.batch_size,
                    device=device,
                    model_dtype=model_dtype,
                    outer_dtype=outer_dtype,
                    trust_remote_code=trust_remote_code,
                    inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                    enable_thinking=enable_thinking,
                    task_types=task_types,
                    fn_names=fn_names,
                )
            else:
                if feedback_to_planner is None:
                    raise RuntimeError("Missing recursive feedback latents for planner stage.")
                planner_to_refiner = run_planner_feedback_latent_stage(
                    model_name_or_path=planner_model,
                    questions=questions,
                    feedback_latents=feedback_to_planner,
                    agent1_inner_aligner_path=args.agent1_inner_aligner_path,
                    outer_12_path=outer_12_path,
                    outer_12_type=outer_12_type,
                    latent_steps=args.latent_steps,
                    batch_size=args.batch_size,
                    device=device,
                    model_dtype=model_dtype,
                    outer_dtype=outer_dtype,
                    trust_remote_code=trust_remote_code,
                    inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                    enable_thinking=enable_thinking,
                )
            planner_to_refiner = [x for x in planner_to_refiner]
            planner_to_refiner_rounds.append(planner_to_refiner)

            refiner_to_solver = run_refiner_latent_stage(
                model_name_or_path=refiner_model,
                questions=questions,
                planner_latents=planner_to_refiner,
                agent2_inner_aligner_path=args.agent2_inner_aligner_path,
                outer_23_path=outer_23_path,
                outer_23_type=outer_23_type,
                latent_steps=args.latent_steps,
                batch_size=args.batch_size,
                device=device,
                model_dtype=model_dtype,
                outer_dtype=outer_dtype,
                trust_remote_code=trust_remote_code,
                inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                enable_thinking=enable_thinking,
                task_types=task_types,
                fn_names=fn_names,
            )
            refiner_to_solver = [x for x in refiner_to_solver]
            refiner_to_solver_rounds.append(refiner_to_solver)

            if round_idx < recursive_rounds - 1:
                feedback_to_planner = run_solver_feedback_latent_stage(
                    model_name_or_path=solver_model,
                    questions=questions,
                    refiner_latents=refiner_to_solver,
                    agent3_inner_aligner_path=args.agent3_inner_aligner_path,
                    outer_31_path=outer_31_path,
                    outer_31_type=outer_31_type,
                    latent_steps=args.latent_steps,
                    batch_size=args.batch_size,
                    device=device,
                    model_dtype=model_dtype,
                    outer_dtype=outer_dtype,
                    trust_remote_code=trust_remote_code,
                    inner_adapter_type_fallback=args.inner_adapter_type_fallback,
                    enable_thinking=enable_thinking,
                    args=args,
                    task_types=task_types,
                    fn_names=fn_names,
                )
                feedback_to_planner = [x for x in feedback_to_planner]
                feedback_to_planner_rounds.append(feedback_to_planner)

        final_refiner_to_solver = refiner_to_solver_rounds[-1]
        solver_outputs = run_solver_latent_stage(
            model_name_or_path=solver_model,
            questions=questions,
            refiner_latents=final_refiner_to_solver,
            args=args,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            task_types=task_types,
            fn_names=fn_names,
        )

        planner_to_refiner_desc_rounds = [
            [format_latent_info(x) for x in round_latents] for round_latents in planner_to_refiner_rounds
        ]
        refiner_to_solver_desc_rounds = [
            [format_latent_info(x) for x in round_latents] for round_latents in refiner_to_solver_rounds
        ]
        feedback_to_planner_desc_rounds = [
            [format_latent_info(x) for x in round_latents] for round_latents in feedback_to_planner_rounds
        ]
        solver_rollout_latents = [x for x in final_refiner_to_solver]

        agent1_outputs = []
        for i in range(len(questions)):
            parts = [
                f"r{rid + 1}_to_agent2={planner_to_refiner_desc_rounds[rid][i]}"
                for rid in range(recursive_rounds)
            ]
            agent1_outputs.append("; ".join(parts))

        final_planner_desc = planner_to_refiner_desc_rounds[-1]
        agent2_inputs = []
        for i, question in enumerate(questions):
            if is_code_eval:
                if task_types is None:
                    raise RuntimeError("Missing task_types for code refiner slot prompt.")
                fn_name = fn_names[i] if fn_names is not None else None
                a2_in = build_code_refiner_prompt_with_slot(
                    question,
                    task_types[i],
                    fn_name=fn_name,
                ).replace(PLANNER_SLOT, final_planner_desc[i])
            else:
                a2_in = build_math_refiner_prompt_with_slot(question).replace(PLANNER_SLOT, final_planner_desc[i])
            agent2_inputs.append(a2_in)

        agent2_outputs = []
        for i in range(len(questions)):
            parts = [
                f"r{rid + 1}_to_agent3={refiner_to_solver_desc_rounds[rid][i]}"
                for rid in range(recursive_rounds)
            ]
            agent2_outputs.append("; ".join(parts))

        final_refiner_desc = refiner_to_solver_desc_rounds[-1]
        agent3_inputs = []
        for i, question in enumerate(questions):
            if is_code_eval:
                if task_types is None:
                    raise RuntimeError("Missing task_types for code solver slot prompt.")
                fn_name = fn_names[i] if fn_names is not None else None
                a3_in = build_code_solver_prompt_with_slots(
                    question,
                    task_types[i],
                    args=args,
                    mas_shape=args.mas_shape,
                    fn_name=fn_name,
                )
            else:
                a3_in = build_math_solver_prompt_with_slots(question, args, mas_shape=args.mas_shape)
            a3_in = a3_in.replace(REFINED_SLOT, final_refiner_desc[i])
            agent3_inputs.append(a3_in)
        agent3_outputs = solver_outputs

        # Recursive-specific logging fields (full chat template)
        a1_round1_prompts = [
            build_planner_prompt_text(planner_questions[i], i)
            for i in range(len(questions))
        ]
        a1_roundk_prompts = [
            build_planner_prompt_text(planner_questions[i], i, FEEDBACK_SLOT)
            for i in range(len(questions))
        ]
        a1_round1_rendered = render_inputs_for_logging(
            model_name_or_path=planner_model,
            user_prompts=a1_round1_prompts,
            trust_remote_code=trust_remote_code,
            agent_name="agent1-r1",
            enable_thinking=enable_thinking,
        )
        a1_roundk_rendered = render_inputs_for_logging(
            model_name_or_path=planner_model,
            user_prompts=a1_roundk_prompts,
            trust_remote_code=trust_remote_code,
            agent_name="agent1-rk",
            enable_thinking=enable_thinking,
        )
        agent1_inputs_for_log = []
        for i in range(len(questions)):
            parts = [f"[Round1 planner input]\n{a1_round1_rendered[i]}"]
            for rid in range(1, recursive_rounds):
                fb_desc = feedback_to_planner_desc_rounds[rid - 1][i]
                parts.append(f"[Round{rid} feedback latent] {fb_desc}")
                parts.append(f"[Round{rid + 1} planner input]\n{a1_roundk_rendered[i]}")
            agent1_inputs_for_log.append("\n\n".join(parts))

    ans_retry_count = 0
    ans_retry_max_new_tokens = int(args.ans_max_new_tokens)
    if ans_retry_max_new_tokens <= 0:
        ans_retry_max_new_tokens = 256 if is_code_eval else 16

    if args.ans:
        agent3_outputs, ans_retry_count = run_answer_retry_stage(
            model_name_or_path=solver_model,
            outputs=agent3_outputs,
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
        retry_target = "code block" if is_code_eval else "boxed/choice answer"
        print(
            f"[ans] retried {ans_retry_count} samples with missing {retry_target} "
            f"by {ans_retry_max_new_tokens}-token continuation."
        )

    if not agent1_inputs_for_log:
        agent1_inputs_for_log = render_inputs_for_logging(
            model_name_or_path=planner_model,
            user_prompts=agent1_inputs,
            trust_remote_code=trust_remote_code,
            agent_name="agent1",
            enable_thinking=enable_thinking,
        )
    if not agent2_inputs_for_log:
        if args.method in {"ours", "ours_recursive"}:
            if is_code_eval:
                if task_types is None:
                    raise RuntimeError("Missing task_types for code refiner slot logging prompts.")
                agent2_slot_prompts = [
                    build_code_refiner_prompt_with_slot(
                        questions[i],
                        task_types[i],
                        fn_name=(fn_names[i] if fn_names is not None else None),
                    )
                    for i in range(len(questions))
                ]
            else:
                agent2_slot_prompts = [
                    build_math_refiner_prompt_with_slot(question) for question in questions
                ]
            agent2_inputs_for_log = render_inputs_for_logging(
                model_name_or_path=refiner_model,
                user_prompts=agent2_slot_prompts,
                trust_remote_code=trust_remote_code,
                agent_name="agent2",
                enable_thinking=enable_thinking,
            )
        else:
            agent2_inputs_for_log = render_inputs_for_logging(
                model_name_or_path=refiner_model,
                user_prompts=agent2_inputs,
                trust_remote_code=trust_remote_code,
                agent_name="agent2",
                enable_thinking=enable_thinking,
            )
    if not agent3_inputs_for_log:
        if args.method in {"ours", "ours_recursive"}:
            if is_code_eval:
                if task_types is None:
                    raise RuntimeError("Missing task_types for code solver slot logging prompts.")
                agent3_slot_prompts = [
                    build_code_solver_prompt_with_slots(
                        questions[i],
                        task_types[i],
                        args=args,
                        mas_shape=args.mas_shape,
                        fn_name=(fn_names[i] if fn_names is not None else None),
                    )
                    for i in range(len(questions))
                ]
            else:
                agent3_slot_prompts = [
                    build_math_solver_prompt_with_slots(question, args, mas_shape=args.mas_shape)
                    for question in questions
                ]
            agent3_inputs_for_log = render_inputs_for_logging(
                model_name_or_path=solver_model,
                user_prompts=agent3_slot_prompts,
                trust_remote_code=trust_remote_code,
                agent_name="agent3",
                enable_thinking=enable_thinking,
            )
        else:
            agent3_inputs_for_log = render_inputs_for_logging(
                model_name_or_path=solver_model,
                user_prompts=agent3_inputs,
                trust_remote_code=trust_remote_code,
                agent_name="agent3",
                enable_thinking=enable_thinking,
            )

    agent3_outputs_by_rollout: List[List[str]] = [list(agent3_outputs)]
    if args.num_rollouts > 1:
        for rollout_idx in range(1, args.num_rollouts):
            if args.do_sample:
                rollout_seed = set_rollout_seed(rollout_idx)
                print(f"[rollout {rollout_idx + 1}/{args.num_rollouts}] sample_seed={rollout_seed}")
            else:
                rollout_seed = int(base_sample_seed + rollout_idx)
            rollout_seeds.append(rollout_seed)

            if not args.do_sample:
                agent3_outputs_by_rollout.append(list(agent3_outputs))
                continue

            if args.method == "text":
                planner_outputs_r, _ = run_text_generation_stage(
                    stage_name=f"planner-r{rollout_idx + 1}",
                    model_name_or_path=planner_model,
                    user_prompts=agent1_inputs,
                    batch_size=args.batch_size,
                    max_new_tokens=planner_stage_max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )
                planner_outputs_r = postprocess_planner_outputs(
                    planner_outputs_r, f"planner-r{rollout_idx + 1}"
                )
                refiner_prompts_r = [
                    build_refiner_prompt_text(questions[i], planner_outputs_r[i], i)
                    for i in range(len(questions))
                ]
                refiner_outputs_r, _ = run_text_generation_stage(
                    stage_name=f"refiner-r{rollout_idx + 1}",
                    model_name_or_path=refiner_model,
                    user_prompts=refiner_prompts_r,
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
                refiner_outputs_r = postprocess_refiner_outputs(
                    refiner_outputs_r, f"refiner-r{rollout_idx + 1}"
                )
                solver_prompts_r = [
                    build_solver_prompt_text(questions[i], refiner_outputs_r[i], i)
                    for i in range(len(questions))
                ]
                rollout_outputs = run_text_generation_stage(
                    stage_name=f"solver-r{rollout_idx + 1}",
                    model_name_or_path=solver_model,
                    user_prompts=solver_prompts_r,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )[0]
            elif args.method == "text_recursive":
                planner_outputs_r1, _ = run_text_generation_stage(
                    stage_name=f"planner-r1-k{rollout_idx + 1}",
                    model_name_or_path=planner_model,
                    user_prompts=agent1_inputs,
                    batch_size=args.batch_size,
                    max_new_tokens=planner_stage_max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )
                planner_outputs_r1 = postprocess_planner_outputs(
                    planner_outputs_r1, f"planner-r1-k{rollout_idx + 1}"
                )
                refiner_prompts_r1 = [
                    build_refiner_prompt_text(questions[i], planner_outputs_r1[i], i)
                    for i in range(len(questions))
                ]
                refiner_outputs_r1, _ = run_text_generation_stage(
                    stage_name=f"refiner-r1-k{rollout_idx + 1}",
                    model_name_or_path=refiner_model,
                    user_prompts=refiner_prompts_r1,
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
                refiner_outputs_r1 = postprocess_refiner_outputs(
                    refiner_outputs_r1, f"refiner-r1-k{rollout_idx + 1}"
                )
                solver_prompts_r1 = [
                    build_solver_prompt_text(questions[i], refiner_outputs_r1[i], i)
                    for i in range(len(questions))
                ]
                solver_outputs_r1, _ = run_text_generation_stage(
                    stage_name=f"solver-r1-k{rollout_idx + 1}",
                    model_name_or_path=solver_model,
                    user_prompts=solver_prompts_r1,
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
                planner_prompts_r2 = [
                    build_planner_prompt_text(planner_questions[i], i, solver_outputs_r1[i])
                    for i in range(len(planner_questions))
                ]
                planner_outputs_r2, _ = run_text_generation_stage(
                    stage_name=f"planner-r2-k{rollout_idx + 1}",
                    model_name_or_path=planner_model,
                    user_prompts=planner_prompts_r2,
                    batch_size=args.batch_size,
                    max_new_tokens=planner_stage_max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )
                planner_outputs_r2 = postprocess_planner_outputs(
                    planner_outputs_r2, f"planner-r2-k{rollout_idx + 1}"
                )
                refiner_prompts_r2 = [
                    build_refiner_prompt_text(questions[i], planner_outputs_r2[i], i)
                    for i in range(len(questions))
                ]
                refiner_outputs_r2, _ = run_text_generation_stage(
                    stage_name=f"refiner-r2-k{rollout_idx + 1}",
                    model_name_or_path=refiner_model,
                    user_prompts=refiner_prompts_r2,
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
                refiner_outputs_r2 = postprocess_refiner_outputs(
                    refiner_outputs_r2, f"refiner-r2-k{rollout_idx + 1}"
                )
                solver_prompts_r2 = [
                    build_solver_prompt_text(questions[i], refiner_outputs_r2[i], i)
                    for i in range(len(questions))
                ]
                rollout_outputs = run_text_generation_stage(
                    stage_name=f"solver-r2-k{rollout_idx + 1}",
                    model_name_or_path=solver_model,
                    user_prompts=solver_prompts_r2,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                )[0]
            else:
                if solver_rollout_latents is None:
                    raise RuntimeError("Missing solver rollout latents for multi-rollout inference.")
                rollout_outputs = run_solver_latent_stage(
                    model_name_or_path=solver_model,
                    questions=questions,
                    refiner_latents=solver_rollout_latents,
                    args=args,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                    task_types=task_types,
                    fn_names=fn_names,
                )

            if args.ans:
                rollout_outputs, _ = run_answer_retry_stage(
                    model_name_or_path=solver_model,
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

            agent3_outputs_by_rollout.append(rollout_outputs)

    total = len(questions)
    result_jsonl_path = args.result_jsonl.strip()
    sample_records: List[Dict[str, object]] = []
    rollout_eval_math: List[List[Tuple[str, Optional[str], bool, str, str]]] = []
    rollout_eval_code: List[List[Dict[str, Any]]] = []
    rollout_correct_counts: List[int] = []

    for rollout_idx, outputs in enumerate(agent3_outputs_by_rollout):
        correct_count = 0

        if is_code_eval:
            if sample_metadata is None:
                raise RuntimeError("Missing LiveCodeBench metadata for code evaluation.")
            eval_rows_code: List[Dict[str, Any]] = []
            eval_start_time = time.time()
            for i in range(total):
                cleaned_output = clean_raw_output(outputs[i])
                parsed_code = extract_python_code(cleaned_output)
                eval_sample = sample_metadata[i].get("eval_sample", {})
                eval_result = evaluate_generated_code(
                    parsed_code,
                    eval_sample,
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
                gold_parsed, pred_parsed, is_correct, gold_norm, pred_norm = compare_answers(
                    gold_answers[i],
                    outputs[i],
                    dataset_name=dataset_name,
                )
                if is_correct:
                    correct_count += 1
                eval_rows_math.append((gold_parsed, pred_parsed, is_correct, gold_norm, pred_norm))
            rollout_eval_math.append(eval_rows_math)

        rollout_correct_counts.append(correct_count)
        rollout_acc = 100.0 * correct_count / total if total > 0 else 0.0
        if args.num_rollouts > 1:
            print(
                f"[rollout {rollout_idx + 1}/{args.num_rollouts}] "
                f"accuracy={rollout_acc:.2f}% ({correct_count}/{total})"
            )

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
        if args.method in {"ours", "ours_recursive"}:
            print("[note] Agent2/Agent3 input logs show slot placeholders to mark latent embedding injection positions.")

        agent3_outputs_for_log = list(agent3_outputs_by_rollout[0])
        if (
            args.method == "text_recursive"
            and text_recursive_solver_outputs_rounds_for_log is not None
        ):
            round_outputs = [list(x) for x in text_recursive_solver_outputs_rounds_for_log]
            round_outputs[-1] = list(agent3_outputs_by_rollout[0])
            agent3_outputs_for_log = [
                "\n\n".join(
                    f"[round{rid}]\n{round_outputs[rid - 1][i]}"
                    for rid in range(1, len(round_outputs) + 1)
                )
                for i in range(total)
            ]

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
                            "method": args.method,
                            "mas_shape": args.mas_shape,
                            "latent_steps": args.latent_steps if args.method in {"ours", "ours_recursive"} else None,
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
                print("\n2) Agent1 Input (full chat template):")
                print(agent1_inputs_for_log[i])
                print("\n3) Agent1 Output:")
                print(agent1_outputs[i])
                print("\n4) Agent2 Input (full chat template):")
                print(agent2_inputs_for_log[i])
                print("\n5) Agent2 Output:")
                print(agent2_outputs[i])
                print("\n6) Agent3 Input (full chat template):")
                print(agent3_inputs_for_log[i])
                print("\n7) Agent3 Output:")
                print(agent3_outputs_for_log[i])
                print("\n8) Parsed python code:")
                print(parsed_code if parsed_code else "<NO_VALID_CODE_BLOCK>")
                print("\n9) Code evaluation:")
                print(json.dumps(eval_result, ensure_ascii=False))
            else:
                gold_parsed, pred_parsed, is_correct, gold_norm, pred_norm = rollout_eval_math[0][i]
                pred_display = pred_parsed if pred_parsed is not None else "<NOT_FOUND>"
                if result_jsonl_path:
                    sample_records.append(
                        {
                            "sample_idx": i,
                            "dataset": args.dataset,
                            "dataset_split": args.dataset_split,
                            "method": args.method,
                            "mas_shape": args.mas_shape,
                            "latent_steps": args.latent_steps if args.method in {"ours", "ours_recursive"} else None,
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
                if dataset_name == "openai/gsm8k":
                    print("\n2) Answer (parsed from gold #### ...):")
                else:
                    print("\n2) Gold answer (raw dataset field):")
                print(gold_parsed)
                print("\n3) Agent1 Input (full chat template):")
                print(agent1_inputs_for_log[i])
                print("\n4) Agent1 Output:")
                print(agent1_outputs[i])
                print("\n5) Agent2 Input (full chat template):")
                print(agent2_inputs_for_log[i])
                print("\n6) Agent2 Output:")
                print(agent2_outputs[i])
                print("\n7) Agent3 Input (full chat template):")
                print(agent3_inputs_for_log[i])
                print("\n8) Agent3 Output:")
                print(agent3_outputs_for_log[i])
                if dataset_name == "openai/gsm8k":
                    print("\n9) Parsed answer from Agent3 output (boxed):")
                else:
                    print("\n9) Parsed answer from Agent3 output:")
                print(pred_display)
                print("\n10) Compare normalized pure answers:")
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
                    "method": args.method,
                    "mas_shape": args.mas_shape,
                    "latent_steps": args.latent_steps if args.method in {"ours", "ours_recursive"} else None,
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
        print("=" * 120)
        print(f"pass@{args.num_rollouts}")
        print(f"pass@{args.num_rollouts}={pass_at_k:.2f}% ({pass_correct_total}/{total})")
        print("=" * 120)
        accuracy = pass_at_k


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
                "method": args.method,
                "mas_shape": args.mas_shape,
                "latent_steps": args.latent_steps if args.method in {"ours", "ours_recursive"} else None,
                "num_samples": total,
                "num_rollouts": args.num_rollouts,
                "sample_seed_base": base_sample_seed,
                "per_rollout_num_correct": rollout_correct_counts,
                "per_rollout_accuracy": [
                    (100.0 * n / total if total > 0 else 0.0) for n in rollout_correct_counts
                ],
                "num_correct": pass_correct_total if args.num_rollouts > 1 else rollout_correct_counts[0],
                "accuracy": accuracy,
                "pass_at_k": pass_at_k,
            }
            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
        print(f"[jsonl] wrote {len(sample_records)} sample records to {result_jsonl_path}")

if __name__ == "__main__":
    main()
