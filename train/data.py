import os
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from mas_prompt import (
    build_code_planner_prompt,
    build_code_refiner_prompt,
    build_code_solver_prompt,
    build_deliberation_reflector_prompt,
    build_deliberation_toolcaller_prompt_with_slot,
    build_distill_expert_prompt,
    build_distill_learner_prompt,
    build_hie_expert_prompt,
    build_hie_summarizer_prompt,
    build_math_planner_prompt,
    build_math_refiner_prompt,
    build_math_solver_prompt,
    get_system_prompt,
)

_CHAT_TEMPLATE_STRING_FALLBACK_WARNED = False
SEQUENTIAL_ROLES = {"planner", "refiner", "solver"}
HIE_ROLES = {"hie_math_expert", "hie_code_expert", "hie_science_expert", "hie_summarizer"}
DISTILL_ROLES = {"distill_expert", "distill_learner"}
DELIBERATION_ROLES = {"deliberation_reflector", "deliberation_toolcaller"}
ALL_MAS_ROLES = SEQUENTIAL_ROLES | HIE_ROLES | DISTILL_ROLES | DELIBERATION_ROLES

HIE_ROLE_FIELD_CANDIDATES = {
    "hie_math_expert": ("hie_math_expert", "math_expert", "expert_math"),
    "hie_code_expert": ("hie_code_expert", "code_expert", "expert_code"),
    "hie_science_expert": ("hie_science_expert", "science_expert", "expert_science"),
    "hie_summarizer": ("hie_summarizer", "summary", "answer"),
}
DISTILL_ROLE_FIELD_CANDIDATES = {
    "distill_expert": ("distill_expert", "refined_plan", "expert_plan"),
    "distill_learner": ("distill_learner", "answer", "learner_answer"),
}
DELIBERATION_ROLE_FIELD_CANDIDATES = {
    "deliberation_reflector": ("deliberation_reflector", "answer"),
    "deliberation_toolcaller": ("deliberation_toolcaller", "answer"),
}


def _load_dataset_split(
    dataset_name: str,
    dataset_split: str,
    dataset_json_field: Optional[str] = None,
):
    key = dataset_name.strip().lower()
    if key in {"gsm8k", "openai/gsm8k"}:
        return load_dataset("openai/gsm8k", "main", split=dataset_split)

    if os.path.isfile(dataset_name):
        suffix = os.path.splitext(dataset_name)[1].lower()
        if suffix == ".json":
            fields_to_try: List[Optional[str]] = []
            if dataset_json_field is not None:
                fields_to_try.append(dataset_json_field)
            else:
                fields_to_try.extend(["data", None])

            last_error: Optional[Exception] = None
            for field in fields_to_try:
                try:
                    if field is None:
                        return load_dataset("json", data_files=dataset_name, split="train")
                    return load_dataset("json", data_files=dataset_name, field=field, split="train")
                except Exception as exc:  # fallback to next field option
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise ValueError(f"Failed to load local json dataset: {dataset_name}")

        if suffix == ".jsonl":
            return load_dataset("json", data_files=dataset_name, split="train")

        raise ValueError(f"Unsupported local dataset file type: {dataset_name}")

    return load_dataset(dataset_name, split=dataset_split)


def _build_texts(batch: dict, text_field: Optional[str]) -> List[str]:
    if text_field and text_field in batch:
        return batch[text_field]
    if "question" in batch and "answer" in batch:
        return [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(batch["question"], batch["answer"])
        ]
    for _, value in batch.items():
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value
    raise ValueError("No usable text field found in dataset batch.")


def _build_gsm8k_user_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}:\n"
    )


def _extract_gsm8k_final_answer(answer: str) -> str:
    match = re.search(r"####\s*(.+)$", answer, flags=re.DOTALL)
    if not match:
        return answer.strip()
    final = match.group(1).strip()
    return final if final else answer.strip()


def _format_gsm8k_answer(answer: str) -> str:
    final = _extract_gsm8k_final_answer(answer)
    reason_text = answer.rstrip()
    return f"{reason_text}\nFinal Answer: \\boxed{{{final}}}"


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int:
    if not needle:
        return -1
    end = len(haystack) - len(needle) + 1
    for idx in range(start, max(start, end)):
        if haystack[idx : idx + len(needle)] == list(needle):
            return idx
    return -1


def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages,
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


def _normalize_template_ids(
    tokenizer: PreTrainedTokenizerBase,
    value: Any,
    max_length: int,
) -> List[int]:
    global _CHAT_TEMPLATE_STRING_FALLBACK_WARNED

    # Some tokenizers/templates may ignore tokenize=True and return text.
    if isinstance(value, str):
        if not _CHAT_TEMPLATE_STRING_FALLBACK_WARNED:
            print(
                "[warn] apply_chat_template(tokenize=True) returned str; "
                "falling back to tokenizer(...) to get input_ids."
            )
            _CHAT_TEMPLATE_STRING_FALLBACK_WARNED = True
        return tokenizer(
            value,
            truncation=True,
            max_length=max_length,
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
        # Handle batched output shape [1, seq_len]
        if isinstance(value[0], list):
            return _normalize_template_ids(tokenizer, value[0], max_length=max_length)
        return [int(x) for x in value]

    raise ValueError(f"Unsupported chat template output type: {type(value)}")


def _tokenize_chat_example(
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    assistant_text: str,
    max_length: int,
    enable_thinking: bool,
    system_prompt: Optional[str] = None,
    marker_candidates: Optional[List[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    system_prompt = str(system_prompt or get_system_prompt()).strip()
    if not hasattr(tokenizer, "apply_chat_template"):
        prompt_text = f"{system_prompt}\n\n{user_prompt}"
        full_text = prompt_text + assistant_text
        if tokenizer.eos_token:
            full_text += tokenizer.eos_token

        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )["input_ids"]
        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )["input_ids"]
    else:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]

        prompt_ids = _apply_chat_template(
            tokenizer,
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        full_ids = _apply_chat_template(
            tokenizer,
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )

    full_ids = _normalize_template_ids(tokenizer, full_ids, max_length=max_length)
    prompt_ids = _normalize_template_ids(tokenizer, prompt_ids, max_length=max_length)
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    prompt_len = min(len(prompt_ids), len(full_ids))
    loss_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
    latent_mask = loss_mask.copy()

    if marker_candidates:
        marker_pos = -1
        for marker_text in marker_candidates:
            marker_ids = tokenizer(marker_text, add_special_tokens=False)["input_ids"]
            marker_pos = _find_subsequence(full_ids, marker_ids, start=prompt_len)
            if marker_pos >= 0:
                break

        if marker_pos >= 0:
            for i in range(marker_pos, len(latent_mask)):
                latent_mask[i] = 0
        else:
            print(
                "WARNING: target marker not found; using full loss mask as latent mask for this sample."
            )

    return full_ids, loss_mask, latent_mask


def _tokenize_gsm8k_chat_example(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    answer: str,
    max_length: int,
    enable_thinking: bool,
) -> Tuple[List[int], List[int], List[int]]:
    answer_text = _format_gsm8k_answer(answer)
    return _tokenize_chat_example(
        tokenizer,
        user_prompt=_build_gsm8k_user_prompt(question),
        assistant_text=answer_text,
        max_length=max_length,
        enable_thinking=enable_thinking,
        marker_candidates=["\nFinal Answer:", "Final Answer:"],
    )


def _tokenize_role_chat_example(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    planner_plan: str,
    refined_plan: str,
    answer: str,
    max_length: int,
    enable_thinking: bool,
    mas_role: str,
    solver_pre_question: int,
    task_type: str = "complete",
    mas_task: str = "math",
    fn_name: Optional[str] = None,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    question = str(question).strip()
    planner_plan = str(planner_plan).strip()
    refined_plan = str(refined_plan).strip()
    answer = str(answer).strip()

    if not question:
        return None

    mas_task = str(mas_task or "math").strip().lower()
    if mas_task not in {"math", "code", "choice"}:
        raise ValueError(f"Unsupported mas_task: {mas_task}")

    if mas_task == "code":
        if mas_role == "planner":
            user_prompt = build_code_planner_prompt(question, task_type=task_type, fn_name=fn_name)
            assistant_text = planner_plan
        elif mas_role == "refiner":
            user_prompt = build_code_refiner_prompt(question, planner_plan, task_type=task_type, fn_name=fn_name)
            assistant_text = refined_plan
        elif mas_role == "solver":
            solver_args = SimpleNamespace(solver_pre_question=int(solver_pre_question))
            user_prompt = build_code_solver_prompt(
                question,
                refined_plan,
                task_type=task_type,
                args=solver_args,
                fn_name=fn_name,
            )
            assistant_text = answer
        else:
            raise ValueError(f"Unsupported mas_role: {mas_role}")
    else:
        if mas_role == "planner":
            user_prompt = build_math_planner_prompt(question)
            assistant_text = planner_plan
        elif mas_role == "refiner":
            user_prompt = build_math_refiner_prompt(question, planner_plan)
            assistant_text = refined_plan
        elif mas_role == "solver":
            solver_args = SimpleNamespace(solver_pre_question=int(solver_pre_question))
            user_prompt = build_math_solver_prompt(question, refined_plan, args=solver_args)
            assistant_text = answer
        else:
            raise ValueError(f"Unsupported mas_role: {mas_role}")

    if not assistant_text:
        return None

    return _tokenize_chat_example(
        tokenizer,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        max_length=max_length,
        enable_thinking=enable_thinking,
        marker_candidates=None,
    )


def _tokenize_hie_role_chat_example(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    assistant_text: str,
    max_length: int,
    enable_thinking: bool,
    mas_role: str,
    task_type: str = "complete",
    mas_task: str = "math",
    fn_name: Optional[str] = None,
    math_expert_output: str = "",
    code_expert_output: str = "",
    science_expert_output: str = "",
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    question = str(question).strip()
    assistant_text = str(assistant_text).strip()
    mas_role = str(mas_role or "").strip().lower()
    if not question or not assistant_text:
        return None
    if mas_role not in HIE_ROLES:
        raise ValueError(f"Unsupported hie mas_role: {mas_role}")

    if mas_role == "hie_summarizer":
        user_prompt = build_hie_summarizer_prompt(
            question,
            math_expert_output=math_expert_output,
            code_expert_output=code_expert_output,
            science_expert_output=science_expert_output,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )
    else:
        user_prompt = build_hie_expert_prompt(
            question,
            mas_role,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )

    return _tokenize_chat_example(
        tokenizer,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        max_length=max_length,
        enable_thinking=enable_thinking,
        marker_candidates=None,
    )


def _tokenize_distill_role_chat_example(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    expert_text: str,
    learner_text: str,
    max_length: int,
    enable_thinking: bool,
    mas_role: str,
    task_type: str = "complete",
    mas_task: str = "math",
    fn_name: Optional[str] = None,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    question = str(question).strip()
    expert_text = str(expert_text).strip()
    learner_text = str(learner_text).strip()
    mas_role = str(mas_role or "").strip().lower()
    if not question:
        return None
    if mas_role not in DISTILL_ROLES:
        raise ValueError(f"Unsupported distill mas_role: {mas_role}")

    if mas_role == "distill_expert":
        assistant_text = expert_text
        if not assistant_text:
            return None
        user_prompt = build_distill_expert_prompt(
            question,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )
    else:
        assistant_text = learner_text
        if not expert_text or not assistant_text:
            return None
        user_prompt = build_distill_learner_prompt(
            question,
            expert_plan=expert_text,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )

    return _tokenize_chat_example(
        tokenizer,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        max_length=max_length,
        enable_thinking=enable_thinking,
        system_prompt=get_system_prompt(mas_design="distill", mas_role=mas_role),
        marker_candidates=None,
    )


def _tokenize_deliberation_role_chat_example(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    assistant_text: str,
    max_length: int,
    enable_thinking: bool,
    mas_role: str,
    task_type: str = "complete",
    mas_task: str = "math",
    fn_name: Optional[str] = None,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    question = str(question).strip()
    assistant_text = str(assistant_text).strip()
    mas_role = str(mas_role or "").strip().lower()
    if not question or not assistant_text:
        return None
    if mas_role not in DELIBERATION_ROLES:
        raise ValueError(f"Unsupported deliberation mas_role: {mas_role}")

    if mas_role == "deliberation_reflector":
        user_prompt = build_deliberation_reflector_prompt(
            question,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )
    else:
        user_prompt = build_deliberation_toolcaller_prompt_with_slot(
            question,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )

    return _tokenize_chat_example(
        tokenizer,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        max_length=max_length,
        enable_thinking=enable_thinking,
        system_prompt=get_system_prompt(mas_design="deliberation", mas_role=mas_role),
        marker_candidates=None,
    )


def _batch_values(batch: dict, keys: Sequence[str], default: str = "") -> List[str]:
    for key in keys:
        if key in batch:
            return [default if value is None else str(value) for value in batch[key]]
    if "question" not in batch:
        return []
    return [default] * len(batch["question"])


def load_and_tokenize_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_split: str,
    max_length: int,
    text_field: Optional[str] = None,
    enable_thinking: bool = False,
    mas_role: Optional[str] = None,
    dataset_json_field: Optional[str] = None,
    solver_pre_question: int = 0,
    mas_task: str = "math",
    mas_design: str = "sequential",
):
    dataset = _load_dataset_split(dataset_name, dataset_split, dataset_json_field=dataset_json_field)

    mas_task = str(mas_task or "math").strip().lower()
    if mas_task not in {"math", "code", "choice"}:
        raise ValueError(f"Unsupported mas_task: {mas_task}")
    mas_design = str(mas_design or "sequential").strip().lower()
    if mas_design not in {"sequential", "hie", "distill", "deliberation"}:
        raise ValueError(f"Unsupported mas_design: {mas_design}")

    def tokenize_fn(batch: dict) -> dict:
        if mas_role is not None:
            role_name = str(mas_role).strip().lower()
            if role_name not in ALL_MAS_ROLES:
                raise ValueError(f"Unsupported mas_role: {mas_role}")
            if role_name in HIE_ROLES:
                if "question" not in batch:
                    raise ValueError("hie mas_role data requires question field.")

                input_ids, loss_masks, latent_masks = [], [], []
                assistant_values = _batch_values(batch, HIE_ROLE_FIELD_CANDIDATES[role_name])
                type_values = batch.get("type", ["complete"] * len(batch["question"]))
                fn_values = batch.get("fn_name", [None] * len(batch["question"]))
                task_values = batch.get("task_family", [mas_task] * len(batch["question"]))
                math_values = _batch_values(batch, HIE_ROLE_FIELD_CANDIDATES["hie_math_expert"])
                code_values = _batch_values(batch, HIE_ROLE_FIELD_CANDIDATES["hie_code_expert"])
                science_values = _batch_values(batch, HIE_ROLE_FIELD_CANDIDATES["hie_science_expert"])
                for q, assistant, t, fn, task_value, math_text, code_text, science_text in zip(
                    batch["question"],
                    assistant_values,
                    type_values,
                    fn_values,
                    task_values,
                    math_values,
                    code_values,
                    science_values,
                ):
                    packed = _tokenize_hie_role_chat_example(
                        tokenizer,
                        q,
                        assistant,
                        max_length,
                        enable_thinking=enable_thinking,
                        mas_role=role_name,
                        task_type=t,
                        mas_task=task_value,
                        fn_name=fn,
                        math_expert_output=math_text,
                        code_expert_output=code_text,
                        science_expert_output=science_text,
                    )
                    if packed is None:
                        continue
                    ids, mask, latent = packed
                    input_ids.append(ids)
                    loss_masks.append(mask)
                    latent_masks.append(latent)
                return {"input_ids": input_ids, "loss_mask": loss_masks, "latent_mask": latent_masks}

            if role_name in DISTILL_ROLES:
                if "question" not in batch:
                    raise ValueError("distill mas_role data requires question field.")

                input_ids, loss_masks, latent_masks = [], [], []
                expert_values = _batch_values(batch, DISTILL_ROLE_FIELD_CANDIDATES["distill_expert"])
                learner_values = _batch_values(batch, DISTILL_ROLE_FIELD_CANDIDATES["distill_learner"])
                type_values = batch.get("type", ["complete"] * len(batch["question"]))
                fn_values = batch.get("fn_name", [None] * len(batch["question"]))
                task_values = batch.get("task_family", [mas_task] * len(batch["question"]))
                for q, expert_text, learner_text, t, fn, task_value in zip(
                    batch["question"],
                    expert_values,
                    learner_values,
                    type_values,
                    fn_values,
                    task_values,
                ):
                    packed = _tokenize_distill_role_chat_example(
                        tokenizer,
                        q,
                        expert_text,
                        learner_text,
                        max_length,
                        enable_thinking=enable_thinking,
                        mas_role=role_name,
                        task_type=t,
                        mas_task=task_value,
                        fn_name=fn,
                    )
                    if packed is None:
                        continue
                    ids, mask, latent = packed
                    input_ids.append(ids)
                    loss_masks.append(mask)
                    latent_masks.append(latent)
                return {"input_ids": input_ids, "loss_mask": loss_masks, "latent_mask": latent_masks}

            if role_name in DELIBERATION_ROLES:
                if "question" not in batch:
                    raise ValueError("deliberation mas_role data requires question field.")

                input_ids, loss_masks, latent_masks = [], [], []
                assistant_values = _batch_values(batch, DELIBERATION_ROLE_FIELD_CANDIDATES[role_name])
                type_values = batch.get("type", ["complete"] * len(batch["question"]))
                fn_values = batch.get("fn_name", [None] * len(batch["question"]))
                task_values = batch.get("task_family", [mas_task] * len(batch["question"]))
                for q, assistant_text, t, fn, task_value in zip(
                    batch["question"],
                    assistant_values,
                    type_values,
                    fn_values,
                    task_values,
                ):
                    packed = _tokenize_deliberation_role_chat_example(
                        tokenizer,
                        q,
                        assistant_text,
                        max_length,
                        enable_thinking=enable_thinking,
                        mas_role=role_name,
                        task_type=t,
                        mas_task=task_value,
                        fn_name=fn,
                    )
                    if packed is None:
                        continue
                    ids, mask, latent = packed
                    input_ids.append(ids)
                    loss_masks.append(mask)
                    latent_masks.append(latent)
                return {"input_ids": input_ids, "loss_mask": loss_masks, "latent_mask": latent_masks}

            if "question" not in batch or "plan" not in batch:
                raise ValueError("mas_role data requires question and plan fields.")
            if role_name in {"refiner", "solver"} and "refined_plan" not in batch:
                raise ValueError(f"mas_role={mas_role} requires refined_plan field.")
            if role_name == "solver" and "answer" not in batch:
                raise ValueError("mas_role=solver requires answer field.")

            input_ids, loss_masks, latent_masks = [], [], []
            refined_values = batch.get("refined_plan", [""] * len(batch["question"]))
            answer_values = batch.get("answer", [""] * len(batch["question"]))
            type_values = batch.get("type", ["complete"] * len(batch["question"]))
            fn_values = batch.get("fn_name", [None] * len(batch["question"]))
            task_values = batch.get("task_family", [mas_task] * len(batch["question"]))
            for q, p, rp, a, t, fn, task_value in zip(
                batch["question"],
                batch["plan"],
                refined_values,
                answer_values,
                type_values,
                fn_values,
                task_values,
            ):
                packed = _tokenize_role_chat_example(
                    tokenizer,
                    q,
                    p,
                    rp,
                    a,
                    max_length,
                    enable_thinking=enable_thinking,
                    mas_role=role_name,
                    solver_pre_question=solver_pre_question,
                    task_type=t,
                    mas_task=task_value,
                    fn_name=fn,
                )
                if packed is None:
                    continue
                ids, mask, latent = packed
                input_ids.append(ids)
                loss_masks.append(mask)
                latent_masks.append(latent)
            return {"input_ids": input_ids, "loss_mask": loss_masks, "latent_mask": latent_masks}

        if "question" in batch and "cot" in batch and "answer" in batch:
            input_ids, loss_masks, latent_masks = [], [], []
            for q, cot, a in zip(batch["question"], batch["cot"], batch["answer"]):
                cot_text = str(cot).rstrip()
                answer_text = str(a).strip()
                gsm8k_style_answer = f"{cot_text}\n#### {answer_text}"
                ids, mask, latent = _tokenize_gsm8k_chat_example(
                    tokenizer,
                    q,
                    gsm8k_style_answer,
                    max_length,
                    enable_thinking=enable_thinking,
                )
                input_ids.append(ids)
                loss_masks.append(mask)
                latent_masks.append(latent)
            return {"input_ids": input_ids, "loss_mask": loss_masks, "latent_mask": latent_masks}

        if "question" in batch and "answer" in batch:
            input_ids, loss_masks, latent_masks = [], [], []
            for q, a in zip(batch["question"], batch["answer"]):
                ids, mask, latent = _tokenize_gsm8k_chat_example(
                    tokenizer,
                    q,
                    a,
                    max_length,
                    enable_thinking=enable_thinking,
                )
                input_ids.append(ids)
                loss_masks.append(mask)
                latent_masks.append(latent)
            return {"input_ids": input_ids, "loss_mask": loss_masks, "latent_mask": latent_masks}

        texts = _build_texts(batch, text_field)
        tokenized = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        loss_masks = [[1] * len(ids) for ids in tokenized["input_ids"]]
        latent_masks = [mask.copy() for mask in loss_masks]
        return {
            "input_ids": tokenized["input_ids"],
            "loss_mask": loss_masks,
            "latent_mask": latent_masks,
        }

    return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)


class LatentDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.inner = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features: List[dict]) -> dict:
        loss_masks = [feature.pop("loss_mask") for feature in features]
        latent_masks = [feature.pop("latent_mask") for feature in features]
        batch = self.inner(features)

        max_len = batch["input_ids"].size(1)
        padded_masks = []
        padded_latent = []

        for mask in loss_masks:
            pad_len = max_len - len(mask)
            if pad_len > 0:
                padded_mask = [0] * pad_len + mask
            else:
                padded_mask = mask[-max_len:]
            padded_masks.append(padded_mask)

        for mask in latent_masks:
            pad_len = max_len - len(mask)
            if pad_len > 0:
                padded_mask = [0] * pad_len + mask
            else:
                padded_mask = mask[-max_len:]
            padded_latent.append(padded_mask)

        batch["loss_mask"] = torch.tensor(padded_masks, dtype=torch.float32)
        batch["latent_mask"] = torch.tensor(padded_latent, dtype=torch.float32)
        return batch


def build_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_split: str,
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    text_field: Optional[str] = None,
    enable_thinking: bool = False,
    mas_role: Optional[str] = None,
    dataset_json_field: Optional[str] = None,
    solver_pre_question: int = 0,
    mas_task: str = "math",
    mas_design: str = "sequential",
):
    dataset = load_and_tokenize_dataset(
        tokenizer,
        dataset_name,
        dataset_split,
        max_length,
        text_field,
        enable_thinking,
        mas_role,
        dataset_json_field,
        solver_pre_question,
        mas_task,
        mas_design,
    )
    collator = LatentDataCollator(tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        drop_last=True,
    )
