from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset


def _sample_first_nonempty_text(sample: Mapping, keys: Sequence[str]) -> str:
    for key in keys:
        value = sample.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_dataset_key(name: str) -> str:
    return str(name or "").strip().lower()


def is_extra_eval_dataset(name: str) -> bool:
    key = _normalize_dataset_key(name)
    return key in {
        "hotpotqa",
        "hotpot_qa",
        "hotpotqa/hotpot_qa",
        "hotpotqa_distractor",
        "hotpotqa_distractor_validation",
        "bamboogle",
        "chiayewken/bamboogle",
    }


def _build_hotpotqa_question_with_context(sample: Mapping) -> str:
    return _build_hotpotqa_question_with_context_flags(
        sample,
        context_mode="distractor",
        question_flag=False,
        question_prefix="[Question Start]",
        question_suffix="[Question End]",
    )


def _build_hotpotqa_question_with_context_flags(
    sample: Mapping,
    *,
    context_mode: str,
    question_flag: bool,
    question_prefix: str,
    question_suffix: str,
) -> str:
    question = _sample_first_nonempty_text(sample, ("question", "Question", "query", "prompt"))
    rendered_question = question
    if question_flag:
        rendered_question = (
            f"{str(question_prefix).rstrip()}\n"
            f"{question}\n"
            f"{str(question_suffix).rstrip()}"
        )

    if str(context_mode or "distractor").strip().lower() in {"question_only", "question-only", "qonly", "no_context", "no-context"}:
        return (
            "Answer the question. You may use available tools if needed.\n\n"
            f"Question:\n{rendered_question}"
        )

    context = sample.get("context")

    titles: List[str] = []
    sentences: List[List[str]] = []
    if isinstance(context, Mapping):
        raw_titles = context.get("title") or []
        raw_sentences = context.get("sentences") or []
        if isinstance(raw_titles, list):
            titles = [str(x).strip() for x in raw_titles]
        if isinstance(raw_sentences, list):
            for item in raw_sentences:
                if isinstance(item, list):
                    sentences.append([str(x).strip() for x in item if str(x).strip()])
                elif item is None:
                    sentences.append([])
                else:
                    text = str(item).strip()
                    sentences.append([text] if text else [])

    blocks: List[str] = []
    for idx, title in enumerate(titles):
        sent_list = sentences[idx] if idx < len(sentences) else []
        body = " ".join(x for x in sent_list if x)
        if not title and not body:
            continue
        block = f"[{idx + 1}] {title}".strip()
        if body:
            block = f"{block}\n{body}"
        blocks.append(block)

    if not blocks:
        return rendered_question

    return (
        "Answer the question using the provided context documents.\n\n"
        "Context Documents:\n"
        f"{chr(10).join(chr(10) + block if i > 0 else block for i, block in enumerate(blocks))}\n\n"
        f"Question:\n{rendered_question}"
    )


def load_extra_eval_questions_and_answers(
    dataset: str,
    dataset_split: str,
    num_samples: int,
    shuffle: bool,
    seed: int,
    return_metadata: bool = False,
    hotpot_context_mode: str = "distractor",
    hotpot_question_flag: bool = False,
    hotpot_question_prefix: str = "[Question Start]",
    hotpot_question_suffix: str = "[Question End]",
):
    key = _normalize_dataset_key(dataset)

    if key in {
        "hotpotqa",
        "hotpot_qa",
        "hotpotqa/hotpot_qa",
        "hotpotqa_distractor",
        "hotpotqa_distractor_validation",
    }:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=dataset_split)
        if len(ds) == 0:
            raise ValueError("Loaded HotpotQA distractor split is empty.")
        if shuffle:
            ds = ds.shuffle(seed=seed)
        if num_samples > 0:
            ds = ds.select(range(min(num_samples, len(ds))))

        questions = []
        gold_answers = []
        sample_metadata: List[Dict[str, Any]] = []
        for sample in ds:
            questions.append(
                _build_hotpotqa_question_with_context_flags(
                    sample,
                    context_mode=hotpot_context_mode,
                    question_flag=bool(hotpot_question_flag),
                    question_prefix=hotpot_question_prefix,
                    question_suffix=hotpot_question_suffix,
                )
            )
            gold_answers.append(_sample_first_nonempty_text(sample, ("answer", "Answer")))
            sample_metadata.append(
                {
                    "id": sample.get("id"),
                    "type": sample.get("type"),
                    "level": sample.get("level"),
                }
            )

        if return_metadata:
            dataset_name = "hotpotqa_question_only" if str(hotpot_context_mode).strip().lower() in {"question_only", "question-only", "qonly", "no_context", "no-context"} else "hotpotqa_distractor"
            return dataset_name, questions, gold_answers, sample_metadata
        dataset_name = "hotpotqa_question_only" if str(hotpot_context_mode).strip().lower() in {"question_only", "question-only", "qonly", "no_context", "no-context"} else "hotpotqa_distractor"
        return dataset_name, questions, gold_answers

    if key in {"bamboogle", "chiayewken/bamboogle"}:
        ds = load_dataset("chiayewken/bamboogle", split=dataset_split)
        if len(ds) == 0:
            raise ValueError("Loaded Bamboogle split is empty.")
        if shuffle:
            ds = ds.shuffle(seed=seed)
        if num_samples > 0:
            ds = ds.select(range(min(num_samples, len(ds))))

        questions = [_sample_first_nonempty_text(sample, ("Question", "question", "query", "prompt")) for sample in ds]
        gold_answers = [_sample_first_nonempty_text(sample, ("Answer", "answer")) for sample in ds]
        if return_metadata:
            return "bamboogle", questions, gold_answers, None
        return "bamboogle", questions, gold_answers

    return None
