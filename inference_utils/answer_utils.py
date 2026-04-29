# inference_utils.py

import re
from typing import Optional, Tuple

import torch


_GSM8K_KEYS = {"gsm8k", "openai/gsm8k"}
_MATH500_KEYS = {"math500", "math-500", "huggingfaceh4/math-500"}
_MEDQA_KEYS = {
    "medqa",
    "local/medqa",
    "dataset/medqa.json",
    "./dataset/medqa.json",
}
_GPQA_KEYS = {
    "gpqa",
    "gpqa_diamond",
    "idavidrein/gpqa",
    "idavidrein/gpqa:gpqa_diamond",
    "idavidrein/gpqa_diamond",
}

def _dataset_key(name: str) -> str:
    return name.strip().lower()


def _ensure_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def truncate_text_chars(text: str, max_chars: int) -> str:
    text = _ensure_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _is_gsm8k_dataset(name: str) -> bool:
    return _dataset_key(name) in _GSM8K_KEYS


def _is_math500_dataset(name: str) -> bool:
    return _dataset_key(name) in _MATH500_KEYS


def is_medqa_dataset(name: str) -> bool:
    return _dataset_key(name) in _MEDQA_KEYS


def is_gpqa_dataset(name: str) -> bool:
    return _dataset_key(name) in _GPQA_KEYS


def is_choice_dataset(name: str) -> bool:
    key = _dataset_key(name)
    return key in _MEDQA_KEYS or key in _GPQA_KEYS



def ensure_choice_instruction(question: str) -> str:
    question = _ensure_text(question).rstrip()
    if "choose the correct option" in question.lower():
        return question
    return (
        f"{question}\n\n"
        "Choose the correct option (A/B/C/D)."
    )


def strip_choice_instruction_lines(question: str) -> str:
    """Remove extra choice-output instructions, keep stem + A/B/C/D options."""
    text = _ensure_text(question)
    if not text:
        return text

    choose_pat = re.compile(r"^\s*Choose\s+the\s+correct\s+option\s*\(A/B/C/D\)\.?\s*$", re.IGNORECASE)
    final_choice_pat = re.compile(r"^\s*Final\s*Choice\s*:\s*.*$", re.IGNORECASE)

    kept_lines = []
    for line in text.splitlines():
        if choose_pat.match(line):
            continue
        if final_choice_pat.match(line):
            continue
        kept_lines.append(line)

    # Collapse overly long blank runs introduced by line removal.
    out_lines = []
    prev_blank = False
    for line in kept_lines:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        out_lines.append(line)
        prev_blank = is_blank
    return "\n".join(out_lines).strip()


def is_gemma_model_name(model_name_or_path: str) -> bool:
    return "gemma" in _dataset_key(_ensure_text(model_name_or_path))


def soften_planner_format_instruction(prompt: str) -> str:
    """Replace rigid Step-1 template instruction with a softer instruction."""
    text = _ensure_text(prompt)
    if not text:
        return text

    replacement = (
        "Provide a clear step-by-step plan (within 3-5 steps) to solve the question. "
        "Do not calculate the final answer."
    )

    text = text.replace(
        "Your response should be in the format of:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ...",
        replacement,
    )
    text = text.replace(
        "Output only a concise plan in the format:\n"
        "Step 1: ...\n"
        "...\n"
        "Step n: ...",
        replacement,
    )
    return text


def _normalize_option_text(text: str) -> str:
    text = _ensure_text(text).strip()
    text = re.sub(r"^\s*[A-Da-d]\s*[\.\):\-]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _extract_choice_core(text: str) -> Optional[str]:
    text = _ensure_text(text).strip()
    if not text:
        return None

    # Prefer explicit "choice/answer/option" mentions.
    keyword_match = re.search(
        r"(?:final\s*(?:choice|answer)|correct\s*(?:choice|option|answer)|choice|option|answer)\s*[:\-]?\s*[\(\[]?\s*([A-Da-d])\b",
        text,
        flags=re.IGNORECASE,
    )
    if keyword_match:
        return keyword_match.group(1).upper()

    # Accept short direct forms like "A", "(B)", "C: ...".
    direct_match = re.match(r"^\s*[\(\[]?\s*([A-Da-d])(?:\s*[\)\]]|\s*[:\.\-]|$)", text)
    if direct_match:
        return direct_match.group(1).upper()

    return None


def extract_choice_answer(text: str, default: Optional[str] = None) -> Optional[str]:
    text = _ensure_text(text)
    candidates = []

    boxed = extract_boxed_answer(text)
    if boxed is not None:
        candidates.append(boxed)

    final_lines = re.findall(
        r"Final\s*(?:Choice|Answer)\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if final_lines:
        candidates.append(final_lines[-1])

    candidates.append(text.strip())

    for candidate in candidates:
        letter = _extract_choice_core(candidate)
        if letter is not None:
            return letter

    if default is None:
        return None

    fallback = _extract_choice_core(default)
    return fallback if fallback is not None else "A"


def medqa_gold_to_choice(sample: dict, default_choice: str = "A") -> str:
    answer_raw = _ensure_text(sample.get("answer", ""))
    answer_choice = extract_choice_answer(answer_raw, default=None)
    if answer_choice is not None:
        return answer_choice

    options = sample.get("options")
    if isinstance(options, list):
        norm_answer = _normalize_option_text(answer_raw)
        for opt in options:
            opt_str = _ensure_text(opt)
            label = _extract_choice_core(opt_str)
            if label is None:
                continue
            norm_opt = _normalize_option_text(opt_str)
            if norm_answer and (norm_answer == norm_opt or norm_answer in norm_opt or norm_opt in norm_answer):
                return label

    return extract_choice_answer(default_choice, default="A")


def extract_boxed_answer(text: str) -> Optional[str]:
    text = _ensure_text(text)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    else:
        boxed = re.findall(r"\\boxed\{(.*)\}", text)
        if boxed:
            return boxed[-1].strip()

    return None


def extract_pred_answer(text: str) -> Optional[str]:
    text = _ensure_text(text)
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    final_matches = re.findall(r"Final\\s+Answer\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if final_matches:
        final_answer = final_matches[-1].strip()
        if final_answer:
            return final_answer

    fallback = text.strip()
    return fallback if fallback else None


def extract_gsm8k_gold_answer(text: str) -> str:
    text = _ensure_text(text)
    hash_match = re.search(r"####\s*(.+)$", text, flags=re.DOTALL)
    if hash_match:
        candidate = hash_match.group(1).strip()
        if candidate:
            return candidate
    return text.strip()


def extract_gold_answer(text: str, dataset_name: str) -> str:
    text = _ensure_text(text)
    if is_choice_dataset(dataset_name):
        choice = extract_choice_answer(text, default=None)
        return choice if choice is not None else "A"
    if _is_gsm8k_dataset(dataset_name):
        return extract_gsm8k_gold_answer(text)
    return text.strip()


def normalize_answer_string(text: str) -> str:
    text = _ensure_text(text)
    # Rule:
    # 1) Drop decimal point and everything on its right.
    # 2) Keep digits only.
    integer_part = text.split(".", 1)[0]
    digits = "".join(re.findall(r"\d", integer_part))
    if not digits:
        return ""
    normalized = digits.lstrip("0")
    return normalized if normalized else "0"


def normalize_raw_no_space(text: str) -> str:
    text = _ensure_text(text)
    return re.sub(r"\s+", "", text.strip())


def normalize_freeform_answer_string(text: str) -> str:
    text = _ensure_text(text).strip().lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_freeform_em_string(text: str) -> str:
    text = normalize_freeform_answer_string(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _replace_text_macros(text: str) -> str:
    return re.sub(r"\\text\{([^{}]*)\}", r"\1", text)


def normalize_latex_text_string(text: str) -> str:
    text = _ensure_text(text)
    normalized = text.strip()
    normalized = _replace_text_macros(normalized)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\,", "").replace("\\;", "")
    normalized = normalized.replace("\\!", "").replace("\\:", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("\"", "").replace("'", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _replace_simple_fractions(text: str) -> str:
    def repl(match: re.Match) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator == 0:
            return match.group(0)
        return str(numerator / denominator)

    return re.sub(r"\\frac\{\s*(-?\d+)\s*\}\{\s*(-?\d+)\s*\}", repl, text)


def normalize_int_from_first_number(text: str) -> str:
    text = _ensure_text(text)
    normalized = text.strip()
    normalized = _replace_text_macros(normalized)
    normalized = _replace_simple_fractions(normalized)
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("\\circ", "")
    normalized = normalized.replace("^\\circ", "")

    number_match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not number_match:
        return ""

    value_str = number_match.group(0)
    int_part = value_str.split(".", 1)[0]
    if int_part in {"", "-", "+"}:
        return ""
    try:
        return str(int(int_part))
    except (ValueError, OverflowError):
        return ""


def normalize_date_string(text: str) -> str:
    text = normalize_freeform_answer_string(text)
    if not text:
        return ""

    text = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text)
    looks_datey = bool(
        re.search(r"\b\d{4}\b", text)
        or re.search(
            r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\b",
            text,
        )
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    )
    if not looks_datey:
        return ""

    try:
        from dateutil import parser as date_parser  # type: ignore
    except Exception:
        return ""

    try:
        parsed = date_parser.parse(text, fuzzy=False, default=None)
    except Exception:
        return ""

    has_year = bool(re.search(r"\b\d{4}\b", text))
    has_month = bool(
        re.search(
            r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)\b",
            text,
        )
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", text)
    )
    has_day = bool(
        re.search(r"\b\d{1,2}(st|nd|rd|th)?\b", text)
        and has_month
    )

    if has_year and has_month and has_day:
        return parsed.strftime("%Y-%m-%d")
    if has_year and has_month:
        return parsed.strftime("%Y-%m")
    if has_year:
        return parsed.strftime("%Y")
    return ""


def compare_answers(
    gold_text: str,
    pred_text: str,
    dataset_name: str = "openai/gsm8k",
) -> Tuple[str, Optional[str], bool, str, str]:
    gold_answer = extract_gold_answer(gold_text, dataset_name)

    if is_choice_dataset(dataset_name):
        # Per request, default to choice A if parsing fails, then override when parsed.
        pred_answer = extract_choice_answer(pred_text, default="A")
        gold_choice = extract_choice_answer(gold_answer, default="A")
        pred_choice = extract_choice_answer(pred_answer, default="A")
        correct = bool(gold_choice == pred_choice)
        return (
            gold_choice,
            pred_choice,
            correct,
            f"choice:{gold_choice.lower()}",
            f"choice:{pred_choice.lower()}",
        )

    if _is_math500_dataset(dataset_name):
        pred_answer = extract_pred_answer(pred_text)
        if pred_answer is None:
            return gold_answer, None, False, "", ""

        strategies = [
            ("intpart", normalize_int_from_first_number),
            ("latex_text", normalize_latex_text_string),
            ("nospace", normalize_raw_no_space),
            ("digits", normalize_answer_string),
        ]

        fallback_gold = ""
        fallback_pred = ""
        for strategy_name, strategy_fn in strategies:
            gold_norm = strategy_fn(gold_answer)
            pred_norm = strategy_fn(pred_answer)
            if gold_norm and pred_norm and not fallback_gold:
                fallback_gold = f"{strategy_name}:{gold_norm}"
                fallback_pred = f"{strategy_name}:{pred_norm}"
            if gold_norm and pred_norm and gold_norm == pred_norm:
                return (
                    gold_answer,
                    pred_answer,
                    True,
                    f"{strategy_name}:{gold_norm}",
                    f"{strategy_name}:{pred_norm}",
                )

        return gold_answer, pred_answer, False, fallback_gold, fallback_pred

    pred_answer = extract_pred_answer(pred_text)
    if pred_answer is None:
        return gold_answer, None, False, "", ""

    strategies = [
        ("date", normalize_date_string),
        ("em", normalize_freeform_em_string),
        ("lower", normalize_freeform_answer_string),
    ]

    fallback_gold = ""
    fallback_pred = ""
    for strategy_name, strategy_fn in strategies:
        gold_norm = strategy_fn(gold_answer)
        pred_norm = strategy_fn(pred_answer)
        if gold_norm and pred_norm and not fallback_gold:
            fallback_gold = f"{strategy_name}:{gold_norm}"
            fallback_pred = f"{strategy_name}:{pred_norm}"
        if gold_norm and pred_norm and gold_norm == pred_norm:
            return (
                gold_answer,
                pred_answer,
                True,
                f"{strategy_name}:{gold_norm}",
                f"{strategy_name}:{pred_norm}",
            )

    return gold_answer, pred_answer, False, fallback_gold, fallback_pred

    pred_answer = extract_boxed_answer(pred_text)
    gold_norm = normalize_answer_string(gold_answer)
    if pred_answer is None:
        return gold_answer, None, False, gold_norm, ""
    pred_norm = normalize_answer_string(pred_answer)
    if not gold_norm or not pred_norm:
        return gold_answer, pred_answer, False, gold_norm, pred_norm
    correct = gold_norm == pred_norm
    return gold_answer, pred_answer, correct, gold_norm, pred_norm


def format_latent_info(latent: torch.Tensor) -> str:
    steps = int(latent.size(0)) if latent.ndim >= 1 else 0
    hidden = int(latent.size(1)) if latent.ndim >= 2 else 0
    dtype = str(latent.dtype).replace("torch.", "")
    return f"<latent_embedding steps={steps} hidden={hidden} dtype={dtype}>"
