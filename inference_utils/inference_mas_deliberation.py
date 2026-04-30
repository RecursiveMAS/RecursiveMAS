import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from . import inference_mas as base
from .lcb_utils import (
    clean_raw_output,
    evaluate_generated_code,
    extract_python_code,
    is_code_eval_dataset,
    is_mbppplus_dataset,
)
from prompts import (
    DELIBERATION_FEEDBACK_SLOT,
    DELIBERATION_REFLECTOR_SLOT,
    build_deliberation_reflector_prompt,
    build_deliberation_reflector_prompt_with_feedback_slot,
    build_deliberation_toolcaller_prompt,
    build_deliberation_toolcaller_prompt_with_slot,
    get_system_prompt,
)


TOOL_RE = re.compile(r"<(python|search)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE)
UNCLOSED_TOOL_RE = re.compile(r"<(python|search)>(?!.*</\1>)", re.DOTALL | re.IGNORECASE)
DEFAULT_TAVILY_SEARCH_DEPTH = "advanced"
DEFAULT_TAVILY_MAX_RESULTS = 4
_TAVILY_CLIENT = None
_TAVILY_FALLBACK_NOTED = False

@dataclass(frozen=True)
class ToolCall:
    name: str
    content: str
    start: int
    end: int


@dataclass
class ToolLoopState:
    index: int
    assistant_text: str = ""
    cursor: int = 0
    tool_round: int = 0
    done: bool = False


class ToolTagStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: Any, start_length: int) -> None:
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.triggered_indices: set[int] = set()
        tag_texts = (
            "</python>",
            "</search>",
            "</Python>",
            "</Search>",
            "</PYTHON>",
            "</SEARCH>",
        )
        self.tag_texts_lower = tuple(text.lower() for text in tag_texts)
        self.tail_decode_tokens = 16

    def _has_closing_tag_in_tail_text(self, generated_ids: torch.LongTensor) -> bool:
        if generated_ids.numel() == 0:
            return False
        tail_ids = generated_ids[-self.tail_decode_tokens :]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False).lower()
        return any(tag_text in tail_text for tag_text in self.tag_texts_lower)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        self.triggered_indices.clear()
        for row_idx in range(input_ids.shape[0]):
            generated_ids = input_ids[row_idx, self.start_length :]
            if self._has_closing_tag_in_tail_text(generated_ids):
                self.triggered_indices.add(row_idx)
        return bool(self.triggered_indices)


def infer_deliberation_task(dataset_name: str, is_code_eval: bool) -> str:
    if is_code_eval:
        return "code"
    if base.is_reasoning_dataset(dataset_name):
        return "reasoning"
    if base.is_choice_dataset(dataset_name):
        return "choice"
    return "math"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mas_shape", type=str, default="deliberation", choices=["deliberation"])
    parser.add_argument("--dataset", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=-1)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_recursive_rounds", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--reflector_model_name_or_path", type=str, required=True)
    parser.add_argument("--toolcaller_model_name_or_path", type=str, required=True)

    parser.add_argument("--latent_steps", type=int, default=10)
    parser.add_argument("--reflector_inner_aligner_path", type=str, default="")
    parser.add_argument("--toolcaller_inner_aligner_path", type=str, default="")
    parser.add_argument("--outer_rt_path", type=str, default=None)
    parser.add_argument("--outer_tr_path", type=str, default=None)

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

    parser.add_argument("--max_tool_rounds", type=int, default=5)
    parser.add_argument("--python_timeout", type=float, default=10.0)
    parser.add_argument("--python_cwd", type=str, default=".")
    parser.add_argument("--result_max_chars", type=int, default=6000)
    parser.add_argument("--echo_last_expr", action="store_true")
    parser.add_argument("--quiet_tools", action="store_true")

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


def resolve_deliberation_outer_paths(
    outer_rt_path: Optional[str],
    outer_tr_path: Optional[str],
) -> Tuple[str, str]:
    if outer_rt_path is None or outer_tr_path is None:
        raise ValueError("Please provide both --outer_rt_path and --outer_tr_path.")
    return outer_rt_path, outer_tr_path


def resolve_deliberation_outer_types(
    outer_cfg_path: Optional[str],
    fallback_type: str,
) -> Tuple[str, str]:
    rt_type = fallback_type
    tr_type = fallback_type
    if outer_cfg_path and os.path.isfile(outer_cfg_path):
        with open(outer_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rt_type = cfg.get("outer_rt_type", rt_type)
        tr_type = cfg.get("outer_tr_type", tr_type)
    return rt_type, tr_type


def _outer_out_dim(path: str) -> int:
    return base.infer_outer_adapter_out_dim_from_file(path)


def render_chat_prompt_with_system(tokenizer, user_prompt: str, enable_thinking: bool, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered = base.apply_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return base._normalize_template_text(tokenizer, rendered)


def render_chat_prompt_ids_with_system(tokenizer, user_prompt: str, enable_thinking: bool, system_prompt: str) -> List[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered = base.apply_chat_template(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return base._normalize_template_ids(tokenizer, rendered)


def split_prompt_ids_by_slots_with_system(
    tokenizer,
    user_prompt_with_slots: str,
    slot_texts: Sequence[str],
    enable_thinking: bool,
    system_prompt: str,
) -> List[List[int]]:
    full_text = render_chat_prompt_with_system(tokenizer, user_prompt_with_slots, enable_thinking, system_prompt)
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


def build_reflector_prompt_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
    feedback_text: Optional[str] = None,
) -> str:
    task_type = task_types[sample_idx] if (mas_task == "code" and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    if feedback_text is None:
        return build_deliberation_reflector_prompt(
            question,
            mas_task=mas_task,
            task_type=task_type,
            fn_name=fn_name,
        )
    return build_deliberation_reflector_prompt_with_feedback_slot(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    ).replace(DELIBERATION_FEEDBACK_SLOT, feedback_text)


def build_toolcaller_prompt_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
    reflector_signal_text: str,
) -> str:
    task_type = task_types[sample_idx] if (mas_task == "code" and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    return build_deliberation_toolcaller_prompt(
        question,
        reflector_signal=reflector_signal_text,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def build_toolcaller_prompt_with_slot_text(
    question: str,
    sample_idx: int,
    mas_task: str,
    task_types: Optional[Sequence[str]],
    fn_names: Optional[Sequence[Optional[str]]],
) -> str:
    task_type = task_types[sample_idx] if (mas_task == "code" and task_types is not None) else "complete"
    fn_name = fn_names[sample_idx] if fn_names is not None else None
    return build_deliberation_toolcaller_prompt_with_slot(
        question,
        mas_task=mas_task,
        task_type=task_type,
        fn_name=fn_name,
    )


def find_next_tool(text: str, cursor: int) -> Optional[ToolCall]:
    match = TOOL_RE.search(text, cursor)
    if not match:
        return None
    return ToolCall(
        name=match.group(1).lower(),
        content=match.group(2).strip(),
        start=match.start(),
        end=match.end(),
    )


def has_unclosed_tool(text: str, cursor: int) -> bool:
    return UNCLOSED_TOOL_RE.search(text, cursor) is not None


def truncate_result(text: str, max_chars: int) -> str:
    text = text.replace("</result>", "< /result>")
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... [truncated {omitted} chars]"


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def echo_last_expression(code: str) -> str:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return code
    if not module.body or not isinstance(module.body[-1], ast.Expr):
        return code
    last_expr = module.body[-1]
    assign = ast.Assign(
        targets=[ast.Name(id="__tool_last_expr", ctx=ast.Store())],
        value=last_expr.value,
    )
    print_call = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id="repr", ctx=ast.Load()),
                    args=[ast.Name(id="__tool_last_expr", ctx=ast.Load())],
                    keywords=[],
                )
            ],
            keywords=[],
        )
    )
    ast.copy_location(assign, last_expr)
    ast.copy_location(print_call, last_expr)
    module.body[-1] = assign
    module.body.append(print_call)
    ast.fix_missing_locations(module)
    return ast.unparse(module)


def run_python_tool(code: str, timeout: float, cwd: str, echo_last_expr_flag: bool, max_chars: int) -> str:
    if echo_last_expr_flag:
        code = echo_last_expression(code)
    prelude = (
        "import math\n"
        "import itertools\n"
        "import functools\n"
        "import fractions\n"
        "import statistics\n"
        "try:\n"
        "    import sympy as sp\n"
        "except Exception:\n"
        "    sp = None\n\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        script_path = handle.name
        handle.write(prelude)
        handle.write(code)
        handle.write("\n")
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        pieces = []
        if completed.stdout:
            pieces.append(completed.stdout.rstrip())
        if completed.stderr:
            pieces.append("[stderr]\n" + completed.stderr.rstrip())
        if completed.returncode != 0:
            pieces.append(f"[exit_code] {completed.returncode}")
        result = "\n".join(pieces).strip() or "[no output]"
        return truncate_result(result, max_chars)
    except subprocess.TimeoutExpired:
        return f"[timeout] Python tool exceeded {timeout} seconds."
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def run_search_tool(query: str, max_chars: int) -> str:
    result = (
        f"Dummy search result for query: {query}\n"
        "External search is disabled in the release runner."
    )
    return truncate_result(result, max_chars)


def note_tavily_fallback(reason: str) -> None:
    global _TAVILY_FALLBACK_NOTED
    if _TAVILY_FALLBACK_NOTED:
        return
    print(f"[note] {reason}; falling back to dummy search.", file=sys.stderr)
    _TAVILY_FALLBACK_NOTED = True


def get_tavily_api_key() -> str:
    api_key = str(os.environ.get("TAVILY_API_KEY", "")).strip()
    if api_key:
        return api_key

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(repo_root, ".env")
    if not os.path.isfile(env_path):
        return ""

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != "TAVILY_API_KEY":
                    continue
                api_key = value.strip().strip("'\"")
                if api_key:
                    os.environ["TAVILY_API_KEY"] = api_key
                    return api_key
    except OSError:
        return ""
    return ""


def get_tavily_client():
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        from tavily import TavilyClient
        _TAVILY_CLIENT = TavilyClient(api_key=get_tavily_api_key())
    return _TAVILY_CLIENT


def run_tavily_search_tool(query: str, max_chars: int) -> str:
    api_key = get_tavily_api_key()
    if not api_key or api_key == "xxx":
        note_tavily_fallback("Tavily API key is missing")
        return run_search_tool(query, max_chars)

    try:
        client = get_tavily_client()
    except Exception as exc:
        note_tavily_fallback(f"Tavily import/init failed: {exc}")
        return run_search_tool(query, max_chars)

    try:
        response = client.search(
            query=query,
            search_depth=DEFAULT_TAVILY_SEARCH_DEPTH,
        )
    except Exception as exc:
        return truncate_result(f"[search error] {type(exc).__name__}: {exc}", max_chars)

    lines = [f"Query: {query}"]
    results = response.get("results") or []
    if results:
        lines.append("Results:")
    for idx, item in enumerate(results[:DEFAULT_TAVILY_MAX_RESULTS], start=1):
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        content = compact_whitespace(str(item.get("content") or "").strip())
        if len(content) > 320:
            content = content[:320].rstrip() + " ..."
        lines.append(f"[{idx}] {title}")
        if url:
            lines.append(f"URL: {url}")
        if content:
            lines.append(content)
    if not results:
        lines.append("[no search results]")
    return truncate_result("\n".join(lines), max_chars)


def execute_tool_call(tool_call: ToolCall, args: argparse.Namespace) -> str:
    if tool_call.name == "python":
        return run_python_tool(
            code=tool_call.content,
            timeout=args.python_timeout,
            cwd=args.python_cwd,
            echo_last_expr_flag=args.echo_last_expr,
            max_chars=args.result_max_chars,
        )
    return run_tavily_search_tool(tool_call.content, args.result_max_chars)


def apply_next_tool_if_present(state: ToolLoopState, args: argparse.Namespace) -> bool:
    tool_call = find_next_tool(state.assistant_text, state.cursor)
    if not tool_call:
        return False

    state.tool_round += 1
    if state.tool_round > args.max_tool_rounds:
        state.assistant_text = state.assistant_text[: tool_call.end]
        state.assistant_text += f" <result> [tool limit reached after {args.max_tool_rounds} calls] </result>"
        state.done = True
        return True

    state.assistant_text = state.assistant_text[: tool_call.end]
    if not args.quiet_tools:
        preview = tool_call.content.replace("\n", "\\n")
        print(f"[sample:{state.index} tool:{state.tool_round}] {tool_call.name}: {preview[:240]}", file=sys.stderr)

    result = execute_tool_call(tool_call, args)
    state.assistant_text += f" <result> {result} </result>"
    state.cursor = len(state.assistant_text)
    return True


def get_eos_token_ids(tokenizer: Any, model: Any) -> set[int]:
    eos_ids = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    ):
        if value is None:
            continue
        if isinstance(value, int):
            eos_ids.add(value)
        else:
            eos_ids.update(int(item) for item in value if item is not None)
    return eos_ids


def build_generation_kwargs_with_tool_stop(
    tokenizer: Any,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    stopping_criteria: StoppingCriteriaList,
) -> Dict[str, Any]:
    generation_kwargs = base.build_generation_kwargs(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    generation_kwargs["stopping_criteria"] = stopping_criteria
    return generation_kwargs


def generate_batch_from_embeds(
    tokenizer: Any,
    model: Any,
    batch_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[bool], set[int], List[int]]:
    input_length = int(attention_mask.size(1))
    stop_criteria = ToolTagStoppingCriteria(tokenizer, input_length)
    generation_kwargs = build_generation_kwargs_with_tool_stop(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )
    eos_token_ids = get_eos_token_ids(tokenizer, model)
    with torch.inference_mode():
        generated = model.generate(
            inputs_embeds=batch_embeds,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            **generation_kwargs,
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    gen_ids = sequences[:, input_length:] if sequences.size(1) > max_new_tokens else sequences

    texts: List[str] = []
    finished: List[bool] = []
    token_counts: List[int] = []
    for row_idx in range(gen_ids.size(0)):
        row = gen_ids[row_idx]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
        token_values = [int(token_id) for token_id in row.detach().cpu().tolist()]
        finished.append(bool(eos_token_ids.intersection(token_values)))
        token_counts.append(int(len(token_values)))
    return texts, finished, set(stop_criteria.triggered_indices), token_counts


def generate_batch_from_text_prompts(
    tokenizer: Any,
    model: Any,
    prompts: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[List[str], List[bool], set[int], List[int]]:
    batch_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    ).to(device)
    input_length = int(batch_inputs["input_ids"].size(1))
    stop_criteria = ToolTagStoppingCriteria(tokenizer, input_length)
    generation_kwargs = build_generation_kwargs_with_tool_stop(
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        stopping_criteria=StoppingCriteriaList([stop_criteria]),
    )
    eos_token_ids = get_eos_token_ids(tokenizer, model)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            return_dict_in_generate=True,
            **generation_kwargs,
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    gen_ids = sequences[:, input_length:]

    texts: List[str] = []
    finished: List[bool] = []
    token_counts: List[int] = []
    for row_idx in range(gen_ids.size(0)):
        row = gen_ids[row_idx]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
        token_values = [int(token_id) for token_id in row.detach().cpu().tolist()]
        finished.append(bool(eos_token_ids.intersection(token_values)))
        token_counts.append(int(len(token_values)))
    return texts, finished, set(stop_criteria.triggered_indices), token_counts


def run_reflector_latent_stage(
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
    feedback_latents: Optional[Sequence[torch.Tensor]],
) -> List[torch.Tensor]:
    if latent_steps == 0:
        out_dim = _outer_out_dim(outer_path)
        return [torch.empty((0, out_dim), dtype=torch.float32) for _ in questions]

    system_prompt = get_system_prompt("deliberation")
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="deliberation_reflector",
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
            prompt_payloads.append(
                split_prompt_ids_by_slots_with_system(
                    tokenizer,
                    build_deliberation_reflector_prompt_with_feedback_slot(
                        question,
                        mas_task=mas_task,
                        task_type=(task_types[idx] if (mas_task == "code" and task_types is not None) else "complete"),
                        fn_name=(fn_names[idx] if fn_names is not None else None),
                    ),
                    [DELIBERATION_FEEDBACK_SLOT],
                    enable_thinking,
                    system_prompt,
                )
            )
        else:
            prompt_payloads.append(
                render_chat_prompt_ids_with_system(
                    tokenizer,
                    build_reflector_prompt_text(question, idx, mas_task, task_types, fn_names),
                    enable_thinking,
                    system_prompt,
                )
            )

    outputs: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="delib_reflector_latent",
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


def run_toolcaller_feedback_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    reflector_latents: Sequence[torch.Tensor],
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

    system_prompt = get_system_prompt("deliberation")
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        agent_name="deliberation_toolcaller_feedback",
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
        split_prompt_ids_by_slots_with_system(
            tokenizer,
            build_toolcaller_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names),
            [DELIBERATION_REFLECTOR_SLOT],
            enable_thinking,
            system_prompt,
        )
        for idx, question in enumerate(questions)
    ]

    outputs: List[torch.Tensor] = []
    total_batches = (len(questions) + batch_size - 1) // batch_size
    for start, end in tqdm(
        base.batch_iter_indices(len(questions), batch_size),
        total=total_batches,
        desc="delib_toolcaller_feedback_latent",
    ):
        embed_seqs: List[torch.Tensor] = []
        for idx in range(start, end):
            seg_prefix, seg_suffix = prompt_segments[idx]
            prefix = base.token_ids_to_embeds(embed_layer, seg_prefix, device=device, dtype=embed_dtype)
            suffix = base.token_ids_to_embeds(embed_layer, seg_suffix, device=device, dtype=embed_dtype)
            seq = torch.cat(
                [
                    prefix,
                    reflector_latents[idx].to(device=device, dtype=embed_dtype),
                    suffix,
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
        self_latent = base.run_inner_adapter(inner, hidden_rollout, output_dtype=embed_dtype)
        mapped = base.run_outer_adapter(outer, self_latent, output_dtype=torch.float32)
        for i in range(mapped.size(0)):
            outputs.append(mapped[i].detach().cpu())

    base.release_resources(model, tokenizer, inner, outer)
    return outputs


def run_text_tool_generation_stage(
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
    args: argparse.Namespace,
    system_prompt: str,
) -> Tuple[List[str], List[str]]:
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name=stage_name,
    )
    rendered_prompts = [render_chat_prompt_with_system(tokenizer, p, enable_thinking, system_prompt) for p in user_prompts]


    states = [ToolLoopState(index=i) for i in range(len(rendered_prompts))]
    remaining_budgets = [int(max_new_tokens) for _ in range(len(rendered_prompts))]
    while True:
        active = [state for state in states if (not state.done and remaining_budgets[state.index] > 0)]
        if not active:
            break
        for batch_states in [active[i : i + batch_size] for i in range(0, len(active), batch_size)]:
            batch_budget = min(remaining_budgets[state.index] for state in batch_states)
            if batch_budget <= 0:
                for state in batch_states:
                    state.done = True
                continue
            batch_prompts = [rendered_prompts[state.index] + state.assistant_text for state in batch_states]
            continuations, finished, triggered_rows, token_counts = generate_batch_from_text_prompts(
                tokenizer=tokenizer,
                model=model,
                prompts=batch_prompts,
                max_new_tokens=batch_budget,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
            batch_stopped_for_tool = bool(triggered_rows)
            for row_idx, state in enumerate(batch_states):
                remaining_budgets[state.index] = max(0, remaining_budgets[state.index] - int(token_counts[row_idx]))
                state.assistant_text += continuations[row_idx]
                if apply_next_tool_if_present(state, args):
                    if remaining_budgets[state.index] <= 0:
                        state.done = True
                    continue
                if has_unclosed_tool(state.assistant_text, state.cursor):
                    if not args.quiet_tools:
                        print(f"[sample:{state.index}] detected an unclosed tool tag; continuing generation", file=sys.stderr)
                    if remaining_budgets[state.index] <= 0:
                        state.done = True
                    continue
                if remaining_budgets[state.index] <= 0 or finished[row_idx] or not batch_stopped_for_tool:
                    state.done = True
    outputs = [state.assistant_text.strip() for state in states]
    base.release_resources(model, tokenizer)
    return outputs, rendered_prompts


def run_toolcaller_text_from_latent_stage(
    model_name_or_path: str,
    questions: Sequence[str],
    reflector_latents: Sequence[torch.Tensor],
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
    args: argparse.Namespace,
) -> Tuple[List[str], List[str]]:
    system_prompt = get_system_prompt("deliberation")
    model, tokenizer = base.load_agent_model_and_tokenizer(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        agent_name="deliberation_toolcaller",
    )
    embed_layer = model.get_input_embeddings()
    embed_dtype = embed_layer.weight.dtype

    prompt_segments = [
        split_prompt_ids_by_slots_with_system(
            tokenizer,
            build_toolcaller_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names),
            [DELIBERATION_REFLECTOR_SLOT],
            enable_thinking,
            system_prompt,
        )
        for idx, question in enumerate(questions)
    ]
    rendered_prompt_logs = [
        render_chat_prompt_with_system(
            tokenizer,
            build_toolcaller_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names),
            enable_thinking,
            system_prompt,
        )
        for idx, question in enumerate(questions)
    ]


    states = [ToolLoopState(index=i) for i in range(len(questions))]
    remaining_budgets = [int(max_new_tokens) for _ in range(len(questions))]
    while True:
        active = [state for state in states if (not state.done and remaining_budgets[state.index] > 0)]
        if not active:
            break
        for batch_states in [active[i : i + batch_size] for i in range(0, len(active), batch_size)]:
            batch_budget = min(remaining_budgets[state.index] for state in batch_states)
            if batch_budget <= 0:
                for state in batch_states:
                    state.done = True
                continue
            embed_seqs: List[torch.Tensor] = []
            for state in batch_states:
                idx = state.index
                seg_prefix, seg_suffix = prompt_segments[idx]
                prefix = base.token_ids_to_embeds(embed_layer, seg_prefix, device=device, dtype=embed_dtype)
                suffix = base.token_ids_to_embeds(embed_layer, seg_suffix, device=device, dtype=embed_dtype)
                latent = reflector_latents[idx].to(device=device, dtype=embed_dtype)
                if state.assistant_text:
                    assistant_ids = tokenizer(state.assistant_text, add_special_tokens=False)["input_ids"]
                    assistant = base.token_ids_to_embeds(embed_layer, assistant_ids, device=device, dtype=embed_dtype)
                    seq = torch.cat([prefix, latent, suffix, assistant], dim=0)
                else:
                    seq = torch.cat([prefix, latent, suffix], dim=0)
                embed_seqs.append(seq)

            batch_embeds, attention_mask = base.pad_left_embeds(embed_seqs, device=device)
            continuations, finished, triggered_rows, token_counts = generate_batch_from_embeds(
                tokenizer=tokenizer,
                model=model,
                batch_embeds=batch_embeds,
                attention_mask=attention_mask,
                max_new_tokens=batch_budget,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            batch_stopped_for_tool = bool(triggered_rows)
            for row_idx, state in enumerate(batch_states):
                remaining_budgets[state.index] = max(0, remaining_budgets[state.index] - int(token_counts[row_idx]))
                state.assistant_text += continuations[row_idx]
                if apply_next_tool_if_present(state, args):
                    if remaining_budgets[state.index] <= 0:
                        state.done = True
                    continue
                if has_unclosed_tool(state.assistant_text, state.cursor):
                    if not args.quiet_tools:
                        print(f"[sample:{state.index}] detected an unclosed tool tag; continuing generation", file=sys.stderr)
                    if remaining_budgets[state.index] <= 0:
                        state.done = True
                    continue
                if remaining_budgets[state.index] <= 0 or finished[row_idx] or not batch_stopped_for_tool:
                    state.done = True

    outputs = [state.assistant_text.strip() for state in states]
    base.release_resources(model, tokenizer)
    return outputs, rendered_prompt_logs


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
    system_prompt = get_system_prompt("deliberation")

    outer_rt_path, outer_tr_path = resolve_deliberation_outer_paths(
        args.outer_rt_path,
        args.outer_tr_path,
    )
    outer_rt_type = outer_tr_type = args.outer_adapter_type_fallback

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

    mas_task = infer_deliberation_task(dataset_name, is_code_eval)
    print(
        f"Running method=ours_recursive on {len(questions)} samples "
        f"(reflector={args.reflector_model_name_or_path}, toolcaller={args.toolcaller_model_name_or_path}, mas_shape={args.mas_shape})"
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
    reflector_inputs_for_log: List[str] = []
    toolcaller_inputs_for_log: List[str] = []

    if args.method == "text":
        reflector_prompts = [
            build_reflector_prompt_text(question, idx, mas_task, task_types, fn_names)
            for idx, question in enumerate(questions)
        ]
        reflector_outputs, reflector_inputs_for_log = run_text_tool_generation_stage(
            stage_name="deliberation_reflector_text",
            model_name_or_path=args.reflector_model_name_or_path,
            user_prompts=reflector_prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            args=args,
            system_prompt=system_prompt,
        )
        toolcaller_prompts = [
            build_toolcaller_prompt_text(question, idx, mas_task, task_types, fn_names, reflector_outputs[idx])
            for idx, question in enumerate(questions)
        ]
        final_outputs, toolcaller_inputs_for_log = run_text_tool_generation_stage(
            stage_name="deliberation_toolcaller_text",
            model_name_or_path=args.toolcaller_model_name_or_path,
            user_prompts=toolcaller_prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            args=args,
            system_prompt=system_prompt,
        )
        final_reflector_latents = None
    else:
        current_feedbacks: Optional[List[torch.Tensor]] = None
        final_reflector_latents: Optional[List[torch.Tensor]] = None
        final_outputs: List[str] = []
        for round_idx in range(args.num_recursive_rounds):
            reflector_inputs_for_log = [
                build_reflector_prompt_text(
                    question,
                    idx,
                    mas_task,
                    task_types,
                    fn_names,
                    feedback_text=(None if current_feedbacks is None else DELIBERATION_FEEDBACK_SLOT),
                )
                for idx, question in enumerate(questions)
            ]
            reflector_latents = run_reflector_latent_stage(
                model_name_or_path=args.reflector_model_name_or_path,
                questions=questions,
                inner_aligner_path=args.reflector_inner_aligner_path,
                outer_path=outer_rt_path,
                outer_type=outer_rt_type,
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
            final_reflector_latents = reflector_latents
            toolcaller_inputs_for_log = [
                build_toolcaller_prompt_with_slot_text(question, idx, mas_task, task_types, fn_names)
                for idx, question in enumerate(questions)
            ]
            if round_idx == args.num_recursive_rounds - 1:
                final_outputs, _ = run_toolcaller_text_from_latent_stage(
                    model_name_or_path=args.toolcaller_model_name_or_path,
                    questions=questions,
                    reflector_latents=reflector_latents,
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
                    args=args,
                )
            else:
                current_feedbacks = run_toolcaller_feedback_latent_stage(
                    model_name_or_path=args.toolcaller_model_name_or_path,
                    questions=questions,
                    reflector_latents=reflector_latents,
                    inner_aligner_path=args.toolcaller_inner_aligner_path,
                    outer_path=outer_tr_path,
                    outer_type=outer_tr_type,
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
            model_name_or_path=args.toolcaller_model_name_or_path,
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
        if args.method == "ours_recursive" and final_reflector_latents is None:
            raise RuntimeError("Missing final reflector latents for multi-rollout generation.")
        for rollout_idx in range(1, args.num_rollouts):
            rollout_seed = set_rollout_seed(rollout_idx) if args.do_sample else int(base_sample_seed + rollout_idx)
            rollout_seeds.append(rollout_seed)
            if not args.do_sample:
                outputs_by_rollout.append(list(final_outputs))
                continue
            if args.method == "text":
                reflector_prompts = [
                    build_reflector_prompt_text(question, idx, mas_task, task_types, fn_names)
                    for idx, question in enumerate(questions)
                ]
                reflector_outputs, _ = run_text_tool_generation_stage(
                    stage_name=f"deliberation_reflector_text_r{rollout_idx + 1}",
                    model_name_or_path=args.reflector_model_name_or_path,
                    user_prompts=reflector_prompts,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                    args=args,
                    system_prompt=system_prompt,
                )
                toolcaller_prompts = [
                    build_toolcaller_prompt_text(question, idx, mas_task, task_types, fn_names, reflector_outputs[idx])
                    for idx, question in enumerate(questions)
                ]
                rollout_outputs, _ = run_text_tool_generation_stage(
                    stage_name=f"deliberation_toolcaller_text_r{rollout_idx + 1}",
                    model_name_or_path=args.toolcaller_model_name_or_path,
                    user_prompts=toolcaller_prompts,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    dtype=model_dtype,
                    trust_remote_code=trust_remote_code,
                    enable_thinking=enable_thinking,
                    args=args,
                    system_prompt=system_prompt,
                )
            else:
                rollout_outputs, _ = run_toolcaller_text_from_latent_stage(
                    model_name_or_path=args.toolcaller_model_name_or_path,
                    questions=questions,
                    reflector_latents=final_reflector_latents,
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
                    args=args,
                )
            if args.ans:
                rollout_outputs, _ = base.run_answer_retry_stage(
                    model_name_or_path=args.toolcaller_model_name_or_path,
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

    method_name = f"{args.method}_deliberation"
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
                            "reflector_input": reflector_inputs_for_log[i] if i < len(reflector_inputs_for_log) else "",
                            "toolcaller_input": toolcaller_inputs_for_log[i] if i < len(toolcaller_inputs_for_log) else "",
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
                            "reflector_input": reflector_inputs_for_log[i] if i < len(reflector_inputs_for_log) else "",
                            "toolcaller_input": toolcaller_inputs_for_log[i] if i < len(toolcaller_inputs_for_log) else "",
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
