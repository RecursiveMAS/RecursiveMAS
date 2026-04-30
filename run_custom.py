#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from load_from_repo import STYLE_SPECS


TASK_DATASET = {
    "math": "math500",
    "reasoning": "math500",
    "choice": "gpqa",
    "code": "mbppplus",
}


def _runner_helpers():
    from run import (
        build_cli_for_style,
        infer_dataset_split,
        infer_max_new_tokens,
        resolve_style_paths,
    )

    return build_cli_for_style, infer_dataset_split, infer_max_new_tokens, resolve_style_paths


def _inference_modules():
    from inference_utils import (
        inference_mas,
        inference_mas_deliberation,
        inference_mas_distill,
        inference_mas_mixture,
    )

    return inference_mas, inference_mas_deliberation, inference_mas_distill, inference_mas_mixture


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run released RecursiveMAS agents on custom question(s).",
    )
    p.add_argument("--style", required=True, choices=list(STYLE_SPECS.keys()))
    p.add_argument(
        "-q",
        "--question",
        action="append",
        default=[],
        help="Question text. Can be passed multiple times.",
    )
    p.add_argument(
        "--questions_file",
        type=str,
        default="",
        help=(
            "Text, JSON, or JSONL file containing questions. JSON may be a list of "
            "strings, a list of objects, {'questions': [...]}, or a single object."
        ),
    )
    p.add_argument(
        "--task",
        default="math",
        choices=["math", "reasoning", "choice", "code"],
        help="Prompt/evaluation family to use for custom inputs.",
    )
    p.add_argument("--output_jsonl", type=str, default="", help="Write clean outputs as JSONL.")
    p.add_argument("--pipeline_logs", type=str, default="", help="Optional file for captured internal runner logs.")
    p.add_argument("--show_pipeline_logs", action="store_true", help="Also print internal runner logs.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_seed", type=int, default=-1)
    p.add_argument("--num_recursive_rounds", type=int, default=3)
    p.add_argument("--num_rollouts", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--latent_length", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=-1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--min_p", type=float, default=-1.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--dtype", default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--outer_dtype", default="auto", choices=["float32", "float16", "bfloat16", "auto"])
    p.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    p.add_argument("--device", default=None)
    p.add_argument("--enable_thinking", type=int, default=0, choices=[0, 1])
    p.add_argument("--greedy", action="store_true", help="Disable sampling.")
    p.add_argument("--no_ans", action="store_true", help="Disable answer-format retry stage.")
    p.add_argument("--ans_max_new_tokens", type=int, default=-1)
    p.add_argument("--mbppplus_timeout_s", type=int, default=10)
    p.add_argument("--mbppplus_num_prompt_tests", type=int, default=3)
    p.add_argument("--python_timeout", type=float, default=10.0)
    p.add_argument("--python_cwd", type=str, default=".")
    p.add_argument("--result_max_chars", type=int, default=6000)
    return p


def _first_text_field(item: Mapping[str, Any]) -> str:
    for key in ("question", "query", "prompt", "text", "input"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Question object is missing one of question/query/prompt/text/input: {item}")


def _record_from_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"question": item.strip()}
    if isinstance(item, Mapping):
        record = dict(item)
        record["question"] = _first_text_field(item)
        return record
    raise ValueError(f"Unsupported question item type: {type(item)}")


def _records_from_json(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [_record_from_item(item) for item in data]
    if isinstance(data, Mapping):
        if "questions" in data:
            questions = data["questions"]
            if not isinstance(questions, list):
                raise ValueError("JSON field 'questions' must be a list.")
            return [_record_from_item(item) for item in questions]
        return [_record_from_item(data)]
    raise ValueError("JSON questions file must be a list, object, or {'questions': [...]}.")


def _records_from_text(text: str) -> List[Dict[str, Any]]:
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n")]
    records = [block for block in blocks if block]
    if len(records) <= 1:
        records = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"question": item} for item in records]


def load_question_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for question in args.question:
        if str(question).strip():
            records.append({"question": str(question).strip()})

    if args.questions_file:
        path = Path(args.questions_file)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            records.extend(_records_from_json(json.loads(text)))
        elif suffix in {".jsonl", ".ndjson"}:
            for line_no, raw_line in enumerate(text.splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(_record_from_item(json.loads(line)))
                except json.JSONDecodeError:
                    records.append({"question": line})
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
        else:
            records.extend(_records_from_text(text))

    cleaned = []
    for idx, record in enumerate(records):
        question = str(record.get("question", "")).strip()
        if question:
            out = dict(record)
            out["question"] = question
            out.setdefault("question_id", idx)
            cleaned.append(out)
    if not cleaned:
        raise ValueError("Provide at least one question with -q/--question or --questions_file.")
    return cleaned


def _custom_dataset_name(task: str) -> str:
    if task == "reasoning":
        return "custom_reasoning"
    if task == "choice":
        return "gpqa_diamond"
    if task == "code":
        return "mbppplus"
    return "math500"


def _build_metadata(records: List[Dict[str, Any]], task: str) -> Optional[List[Dict[str, Any]]]:
    if task != "code":
        return None
    metadata = []
    for record in records:
        task_type = str(record.get("task_type") or ("function" if record.get("fn_name") else "complete"))
        metadata.append(
            {
                "question_id": record.get("question_id"),
                "task_type": task_type,
                "fn_name": record.get("fn_name"),
                "eval_sample": {
                    "mode": "custom_code",
                    "fn_name": record.get("fn_name"),
                    "inputs": [],
                    "outputs": [],
                },
                "gold_answer": "",
                "build_error": "custom question has no tests",
            }
        )
    return metadata


def install_custom_dataset_patches(records: List[Dict[str, Any]], task: str) -> None:
    (
        inference_mas,
        inference_mas_deliberation,
        inference_mas_distill,
        inference_mas_mixture,
    ) = _inference_modules()
    questions = [str(record["question"]) for record in records]
    metadata = _build_metadata(records, task)
    dataset_name = _custom_dataset_name(task)

    def custom_loader(*args, return_metadata: bool = False, **kwargs):
        gold_answers = ["" for _ in questions]
        if return_metadata:
            return dataset_name, list(questions), gold_answers, metadata
        return dataset_name, list(questions), gold_answers

    def custom_compare(gold_text: str, pred_text: str, dataset_name: str = "math500"):
        return "", str(pred_text or ""), False, "", ""

    def custom_code_eval(code: str, eval_sample: Mapping[str, Any], timeout_s: int):
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": 0,
            "failed_test_index": None,
            "error_type": "NO_TESTS",
            "detail": "Custom code question was generated but not evaluated.",
        }

    inference_mas.load_eval_questions_and_answers = custom_loader
    inference_mas.compare_answers = custom_compare
    inference_mas.evaluate_generated_code = custom_code_eval
    inference_mas_mixture.evaluate_generated_code = custom_code_eval
    inference_mas_distill.evaluate_generated_code = custom_code_eval
    inference_mas_deliberation.evaluate_generated_code = custom_code_eval


def _remove_flag(cli: List[str], flag: str, takes_value: bool) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(cli):
        if cli[i] == flag:
            i += 2 if takes_value else 1
            continue
        out.append(cli[i])
        i += 1
    return out


def build_custom_cli(args: argparse.Namespace, result_jsonl: str) -> Tuple[object, List[str]]:
    build_cli_for_style, infer_dataset_split, infer_max_new_tokens, resolve_style_paths = _runner_helpers()
    dataset = TASK_DATASET[args.task]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else infer_max_new_tokens(args.style, dataset)
    paths = resolve_style_paths(args.style, dataset)
    family = str(STYLE_SPECS[args.style]["family"])
    runner_args = argparse.Namespace(
        style=args.style,
        dataset=dataset,
        dataset_split=infer_dataset_split(dataset, ""),
        seed=args.seed,
        sample_seed=args.sample_seed,
        num_recursive_rounds=args.num_recursive_rounds,
        batch_size=args.batch_size,
        latent_length=args.latent_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
    )
    module, cli = build_cli_for_style(
        args=runner_args,
        family=family,
        dataset_arg=dataset,
        dataset_split=runner_args.dataset_split,
        paths=paths,
        latent_steps=args.latent_length,
        max_new_tokens=max_new_tokens,
    )
    if args.greedy:
        cli = _remove_flag(cli, "--do_sample", takes_value=False)
    if args.no_ans or args.task == "reasoning":
        cli = _remove_flag(cli, "--ans", takes_value=False)
    cli.extend(
        [
            "--num_rollouts",
            str(args.num_rollouts),
            "--min_p",
            str(args.min_p),
            "--repetition_penalty",
            str(args.repetition_penalty),
            "--dtype",
            args.dtype,
            "--outer_dtype",
            args.outer_dtype,
            "--enable_thinking",
            str(args.enable_thinking),
            "--ans_max_new_tokens",
            str(args.ans_max_new_tokens),
            "--mbppplus_timeout_s",
            str(args.mbppplus_timeout_s),
            "--mbppplus_num_prompt_tests",
            str(args.mbppplus_num_prompt_tests),
            "--result_jsonl",
            result_jsonl,
        ]
    )
    if family == "deliberation":
        cli.extend(
            [
                "--python_timeout",
                str(args.python_timeout),
                "--python_cwd",
                args.python_cwd,
                "--result_max_chars",
                str(args.result_max_chars),
            ]
        )
    return module, cli


def run_module_capture(module: object, cli_args: List[str]) -> str:
    old_argv = sys.argv[:]
    stdout = io.StringIO()
    try:
        sys.argv = [getattr(module, "__file__", None) or getattr(module, "__name__", "module")] + cli_args
        with contextlib.redirect_stdout(stdout):
            module.main()
    except Exception:
        captured = stdout.getvalue()
        if captured.strip():
            print(captured, file=sys.stderr, end="" if captured.endswith("\n") else "\n")
        raise
    finally:
        sys.argv = old_argv
    return stdout.getvalue()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("type") != "summary":
                rows.append(row)
    return rows


def _single_output(row: Mapping[str, Any]) -> str:
    for key in ("raw_output", "pred_answer_parsed", "pred_code_parsed"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def clean_rows(
    rows: Iterable[Dict[str, Any]],
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        idx = int(row.get("sample_idx", len(out)))
        question_record = records[idx] if idx < len(records) else {"question": row.get("question", "")}
        clean: Dict[str, Any] = {
            "sample_idx": idx,
            "question_id": question_record.get("question_id", idx),
            "style": args.style,
            "task": args.task,
            "question": str(question_record.get("question", row.get("question", ""))),
        }
        if "rollouts" in row and isinstance(row["rollouts"], list):
            clean["rollouts"] = [
                {
                    "rollout_idx": rollout.get("rollout_idx", rid),
                    "sample_seed": rollout.get("sample_seed"),
                    "output": str(
                        rollout.get("raw_output")
                        or rollout.get("pred_answer_parsed")
                        or rollout.get("pred_code_parsed")
                        or ""
                    ),
                }
                for rid, rollout in enumerate(row["rollouts"])
                if isinstance(rollout, Mapping)
            ]
        else:
            clean["output"] = _single_output(row)
            if row.get("sample_seed") is not None:
                clean["sample_seed"] = row.get("sample_seed")
        out.append(clean)
    return out


def write_clean_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_clean_outputs(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        print("=" * 100)
        print(f"Sample {int(row['sample_idx']) + 1}")
        print("-" * 100)
        print("Question:")
        print(row["question"])
        if "rollouts" in row:
            for rollout in row["rollouts"]:
                print(f"\nOutput [rollout {int(rollout.get('rollout_idx', 0)) + 1}]:")
                print(rollout.get("output", ""))
        else:
            print("\nOutput:")
            print(row.get("output", ""))


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")
    parser = build_parser()
    args = parser.parse_args()
    try:
        records = load_question_records(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    install_custom_dataset_patches(records, args.task)

    with tempfile.TemporaryDirectory(prefix="recursivemas_custom_") as tmp_dir:
        raw_jsonl = str(Path(tmp_dir) / "pipeline_results.jsonl")
        module, cli = build_custom_cli(args, raw_jsonl)
        captured = run_module_capture(module, cli)
        if args.pipeline_logs:
            Path(args.pipeline_logs).write_text(captured, encoding="utf-8")
        if args.show_pipeline_logs and captured:
            print(captured, end="" if captured.endswith("\n") else "\n")

        raw_rows = read_jsonl(raw_jsonl)
        clean = clean_rows(raw_rows, records, args)
        if args.output_jsonl:
            write_clean_jsonl(args.output_jsonl, clean)
            print(f"[jsonl] wrote {len(clean)} records to {args.output_jsonl}")
        print_clean_outputs(clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
