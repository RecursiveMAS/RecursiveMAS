#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from hf_resolver import (
    resolve_inner_adapter,
    resolve_medqa_dataset_arg,
    resolve_outer_paths,
    snapshot_repo,
    task_for_inner_repo,
)
from load_from_repo import DATASET_DEFAULT_SPLIT, STYLE_SPECS
from inference_utils import (
    inference_mas,
    inference_mas_deliberation,
    inference_mas_distill,
    inference_mas_mixture,
)

GPQA_DEFAULT_CHOICE_OLD_PROMPT = 2
MBPPPLUS_TEMPERATURE = 0.2


class RunCapture:
    def __init__(self) -> None:
        self.stdout = io.StringIO()

    def get_text(self) -> str:
        return self.stdout.getvalue()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Release inference runner for RecursiveMAS HF checkpoints.")
    p.add_argument("--style", required=True, choices=list(STYLE_SPECS.keys()))
    p.add_argument("--dataset", required=True, default="math500", choices=["math500", "medqa", "gpqa", "mbppplus", "aime25", "aime26", "livecodebench", "bamboogle", "hotpotqa"])
    p.add_argument("--dataset_split", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_seed", type=int, default=-1)
    p.add_argument("--num_recursive_rounds", type=int, default=3)
    p.add_argument("--num_samples", type=int, default=-1, help="Limit number of eval questions (-1 = all). Useful for quick canaries.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--latent_length", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--num_rollouts", type=int, default=1, help="Stochastic rollouts for pass@k. AIME defaults to 10 (pass@10) if not set.")
    p.add_argument("--lcb_use_private_tests", type=int, default=1, choices=[0, 1], help="LCB: 1 = public + private (hidden) tests, 0 = public only.")
    # search-QA (deliberation + Tavily): keys file + per-sample output for later LLM-judge
    p.add_argument("--tavily_keys_file", default="", help="File of Tavily keys (deliberation search datasets).")
    p.add_argument("--tavily_sentinel_file", default="", help="Sentinel path written when all Tavily keys exhausted.")
    p.add_argument("--result_jsonl", default="", help="Per-sample output (question/gold/pred/raw_output) for LLM-judge.")
    p.add_argument("--trust_remote_code", type=int, default=1, choices=[0, 1])
    p.add_argument("--device", default=None)
    p.add_argument(
        "--ckpt_override",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Override a role/outer repo with a locally-trained checkpoint dir (no HF download), "
             "e.g. --ckpt_override solver=/path/trained_solver --ckpt_override outer=/path/trained_outer. "
             "Repeatable. Keys are the style's repo keys: sequential {planner,critic,solver,outer}; "
             "mixture {math,code,science,summarizer,outer}; distillation {expert,learner,outer}; "
             "deliberation {reflector,toolcaller,outer}.",
    )
    return p


def is_search_dataset(dataset: str) -> bool:
    return dataset.strip().lower() in {"bamboogle", "hotpotqa"}


def validate_style_dataset(args: argparse.Namespace) -> None:
    """Fail fast on (style, dataset) pairs that would silently produce a meaningless metric.

    The search-QA datasets (``bamboogle``/``hotpotqa``) are answered with the Deliberation
    pipeline only: the Tool-Caller issues real web searches and the open-ended answers are
    graded by an LLM judge. Running them under any other style would fall through to
    string-match scoring and report a number that does not reflect the task, so we reject
    the combination up front instead of returning a wrong result.
    """
    if is_search_dataset(args.dataset):
        if args.style != "deliberation":
            raise SystemExit(
                f"[error] dataset '{args.dataset}' is a search-QA task supported only by "
                f"--style deliberation (got --style {args.style}). "
                f"Re-run with --style deliberation, or choose a non-search dataset."
            )
        if not args.tavily_keys_file:
            raise SystemExit(
                f"[error] dataset '{args.dataset}' needs web search: pass "
                f"--tavily_keys_file <path> (a file of Tavily keys) and set the "
                f"API_KEY / API_BASE_URL / API_MODEL judge env vars. See inference/README.md."
            )


def infer_dataset_split(dataset: str, explicit: str) -> str:
    if explicit:
        return explicit
    return DATASET_DEFAULT_SPLIT.get(dataset.lower(), "test")


def infer_max_new_tokens(style: str, dataset: str) -> int:
    ds = dataset.lower()
    if ds == "math500":
        if style == "sequential_light":
            return 1000
        return 2000
    if ds in {"aime25", "aime26"}:
        # AIME pass@10: light family uses 8k, all larger families use 16k.
        return 8192 if style == "sequential_light" else 16000
    if ds == "lcb":
        # LiveCodeBench: 4096 generation tokens.
        return 4096
    if ds in {"bamboogle", "hotpotqa"}:
        # Search-QA (deliberation + Tavily): 4000 generation tokens.
        return 4000
    return 4000


def infer_temperature(dataset: str, explicit: float) -> float:
    if dataset.lower() == "mbppplus":
        return MBPPPLUS_TEMPERATURE
    # lcb uses the recommended temperature, or --temperature if explicitly provided.
    return explicit


def _has_cli_flag(flag: str) -> bool:
    # Matches both "--flag value" and "--flag=value" forms.
    prefix = flag + "="
    return any(arg == flag or arg.startswith(prefix) for arg in sys.argv[1:])


def apply_recommended_settings(args: argparse.Namespace) -> None:
    recommended = inference_mas.get_release_recommended_settings(args.style, args.dataset)
    if recommended is None:
        return

    field_to_flag = {
        "seed": "--seed",
        "batch_size": "--batch_size",
        "latent_length": "--latent_length",
        "num_recursive_rounds": "--num_recursive_rounds",
        "temperature": "--temperature",
    }
    int_fields = {"seed", "batch_size", "latent_length", "num_recursive_rounds"}
    mismatches: List[str] = []
    for field_name, recommended_value in recommended.items():
        flag = field_to_flag.get(field_name)
        if flag is None:
            continue
        recommended_value = int(recommended_value) if field_name in int_fields else float(recommended_value)
        explicit = _has_cli_flag(flag)
        if not explicit:
            setattr(args, field_name, recommended_value)
            continue
        if getattr(args, field_name) != recommended_value:
            mismatches.append(f"{field_name}={recommended_value}")

    if mismatches:
        joined = ", ".join(mismatches)
        print(
            f"[note] We recommend to use provided settings to run "
            f"{args.style} on {args.dataset}: {joined}"
        )


def resolve_style_paths(style: str, dataset: str, repo_overrides: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
    spec = STYLE_SPECS[style]
    repos = dict(spec["repos"])
    if repo_overrides:
        repos.update(repo_overrides)
    task = task_for_inner_repo(dataset)
    out: Dict[str, Path] = {}

    def materialize(key: str) -> Path:
        return snapshot_repo(str(repos[key]))

    family = str(spec["family"])
    if family == "sequential":
        for key in ["planner", "critic", "solver", "outer"]:
            out[key] = materialize(key)
        out["planner_adapter"] = resolve_inner_adapter(out["planner"], task)
        out["critic_adapter"] = resolve_inner_adapter(out["critic"], task)
        out["solver_adapter"] = resolve_inner_adapter(out["solver"], task)
        outer_paths = resolve_outer_paths(out["outer"], task=task)
        out["outer_12"] = outer_paths["outer_12"]
        out["outer_23"] = outer_paths["outer_23"]
        out["outer_31"] = outer_paths["outer_31"]
        return out

    if family == "mixture":
        for key in ["math", "code", "science", "summarizer", "outer"]:
            out[key] = materialize(key)
        out["math_adapter"] = resolve_inner_adapter(out["math"], None)
        out["code_adapter"] = resolve_inner_adapter(out["code"], None)
        out["science_adapter"] = resolve_inner_adapter(out["science"], None)
        out["summarizer_adapter"] = resolve_inner_adapter(out["summarizer"], None)
        outer_paths = resolve_outer_paths(out["outer"], task=None)
        for key in ["outer_1s", "outer_2s", "outer_3s", "outer_s1", "outer_s2", "outer_s3"]:
            out[key] = outer_paths[key]
        return out

    if family == "distillation":
        for key in ["expert", "learner", "outer"]:
            out[key] = materialize(key)
        out["expert_adapter"] = resolve_inner_adapter(out["expert"], task)
        out["learner_adapter"] = resolve_inner_adapter(out["learner"], task)
        outer_paths = resolve_outer_paths(out["outer"], task=task)
        out["outer_el"] = outer_paths["outer_el"]
        out["outer_le"] = outer_paths["outer_le"]
        return out

    if family == "deliberation":
        for key in ["reflector", "toolcaller", "outer"]:
            out[key] = materialize(key)
        out["reflector_adapter"] = resolve_inner_adapter(out["reflector"], None)
        out["toolcaller_adapter"] = resolve_inner_adapter(out["toolcaller"], None)
        outer_paths = resolve_outer_paths(out["outer"], task=None)
        out["outer_rt"] = outer_paths["outer_rt"]
        out["outer_tr"] = outer_paths["outer_tr"]
        return out

    raise ValueError(f"Unsupported style family: {family}")


def build_common_cli(args: argparse.Namespace, dataset_arg: str, dataset_split: str, latent_steps: int, max_new_tokens: int) -> List[str]:
    temperature = infer_temperature(args.dataset, args.temperature)
    out = [
        "--dataset", dataset_arg,
        "--dataset_split", dataset_split,
        "--num_samples", str(args.num_samples),
        "--seed", str(args.seed),
        "--sample_seed", str(args.sample_seed),
        "--num_rollouts", str(args.num_rollouts),
        "--num_recursive_rounds", str(args.num_recursive_rounds),
        "--batch_size", str(args.batch_size),
        "--latent_steps", str(latent_steps),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top_p", str(args.top_p),
        "--top_k", str(args.top_k),
        "--ans_max_new_tokens", "-1",
        "--mbppplus_timeout_s", "10",
        "--mbppplus_num_prompt_tests", "3",
        "--dtype", "auto",
        "--outer_dtype", "auto",
        "--trust_remote_code", str(args.trust_remote_code),
        "--enable_thinking", "0",
    ]
    if args.dataset.lower() == "lcb":
        # private=1 => public + private (hidden) tests; 0 => public only. 6s/test.
        out.extend(["--lcb_use_private_tests", str(args.lcb_use_private_tests), "--lcb_timeout_s", "6"])
    if args.device is not None:
        out.extend(["--device", str(args.device)])
    out.append("--do_sample")
    out.append("--ans")
    return out


def extract_metric(output_text: str) -> Tuple[str, float]:
    passk = re.findall(r"pass@(\d+)=([0-9]+(?:\.[0-9]+)?)%", output_text)
    if passk:
        k, val = passk[-1]
        return f"pass@{k}", float(val)
    matches = re.findall(r"accuracy=([0-9]+(?:\.[0-9]+)?)%", output_text)
    if matches:
        return "accuracy", float(matches[-1])
    raise RuntimeError("Failed to parse final metric from inference output.")


def run_module(module, cli_args: List[str]) -> Tuple[str, float, str]:
    old_argv = sys.argv[:]
    capture = RunCapture()
    try:
        sys.argv = [module.__file__ or module.__name__] + cli_args
        with contextlib.redirect_stdout(capture.stdout):
            module.main()
    except Exception:
        captured = capture.get_text()
        if captured.strip():
            print(captured, file=sys.stderr, end="" if captured.endswith("\n") else "\n")
        raise
    finally:
        sys.argv = old_argv
    text = capture.get_text()
    metric_name, metric_value = extract_metric(text)
    return metric_name, metric_value, text


def build_cli_for_style(
    args: argparse.Namespace,
    family: str,
    dataset_arg: str,
    dataset_split: str,
    paths: Dict[str, Path],
    latent_steps: int,
    max_new_tokens: int,
) -> Tuple[object, List[str]]:
    common = build_common_cli(args, dataset_arg=dataset_arg, dataset_split=dataset_split, latent_steps=latent_steps, max_new_tokens=max_new_tokens)

    if family == "sequential":
        choice_old_prompt = GPQA_DEFAULT_CHOICE_OLD_PROMPT if args.dataset.lower() == "gpqa" else 0
        cli = [
            "--mas_shape", "chain",
            "--agent1_model_name_or_path", str(paths["planner"]),
            "--agent2_model_name_or_path", str(paths["critic"]),
            "--agent3_model_name_or_path", str(paths["solver"]),
            "--agent1_inner_aligner_path", str(paths["planner_adapter"]),
            "--agent2_inner_aligner_path", str(paths["critic_adapter"]),
            "--agent3_inner_aligner_path", str(paths["solver_adapter"]),
            "--outer_12_path", str(paths["outer_12"]),
            "--outer_23_path", str(paths["outer_23"]),
            "--outer_31_path", str(paths["outer_31"]),
            "--choice_old_prompt", str(choice_old_prompt),
            "--solver_pre_question", "0",
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
        ] + common
        return inference_mas, cli

    if family == "mixture":
        cli = [
            "--mas_shape", "hie",
            "--agent1_model_name_or_path", str(paths["math"]),
            "--agent2_model_name_or_path", str(paths["code"]),
            "--agent3_model_name_or_path", str(paths["science"]),
            "--agent4_model_name_or_path", str(paths["summarizer"]),
            "--agent1_inner_aligner_path", str(paths["math_adapter"]),
            "--agent2_inner_aligner_path", str(paths["code_adapter"]),
            "--agent3_inner_aligner_path", str(paths["science_adapter"]),
            "--agent4_inner_aligner_path", str(paths["summarizer_adapter"]),
            "--outer_1s_path", str(paths["outer_1s"]),
            "--outer_2s_path", str(paths["outer_2s"]),
            "--outer_3s_path", str(paths["outer_3s"]),
            "--outer_s1_path", str(paths["outer_s1"]),
            "--outer_s2_path", str(paths["outer_s2"]),
            "--outer_s3_path", str(paths["outer_s3"]),
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
        ] + common
        return inference_mas_mixture, cli

    if family == "distillation":
        cli = [
            "--mas_shape", "distill",
            "--expert_model_name_or_path", str(paths["expert"]),
            "--learner_model_name_or_path", str(paths["learner"]),
            "--expert_inner_aligner_path", str(paths["expert_adapter"]),
            "--learner_inner_aligner_path", str(paths["learner_adapter"]),
            "--outer_el_path", str(paths["outer_el"]),
            "--outer_le_path", str(paths["outer_le"]),
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
        ] + common
        return inference_mas_distill, cli

    if family == "deliberation":
        cli = [
            "--mas_shape", "deliberation",
            "--reflector_model_name_or_path", str(paths["reflector"]),
            "--toolcaller_model_name_or_path", str(paths["toolcaller"]),
            "--reflector_inner_aligner_path", str(paths["reflector_adapter"]),
            "--toolcaller_inner_aligner_path", str(paths["toolcaller_adapter"]),
            "--outer_rt_path", str(paths["outer_rt"]),
            "--outer_tr_path", str(paths["outer_tr"]),
            "--inner_adapter_type_fallback", "ln_res_adapter",
            "--outer_adapter_type_fallback", "outer_ln_res_adapter",
            "--max_tool_rounds", "5",
            "--python_timeout", "10.0",
            "--python_cwd", ".",
            "--result_max_chars", "6000",
        ] + common
        cli.append("--quiet_tools")
        # search-QA datasets: enable real Tavily (multi-key rotation) + per-sample jsonl for LLM-judge
        if is_search_dataset(args.dataset) and args.tavily_keys_file:
            cli += [
                "--search_provider", "tavily",
                "--tavily_keys_file", str(args.tavily_keys_file),
                "--tavily_exhausted_file", "/tmp/tavily_exhausted.txt",
                "--tavily_search_depth", "advanced",
                "--tavily_max_results", "4",
            ]
            if args.tavily_sentinel_file:
                cli += ["--tavily_sentinel_file", str(args.tavily_sentinel_file)]
        if args.result_jsonl:
            cli += ["--result_jsonl", str(args.result_jsonl)]
        return inference_mas_deliberation, cli

    raise ValueError(f"Unsupported style family: {family}")


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")

    args = build_parser().parse_args()
    if args.dataset.lower() == "livecodebench":
        args.dataset = "lcb"  # internal dataset key
    validate_style_dataset(args)
    apply_recommended_settings(args)
    if args.dataset.lower() in {"aime25", "aime26"} and not _has_cli_flag("--num_rollouts"):
        args.num_rollouts = 10  # AIME defaults to pass@10
    if args.dataset.lower() == "lcb" and not _has_cli_flag("--temperature"):
        # LCB default temperature; only when the recommended table didn't already set one.
        recommended = inference_mas.get_release_recommended_settings(args.style, args.dataset)
        if not (recommended and "temperature" in recommended):
            args.temperature = 0.2
    repo_root = Path(__file__).resolve().parent
    dataset_arg = resolve_medqa_dataset_arg(args.dataset, repo_root)
    dataset_split = infer_dataset_split(args.dataset, args.dataset_split)
    repo_overrides: Dict[str, str] = {}
    for item in args.ckpt_override:
        if "=" not in item:
            raise ValueError(f"--ckpt_override must be KEY=PATH, got: {item!r}")
        key, path = item.split("=", 1)
        repo_overrides[key.strip()] = path.strip()
    paths = resolve_style_paths(args.style, args.dataset, repo_overrides=repo_overrides)
    family = str(STYLE_SPECS[args.style]["family"])
    max_new_tokens = infer_max_new_tokens(args.style, args.dataset)
    print(f"[run] style={args.style} dataset={args.dataset} rounds={args.num_recursive_rounds} batch_size={args.batch_size} latent_length={args.latent_length} max_new_tokens={max_new_tokens}")
    module, cli = build_cli_for_style(
        args=args,
        family=family,
        dataset_arg=dataset_arg,
        dataset_split=dataset_split,
        paths=paths,
        latent_steps=args.latent_length,
        max_new_tokens=max_new_tokens,
    )
    metric_name, metric_value, _ = run_module(module, cli)
    print(f"[result] {metric_name}={metric_value:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
