import base64
import json
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import zlib
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional

from huggingface_hub import hf_hub_download


DATASET_REPO = "livecodebench/code_generation_lite"
DATASET_VERSION = "release_v6"
RELEASE_V6_FILES = [
    "test.jsonl",
    "test2.jsonl",
    "test3.jsonl",
    "test4.jsonl",
    "test5.jsonl",
    "test6.jsonl",
]

_LCB_KEYS = {
    "lcb",
    "livecodebench",
    "livecodebench_v6",
    "livecodebench/code_generation_lite",
    "livecodebench/code_generation_lite:release_v6",
}

_MBPPPLUS_KEYS = {
    "mbppplus",
    "mbpp+",
    "evalplus/mbppplus",
}

PYTHON_CODE_FENCE_PATTERN = re.compile(r"```python\s*(.*?)```", re.IGNORECASE | re.DOTALL)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
MBPPPLUS_ASSERTION_FN_PATTERN = re.compile(r"assertion\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MBPPPLUS_ASSERT_FN_PATTERN = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
MBPPPLUS_DEF_PATTERN = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)

IMPORT_STRING = "\n".join(
    [
        "from string import *",
        "from re import *",
        "from datetime import *",
        "from collections import *",
        "from heapq import *",
        "from bisect import *",
        "from copy import *",
        "from math import *",
        "from random import *",
        "from statistics import *",
        "from itertools import *",
        "from functools import *",
        "from operator import *",
        "from io import *",
        "from sys import *",
        "from json import *",
        "from builtins import *",
        "from typing import *",
        "import string",
        "import re",
        "import datetime",
        "import collections",
        "import heapq",
        "import bisect",
        "import copy",
        "import math",
        "import random",
        "import statistics",
        "import itertools",
        "import functools",
        "import operator",
        "import io",
        "import sys",
        "import json",
        "sys.setrecursionlimit(50000)",
    ]
)


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


def _dataset_key(name: str) -> str:
    return str(name or "").strip().lower()


def is_lcb_dataset(name: str) -> bool:
    return _dataset_key(name) in _LCB_KEYS


def is_mbppplus_dataset(name: str) -> bool:
    return _dataset_key(name) in _MBPPPLUS_KEYS


def is_code_eval_dataset(name: str) -> bool:
    return is_lcb_dataset(name) or is_mbppplus_dataset(name)


def _cut(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>..."


def sanitize_code_for_execution(code: str) -> str:
    code = str(code or "").strip()
    if not code:
        return ""

    if code.startswith("```"):
        parts = code.split("\n", 1)
        code = parts[1] if len(parts) > 1 else ""

    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().lower().startswith("```"):
            continue
        cleaned_lines.append(line)

    code = "\n".join(cleaned_lines).strip()

    if code.lower().startswith("python\n"):
        code = code.split("\n", 1)[1].strip() if "\n" in code else ""
    elif code.lower() == "python":
        code = ""

    return code


def clean_raw_output(text: str) -> str:
    text = str(text or "")
    if not text:
        return ""
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def extract_python_code(text: str) -> str:
    text = clean_raw_output(text)
    if not text:
        return ""

    blocks = PYTHON_CODE_FENCE_PATTERN.findall(text)
    if not blocks:
        return ""

    first = blocks[0].strip()
    if not first:
        return ""
    return sanitize_code_for_execution(first)


def build_code_reparse_suffix() -> str:
    return (
        "Your previous output format was invalid. "
        "Now output only final code and nothing else.\n"
        "Final Code:\n```python\n"
    )


def load_release_v6_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for filename in RELEASE_V6_FILES:
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=filename,
        )
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        records.append(row)
    return records


def load_mbppplus_records(
    split: str = "test",
    subset: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    records: List[Dict[str, Any]] = []
    for item in ds:
        if isinstance(item, Mapping):
            records.append(dict(item))
    return records


def _infer_mbppplus_fn_name(
    test_script: str,
    test_list: Optional[List[str]] = None,
    reference_code: str = "",
) -> Optional[str]:
    text = str(test_script or "")
    names = MBPPPLUS_ASSERTION_FN_PATTERN.findall(text)
    if names:
        return names[-1]

    names = MBPPPLUS_ASSERT_FN_PATTERN.findall(text)
    if names:
        return names[-1]

    for case in (test_list or []):
        c = str(case or "")
        m = MBPPPLUS_ASSERT_FN_PATTERN.search(c)
        if m:
            return m.group(1)

    defs = MBPPPLUS_DEF_PATTERN.findall(str(reference_code or ""))
    if defs:
        return defs[0]
    return None


def build_mbppplus_sample_meta(
    row: Mapping[str, Any],
    *,
    max_prompt_tests: int = 3,
) -> Dict[str, Any]:
    prompt = str(row.get("prompt", "") or "").strip()
    test_script = str(row.get("test", "") or "")
    raw_test_list = row.get("test_list", [])
    test_list = [str(x) for x in raw_test_list] if isinstance(raw_test_list, list) else []

    fn_name = _infer_mbppplus_fn_name(
        test_script=test_script,
        test_list=test_list,
        reference_code=str(row.get("code", "") or ""),
    )

    keep_n = max(0, int(max_prompt_tests))
    preview_tests = test_list[:keep_n]
    if preview_tests:
        question = (
            f"{prompt}\n\n"
            "Your solution should satisfy assertions like:\n"
            + "\n".join(preview_tests)
        )
    else:
        question = prompt

    eval_sample = {
        "fn_name": fn_name if fn_name else None,
        "mode": "mbppplus_script",
        "testtype": "python_script",
        "test_script": test_script,
        "inputs": [],
        "outputs": [],
        "num_tests": len(test_list),
    }

    build_error: Optional[str] = None
    if not prompt:
        build_error = "empty prompt"
    elif not test_script:
        build_error = "empty test script"

    return {
        "question_id": row.get("task_id"),
        "question": question,
        "task_type": "function",
        "fn_name": fn_name if fn_name else None,
        "eval_sample": eval_sample,
        "gold_answer": test_script,
        "build_error": build_error,
    }


def decode_private_test_cases(private_blob: str) -> List[Dict[str, Any]]:
    private_blob = str(private_blob or "")
    if not private_blob:
        return []

    try:
        loaded = json.loads(private_blob)
        if isinstance(loaded, list):
            return loaded
    except Exception:
        pass

    decoded = base64.b64decode(private_blob.encode("utf-8"))
    decompressed = zlib.decompress(decoded)
    unpickled = pickle.loads(decompressed)
    if isinstance(unpickled, (bytes, bytearray)):
        unpickled = unpickled.decode("utf-8")
    if isinstance(unpickled, str):
        return json.loads(unpickled)
    if isinstance(unpickled, list):
        return unpickled
    raise ValueError("Unsupported private_test_cases format after decode")


def build_eval_sample(row: Mapping[str, Any], use_private: bool) -> Dict[str, Any]:
    metadata = json.loads(str(row.get("metadata", "{}") or "{}"))
    fn_name = metadata.get("func_name")

    public_tests = json.loads(str(row.get("public_test_cases", "[]") or "[]"))
    private_tests: List[Dict[str, Any]] = []
    if use_private:
        private_raw = str(row.get("private_test_cases", "") or "")
        if private_raw:
            private_tests = decode_private_test_cases(private_raw)

    tests = public_tests + private_tests
    if not tests:
        raise ValueError("No tests found")

    return {
        "fn_name": fn_name,
        "mode": "functional" if fn_name is not None else "stdin",
        "testtype": tests[0].get("testtype", "stdin"),
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "num_public": len(public_tests),
        "num_private": len(private_tests),
    }


def build_lcb_sample_meta(
    row: Mapping[str, Any],
    *,
    use_private_tests: bool,
) -> Dict[str, Any]:
    question_content = str(row.get("question_content", "") or "").strip()
    starter_code = row.get("starter_code")
    prompt_question = question_content
    if starter_code and str(starter_code).strip():
        prompt_question += f"\n\nStarter code:\n{starter_code}"

    build_error: Optional[str] = None
    try:
        eval_sample = build_eval_sample(row, use_private=use_private_tests)
    except Exception as exc:
        build_error = repr(exc)
        eval_sample = {
            "fn_name": None,
            "mode": "stdin",
            "testtype": "stdin",
            "inputs": [],
            "outputs": [],
            "num_public": 0,
            "num_private": 0,
        }

    fn_name = eval_sample.get("fn_name")
    task_type = "function" if str(eval_sample.get("mode", "stdin")) == "functional" else "complete"
    return {
        "question_id": row.get("question_id"),
        "question": prompt_question,
        "task_type": task_type,
        "fn_name": fn_name if fn_name else None,
        "eval_sample": eval_sample,
        "gold_answer": str(row.get("solution", "") or ""),
        "build_error": build_error,
    }


def _line_to_decimals(line: str) -> Optional[List[Decimal]]:
    try:
        return [Decimal(elem) for elem in line.split()]
    except Exception:
        return None


def _normalize_lines(text: str) -> List[str]:
    text = str(text or "").strip()
    if text == "":
        return []
    return [line.strip() for line in text.split("\n")]


def compare_stdio_output(prediction: str, expected: str) -> bool:
    pred_lines = _normalize_lines(prediction)
    exp_lines = _normalize_lines(expected)
    if len(pred_lines) != len(exp_lines):
        return False

    for pred_line, exp_line in zip(pred_lines, exp_lines):
        if pred_line == exp_line:
            continue
        pred_dec = _line_to_decimals(pred_line)
        exp_dec = _line_to_decimals(exp_line)
        if pred_dec is None or exp_dec is None or pred_dec != exp_dec:
            return False
    return True


def run_stdio_tests(code: str, inputs: List[str], outputs: List[str], timeout_s: int) -> Dict[str, Any]:
    passed = 0
    total = len(inputs)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name

    try:
        for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=inp,
                    text=True,
                    capture_output=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return {
                    "all_passed": False,
                    "passed_tests": passed,
                    "total_tests": total,
                    "failed_test_index": idx,
                    "error_type": "TLE",
                    "detail": "subprocess timeout",
                }

            if proc.returncode != 0:
                return {
                    "all_passed": False,
                    "passed_tests": passed,
                    "total_tests": total,
                    "failed_test_index": idx,
                    "error_type": "RE",
                    "detail": _cut(proc.stderr, 800),
                }

            pred = proc.stdout
            if not compare_stdio_output(pred, exp):
                return {
                    "all_passed": False,
                    "passed_tests": passed,
                    "total_tests": total,
                    "failed_test_index": idx,
                    "error_type": "WA",
                    "detail": json.dumps(
                        {
                            "input": _cut(inp, 400),
                            "expected": _cut(exp, 400),
                            "prediction": _cut(pred, 400),
                        },
                        ensure_ascii=False,
                    ),
                }
            passed += 1

        return {
            "all_passed": True,
            "passed_tests": passed,
            "total_tests": total,
            "failed_test_index": None,
            "error_type": None,
            "detail": "OK",
        }
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def _compile_functional_code(code: str, timeout_s: int):
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        module = ModuleType("candidate", "")
        exec(code, module.__dict__)
        if "class Solution" in code and hasattr(module, "Solution"):
            return module.Solution()
        return module
    finally:
        signal.alarm(0)


def run_functional_tests(
    code: str,
    fn_name: str,
    inputs: List[str],
    outputs: List[str],
    timeout_s: int,
) -> Dict[str, Any]:
    passed = 0
    total = len(inputs)

    full_code = IMPORT_STRING + "\n\n" + code

    try:
        compiled = _compile_functional_code(full_code, timeout_s)
    except TimeoutException:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0,
            "error_type": "TLE",
            "detail": "compile timeout",
        }
    except Exception as exc:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0,
            "error_type": "CE/RE",
            "detail": repr(exc),
        }

    method = getattr(compiled, fn_name, None)
    if method is None:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0,
            "error_type": "NO_FUNC",
            "detail": f"Required callable `{fn_name}` not found",
        }

    signal.signal(signal.SIGALRM, _timeout_handler)

    for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
        try:
            args = [json.loads(line) for line in inp.split("\n")]
            expected = json.loads(exp)
        except Exception as exc:
            return {
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": total,
                "failed_test_index": idx,
                "error_type": "BAD_TEST_FORMAT",
                "detail": repr(exc),
            }

        try:
            signal.alarm(timeout_s)
            pred = method(*args)
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            return {
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": total,
                "failed_test_index": idx,
                "error_type": "TLE",
                "detail": "function timeout",
            }
        except Exception as exc:
            signal.alarm(0)
            return {
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": total,
                "failed_test_index": idx,
                "error_type": "RE",
                "detail": repr(exc),
            }

        if isinstance(pred, tuple):
            pred = list(pred)

        if pred != expected:
            return {
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": total,
                "failed_test_index": idx,
                "error_type": "WA",
                "detail": json.dumps(
                    {
                        "input_args": _cut(str(args), 400),
                        "expected": _cut(str(expected), 400),
                        "prediction": _cut(str(pred), 400),
                    },
                    ensure_ascii=False,
                ),
            }

        passed += 1

    return {
        "all_passed": True,
        "passed_tests": passed,
        "total_tests": total,
        "failed_test_index": None,
        "error_type": None,
        "detail": "OK",
    }




def run_functional_tests_subprocess(
    code: str,
    fn_name: str,
    inputs: List[str],
    outputs: List[str],
    timeout_s: int,
) -> Dict[str, Any]:
    total = len(inputs)

    runner = r"""
import json
import signal
import sys
from types import ModuleType


class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out")


def _cut(text, max_chars):
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>..."


def main():
    payload = json.loads(sys.stdin.read())
    code = str(payload.get("code", "") or "")
    fn_name = str(payload.get("fn_name", "") or "")
    inputs = list(payload.get("inputs", []) or [])
    outputs = list(payload.get("outputs", []) or [])
    timeout_s = int(payload.get("timeout_s", 6))
    import_string = str(payload.get("import_string", "") or "")

    full_code = import_string + "\n\n" + code

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
        module = ModuleType("candidate", "")
        exec(full_code, module.__dict__)
        signal.alarm(0)
        compiled = module.Solution() if ("class Solution" in full_code and hasattr(module, "Solution")) else module
    except TimeoutException:
        signal.alarm(0)
        print(json.dumps({
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": len(inputs),
            "failed_test_index": 0 if len(inputs) > 0 else None,
            "error_type": "TLE",
            "detail": "compile timeout",
        }, ensure_ascii=False))
        return
    except Exception as exc:
        signal.alarm(0)
        print(json.dumps({
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": len(inputs),
            "failed_test_index": 0 if len(inputs) > 0 else None,
            "error_type": "CE/RE",
            "detail": repr(exc),
        }, ensure_ascii=False))
        return

    method = getattr(compiled, fn_name, None)
    if method is None:
        print(json.dumps({
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": len(inputs),
            "failed_test_index": 0 if len(inputs) > 0 else None,
            "error_type": "NO_FUNC",
            "detail": f"Required callable `{fn_name}` not found",
        }, ensure_ascii=False))
        return

    passed = 0
    signal.signal(signal.SIGALRM, _timeout_handler)

    for idx, (inp, exp) in enumerate(zip(inputs, outputs)):
        try:
            args = [json.loads(line) for line in str(inp).split("\n") if str(line).strip() != ""]
            expected = json.loads(exp)
        except Exception as exc:
            print(json.dumps({
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": len(inputs),
                "failed_test_index": idx,
                "error_type": "BAD_TEST_FORMAT",
                "detail": repr(exc),
            }, ensure_ascii=False))
            return

        try:
            signal.alarm(timeout_s)
            pred = method(*args)
            signal.alarm(0)
        except TimeoutException:
            signal.alarm(0)
            print(json.dumps({
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": len(inputs),
                "failed_test_index": idx,
                "error_type": "TLE",
                "detail": "function timeout",
            }, ensure_ascii=False))
            return
        except Exception as exc:
            signal.alarm(0)
            print(json.dumps({
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": len(inputs),
                "failed_test_index": idx,
                "error_type": "RE",
                "detail": repr(exc),
            }, ensure_ascii=False))
            return

        if isinstance(pred, tuple):
            pred = list(pred)

        if pred != expected:
            print(json.dumps({
                "all_passed": False,
                "passed_tests": passed,
                "total_tests": len(inputs),
                "failed_test_index": idx,
                "error_type": "WA",
                "detail": json.dumps({
                    "input_args": _cut(str(args), 400),
                    "expected": _cut(str(expected), 400),
                    "prediction": _cut(str(pred), 400),
                }, ensure_ascii=False),
            }, ensure_ascii=False))
            return

        passed += 1

    print(json.dumps({
        "all_passed": True,
        "passed_tests": passed,
        "total_tests": len(inputs),
        "failed_test_index": None,
        "error_type": None,
        "detail": "OK",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
"""

    payload = {
        "code": str(code or ""),
        "fn_name": str(fn_name or ""),
        "inputs": inputs,
        "outputs": outputs,
        "timeout_s": int(timeout_s),
        "import_string": IMPORT_STRING,
    }

    # Keep eval isolated from the main inference process to avoid hard crashes
    # from adversarial/unsafe functional code (e.g., os._exit / ctypes faults).
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=max(10, int(timeout_s) * max(1, total) + 5),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0 if total > 0 else None,
            "error_type": "TLE",
            "detail": "functional-eval subprocess timeout",
        }

    if proc.returncode != 0:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0 if total > 0 else None,
            "error_type": "EVAL_CRASH",
            "detail": _cut(proc.stderr, 800),
        }

    try:
        parsed = json.loads(proc.stdout.strip())
    except Exception as exc:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0 if total > 0 else None,
            "error_type": "EVAL_BAD_OUTPUT",
            "detail": _cut(f"parse_error={repr(exc)} stdout={proc.stdout}", 800),
        }

    if not isinstance(parsed, dict):
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0 if total > 0 else None,
            "error_type": "EVAL_BAD_OUTPUT",
            "detail": _cut(str(parsed), 800),
        }

    return parsed

def run_mbppplus_script_tests(
    code: str,
    test_script: str,
    timeout_s: int,
    num_tests: int = 1,
) -> Dict[str, Any]:
    code = str(code or "").strip()
    test_script = str(test_script or "").strip()
    total = int(num_tests) if int(num_tests) > 0 else 1

    if not code:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total,
            "failed_test_index": 0,
            "error_type": "EMPTY_CODE",
            "detail": "No code after parsing",
        }
    if not test_script:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": 0,
            "failed_test_index": None,
            "error_type": "NO_TESTS",
            "detail": "No executable tests available",
        }

    full_code = IMPORT_STRING + "\n\n" + code + "\n\n" + test_script + "\n"

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                text=True,
                capture_output=True,
                timeout=max(1, int(timeout_s)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "all_passed": False,
                "passed_tests": 0,
                "total_tests": total,
                "failed_test_index": 0,
                "error_type": "TLE",
                "detail": f"Execution exceeded {timeout_s} seconds",
            }

        if proc.returncode != 0:
            return {
                "all_passed": False,
                "passed_tests": 0,
                "total_tests": total,
                "failed_test_index": 0,
                "error_type": "RE",
                "detail": _cut(proc.stderr or proc.stdout, 1000),
            }

        return {
            "all_passed": True,
            "passed_tests": total,
            "total_tests": total,
            "failed_test_index": None,
            "error_type": None,
            "detail": "OK",
        }
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


def evaluate_generated_code(code: str, eval_sample: Mapping[str, Any], timeout_s: int) -> Dict[str, Any]:
    mode = str(eval_sample.get("mode", "stdin"))
    inputs = list(eval_sample.get("inputs", []) or [])
    total_tests = len(inputs)

    if mode == "mbppplus_script":
        return run_mbppplus_script_tests(
            code=code,
            test_script=str(eval_sample.get("test_script", "") or ""),
            timeout_s=timeout_s,
            num_tests=int(eval_sample.get("num_tests", 1) or 1),
        )

    if not str(code or "").strip():
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": total_tests,
            "failed_test_index": 0 if total_tests > 0 else None,
            "error_type": "EMPTY_CODE",
            "detail": "No code after parsing",
        }
    if total_tests <= 0:
        return {
            "all_passed": False,
            "passed_tests": 0,
            "total_tests": 0,
            "failed_test_index": None,
            "error_type": "NO_TESTS",
            "detail": "No executable tests available",
        }

    if mode == "functional":
        fn_name = str(eval_sample.get("fn_name", "") or "")
        outputs = list(eval_sample.get("outputs", []) or [])
        isolated = str(os.environ.get("LCB_FUNCTIONAL_ISOLATED", "1")).strip().lower() not in {"0", "false", "no"}
        if isolated:
            return run_functional_tests_subprocess(
                code=code,
                fn_name=fn_name,
                inputs=inputs,
                outputs=outputs,
                timeout_s=timeout_s,
            )
        return run_functional_tests(
            code=code,
            fn_name=fn_name,
            inputs=inputs,
            outputs=outputs,
            timeout_s=timeout_s,
        )

    return run_stdio_tests(
        code=code,
        inputs=inputs,
        outputs=list(eval_sample.get("outputs", []) or []),
        timeout_s=timeout_s,
    )


    if str(eval_sample.get("mode", "stdin")) == "functional":
        fn_name = str(eval_sample.get("fn_name", "") or "")
        inputs = list(eval_sample.get("inputs", []) or [])
        outputs = list(eval_sample.get("outputs", []) or [])
        isolated = str(os.environ.get("LCB_FUNCTIONAL_ISOLATED", "1")).strip().lower() not in {"0", "false", "no"}
        if isolated:
            return run_functional_tests_subprocess(
                code=code,
                fn_name=fn_name,
                inputs=inputs,
                outputs=outputs,
                timeout_s=timeout_s,
            )
        return run_functional_tests(
            code=code,
            fn_name=fn_name,
            inputs=inputs,
            outputs=outputs,
            timeout_s=timeout_s,
        )

    return run_stdio_tests(
        code=code,
        inputs=list(eval_sample.get("inputs", []) or []),
        outputs=list(eval_sample.get("outputs", []) or []),
        timeout_s=timeout_s,
    )
