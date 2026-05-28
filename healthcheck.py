#!/usr/bin/env python3
"""
Three-level health check for the RecursiveMAS Docker container.

Level 1  — Python deps + internal imports  (always fast, no network)
Level 2  — CUDA device availability        (fast, requires GPU passthrough)
Level 3  — HuggingFace Hub reachability   (fast, requires outbound network)

Exit code 0 = all requested levels passed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
INFO = f"{BLUE}[INFO]{RESET}"

Results: List[bool] = []


def check(label: str, fn: Callable[[], str | None]) -> bool:
    try:
        detail = fn()
        suffix = f": {detail}" if detail else ""
        print(f"{PASS} {label}{suffix}")
        Results.append(True)
        return True
    except Exception as exc:
        print(f"{FAIL} {label}: {exc}")
        Results.append(False)
        return False


# ---------------------------------------------------------------------------
# Level 1 — dependencies + internal modules
# ---------------------------------------------------------------------------

def _torch_version() -> str:
    import torch
    return f"version={torch.__version__}"


def _transformers_version() -> str:
    import transformers  # noqa: F401
    return f"version={transformers.__version__}"


def _hf_hub_version() -> str:
    import huggingface_hub  # noqa: F401
    return f"version={huggingface_hub.__version__}"


def _accelerate_version() -> str:
    import accelerate  # noqa: F401
    return f"version={accelerate.__version__}"


def _internal_imports() -> str:
    repo_root = str(Path(__file__).resolve().parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from load_from_repo import STYLE_SPECS  # noqa: F401
    from modeling import INNER_ADAPTER_TYPE  # noqa: F401
    from prompts import SYSTEM_PROMPT  # noqa: F401
    return f"{len(STYLE_SPECS)} styles registered"


def level1() -> None:
    print(f"\n{BLUE}[Level 1] Python dependencies + internal modules{RESET}")
    check("torch", _torch_version)
    check("transformers", _transformers_version)
    check("huggingface_hub", _hf_hub_version)
    check("accelerate", _accelerate_version)
    check("internal modules (modeling, load_from_repo, prompts)", _internal_imports)


# ---------------------------------------------------------------------------
# Level 2 — CUDA
# ---------------------------------------------------------------------------

def _cuda_available() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False — "
            "check that --gpus / nvidia container runtime is active"
        )
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    return f"{n} device(s): {', '.join(names)}"


def _cuda_alloc() -> str:
    import torch
    t = torch.zeros(4, device="cuda")
    del t
    torch.cuda.empty_cache()
    return "small tensor allocation + free on cuda:0 OK"


def _hf_cache_status() -> str:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = hf_home / "hub"
    if not hub.exists():
        return f"cache dir {hub} does not exist yet (models not downloaded)"
    cached = [d.name for d in hub.iterdir() if d.is_dir() and d.name.startswith("models--")]
    return f"{len(cached)} model repo(s) in {hub}"


def level2() -> None:
    print(f"\n{BLUE}[Level 2] CUDA / GPU{RESET}")
    check("CUDA available", _cuda_available)
    check("CUDA tensor alloc", _cuda_alloc)
    print(f"\n{BLUE}[Level 2b] HuggingFace cache{RESET}")
    check("HF_HOME cache", _hf_cache_status)


# ---------------------------------------------------------------------------
# Level 3 — HuggingFace Hub reachability (metadata only, no weight download)
# ---------------------------------------------------------------------------

def _hf_hub_reachable() -> str:
    from huggingface_hub import list_repo_files
    # Outerlinks repo is small; listing files is a lightweight metadata call.
    files = list(list_repo_files("RecursiveMAS/Sequential-Light-Outerlinks", repo_type="model"))
    return f"HF Hub reachable — {len(files)} file(s) in Sequential-Light-Outerlinks"


def level3() -> None:
    print(f"\n{BLUE}[Level 3] HuggingFace Hub connectivity{RESET}")
    check("HF Hub reachable (metadata only)", _hf_hub_reachable)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="RecursiveMAS container health check")
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Run checks up to this level (default: 3 = all)",
    )
    args = parser.parse_args()

    print("=" * 54)
    print("  RecursiveMAS — container health check")
    print("=" * 54)

    level1()
    if args.level >= 2:
        level2()
    if args.level >= 3:
        level3()

    passed = sum(Results)
    total = len(Results)
    print("\n" + "=" * 54)
    if passed == total:
        print(f"{GREEN}All {total}/{total} checks passed.{RESET}")
        return 0
    else:
        failed = total - passed
        print(f"{RED}{failed}/{total} check(s) failed.{RESET}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
