#!/usr/bin/env python3
"""
Gradio web UI for RecursiveMAS.

Models are loaded once on first use and kept warm in VRAM.
Style switching evicts the old models before loading the new set.

Launch:
    python serve.py                          # http://0.0.0.0:7860
    python serve.py --port 8080 --share      # public Gradio link
"""
from __future__ import annotations

import contextlib
import gc
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

# Load .env (TAVILY_API_KEY needed for deliberation style)
_env_path = THIS_DIR / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")

# ── Model cache — patch BEFORE importing any pipeline module ─────────────────
# All submodules import inference_mas as `base` and call base.load_agent_model_and_tokenizer
# at call time (not import time), so patching the module attribute propagates correctly.

import inference_utils.inference_mas as _base  # noqa: E402

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_CURRENT_STYLE: Optional[str] = None
_ORIG_LOAD = _base.load_agent_model_and_tokenizer


def _cached_load(model_name_or_path, device, dtype, trust_remote_code, agent_name):
    from modeling import resolve_local_pretrained_path
    key = resolve_local_pretrained_path(str(model_name_or_path))
    if key not in _MODEL_CACHE:
        print(f"[serve] loading {agent_name} into VRAM …", flush=True)
        model, tok = _ORIG_LOAD(model_name_or_path, device, dtype, trust_remote_code, agent_name)
        _MODEL_CACHE[key] = (model, tok)
        print(f"[serve] {agent_name} cached.", flush=True)
    else:
        print(f"[serve] cache hit: {agent_name}", flush=True)
    return _MODEL_CACHE[key]


def _noop_release(*_):
    pass  # keep models warm between requests


_base.load_agent_model_and_tokenizer = _cached_load
_base.release_resources = _noop_release

# ── Import run.py utilities (safe now that patching is done) ─────────────────
from run import (  # noqa: E402
    STYLE_SPECS,
    build_cli_for_style,
    infer_max_new_tokens,
    resolve_style_paths,
)
import gradio as gr  # noqa: E402


# ── VRAM management ───────────────────────────────────────────────────────────

def _evict_cache() -> None:
    global _MODEL_CACHE
    for model, tok in _MODEL_CACHE.values():
        del model, tok
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[serve] VRAM cache evicted.", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_single_question(
    style: str,
    question: str,
    device: str,
    num_rounds: int,
    latent_steps: int,
) -> Tuple[str, str]:
    """
    Run the MAS pipeline on one question.
    Returns (captured_stdout, parsed_answer_string).
    """
    import argparse as _ap

    # Write question to a temporary medqa-format JSON
    tmp_json = tempfile.mktemp(suffix=".json")
    result_jsonl = tempfile.mktemp(suffix=".jsonl")
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump([{"question": question, "answer": ""}], f)

    try:
        # Resolve HF model paths (cached locally after first download)
        paths = resolve_style_paths(style, "math500")
        family = str(STYLE_SPECS[style]["family"])
        max_new_tokens = infer_max_new_tokens(style, "math500")

        # Minimal args namespace consumed by build_cli_for_style
        fake_args = _ap.Namespace(
            dataset="math500",
            dataset_split="",
            num_recursive_rounds=num_rounds,
            batch_size=1,
            latent_length=latent_steps,
            temperature=0.6,
            top_p=0.95,
            top_k=-1,
            trust_remote_code=1,
            device=device,
        )

        module, cli_args = build_cli_for_style(
            args=fake_args,
            family=family,
            dataset_arg=tmp_json,   # actual dataset = temp JSON (single question)
            dataset_split="train",
            paths=paths,
            latent_steps=latent_steps,
            max_new_tokens=max_new_tokens,
        )
        cli_args += ["--result_jsonl", result_jsonl, "--num_samples", "-1"]

        # Run with stdout captured
        captured = io.StringIO()
        old_argv, sys.argv = sys.argv[:], [module.__file__ or "serve"] + cli_args
        try:
            with contextlib.redirect_stdout(captured):
                module.main()
        finally:
            sys.argv = old_argv

        stdout = captured.getvalue()

        # Read structured result
        parsed = ""
        if os.path.isfile(result_jsonl):
            with open(result_jsonl, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    parsed = str(
                        rec.get("pred_answer_parsed")
                        or rec.get("pred_code_parsed")
                        or ""
                    )
                    break

        return stdout, parsed

    finally:
        for p in (tmp_json, result_jsonl):
            try:
                os.unlink(p)
            except OSError:
                pass


def _parse_agent_outputs(stdout: str) -> Dict[str, str]:
    """Extract Agent1 / Agent2 / Agent3 text outputs from captured stdout."""
    # The pipeline prints labelled sections like "3) Agent1 Output:" etc.
    MARKERS = {
        "Agent1 Output:": "agent1",
        "Agent2 Output:": "agent2",
        "Agent3 Output:": "agent3",
    }
    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    buf: List[str] = []

    for line in stdout.splitlines():
        matched = False
        for marker, key in MARKERS.items():
            if marker in line:
                if current_key and buf:
                    sections[current_key] = "\n".join(buf).strip()
                current_key = key
                buf = []
                matched = True
                break
        if matched:
            continue
        if current_key:
            # Section ends at a separator or a new numbered label
            if line.startswith("=" * 20) or (
                len(line) > 2 and line[0].isdigit() and line[1] in (").", ") ")
            ):
                sections[current_key] = "\n".join(buf).strip()
                current_key = None
                buf = []
            else:
                buf.append(line)

    if current_key and buf:
        sections[current_key] = "\n".join(buf).strip()

    return sections


def _build_reply(style: str, parsed: str, stdout: str) -> str:
    agents = _parse_agent_outputs(stdout)
    parts: List[str] = [f"**Style:** `{style}`"]

    if parsed:
        parts.append(f"\n**Answer:** `{parsed}`")

    solver_text = agents.get("agent3", "")
    if solver_text:
        parts.append("\n---\n**Solver output:**\n" + solver_text)

    # Wrap intermediate agent outputs in collapsible details
    for key, label in [("agent1", "Planner"), ("agent2", "Critic / Refiner")]:
        text = agents.get(key, "")
        if text:
            parts.append(
                f"\n<details><summary>{label} output</summary>\n\n{text}\n\n</details>"
            )

    return "\n".join(parts) if len(parts) > 1 else (parsed or stdout[:3000])


# ── Gradio event handler ──────────────────────────────────────────────────────

def respond(
    message: str,
    history: List[Dict],
    style: str,
    num_rounds: int,
    latent_steps: int,
    device: str,
) -> Tuple[List[Dict], List[Dict], str]:
    global _CURRENT_STYLE

    if not message.strip():
        return history, history, ""

    if _CURRENT_STYLE != style and _MODEL_CACHE:
        _evict_cache()
    _CURRENT_STYLE = style

    try:
        stdout, parsed = _run_single_question(style, message, device, num_rounds, latent_steps)
        reply = _build_reply(style, parsed, stdout)
    except Exception as exc:
        reply = f"❌ Error during inference:\n```\n{exc}\n```"

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history, new_history, ""


# ── Gradio layout ─────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    device_opts = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

    with gr.Blocks(title="RecursiveMAS") as demo:
        gr.Markdown(
            "# RecursiveMAS\n"
            "Multi-agent reasoning via latent-space recursion.  \n"
            "Models are loaded into VRAM on the first request and stay warm for subsequent ones."
        )

        with gr.Row():
            # ── Settings panel ────────────────────────────────────────────
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Settings")
                style_dd = gr.Dropdown(
                    choices=list(STYLE_SPECS.keys()),
                    value="sequential_light",
                    label="Collaboration style",
                )
                rounds_sl = gr.Slider(1, 5, value=3, step=1, label="Recursive rounds")
                latent_sl = gr.Slider(8, 64, value=32, step=8, label="Latent steps")
                device_dd = gr.Dropdown(choices=device_opts, value=device_opts[0], label="Device")
                gr.Markdown(
                    "**Approx. VRAM**\n"
                    "- `sequential_light` ≈ 5 GB\n"
                    "- `sequential_scaled` ≈ 12 GB\n"
                    "- `mixture` ≈ 15 GB\n"
                    "- `distillation` ≈ 18 GB\n"
                    "- `deliberation` ≈ 12 GB"
                )

            # ── Chat panel ────────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=520, label="", show_label=False)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a math, science, or reasoning question…",
                        label="",
                        lines=2,
                        scale=5,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                gr.Button("Clear").click(lambda: ([], []), outputs=[chatbot, gr.State([])])

        state = gr.State([])

        for trigger in (send_btn.click, msg.submit):
            trigger(
                respond,
                inputs=[msg, state, style_dd, rounds_sl, latent_sl, device_dd],
                outputs=[chatbot, state, msg],
            )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser(description="RecursiveMAS web UI")
    p.add_argument("--host", default="0.0.0.0", help="Bind host")
    p.add_argument("--port", type=int, default=7860, help="Bind port")
    p.add_argument("--share", action="store_true", help="Create a public Gradio tunnel")
    cfg = p.parse_args()

    build_ui().launch(server_name=cfg.host, server_port=cfg.port, share=cfg.share, theme=gr.themes.Soft())
