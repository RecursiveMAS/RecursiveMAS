from __future__ import annotations

from typing import Dict

STYLE_SPECS: Dict[str, Dict[str, object]] = {
    "sequential_light": {
        "family": "sequential",
        "repos": {
            "planner": "RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B",
            "critic": "RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B",
            "solver": "RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B",
            "outer": "RecursiveMAS/Sequential-Light-Outerlinks",
        },
    },
    "sequential_scaled": {
        "family": "sequential",
        "repos": {
            "planner": "RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B",
            "critic": "RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B",
            "solver": "RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B",
            "outer": "RecursiveMAS/Sequential-Scaled-Outerlinks",
        },
    },
    "mixture": {
        "family": "mixture",
        "repos": {
            "math": "RecursiveMAS/Mixture-Math-DeepSeek-R1-Distill-Qwen-1.5B",
            "code": "RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B",
            "science": "RecursiveMAS/Mixture-Science-BioMistral-7B",
            "summarizer": "RecursiveMAS/Mixture-Summarizer-Qwen3.5-2B",
            "outer": "RecursiveMAS/Mixture-Outerlinks",
        },
    },
    "distillation": {
        "family": "distillation",
        "repos": {
            "expert": "RecursiveMAS/Distillation-Expert-Qwen3.5-9B",
            "learner": "RecursiveMAS/Distillation-Learner-Qwen3.5-4B",
            "outer": "RecursiveMAS/Distillation-Outerlinks",
        },
    },
    "deliberation": {
        "family": "deliberation",
        "repos": {
            "reflector": "RecursiveMAS/Deliberation-Reflector-Qwen3.5-4B",
            "toolcaller": "RecursiveMAS/Deliberation-Toolcaller-Qwen3.5-4B",
            "outer": "RecursiveMAS/Deliberation-Outerlinks",
        },
    },
}

DATASET_DEFAULT_SPLIT = {
    "math500": "test",
    "math-500": "test",
    "huggingfaceh4/math-500": "test",
    "gpqa": "train",
    "gpqa_diamond": "train",
    "idavidrein/gpqa": "train",
    "medqa": "train",
    "local/medqa": "train",
    "mbppplus": "test",
    "evalplus/mbppplus": "test",
    "aime25": "test",
    "math-ai/aime25": "test",
    "aime26": "train",
    "matharena/aime_2026": "train",
    "lcb": "test",
    "livecodebench": "test",
    "livecodebench/code_generation_lite": "test",
    "bamboogle": "test",
    "chiayewken/bamboogle": "test",
    "hotpotqa": "validation",
    "hotpot_qa": "validation",
}

