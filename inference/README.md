# 🔎 RecursiveMAS Inference

`./inference`contains the unified inference runner for RecursiveMAS. By default, it loads public Hugging Face checkpoints for plug-and-play use; it can also evaluate locally trained checkpoints to reproduce paper experiments.

## 🗺️ Inference Code Map

```text
inference/
├── run.py                         # unified inference entry point
├── load_from_repo.py
├── hf_resolver.py
├── modeling.py                    # RecursiveLink and model adapter modules
├── system_loader.py
├── prompts.py                     # inference prompts by collaboration style
├── dataset/                       # local offline eval datasets
└── inference_utils/
    ├── inference_mas.py           # RecursiveMAS pipeline (sequential) and shared utilities
    ├── inference_mas_mixture.py
    ├── inference_mas_distill.py
    ├── inference_mas_deliberation.py
    ├── lcb_utils.py               # code-eval execution and judging helpers
    ├── eval_datasets_extra.py     # benchmark evaluation helpers
    ├── tavily_search.py           # Tavily search integration
    └── llm_judge.py
```

## 🗂️ Supported Tasks

| Benchmarks | Task | Metric |
| --- | --- | --- |
| `math500` | math reasoning | accuracy |
| `gpqa` | graduate-level science QA | accuracy |
| `medqa` | medical QA | accuracy |
| `mbppplus` | code generation | test pass rate |
| `aime25`, `aime26` | competition math | pass@10 |
| `livecodebench` | code generation | pass@1 |
| `bamboogle`, `hotpotqa` | open-domain search QA | EM/LLM-judge |

You can also add your own offline evaluation tasks under `./inference/dataset`, we provide `medqa.json` as a reference example.

## 🚀 Plug-and-Play

Download our reference checkpoints under the [RecursiveMAS Hugging Face organization](https://huggingface.co/RecursiveMAS/models).

Then run the commands below:

```bash
python inference/run.py \
  --style sequential_scaled \
  --dataset math500 \
  --device cuda
```

This is the fastest way to explore a released RecursiveMAS configuration. To reproduce our paper experiments, please first follow our training pipeline in the [training guide](../train/README.md) to train the corresponding configurations, the run the inference and evaluation steps below.

## 📈 Run Paper Experiments

The table below summarizes the key command-line arguments used for inference and evaluation.


| Flag | Description |
| --- | --- |
| `--style` | RecursiveMAS family: `sequential_light`, `sequential_scaled`, `mixture`, `distillation`, `deliberation` |
| `--dataset` | Evaluation dataset name |
| `--device` | Device string such as `cuda`, `cuda:0`, or `cpu` |
| `--num_samples` | Optional limit for quick canary runs |
| `--ckpt_override` | Load locally trained, task-specific checkpoints instead of released HF checkpoints |

Please first save your trained checkpoints, then use `--ckpt_override KEY=PATH` to run the matching task-specific configuration below.

---

### Sequential-Style (Light/Scaled)

```bash
python inference/run.py \
  --style sequential_light \
  --dataset math500 \
  --device cuda \
  --ckpt_override planner=train/ckpts/seq_light/planner_math \
  --ckpt_override critic=train/ckpts/seq_light/refiner_math \
  --ckpt_override solver=train/ckpts/seq_light/solver_math \
  --ckpt_override outer=train/ckpts/seq_light/outer_math
```
Modify `--style`, `--dataset`, and `ckpt_override` to switch to other configurations.

---

### Distillation-Style

```bash
python inference/run.py \
  --style distillation \
  --dataset math500 \
  --device cuda \
  --ckpt_override expert=train/ckpts/distill/expert_math \
  --ckpt_override learner=train/ckpts/distill/learner_math \
  --ckpt_override outer=train/ckpts/distill/outer_math
```

Modify `--style`, `--dataset`, and `ckpt_override` to switch to other configurations.

---

### Mixture-Style

```bash
python inference/run.py \
  --style mixture \
  --dataset math500 \
  --device cuda \
  --ckpt_override math=train/ckpts/mixture/math_expert \
  --ckpt_override code=train/ckpts/mixture/code_expert \
  --ckpt_override science=train/ckpts/mixture/science_expert \
  --ckpt_override summarizer=train/ckpts/mixture/summarizer \
  --ckpt_override outer=train/ckpts/mixture/outer
```

Modify `--style`, `--dataset`, and `ckpt_override` to switch to other configurations.

---

### Deliberation-Style

```bash
python inference/run.py \
  --style deliberation \
  --dataset bamboogle \
  --device cuda \
  --tavily_keys_file keys.txt \
  --ckpt_override reflector=train/ckpts/deliberation/reflector \
  --ckpt_override toolcaller=train/ckpts/deliberation/toolcaller \
  --ckpt_override outer=train/ckpts/deliberation/outer
```

Modify `--style`, `--dataset`, and `ckpt_override` to switch to other configurations.
