# 🔥 RecursiveMAS Training

`./train` contains the training pipeline with task-specific configurations. The overall Inner-Outer Loop training includes two phases:

1. **Inner-Loop Training (`train/train_inner.py`)**: train each agent role-specific *inner* RecursiveLink (frozen base model + a small
  `ln_res_adapter`). Run once per role.
2. **Outer-Loop Training (`train/train_outer.py`)**: Connect all agents together and train the *outer* RecursiveLink between agents through recursion. Outer-loop training loads the inner adapters trained in the first stage and keeps them frozen. Only the outer RecursiveLink modules are optimized. Note gradients are allowed to pass through the frozen inner adapters so the outer links can be trained across recursion rounds.

> 🎯 To repduce our experiment results across all downstream tasks, we recommand training a matching RecursiveMAS configuration with our provided pipeline. Then use the resulting local checkpoints for downstream inference and evaluation.

## 🗺️ Training Code Map

```text
train/
├── train_inner.py              # inner-loop training
├── train_outer.py              # outer-loop training
├── data.py                     # dataset loading and tokenization
├── model.py                    # RecursiveLink and model adapter modules
├── mas_prompt.py               # training prompts by collaboration style
├── outer/
│   ├── common.py               # shared outer-training utilities
│   ├── sequential.py
│   ├── mixture.py
│   ├── distillation.py
│   └── deliberation.py
└── data/
    └── README.md               # Hugging Face dataset mapping and local backup format
```

## 💫 Getting Start

Pleae first make sure to complete the environment setups in the top-level [README](../README.md).



### Training Data

Hugging Face datasets are the recommended data path. Each dataset has a row-format `train` split and can be passed directly with `--dataset_name`.

**See [data/README.md](data/README.md) for dataset details and full mapping.**

---

### Sequential-Style (Light/Scaled)

Sequential-Light and Sequential-Scaled use the same data and training family; they only differ in backbone models.


| Role | Backbone Model (Light) | Backbone Model (Scaled) |
| --- | --- | ---|
| planner | `Qwen/Qwen3-1.7B` | `google/gemma-3-4b-it`|
| critic/refiner | `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` |
| solver | `Qwen/Qwen2.5-Math-1.5B-Instruct` | `Qwen/Qwen3.5-4B` |


Inner-Loop Training for Math/Science Domain Tasks:

```bash
python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --mas_design sequential \
  --mas_role planner \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --save_dir train/ckpts/seq_light/planner_math

python train/train_inner.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --mas_design sequential \
  --mas_role refiner \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --save_dir train/ckpts/seq_light/refiner_math

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --mas_design sequential \
  --mas_role solver \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --save_dir train/ckpts/seq_light/solver_math
```

Outer-Loop Training for Math/Science Domain Tasks:

```bash
python train/train_outer.py \
  --style sequential_light \
  --agent1_model_name_or_path Qwen/Qwen3-1.7B \
  --agent2_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --agent1_inner_aligner_path train/ckpts/seq_light/planner_math \
  --agent2_inner_aligner_path train/ckpts/seq_light/refiner_math \
  --agent3_inner_aligner_path train/ckpts/seq_light/solver_math \
  --mas_task math \
  --dataset_name RecursiveMAS/Sequential-Math \
  --save_dir train/ckpts/seq_light/outer_math
```

Inner-Outer Loop Training for coding domain tasks, use the similar commands:

```bash
# Inner-Loop training
python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --mas_design sequential \
  --mas_role planner \
  --mas_task code \
  --dataset_name RecursiveMAS/Sequential-Code \
  --save_dir train/ckpts/seq_light/planner_code

python train/train_inner.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --mas_design sequential \
  --mas_role refiner \
  --mas_task code \
  --dataset_name RecursiveMAS/Sequential-Code \
  --save_dir train/ckpts/seq_light/refiner_code

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --mas_design sequential \
  --mas_role solver \
  --mas_task code \
  --dataset_name RecursiveMAS/Sequential-Code \
  --save_dir train/ckpts/seq_light/solver_code

# Outer-Loop training
python train/train_outer.py \
  --style sequential_light \
  --agent1_model_name_or_path Qwen/Qwen3-1.7B \
  --agent2_model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --agent3_model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
  --agent1_inner_aligner_path train/ckpts/seq_light/planner_code \
  --agent2_inner_aligner_path train/ckpts/seq_light/refiner_code \
  --agent3_inner_aligner_path train/ckpts/seq_light/solver_code \
  --mas_task code \
  --dataset_name RecursiveMAS/Sequential-Code \
  --save_dir train/ckpts/seq_light/outer_code
```

For **Sequential-Style (Scaled)** training, use the same commands above, then replace backbone configurations (planner `google/gemma-3-4b-it`, refiner `meta-llama/Llama-3.2-3B-Instruct`, solver `Qwen/Qwen3.5-4B`) and use `--style sequential_scaled`.

---

### Distillation-Style

Distillation-Style uses one expert agent and one learner agent. First train both role-specific Inner RecursiveLinks, then train the Outer RecursiveLinks.

| Role | Backbone Model |
| --- | --- |
| expert | `Qwen/Qwen3.5-9B` |
| learner | `Qwen/Qwen3.5-4B` |

Inner-Loop Training for Math/Science Domain Tasks:

```bash
python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --mas_design distill \
  --mas_role distill_expert \
  --mas_task math \
  --dataset_name RecursiveMAS/Distillation-Math \
  --save_dir train/ckpts/distill/expert_math

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --mas_design distill \
  --mas_role distill_learner \
  --mas_task math \
  --dataset_name RecursiveMAS/Distillation-Math \
  --save_dir train/ckpts/distill/learner_math
```

Outer-Loop Training for Math/Science Domain Tasks:

```bash
python train/train_outer.py \
  --style distillation \
  --expert_model_name_or_path Qwen/Qwen3.5-9B \
  --learner_model_name_or_path Qwen/Qwen3.5-4B \
  --expert_inner_aligner_path train/ckpts/distill/expert_math \
  --learner_inner_aligner_path train/ckpts/distill/learner_math \
  --mas_task math \
  --dataset_name RecursiveMAS/Distillation-Math \
  --save_dir train/ckpts/distill/outer_math
```

Inner-Outer Loop Training for coding domain tasks, use the similar commands:

```bash
# Inner-Loop training
python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-9B \
  --mas_design distill \
  --mas_role distill_expert \
  --mas_task code \
  --dataset_name RecursiveMAS/Distillation-Code \
  --save_dir train/ckpts/distill/expert_code

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --mas_design distill \
  --mas_role distill_learner \
  --mas_task code \
  --dataset_name RecursiveMAS/Distillation-Code \
  --save_dir train/ckpts/distill/learner_code

# Outer-Loop training
python train/train_outer.py \
  --style distillation \
  --expert_model_name_or_path Qwen/Qwen3.5-9B \
  --learner_model_name_or_path Qwen/Qwen3.5-4B \
  --expert_inner_aligner_path train/ckpts/distill/expert_code \
  --learner_inner_aligner_path train/ckpts/distill/learner_code \
  --mas_task code \
  --dataset_name RecursiveMAS/Distillation-Code \
  --save_dir train/ckpts/distill/outer_code
```

---

### Mixture-Style 

Mixture-Style uses three domain-expert agents and one summarizer agent. First train each role-specific Inner RecursiveLink, then train the Outer RecursiveLinks that connect experts and summarizer.

| Role | Backbone Model |
| --- | --- |
| math expert | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| code expert | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| science expert | `BioMistral/BioMistral-7B` |
| summarizer | `Qwen/Qwen3.5-2B` |

Inner-Loop Training:

```bash
python train/train_inner.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --mas_design hie \
  --mas_role hie_math_expert \
  --mas_task math \
  --dataset_name RecursiveMAS/Mixture-Math \
  --save_dir train/ckpts/mixture/math_expert

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
  --mas_design hie \
  --mas_role hie_code_expert \
  --mas_task code \
  --dataset_name RecursiveMAS/Mixture-Code \
  --save_dir train/ckpts/mixture/code_expert

python train/train_inner.py \
  --model_name_or_path BioMistral/BioMistral-7B \
  --mas_design hie \
  --mas_role hie_science_expert \
  --mas_task choice \
  --dataset_name RecursiveMAS/Mixture-Science \
  --save_dir train/ckpts/mixture/science_expert

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-2B \
  --mas_design hie \
  --mas_role hie_summarizer \
  --mas_task math \
  --dataset_name RecursiveMAS/Mixture-Summarizer \
  --save_dir train/ckpts/mixture/summarizer
```

Outer-Loop Training:

```bash
python train/train_outer.py \
  --style mixture \
  --agent1_model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --agent2_model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
  --agent3_model_name_or_path BioMistral/BioMistral-7B \
  --agent4_model_name_or_path Qwen/Qwen3.5-2B \
  --agent1_inner_aligner_path train/ckpts/mixture/math_expert \
  --agent2_inner_aligner_path train/ckpts/mixture/code_expert \
  --agent3_inner_aligner_path train/ckpts/mixture/science_expert \
  --agent4_inner_aligner_path train/ckpts/mixture/summarizer \
  --mas_task math \
  --dataset_name RecursiveMAS/Mixture-Outer \
  --save_dir train/ckpts/mixture/outer
```

---

### Deliberation-Style

Deliberation-Style uses a reflector agent and a tool-caller agent. First train both role-specific Inner RecursiveLinks, then train the Outer RecursiveLinks for reflection and tool-use coordination.

| Role | Backbone Model |
| --- | --- |
| reflector | `Qwen/Qwen3.5-4B` |
| tool-caller | `Qwen/Qwen3.5-4B` |

Inner-Loop Training:

```bash
python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --mas_design deliberation \
  --mas_role deliberation_reflector \
  --mas_task math \
  --dataset_name RecursiveMAS/Deliberation \
  --save_dir train/ckpts/deliberation/reflector

python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3.5-4B \
  --mas_design deliberation \
  --mas_role deliberation_toolcaller \
  --mas_task math \
  --dataset_name RecursiveMAS/Deliberation \
  --save_dir train/ckpts/deliberation/toolcaller
```

Outer-Loop Training:

```bash
python train/train_outer.py \
  --style deliberation \
  --reflector_model_name_or_path Qwen/Qwen3.5-4B \
  --toolcaller_model_name_or_path Qwen/Qwen3.5-4B \
  --reflector_inner_aligner_path train/ckpts/deliberation/reflector \
  --toolcaller_inner_aligner_path train/ckpts/deliberation/toolcaller \
  --mas_task math \
  --dataset_name RecursiveMAS/Deliberation \
  --save_dir train/ckpts/deliberation/outer
```

---

### (Optional) Train RecursiveMAS with Local Data

If you need training with offline data (e.g., your own training data), place the original wrapped JSON files under `train/data/` and add `--dataset_json_field data`. Then run commands in the following:

```bash
python train/train_inner.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --mas_design sequential \
  --mas_role planner \
  --mas_task math \
  --dataset_name train/data/Sequential-Math.json \
  --dataset_json_field data \
  --save_dir train/ckpts/seq_light/planner_math
```


## ⚙️ Configurations

Below are default configurations used in our experiments.

| Setting | Default |
| --- | --- |
| Training length | `--max_steps 20000` |
| Inner LR | `--adapter_lr 5e-4` |
| Outer LR | `--outer_lr 5e-4` |
| Scheduler (Inner & Outer) | `--lr_scheduler_type cosine`, `--warmup_steps 10` |
| Inner loss | `--adapter_cos_weight 1.0`, `--adapter_mse_weight 0.0` |
| Outer loss | `--supervise_final_only 1` (optimize the final recursion round's loss)|
| Recursion Rounds/Depth | `--num_recursive_rounds 3` |
| Precision | `--dtype bfloat16` |



## 🔎 Evaluation

After training a matching collaboration style and task setting, you can use the resulting checkpoints for inference and downstream evaluation. 

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

**Please refer to the [inference guide](../inference/README.md) for more inference and evaluation details.**
