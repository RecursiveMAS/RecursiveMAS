<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo.png">
    <img alt="RecursiveMAS" src="assets/logo.png" width=300>
  </picture>
</p>

<h3 align="center">
Scaling agent collaboration through latent-space recursion.
</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2604.25917"><img src="https://img.shields.io/badge/arXiv-2604.25917-B31B1B.svg?logo=arxiv" alt="Arxiv"></a>
    <a href="https://huggingface.co/papers/2604.25917"><img src="https://img.shields.io/badge/HF%20Daily%20Paper-2604.25917-FFD21E.svg?logo=huggingface" alt="HF Daily Paper"></a>
    <a href="https://recursivemas.github.io/"><img src="https://img.shields.io/badge/Website-RecursiveMAS-2176BC?logo=GoogleChrome" alt="Website"></a>
    <a href="https://github.com/RecursiveMAS/RecursiveMAS"><img src="https://img.shields.io/badge/Github-RecursiveMAS-2D8CFF.svg?logo=github" alt="RecursiveMAS"></a>
    <a href="https://huggingface.co/RecursiveMAS/collections"><img src="https://img.shields.io/badge/Huggingface-Collections-FFD21E.svg?logo=huggingface" alt="Huggingface Collection"></a>
    <a href="https://huggingface.co/RecursiveMAS/models"><img src="https://img.shields.io/badge/Huggingface-Models-FFD21E.svg?logo=huggingface" alt="Huggingface Model"></a>
    <a href="https://www.linkedin.com/posts/jiaruzou_recursivemas-recurisvelearning-multiagentsystems-ugcPost-7455645681341493248-ioLJ/?utm_source=share&utm_medium=member_desktop&rcm=ACoAADc5TzgBN_tNOuzpi7kE7n6dZ0y13EkxZOs"><img src="https://img.shields.io/badge/LinkedIn-Coverage-0A66C2.svg?logo=linkedin&logoColor=white" alt="LinkedIn Coverage"></a>
    <a href="https://x.com/Jiaru_Zou/status/2049551828296389118"><img src="https://img.shields.io/badge/Twitter-Coverage-1DA1F2.svg?logo=x" alt="Twitter Coverage"></a>
    <a href="https://venturebeat.com/ai/how-recursivemas-speeds-up-multi-agent-inference-by-2-4x-and-reduces-token-usage-by-75"><img src="https://img.shields.io/badge/Venture-Beat-EE1C25.svg?labelColor=111111&color=EE1C25&logo=venturebeat&logoColor=white" alt="VentureBeat Coverage"></a>
    <a href="https://www.youtube.com/watch?v=dUmT0OIGoqE"><img src="https://img.shields.io/badge/YouTube-Tutorial-FF0000?logo=youtube" alt="Youtube"></a>
    <a href="https://vishalmysore.github.io/recursiveMASDemo/"><img src="https://img.shields.io/badge/Playground-Demo-00B894.svg?logo=googlechrome&logoColor=white" alt="Playground Demo"></a>
</p>


<p align="center">
  <video src="https://github.com/user-attachments/assets/9c09261a-c9e7-4851-8462-eeda69989b4e" controls width="300"></video>
</p>

## 📰 News

**[2026.06.23]** Try it out! RecursiveMAS now has a [**🧩YouTube tutorial**](https://www.youtube.com/watch?v=dUmT0OIGoqE) and an [**🎮interactive playground demo**](https://vishalmysore.github.io/recursiveMASDemo/)! Special thanks to [@TwoMinutePapers](https://www.youtube.com/@TwoMinutePapers) and [@vishalmysore](https://github.com/vishalmysore)!

**[2026.05.24]** Check out the [VentureBeat article](https://t.co/KSQwBwpC4W) featuring our research on RecursiveMAS!

**[2026.05.01]** Ours paper is featured as [🤗 HuggingFace 1st Paper of the Week/Day](https://huggingface.co/papers/2604.25917)!

**[2026.04.28]** All [collaboration styles](https://huggingface.co/RecursiveMAS/collections) and [model checkpoints](https://huggingface.co/RecursiveMAS/models), with [examplified downstream inference](https://github.com/RecursiveMAS/RecursiveMAS) are now available. Stay tuned for the complete training/inference pipeline and additional features!

**[2026.04.28]** We have released the [RecursiveMAS paper](https://huggingface.co/papers/2604.25917)! 


## 🌟 Introduction

<p align="center">
  <img src="assets/exps.png" width="100%" alt="RecursiveMAS Overview">
</p>

**RecursiveMAS** is a multi-agent framework that scales agent collaboration through **latent-space recursion**. Instead of treating each LLM agent as an isolated module, RecursiveMAS casts the entire multi-agent system as a **unified recursive computation**. Heterogeneous agents are connected through lightweight RecursiveLink modules, allowing agents to iteratively exchange, refine, and evolve their latent states across recursion rounds.

## 📋 Supported Features

✅ Release All Collaboration Patterns (Sequential, Mixture, Deliberation, Distillation).

✅ Release Demo Code for Inference (Commands Provided Below).

☑️ Add Complete Inference Pipeline Across All Downstreams.

☑️ Add All Training Data & Implementation Details.

☑️ Add Additional Supported Model Family & MAS Collaboration Patterns.

## 🛠️ Environment Setup

This repository provides the code for running RecursiveMAS under different multi-agent collaboration styles. 

To begin with, we recommend creating a new conda environment:

```bash
conda create -n recursivemas python=3.10 -y
conda activate recursivemas
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For Deliberation-Style, the Tool-Caller Agent requires external search tools to retrieve information. 
Please set up a search API key (e.g., a Tavily API key) in `.env` file:
```bash
TAVILY_API_KEY=your_tavily_api_key_here
```

## 💥 Quick Start

### 🤖 Load Model Checkpoints

To run RecursiveMAS, you need to download and store the checkpoints for each agent role in the multi-agent system from our Hugging Face release.

The checkpoints are organized by collaboration style. Each collection contains the individual role-specific agent together with their RecursiveLink modules.

### [Sequential-Style (Light) MAS Collection](https://huggingface.co/collections/RecursiveMAS/sequential-style-recursivemas)

| **Model Organization** | **Download** |
| ---------------------- | ------------ |
| Sequential-Light-Planner-Qwen3-1.7B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Light-Planner-Qwen3-1.7B) |
| Sequential-Light-Critic-Llama3.2-1B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Light-Critic-Llama3.2-1B) |
| Sequential-Light-Solver-Qwen2.5-Math-1.5B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Light-Solver-Qwen2.5-Math-1.5B) |
| Sequential-Light-Outerlinks | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Light-Outerlinks) |

### [Sequential-Style (Scaled) MAS Collection](https://huggingface.co/collections/RecursiveMAS/sequential-style-recursivemas)

| **Model Organization** | **Download** |
| ---------------------- | ------------ |
| Sequential-Scaled-Planner-Gemma3-4B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Scaled-Planner-Gemma3-4B) |
| Sequential-Scaled-Critic-Llama3.2-3B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Scaled-Critic-Llama3.2-3B) |
| Sequential-Scaled-Solver-Qwen3.5-4B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Scaled-Solver-Qwen3.5-4B) |
| Sequential-Scaled-Outerlinks | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Sequential-Scaled-Outerlinks) |

### [Mixture-Style MAS Collection](https://huggingface.co/collections/RecursiveMAS/mixture-style-recursivemas)

| **Model Organization** | **Download** |
| ---------------------- | ------------ |
| Mixture-Math-DeepSeek-R1-Distill-Qwen-1.5B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Mixture-Math-DeepSeek-R1-Distill-Qwen-1.5B) |
| Mixture-Code-Qwen2.5-Coder-3B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Mixture-Code-Qwen2.5-Coder-3B) |
| Mixture-Science-BioMistral-7B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Mixture-Science-BioMistral-7B) |
| Mixture-Summarizer-Qwen3.5-2B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Mixture-Summarizer-Qwen3.5-2B) |
| Mixture-Outerlinks | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Mixture-Outerlinks) |

### [Distillation-Style MAS Collection](https://huggingface.co/collections/RecursiveMAS/distillation-style-recursivemas)

| **Model Organization** | **Download** |
| ---------------------- | ------------ |
| Distillation-Expert-Qwen3.5-9B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Distillation-Expert-Qwen3.5-9B) |
| Distillation-Learner-Qwen3.5-4B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Distillation-Learner-Qwen3.5-4B) |
| Distillation-Outerlinks | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Distillation-Outerlinks) |

### [Deliberation-Style MAS Collection](https://huggingface.co/collections/RecursiveMAS/deliberation-style-recursivemas)

| **Model Organization** | **Download** |
| ---------------------- | ------------ |
| Deliberation-Reflector-Qwen3.5-4B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Deliberation-Reflector-Qwen3.5-4B) |
| Deliberation-Toolcaller-Qwen3.5-4B | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Deliberation-Toolcaller-Qwen3.5-4B) |
| Deliberation-Outerlinks | [🤗 HuggingFace](https://huggingface.co/RecursiveMAS/Deliberation-Outerlinks) |


Here is an example of how to load the whole MAS pipeline:

```python
from system_loader import load_mas_system

mas = load_mas_system(
    style="sequential_light",
    device="cuda",
    trust_remote_code=True,
)

planner = mas.agents["planner"].model
critic = mas.agents["critic"].model
solver = mas.agents["solver"].model
```

Detailed running code for loading agents and running RecursiveMAS on downstream tasks is provided in `run.py`. 


### 🔍 Clone the Repository

Next, clone our repository and enter the project directory:

```bash
git clone https://github.com/RecursiveMAS/RecursiveMAS.git
cd RecursiveMAS
```

The current repository is organized as follows:

```text
RecursiveMAS/
├── README.md
├── __init__.py
├── run.py
├── load_from_repo.py
├── hf_resolver.py
├── modeling.py
├── system_loader.py
├── prompts.py
├── requirements.txt
├── assets/
├── dataset/
└── inference_utils/
    ├── __init__.py
    ├── answer_utils.py
    ├── lcb_utils.py
    ├── reflector_tool_notes.py
    ├── inference_mas.py
    ├── inference_mas_mixture.py
    ├── inference_mas_distill.py
    └── inference_mas_deliberation.py
```

The key components are:

- `run.py`: the unified entry point for running RecursiveMAS inference.
- `load_from_repo.py`: maps each MAS style to our released Hugging Face checkpoints and dataset defaults.
- `hf_resolver.py`: resolves and load the Hugging Face checkpoints.
- `modeling.py`: implements RecursiveLink modules.
- `system_loader.py`: provides a high-level API for loading a full released multi-agent system.
- `prompts.py`: stores prompts for different MAS collaboration styles.
- `inference_utils/`: contains inference pipelines and evaluation utilities for different MAS structures.

### ⚙️ Running RecursiveMAS at Different Scales

We provide Sequential-style RecursiveMAS under both lightweight and scaled settings.

- **Sequential-style (Light)** uses lightweight agents for efficient recursive collaboration.
```bash
python run.py --style sequential_light --batch_size 32 --temperature 0.6 --top_p 0.95 --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

- **Sequential-style (Scaled)** uses stronger LLM agents to further improve reasoning performance.
```bash
python run.py --style sequential_scaled --batch_size 16 --temperature 0.6 --top_p 0.95 --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

### 🧩 Exploring Various Collaboration Patterns

RecursiveMAS can also be adapted to different MAS collaboration patterns beyond the sequential setting.

- **Mixture-style RecursiveMAS** coordinates multiple domain-specialized agents and aggregates their information through a summarizer.
```bash
python run.py --style mixture --batch_size 16 --temperature 0.6 --top_p 0.95 --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

- **Distillation-style RecursiveMAS** enables a larger Expert and a smaller Learner to interact recursively, improving the Learner while retaining better efficiency.
```bash
python run.py --style distillation --batch_size 16 --temperature 0.6 --top_p 0.95 --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

- **Deliberation-style RecursiveMAS** supports recursive coordination between a Reflector and a Tool-Caller for tool-integrated reasoning.
```bash
python run.py --style deliberation --batch_size 16 --temperature 0.6 --top_p 0.95 --dataset math500 --seed 42 --trust_remote_code 1 --device cuda
```

## 🙏 Acknowledgements

This project is built upon the excellent open-source community. We sincerely thank the developers and maintainers of the following libraries and resources:

- [vLLM](https://github.com/vllm-project/vllm) for supporting efficient LLM inference and serving.
- [ARPO](https://github.com/RUC-NLPIR/ARPO) for providing useful references on agentic tool-use systems and efficient tool-calling workflows.
- [TextGrad](https://github.com/zou-group/textgrad) for its pioneering framework on text-based optimization and natural-language feedback for compound agentic systems.

<!-- 
## 🚀 Contributing

We welcome discussions and contributions to RecursiveMAS. If you would like to suggest improvements, please feel free to contact us.

- [Xiyuan Yang](mailto:xiyuany4@illinois.edu)
- [Jiaru Zou](mailto:jiaru@stanford.edu)

--- -->

## 📚 Citation
```text
@misc{recursivemas,
      title={Recursive Multi-Agent Systems}, 
      author={Xiyuan Yang and Jiaru Zou and Rui Pan and Ruizhong Qiu and Pan Lu and Shizhe Diao and Jindong Jiang and Hanghang Tong and Tong Zhang and Markus J. Buehler and Jingrui He and James Zou},
      year={2026},
      eprint={2604.25917},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.25917}, 
}
```
