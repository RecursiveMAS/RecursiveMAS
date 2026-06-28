# 🗂️ Training Data

RecursiveMAS training data is hosted on Hugging Face. You can access them from [here](https://huggingface.co/RecursiveMAS/datasets).

## 🤗 Hugging Face Datasets

Use Hugging Face datasets directly whenever possible. The uploaded datasets are row-format datasets with a `train` split.

```bash
python train/train_inner.py \
  --dataset_name RecursiveMAS/Sequential-Math \
  ...
```

### 📚 Dataset Map

Benchmarks are grouped by the training setup they correspond to. For styles with separate math/code datasets, train the matching setup before evaluating that benchmark family.

| Hugging Face dataset | Descriptions | Downstream benchmarks |
| --- | --- | --- |
| [`🤗 RecursiveMAS/Sequential-Math`](https://huggingface.co/datasets/RecursiveMAS/Sequential-Math) | Sequential planner, critic/refiner, and solver Inner Loop & Outer Loop training for math/science benchmarks | `math500`, `gpqa`, `medqa`, `aime25`, `aime26` |
| [`🤗 RecursiveMAS/Sequential-Code`](https://huggingface.co/datasets/RecursiveMAS/Sequential-Code) | Sequential planner, critic/refiner, and solver Inner Loop & Outer Loop training for code benchmarks | `mbppplus`, `livecodebench` |
| [`🤗 RecursiveMAS/Distillation-Math`](https://huggingface.co/datasets/RecursiveMAS/Distillation-Math) | Distillation expert and learner Inner Loop & Outer Loop training for math/science benchmarks | `math500`, `gpqa`, `medqa`, `aime25`, `aime26` |
| [`🤗 RecursiveMAS/Distillation-Code`](https://huggingface.co/datasets/RecursiveMAS/Distillation-Code) | Distillation expert and learner Inner Loop & Outer Loop training for code benchmarks | `mbppplus`, `livecodebench` |
| [`🤗 RecursiveMAS/Mixture-Math`](https://huggingface.co/datasets/RecursiveMAS/Mixture-Math) | Mixture math expert Inner RecursiveLink training | --- |
| [`🤗 RecursiveMAS/Mixture-Code`](https://huggingface.co/datasets/RecursiveMAS/Mixture-Code) | Mixture code expert Inner Loop training | --- |
| [`🤗 RecursiveMAS/Mixture-Science`](https://huggingface.co/datasets/RecursiveMAS/Mixture-Science) | Mixture science expert Inner Loop training | --- |
| [`🤗 RecursiveMAS/Mixture-Summarizer`](https://huggingface.co/datasets/RecursiveMAS/Mixture-Summarizer) | Mixture summarizer Inner Loop training | --- |
| [`🤗 RecursiveMAS/Mixture-Outer`](https://huggingface.co/datasets/RecursiveMAS/Mixture-Outer) | Mixture Outer Loop training | `gpqa`, `medqa`, `aime26`, `livecodebench`|
| [`🤗 RecursiveMAS/Deliberation`](https://huggingface.co/datasets/RecursiveMAS/Deliberation) | Deliberation reflector and tool-caller Inner Loop & Outer Loop training | `gpqa`, `aime26`, `bamboogle`, `hotpotqa` |

### (Optional) Local Training Data

You can also use your own training data for RecursiveMAS. For offline data storage, place local JSON files under `./train/data`. Each file should contain a data array formatted as follows:
```json
{
  "data": [
    {
      "question": "...",
      "answer": "..."
    }
  ]
}
```

Add both flags to the training command:

```bash
--dataset_name train/data/Sequential-Math.json \
--dataset_json_field data
```

> 📌 Note: Please only use `--dataset_json_field` for offline data loading. For Hugging Face datasets, you can simply ignore this argument.

See the [training guide](../README.md) for complete training commands.
