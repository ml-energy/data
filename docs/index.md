# The ML.ENERGY Data & Toolkit

[ML.ENERGY](https://ml.energy) publishes open-source datasets.
To aid in working with these datasets, we also provide a Python toolkit: `mlenergy-data`.

We currently have [The ML.ENERGY Benchmark v3.0](https://github.com/ml-energy/benchmark) dataset, which includes LLM and diffusion inference runs on NVIDIA H100 and B200 GPUs.
Actual data are currently stored in Hugging Face Hub: [`ml-energy/benchmark-v3`](https://huggingface.co/datasets/ml-energy/benchmark-v3).

## What the toolkit does

- **Load and filter benchmark runs** with typed, immutable collection classes (`LLMRuns`, `DiffusionRuns`).
- **Extract bulk data** — power timelines, ITL samples, output lengths — as DataFrames.
- **Fit models** — logistic power/latency curves, ITL latency distributions.
- **Build data packages** for publishing to Hugging Face Hub.

## Installation

```bash
pip install mlenergy-data
```

## Quick example

```python
from mlenergy_data.records import LLMRuns

runs = LLMRuns.from_hf()

# Find the most energy-efficient model on GPQA
best = min(runs.task("gpqa"), key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok on {best.gpu_model}")
```

## Who uses it

- [**The ML.ENERGY Leaderboard v3.0**](https://ml.energy/leaderboard): Benchmark results are loaded and compiled into the leaderboard web app data format.
- [**OpenG2G**](TODO): Datacenter-grid coordination simulation framework; loads benchmark data and fits models.
- [**The ML.ENERGY blog**](https://ml.energy/blog): Analysis scripts for blog posts.

## Next steps

- [Guide](guide.md): Progressive walkthrough from loading data to fitting models.
- [API Reference](api/records.md): Auto-generated from docstrings.
