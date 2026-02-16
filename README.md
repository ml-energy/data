# The ML.ENERGY Data Toolkit

A Python toolkit for [ML.ENERGY](https://ml.energy) datasets: loading raw results, filtering and analyzing runs, fitting models, and building data packages.

We currently have [The ML.ENERGY Benchmark v3.0](https://github.com/ml-energy/benchmark) dataset, which includes LLM and diffusion inference runs on NVIDIA H100 and B200 GPUs.
Actual data are stored in Hugging Face Hub: [`ml-energy/benchmark-v3`](https://huggingface.co/datasets/ml-energy/benchmark-v3).
This repository contains the toolkit code, not the data itself.

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

# Column access via .data
energies = runs.data.energy_per_token_joules  # list[float]
```

Filter, group, and compare across GPU generations and model architectures:

```python
# Compare GPU generations: best energy efficiency per model on GPQA
for gpu, group in runs.task("gpqa").group_by("gpu_model").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{gpu}: {best.nickname} @ {best.energy_per_token_joules:.3f} J/tok, "
          f"{best.output_throughput_tokens_per_sec:.0f} tok/s")

# MoE, Dense, Hybrid: who's more energy-efficient?
for arch, group in runs.task("gpqa").gpu("B200").group_by("architecture").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{arch}: {best.nickname} @ {best.energy_per_token_joules:.3f} J/tok")
```

## Who uses it

- [**The ML.ENERGY Leaderboard v3.0**](https://ml.energy/leaderboard): Benchmark results are loaded and compiled into the leaderboard web app data format.
- [**OpenG2G**](TODO): Datacenter-grid coordination simulation framework; loads benchmark data and fits models.
- [**The ML.ENERGY blog**](https://ml.energy/blog): Analysis scripts for blog posts.

## Documentation

See the full [documentation site](https://ml-energy.github.io/mlenergy-data/) for:

- [Usage guide](https://ml-energy.github.io/mlenergy-data/guide/) — progressive walkthrough from loading data to fitting models.
- [API reference](https://ml-energy.github.io/mlenergy-data/api/records/) — auto-generated from docstrings.

## Citation

```bibtex
@inproceedings{mlenergy-neuripsdb25,
    title={The {ML.ENERGY Benchmark}: Toward Automated Inference Energy Measurement and Optimization},
    author={Jae-Won Chung and Jeff J. Ma and Ruofan Wu and Jiachen Liu and Oh Jun Kweon and Yuxuan Xia and Zhiyu Wu and Mosharaf Chowdhury},
    year={2025},
    booktitle={NeurIPS Datasets and Benchmarks},
}
```
