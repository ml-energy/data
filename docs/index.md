# The ML.ENERGY Data & Toolkit

[ML.ENERGY](https://ml.energy) publishes open-source datasets.
To aid in working with these datasets, we also provide a Python toolkit: `mlenergy-data`.

We currently have [The ML.ENERGY Benchmark v3.0](https://github.com/ml-energy/benchmark) dataset, which includes LLM and diffusion inference runs on NVIDIA H100 and B200 GPUs.
Actual data are stored in Hugging Face Hub: [`ml-energy/benchmark-v3`](https://huggingface.co/datasets/ml-energy/benchmark-v3).

## What the Toolkit Does

- **Load and filter benchmark runs** with typed, immutable collection classes (`LLMRuns`, `DiffusionRuns`).
- **Extract bulk per-request detailed data** (e.g., power timelines, ITL samples, output lengths) as DataFrames.
- **Fit models**: logistic power/latency curves and ITL latency distributions.

## Installation

```bash
pip install mlenergy-data
```

## Dataset Access

The benchmark dataset ([`ml-energy/benchmark-v3`](https://huggingface.co/datasets/ml-energy/benchmark-v3)) is gated on Hugging Face Hub.
Before using the toolkit to load data from HF, you need to:

1. Visit the [dataset page](https://huggingface.co/datasets/ml-energy/benchmark-v3) and request access (granted automatically).
2. Set the `HF_TOKEN` environment variable to a [Hugging Face access token](https://huggingface.co/settings/tokens).

## Quick Example

```python
from mlenergy_data.records import LLMRuns

runs = LLMRuns.from_hf()

# Find the minimum-energy model on GPQA
gpqa_runs = runs.task("gpqa")
best = min(gpqa_runs, key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok on {best.gpu_model}")

energies = [r.energy_per_token_joules for r in runs]
```

Filter, group, and compare across GPU generations and model architectures:

```python
# Compare GPU generations: best energy efficiency per model on GPQA
for gpu, group in runs.task("gpqa").group_by("gpu_model").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{gpu}: {best.nickname} @ {best.energy_per_token_joules:.3f} J/tok, "
          f"{best.output_throughput_tokens_per_sec:.0f} tok/s")

# MoE, Dense, Hybrid: who's more energy-efficient?
for arch, group in runs.task("gpqa").gpu_model("B200").group_by("architecture").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{arch}: {best.nickname} @ {best.energy_per_token_joules:.3f} J/tok")
```

## Who Uses It

- [**The ML.ENERGY Leaderboard v3.0**](https://ml.energy/leaderboard): Benchmark results are loaded and compiled into the leaderboard web app data format.
- [**OpenG2G**](https://gpu2grid.io/openg2g/): Datacenter-grid coordination simulation framework; loads benchmark data and fits models.
- [**The ML.ENERGY blog**](https://ml.energy/blog): Analysis scripts for blog posts.

See the [Guide](guide.md) page for more details, together with a progressive walkthrough.

## Next Steps

- [Guide](guide.md): Progressive walkthrough from loading data to fitting models.
- [API Reference](api/records.md): Auto-generated from docstrings.

## Citation

```bibtex
@inproceedings{mlenergy-neuripsdb25,
    title={The {ML.ENERGY Benchmark}: Toward Automated Inference Energy Measurement and Optimization},
    author={Jae-Won Chung and Jeff J. Ma and Ruofan Wu and Jiachen Liu and Oh Jun Kweon and Yuxuan Xia and Zhiyu Wu and Mosharaf Chowdhury},
    year={2025},
    booktitle={NeurIPS Datasets and Benchmarks},
}
```
