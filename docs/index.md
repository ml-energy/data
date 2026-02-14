# mlenergy-data

Python toolkit for [ML.ENERGY](https://ml.energy) Benchmark datasets.

Currently supports **Benchmark v3.0** (LLM and diffusion inference on NVIDIA H100 and B200 GPUs). The library is designed to accommodate future benchmark versions and datasets.

## What it does

- **Load and filter benchmark runs** with typed, immutable collection classes (`LLMRuns`, `DiffusionRuns`).
- **Extract bulk data** — power timelines, ITL samples, output lengths — as DataFrames.
- **Fit models** — logistic power/latency curves, ITL latency distributions.
- **Build data packages** for publishing to Hugging Face Hub.

!!! note "Hugging Face Hub"
    Benchmark datasets will be published to the Hugging Face Hub. This is a work in progress.

## Installation

```bash
pip install -e .
```

## Quick example

```python
from mlenergy_data.records import LLMRuns

runs = LLMRuns.from_directory("/path/to/compiled/data")

# Find the most energy-efficient model on GPQA
best = min(runs.task("gpqa"), key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok on {best.gpu_model}")
```

## Who uses it

- [**ML.ENERGY Leaderboard v3.0**](https://ml.energy/leaderboard) — data pipeline for the public leaderboard.
- **OpenG2G** — datacenter-grid simulation framework; loads benchmark data and fits models.
- **ML.ENERGY blog** — analysis scripts for blog posts.

## Next steps

- [Guide](guide.md) — progressive walkthrough from loading data to fitting models.
- [API Reference](api/records.md) — auto-generated from docstrings.
