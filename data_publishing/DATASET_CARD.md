# The ML.ENERGY Benchmark Dataset v3

This dataset contains benchmark results from [The ML.ENERGY Benchmark](https://github.com/ml-energy/benchmark).
You can use [The ML.ENERGY Leaderboard](https://ml.energy) to explore the benchmarking results.

## Subsets

- **llm**: LLM benchmark runs with full metrics (energy, throughput, latency, etc.)
- **diffusion**: Diffusion model benchmark runs

## Usage

You can programmatically utilize the dataset using the ML.ENERGY data toolkit.

```bash
pip install mlenergy-data
```

```python
from mlenergy_data.records import LLMRuns, DiffusionRuns

# Load (fast, parquet only ~few MB)
llm = LLMRuns.from_hf("ml-energy/benchmark-v3")

# Filter and analyze (parquet only, no download)
for r in llm.task("gpqa").gpu("B200"):
    print(r.nickname, r.energy_per_token_joules)

# Bulk data methods auto-download raw files as needed
out = llm.task("gpqa").output_lengths()
```

## Schema

### LLM Runs

| Field | Type | Description |
|-------|------|-------------|
| task | str | Benchmark task (gpqa, lm-arena-chat, etc.) |
| model_id | str | Model identifier (org/model) |
| nickname | str | Human-readable model name |
| gpu_model | str | GPU model (H100, B200) |
| num_gpus | int | Number of GPUs used |
| max_num_seqs | int | Maximum batch size |
| energy_per_token_joules | float | Energy per output token (J) |
| output_throughput_tokens_per_sec | float | Output throughput (tok/s) |
| avg_power_watts | float | Average GPU power (W) |
| mean_itl_ms | float | Mean inter-token latency (ms) |
| is_stable | bool | Whether the run passed stability checks |

See the parquet files for the full list of fields.

### Diffusion Runs

| Field | Type | Description |
|-------|------|-------------|
| task | str | Benchmark task (text-to-image, text-to-video) |
| model_id | str | Model identifier (org/model) |
| nickname | str | Human-readable model name |
| gpu_model | str | GPU model (H100, B200, etc.) |
| num_gpus | int | Number of GPUs used |
| batch_size | int | Batch size |
| energy_per_generation_joules | float | Energy per generation (J) |
| height | int | Output height in pixels |
| width | int | Output width in pixels |
| batch_latency_s | float | Batch latency (s) |
| avg_power_watts | float | Average GPU power (W) |

## Issues and Discussions

Please direct any issues, questions, or discussions to [The ML.ENERGY Data Toolkit GitHub repository](https://github.com/ml-energy/data).

## Citation

```bibtex
@inproceedings{mlenergy-neuripsdb25,
    title={The {ML.ENERGY Benchmark}: Toward Automated Inference Energy Measurement and Optimization}, 
    author={Jae-Won Chung and Jeff J. Ma and Ruofan Wu and Jiachen Liu and Oh Jun Kweon and Yuxuan Xia and Zhiyu Wu and Mosharaf Chowdhury},
    year={2025},
    booktitle={NeurIPS Datasets and Benchmarks},
}
```

## License

Apache-2.0
