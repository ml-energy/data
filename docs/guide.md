# Toolkit Guide

## Real-World Examples

For full working examples of the toolkit in production, see:

| Project | Description | Script |
|---------|-------------|-------|
| [The ML.ENERGY Leaderboard](https://ml.energy/leaderboard) | Builds the leaderboard JSON data from benchmark runs | [Link](https://github.com/ml-energy/leaderboard/blob/master/scripts/build_data.py) |
| [The ML.ENERGY Blog](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/) | Analysis for the blog post on the V3 benchmark results | [Link](https://github.com/ml-energy/blog/blob/master/analysis/ml-energy-leaderboard-v3.0.py) |
| [OpenG2G Simulation](TODO) | Power traces and models for datacenter–grid simulation | [Link](TODO) |


## Dataset Access

The benchmark dataset ([`ml-energy/benchmark-v3`](https://huggingface.co/datasets/ml-energy/benchmark-v3)) is gated on Hugging Face Hub.
Before loading data with `from_hf()`, you need to:

1. Visit the [dataset page](https://huggingface.co/datasets/ml-energy/benchmark-v3) and request access (granted automatically).
2. Set the `HF_TOKEN` environment variable to a [Hugging Face access token](https://huggingface.co/settings/tokens).

## Loading Benchmark Runs

`LLMRuns` and `DiffusionRuns` are typed, immutable collections.
Each run is a frozen dataclass (`LLMRun` / `DiffusionRun`) with IDE autocomplete and type checking.

```python
from mlenergy_data.records import LLMRuns, DiffusionRuns

# Load from Hugging Face Hub
runs = LLMRuns.from_hf()

# Diffusion runs
diff = DiffusionRuns.from_hf()
```

Or load from a local compiled data directory:

```python {test="skip"}
root = "/path/to/compiled/data"
runs = LLMRuns.from_directory(root)
runs = LLMRuns.from_directory(root, stable_only=False)
diff = DiffusionRuns.from_directory(root)
```

!!! Note
    A "compiled data directory" is one built by `data_publishing/build_hf_data.py` (or downloaded from HF Hub). It contains parquet summary files under `runs/`, raw result files under `llm/` and `diffusion/`, and benchmark config files under `configs/`.

## Filtering

All filter methods return a new collection — chain freely:

```python
# Single filter
gpqa = runs.task("gpqa")
h100 = runs.gpu_model("H100")
fp8 = runs.precision("fp8")

# Chained filters
best_candidates = runs.task("gpqa").gpu_model("B200").precision("fp8")

# Multiple values (OR within a filter)
chat_or_gpqa = runs.task("gpqa", "lm-arena-chat")

# By nickname
deepseek = runs.nickname("DeepSeek R1")

# Architecture (LLM only)
moe_models = runs.architecture("MoE")

# Batch size: exact values or range
batch_128 = runs.max_num_seqs(128)
large_batch = runs.max_num_seqs(min=64)
mid_batch = runs.max_num_seqs(min=16, max=128)

# GPU count: exact or range
single_gpu = runs.num_gpus(1)
multi_gpu = runs.num_gpus(min=2)

# Arbitrary predicate
big_models = runs.where(lambda r: r.total_params_billions > 70)
```

## Data Access

Iterate the collection to get individual typed records:

```python
for r in runs.task("gpqa"):
    print(r.energy_per_token_joules, r.nickname)

best = min(runs.task("gpqa"), key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok")
```

```python
import matplotlib.pyplot as plt

plt.scatter(
    [r.max_num_seqs for r in runs],
    [r.energy_per_token_joules for r in runs],
)
plt.xlabel("Batch size")
plt.ylabel("Energy per token (J)")
```

**Indexing and concatenation:**

```python
first_run = runs[0]

# Concatenate collections
h100 = runs.gpu_model("H100")
b200 = runs.gpu_model("B200")
combined = h100 + b200
```

## Grouping

```python
# Group by task
for task, group in runs.group_by("task").items():
    print(f"{task}: {len(group)} runs")

# Group by multiple fields
for (model, batch), g in runs.group_by("model_id", "max_num_seqs").items():
    best = min(g, key=lambda r: r.energy_per_token_joules)
    print(f"{model} @ batch={batch}: {best.energy_per_token_joules:.3f} J/tok")
```

## Analysis Patterns

Python is the analysis layer — no special helper functions needed:

```python
# Compare GPU generations on a task
for gpu, group in runs.task("lm-arena-chat").group_by("gpu_model").items():
    best = min(group, key=lambda r: r.output_throughput_tokens_per_sec)
    print(f"{gpu}: {best.nickname} @ {best.output_throughput_tokens_per_sec:.0f} tok/s")

# Comparing GPUs for a specific model
llama70b = runs.model_id("meta-llama/Llama-3.1-70B-Instruct")
for gpu, g in llama70b.group_by("gpu_model").items():
    plt.scatter(
        [r.max_num_seqs for r in g],
        [r.energy_per_token_joules for r in g],
        label=gpu,
    )
plt.legend()
```

## Bulk Raw Data

By default, only the compact Parquet files are downloaded, but for some analyses, you need the raw results file.

These methods return pandas DataFrames for numerical analysis.
When loaded from HF Hub (`from_hf()`), the library automatically downloads only the raw files needed for the current collection. The download scope is determined by your filters (e.g., `task()`, `gpu_model()`). HF Hub caches files locally, so repeated calls are instant.

To eagerly download all raw files upfront, use `download_raw_files()`:

```python {test="skip-bulk"}
# Eagerly download all raw files for a filtered collection
runs = LLMRuns.from_hf().task("lm-arena-chat").gpu_model("H100").download_raw_files()
power_tl = runs.timelines(metric="power.device_instant")  # no download delay
```

```python {test="skip"}
# Power timelines (long-form)
power_tl = runs.timelines(metric="power.device_instant")

# Temperature timelines
temp_tl = runs.timelines(metric="temperature")

# Output lengths
out_df = runs.output_lengths()

# Full DataFrame (one row per run, all fields as columns)
df = runs.to_dataframe()
```

Per-record methods give you direct access to raw data without needing a join key:

```python {test="skip"}
run = runs[0]

# Parsed results.json (downloaded lazily from HF Hub, then cached)
data = run.read_results_json()

# Per-record data extraction
lengths = run.output_lengths()
itls = run.inter_token_latencies()
tl = run.timelines(metric="power.device_instant")
```

## Diffusion Runs

`DiffusionRuns` follows the same patterns:

```python
from mlenergy_data.records import DiffusionRuns

diff = DiffusionRuns.from_hf()
t2i = diff.task("text-to-image")
best = min(t2i, key=lambda r: r.energy_per_generation_joules)
print(f"{best.nickname}: {best.energy_per_generation_joules:.3f} J/image")

# Task field and convenience properties
r = diff[0]
r.task               # "text-to-image" or "text-to-video"
r.is_text_to_image   # True for text-to-image tasks
r.is_text_to_video   # True for text-to-video tasks

# Available filters: task(), model(), gpu(), nickname(), batch(),
# num_gpus(), precision(), where()
```

## Model Fitting

### Logistic Curves

`LogisticModel` models a four-parameter logistic `y = b0 + L * sigmoid(k * (x - x0))` where `x = log2(batch_size)`:

```python
import numpy as np
from mlenergy_data.modeling import LogisticModel

# Fit from data
x = np.log2([8, 16, 32, 64, 128, 256])
y_power = np.array([200, 250, 320, 400, 480, 530])
fit = LogisticModel.fit(x, y_power)

# Evaluate at a specific batch size
predicted = fit.eval(batch=128)

# Serialize / deserialize
d = fit.to_dict()  # {"L": ..., "x0": ..., "k": ..., "b0": ...}
fit2 = LogisticModel.from_dict(d)
```

### ITL Latency Distributions

`ITLMixtureModel` fits a two-component lognormal mixture for inter-token latency:

```python
from mlenergy_data.modeling import ITLMixtureModel

# Fit from raw ITL samples (seconds)
model = ITLMixtureModel.fit(itl_samples_s, max_samples=2048, seed=0)

# Analytical mean and variance
mean, var = model.mean_var()

# Simulate average ITL across replicas
rng = np.random.default_rng(0)
avg_itl = model.sample_avg(n_replicas=180, rng=rng)

# Serialize / deserialize
d = model.to_dict()
model2 = ITLMixtureModel.from_dict(d)
```
