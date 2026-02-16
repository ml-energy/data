# Guide

## Real-world examples

For full working examples of the toolkit in production, see:

- [ML.ENERGY Leaderboard data build](TODO) -- builds the leaderboard JSON data from benchmark runs
- [ML.ENERGY Blog analysis scripts](TODO) -- generates figures for the ML.ENERGY blog
- [OpenG2G simulation data build](TODO) -- builds power traces, logistic fits, and latency fits for grid simulation

## Loading benchmark runs

`LLMRuns` and `DiffusionRuns` are typed, immutable collections.
Each run is a frozen dataclass (`LLMRun` / `DiffusionRun`) with IDE autocomplete and type checking.

```python
from mlenergy_data.records import LLMRuns, DiffusionRuns

# Load all stable LLM runs from a compiled data directory
runs = LLMRuns.from_directory("/path/to/compiled/data")

# Include unstable runs
runs = LLMRuns.from_directory(root, stable_only=False)

# Diffusion runs
diff = DiffusionRuns.from_directory(root)

# Load from Hugging Face Hub
runs = LLMRuns.from_hf()
diff = DiffusionRuns.from_hf()
```

## Filtering

All filter methods return a new collection — chain freely:

```python
# Single filter
gpqa = runs.task("gpqa")
h100 = runs.gpu("H100")
fp8 = runs.precision("fp8")

# Chained filters
best_candidates = runs.task("gpqa").gpu("B200").precision("fp8")

# Multiple values (OR within a filter)
chat_or_gpqa = runs.task("gpqa", "lm-arena-chat")

# By nickname
deepseek = runs.nickname("DeepSeek R1")

# Batch size: exact values or range
batch_128 = runs.batch(128)
large_batch = runs.batch(min=64)
mid_batch = runs.batch(min=16, max=128)

# GPU count: exact or range
single_gpu = runs.num_gpus(1)
multi_gpu = runs.num_gpus(min=2)

# Arbitrary predicate
big_models = runs.where(lambda r: r.total_params_billions > 70)
```

## Data access

There are two ways to access run data:

**Per-record (row):** Iterate the collection to get individual typed records.

```python
for r in runs.task("gpqa"):
    print(r.energy_per_token_joules, r.nickname)

best = min(runs.task("gpqa"), key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok")
```

**Per-field (column):** Use the `data` property for typed field arrays.

```python
energies = runs.data.energy_per_token_joules  # list[float]
gpus = runs.data.num_gpus                     # list[int]
names = runs.data.nickname                    # list[str]
```

The `data` property provides full IDE autocomplete and type checking.
Each attribute returns a `list[T]` with one element per run, in iteration order.

```python
import matplotlib.pyplot as plt

plt.scatter(runs.data.max_num_seqs, runs.data.energy_per_token_joules)
plt.xlabel("Batch size")
plt.ylabel("Energy per token (J)")
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

## Analysis patterns

Python is the analysis layer — no special helper functions needed:

```python
# Best energy per token for each model on a task
for model_id, group in runs.task("gpqa").group_by("model_id").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{best.nickname}: {best.energy_per_token_joules:.3f} J/tok")

# Comparing GPUs
llama70b = runs.model("meta-llama/Llama-3.1-70B-Instruct")
for gpu, g in llama70b.group_by("gpu_model").items():
    plt.scatter(g.data.max_num_seqs, g.data.energy_per_token_joules, label=gpu)
plt.legend()
```

## Bulk data

These methods return pandas DataFrames for numerical analysis.
When loaded from HF Hub (`from_hf()`), they automatically download only the raw files needed for the current collection. The download scope is determined by your filters. HF Hub caches files locally, so repeated calls are instant.

```python
# Power timelines (long-form)
power_tl = runs.timelines(metric="power.device_instant")
# Columns: results_path, domain, task, model_id, num_gpus, max_num_seqs, batch_size, timestamp, relative_time_s, value, metric

# Temperature timelines
temp_tl = runs.timelines(metric="temperature")

# Output lengths
out_df = runs.output_lengths()
# Columns: results_path, task, model_id, num_gpus, max_num_seqs, output_len, success

# Full DataFrame (one row per run, all fields as columns)
df = runs.to_dataframe()
```

## Diffusion runs

`DiffusionRuns` follows the same patterns:

```python
from mlenergy_data.records import DiffusionRuns

diff = DiffusionRuns.from_directory(root)
t2i = diff.task("text-to-image")
best = min(t2i, key=lambda r: r.energy_per_generation_joules)
print(f"{best.nickname}: {best.energy_per_generation_joules:.3f} J/image")

# Available filters: task(), model(), gpu(), nickname(), batch(),
# num_gpus(), precision(), where()
```

## Model fitting

### Logistic curves

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

### ITL latency distributions

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

## Building a Hugging Face data package

```bash
python data_publishing/build_hf_data.py \
  --results-dir /path/to/llm/h100/current/run \
  --results-dir /path/to/diffusion/h100/current/run \
  --out-dir /tmp/hf_pkg

# Upload to Hugging Face Hub
hf upload-large-folder ml-energy/benchmark-v3 /tmp/hf_pkg --repo-type dataset
```
