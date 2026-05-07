# Recipes

Longer, idiomatic patterns drawn from real production code (the leaderboard, blog, and OpenG2G use cases). Read these when the user asks for an analysis that goes beyond a single filter+min.

## FP8 vs BF16 pairing of the same model

Most FP8 variants append `" FP8"` to the BF16 nickname. Pair them by stripping the suffix; key on the (base nickname, gpu, num_gpus, batch). Skip DeepSeek R1/V3.1 — they are natively FP8 and have no BF16 sibling.

```python
from collections import defaultdict
from mlenergy.data.records import LLMRuns

llm = LLMRuns.from_hf()

groups: dict[tuple, dict[str, "LLMRun"]] = defaultdict(dict)
for r in llm.task("lm-arena-chat"):
    base = r.nickname.removesuffix(" FP8")
    key = (base, r.gpu_model, r.num_gpus, r.max_num_seqs)
    groups[key][r.weight_precision] = r

for (base, gpu, n_gpus, batch), prec_runs in groups.items():
    if "fp8" in prec_runs and "bfloat16" in prec_runs:
        fp8, bf16 = prec_runs["fp8"], prec_runs["bfloat16"]
        ratio = fp8.energy_per_token_joules / bf16.energy_per_token_joules
        print(f"{base} on {n_gpus}x{gpu} batch={batch}: FP8 / BF16 = {ratio:.2f}")
```

## Iso-latency comparison (B200 vs H100)

For each (model, precision) that has runs on both GPUs, find the lowest-energy config that meets a median-ITL deadline.

```python
def best_for_deadline(runs, model, gpu, precision, deadline_ms, latency_attr, energy_attr):
    matching = [
        r for r in runs
        if r.nickname == model and r.gpu_model == gpu
        and r.weight_precision == precision
        and getattr(r, latency_attr) <= deadline_ms
    ]
    return min(matching, key=lambda r: getattr(r, energy_attr)) if matching else None

llm = LLMRuns.from_hf()
gpqa = llm.task("gpqa")

deadline_ms = 100
for model, prec in sorted({(r.nickname, r.weight_precision) for r in gpqa}):
    h = best_for_deadline(gpqa, model, "H100", prec, deadline_ms, "median_itl_ms", "energy_per_token_joules")
    b = best_for_deadline(gpqa, model, "B200", prec, deadline_ms, "median_itl_ms", "energy_per_token_joules")
    if h and b:
        gain = (h.energy_per_token_joules - b.energy_per_token_joules) / h.energy_per_token_joules * 100
        print(f"{model:40s} {prec:8s} H100={h.energy_per_token_joules:.3f}  B200={b.energy_per_token_joules:.3f}  gain={gain:+.0f}%")
```

For diffusion, swap to `r.batch_latency_s` (latency) and `r.energy_per_generation_joules` (energy).

## Multi-GPU scaling: same batch, more GPUs

Across all (model, gpu_type) groups, count how often adding GPUs *at the same batch size* trades energy for latency.

```python
from collections import defaultdict

llm = LLMRuns.from_hf().task("gpqa", "lm-arena-chat", "sourcegraph-fim")
groups: dict[tuple[str, str], dict[int, list]] = defaultdict(lambda: defaultdict(list))
for r in llm:
    groups[(r.nickname, r.gpu_model)][r.num_gpus].append(r)

for gpu in ("B200", "H100"):
    e_up = lat_down = total = 0
    for (nick, g), per_count in groups.items():
        if g != gpu:
            continue
        counts = sorted(per_count.keys())
        for i, n in enumerate(counts):
            for m in counts[i + 1:]:
                a = {r.max_num_seqs: r for r in per_count[n]}
                b = {r.max_num_seqs: r for r in per_count[m]}
                for batch in set(a) & set(b):
                    total += 1
                    if b[batch].energy_per_token_joules >= a[batch].energy_per_token_joules:
                        e_up += 1
                    if b[batch].median_itl_ms < a[batch].median_itl_ms:
                        lat_down += 1
    print(f"{gpu}: more GPUs at same batch -> energy up {e_up}/{total}, latency down {lat_down}/{total}")
```

## Power timeline: median-aggregate across runs

Used by OpenG2G to build a representative per-GPU power trace per (model, batch_size, num_gpus). Resample each run to a common grid, then take the median.

```python
import numpy as np
import pandas as pd

sub = (
    llm.task("lm-arena-chat")
       .nickname("Llama 3.1 8B Instruct")
       .gpu_model("H100")
       .num_gpus(1)
       .max_num_seqs(64)
       .download_raw_files(file="results")
)

# Concatenate timelines from each run
frames = []
for i, run in enumerate(sub):
    tl = run.timelines(metric="power.device_instant")
    tl["run_index"] = i
    frames.append(tl)
all_tl = pd.concat(frames, ignore_index=True)

dt_s = 0.1
t_end = float(np.median([all_tl[all_tl.run_index == i].relative_time_s.max() for i in range(len(sub))]))
grid = np.arange(0.0, t_end + 1e-12, dt_s)
mat = np.vstack([
    np.interp(grid, g.relative_time_s.to_numpy(), g.value.to_numpy())
    for _, g in all_tl.groupby("run_index", sort=True)
])
median_trace_w = np.median(mat, axis=0)   # per-timestep median power across runs (total across GPUs)
per_gpu_w = median_trace_w / sub[0].num_gpus
```

The full version with multi-batch grouping lives at `_build_trace_store_from_timelines` in OpenG2G's `inference.py`.

## Output-length distribution per (model, task)

For leaderboard-style histograms with consistent bins across configurations:

```python
import numpy as np

sub = llm.task("gpqa").nickname("DeepSeek R1").download_raw_files(file="results")
all_lengths = []
per_run = []
for r in sub:
    lens = np.array(r.output_lengths(), dtype=int)
    per_run.append(lens)
    all_lengths.extend(lens.tolist())

_, bin_edges = np.histogram(all_lengths, bins=50)
agg_counts, _ = np.histogram(all_lengths, bins=bin_edges)
per_run_counts = [np.histogram(lens, bins=bin_edges)[0] for lens in per_run]
```

## Best-of-K with multiple constraints

Find the lowest energy-per-token config that simultaneously meets:
- median ITL deadline
- p95 ITL deadline
- minimum throughput floor

```python
ok = llm.task("lm-arena-chat").nickname("Qwen 3 32B").gpu_model("B200").where(
    lambda r: r.median_itl_ms <= 80
              and r.p95_itl_ms <= 250
              and r.output_throughput_tokens_per_sec >= 800
)
best = min(ok, key=lambda r: r.energy_per_token_joules) if ok else None
```

## When a run isn't there

If `min(empty_collection, key=...)` raises `ValueError`, that's a sign the (model, GPU, num_gpus, task) cell isn't populated. Check `len(sub)` before reducing, or wrap with `if sub:`. The catalog ([`DATA_CATALOG.md`](DATA_CATALOG.md)) lists what cells exist.
