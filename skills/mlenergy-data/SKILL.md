---
name: mlenergy-data
description: Answer natural-language questions about The ML.ENERGY Benchmark (LLM and diffusion inference energy/latency/throughput measurement data on GPUs) using the `mlenergy-data` Python toolkit. Use when the user asks about energy per token, energy per image/video, throughput, ITL latency, power timelines, GPU comparisons, batch-size sweeps, MoE/Dense/Hybrid architecture comparisons, or FP8/BF16/MXFP4 precision comparisons.
license: Apache-2.0
compatibility: Requires Python 3.10+, `pip install mlenergy-data`, and a Hugging Face access token in `HF_TOKEN` (the dataset is gated).
---

# ML.ENERGY Data Toolkit

The toolkit (`mlenergy_data`) loads and analyzes [The ML.ENERGY Benchmark v3.0](https://huggingface.co/datasets/ml-energy/benchmark-v3) — LLM and diffusion inference runs on NVIDIA H100 and B200 GPUs, published as parquet summaries plus per-run raw JSONs on HF Hub.

Use this skill whenever the user wants to query, compare, aggregate, plot, or model the benchmark data.

## What the data covers (orient before coding)

- **Two domains**, each with its own collection class:
  - LLM (text + multimodal LLM): `LLMRuns` / `LLMRun`. Tasks: `gpqa`, `lm-arena-chat`, `sourcegraph-fim`, `image-chat`, `video-chat`. Image-chat and video-chat are LLMs that ingest images/videos — they are NOT diffusion.
  - Diffusion (text-to-image, text-to-video): `DiffusionRuns` / `DiffusionRun`. Tasks: `text-to-image`, `text-to-video`.
- GPUs: `H100`, `B200`. Multi-GPU configs at `num_gpus ∈ {1, 2, 4, 8}`.
- LLM batch sizes (`max_num_seqs`): from 8 up to 4096, mostly powers of 2 (8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096).
- Diffusion batch sizes: 1, 2, 4, 8, 16, 32.
- LLM precisions: `bfloat16`, `fp8`, `mxfp4` (mxfp4 is GPT OSS only). Diffusion precisions: only `bfloat16` in the current release.
- LLM architectures: `"Dense Transformer"`, `"MoE"`, `"Mamba-Transformer Hybrid"` (the literal strings — not "Dense"/"Hybrid").
- A complete catalog of models, tasks-per-model, GPU coverage, and image/video sizes is in [`references/DATA_CATALOG.md`](references/DATA_CATALOG.md). Read it before guessing model names or whether a (model, GPU, num_gpus) cell exists.

## Setup

**The dataset is gated on Hugging Face Hub.** The user must have requested access *and* set an `HF_TOKEN` for any `from_hf()` call to succeed.

If running inside this repo, `source .venv/bin/activate` first. Otherwise `pip install mlenergy-data` (or `uv pip install mlenergy-data`).

```python
from mlenergy_data.records import LLMRuns, DiffusionRuns

llm = LLMRuns.from_hf()           # default: stable_only=True, ~few MB parquet only
diff = DiffusionRuns.from_hf()
```

Both classes also have `from_directory(root)` (compiled local copy) and `from_raw_results(*roots)` (re-parse raw `results.json` files). Default to `from_hf()` unless the user specifies a local directory.

`from_hf()` only downloads the parquet summary. Raw files (per-request output lengths, per-token ITL, power timelines) auto-download on demand when you call `output_lengths()`, `inter_token_latencies()`, `timelines()`, or `read_results_json()`. **Always filter the collection before calling those methods** — the full raw dataset is ~100 GB.

### If the user hits a gated-dataset error

Recognize these signals from a `from_hf()` call (or any raw-data method that triggers a download):

- A `RuntimeError` containing the phrase `"This is a gated dataset"` (the toolkit wraps HF errors with a friendly message).
- A `huggingface_hub.errors.GatedRepoError` or `RepositoryNotFoundError`.
- HTTP 401 / 403 in the traceback, or a message about an expired/missing token.
- The user reports they can't load anything but the toolkit is installed correctly.

Don't just dump the traceback. Walk the user through the fix in plain language:

1. **Request access to the dataset.** Open https://huggingface.co/datasets/ml-energy/benchmark-v3 in a browser and click "Request access". Approval is automatic and usually instant — the page should show "You have been granted access" within seconds. If the user isn't logged into Hugging Face, they'll need a free account first (https://huggingface.co/join).
2. **Create an access token.** Go to https://huggingface.co/settings/tokens and create a new token. A "Read" token is sufficient — no write permissions needed. Copy the token (it starts with `hf_…`).
3. **Export it in the shell.** `export HF_TOKEN=hf_…` for the current session, or add it to `~/.zshrc` / `~/.bashrc` / `.env` for persistence. The user must do this in the *same shell* that runs Python — exporting in one terminal does not affect another.
4. **Retry.** Re-run the `from_hf()` call. If it still fails, ask the user to (a) confirm the access page now shows "granted", (b) verify `echo $HF_TOKEN` prints a non-empty `hf_…` string in the same shell.

If the user can't get access for any reason, stop and tell them — don't fabricate results.

## The collection model

`LLMRuns` and `DiffusionRuns` are **immutable, chainable, lazy filtered collections**. Every filter returns a new collection; you iterate to get individual frozen-dataclass records.

```python
gpqa_b200 = llm.task("gpqa").gpu_model("B200")     # chain freely
len(gpqa_b200)                                      # count
gpqa_b200[0]                                        # index
list(gpqa_b200)                                     # materialize
gpqa_b200.to_dataframe()                            # one row per run
```

Iteration yields typed records — the IDE/autocomplete works. Use `min`/`max`/`sorted` directly with a `key=` lambda; do not write helper functions.

## Filter cheat sheet

| Method | LLMRuns | DiffusionRuns | Notes |
|---|---|---|---|
| `task(*names)` | yes | yes | OR within args |
| `gpu_model(*names)` | yes | yes | e.g. `"H100"`, `"B200"` |
| `num_gpus(*counts)` or `num_gpus(min=, max=)` | yes | yes | exact OR range, not both |
| `max_num_seqs(*sizes)` or `max_num_seqs(min=, max=)` | yes | — | LLM batch size |
| `batch_size(*sizes)` or `batch_size(min=, max=)` | — | yes | diffusion batch size |
| `precision(*values)` | yes | yes | `"bfloat16"`, `"fp8"`, `"mxfp4"` |
| `architecture(*values)` | yes | — | `"Dense Transformer"`, `"MoE"`, `"Mamba-Transformer Hybrid"` |
| `nickname(*names)` | yes | yes | human-friendly display name |
| `model_id(*ids)` | yes | yes | `"org/model"` HF identifier |
| `where(predicate)` | yes | yes | arbitrary lambda |
| `stable()` / `unstable()` | yes | — | LLM-only (see Stability) |
| `group_by(*fields)` | yes | yes | returns `{value: subcollection}` |
| `+` (concat) | yes | yes | union of two collections |

Filters are AND across calls (`.task("gpqa").gpu_model("B200")` ⇒ both). Within a call, multiple positional args are OR (`.task("gpqa", "lm-arena-chat")`). Keyword `min=`/`max=` are inclusive bounds.

## Field cheat sheet

The full set of fields lives in the `LLMRun` and `DiffusionRun` docstrings (`mlenergy_data/records/runs.py`) and in `to_dataframe()`. The most common ones for analysis:

**LLMRun (35 public fields, see [`references/DATA_CATALOG.md`](references/DATA_CATALOG.md) for the full list):**
- Identity: `task`, `model_id`, `nickname`, `architecture`, `weight_precision`, `gpu_model`, `num_gpus`, `max_num_seqs`
- Size: `total_params_billions`, `activated_params_billions` (equals total for dense; smaller for MoE)
- Energy/throughput: `energy_per_token_joules`, `energy_per_request_joules`, `output_throughput_tokens_per_sec`, `request_throughput_req_per_sec`, `avg_power_watts`
- Workload: `total_output_tokens`, `completed_requests`, `avg_output_len`, `avg_batch_size` (actual concurrent sequences observed in steady state — usually < `max_num_seqs`)
- Latency: `mean_itl_ms`, `median_itl_ms`, `p50_itl_ms`, `p90_itl_ms`, `p95_itl_ms`, `p99_itl_ms` (ITL = inter-token latency)
- Parallelism: `tensor_parallel`, `expert_parallel`, `data_parallel`
- Stability: `is_stable`, `unstable_reason`

**DiffusionRun (22 public fields):**
- Identity: `task`, `model_id`, `nickname`, `weight_precision`, `gpu_model`, `num_gpus`, `batch_size`
- Generation params: `inference_steps`, `height`, `width`, `num_frames` (video only), `fps` (video only)
- Parallelism: `ulysses_degree`, `ring_degree`, `use_torch_compile`
- Energy: `energy_per_generation_joules` (generic — image OR video), `avg_power_watts`, `batch_latency_s`, `throughput_generations_per_sec`
- `is_text_to_image`, `is_text_to_video` are convenience properties.

DiffusionRun has no `architecture` field and no stability metadata. There is no `energy_per_image_joules` / `energy_per_video_joules` — use `energy_per_generation_joules` and check `r.task` if you need a label.

## Recipes for common questions

These map natural-language asks to short, idiomatic code. For longer or trickier patterns, [`references/RECIPES.md`](references/RECIPES.md) covers:

- **FP8 vs BF16 pairing of the same model** — strip the `" FP8"` nickname suffix, group on `(base, gpu, num_gpus, batch)`, compare energy/latency ratios.
- **Iso-latency comparison (B200 vs H100)** — for each (model, precision), find the lowest-energy config under a median-ITL deadline on each GPU, compute the gain.
- **Multi-GPU scaling: same batch, more GPUs** — count, across a task suite, how often adding GPUs at the same batch trades energy for latency.
- **Power timeline: median-aggregate across runs** — resample per-run timelines onto a common grid and take the median (the OpenG2G simulation pattern).
- **Output-length distribution per (model, task)** — leaderboard-style histograms with consistent bins.
- **Best-of-K with multiple constraints** — find the lowest-energy config that simultaneously meets median-ITL, p95-ITL, and throughput floors.
- **When a run isn't there** — handling empty subsets and the sparse `(model, gpu, num_gpus, task)` grid.

Read RECIPES.md only if the user's question matches one of the patterns above.

### "Which model has the lowest energy per token on GPQA?"
```python
best = min(llm.task("gpqa"), key=lambda r: r.energy_per_token_joules)
print(f"{best.nickname} on {best.num_gpus}x{best.gpu_model} batch={best.max_num_seqs}: "
      f"{best.energy_per_token_joules:.3f} J/tok")
```

### "Best energy efficiency per GPU type, on GPQA"
```python
for gpu, group in llm.task("gpqa").group_by("gpu_model").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{gpu}: {best.nickname} {best.num_gpus}x batch={best.max_num_seqs} "
          f"-> {best.energy_per_token_joules:.4f} J/tok")
```

### "Compare best efficiency between B200 and H100 on a task"
```python
chat = llm.task("lm-arena-chat")
b = min(chat.gpu_model("B200"), key=lambda r: r.energy_per_token_joules)
h = min(chat.gpu_model("H100"), key=lambda r: r.energy_per_token_joules)
delta = (h.energy_per_token_joules - b.energy_per_token_joules) / h.energy_per_token_joules * 100
print(f"B200: {b.nickname} {b.num_gpus}x batch={b.max_num_seqs} -> {b.energy_per_token_joules:.4f} J/tok")
print(f"H100: {h.nickname} {h.num_gpus}x batch={h.max_num_seqs} -> {h.energy_per_token_joules:.4f} J/tok")
print(f"B200 vs H100: {delta:+.1f}%  (positive = B200 wins)")
```

### "Best per model, restricted to B200"
```python
for nick, group in llm.task("gpqa").gpu_model("B200").group_by("nickname").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{nick}: {best.energy_per_token_joules:.4f} J/tok @ batch={best.max_num_seqs}")
```

### "Energy per token vs batch size for one model"
```python
runs = sorted(
    llm.task("gpqa").nickname("DeepSeek R1").gpu_model("B200").num_gpus(8),
    key=lambda r: r.max_num_seqs,
)
batches = [r.max_num_seqs for r in runs]
energies = [r.energy_per_token_joules for r in runs]
```

### "MoE vs dense at similar active parameters"
```python
gpqa = llm.task("gpqa").gpu_model("B200").precision("bfloat16")
for arch, group in gpqa.group_by("architecture").items():
    best = min(group, key=lambda r: r.energy_per_token_joules)
    print(f"{arch}: {best.nickname} ({best.activated_params_billions:.0f}B active) "
          f"@ {best.energy_per_token_joules:.3f} J/tok")
```

### "Find configs that meet a latency SLA, then minimize energy"
```python
median_itl_deadline_ms = 100
candidates = llm.task("lm-arena-chat").nickname("Qwen 3 235B A22B Instruct FP8").gpu_model("B200")
ok = candidates.where(lambda r: r.median_itl_ms <= median_itl_deadline_ms)
best = min(ok, key=lambda r: r.energy_per_token_joules) if ok else None
```

### "Lowest energy per image (text-to-image) on B200"
```python
t2i = diff.task("text-to-image").gpu_model("B200")
best = min(t2i, key=lambda r: r.energy_per_generation_joules)
print(f"{best.nickname} {best.height}x{best.width} batch={best.batch_size}: "
      f"{best.energy_per_generation_joules:.0f} J/image")
```

### "Average power timeline of a run"
```python
run = llm.task("gpqa").nickname("Qwen 3 8B").gpu_model("B200").num_gpus(1).max_num_seqs(64)[0]
tl = run.timelines(metric="power.device_instant")   # auto-downloads raw on first call
# columns: timestamp, relative_time_s, value, metric (long-form, summed across GPUs)
```

`timelines()` accepts `metric ∈ {"power.device_instant", "power.device_average", "temperature"}`. For LLM runs, the timeline is restricted to the steady-state window. For diffusion runs, it returns the full run.

### "Distribution of output lengths for a model"
```python
sub = llm.task("gpqa").nickname("DeepSeek R1").gpu_model("B200").num_gpus(8).max_num_seqs(128)
df = sub.output_lengths()                           # columns: task, model_id, num_gpus, max_num_seqs, output_len
df["output_len"].describe()
```

### "Per-token inter-token latency samples for distributional analysis"
```python
sub = llm.task("lm-arena-chat").nickname("Qwen 3 8B").gpu_model("B200").num_gpus(1).max_num_seqs(64)
df = sub.inter_token_latencies()                    # columns: ..., itl_s (seconds)
```

## Bulk raw data — when and how

By default, only the parquet summary is downloaded (~few MB). The collection is enough to answer any aggregate-stat question. Reach for raw files only when:

- **Power/temperature timeline** of a run → `run.timelines(metric=...)` or `runs.timelines(metric=...)`.
- **Per-request output lengths** → `run.output_lengths()` or `runs.output_lengths()`.
- **Per-token inter-token latencies** (for distributional analysis or percentiles beyond what's pre-computed) → `run.inter_token_latencies()` or `runs.inter_token_latencies()`.
- **Anything not exposed as a field** → `run.read_results_json()` returns the parsed `results.json` payload; `run.read_prometheus_json()` returns vLLM/Prometheus stats (LLM only).

Auto-download is per-record and lazy — the first `timelines()` or `output_lengths()` call on a run hits the Hub once, then HF caches it locally (`HF_HOME`). To pre-warm in parallel:

```python
sub = llm.task("gpqa").gpu_model("B200").download_raw_files()  # both results + prometheus
sub = llm.task("gpqa").gpu_model("B200").download_raw_files(file="results")  # results only
```

Always filter before `download_raw_files()` (or before bulk methods) — the unfiltered dataset is ~100 GB.

## Units

Mixing these up silently produces wrong answers — there's no type system to catch it.

- **Energy**: joules (`*_joules`).
- **Power**: watts (`avg_power_watts`).
- **Duration**: seconds (`steady_state_duration_seconds`, `batch_latency_s`).
- **ITL percentile fields** (`mean_itl_ms`, `median_itl_ms`, `p50/p90/p95/p99_itl_ms`): **milliseconds**.
- **Raw ITL samples** from `run.inter_token_latencies()` and `runs.inter_token_latencies()`: **seconds**.
- **Throughput**: tokens/sec (LLM `output_throughput_tokens_per_sec`), requests/sec (LLM `request_throughput_req_per_sec`), generations/sec (diffusion `throughput_generations_per_sec`).

When comparing a percentile against a deadline, the deadline must be in ms.

## Critical gotchas

[`references/PITFALLS.md`](references/PITFALLS.md) has the full list (API edge cases, ITL smoothing semantics for raw JSON, reproducibility notes). The ones that cause silent wrong answers — read these every time:

1. **FP8 nickname suffix is inconsistent.** Most FP8 variants have `" FP8"` appended to the BF16 nickname (e.g. `"Qwen 3 235B A22B Instruct"` ↔ `"Qwen 3 235B A22B Instruct FP8"`). But `DeepSeek R1` and `DeepSeek V3.1` are **natively FP8** — the nickname has no suffix, and `weight_precision == "fp8"`. To enumerate FP8 vs BF16 pairs of the *same architecture*, key on the BF16 nickname stripped of `" FP8"` (see `_analyze_fp8_vs_bf16` in the blog script).

2. **Architecture strings are literal.** Use `"Dense Transformer"` (not `"Dense"`), `"MoE"`, and `"Mamba-Transformer Hybrid"`. README/docs occasionally show `"Dense"` for brevity — that string does not match.

3. **Stability is LLM-only.** `LLMRuns.from_hf()` defaults to `stable_only=True` and excludes runs that failed quality checks (low batch utilization, short steady state, cascade-from-unstable-batch). To inspect rejected runs, call `LLMRuns.from_hf(stable_only=False)` and use `.unstable()`. `DiffusionRuns` has no stability filtering — every parquet row is returned.

4. **`max_num_seqs` is the configured cap, `avg_batch_size` is what was actually achieved** during steady state. They differ when client request load couldn't fill the configured batch.

5. **Image-chat and video-chat are LLM tasks**, not diffusion. They use multimodal LLMs (`Qwen 3 VL`, `Llama 4 Scout`, etc.) and live in `LLMRuns`.

6. **Diffusion has only `bfloat16`** in the current release. Filtering `diff.precision("fp8")` returns an empty collection. Diffusion has no `is_stable`, no `architecture`, and no `use_torch_compile=True` runs.

7. **Bulk methods auto-download**, so an unfiltered `runs.timelines()` or `runs.download_raw_files()` will download ~100 GB. Always filter first.

8. **Tasks are not interchangeable.** Each model is benchmarked on a specific subset of tasks (see [`references/DATA_CATALOG.md`](references/DATA_CATALOG.md)). DeepSeek R1 has only `gpqa`; Qwen 3 Coder has only `sourcegraph-fim`; etc. Always check coverage before requesting a comparison.

9. **`energy_per_request_joules` and `request_throughput_req_per_sec` can be `None`** when the run had no completed requests or `avg_output_len` is missing. The identity `avg_output_len × energy_per_token_joules ≈ energy_per_request_joules` holds only when the right-hand side is non-null. Don't multiply blindly.

10. **`group_by` keys are tuples for multi-field groupings.** `runs.group_by("model_id", "task")` returns `{(model_id, task): subcollection}`.

11. **Power timeline `value` is summed across GPUs and windowed differently per domain.** `LLMRun.timelines()` slices to the steady-state window (post-warm-up); `DiffusionRun.timelines()` returns the full run. The `value` column is total power across all GPUs — divide by `r.num_gpus` for per-GPU, and don't compare LLM and diffusion timelines without accounting for the different windowing.

## Working code-first

When the user asks an analysis question, default to:

1. Filter the collection (one chain — no intermediate variables unless they're reused).
2. Iterate / `min` / `max` / `sorted` with a `key=` lambda for the answer.
3. Print the answer in a single line that names the model, GPU, batch, and the metric value with units.
4. Reach for `to_dataframe()` only if the user wants a table or a multi-column summary; reach for raw bulk methods only if the answer truly requires per-request or per-token data.

If the user asks for a plot, follow project plotting conventions (zero-anchored axes, deterministic SVG when applicable). The blog script ([`analysis/ml-energy-leaderboard-v3.0.py`](https://github.com/ml-energy/blog/blob/master/analysis/ml-energy-leaderboard-v3.0.py)) is the reference for production-quality plots.

## Where to read more

- **API reference** (auto-generated from docstrings): https://ml.energy/data/api/records/ and https://ml.energy/data/api/modeling/
- **Guide** (progressive walkthrough): https://ml.energy/data/guide/
- **Real-world examples** (study these for production patterns):
  - Leaderboard JSON build: https://github.com/ml-energy/leaderboard/blob/master/scripts/build_data.py
  - Blog figures (FP8 pairing, iso-latency, multi-GPU): https://github.com/ml-energy/blog/blob/master/analysis/ml-energy-leaderboard-v3.0.py
  - OpenG2G simulation traces and ITL fits: https://github.com/gpu2grid/openg2g/blob/master/openg2g/datacenter/workloads/inference.py
- **Bundled references in this skill**:
  - [`references/DATA_CATALOG.md`](references/DATA_CATALOG.md) — exhaustive enumeration of models, tasks, GPU coverage, sizes
  - [`references/RECIPES.md`](references/RECIPES.md) — longer analysis recipes
  - [`references/PITFALLS.md`](references/PITFALLS.md) — extended gotchas list
