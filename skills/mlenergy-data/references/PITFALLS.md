# Extended Pitfalls

The most critical pitfalls — the ones that cause silent wrong answers — are inlined in [`SKILL.md`](../SKILL.md) ("Critical gotchas" and "Units"). This file holds the longer-tail items: API edge cases, raw-JSON details, and reproducibility notes.

## API edge cases

### `num_gpus` / `max_num_seqs` / `batch_size` filters cannot mix exact + range

```python
runs.num_gpus(1, 2)            # exact OR
runs.num_gpus(min=2, max=8)    # inclusive range
runs.num_gpus(1, min=2)        # raises ValueError
runs.num_gpus()                # raises TypeError (no args)
```

### `unstable()` requires a non-stable-only collection

```python
LLMRuns.from_hf().unstable()                      # raises ValueError
LLMRuns.from_hf(stable_only=False).unstable()     # works
```

### `group_by` key shape depends on arg count

- One field: keys are scalars. `group_by("task")` → `{"gpqa": LLMRuns(...), ...}`.
- Multiple fields: keys are tuples. `group_by("model_id", "task")` → `{("Qwen/...", "gpqa"): LLMRuns(...), ...}`.

### `to_dataframe()` strips private fields

Path fields (`_results_path`, `_prometheus_path`) and HF metadata (`_hf_repo_id`, etc.) are excluded. Use `r.read_results_json()` or `r.read_prometheus_json()` to access raw payloads — don't try to read paths from a `to_dataframe()` row.

### `download_raw_files()` returns the same collection

It's an in-place pre-warm — chain it for ergonomics; the return value is the same `LLMRuns`/`DiffusionRuns` object. The optional `file=` arg can be `"results"` or `"prometheus"` for LLM; default is both. For diffusion, only `results.json` exists.

### Cache lives in `HF_HOME`

Repeated calls after the first download are instant. To force a re-download, clear the HF cache or pass `revision=` to `from_hf()`.

## Raw JSON details

### Inter-token latencies in raw `results.json` are not smoothed

For chunked-prefill engines (vLLM), some ITL entries are zero — the engine batched multiple decode tokens into one step. The toolkit's `inter_token_latencies()` and the `*_itl_ms` percentile fields **already handle this** by spreading the accumulated latency across the covered tokens (`_smooth_chunked_itl` in `runs.py`).

Only an issue if you read raw `results.json` directly via `run.read_results_json()` — then you'll see a long zero spike at the front. Use the toolkit's accessors unless you specifically want raw values.

### Per-device timeline columns

`run.timelines()` exposes only the GPU-summed `value`. Per-device samples are present in raw `results.json` under `timeline.power.device_instant.<gpu_id>` if needed. The internal helper `mlenergy_data.records.timelines.extract_device_timeline` returns wide-form per-device columns plus the total — use it directly if you're doing per-GPU analysis.

### Stability cascade rule

A run is unstable if any of: `steady_state_duration < 20s`, `energy_per_token_joules <= 0`, `avg_batch_size / max_num_seqs < 0.85` ("low_batch_utilization"), or a smaller batch in the same `(model, task, GPU, num_gpus)` group was unstable ("cascade_from_unstable_batch:N"). The cascade reflects the assumption that smaller batches are the safe baseline; if the smallest batch is unstable, larger batches don't get a free pass.

To diagnose:

```python
all_llm = LLMRuns.from_hf(stable_only=False)
{r.unstable_reason.split(":")[0] for r in all_llm.unstable()}
```

In the current release: ~95% of unstable runs are `low_batch_utilization`, the remainder are `cascade_from_unstable_batch` and `short_steady_state`.

## Coverage

### (model, GPU, num_gpus, task) cells are sparse

The catalog ([`DATA_CATALOG.md`](DATA_CATALOG.md)) is the authoritative list. When a subset comes up empty for a comparison, prefer to skip and tell the user, rather than fabricate by widening filters.

### Across-the-board comparison: intersect available tasks first

```python
candidates = {
    nick for nick, group in llm.group_by("nickname").items()
    if {"gpqa", "lm-arena-chat"} <= {r.task for r in group}
}
```

## Field semantics

### Diffusion's single `energy_per_generation_joules`

The toolkit uses one generic field for both image and video runs. Check `r.is_text_to_image` / `r.is_text_to_video` (or `r.task`) to decide labels and which size fields are populated. `num_frames` and `fps` are `None` for image runs. The leaderboard-build script splits this into `energy_per_image_joules` / `energy_per_video_joules` in its JSON output, but that's a downstream rename, not a toolkit field.

## `mlenergy_data.modeling` is not recommended for general use

The toolkit exports `LogisticModel` (4-parameter sigmoid for batch sweeps) and `ITLMixtureModel` (two-component lognormal for inter-token latency). Both classes work, but their fit quality is unreliable enough that the skill does not surface them. Concretely, on the v3 dataset:

- **`LogisticModel`** on `energy_per_token_joules`, `avg_power_watts`, `median_itl_ms`, `output_throughput_tokens_per_sec` vs `log2(batch_size)`: median R² ≥ 0.98 across (model, task, gpu, num_gpus) groups, but **22%–38% of groups have max relative residuals above 15%, with worst cases over 100%**. There's no in-band signal that tells you whether your fit is in the good majority or the bad tail.
- **`ITLMixtureModel`**: median empirical-mean recovery is 2%, but **p90 is 172% and the worst case is 1430%** (model predicts ~15× the true mean). Vision-language models on `image-chat` and `video-chat` are the consistent failure mode — the lognormal "stall" component fits a heavy tail that blows up the analytical mean.

If a user explicitly asks for these fits, use them — but always validate by:
- For `LogisticModel`: compute residuals against the actual data points (`fit.eval(batch=b)` vs the measured value at each `b` in your sweep), and reject the fit if `max(|residual / y|) > 0.15` or any held-out point disagrees by more than ~10%.
- For `ITLMixtureModel`: compare `fit.mean_var()` to `np.mean(samples_s), np.std(samples_s)` and reject if either differs by more than ~10–20%. KS-test the model's `sample_one` draws against the empirical distribution.

For most analyses, prefer interpolating directly between measured batch sizes (the dataset is dense) over fitting a parametric model.

## Reproducibility

The `seed` and `num_request_repeats` fields are populated for LLM runs. The toolkit doesn't aggregate across seeds — multiple seeds appear as multiple `LLMRun` rows sharing `(task, model_id, gpu_model, num_gpus, max_num_seqs)`. To measure variability across seeds, group by everything except `seed` and look at the spread of the metric of interest:

```python
keys = ("task", "model_id", "gpu_model", "num_gpus", "max_num_seqs", "weight_precision")
for k, group in llm.group_by(*keys).items():
    if len(group) > 1:
        es = [r.energy_per_token_joules for r in group]
        print(f"{k}: n={len(es)} energy spread {min(es):.4f}..{max(es):.4f}")
```
