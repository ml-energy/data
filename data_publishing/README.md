# Releasing Benchmark Data to Hugging Face Hub

This documents how to build and upload the `ml-energy/benchmark-v3` dataset.

## Prerequisites

- Access to the raw benchmark results directory tree
- Access to the LLM and diffusion config directories (from the benchmark repo)
- `hf` CLI authenticated: `hf auth login`

## 1. Build the data package

The build CLI produces a self-contained directory with parquet summaries, embedded configs, and (optionally) raw results files.

```bash
python data_publishing/build_hf_data.py \
  --results-dir /path/to/benchmark/3.0/llm/h100/current/run \
  --results-dir /path/to/benchmark/3.0/llm/b200/current/run \
  --results-dir /path/to/benchmark/3.0/diffusion/h100/current/run \
  --results-dir /path/to/benchmark/3.0/diffusion/b200/current/run \
  --llm-config-dir /path/to/benchmark/configs/vllm \
  --diffusion-config-dir /path/to/benchmark/configs/xdit \
  --out-dir /path/to/output
```

Add `--skip-raw` to build only parquet + configs (useful for quick iteration).

Output structure:

```
/path/to/output/
├── README.md                    # HF Data Studio YAML config + schema docs
├── LICENSE                      # Apache 2.0
├── runs/
│   ├── llm.parquet              # One row per LLM run
│   └── diffusion.parquet        # One row per diffusion run
├── configs/
│   ├── vllm/...                 # Embedded LLM configs
│   └── xdit/...                 # Embedded diffusion configs
├── llm/                         # Raw results (omitted with --skip-raw)
└── diffusion/                   # Raw results (omitted with --skip-raw)
```

## 2. Verify locally

Load from the built package to confirm it works:

```python
from mlenergy_data.records import LLMRuns, DiffusionRuns

llm = LLMRuns.from_directory("/path/to/output", stable_only=True)
diff = DiffusionRuns.from_directory("/path/to/output")
print(f"LLM: {len(llm)} runs, Diffusion: {len(diff)} runs")
```

Also verify parquet round-trip:

```python
llm = LLMRuns.from_parquet("/path/to/output/runs/llm.parquet")
diff = DiffusionRuns.from_parquet("/path/to/output/runs/diffusion.parquet")
```

## 3. Upload to Hugging Face Hub

### Full upload (parquet + configs + raw results)

Use `upload-large-folder` for resumable uploads of large directories:

```bash
hf upload-large-folder ml-energy/benchmark-v3 /path/to/output --repo-type dataset
```

### Lightweight upload (parquet + configs only)

For testing Data Studio without uploading raw results:

```bash
hf upload ml-energy/benchmark-v3 /path/to/output/runs runs --repo-type dataset
hf upload ml-energy/benchmark-v3 /path/to/output/configs configs --repo-type dataset
hf upload ml-energy/benchmark-v3 /path/to/output/README.md README.md --repo-type dataset
```

## 4. Verify Data Studio

After upload, check https://huggingface.co/datasets/ml-energy/benchmark-v3:

1. Two subsets should appear in the viewer dropdown: **llm** (default), **diffusion**
2. Column types render correctly (floats, strings, booleans, nullable fields)
3. Filtering and SQL Console work

## 5. Verify consumer scripts

After upload, verify that `from_hf()` works end-to-end:

```python
from mlenergy_data.records import LLMRuns, DiffusionRuns

# Load from HF Hub (parquet only, fast)
llm = LLMRuns.from_hf()
diff = DiffusionRuns.from_hf()

# Bulk data methods auto-download raw files as needed
out = llm.task("gpqa").output_lengths()
```
