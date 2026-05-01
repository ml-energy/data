# Data Catalog

Concrete enumeration of what's in `ml-energy/benchmark-v3` as of the current release. Use this as a lookup so you don't have to re-download the parquet to check which combinations exist. To verify any of these are still current, run `LLMRuns.from_hf()` / `DiffusionRuns.from_hf()` and group as needed.

## Counts

- LLM runs (stable): 838 (1138 total; ~300 marked unstable)
- Diffusion runs: 1020

## Tasks

| Task | Domain | Description |
|---|---|---|
| `gpqa` | LLM | GPQA Diamond — long-chain reasoning |
| `lm-arena-chat` | LLM | LM Arena chat-style requests |
| `sourcegraph-fim` | LLM | Code fill-in-the-middle (Sourcegraph) |
| `image-chat` | LLM (multimodal) | Vision-language chat with image inputs |
| `video-chat` | LLM (multimodal) | Vision-language chat with video inputs |
| `text-to-image` | Diffusion | Text-to-image generation |
| `text-to-video` | Diffusion | Text-to-video generation |

Display names used in the leaderboard JSON (only for human output, not for filtering):
GPQA → "GPQA Diamond", lm-arena-chat → "LLM Chat (LM Arena)", sourcegraph-fim → "Fill-in-the-Middle (Sourcegraph)", image-chat → "Image Chat", video-chat → "Video Chat", text-to-image → "Text to Image", text-to-video → "Text to Video".

## GPUs and parallelism

- `gpu_model`: `"H100"`, `"B200"`.
- `num_gpus`: 1, 2, 4, 8.
- LLM parallelism fields: `tensor_parallel`, `expert_parallel`, `data_parallel` (DP for attention + EP for MLP experts is the typical MoE setup).
- Diffusion parallelism: `ulysses_degree`, `ring_degree` (sequence parallelism).

## LLM batch sizes (`max_num_seqs`)

Observed values (powers of 2 plus a few midpoints): 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096. Not every (model, GPU, num_gpus) cell covers the full range.

## Diffusion batch sizes

1, 2, 4, 8, 16, 32.

## Precisions

- LLM `weight_precision`: `"bfloat16"`, `"fp8"`, `"mxfp4"` (mxfp4 is GPT OSS only).
- Diffusion `weight_precision`: `"bfloat16"` only in this release.

## Architectures (LLM)

Three literal strings — match exactly:

| Value | Examples |
|---|---|
| `"Dense Transformer"` | Llama 3.1 8B/70B/405B Instruct, Qwen 3 8B/14B/32B, Gemma 3 12B/27B, Qwen 3 VL 8B/32B Instruct |
| `"MoE"` | Qwen 3 30B/235B (and Coder 30B/480B), GPT OSS 20B/120B, DeepSeek R1/V3.1, Llama 4 Maverick/Scout, Qwen 3 Omni 30B A3B, Qwen 3 VL 30B/235B Instruct |
| `"Mamba-Transformer Hybrid"` | NVIDIA Nemotron Nano 9B V2, 12B V2, 12B V2 VL |

## LLM models (per-model coverage)

Format: `nickname` — `model_id` — `task list` — `(GPU, num_gpus)` cells. Total params (B) and active params (B) listed for MoE/Hybrid where they differ from total.

### Dense Transformer

- **Gemma 3 12B** — google/gemma-3-12b-it — image-chat, lm-arena-chat — (B200,1), (H100,1)
- **Gemma 3 27B** — google/gemma-3-27b-it — image-chat, lm-arena-chat — (B200,1), (H100,1), (H100,2)
- **Llama 3.1 8B Instruct** — meta-llama/Llama-3.1-8B-Instruct — lm-arena-chat — (B200,1), (H100,1)
- **Llama 3.1 70B Instruct** — meta-llama/Llama-3.1-70B-Instruct — lm-arena-chat — (B200,2), (H100,4)
- **Llama 3.3 70B Instruct** — meta-llama/Llama-3.3-70B-Instruct — lm-arena-chat — (B200,2), (H100,4)
- **Llama 3.1 405B Instruct** (BF16, 405B) — meta-llama/Llama-3.1-405B-Instruct — lm-arena-chat — (B200,8)
- **Llama 3.1 405B Instruct FP8** — meta-llama/Llama-3.1-405B-Instruct-FP8 — lm-arena-chat — (B200,4), (H100,8)
- **Qwen 3 8B** — Qwen/Qwen3-8B — gpqa, lm-arena-chat — (B200,1), (H100,1)
- **Qwen 3 14B** — Qwen/Qwen3-14B — gpqa, lm-arena-chat — (B200,1), (H100,1)
- **Qwen 3 32B** — Qwen/Qwen3-32B — gpqa, lm-arena-chat — (B200,1), (H100,1), (H100,2)
- **Qwen 3 VL 8B Instruct** — Qwen/Qwen3-VL-8B-Instruct — image-chat, video-chat — (B200,1), (H100,1)
- **Qwen 3 VL 32B Instruct** — Qwen/Qwen3-VL-32B-Instruct — image-chat, video-chat — (B200,1), (H100,2)

### MoE (total / active)

- **Qwen 3 30B A3B Instruct** — Qwen/Qwen3-30B-A3B-Instruct-2507 — lm-arena-chat — (30B/3B) — (B200,1), (H100,1), (H100,2)
- **Qwen 3 30B A3B Thinking** — Qwen/Qwen3-30B-A3B-Thinking-2507 — gpqa — (30B/3B) — (B200,1), (H100,1), (H100,2)
- **Qwen 3 Coder 30B A3B** — Qwen/Qwen3-Coder-30B-A3B-Instruct — sourcegraph-fim — (30B/3B) — (B200,1), (H100,1), (H100,2)
- **Qwen 3 Omni 30B A3B Instruct** — Qwen/Qwen3-Omni-30B-A3B-Instruct — image-chat, video-chat — (30B/3B) — (B200,1), (H100,1), (H100,2)
- **Qwen 3 VL 30B A3B Instruct** — Qwen/Qwen3-VL-30B-A3B-Instruct — image-chat, video-chat — (30B/3B) — (B200,1), (H100,1), (H100,2)
- **Qwen 3 235B A22B Instruct** — Qwen/Qwen3-235B-A22B-Instruct-2507 — lm-arena-chat — (235B/22B) — (B200,4), (B200,8), (H100,8)
- **Qwen 3 235B A22B Instruct FP8** — Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 — lm-arena-chat — (235B/22B) — (B200,2), (H100,4), (H100,8)
- **Qwen 3 235B A22B Thinking** — Qwen/Qwen3-235B-A22B-Thinking-2507 — gpqa — (235B/22B) — (B200,4), (B200,8), (H100,8)
- **Qwen 3 235B A22B Thinking FP8** — Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 — gpqa — (235B/22B) — (B200,2), (B200,4), (H100,4), (H100,8)
- **Qwen 3 VL 235B A22B Instruct** — Qwen/Qwen3-VL-235B-A22B-Instruct — image-chat, video-chat — (235B/22B) — (B200,4), (B200,8), (H100,8)
- **Qwen 3 VL 235B A22B Instruct FP8** — Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 — image-chat, video-chat — (235B/22B) — (B200,2), (B200,4), (H100,4), (H100,8)
- **Qwen 3 Coder 480B A35B** — Qwen/Qwen3-Coder-480B-A35B-Instruct — sourcegraph-fim — (480B/35B) — (B200,8)
- **Qwen 3 Coder 480B A35B FP8** — Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 — sourcegraph-fim — (480B/35B) — (B200,4), (B200,8)
- **GPT OSS 20B** — openai/gpt-oss-20b — gpqa — (21B/4B), MXFP4 capable — (B200,1), (H100,1)
- **GPT OSS 120B** — openai/gpt-oss-120b — gpqa — (117B/5B), MXFP4 capable — (B200,1), (B200,2), (H100,1), (H100,2)
- **DeepSeek R1** — deepseek-ai/DeepSeek-R1-0528 — gpqa — (671B/37B), **natively FP8** (no `" FP8"` suffix) — (B200,8)
- **DeepSeek V3.1** — deepseek-ai/DeepSeek-V3.1 — gpqa, lm-arena-chat — (671B/37B), **natively FP8** — (B200,8)
- **Llama 4 Maverick 17B 128E Instruct FP8** — meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 — image-chat, lm-arena-chat — (400B/17B) — (H100,8)
- **Llama 4 Scout 17B 16E Instruct** — meta-llama/Llama-4-Scout-17B-16E-Instruct — image-chat, lm-arena-chat — (109B/17B) — (H100,4), (H100,8)

### Mamba-Transformer Hybrid

- **NVIDIA Nemotron Nano 9B V2** — nvidia/NVIDIA-Nemotron-Nano-9B-v2 — gpqa, lm-arena-chat — (B200,1), (H100,1)
- **NVIDIA Nemotron Nano 12B V2** — nvidia/NVIDIA-Nemotron-Nano-12B-v2 — gpqa, lm-arena-chat — (B200,1), (H100,1)
- **NVIDIA Nemotron Nano 12B V2 VL** — nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL — image-chat — (H100,1)

## Diffusion models

| Nickname | model_id | Task | Output size | inference_steps | Notes |
|---|---|---|---|---|---|
| FLUX.1 Dev | black-forest-labs/FLUX.1-dev | text-to-image | 1024×1024 | 50 | |
| Hunyuan-DiT 1.2 | Tencent-Hunyuan/HunyuanDiT-v1.2 | text-to-image | 1024×1024 | 50 | |
| PixArt-Σ | PixArt-alpha/PixArt-Sigma-XL-2-1024-MS | text-to-image | 1024×1024 | 20 | |
| SANA 1.5 1.6B | Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers | text-to-image | 1024×1024 | 20 | |
| SANA 1.5 4.8B | Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers | text-to-image | 1024×1024 | 20 | |
| Stable Diffusion 3.5 Medium | stabilityai/stable-diffusion-3.5-medium | text-to-image | 1024×1024 | 28 | |
| Stable Diffusion 3.5 Large | stabilityai/stable-diffusion-3.5-large | text-to-image | 1024×1024 | 28 | |
| CogVideoX 2B | THUDM/CogVideoX-2b | text-to-video | 480×720, 49 frames @ 8 fps | 50 | |
| CogVideoX 1.5 5B | THUDM/CogVideoX1.5-5B | text-to-video | 768×1360, 81 frames @ 8 fps | 50 | |
| Wan 2.1 1.3B | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | text-to-video | 480×832, 81 frames @ 15 fps | 50 | |
| Wan 2.1 14B | Wan-AI/Wan2.1-T2V-14B-Diffusers | text-to-video | 480×832, 81 frames @ 15 fps | 50 | |
| HunyuanVideo | tencent/HunyuanVideo | text-to-video | 720×1280, 129 frames @ 15 fps | 50 | |

All diffusion runs in this release are `bfloat16`, `use_torch_compile=False`. Diffusion has run-count coverage on (B200, 1/2/4/8) and (H100, 1/2/4/8) for most models — the multi-GPU variants test sequence-parallel (Ulysses + Ring).

## Stability (LLM only)

A run is marked unstable if any of:

- `steady_state_duration < 20s`
- `energy_per_token_joules` missing or non-positive
- `avg_batch_size / max_num_seqs < 0.85` ("low_batch_utilization")
- A smaller batch in the same (model, task, GPU, num_gpus) group was unstable ("cascade_from_unstable_batch:N")

`stable_only=True` (the default for `LLMRuns.from_hf()`) excludes them. Use `LLMRuns.from_hf(stable_only=False)` then `.unstable()` to inspect.
