"""Normalized benchmark run records with typed collection API."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml

from mlenergy_data.raw.path_parser import ParsedPath, parse_results_path
from mlenergy_data.records.timelines import load_timeline_table
from mlenergy_data.sources import download_file

logger = logging.getLogger(__name__)
_MP_CONTEXT = mp.get_context("spawn")


@dataclass(frozen=True)
class _HFSource:
    """HF Hub origin metadata, propagated through filters for lazy raw-file downloads."""

    repo_id: str
    revision: str | None
    snapshot_root: Path


@dataclass(frozen=True)
class _RunFile:
    results_path: Path
    parsed: ParsedPath


@dataclass(frozen=True)
class _ScopedRunFile:
    root: Path
    run: _RunFile


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_int(config: dict[str, str], key: str) -> int | None:
    raw = config.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _infer_weight_precision(model_id: str, model_info: dict[str, Any]) -> str:
    if isinstance(model_info.get("weight_precision"), str):
        return str(model_info["weight_precision"])
    if "FP8" in model_id:
        return "fp8"
    if "DeepSeek-R1" in model_id or "DeepSeek-V3" in model_id:
        return "fp8"
    return "bfloat16"


def _load_json(path: Path) -> dict[str, Any]:
    # results.json can be very large; only cache small JSON files.
    if path.stat().st_size <= 1_000_000:
        return _load_json_cached(str(path))
    return json.loads(path.read_text())


@lru_cache(maxsize=32768)
def _load_json_cached(path_raw: str) -> Any:
    return json.loads(Path(path_raw).read_text())


@lru_cache(maxsize=32768)
def _load_yaml_cached(path_raw: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path_raw).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid monolithic.config.yaml (not object): {path_raw}")
    return raw


def _smooth_chunked_itl(itl_list: list[float]) -> list[float]:
    if not itl_list:
        return []
    first = 0
    for i, x in enumerate(itl_list):
        if x > 0:
            first = i
            break
    else:
        return []
    filtered = itl_list[first:]
    out: list[float] = []
    i = 0
    while i < len(filtered):
        if filtered[i] <= 0:
            i += 1
            continue
        latency = filtered[i]
        j = i + 1
        while j < len(filtered) and filtered[j] == 0:
            j += 1
        n = j - i
        if n == 1:
            out.append(float(latency))
        else:
            out.extend([float(latency) / float(n)] * n)
        i = j
    return out


def _client_itl_percentiles_ms(results: dict[str, Any]) -> dict[str, float]:
    values: list[float] = []
    for req in results.get("results", []):
        if not req.get("success", False):
            continue
        itl = [float(x) for x in req.get("itl", [])]
        values.extend(_smooth_chunked_itl(itl))
    if not values:
        raise ValueError("No valid client ITL values found")
    arr = np.asarray(values, dtype=float)
    return {
        "mean_itl_ms": float(np.mean(arr) * 1000.0),
        "p50_itl_ms": float(np.percentile(arr, 50) * 1000.0),
        "p90_itl_ms": float(np.percentile(arr, 90) * 1000.0),
        "p95_itl_ms": float(np.percentile(arr, 95) * 1000.0),
        "p99_itl_ms": float(np.percentile(arr, 99) * 1000.0),
    }


def _load_model_info(
    *,
    llm_config_dir: Path | None,
    diffusion_config_dir: Path | None,
    parsed: ParsedPath,
    gpu_sku: str | None,
) -> dict[str, Any]:
    if parsed.domain == "llm" and llm_config_dir is not None:
        p = llm_config_dir / parsed.workload / f"{parsed.org}/{parsed.model}" / "model_info.json"
        if p.exists():
            data = _load_json_cached(str(p))
            if not isinstance(data, dict):
                raise ValueError(f"Invalid model_info.json (not object): {p}")
            return data
        raise FileNotFoundError(f"Missing required model_info.json: {p}")
    if parsed.domain == "diffusion" and diffusion_config_dir is not None:
        p = (
            diffusion_config_dir
            / parsed.workload
            / f"{parsed.org}/{parsed.model}"
            / "model_info.json"
        )
        if p.exists():
            data = _load_json_cached(str(p))
            if not isinstance(data, dict):
                raise ValueError(f"Invalid model_info.json (not object): {p}")
            return data
        raise FileNotFoundError(f"Missing required model_info.json: {p}")
    del gpu_sku
    raise ValueError(
        f"Missing required config root for domain={parsed.domain!r} "
        f"(llm_config_dir/diffusion_config_dir)"
    )


def _load_parallelization(
    *,
    llm_config_dir: Path | None,
    parsed: ParsedPath,
    gpu_sku: str | None,
    num_gpus: int,
) -> tuple[int, int, int]:
    if llm_config_dir is None:
        raise ValueError("llm_config_dir is required to load parallelization settings")
    if gpu_sku is None:
        raise ValueError("gpu_sku is required for llm parallelization settings")
    cfg = (
        llm_config_dir
        / parsed.workload
        / f"{parsed.org}/{parsed.model}"
        / gpu_sku
        / "monolithic.config.yaml"
    )
    if not cfg.exists():
        raise FileNotFoundError(f"Missing required monolithic.config.yaml: {cfg}")
    raw = _load_yaml_cached(str(cfg))

    ep = 1
    if bool(raw.get("enable-expert-parallel", False)):
        ep = int(raw.get("expert-parallel-size", num_gpus))
    tp_default = 1 if ep > 1 else num_gpus
    tp = int(raw.get("tensor-parallel-size", tp_default))
    dp_raw = raw.get("data-parallel-size", 1)
    dp = int(dp_raw) if int(dp_raw) != 0 else int(num_gpus)
    return (tp, ep, dp)


def _iter_result_files(
    root: Path,
    *,
    domains: tuple[str, ...],
    tasks: set[str] | None,
) -> list[_RunFile]:
    domain_set = set(domains)
    llm_tasks = {"gpqa", "lm-arena-chat", "sourcegraph-fim", "image-chat", "video-chat"}
    diffusion_tasks = {"text-to-image", "text-to-video"}
    task_patterns: dict[str, list[str] | None] = {"llm": None, "diffusion": None}
    if tasks is not None:
        task_patterns["llm"] = sorted(t for t in tasks if t in llm_tasks)
        task_patterns["diffusion"] = sorted(t for t in tasks if t in diffusion_tasks)
    out: list[_RunFile] = []
    for domain in domain_set:
        patterns: list[str] = []
        domain_tasks = task_patterns.get(domain)
        if domain == "llm" and domain_tasks:
            for task in domain_tasks:
                patterns.append(f"llm/*/current/run/*/{task}/results/**/results.json")
        elif domain == "diffusion" and domain_tasks:
            for task in domain_tasks:
                patterns.append(f"diffusion/*/current/run/diffusion/{task}/**/results.json")
        elif tasks is None:
            patterns.append(f"{domain}/*/current/run/**/results.json")
        else:
            continue

        for pattern in patterns:
            for p in root.glob(pattern):
                try:
                    parsed = parse_results_path(p, root)
                except Exception:
                    continue
                if parsed.domain not in domain_set:
                    continue
                if parsed.snapshot != "current":
                    continue
                if tasks is not None and parsed.workload not in tasks:
                    continue
                out.append(_RunFile(results_path=p, parsed=parsed))
    return out


def _build_row(
    run: _RunFile,
    *,
    root: Path,
    llm_cfg: Path | None,
    diff_cfg: Path | None,
) -> LLMRun | DiffusionRun | None:
    try:
        if run.parsed.domain == "llm":
            return _llm_row(
                root=root,
                run=run,
                llm_config_dir=llm_cfg,
                diffusion_config_dir=diff_cfg,
            )
        if run.parsed.domain == "diffusion":
            return _diffusion_row(
                root=root,
                run=run,
                llm_config_dir=llm_cfg,
                diffusion_config_dir=diff_cfg,
            )
        logger.debug("Skipping unknown domain %r: %s", run.parsed.domain, run.results_path)
        return None
    except Exception:
        logger.warning("Failed to build row for %s", run.results_path, exc_info=True)
        return None


def _resolve_workers(n_workers: int | None, n_files: int) -> int:
    if n_files <= 1:
        return 1
    if n_workers is None or n_workers <= 0:
        cpu = os.cpu_count() or 1
        return min(32, max(4, cpu + 4))
    return max(1, int(n_workers))


def _build_row_worker(
    args: tuple[_RunFile, str, str | None, str | None],
) -> LLMRun | DiffusionRun | None:
    run, root_raw, llm_cfg_raw, diff_cfg_raw = args
    return _build_row(
        run,
        root=Path(root_raw),
        llm_cfg=(Path(llm_cfg_raw) if llm_cfg_raw is not None else None),
        diff_cfg=(Path(diff_cfg_raw) if diff_cfg_raw is not None else None),
    )


def _infer_benchmark_root(path: Path) -> Path:
    if path.name == "run" and len(path.parents) >= 4:
        return path.parents[3]
    for candidate in [path, *path.parents]:
        if (candidate / "llm").exists() or (candidate / "diffusion").exists():
            return candidate
    raise ValueError(f"Could not infer benchmark root from path: {path}")


def _llm_stability_status(
    *,
    root: Path,
    run: _RunFile,
    min_steady_duration: float = 20.0,
    batch_utilization_threshold: float = 0.85,
) -> tuple[bool, str]:
    if run.parsed.domain != "llm":
        return (False, "")
    p = run.results_path
    try:
        results = _load_json(p)
    except Exception as exc:
        return (True, f"failed_to_load_results:{exc}")

    steady_duration = _safe_float(results.get("steady_state_duration"))
    if steady_duration is None or steady_duration < min_steady_duration:
        return (True, f"short_steady_state:{steady_duration}")

    energy_per_tok = _safe_float(results.get("steady_state_energy_per_token"))
    if energy_per_tok is None:
        energy_per_tok = _safe_float(results.get("energy_per_token_joules"))
    if energy_per_tok is None or energy_per_tok <= 0:
        return (True, f"invalid_energy_per_token:{energy_per_tok}")

    max_num_seqs = _coerce_int(run.parsed.config, "max_num_seqs")
    if max_num_seqs is not None and max_num_seqs > 0:
        prom_path = p.with_name("prometheus.json")
        try:
            prometheus = _load_json(prom_path)
            avg_batch_size = _safe_float(
                prometheus.get("steady_state_stats", {}).get("vllm:num_requests_running")
            )
        except Exception:
            avg_batch_size = None
        if avg_batch_size is not None:
            utilization = float(avg_batch_size) / float(max_num_seqs)
            if utilization < batch_utilization_threshold:
                return (True, f"low_batch_utilization:{utilization:.6f}")

    return (False, "")


def _llm_group_key(*, root: Path, run: _RunFile) -> tuple[str, str, str, int]:
    parsed = run.parsed
    model_id = f"{parsed.org}/{parsed.model}"
    gpu_model = str(
        _gpu_sku_from_relpath(root, run.results_path, parsed) or parsed.gpu_family
    ).upper()
    num_gpus = _coerce_int(parsed.config, "num_gpus")
    if num_gpus is None:
        raise ValueError(f"Missing required config num_gpus in {run.results_path}")
    return (model_id, parsed.workload, gpu_model, int(num_gpus))


def _stability_map_for_scoped_runs(
    scoped_runs: list[_ScopedRunFile],
    *,
    n_workers: int | None = None,
) -> dict[str, tuple[bool, str]]:
    logger.info("Computing stability for %d runs", len(scoped_runs))
    status: dict[str, tuple[bool, str]] = {}
    unstable_threshold: dict[tuple[str, str, str, int], int] = {}

    workers = _resolve_workers(n_workers, len(scoped_runs))
    if workers <= 1 or len(scoped_runs) <= 1:
        stability_rows = [_stability_worker(sf) for sf in scoped_runs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
            stability_rows = list(pool.map(_stability_worker, scoped_runs))

    for rid, is_unstable, reason, group_key, max_num_seqs in stability_rows:
        status[rid] = (is_unstable, reason)
        if group_key is None or max_num_seqs is None or not is_unstable:
            continue
        prev = unstable_threshold.get(group_key)
        if prev is None or max_num_seqs < prev:
            unstable_threshold[group_key] = max_num_seqs

    for sf in scoped_runs:
        run = sf.run
        if run.parsed.domain != "llm":
            continue
        max_num_seqs = _coerce_int(run.parsed.config, "max_num_seqs")
        if max_num_seqs is None:
            continue
        key = _llm_group_key(root=sf.root, run=run)
        threshold = unstable_threshold.get(key)
        if threshold is None or max_num_seqs < threshold:
            continue
        rid = str(run.results_path.resolve())
        is_unstable, reason = status[rid]
        if not is_unstable:
            status[rid] = (True, f"cascade_from_unstable_batch:{threshold}")
        else:
            status[rid] = (True, reason)

    n_unstable = sum(1 for v in status.values() if v[0])
    n_stable = len(status) - n_unstable
    logger.info(
        "Stability: %d stable, %d unstable out of %d total", n_stable, n_unstable, len(status)
    )
    return status


def _stability_worker(
    sf: _ScopedRunFile,
) -> tuple[str, bool, str, tuple[str, str, str, int] | None, int | None]:
    run = sf.run
    rid = str(run.results_path.resolve())
    if run.parsed.domain != "llm":
        return (rid, False, "", None, None)
    is_unstable, reason = _llm_stability_status(root=sf.root, run=run)
    max_num_seqs = _coerce_int(run.parsed.config, "max_num_seqs")
    group_key: tuple[str, str, str, int] | None = None
    if max_num_seqs is not None:
        group_key = _llm_group_key(root=sf.root, run=run)
    return (rid, is_unstable, reason, group_key, max_num_seqs)


def _gpu_sku_from_relpath(root: Path, path: Path, parsed: ParsedPath) -> str | None:
    parts = path.relative_to(root).parts
    if parsed.domain != "llm":
        return None
    # llm/{gpu}/{snapshot}/run/{suite}/{task}/results/{org}/{model}/{gpu_sku}/{cfg}/results.json
    if len(parts) < 11:
        return None
    return str(parts[9])


def _llm_row(
    *,
    root: Path,
    run: _RunFile,
    llm_config_dir: Path | None,
    diffusion_config_dir: Path | None,
) -> LLMRun | None:
    p = run.results_path
    parsed = run.parsed
    results = _load_json(p)
    prom_path = p.with_name("prometheus.json")
    prometheus = _load_json(prom_path) if prom_path.exists() else {}

    num_gpus = _coerce_int(parsed.config, "num_gpus")
    max_num_seqs = _coerce_int(parsed.config, "max_num_seqs")
    if num_gpus is None:
        raise ValueError(f"Missing required config num_gpus in {p}")
    if max_num_seqs is None:
        raise ValueError(f"Missing required config max_num_seqs in {p}")

    steady_energy = _safe_float(results.get("steady_state_energy"))
    steady_duration = _safe_float(results.get("steady_state_duration"))
    energy_per_tok = _safe_float(results.get("steady_state_energy_per_token"))
    completed = _safe_float(results.get("completed"))
    total_output_tokens = _safe_float(results.get("total_output_tokens"))
    if steady_energy is None:
        raise ValueError(f"Missing required steady_state_energy in {p}")
    if steady_duration is None or steady_duration <= 0:
        raise ValueError(f"Missing/invalid required steady_state_duration in {p}")

    avg_output_len: float | None = None
    if completed is not None and completed > 0 and total_output_tokens is not None:
        avg_output_len = total_output_tokens / completed

    if energy_per_tok is None:
        energy_per_tok = _safe_float(results.get("energy_per_token_joules"))
    if energy_per_tok is None:
        raise ValueError(f"Missing required steady_state_energy_per_token in {p}")
    if energy_per_tok > 0:
        output_throughput = (steady_energy / energy_per_tok) / steady_duration
    else:
        output_throughput = _safe_float(results.get("output_throughput"))
        if output_throughput is None:
            raise ValueError(
                f"Cannot compute output_throughput: energy_per_token <= 0 and "
                f"no fallback output_throughput in {p}"
            )

    energy_per_req: float | None = None
    request_throughput: float | None = None
    if energy_per_tok > 0 and avg_output_len is not None and avg_output_len > 0:
        energy_per_req = energy_per_tok * avg_output_len
        request_throughput = output_throughput / avg_output_len

    try:
        itl = _client_itl_percentiles_ms(results)
    except Exception:
        return None

    model_id = f"{parsed.org}/{parsed.model}"
    gpu_sku = _gpu_sku_from_relpath(root, p, parsed)
    model_info = _load_model_info(
        llm_config_dir=llm_config_dir,
        diffusion_config_dir=diffusion_config_dir,
        parsed=parsed,
        gpu_sku=gpu_sku,
    )
    total_params = _safe_float(model_info.get("total_parameters_billion"))
    active_params = _safe_float(model_info.get("active_parameters_billion"))
    if total_params is None:
        raise ValueError(f"Missing required model_info.total_parameters_billion for {model_id}")
    if active_params is None:
        active_params = total_params
    architecture_raw = model_info.get("architecture")
    if isinstance(architecture_raw, str) and architecture_raw:
        architecture = architecture_raw
    else:
        architecture = "MoE" if total_params != active_params else "Dense"
    weight_precision = _infer_weight_precision(model_id, model_info)
    nickname_raw = model_info.get("nickname")
    nickname = (
        str(nickname_raw) if isinstance(nickname_raw, str) and nickname_raw else parsed.model
    )

    tp, ep, dp = _load_parallelization(
        llm_config_dir=llm_config_dir,
        parsed=parsed,
        gpu_sku=gpu_sku,
        num_gpus=int(num_gpus),
    )

    avg_batch_size = None
    try:
        avg_batch_size = _safe_float(
            prometheus.get("steady_state_stats", {}).get("vllm:num_requests_running")
        )
    except Exception:
        avg_batch_size = None

    return LLMRun(
        domain="llm",
        task=parsed.workload,
        gpu_model=str(gpu_sku or parsed.gpu_family).upper(),
        model_id=model_id,
        nickname=nickname,
        architecture=architecture,
        total_params_billions=total_params,
        activated_params_billions=active_params,
        weight_precision=weight_precision,
        num_gpus=int(num_gpus),
        max_num_seqs=int(max_num_seqs),
        seed=_coerce_int(parsed.config, "seed"),
        num_request_repeats=_coerce_int(parsed.config, "num_request_repeats"),
        tensor_parallel=int(tp),
        expert_parallel=int(ep),
        data_parallel=int(dp),
        steady_state_energy_joules=float(steady_energy),
        steady_state_duration_seconds=float(steady_duration),
        energy_per_token_joules=energy_per_tok,
        energy_per_request_joules=energy_per_req,
        output_throughput_tokens_per_sec=output_throughput,
        request_throughput_req_per_sec=request_throughput,
        avg_power_watts=float(steady_energy / steady_duration),
        total_output_tokens=total_output_tokens,
        completed_requests=completed,
        avg_output_len=avg_output_len,
        mean_itl_ms=itl["mean_itl_ms"],
        median_itl_ms=itl["p50_itl_ms"],
        p50_itl_ms=itl["p50_itl_ms"],
        p90_itl_ms=itl["p90_itl_ms"],
        p95_itl_ms=itl["p95_itl_ms"],
        p99_itl_ms=itl["p99_itl_ms"],
        avg_batch_size=avg_batch_size,
        is_stable=True,
        unstable_reason="",
        results_path=str(p),
        prometheus_path=str(prom_path),
    )


def _diffusion_row(
    *,
    root: Path,
    run: _RunFile,
    llm_config_dir: Path | None,
    diffusion_config_dir: Path | None,
) -> DiffusionRun | None:
    del llm_config_dir
    p = run.results_path
    parsed = run.parsed
    results = _load_json(p)

    config = parsed.config
    batch_size = _coerce_int(config, "batch")
    if batch_size is None:
        batch_size = _safe_float(results.get("batch_size"))
        batch_size = int(batch_size) if batch_size is not None else None
    if batch_size is None or batch_size <= 0:
        raise ValueError(f"Missing/invalid required batch size in {p}")

    uly = _coerce_int(config, "uly")
    ring = _coerce_int(config, "ring")
    if uly is None:
        uly_val = _safe_float(results.get("ulysses_degree"))
        uly = int(uly_val) if uly_val is not None else None
    if ring is None:
        ring_val = _safe_float(results.get("ring_degree"))
        ring = int(ring_val) if ring_val is not None else None
    if uly is None or uly <= 0:
        raise ValueError(f"Missing/invalid required ulysses degree in {p}")
    if ring is None or ring <= 0:
        raise ValueError(f"Missing/invalid required ring degree in {p}")

    num_gpus = int(max(uly, 1) * max(ring, 1))

    iters = results.get("iteration_energy_measurements")
    if not isinstance(iters, list) or len(iters) == 0:
        raise ValueError(f"Missing required iteration_energy_measurements in {p}")

    batch_energies: list[float] = []
    batch_latencies: list[float] = []
    for it in iters:
        ge = it.get("gpu_energy", {})
        if isinstance(ge, dict):
            total = float(sum(float(v) for v in ge.values()))
        else:
            continue
        t = _safe_float(it.get("time"))
        if t is None or t <= 0:
            continue
        batch_energies.append(total)
        batch_latencies.append(float(t))

    if not batch_energies or not batch_latencies:
        raise ValueError(f"No valid iteration energy/time samples in {p}")

    avg_batch_energy = float(np.mean(batch_energies))
    avg_batch_latency = float(np.mean(batch_latencies))
    avg_power = avg_batch_energy / avg_batch_latency

    energy_per_output = avg_batch_energy / float(batch_size)
    throughput = float(batch_size) / avg_batch_latency

    model_id = f"{parsed.org}/{parsed.model}"
    model_info = _load_model_info(
        llm_config_dir=None,
        diffusion_config_dir=diffusion_config_dir,
        parsed=parsed,
        gpu_sku=None,
    )

    total_params = _safe_float(model_info.get("total_parameters_billion"))
    active_params = _safe_float(model_info.get("active_parameters_billion"))
    if total_params is None:
        raise ValueError(f"Missing required model_info.total_parameters_billion for {model_id}")
    if active_params is None:
        active_params = total_params
    nickname_raw = model_info.get("nickname")
    weight_precision_raw = model_info.get("weight_precision")

    num_frames_raw = _safe_float(results.get("num_frames"))
    fps_raw = _safe_float(results.get("fps"))

    height: int = 0
    width: int = 0
    size = str(config.get("size", ""))
    if "x" in size:
        h, w = size.split("x", 1)
        try:
            height = int(h)
            width = int(w)
        except ValueError as exc:
            raise ValueError(
                f"Invalid size string {size!r} in config for {p}: "
                f"expected 'HxW' with integer dimensions"
            ) from exc

    return DiffusionRun(
        domain="diffusion",
        task=parsed.workload,
        gpu_model=parsed.gpu_family.upper(),
        model_id=model_id,
        nickname=(
            str(nickname_raw) if isinstance(nickname_raw, str) and nickname_raw else parsed.model
        ),
        total_params_billions=total_params,
        activated_params_billions=active_params,
        weight_precision=(
            str(weight_precision_raw)
            if isinstance(weight_precision_raw, str) and weight_precision_raw
            else "bfloat16"
        ),
        num_gpus=int(num_gpus),
        batch_size=int(batch_size),
        inference_steps=_coerce_int(config, "steps"),
        height=height,
        width=width,
        num_frames=int(num_frames_raw) if num_frames_raw is not None else None,
        fps=int(fps_raw) if fps_raw is not None else None,
        ulysses_degree=int(max(uly, 1)),
        ring_degree=int(max(ring, 1)),
        use_torch_compile=str(config.get("tc", "")).lower() == "true",
        batch_latency_s=avg_batch_latency,
        avg_power_watts=avg_power,
        energy_per_generation_joules=energy_per_output,
        throughput_generations_per_sec=throughput,
        results_path=str(p),
    )


def _load_runs_from_roots(
    roots: Sequence[str | Path],
    *,
    domains: tuple[str, ...],
    tasks: set[str] | None,
    llm_cfg: Path | None,
    diff_cfg: Path | None,
    stable_only: bool,
    n_workers: int | None,
) -> list[LLMRun | DiffusionRun]:
    """Shared loading logic for from_raw_results constructors."""
    scoped_files: list[_ScopedRunFile] = []
    seen: set[str] = set()

    for raw_root in roots:
        root_path = Path(raw_root)
        files = _iter_result_files(
            root_path,
            domains=domains,
            tasks=tasks,
        )
        if files:
            for rf in files:
                resolved = str(rf.results_path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    scoped_files.append(_ScopedRunFile(root=root_path, run=rf))
        else:
            try:
                inferred_root = _infer_benchmark_root(root_path)
            except ValueError:
                inferred_root = root_path
            for p in root_path.rglob("results.json"):
                resolved = str(p.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    parsed = parse_results_path(p, inferred_root)
                except Exception:
                    continue
                if parsed.domain not in set(domains):
                    continue
                if parsed.snapshot != "current":
                    continue
                if tasks is not None and parsed.workload not in tasks:
                    continue
                scoped_files.append(_ScopedRunFile(root=inferred_root, run=_RunFile(p, parsed)))

    if not scoped_files:
        logger.info("No result files found")
        return []

    logger.info("Found %d result files across %d roots", len(scoped_files), len(roots))
    workers = _resolve_workers(n_workers, len(scoped_files))

    has_llm = any(sf.run.parsed.domain == "llm" for sf in scoped_files)
    if has_llm:
        status_map = _stability_map_for_scoped_runs(scoped_files, n_workers=workers)
        if stable_only:
            scoped_files = [
                sf
                for sf in scoped_files
                if not status_map.get(str(sf.run.results_path.resolve()), (False, ""))[0]
            ]
            logger.info("After leaving only stable runs: %d files", len(scoped_files))
    else:
        status_map = {}

    if not scoped_files:
        return []

    args_list = [
        (
            sf.run,
            str(sf.root),
            str(llm_cfg) if llm_cfg is not None else None,
            str(diff_cfg) if diff_cfg is not None else None,
        )
        for sf in scoped_files
    ]
    logger.info("Processing %d result files with %d workers", len(scoped_files), workers)
    if workers <= 1 or len(scoped_files) <= 1:
        raw_rows = [_build_row_worker(a) for a in args_list]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=_MP_CONTEXT) as pool:
            raw_rows = list(pool.map(_build_row_worker, args_list))

    rows = [r for r in raw_rows if r is not None]
    logger.info("Built %d rows (%d skipped)", len(rows), len(raw_rows) - len(rows))

    if not rows:
        return []

    for i, row in enumerate(rows):
        rid = str(Path(row.results_path).resolve())
        is_unstable, reason = status_map.get(rid, (False, ""))
        if isinstance(row, LLMRun):
            rows[i] = replace(row, is_stable=not is_unstable, unstable_reason=reason)

    if stable_only:
        rows = [r for r in rows if not isinstance(r, LLMRun) or r.is_stable]

    return rows


def _auto_detect_config(
    roots: tuple[str | Path, ...],
    config_subdir: str,
) -> Path | None:
    """Auto-detect embedded config directory from data roots.

    Looks for ``configs/{config_subdir}`` inside each root directory.
    Returns the first match, or ``None`` if not found.
    """
    for raw_root in roots:
        candidate = Path(raw_root) / "configs" / config_subdir
        if candidate.is_dir():
            logger.info("Auto-detected config dir: %s", candidate)
            return candidate
    return None


@dataclass(frozen=True)
class LLMRun:
    """A single LLM benchmark run.

    Attributes:
        domain: Always ``"llm"``.
        task: Benchmark task (e.g. ``"gpqa"``, ``"lm-arena-chat"``).
        model_id: Full HF model identifier (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``).
        nickname: Human-friendly display name from ``model_info.json``.
        architecture: Model architecture (``"Dense"`` or ``"MoE"``).
        total_params_billions: Total parameter count in billions.
        activated_params_billions: Activated parameter count in billions (equals total for dense).
        weight_precision: Weight precision (e.g. ``"bfloat16"``, ``"fp8"``).
        gpu_model: GPU model identifier (e.g. ``"H100"``, ``"B200"``).
        num_gpus: Number of GPUs used.
        max_num_seqs: Maximum concurrent sequences (batch size).
        seed: Random seed used for the benchmark run.
        num_request_repeats: Number of request repetitions.
        tensor_parallel: Tensor parallelism degree.
        expert_parallel: Expert parallelism degree.
        data_parallel: Data parallelism degree.
        steady_state_energy_joules: Total GPU energy during steady state in joules.
        steady_state_duration_seconds: Duration of steady state in seconds.
        energy_per_token_joules: Steady-state energy per output token in joules.
        energy_per_request_joules: Estimated energy per request in joules.
        output_throughput_tokens_per_sec: Steady-state output throughput in tokens/second.
        request_throughput_req_per_sec: Steady-state request throughput in requests/second.
        avg_power_watts: Average GPU power during steady state in watts.
        total_output_tokens: Total output tokens generated (over full benchmark).
        completed_requests: Number of completed requests (over full benchmark).
        avg_output_len: Average output length in tokens.
        mean_itl_ms: Mean inter-token latency in milliseconds.
        median_itl_ms: Median inter-token latency in milliseconds.
        p50_itl_ms: 50th percentile inter-token latency in milliseconds.
        p90_itl_ms: 90th percentile inter-token latency in milliseconds.
        p95_itl_ms: 95th percentile inter-token latency in milliseconds.
        p99_itl_ms: 99th percentile inter-token latency in milliseconds.
        avg_batch_size: Average concurrent sequences during steady state (from Prometheus).
        is_stable: Whether this run passed stability checks.
        unstable_reason: Reason for instability (empty if stable).
        results_path: Path to the raw ``results.json`` file.
        prometheus_path: Path to the ``prometheus.json`` file.
    """

    domain: str
    task: str
    model_id: str
    nickname: str
    architecture: str
    total_params_billions: float
    activated_params_billions: float
    weight_precision: str
    gpu_model: str
    num_gpus: int
    max_num_seqs: int
    seed: int | None
    num_request_repeats: int | None
    tensor_parallel: int
    expert_parallel: int
    data_parallel: int
    steady_state_energy_joules: float
    steady_state_duration_seconds: float
    energy_per_token_joules: float
    energy_per_request_joules: float | None
    output_throughput_tokens_per_sec: float
    request_throughput_req_per_sec: float | None
    avg_power_watts: float
    total_output_tokens: float | None
    completed_requests: float | None
    avg_output_len: float | None
    mean_itl_ms: float
    median_itl_ms: float
    p50_itl_ms: float
    p90_itl_ms: float
    p95_itl_ms: float
    p99_itl_ms: float
    avg_batch_size: float | None
    is_stable: bool
    unstable_reason: str
    results_path: str
    prometheus_path: str


@dataclass(frozen=True)
class DiffusionRun:
    """A single diffusion model benchmark run.

    Attributes:
        domain: Always ``"diffusion"``.
        task: Benchmark task (``"text-to-image"`` or ``"text-to-video"``).
        model_id: Full HF model identifier.
        nickname: Human-friendly display name from ``model_info.json``.
        total_params_billions: Total parameter count in billions.
        activated_params_billions: Activated parameter count in billions.
        weight_precision: Weight precision (e.g. ``"bfloat16"``, ``"fp8"``).
        gpu_model: GPU model identifier (e.g. ``"H100"``).
        num_gpus: Number of GPUs used.
        batch_size: Batch size.
        inference_steps: Number of diffusion inference steps.
        height: Output height in pixels.
        width: Output width in pixels.
        num_frames: Number of video frames (``None`` for images).
        fps: Video frames per second (``None`` for images).
        ulysses_degree: Ulysses sequence parallelism degree.
        ring_degree: Ring attention parallelism degree.
        use_torch_compile: Whether torch.compile was enabled.
        batch_latency_s: Average batch latency in seconds.
        avg_power_watts: Average GPU power in watts.
        energy_per_generation_joules: Energy per generated output (image or video) in joules.
        throughput_generations_per_sec: Throughput in generations per second.
        results_path: Path to the raw ``results.json`` file.
    """

    domain: str
    task: str
    model_id: str
    nickname: str
    total_params_billions: float
    activated_params_billions: float
    weight_precision: str
    gpu_model: str
    num_gpus: int
    batch_size: int
    inference_steps: int | None
    height: int
    width: int
    num_frames: int | None
    fps: int | None
    ulysses_degree: int | None
    ring_degree: int | None
    use_torch_compile: bool | None
    batch_latency_s: float
    avg_power_watts: float
    energy_per_generation_joules: float
    throughput_generations_per_sec: float
    results_path: str

    @property
    def is_text_to_image(self) -> bool:
        """Whether this is a text-to-image run."""
        return self.task == "text-to-image"

    @property
    def is_text_to_video(self) -> bool:
        """Whether this is a text-to-video run."""
        return self.task == "text-to-video"


_LLM_FIELDS = frozenset(f.name for f in dataclasses.fields(LLMRun))
_DIFFUSION_FIELDS = frozenset(f.name for f in dataclasses.fields(DiffusionRun))


class _LLMRunsData:
    """Typed field accessor for LLMRuns, returning `list[T]` per field.

    Accessed via `LLMRuns.data`. Provides IDE autocomplete and type-safe
    access to field arrays without collisions with filter method names
    (e.g. `runs.data.num_gpus` returns `list[int]` whereas `runs.num_gpus`
    is the filter method).

    When adding fields to `LLMRun`, add a matching `@property` stub
    inside the `if TYPE_CHECKING` block below to keep types in sync.
    """

    __slots__ = ("_cache", "_ensure_raw", "_runs")

    def __init__(
        self,
        runs: tuple[LLMRun, ...],
        cache: dict[str, Any],
        ensure_raw: Callable[[], None],
    ) -> None:
        self._runs = runs
        self._cache = cache
        self._ensure_raw = ensure_raw

    @property
    def results_path(self) -> list[str]:
        """Path to the raw results.json file. Triggers download if needed."""
        self._ensure_raw()
        key = "_data_results_path"
        if key not in self._cache:
            self._cache[key] = [r.results_path for r in self._runs]
        return self._cache[key]

    @property
    def prometheus_path(self) -> list[str]:
        """Path to the prometheus.json file. Triggers download if needed."""
        self._ensure_raw()
        key = "_data_prometheus_path"
        if key not in self._cache:
            self._cache[key] = [r.prometheus_path for r in self._runs]
        return self._cache[key]

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> list[Any]:
            if name not in _LLM_FIELDS:
                raise AttributeError(f"LLMRun has no field {name!r}")
            key = f"_data_{name}"
            if key not in self._cache:
                self._cache[key] = [getattr(r, name) for r in self._runs]
            return self._cache[key]

    if TYPE_CHECKING:

        @property
        def domain(self) -> list[str]:
            """Always ``"llm"``."""
            ...

        @property
        def task(self) -> list[str]:
            """Benchmark task identifier."""
            ...

        @property
        def model_id(self) -> list[str]:
            """Full HF model identifier."""
            ...

        @property
        def nickname(self) -> list[str]:
            """Human-friendly display name."""
            ...

        @property
        def architecture(self) -> list[str]:
            """Model architecture (``"Dense"`` or ``"MoE"``)."""
            ...

        @property
        def total_params_billions(self) -> list[float]:
            """Total parameter count in billions."""
            ...

        @property
        def activated_params_billions(self) -> list[float]:
            """Activated parameter count in billions."""
            ...

        @property
        def weight_precision(self) -> list[str]:
            """Weight precision (e.g. ``"bfloat16"``)."""
            ...

        @property
        def gpu_model(self) -> list[str]:
            """GPU model identifier."""
            ...

        @property
        def num_gpus(self) -> list[int]:
            """Number of GPUs used."""
            ...

        @property
        def max_num_seqs(self) -> list[int]:
            """Maximum concurrent sequences (batch size)."""
            ...

        @property
        def seed(self) -> list[int | None]:
            """Random seed."""
            ...

        @property
        def num_request_repeats(self) -> list[int | None]:
            """Number of request repetitions."""
            ...

        @property
        def tensor_parallel(self) -> list[int]:
            """Tensor parallelism degree."""
            ...

        @property
        def expert_parallel(self) -> list[int]:
            """Expert parallelism degree."""
            ...

        @property
        def data_parallel(self) -> list[int]:
            """Data parallelism degree."""
            ...

        @property
        def steady_state_energy_joules(self) -> list[float]:
            """Total GPU energy during steady state in joules."""
            ...

        @property
        def steady_state_duration_seconds(self) -> list[float]:
            """Duration of steady state in seconds."""
            ...

        @property
        def energy_per_token_joules(self) -> list[float]:
            """Steady-state energy per output token in joules."""
            ...

        @property
        def energy_per_request_joules(self) -> list[float | None]:
            """Estimated energy per request in joules."""
            ...

        @property
        def output_throughput_tokens_per_sec(self) -> list[float]:
            """Steady-state output throughput in tokens/second."""
            ...

        @property
        def request_throughput_req_per_sec(self) -> list[float | None]:
            """Steady-state request throughput in requests/second."""
            ...

        @property
        def avg_power_watts(self) -> list[float]:
            """Average GPU power during steady state in watts."""
            ...

        @property
        def total_output_tokens(self) -> list[float | None]:
            """Total output tokens generated (over full benchmark)."""
            ...

        @property
        def completed_requests(self) -> list[float | None]:
            """Number of completed requests (over full benchmark)."""
            ...

        @property
        def avg_output_len(self) -> list[float | None]:
            """Average output length in tokens."""
            ...

        @property
        def mean_itl_ms(self) -> list[float]:
            """Mean inter-token latency in milliseconds."""
            ...

        @property
        def median_itl_ms(self) -> list[float]:
            """Median inter-token latency in milliseconds."""
            ...

        @property
        def p50_itl_ms(self) -> list[float]:
            """50th percentile inter-token latency in milliseconds."""
            ...

        @property
        def p90_itl_ms(self) -> list[float]:
            """90th percentile inter-token latency in milliseconds."""
            ...

        @property
        def p95_itl_ms(self) -> list[float]:
            """95th percentile inter-token latency in milliseconds."""
            ...

        @property
        def p99_itl_ms(self) -> list[float]:
            """99th percentile inter-token latency in milliseconds."""
            ...

        @property
        def avg_batch_size(self) -> list[float | None]:
            """Average concurrent sequences during steady state (from Prometheus)."""
            ...

        @property
        def is_stable(self) -> list[bool]:
            """Whether this run passed stability checks."""
            ...

        @property
        def unstable_reason(self) -> list[str]:
            """Reason for instability."""
            ...


class _DiffusionRunsData:
    """Typed field accessor for DiffusionRuns, returning `list[T]` per field.

    Accessed via `DiffusionRuns.data`. Provides IDE autocomplete and type-safe
    access to field arrays without collisions with filter method names.

    When adding fields to `DiffusionRun`, add a matching `@property` stub
    inside the `if TYPE_CHECKING` block below to keep types in sync.
    """

    __slots__ = ("_cache", "_ensure_raw", "_runs")

    def __init__(
        self,
        runs: tuple[DiffusionRun, ...],
        cache: dict[str, Any],
        ensure_raw: Callable[[], None],
    ) -> None:
        self._runs = runs
        self._cache = cache
        self._ensure_raw = ensure_raw

    @property
    def results_path(self) -> list[str]:
        """Path to the raw results.json file. Triggers download if needed."""
        self._ensure_raw()
        key = "_data_results_path"
        if key not in self._cache:
            self._cache[key] = [r.results_path for r in self._runs]
        return self._cache[key]

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> list[Any]:
            if name not in _DIFFUSION_FIELDS:
                raise AttributeError(f"DiffusionRun has no field {name!r}")
            key = f"_data_{name}"
            if key not in self._cache:
                self._cache[key] = [getattr(r, name) for r in self._runs]
            return self._cache[key]

    if TYPE_CHECKING:

        @property
        def domain(self) -> list[str]:
            """Always ``"diffusion"``."""
            ...

        @property
        def task(self) -> list[str]:
            """Benchmark task identifier."""
            ...

        @property
        def model_id(self) -> list[str]:
            """Full HF model identifier."""
            ...

        @property
        def nickname(self) -> list[str]:
            """Human-friendly display name."""
            ...

        @property
        def total_params_billions(self) -> list[float]:
            """Total parameter count in billions."""
            ...

        @property
        def activated_params_billions(self) -> list[float]:
            """Activated parameter count in billions."""
            ...

        @property
        def weight_precision(self) -> list[str]:
            """Weight precision."""
            ...

        @property
        def gpu_model(self) -> list[str]:
            """GPU model identifier."""
            ...

        @property
        def num_gpus(self) -> list[int]:
            """Number of GPUs used."""
            ...

        @property
        def batch_size(self) -> list[int]:
            """Batch size."""
            ...

        @property
        def inference_steps(self) -> list[int | None]:
            """Number of diffusion inference steps."""
            ...

        @property
        def height(self) -> list[int]:
            """Output height in pixels."""
            ...

        @property
        def width(self) -> list[int]:
            """Output width in pixels."""
            ...

        @property
        def num_frames(self) -> list[int | None]:
            """Number of video frames."""
            ...

        @property
        def fps(self) -> list[int | None]:
            """Video frames per second."""
            ...

        @property
        def ulysses_degree(self) -> list[int | None]:
            """Ulysses sequence parallelism degree."""
            ...

        @property
        def ring_degree(self) -> list[int | None]:
            """Ring attention parallelism degree."""
            ...

        @property
        def use_torch_compile(self) -> list[bool | None]:
            """Whether torch.compile was enabled."""
            ...

        @property
        def batch_latency_s(self) -> list[float]:
            """Average batch latency in seconds."""
            ...

        @property
        def avg_power_watts(self) -> list[float]:
            """Average GPU power in watts."""
            ...

        @property
        def energy_per_generation_joules(self) -> list[float]:
            """Energy per generated output in joules."""
            ...

        @property
        def throughput_generations_per_sec(self) -> list[float]:
            """Throughput in generations per second."""
            ...


class LLMRuns:
    """Immutable collection of LLM benchmark runs with fluent filtering.

    Supports chained filtering, grouping, iteration, and conversion to
    DataFrames. Two data access patterns are available:

    Per-record (row) -- iterate to get individual `LLMRun` objects:

        for r in runs.task("gpqa"):
            print(r.energy_per_token_joules, r.nickname)

        best = min(runs.task("gpqa"), key=lambda r: r.energy_per_token_joules)

    Per-field (column) -- use the `data` property for typed field arrays:

        energies = runs.data.energy_per_token_joules  # list[float]
        gpus = runs.data.num_gpus                     # list[int]

    Example:

        runs = LLMRuns.from_directory("/path/to/compiled/data")
        best = min(runs.stable().task("gpqa"), key=lambda r: r.energy_per_token_joules)
        energies = runs.task("gpqa").data.energy_per_token_joules
    """

    def __init__(
        self,
        runs: Sequence[LLMRun],
        *,
        _hf_source: _HFSource | None = None,
        _stable_only: bool = False,
    ) -> None:
        self._runs = tuple(runs)
        self._cache: dict[str, Any] = {}
        self._hf_source = _hf_source
        self._stable_only = _stable_only

    def _derive(self, runs: Sequence[LLMRun]) -> LLMRuns:
        """Create a derived collection preserving source metadata."""
        return LLMRuns(runs, _hf_source=self._hf_source, _stable_only=self._stable_only)

    def _ensure_raw_files(self) -> None:
        """Ensure raw results files are available locally.

        When the collection was loaded from HF Hub, downloads the raw
        files for all runs in this collection in parallel. Paths on
        each run record are already absolute (resolved at load time),
        so this only triggers the actual download.

        Files already present in the HF Hub local cache (from previous
        downloads, even across processes) are resolved instantly without
        network I/O.
        """
        if "_raw_files" in self._cache or self._hf_source is None:
            return

        root = self._hf_source.snapshot_root
        files: set[str] = set()
        for run in self._runs:
            files.add(str(Path(run.results_path).relative_to(root)))
            if run.prometheus_path:
                files.add(str(Path(run.prometheus_path).relative_to(root)))

        if not files:
            self._cache["_raw_files"] = True
            return

        logger.info(
            "Downloading %d raw files from %s",
            len(files),
            self._hf_source.repo_id,
        )
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
            futures = [
                pool.submit(
                    download_file,
                    self._hf_source.repo_id,
                    f,
                    revision=self._hf_source.revision,
                )
                for f in files
            ]
            for future in futures:
                future.result()

        self._cache["_raw_files"] = True

    def prefetch(self) -> LLMRuns:
        """Eagerly download all raw files for this collection.

        When loaded from HF Hub, downloads all raw `results.json` and
        `prometheus.json` files for every run in the collection. Useful
        when you know you'll need all raw data and want to pay the download
        cost upfront rather than lazily.

        The full unfiltered dataset is ~100 GB. Filter first to limit
        download size:

            runs = LLMRuns.from_hf().task("gpqa").prefetch()
        """
        self._ensure_raw_files()
        return self

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        stable_only: bool = True,
    ) -> LLMRuns:
        """Load runs from a compiled data directory (parquet-first).

        Reads ``runs/llm.parquet`` from the compiled data repo. No raw file
        parsing or stability re-computation is performed.

        Args:
            root: Compiled data directory containing ``runs/llm.parquet``.
            stable_only: If True (default), only return stable runs.
        """
        root_path = Path(root)
        parquet = root_path / "runs" / "llm.parquet"
        return cls.from_parquet(parquet, base_dir=root_path, stable_only=stable_only)

    @classmethod
    def from_hf(
        cls,
        repo_id: str = "ml-energy/benchmark-v3",
        *,
        revision: str | None = None,
        stable_only: bool = True,
    ) -> LLMRuns:
        """Load LLM runs from a Hugging Face dataset repository.

        Downloads only the parquet summary file (~few MB). Methods that
        need raw data (output_lengths(), timelines(), inter_token_latencies())
        will automatically download the required files on first access.

        Respects the ``HF_HOME`` environment variable for cache location.

        Args:
            repo_id: HF dataset repository ID.
            revision: Git revision (branch, tag, or commit hash).
            stable_only: If True (default), only return stable runs.
                See from_raw_results for the definition of stability.
        """
        parquet_path = download_file(repo_id, "runs/llm.parquet", revision=revision)
        snapshot_root = parquet_path.parent.parent
        source = _HFSource(repo_id, revision, snapshot_root)
        instance = cls.from_parquet(parquet_path, base_dir=snapshot_root, stable_only=stable_only)
        instance._hf_source = source
        return instance

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        *,
        base_dir: Path | None = None,
        stable_only: bool = True,
    ) -> LLMRuns:
        """Construct LLMRuns from a pre-built parquet file.

        Args:
            path: Path to the parquet file.
            base_dir: If provided, resolve relative results_path and
                prometheus_path against this directory.
            stable_only: If True (default), only return stable runs.
        """
        df = pd.read_parquet(path)
        if stable_only and "is_stable" in df.columns:
            df = df[df["is_stable"]]
        if base_dir is not None:
            col_set = set(df.columns)
            for col in ("results_path", "prometheus_path"):
                if col in col_set:
                    series: pd.Series[str] = df[col]
                    df[col] = series.apply(lambda p: str(base_dir / p) if pd.notna(p) else p)
        runs_list: list[LLMRun] = []
        records: list[dict[str, Any]] = df.to_dict(orient="records")
        for rec in records:
            kw = {k: rec[k] for k in _LLM_FIELDS if k in rec}
            for k, v in kw.items():
                if pd.isna(v):
                    kw[k] = None
            runs_list.append(LLMRun(**kw))
        runs_list.sort(key=lambda r: (r.task, r.model_id, r.gpu_model, r.num_gpus, r.max_num_seqs))
        logger.info("LLMRuns.from_parquet: returning %d runs", len(runs_list))
        return cls(runs_list, _stable_only=stable_only)

    @classmethod
    def from_raw_results(
        cls,
        *roots: str | Path,
        tasks: set[str] | None = None,
        config_dir: str | Path | None = None,
        stable_only: bool = True,
        n_workers: int | None = None,
    ) -> LLMRuns:
        """Load runs from raw benchmark result directories.

        Parses ``results.json`` files, computes stability, and returns
        the filtered collection.

        A run is considered **unstable** if any of the following hold:

        - The steady-state duration is shorter than 20 seconds.
        - The energy-per-token value is missing or non-positive.
        - The average batch utilization during steady state is below 85%
          of the configured ``max_num_seqs``.
        - **Cascade rule**: if any batch size for a (model, task, GPU,
          num_gpus) group is unstable, all larger batch sizes in the same
          group are also marked unstable.

        Stability is computed jointly across all roots so the cascade
        rule works cross-root.

        Args:
            roots: One or more benchmark root directories (or results sub-dirs).
            tasks: If given, only load these tasks.
            config_dir: Path to LLM config directory (model_info.json, etc.).
            stable_only: If True (default), only return stable runs.
                Pass False to include all runs; each run's ``is_stable``
                and ``unstable_reason`` fields indicate its status.
            n_workers: Number of parallel workers (default: auto).
        """
        if config_dir is not None:
            llm_cfg = Path(config_dir)
        else:
            llm_cfg = _auto_detect_config(roots, "vllm")
        rows = _load_runs_from_roots(
            roots,
            domains=("llm",),
            tasks=tasks,
            llm_cfg=llm_cfg,
            diff_cfg=None,
            stable_only=stable_only,
            n_workers=n_workers,
        )
        llm_runs = [r for r in rows if isinstance(r, LLMRun)]
        llm_runs.sort(key=lambda r: (r.task, r.model_id, r.gpu_model, r.num_gpus, r.max_num_seqs))
        result = cls(llm_runs, _stable_only=stable_only)
        logger.info("LLMRuns.from_raw_results: returning %d runs", len(llm_runs))
        return result

    def task(self, *tasks: str) -> LLMRuns:
        """Filter to runs matching any of the given tasks."""
        return self._filter("task", tasks)

    def model(self, *model_ids: str) -> LLMRuns:
        """Filter to runs matching any of the given model IDs."""
        return self._filter("model_id", model_ids)

    def gpu(self, *gpu_models: str) -> LLMRuns:
        """Filter to runs matching any of the given GPU models."""
        return self._filter("gpu_model", gpu_models)

    def num_gpus(self, *counts: int, min: int | None = None, max: int | None = None) -> LLMRuns:
        """Filter to runs matching given GPU counts or a range.

        Args:
            counts: Exact GPU counts to include.
            min: Minimum GPU count (inclusive).
            max: Maximum GPU count (inclusive).
        """
        if counts and (min is not None or max is not None):
            raise ValueError("Cannot combine exact values with min/max range")
        if counts:
            return self._filter("num_gpus", counts)
        if min is not None or max is not None:
            return self._filter_range("num_gpus", min, max)
        raise TypeError("num_gpus() requires at least one argument")

    def batch(self, *sizes: int, min: int | None = None, max: int | None = None) -> LLMRuns:
        """Filter to runs matching given batch sizes or a range.

        Args:
            sizes: Exact batch sizes to include.
            min: Minimum batch size (inclusive).
            max: Maximum batch size (inclusive).
        """
        if sizes and (min is not None or max is not None):
            raise ValueError("Cannot combine exact values with min/max range")
        if sizes:
            return self._filter("max_num_seqs", sizes)
        if min is not None or max is not None:
            return self._filter_range("max_num_seqs", min, max)
        raise TypeError("batch() requires at least one argument")

    def precision(self, *prec: str) -> LLMRuns:
        """Filter to runs matching any of the given weight precisions."""
        return self._filter("weight_precision", prec)

    def architecture(self, *arch: str) -> LLMRuns:
        """Filter to runs matching any of the given architectures."""
        return self._filter("architecture", arch)

    def nickname(self, *nicknames: str) -> LLMRuns:
        """Filter to runs matching any of the given nicknames."""
        return self._filter("nickname", nicknames)

    def stable(self) -> LLMRuns:
        """Filter to stable runs only."""
        key = "_stable"
        if key not in self._cache:
            self._cache[key] = self._derive([r for r in self._runs if r.is_stable])
        return self._cache[key]

    def unstable(self) -> LLMRuns:
        """Filter to unstable runs only.

        Raises:
            ValueError: If this collection was loaded with ``stable_only=True``,
                since unstable runs were already filtered out at load time.
        """
        if self._stable_only:
            raise ValueError(
                "Cannot filter unstable runs: this collection was loaded with "
                "stable_only=True. Reload with stable_only=False to access unstable runs."
            )
        key = "_unstable"
        if key not in self._cache:
            self._cache[key] = self._derive([r for r in self._runs if not r.is_stable])
        return self._cache[key]

    def where(self, predicate: Callable[[LLMRun], bool]) -> LLMRuns:
        """Filter runs by an arbitrary predicate.

        Args:
            predicate: Function that takes an `LLMRun` and returns True to keep it.
        """
        return self._derive([r for r in self._runs if predicate(r)])

    def _filter(self, field: str, values: tuple[Any, ...]) -> LLMRuns:
        key = f"_filter_{field}_{values}"
        if key not in self._cache:
            value_set = set(values)
            self._cache[key] = self._derive(
                [r for r in self._runs if getattr(r, field) in value_set]
            )
        return self._cache[key]

    def _filter_range(self, field: str, min_val: Any, max_val: Any) -> LLMRuns:
        key = f"_filter_range_{field}_{min_val}_{max_val}"
        if key not in self._cache:
            filtered = list(self._runs)
            if min_val is not None:
                filtered = [r for r in filtered if getattr(r, field) >= min_val]
            if max_val is not None:
                filtered = [r for r in filtered if getattr(r, field) <= max_val]
            self._cache[key] = self._derive(filtered)
        return self._cache[key]

    @property
    def data(self) -> _LLMRunsData:
        """Typed field accessor returning `list[T]` per field.

        Provides column-oriented access to run fields with full type safety:

            runs.data.energy_per_token_joules  # list[float]
            runs.data.num_gpus                 # list[int]
            runs.data.nickname                 # list[str]

        Each property returns a plain `list` with one element per run,
        in the same order as iteration.
        """
        key = "_data_accessor"
        if key not in self._cache:
            self._cache[key] = _LLMRunsData(self._runs, self._cache, self._ensure_raw_files)
        return self._cache[key]

    def group_by(self, *fields: str) -> dict[Any, LLMRuns]:
        """Group runs by one or more fields.

        Args:
            fields: One or more `LLMRun` field names to group by.

        Returns:
            Single field: `{value: LLMRuns, ...}`.
            Multiple fields: `{(v1, v2, ...): LLMRuns, ...}`.
        """
        key = f"_group_by_{fields}"
        if key not in self._cache:
            groups: dict[Any, list[LLMRun]] = defaultdict(list)
            for r in self._runs:
                if len(fields) == 1:
                    k = getattr(r, fields[0])
                else:
                    k = tuple(getattr(r, f) for f in fields)
                groups[k].append(r)
            self._cache[key] = {k: self._derive(v) for k, v in groups.items()}
        return self._cache[key]

    def __iter__(self) -> Iterator[LLMRun]:
        return iter(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def __getitem__(self, index: int) -> LLMRun:
        return self._runs[index]

    def __bool__(self) -> bool:
        return len(self._runs) > 0

    def __add__(self, other: LLMRuns) -> LLMRuns:
        source = self._hf_source if self._hf_source == other._hf_source else None
        stable_only = self._stable_only and other._stable_only
        return LLMRuns(
            list(self._runs) + list(other._runs),
            _hf_source=source,
            _stable_only=stable_only,
        )

    def __repr__(self) -> str:
        return f"LLMRuns({len(self._runs)} runs)"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per run."""
        if not self._runs:
            return pd.DataFrame()
        return pd.DataFrame([dataclasses.asdict(r) for r in self._runs])

    def output_lengths(self, *, include_unsuccessful: bool = False) -> pd.DataFrame:
        """Extract per-request output lengths.

        When loaded from HF Hub, automatically downloads the raw files
        needed for the current (possibly filtered) collection.

        Args:
            include_unsuccessful: If True, include requests that failed
                during benchmarking (`success=False` in `results.json`).
                Defaults to False (only successful requests).

        Returns:
            DataFrame with columns: results_path, task, model_id,
            num_gpus, max_num_seqs, output_len, success

        Raises:
            FileNotFoundError: If raw results files are not available locally
                and the collection was not loaded from HF Hub.
        """
        cols = [
            "results_path",
            "task",
            "model_id",
            "num_gpus",
            "max_num_seqs",
            "output_len",
            "success",
        ]
        if not self._runs:
            return pd.DataFrame(columns=cols)

        self._ensure_raw_files()
        rows: list[dict[str, Any]] = []
        for run in self._runs:
            p = Path(run.results_path)
            if not p.exists():
                raise FileNotFoundError(f"Raw results file not found: {p}")
            data = _load_json(p)
            reqs = data.get("results")
            if not isinstance(reqs, list):
                raise ValueError(f"Missing required 'results' list in {p}")
            for req in reqs:
                success = bool(req.get("success", False))
                if not include_unsuccessful and not success:
                    continue
                output_len = req.get("output_len")
                if output_len is None:
                    if success:
                        raise ValueError(
                            f"Missing required output_len for successful request in {p}"
                        )
                    continue
                rows.append(
                    {
                        "results_path": run.results_path,
                        "task": run.task,
                        "model_id": run.model_id,
                        "num_gpus": run.num_gpus,
                        "max_num_seqs": run.max_num_seqs,
                        "output_len": int(output_len),
                        "success": success,
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    def inter_token_latencies(self) -> pd.DataFrame:
        """Extract per-token inter-token latency samples.

        Reads raw results files and extracts ITL values from each successful
        request. Chunked-prefill artifacts (zero-valued ITL entries) are
        smoothed by spreading the accumulated latency across the covered tokens.

        When loaded from HF Hub, automatically downloads the raw files
        needed for the current (possibly filtered) collection.

        Returns:
            DataFrame with columns: results_path, task, model_id,
            num_gpus, max_num_seqs, itl_s

        Raises:
            FileNotFoundError: If raw results files are not available locally
                and the collection was not loaded from HF Hub.
        """
        cols = ["results_path", "task", "model_id", "num_gpus", "max_num_seqs", "itl_s"]
        if not self._runs:
            return pd.DataFrame(columns=cols)

        self._ensure_raw_files()
        rows: list[dict[str, Any]] = []
        for run in self._runs:
            p = Path(run.results_path)
            if not p.exists():
                raise FileNotFoundError(f"Raw results file not found: {p}")
            data = _load_json(p)
            vals: list[float] = []
            for req in data.get("results", []):
                if not req.get("success", False):
                    continue
                raw_itl = [float(x) for x in req.get("itl", [])]
                vals.extend(_smooth_chunked_itl(raw_itl))
            arr = [x for x in vals if x > 0 and np.isfinite(x)]
            if not arr:
                raise ValueError(f"No valid ITL samples in run: {run.results_path}")
            for v in arr:
                rows.append(
                    {
                        "results_path": run.results_path,
                        "task": run.task,
                        "model_id": run.model_id,
                        "num_gpus": run.num_gpus,
                        "max_num_seqs": run.max_num_seqs,
                        "itl_s": v,
                    }
                )
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    def timelines(self, *, metric: str = "power.device_instant") -> pd.DataFrame:
        """Extract power/temperature timeseries.

        When loaded from HF Hub, automatically downloads the raw files
        needed for the current (possibly filtered) collection.

        Args:
            metric: Which timeline to extract. Supported values:
                `"power.device_instant"`, `"power.device_average"`,
                `"temperature"`.

        Returns:
            DataFrame with columns: results_path, domain, task, model_id,
            num_gpus, max_num_seqs, batch_size, timestamp,
            relative_time_s, value, metric
        """
        self._ensure_raw_files()
        records = [
            {
                "results_path": r.results_path,
                "domain": r.domain,
                "task": r.task,
                "model_id": r.model_id,
                "num_gpus": r.num_gpus,
                "max_num_seqs": r.max_num_seqs,
            }
            for r in self._runs
        ]
        if not records:
            return pd.DataFrame(
                columns=[
                    "results_path",
                    "domain",
                    "task",
                    "model_id",
                    "num_gpus",
                    "max_num_seqs",
                    "batch_size",
                    "timestamp",
                    "relative_time_s",
                    "value",
                    "metric",
                ]
            )
        return load_timeline_table(records, metric=metric)


class DiffusionRuns:
    """Immutable collection of diffusion model benchmark runs with fluent filtering.

    Same collection pattern as `LLMRuns`, with diffusion-specific filters.
    Two data access patterns are available:

    Per-record (row) -- iterate to get individual `DiffusionRun` objects:

        for r in runs.task("text-to-image"):
            print(r.energy_per_generation_joules, r.nickname)

    Per-field (column) -- use the `data` property for typed field arrays:

        powers = runs.data.avg_power_watts  # list[float]
    """

    def __init__(
        self, runs: Sequence[DiffusionRun], *, _hf_source: _HFSource | None = None
    ) -> None:
        self._runs = tuple(runs)
        self._cache: dict[str, Any] = {}
        self._hf_source = _hf_source

    def _derive(self, runs: Sequence[DiffusionRun]) -> DiffusionRuns:
        """Create a derived collection preserving source metadata."""
        return DiffusionRuns(runs, _hf_source=self._hf_source)

    def _ensure_raw_files(self) -> None:
        """Ensure raw results files are available locally.

        When the collection was loaded from HF Hub, downloads the raw
        files for all runs in this collection in parallel. Paths on
        each run record are already absolute (resolved at load time),
        so this only triggers the actual download.

        Files already present in the HF Hub local cache (from previous
        downloads, even across processes) are resolved instantly without
        network I/O.
        """
        if "_raw_files" in self._cache or self._hf_source is None:
            return

        root = self._hf_source.snapshot_root
        files: set[str] = set()
        for run in self._runs:
            files.add(str(Path(run.results_path).relative_to(root)))

        if not files:
            self._cache["_raw_files"] = True
            return

        logger.info(
            "Downloading %d raw files from %s",
            len(files),
            self._hf_source.repo_id,
        )
        with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
            futures = [
                pool.submit(
                    download_file,
                    self._hf_source.repo_id,
                    f,
                    revision=self._hf_source.revision,
                )
                for f in files
            ]
            for future in futures:
                future.result()

        self._cache["_raw_files"] = True

    def prefetch(self) -> DiffusionRuns:
        """Eagerly download all raw files for this collection.

        When loaded from HF Hub, downloads all raw `results.json` files
        for every run in the collection. Useful when you know you'll need
        all raw data and want to pay the download cost upfront rather than
        lazily.

        The full unfiltered dataset is ~100 GB. Filter first to limit
        download size:

            runs = DiffusionRuns.from_hf().task("text-to-image").prefetch()
        """
        self._ensure_raw_files()
        return self

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
    ) -> DiffusionRuns:
        """Load runs from a compiled data directory (parquet-first).

        Reads ``runs/diffusion.parquet`` from the compiled data repo.

        Args:
            root: Compiled data directory containing ``runs/diffusion.parquet``.
        """
        root_path = Path(root)
        parquet = root_path / "runs" / "diffusion.parquet"
        return cls.from_parquet(parquet, base_dir=root_path)

    @classmethod
    def from_hf(
        cls,
        repo_id: str = "ml-energy/benchmark-v3",
        *,
        revision: str | None = None,
    ) -> DiffusionRuns:
        """Load diffusion runs from a Hugging Face dataset repository.

        Downloads only the parquet summary file (~few MB). Methods that
        need raw data (timelines()) will automatically download the
        required files on first access.

        Respects the ``HF_HOME`` environment variable for cache location.

        Args:
            repo_id: HF dataset repository ID.
            revision: Git revision (branch, tag, or commit hash).
        """
        parquet_path = download_file(repo_id, "runs/diffusion.parquet", revision=revision)
        snapshot_root = parquet_path.parent.parent
        source = _HFSource(repo_id, revision, snapshot_root)
        instance = cls.from_parquet(parquet_path, base_dir=snapshot_root)
        instance._hf_source = source
        return instance

    @classmethod
    def from_parquet(
        cls,
        path: Path,
        *,
        base_dir: Path | None = None,
    ) -> DiffusionRuns:
        """Construct DiffusionRuns from a pre-built parquet file.

        Args:
            path: Path to the parquet file.
            base_dir: If provided, resolve relative results_path against
                this directory.
        """
        df = pd.read_parquet(path)
        diff_col_set = set(df.columns)
        if base_dir is not None and "results_path" in diff_col_set:
            rp_series: pd.Series[str] = df["results_path"]
            df["results_path"] = rp_series.apply(lambda p: str(base_dir / p) if pd.notna(p) else p)
        runs_list: list[DiffusionRun] = []
        records: list[dict[str, Any]] = df.to_dict(orient="records")
        for rec in records:
            kw = {k: rec[k] for k in _DIFFUSION_FIELDS if k in rec}
            for k, v in kw.items():
                if pd.isna(v):
                    kw[k] = None
            runs_list.append(DiffusionRun(**kw))
        runs_list.sort(key=lambda r: (r.task, r.model_id, r.gpu_model, r.num_gpus, r.batch_size))
        logger.info("DiffusionRuns.from_parquet: returning %d runs", len(runs_list))
        return cls(runs_list)

    @classmethod
    def from_raw_results(
        cls,
        *roots: str | Path,
        tasks: set[str] | None = None,
        config_dir: str | Path | None = None,
        n_workers: int | None = None,
    ) -> DiffusionRuns:
        """Load runs from raw benchmark result directories.

        Parses ``results.json`` files and returns the collection.

        Args:
            roots: One or more benchmark root directories (or results sub-dirs).
            tasks: If given, only load these tasks.
            config_dir: Path to diffusion config directory.
            n_workers: Number of parallel workers (default: auto).
        """
        if config_dir is not None:
            diff_cfg = Path(config_dir)
        else:
            diff_cfg = _auto_detect_config(roots, "xdit")
        rows = _load_runs_from_roots(
            roots,
            domains=("diffusion",),
            tasks=tasks,
            llm_cfg=None,
            diff_cfg=diff_cfg,
            stable_only=False,
            n_workers=n_workers,
        )
        diff_runs = [r for r in rows if isinstance(r, DiffusionRun)]
        diff_runs.sort(key=lambda r: (r.task, r.model_id, r.gpu_model, r.num_gpus, r.batch_size))
        logger.info("DiffusionRuns.from_raw_results: returning %d runs", len(diff_runs))
        return cls(diff_runs)

    def task(self, *tasks: str) -> DiffusionRuns:
        """Filter to runs matching any of the given tasks."""
        return self._filter("task", tasks)

    def model(self, *model_ids: str) -> DiffusionRuns:
        """Filter to runs matching any of the given model IDs."""
        return self._filter("model_id", model_ids)

    def gpu(self, *gpu_models: str) -> DiffusionRuns:
        """Filter to runs matching any of the given GPU models."""
        return self._filter("gpu_model", gpu_models)

    def num_gpus(
        self, *counts: int, min: int | None = None, max: int | None = None
    ) -> DiffusionRuns:
        """Filter to runs matching given GPU counts or a range.

        Args:
            counts: Exact GPU counts to include.
            min: Minimum GPU count (inclusive).
            max: Maximum GPU count (inclusive).
        """
        if counts and (min is not None or max is not None):
            raise ValueError("Cannot combine exact values with min/max range")
        if counts:
            return self._filter("num_gpus", counts)
        if min is not None or max is not None:
            return self._filter_range("num_gpus", min, max)
        raise TypeError("num_gpus() requires at least one argument")

    def nickname(self, *nicknames: str) -> DiffusionRuns:
        """Filter to runs matching any of the given nicknames."""
        return self._filter("nickname", nicknames)

    def batch(self, *sizes: int, min: int | None = None, max: int | None = None) -> DiffusionRuns:
        """Filter to runs matching given batch sizes or a range.

        Args:
            sizes: Exact batch sizes to include.
            min: Minimum batch size (inclusive).
            max: Maximum batch size (inclusive).
        """
        if sizes and (min is not None or max is not None):
            raise ValueError("Cannot combine exact values with min/max range")
        if sizes:
            return self._filter("batch_size", sizes)
        if min is not None or max is not None:
            return self._filter_range("batch_size", min, max)
        raise TypeError("batch() requires at least one argument")

    def precision(self, *prec: str) -> DiffusionRuns:
        """Filter to runs matching any of the given weight precisions."""
        return self._filter("weight_precision", prec)

    def where(self, predicate: Callable[[DiffusionRun], bool]) -> DiffusionRuns:
        """Filter runs by an arbitrary predicate.

        Args:
            predicate: Function that takes a `DiffusionRun` and returns True to keep it.
        """
        return self._derive([r for r in self._runs if predicate(r)])

    def _filter(self, field: str, values: tuple[Any, ...]) -> DiffusionRuns:
        key = f"_filter_{field}_{values}"
        if key not in self._cache:
            value_set = set(values)
            self._cache[key] = self._derive(
                [r for r in self._runs if getattr(r, field) in value_set]
            )
        return self._cache[key]

    def _filter_range(self, field: str, min_val: Any, max_val: Any) -> DiffusionRuns:
        key = f"_filter_range_{field}_{min_val}_{max_val}"
        if key not in self._cache:
            filtered = list(self._runs)
            if min_val is not None:
                filtered = [r for r in filtered if getattr(r, field) >= min_val]
            if max_val is not None:
                filtered = [r for r in filtered if getattr(r, field) <= max_val]
            self._cache[key] = self._derive(filtered)
        return self._cache[key]

    @property
    def data(self) -> _DiffusionRunsData:
        """Typed field accessor returning `list[T]` per field.

        Provides column-oriented access to run fields with full type safety:

            runs.data.avg_power_watts          # list[float]
            runs.data.num_gpus                 # list[int]
            runs.data.nickname                 # list[str]

        Each property returns a plain `list` with one element per run,
        in the same order as iteration.
        """
        key = "_data_accessor"
        if key not in self._cache:
            self._cache[key] = _DiffusionRunsData(self._runs, self._cache, self._ensure_raw_files)
        return self._cache[key]

    def group_by(self, *fields: str) -> dict[Any, DiffusionRuns]:
        """Group runs by one or more fields.

        Args:
            fields: One or more `DiffusionRun` field names to group by.

        Returns:
            Single field: `{value: DiffusionRuns, ...}`.
            Multiple fields: `{(v1, v2, ...): DiffusionRuns, ...}`.
        """
        key = f"_group_by_{fields}"
        if key not in self._cache:
            groups: dict[Any, list[DiffusionRun]] = defaultdict(list)
            for r in self._runs:
                if len(fields) == 1:
                    k = getattr(r, fields[0])
                else:
                    k = tuple(getattr(r, f) for f in fields)
                groups[k].append(r)
            self._cache[key] = {k: self._derive(v) for k, v in groups.items()}
        return self._cache[key]

    def __iter__(self) -> Iterator[DiffusionRun]:
        return iter(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def __getitem__(self, index: int) -> DiffusionRun:
        return self._runs[index]

    def __bool__(self) -> bool:
        return len(self._runs) > 0

    def __add__(self, other: DiffusionRuns) -> DiffusionRuns:
        source = self._hf_source if self._hf_source == other._hf_source else None
        return DiffusionRuns(list(self._runs) + list(other._runs), _hf_source=source)

    def __repr__(self) -> str:
        return f"DiffusionRuns({len(self._runs)} runs)"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with one row per run."""
        if not self._runs:
            return pd.DataFrame()
        return pd.DataFrame([dataclasses.asdict(r) for r in self._runs])

    def timelines(self, *, metric: str = "power.device_instant") -> pd.DataFrame:
        """Extract power/temperature timeseries.

        When loaded from HF Hub, automatically downloads the raw files
        needed for the current (possibly filtered) collection.

        Args:
            metric: Which timeline to extract. Supported values:
                `"power.device_instant"`, `"power.device_average"`,
                `"temperature"`.

        Returns:
            DataFrame with columns: results_path, domain, task, model_id,
            num_gpus, max_num_seqs, batch_size, timestamp,
            relative_time_s, value, metric
        """
        self._ensure_raw_files()
        records = [
            {
                "results_path": r.results_path,
                "domain": r.domain,
                "task": r.task,
                "model_id": r.model_id,
                "num_gpus": r.num_gpus,
                "batch_size": r.batch_size,
            }
            for r in self._runs
        ]
        if not records:
            return pd.DataFrame(
                columns=[
                    "results_path",
                    "domain",
                    "task",
                    "model_id",
                    "num_gpus",
                    "max_num_seqs",
                    "batch_size",
                    "timestamp",
                    "relative_time_s",
                    "value",
                    "metric",
                ]
            )
        return load_timeline_table(records, metric=metric)
