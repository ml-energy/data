from __future__ import annotations

import json
from pathlib import Path

from mlenergy_data.records.runs import LLMRuns
from mlenergy_data.records.timelines import (
    extract_power_device_instant,
    extract_temperature_timeseries,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_benchmark_fixture(
    root: Path,
    cfg_root: Path,
    *,
    gpu_sku: str = "H100",
    model_id: str = "org/model",
    task: str = "gpqa",
    num_gpus: int = 1,
    max_num_seqs: int = 16,
    seed: int = 1,
    avg_batch: float = 16.0,
) -> None:
    """Create a minimal benchmark directory with one run."""
    run_dir = (
        root
        / "llm"
        / gpu_sku.lower()
        / "current"
        / "run"
        / "llm"
        / task
        / "results"
        / model_id.replace("/", "/")
        / gpu_sku
        / (
            f"num_gpus+{num_gpus}+max_num_seqs+{max_num_seqs}"
            f"+max_num_batched_tokens+2048+num_request_repeats+1+seed+{seed}"
        )
    )
    _write_json(
        run_dir / "results.json",
        {
            "steady_state_energy_per_token": 0.2,
            "steady_state_energy": 200.0,
            "steady_state_duration": 30.0,
            "output_throughput": 20.0,
            "request_throughput": 2.0,
            "total_output_tokens": 1000,
            "completed": 100,
            "results": [
                {"success": True, "itl": [0.01, 0.0, 0.0, 0.02], "output_len": 4},
                {"success": True, "itl": [0.01, 0.01], "output_len": 2},
            ],
        },
    )
    _write_json(
        run_dir / "prometheus.json",
        {"steady_state_stats": {"vllm:num_requests_running": avg_batch}},
    )

    org, model = model_id.split("/", 1)
    model_info_dir = cfg_root / task / org / model
    _write_json(
        model_info_dir / "model_info.json",
        {
            "nickname": "Model A",
            "total_parameters_billion": 10,
            "active_parameters_billion": 10,
            "architecture": "Dense",
            "weight_precision": "bfloat16",
        },
    )
    (model_info_dir / gpu_sku).mkdir(parents=True, exist_ok=True)
    (model_info_dir / gpu_sku / "monolithic.config.yaml").write_text(
        "tensor-parallel-size: 1\nenable-expert-parallel: false\ndata-parallel-size: 1\n"
    )


def test_extract_power_timeline_with_ffill() -> None:
    results = {
        "timeline": {
            "steady_state_start_time": 10.0,
            "steady_state_end_time": 12.0,
            "power": {
                "device_instant": {
                    "0": [[10.0, 100.0], [11.0, 110.0], [12.0, 120.0]],
                    "1": [[10.0, 200.0], [12.0, 220.0]],
                },
                "device_average": {
                    "0": [[10.0, 90.0], [11.0, 95.0], [12.0, 100.0]],
                    "1": [[10.0, 180.0], [11.0, 190.0], [12.0, 200.0]],
                },
            },
            "temperature": {
                "device_instant": {
                    "0": [[10.0, 60.0], [11.0, 61.0], [12.0, 62.0]],
                    "1": [[10.0, 65.0], [11.0, 66.0], [12.0, 67.0]],
                }
            },
        }
    }

    df = extract_power_device_instant(results)
    assert list(df["relative_time_s"]) == [0.0, 1.0, 2.0]
    # GPU1 missing at t=11 should be ffilled from 200.
    assert float(df.loc[1, "power_device_instant_device_1"]) == 200.0
    assert float(df.loc[1, "power_device_instant_total"]) == 310.0

    tdf = extract_temperature_timeseries(results)
    assert "temperature_device_instant_total" in tdf.columns


def test_llm_runsfrom_raw_results(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    assert len(runs) == 1
    r = runs[0]
    assert r.task == "gpqa"
    assert r.nickname == "Model A"
    assert r.max_num_seqs == 16
    assert r.is_stable is True


def test_llm_runs_filtering(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    assert len(runs) == 2

    batch16 = runs.batch(16)
    assert len(batch16) == 1
    assert batch16[0].max_num_seqs == 16

    batch32 = runs.batch(32)
    assert len(batch32) == 1
    assert batch32[0].max_num_seqs == 32

    both = runs.batch(16, 32)
    assert len(both) == 2

    gpu_filter = runs.gpu("H100")
    assert len(gpu_filter) == 2

    task_filter = runs.task("gpqa")
    assert len(task_filter) == 2

    model_filter = runs.model("org/model")
    assert len(model_filter) == 2


def test_llm_runs_chaining(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    chained = runs.task("gpqa").gpu("H100").batch(16)
    assert len(chained) == 1
    assert chained[0].max_num_seqs == 16


def test_llm_runs_field_access(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    batches = runs.data.max_num_seqs
    assert isinstance(batches, list)
    assert set(batches) == {16, 32}

    powers = runs.data.avg_power_watts
    assert isinstance(powers, list)
    assert len(powers) == 2


def test_llm_runs_group_by(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    groups = runs.group_by("max_num_seqs")
    assert len(groups) == 2
    assert 16 in groups
    assert 32 in groups
    assert len(groups[16]) == 1
    assert len(groups[32]) == 1


def test_llm_runs_iteration_and_bool(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    assert bool(runs)
    assert len(list(runs)) == 1

    empty = LLMRuns([])
    assert not bool(empty)
    assert len(empty) == 0


def test_llm_runs_concatenation(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    doubled = runs + runs
    assert len(doubled) == 2


def test_llm_runs_to_dataframe(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    df = runs.to_dataframe()
    assert len(df) == 1
    assert "task" in df.columns
    assert "energy_per_token_joules" in df.columns
    assert df.iloc[0]["task"] == "gpqa"


def test_llm_runs_output_lengths(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    out = runs.output_lengths()
    assert len(out) == 2
    assert set(out["output_len"].tolist()) == {4, 2}


def test_llm_runs_stability_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1, avg_batch=16.0)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2, avg_batch=8.0)

    runs_stable = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=True)
    assert len(runs_stable) == 1
    assert runs_stable[0].is_stable is True

    runs_all = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    assert len(runs_all) == 2

    stable = runs_all.stable()
    unstable = runs_all.unstable()
    assert len(stable) + len(unstable) == len(runs_all)


def test_llm_runs_where_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    big_batch = runs.where(lambda r: r.max_num_seqs > 20)
    assert len(big_batch) == 1
    assert big_batch[0].max_num_seqs == 32


def test_llm_runs_empty_field_access() -> None:
    empty = LLMRuns([])
    arr = empty.data.energy_per_token_joules
    assert isinstance(arr, list)
    assert len(arr) == 0


def test_llm_runs_invalid_field_access() -> None:
    empty = LLMRuns([])
    try:
        _ = empty.data.nonexistent_field  # type: ignore[attr-defined]
        raise AssertionError("Should raise AttributeError")
    except AttributeError:
        pass


def test_llm_runs_repr() -> None:
    empty = LLMRuns([])
    assert "0 runs" in repr(empty)


def test_llm_runs_multi_field_group_by(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    groups = runs.group_by("task", "gpu_model")
    assert len(groups) == 1
    key = ("gpqa", "H100")
    assert key in groups
    assert len(groups[key]) == 2


def test_llm_runs_label_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    label = runs[0].model_label
    filtered = runs.label(label)
    assert len(filtered) == 1

    empty = runs.label("nonexistent-label")
    assert len(empty) == 0


def test_llm_runs_num_gpus_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    one_gpu = runs.num_gpus(1)
    assert len(one_gpu) == 1

    two_gpu = runs.num_gpus(2)
    assert len(two_gpu) == 0


def test_llm_runs_precision_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    bf16 = runs.precision("bfloat16")
    assert len(bf16) == 1

    fp8 = runs.precision("fp8")
    assert len(fp8) == 0


def test_llm_runs_architecture_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    dense = runs.architecture("Dense")
    assert len(dense) == 1

    moe = runs.architecture("MoE")
    assert len(moe) == 0


def test_llm_runs_field_access_caching(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    arr1 = runs.data.max_num_seqs
    arr2 = runs.data.max_num_seqs
    assert arr1 is arr2


def test_llm_runs_timelines(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    # Fixture doesn't include timeline data, so this should raise
    try:
        runs.timelines()
        raise AssertionError("Should raise due to missing timeline data")
    except (ValueError, KeyError):
        pass


def _make_diffusion_fixture(
    root: Path,
    cfg_root: Path,
    *,
    gpu: str = "h100",
    model_id: str = "org/diffmodel",
    task: str = "text-to-image",
    batch: int = 1,
    uly: int = 1,
    ring: int = 1,
    size: str = "512x512",
) -> None:
    """Create a minimal diffusion benchmark directory with one run."""
    # diffusion/{gpu}/{snapshot}/run/diffusion/{task}/{org}/{model}/{config}/results.json
    # Config uses dash-separated key-value pairs joined by +
    run_dir = (
        root
        / "diffusion"
        / gpu
        / "current"
        / "run"
        / "diffusion"
        / task
        / model_id.replace("/", "/")
        / f"batch-{batch}+uly-{uly}+ring-{ring}+size-{size}+steps-20"
    )
    _write_json(
        run_dir / "results.json",
        {
            "batch_size": batch,
            "ulysses_degree": uly,
            "ring_degree": ring,
            "num_frames": 16 if task == "text-to-video" else None,
            "fps": 8 if task == "text-to-video" else None,
            "iteration_energy_measurements": [
                {"gpu_energy": {"0": 50.0}, "time": 2.0},
                {"gpu_energy": {"0": 55.0}, "time": 2.1},
            ],
        },
    )

    org, model = model_id.split("/", 1)
    model_info_dir = cfg_root / task / org / model
    _write_json(
        model_info_dir / "model_info.json",
        {
            "nickname": "DiffModel A",
            "total_parameters_billion": 3.0,
            "active_parameters_billion": 3.0,
            "weight_precision": "bfloat16",
        },
    )


def test_diffusion_runsfrom_raw_results(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    assert len(runs) == 1
    r = runs[0]
    assert r.task == "text-to-image"
    assert r.nickname == "DiffModel A"
    assert r.batch_size == 1
    assert r.height == 512
    assert r.width == 512
    assert r.is_text_to_image is True
    assert r.is_text_to_video is False


def test_diffusion_runs_filtering(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)
    _make_diffusion_fixture(root, cfg_root, batch=2, size="1024x1024")

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    assert len(runs) == 2

    batch1 = runs.batch(1)
    assert len(batch1) == 1
    assert batch1[0].batch_size == 1

    gpu_filter = runs.gpu("H100")
    assert len(gpu_filter) == 2

    model_filter = runs.model("org/diffmodel")
    assert len(model_filter) == 2


def test_diffusion_runs_field_access(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)
    _make_diffusion_fixture(root, cfg_root, batch=2, size="1024x1024")

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    powers = runs.data.avg_power_watts
    assert isinstance(powers, list)
    assert len(powers) == 2


def test_diffusion_runs_group_by(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)
    _make_diffusion_fixture(root, cfg_root, batch=2, size="1024x1024")

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    groups = runs.group_by("batch_size")
    assert len(groups) == 2
    assert 1 in groups
    assert 2 in groups


def test_diffusion_runs_concatenation_and_repr(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    doubled = runs + runs
    assert len(doubled) == 2
    assert "1 runs" in repr(runs)


def test_diffusion_runs_to_dataframe(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    df = runs.to_dataframe()
    assert len(df) == 1
    assert "task" in df.columns
    assert df.iloc[0]["task"] == "text-to-image"


def test_llm_runs_nickname_filter(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    filtered = runs.nickname("Model A")
    assert len(filtered) == 1

    empty = runs.nickname("Nonexistent")
    assert len(empty) == 0


def test_diffusion_runs_nickname_filter(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    filtered = runs.nickname("DiffModel A")
    assert len(filtered) == 1

    empty = runs.nickname("Nonexistent")
    assert len(empty) == 0


def test_llm_runs_batch_range(tmp_path: Path) -> None:
    import pytest

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=8, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=2)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=3)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    assert len(runs) == 3

    at_least_16 = runs.batch(min=16)
    assert len(at_least_16) == 2
    assert all(r.max_num_seqs >= 16 for r in at_least_16)

    at_most_16 = runs.batch(max=16)
    assert len(at_most_16) == 2
    assert all(r.max_num_seqs <= 16 for r in at_most_16)

    range_8_16 = runs.batch(min=8, max=16)
    assert len(range_8_16) == 2

    with pytest.raises(ValueError):
        runs.batch(16, min=8)
    with pytest.raises(TypeError):
        runs.batch()


def test_llm_runs_num_gpus_range(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1, num_gpus=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=2, num_gpus=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
    at_least_2 = runs.num_gpus(min=2)
    assert len(at_least_2) == 1
    assert at_least_2[0].num_gpus == 2


def test_diffusion_runs_batch_range(tmp_path: Path) -> None:
    import pytest

    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)
    _make_diffusion_fixture(root, cfg_root, batch=4, size="1024x1024")

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    big = runs.batch(min=2)
    assert len(big) == 1
    assert big[0].batch_size == 4

    with pytest.raises(ValueError):
        runs.batch(1, min=1)
    with pytest.raises(TypeError):
        runs.batch()


def test_llm_runs_data_accessor(tmp_path: Path) -> None:
    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
    _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

    runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)

    energies = runs.data.energy_per_token_joules
    assert isinstance(energies, list)
    assert len(energies) == 2

    gpus = runs.data.num_gpus
    assert isinstance(gpus, list)
    assert all(isinstance(g, int) for g in gpus)

    # Accessor is cached (same object on repeated access)
    assert runs.data is runs.data

    # Nonexistent field raises AttributeError
    import pytest

    with pytest.raises(AttributeError):
        _ = runs.data.nonexistent_field  # type: ignore[attr-defined]


def test_diffusion_runs_data_accessor(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)

    powers = runs.data.avg_power_watts
    assert isinstance(powers, list)
    assert len(powers) == 1

    tasks = runs.data.task
    assert isinstance(tasks, list)
    assert tasks[0] == "text-to-image"

    assert runs.data is runs.data


def test_diffusion_runs_num_gpus_filter(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1, uly=1, ring=1)
    _make_diffusion_fixture(root, cfg_root, batch=1, uly=2, ring=1, size="1024x1024")

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    one_gpu = runs.num_gpus(1)
    assert len(one_gpu) == 1

    two_gpu = runs.num_gpus(2)
    assert len(two_gpu) == 1


def test_diffusion_runs_precision_filter(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    bf16 = runs.precision("bfloat16")
    assert len(bf16) == 1

    fp8 = runs.precision("fp8")
    assert len(fp8) == 0


def test_diffusion_runs_label_filter(tmp_path: Path) -> None:
    from mlenergy_data.records.runs import DiffusionRuns

    root = tmp_path / "bench"
    cfg_root = tmp_path / "cfg"
    _make_diffusion_fixture(root, cfg_root, batch=1)

    runs = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
    label = runs[0].model_label
    filtered = runs.label(label)
    assert len(filtered) == 1

    empty = runs.label("nonexistent-label")
    assert len(empty) == 0
