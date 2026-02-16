"""Tests for parquet round-trip and from_hf loading."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from mlenergy_data.records.runs import (
    DiffusionRun,
    DiffusionRuns,
    LLMRun,
    LLMRuns,
    _HFSource,
)

LLM_FIELDS = frozenset(f.name for f in dataclasses.fields(LLMRun))
DIFFUSION_FIELDS = frozenset(f.name for f in dataclasses.fields(DiffusionRun))


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
    """Create a minimal LLM benchmark directory with one run."""
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


class TestLLMParquetRoundTrip:
    """Test LLM runs survive parquet serialization + deserialization."""

    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

        original = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        assert len(original) == 2

        parquet_path = tmp_path / "llm.parquet"
        original.to_dataframe().to_parquet(parquet_path, index=False)

        restored = LLMRuns.from_parquet(parquet_path, stable_only=False)
        assert len(restored) == len(original)

        for orig, rest in zip(
            sorted(original, key=lambda r: r.results_path),
            sorted(restored, key=lambda r: r.results_path),
            strict=True,
        ):
            for field in LLM_FIELDS:
                orig_val = getattr(orig, field)
                rest_val = getattr(rest, field)
                if isinstance(orig_val, float):
                    assert abs(orig_val - rest_val) < 1e-6, (
                        f"Field {field}: {orig_val} != {rest_val}"
                    )
                else:
                    assert orig_val == rest_val, f"Field {field}: {orig_val!r} != {rest_val!r}"

    def test_stable_only_filter(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1, avg_batch=16.0)
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2, avg_batch=8.0)

        all_runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)

        parquet_path = tmp_path / "llm.parquet"
        all_runs.to_dataframe().to_parquet(parquet_path, index=False)

        stable_only = LLMRuns.from_parquet(parquet_path, stable_only=True)
        all_restored = LLMRuns.from_parquet(parquet_path, stable_only=False)

        assert len(all_restored) == len(all_runs)
        assert len(stable_only) < len(all_restored)
        assert all(r.is_stable for r in stable_only)

    def test_task_filter(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        parquet_path = tmp_path / "llm.parquet"
        runs.to_dataframe().to_parquet(parquet_path, index=False)

        filtered = LLMRuns.from_parquet(parquet_path, stable_only=False).task("gpqa")
        assert len(filtered) == 1

        empty = LLMRuns.from_parquet(parquet_path, stable_only=False).task("nonexistent")
        assert len(empty) == 0

    def test_base_dir_resolution(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        df = runs.to_dataframe()

        df["results_path"] = "relative/path/results.json"
        df["prometheus_path"] = "relative/path/prometheus.json"

        parquet_path = tmp_path / "llm.parquet"
        df.to_parquet(parquet_path, index=False)

        base = tmp_path / "base"
        restored = LLMRuns.from_parquet(parquet_path, base_dir=base, stable_only=False)
        assert restored[0].results_path == str(base / "relative/path/results.json")
        assert restored[0].prometheus_path == str(base / "relative/path/prometheus.json")


class TestDiffusionParquetRoundTrip:
    """Test diffusion runs survive parquet serialization + deserialization."""

    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_diffusion_fixture(root, cfg_root, batch=1)
        _make_diffusion_fixture(root, cfg_root, batch=2, size="1024x1024")

        original = DiffusionRuns.from_raw_results(root, config_dir=cfg_root)
        assert len(original) == 2

        parquet_path = tmp_path / "diffusion.parquet"
        original.to_dataframe().to_parquet(parquet_path, index=False)

        restored = DiffusionRuns.from_parquet(parquet_path)
        assert len(restored) == len(original)

        for orig, rest in zip(
            sorted(original, key=lambda r: r.results_path),
            sorted(restored, key=lambda r: r.results_path),
            strict=True,
        ):
            for field in DIFFUSION_FIELDS:
                orig_val = getattr(orig, field)
                rest_val = getattr(rest, field)
                if orig_val is None and rest_val is None:
                    continue
                if isinstance(orig_val, float):
                    assert abs(orig_val - rest_val) < 1e-6, (
                        f"Field {field}: {orig_val} != {rest_val}"
                    )
                else:
                    assert orig_val == rest_val, f"Field {field}: {orig_val!r} != {rest_val!r}"


class TestAutoDetectConfig:
    """Test config auto-detection in from_directory."""

    def test_auto_detect_llm_config(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "bench" / "configs" / "vllm"
        _make_benchmark_fixture(root, cfg_root)

        runs = LLMRuns.from_raw_results(root, stable_only=False)
        assert len(runs) == 1
        assert runs[0].task == "gpqa"

    def test_auto_detect_diffusion_config(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "bench" / "configs" / "xdit"
        _make_diffusion_fixture(root, cfg_root)

        runs = DiffusionRuns.from_raw_results(root)
        assert len(runs) == 1

    def test_explicit_config_overrides_auto_detect(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        auto_cfg = tmp_path / "bench" / "configs" / "vllm"
        explicit_cfg = tmp_path / "explicit_cfg"
        _make_benchmark_fixture(root, auto_cfg)
        _make_benchmark_fixture(root, explicit_cfg, seed=2)

        runs = LLMRuns.from_raw_results(root, config_dir=explicit_cfg, stable_only=False)
        assert len(runs) >= 1


class TestFromDirectory:
    """Test the parquet-first from_directory API."""

    def _build_compiled_repo(
        self,
        tmp_path: Path,
        *,
        num_llm_runs: int = 2,
        num_diff_runs: int = 1,
    ) -> Path:
        """Build a compiled data repo with parquet files from raw fixtures."""
        root = tmp_path / "raw_bench"
        llm_cfg = tmp_path / "llm_cfg"
        diff_cfg = tmp_path / "diff_cfg"

        for i in range(num_llm_runs):
            _make_benchmark_fixture(root, llm_cfg, max_num_seqs=16 * (i + 1), seed=i + 1)
        for i in range(num_diff_runs):
            _make_diffusion_fixture(
                root,
                diff_cfg,
                batch=i + 1,
            )

        llm = LLMRuns.from_raw_results(root, config_dir=llm_cfg, stable_only=False)
        diff = DiffusionRuns.from_raw_results(root, config_dir=diff_cfg)

        compiled = tmp_path / "compiled"
        runs_dir = compiled / "runs"
        runs_dir.mkdir(parents=True)
        llm.to_dataframe().to_parquet(runs_dir / "llm.parquet", index=False)
        diff.to_dataframe().to_parquet(runs_dir / "diffusion.parquet", index=False)
        return compiled

    def test_llm_from_directory(self, tmp_path: Path) -> None:
        compiled = self._build_compiled_repo(tmp_path)
        runs = LLMRuns.from_directory(compiled, stable_only=False)
        assert len(runs) == 2
        assert all(r.task == "gpqa" for r in runs)

    def test_llm_from_directory_stable_only(self, tmp_path: Path) -> None:
        compiled = self._build_compiled_repo(tmp_path)
        all_runs = LLMRuns.from_directory(compiled, stable_only=False)
        stable_runs = LLMRuns.from_directory(compiled, stable_only=True)
        assert len(stable_runs) <= len(all_runs)
        assert all(r.is_stable for r in stable_runs)

    def test_llm_from_directory_task_filter(self, tmp_path: Path) -> None:
        compiled = self._build_compiled_repo(tmp_path)
        filtered = LLMRuns.from_directory(compiled, stable_only=False).task("gpqa")
        assert len(filtered) == 2
        empty = LLMRuns.from_directory(compiled, stable_only=False).task("nonexistent")
        assert len(empty) == 0

    def test_diffusion_from_directory(self, tmp_path: Path) -> None:
        compiled = self._build_compiled_repo(tmp_path)
        runs = DiffusionRuns.from_directory(compiled)
        assert len(runs) == 1

    def test_diffusion_from_directory_task_filter(self, tmp_path: Path) -> None:
        compiled = self._build_compiled_repo(tmp_path)
        filtered = DiffusionRuns.from_directory(compiled).task("text-to-image")
        assert len(filtered) == 1
        empty = DiffusionRuns.from_directory(compiled).task("nonexistent")
        assert len(empty) == 0

    def test_from_directory_base_dir_resolves_paths(self, tmp_path: Path) -> None:
        """Verify relative paths in parquet are resolved against the compiled root."""
        root = tmp_path / "raw_bench"
        cfg = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg)

        raw = LLMRuns.from_raw_results(root, config_dir=cfg, stable_only=False)
        df = raw.to_dataframe()
        df["results_path"] = "relative/path/results.json"
        df["prometheus_path"] = "relative/path/prometheus.json"

        compiled = tmp_path / "compiled"
        runs_dir = compiled / "runs"
        runs_dir.mkdir(parents=True)
        df.to_parquet(runs_dir / "llm.parquet", index=False)

        runs = LLMRuns.from_directory(compiled, stable_only=False)
        assert runs[0].results_path == str(compiled / "relative/path/results.json")
        assert runs[0].prometheus_path == str(compiled / "relative/path/prometheus.json")


class TestMissingRawDataError:
    """Test clear error when accessing raw data methods without raw files."""

    def test_output_lengths_missing_raw(self, tmp_path: Path) -> None:
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        df = runs.to_dataframe()
        df["results_path"] = "/nonexistent/path/results.json"

        parquet_path = tmp_path / "llm.parquet"
        df.to_parquet(parquet_path, index=False)

        restored = LLMRuns.from_parquet(parquet_path, stable_only=False)
        with pytest.raises(FileNotFoundError, match="Raw results file not found"):
            restored.output_lengths()


class TestHFSourcePropagation:
    """Test _hf_source propagation through filter chains."""

    @pytest.fixture()
    def hf_runs(self, tmp_path: Path) -> LLMRuns:
        """Build an LLMRuns with _hf_source set."""
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=2)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        source = _HFSource(repo_id="test/repo", revision=None, cache_dir=None)
        return LLMRuns(list(runs), _hf_source=source)

    def test_filter_propagates_source(self, hf_runs: LLMRuns) -> None:
        filtered = hf_runs.task("gpqa")
        assert filtered._hf_source == hf_runs._hf_source

    def test_where_propagates_source(self, hf_runs: LLMRuns) -> None:
        filtered = hf_runs.where(lambda r: r.max_num_seqs == 16)
        assert filtered._hf_source == hf_runs._hf_source

    def test_stable_propagates_source(self, hf_runs: LLMRuns) -> None:
        filtered = hf_runs.stable()
        assert filtered._hf_source == hf_runs._hf_source

    def test_group_by_propagates_source(self, hf_runs: LLMRuns) -> None:
        groups = hf_runs.group_by("max_num_seqs")
        for group in groups.values():
            assert group._hf_source == hf_runs._hf_source

    def test_add_same_source_preserves(self, hf_runs: LLMRuns) -> None:
        a = hf_runs.batch(16)
        b = hf_runs.batch(32)
        combined = a + b
        assert combined._hf_source == hf_runs._hf_source

    def test_add_different_source_drops(self, hf_runs: LLMRuns) -> None:
        other_source = _HFSource(repo_id="other/repo", revision=None, cache_dir=None)
        other = LLMRuns(list(hf_runs), _hf_source=other_source)
        combined = hf_runs + other
        assert combined._hf_source is None

    def test_auto_fetch_output_lengths(self, tmp_path: Path) -> None:
        """_ensure_raw_files triggers download_file for HF-sourced collections."""
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=1)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        source = _HFSource(repo_id="test/repo", revision=None, cache_dir=None)

        actual_path = runs[0].results_path
        hf_runs = LLMRuns(list(runs), _hf_source=source)

        with patch("mlenergy_data.records.runs.download_file", return_value=actual_path) as mock:
            df = hf_runs.output_lengths()
            assert len(df) > 0
            assert mock.call_count >= 1
            called_filenames = {call.args[1] for call in mock.call_args_list}
            assert runs[0].results_path in called_filenames


class TestSelectiveDownload:
    """Verify that filtering narrows the set of raw files downloaded."""

    @pytest.fixture()
    def multi_run_setup(self, tmp_path: Path) -> tuple[LLMRuns, dict[int, str]]:
        """Build 3 runs with different batch sizes, return HF-sourced LLMRuns."""
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=8, seed=1)
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=16, seed=2)
        _make_benchmark_fixture(root, cfg_root, max_num_seqs=32, seed=3)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        assert len(runs) == 3

        source = _HFSource(repo_id="test/repo", revision=None, cache_dir=None)
        hf_runs = LLMRuns(list(runs), _hf_source=source)
        paths_by_batch = {r.max_num_seqs: r.results_path for r in runs}
        return hf_runs, paths_by_batch

    def _mock_download(self, paths_by_batch: dict[int, str]):
        """Return a side_effect that maps filenames to real local paths."""
        all_paths = set(paths_by_batch.values())
        prom_paths = {p.replace("results.json", "prometheus.json") for p in all_paths}
        known = all_paths | prom_paths

        def _side_effect(repo_id: str, filename: str, **kwargs: object) -> str:
            if filename in known:
                return filename
            raise FileNotFoundError(f"mock: unknown file {filename}")

        return _side_effect

    def test_filtered_downloads_only_matching_files(
        self,
        multi_run_setup: tuple[LLMRuns, dict[int, str]],
    ) -> None:
        """Filtering to batch=8 should only download files for that one run."""
        hf_runs, paths_by_batch = multi_run_setup
        filtered = hf_runs.batch(8)
        assert len(filtered) == 1

        with patch(
            "mlenergy_data.records.runs.download_file",
            side_effect=self._mock_download(paths_by_batch),
        ) as mock:
            filtered.output_lengths()
            downloaded = {call.args[1] for call in mock.call_args_list}
            assert paths_by_batch[8] in downloaded
            assert paths_by_batch[16] not in downloaded
            assert paths_by_batch[32] not in downloaded

    def test_unfiltered_downloads_all_files(
        self,
        multi_run_setup: tuple[LLMRuns, dict[int, str]],
    ) -> None:
        """The full collection should download files for all 3 runs."""
        hf_runs, paths_by_batch = multi_run_setup

        with patch(
            "mlenergy_data.records.runs.download_file",
            side_effect=self._mock_download(paths_by_batch),
        ) as mock:
            hf_runs.output_lengths()
            downloaded = {call.args[1] for call in mock.call_args_list}
            for batch, path in paths_by_batch.items():
                assert path in downloaded, f"batch={batch} file not downloaded"

    def test_chained_filter_narrows_downloads(self, tmp_path: Path) -> None:
        """Chaining .task().batch() should narrow downloads to the intersection."""
        root = tmp_path / "bench"
        cfg_root = tmp_path / "cfg"
        # Two tasks x two batch sizes = 4 runs
        _make_benchmark_fixture(root, cfg_root, task="gpqa", max_num_seqs=16, seed=1)
        _make_benchmark_fixture(root, cfg_root, task="gpqa", max_num_seqs=32, seed=2)
        _make_benchmark_fixture(root, cfg_root, task="lm-arena-chat", max_num_seqs=16, seed=3)
        _make_benchmark_fixture(root, cfg_root, task="lm-arena-chat", max_num_seqs=32, seed=4)

        runs = LLMRuns.from_raw_results(root, config_dir=cfg_root, stable_only=False)
        assert len(runs) == 4

        source = _HFSource(repo_id="test/repo", revision=None, cache_dir=None)
        hf_runs = LLMRuns(list(runs), _hf_source=source)

        # Index by (task, batch) for clear assertions
        path_index = {(r.task, r.max_num_seqs): r.results_path for r in runs}
        all_paths = set(path_index.values())
        prom_paths = {p.replace("results.json", "prometheus.json") for p in all_paths}
        known = all_paths | prom_paths

        def _side_effect(repo_id: str, filename: str, **kwargs: object) -> str:
            if filename in known:
                return filename
            raise FileNotFoundError(f"mock: unknown file {filename}")

        # Filter to gpqa + batch=16: should be exactly 1 run
        filtered = hf_runs.task("gpqa").batch(16)
        assert len(filtered) == 1

        with patch(
            "mlenergy_data.records.runs.download_file",
            side_effect=_side_effect,
        ) as mock:
            filtered.output_lengths()
            downloaded = {call.args[1] for call in mock.call_args_list}
            assert path_index[("gpqa", 16)] in downloaded
            assert path_index[("gpqa", 32)] not in downloaded
            assert path_index[("lm-arena-chat", 16)] not in downloaded
            assert path_index[("lm-arena-chat", 32)] not in downloaded

    def test_timelines_uses_selective_download(
        self,
        multi_run_setup: tuple[LLMRuns, dict[int, str]],
    ) -> None:
        """timelines() on a filtered collection should also scope downloads."""
        hf_runs, paths_by_batch = multi_run_setup
        filtered = hf_runs.batch(32)
        assert len(filtered) == 1

        with patch(
            "mlenergy_data.records.runs.download_file",
            side_effect=self._mock_download(paths_by_batch),
        ) as mock:
            with contextlib.suppress(Exception):
                filtered.timelines(metric="power.device_instant")
            downloaded = {call.args[1] for call in mock.call_args_list}
            assert paths_by_batch[32] in downloaded
            assert paths_by_batch[8] not in downloaded
            assert paths_by_batch[16] not in downloaded

    def test_prefetch_downloads_current_scope(
        self,
        multi_run_setup: tuple[LLMRuns, dict[int, str]],
    ) -> None:
        """prefetch() on a filtered collection only downloads that subset."""
        hf_runs, paths_by_batch = multi_run_setup
        filtered = hf_runs.batch(8, 16)
        assert len(filtered) == 2

        with patch(
            "mlenergy_data.records.runs.download_file",
            side_effect=self._mock_download(paths_by_batch),
        ) as mock:
            filtered.prefetch()
            downloaded = {call.args[1] for call in mock.call_args_list}
            assert paths_by_batch[8] in downloaded
            assert paths_by_batch[16] in downloaded
            assert paths_by_batch[32] not in downloaded

    def test_cache_prevents_repeated_downloads(
        self,
        multi_run_setup: tuple[LLMRuns, dict[int, str]],
    ) -> None:
        """Second call to output_lengths() should not re-download."""
        hf_runs, paths_by_batch = multi_run_setup
        filtered = hf_runs.batch(8)

        with patch(
            "mlenergy_data.records.runs.download_file",
            side_effect=self._mock_download(paths_by_batch),
        ) as mock:
            filtered.output_lengths()
            first_count = mock.call_count
            assert first_count >= 1
            filtered.output_lengths()
            assert mock.call_count == first_count
