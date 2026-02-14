from __future__ import annotations

import json
from pathlib import Path

from mlenergy_data.raw.path_parser import parse_results_path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_parse_llm_results_path(tmp_path: Path):
    root = tmp_path / "bench"
    p = (
        root
        / "llm"
        / "h100"
        / "current"
        / "run"
        / "llm"
        / "lm-arena-chat"
        / "results"
        / "meta-llama"
        / "Llama-3.1-8B-Instruct"
        / "H100"
        / (
            "num_gpus+1+max_num_seqs+64+num_request_repeats+1+dataset_split+train+"
            "num_requests+1024+seed+48105"
        )
        / "results.json"
    )
    _write_json(p, {"steady_state_energy": 1.0, "steady_state_duration": 1.0})

    parsed = parse_results_path(p, root)
    assert parsed.domain == "llm"
    assert parsed.gpu_family == "h100"
    assert parsed.workload == "lm-arena-chat"
    assert parsed.config["num_gpus"] == "1"


def test_parse_diffusion_results_path(tmp_path: Path):
    root = tmp_path / "bench"
    p = (
        root
        / "diffusion"
        / "b200"
        / "current"
        / "run"
        / "diffusion"
        / "text-to-image"
        / "black-forest-labs"
        / "FLUX.1-dev"
        / "batch-4+size-1024x1024+steps-50+seed-48105+uly-1+ring-8+tc-False"
        / "results.json"
    )
    _write_json(p, {"total_energy": 10.0, "total_images": 4})

    parsed = parse_results_path(p, root)
    assert parsed.domain == "diffusion"
    assert parsed.workload == "text-to-image"
    assert parsed.config["batch"] == "4"
