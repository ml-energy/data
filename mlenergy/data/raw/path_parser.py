"""Path parsing helpers for ML.ENERGY raw benchmark layouts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedPath:
    """Parsed components for a raw run file path.

    Attributes:
        domain: Benchmark domain (`"llm"` or `"diffusion"`).
        gpu_family: GPU family identifier (e.g. `"h100"`).
        snapshot: Snapshot identifier (e.g. `"current"`).
        workload_family: Top-level workload suite name.
        workload: Specific workload/task name.
        org: Model organization (e.g. `"meta-llama"`).
        model: Model name (e.g. `"Llama-3.1-8B"`).
        config: Parsed key-value configuration from the path.
        relpath: Original path relative to the benchmark root.
    """

    domain: str
    gpu_family: str
    snapshot: str
    workload_family: str
    workload: str
    org: str
    model: str
    config: dict[str, str]
    relpath: str


def _parse_llm(parts: list[str], relpath: str) -> ParsedPath:
    # llm/{gpu}/{snapshot}/run/{suite}/{workload}/results/{org}/{model}/
    #     {gpu_sku}/{config}/results.json
    suite = parts[3]
    workload = parts[4]
    org = parts[6]
    model = parts[7]
    cfg_raw = parts[9]
    config = _parse_plus_kv(cfg_raw)
    config["suite"] = suite
    return ParsedPath(
        domain="llm",
        gpu_family=parts[0],
        snapshot=parts[1],
        workload_family=suite,
        workload=workload,
        org=org,
        model=model,
        config=config,
        relpath=relpath,
    )


def _parse_diffusion(parts: list[str], relpath: str) -> ParsedPath:
    # diffusion/{gpu}/{snapshot}/run/diffusion/{workload}/{org}/{model}/{config}/results.json
    workload = parts[4]
    org = parts[5]
    model = parts[6]
    cfg_raw = parts[7]
    config = _parse_dash_kv(cfg_raw)
    return ParsedPath(
        domain="diffusion",
        gpu_family=parts[0],
        snapshot=parts[1],
        workload_family="diffusion",
        workload=workload,
        org=org,
        model=model,
        config=config,
        relpath=relpath,
    )


def _parse_plus_kv(raw: str) -> dict[str, str]:
    toks = raw.split("+")
    out: dict[str, str] = {}
    for i in range(0, len(toks) - 1, 2):
        out[toks[i]] = toks[i + 1]
    return out


def _parse_dash_kv(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in raw.split("+"):
        if "-" not in tok:
            continue
        k, v = tok.split("-", 1)
        out[k] = v
    return out


def parse_results_path(path: str | Path, root: str | Path) -> ParsedPath:
    """Parse a raw `results.json` path under benchmark root."""
    p = Path(path)
    rel = p.relative_to(Path(root)).as_posix()
    parts = rel.split("/")
    if len(parts) < 5:
        raise ValueError(f"Unrecognized results path: {rel}")

    domain = parts[0]
    if domain == "llm":
        return _parse_llm(parts[1:], rel)
    if domain == "diffusion":
        return _parse_diffusion(parts[1:], rel)

    raise ValueError(f"Unsupported domain={domain!r} for path={rel}")
