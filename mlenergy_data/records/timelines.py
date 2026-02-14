"""Timeline extraction helpers for benchmark `results.json` payloads."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_metric_series(
    timeline: dict[str, Any],
    *,
    metric_group: str,
    metric_name: str,
) -> dict[str, list[list[float]]]:
    group = timeline.get(metric_group)
    if not isinstance(group, dict):
        raise ValueError(f"timeline.{metric_group} is missing or not a dict")
    series = group.get(metric_name)
    if not isinstance(series, dict):
        raise ValueError(f"timeline.{metric_group}.{metric_name} is missing or not a dict")
    return series


def extract_device_timeline(
    results: dict[str, Any],
    *,
    metric_group: str,
    metric_name: str,
    steady_state_only: bool = True,
) -> pd.DataFrame:
    """Extract aligned per-device timeseries table from `results` timeline.

    Args:
        results: Parsed `results.json` dict.
        metric_group: Top-level metric group (e.g. `"power"`).
        metric_name: Metric name within the group (e.g. `"device_instant"`).
        steady_state_only: If True, restrict to steady-state window.

    Returns:
        DataFrame with columns `timestamp`, `relative_time_s`,
        `{metric_group}_{metric_name}_device_<id>` for each device,
        and `{metric_group}_{metric_name}_total` (sum over devices).
    """
    timeline = results.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("results.timeline is missing or not a dict")

    device_inst = _resolve_metric_series(
        timeline,
        metric_group=metric_group,
        metric_name=metric_name,
    )
    if not device_inst:
        raise ValueError(f"No data in timeline.{metric_group}.{metric_name}")

    if steady_state_only:
        ss_start = timeline.get("steady_state_start_time")
        ss_end = timeline.get("steady_state_end_time")
        if ss_start is None or ss_end is None:
            raise ValueError("Steady-state bounds missing from results.timeline")
        ss_start_f = float(ss_start)
        ss_end_f = float(ss_end)
    else:
        ss_start_f = 0.0
        ss_end_f = 0.0

    gpu_ids = sorted(device_inst.keys(), key=lambda x: int(x))
    base_gpu = gpu_ids[0]
    base_points = device_inst[base_gpu]

    timestamps: list[float] = []
    for ts, _value in base_points:
        if steady_state_only:
            if ss_start_f <= float(ts) <= ss_end_f:
                timestamps.append(float(ts))
        else:
            timestamps.append(float(ts))

    if not timestamps:
        raise ValueError("No timeline samples found in selected range")

    df = pd.DataFrame({"timestamp": timestamps})
    t0 = ss_start_f if steady_state_only else float(df["timestamp"].iloc[0])
    df["relative_time_s"] = df["timestamp"] - t0

    cols: list[str] = []
    for gid in gpu_ids:
        series = device_inst[gid]
        if steady_state_only:
            filtered = [
                (float(ts), float(v)) for ts, v in series if ss_start_f <= float(ts) <= ss_end_f
            ]
        else:
            filtered = [(float(ts), float(v)) for ts, v in series]

        ts_to_value = {ts: v for ts, v in filtered}
        col = f"{metric_group}_{metric_name}_device_{gid}"
        cols.append(col)
        df[col] = df["timestamp"].map(ts_to_value)

    df = df.sort_values("relative_time_s").reset_index(drop=True)
    df[cols] = df[cols].ffill()
    total_col = f"{metric_group}_{metric_name}_total"
    df[total_col] = df[cols].sum(axis=1)
    return df


def extract_power_device_instant(results: dict[str, Any]) -> pd.DataFrame:
    """Extract `timeline.power.device_instant` as aligned timeseries."""
    return extract_device_timeline(results, metric_group="power", metric_name="device_instant")


def extract_power_device_average(results: dict[str, Any]) -> pd.DataFrame:
    """Extract `timeline.power.device_average` as aligned timeseries."""
    return extract_device_timeline(results, metric_group="power", metric_name="device_average")


def extract_temperature_timeseries(results: dict[str, Any]) -> pd.DataFrame:
    """Extract device temperature timeseries.

    Temperature metric names vary across generators. This resolves in order:
    `timeline.temperature.device_instant`,
    `timeline.temperature.device_average`,
    `timeline.temperature.device`.
    """
    timeline = results.get("timeline")
    if not isinstance(timeline, dict):
        raise ValueError("results.timeline is missing or not a dict")
    group = timeline.get("temperature")
    if not isinstance(group, dict):
        raise ValueError("timeline.temperature missing")

    for metric_name in ("device_instant", "device_average", "device"):
        if metric_name in group:
            return extract_device_timeline(
                results,
                metric_group="temperature",
                metric_name=metric_name,
            )
    raise ValueError("No recognized timeline.temperature device metric")


def load_timeline_table(
    records: list[dict[str, Any]] | pd.DataFrame,
    *,
    metric: str = "power.device_instant",
) -> pd.DataFrame:
    """Extract long-form timeline table for runs.

    Accepts either a list of record dicts (from collection types) or a
    DataFrame with the required columns.

    Args:
        records: List of record dicts or DataFrame with run metadata.
        metric: Metric to extract. Supported values: `"power.device_instant"`,
            `"power.device_average"`, `"temperature"`.

    Returns:
        Long-form DataFrame with columns `results_path`, `domain`, `task`,
        `model_id`, `model_label`, `num_gpus`, `max_num_seqs`,
        `batch_size`, `timestamp`, `relative_time_s`, `value`,
        `metric`.
    """
    if isinstance(records, pd.DataFrame):
        rec_list: list[dict[str, Any]] = records.to_dict(orient="records")
    else:
        rec_list = records

    logger.info("Extracting timelines (metric=%s) for %d runs", metric, len(rec_list))
    rows: list[pd.DataFrame] = []
    for rec in rec_list:
        if rec.get("model_label") is None:
            raise ValueError(
                "Missing required model_label in run table for "
                f"results_path={rec.get('results_path')}"
            )
        p = Path(str(rec.get("resolved_path", rec["results_path"])))
        payload = json.loads(p.read_text())
        if metric == "power.device_instant":
            wide = extract_power_device_instant(payload)
            value_col = "power_device_instant_total"
        elif metric == "power.device_average":
            wide = extract_power_device_average(payload)
            value_col = "power_device_average_total"
        elif metric == "temperature":
            wide = extract_temperature_timeseries(payload)
            value_col = [c for c in wide.columns if c.endswith("_total")]
            if not value_col:
                raise ValueError("temperature timeline extraction did not produce *_total column")
            value_col = value_col[0]
        else:
            raise ValueError(f"Unsupported metric={metric!r}")

        local = pd.DataFrame(
            {
                "results_path": rec["results_path"],
                "domain": rec["domain"],
                "task": rec["task"],
                "model_id": rec["model_id"],
                "model_label": rec["model_label"],
                "num_gpus": rec["num_gpus"],
                "max_num_seqs": rec.get("max_num_seqs"),
                "batch_size": rec.get("batch_size"),
                "timestamp": wide["timestamp"],
                "relative_time_s": wide["relative_time_s"],
                "value": wide[value_col],
                "metric": metric,
            }
        )
        rows.append(local)

    if not rows:
        return pd.DataFrame(
            columns=[
                "results_path",
                "domain",
                "task",
                "model_id",
                "model_label",
                "num_gpus",
                "max_num_seqs",
                "batch_size",
                "timestamp",
                "relative_time_s",
                "value",
                "metric",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["results_path", "relative_time_s"]).reset_index(drop=True)
    logger.info("Extracted %d timeline rows from %d runs", len(out), len(rows))
    return out
