"""Timeline extraction helpers for benchmark `results.json` payloads."""

from __future__ import annotations

import pandas as pd


def extract_device_timeline(
    device_series: dict[str, list[list[float]]],
    *,
    label: str,
    steady_state_start: float | None = None,
    steady_state_end: float | None = None,
) -> pd.DataFrame:
    """Extract aligned per-device timeseries from a gpu_id -> samples mapping.

    Args:
        device_series: Mapping of GPU ID (as string) to list of
            `[timestamp, value]` pairs.
        label: Base name for columns. Produces `{label}_device_{id}`
            per GPU and `{label}_total` (sum over devices).
        steady_state_start: If both start and end are provided, restrict
            to this time window.
        steady_state_end: See `steady_state_start`.

    Returns:
        DataFrame with columns `timestamp`, `relative_time_s`,
        `{label}_device_<id>` for each device, and `{label}_total`.
    """
    if not device_series:
        raise ValueError("device_series is empty")

    ss_start: float | None = None
    ss_end: float | None = None
    if steady_state_start is not None and steady_state_end is not None:
        ss_start = float(steady_state_start)
        ss_end = float(steady_state_end)

    gpu_ids = sorted(int(k) for k in device_series)
    base_gpu = str(gpu_ids[0])
    base_points = device_series[base_gpu]

    timestamps: list[float] = []
    for ts, _value in base_points:
        t = float(ts)
        if ss_start is not None and ss_end is not None:
            if ss_start <= t <= ss_end:
                timestamps.append(t)
        else:
            timestamps.append(t)

    if not timestamps:
        raise ValueError("No timeline samples found in selected range")

    df = pd.DataFrame({"timestamp": timestamps})
    t0 = ss_start if ss_start is not None else float(df["timestamp"].iloc[0])
    df["relative_time_s"] = df["timestamp"] - t0

    cols: list[str] = []
    for gid in gpu_ids:
        series = device_series[str(gid)]
        if ss_start is not None and ss_end is not None:
            filtered = [
                (float(ts), float(v)) for ts, v in series if ss_start <= float(ts) <= ss_end
            ]
        else:
            filtered = [(float(ts), float(v)) for ts, v in series]

        ts_to_value = {ts: v for ts, v in filtered}
        col = f"{label}_device_{gid}"
        cols.append(col)
        df[col] = df["timestamp"].map(ts_to_value)

    df = df.sort_values("relative_time_s").reset_index(drop=True)
    df[cols] = df[cols].ffill()
    df[f"{label}_total"] = df[cols].sum(axis=1)
    return df
