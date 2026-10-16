"""Time-based features for sensor health and cross-sensor event evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "run_id", "sensor_id", "group_id", "value"}


def _one_sensor_features(sensor: pd.DataFrame) -> pd.DataFrame:
    sensor = sensor.sort_values("timestamp").copy()
    series = pd.Series(sensor["value"].to_numpy(dtype=float),
                       index=pd.DatetimeIndex(sensor["timestamp"], name="timestamp"))
    diff = series.diff()
    out = sensor.set_index("timestamp")

    for minutes in (5, 15, 60):
        window = f"{minutes}min"
        rolling = series.rolling(window, min_periods=2)
        mean = rolling.mean()
        std = rolling.std().fillna(0.0)
        out[f"mean_{minutes}m"] = mean.to_numpy()
        out[f"std_{minutes}m"] = std.to_numpy()
        def slope(values: np.ndarray) -> float:
            valid = values[np.isfinite(values)]
            return float((valid[-1] - valid[0]) / minutes) if len(valid) > 1 else np.nan

        out[f"slope_{minutes}m"] = rolling.apply(slope, raw=True).to_numpy()
        out[f"diff_mean_abs_{minutes}m"] = diff.abs().rolling(window, min_periods=1).mean().to_numpy()
        out[f"diff_max_abs_{minutes}m"] = diff.abs().rolling(window, min_periods=1).max().to_numpy()

    baseline_mean = series.rolling("60min", min_periods=5).mean()
    baseline_std = series.rolling("60min", min_periods=5).std().replace(0, np.nan)
    out["self_zscore_60m"] = ((series - baseline_mean) / baseline_std).replace([np.inf, -np.inf], np.nan).to_numpy()
    out["missing_ratio_60m"] = series.isna().astype(float).rolling("60min", min_periods=1).mean().to_numpy()
    repeated = series.eq(series.shift(1)) & series.notna()
    out["repeated_count_15m"] = repeated.astype(float).rolling("15min", min_periods=1).sum().to_numpy()
    return out.reset_index()


def extract_features(readings: pd.DataFrame, consensus_z: float = 3.0) -> pd.DataFrame:
    """Add rolling self-health and peer-consensus features to long-form readings.

    The input may contain missing sensor rows and irregular timestamps. Time-based
    windows are used, and rows are never joined across independent runs.
    """
    missing = REQUIRED_COLUMNS - set(readings.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(sorted(missing))}")
    if consensus_z <= 0:
        raise ValueError("consensus_z must be positive")

    data = readings.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["timestamp"])
    keys = ["run_id", "sensor_id"]
    features = pd.concat(
        [_one_sensor_features(group) for _, group in data.groupby(keys, sort=False, dropna=False)],
        ignore_index=True,
    )

    peers = ["run_id", "group_id", "timestamp"]
    grouped = features.groupby(peers, sort=False, dropna=False)["value"]
    features["peer_count"] = grouped.transform("count") - features["value"].notna().astype(int)
    features["peer_median"] = grouped.transform("median")
    features["peer_mad"] = grouped.transform(lambda x: (x - x.median()).abs().median())
    scale = (1.4826 * features["peer_mad"]).clip(lower=0.18)
    features["peer_residual"] = features["value"] - features["peer_median"]
    features["peer_zscore"] = (features["peer_residual"] / scale).replace([np.inf, -np.inf], np.nan)
    features["peer_deviation"] = features["peer_zscore"].abs().ge(consensus_z) & features["peer_zscore"].notna()

    group_series = features[peers + ["peer_median"]].drop_duplicates(peers).sort_values(peers)
    baseline_parts = []
    for _, group in group_series.groupby(["run_id", "group_id"], sort=False, dropna=False):
        group = group.sort_values("timestamp").copy()
        history = group[["timestamp", "peer_median"]].rename(columns={
            "timestamp": "history_timestamp", "peer_median": "previous_day_median",
        }).sort_values("history_timestamp")
        lookup = group[["timestamp", "peer_median"]].copy()
        lookup["baseline_lookup"] = lookup["timestamp"] - pd.Timedelta(hours=24)
        paired = pd.merge_asof(
            lookup.sort_values("baseline_lookup"), history,
            left_on="baseline_lookup", right_on="history_timestamp",
            direction="nearest", tolerance=pd.Timedelta(minutes=30),
        ).sort_values("timestamp")
        residual = paired["peer_median"] - paired["previous_day_median"]
        residual_series = pd.Series(residual.to_numpy(dtype=float),
                                    index=pd.DatetimeIndex(paired["timestamp"]))
        variability = residual_series.rolling("24h", min_periods=60).std().shift(1).clip(lower=0.25)
        group["group_baseline"] = paired["previous_day_median"].to_numpy()
        group["group_scale"] = variability.to_numpy()
        baseline_parts.append(group[peers + ["group_baseline", "group_scale"]])
    baseline_features = pd.concat(baseline_parts, ignore_index=True)
    features = features.merge(baseline_features, on=peers, how="left", validate="many_to_one")
    features["group_baseline_zscore"] = (
        (features["value"] - features["group_baseline"]) / features["group_scale"]
    ).replace([np.inf, -np.inf], np.nan)
    features["group_sensor_deviation"] = features["group_baseline_zscore"].abs().ge(4.0) & features["group_baseline_zscore"].notna()
    features["event_consensus"] = features.groupby(peers, sort=False, dropna=False)["group_sensor_deviation"].transform("mean")
    features.loc[features["peer_count"] <= 0, "event_consensus"] = np.nan
    return features.sort_values(["run_id", "sensor_id", "timestamp"]).reset_index(drop=True)
