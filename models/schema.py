"""Shared model feature and artifact schema."""

FEATURE_COLUMNS = [
    "value", "mean_5m", "std_5m", "slope_5m", "diff_mean_abs_5m", "diff_max_abs_5m",
    "mean_15m", "std_15m", "slope_15m", "diff_mean_abs_15m", "diff_max_abs_15m",
    "mean_60m", "std_60m", "slope_60m", "diff_mean_abs_60m", "diff_max_abs_60m",
    "self_zscore_60m", "missing_ratio_60m", "repeated_count_15m", "peer_count",
    "peer_median", "peer_mad", "peer_residual", "peer_zscore", "group_baseline_zscore", "event_consensus",
]
