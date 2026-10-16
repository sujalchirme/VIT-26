"""Rule and model fusion for environment alarms and sensor-health decisions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.config import load_config


def _truth(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _reason(row: pd.Series, thresholds: dict[str, object]) -> str:
    evidence: list[str] = []
    if row.get("peer_count", 0) == 0:
        evidence.append("no peer sensors available; using self-consistency features")
    if row["predicted_fault_type"] != "normal":
        evidence.append(f"pattern matches {row['predicted_fault_type']}")
    probability = row.get("model_fault_probability")
    if pd.notna(probability) and probability >= thresholds["fault_classifier_probability"]:
        evidence.append(f"fault classifier confidence {probability:.0%}")
    if _truth(row.get("isolation_anomaly", False)):
        evidence.append("reading is outside the learned healthy-sensor pattern")
    peer_z = row.get("peer_zscore")
    if pd.notna(peer_z) and abs(peer_z) >= thresholds["peer_deviation_z"]:
        evidence.append(f"{peer_z:+.1f}σ from peer median")
    consensus = row.get("event_consensus")
    if pd.notna(consensus) and consensus >= thresholds["event_consensus"]:
        evidence.append(f"{consensus:.0%} of the group also deviates from its daily baseline")
    group_z = row.get("group_baseline_zscore")
    if pd.notna(group_z) and abs(group_z) >= thresholds["environment_group_z"]:
        evidence.append(f"{group_z:+.1f}σ from the group’s same-time-yesterday baseline")
    missing_ratio = row.get("missing_ratio_60m")
    if pd.notna(missing_ratio) and missing_ratio > thresholds["dropout_ratio_60m"]:
        evidence.append(f"{missing_ratio:.0%} missing readings in the last hour")
    repeated = row.get("repeated_count_15m")
    if pd.notna(repeated) and repeated >= thresholds["stuck_repeats_15m"]:
        evidence.append(f"{int(repeated)} repeated readings in 15 minutes")
    peer_agreement = row.get("peer_agreement")
    if isinstance(peer_agreement, str) and peer_agreement:
        evidence.append(peer_agreement)
    if not evidence:
        evidence.append("no strong sensor fault or group event evidence")
    return "; ".join(evidence)


def decide(
    features: pd.DataFrame,
    initial_trust: float | None = None,
    model_bundle: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Predict fault class, environmental abnormality, alarm decision and trust.

    Labels such as `is_fault`, `fault_type`, and `is_event` are never read here.
    They remain in the result only for offline evaluation.
    """
    cfg = config or load_config()
    thresholds = cfg["detection"]
    trust_cfg = cfg["trust"]
    if initial_trust is None:
        initial_trust = float(trust_cfg["initial"])

    required = {
        "run_id", "sensor_id", "timestamp", "value", "peer_count", "peer_deviation",
        "peer_zscore", "event_consensus", "self_zscore_60m", "group_baseline_zscore",
        "diff_max_abs_5m", "slope_60m", "std_15m", "repeated_count_15m", "missing_ratio_60m",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")

    data = features.sort_values(["run_id", "sensor_id", "timestamp"]).reset_index(drop=True).copy()
    peer_deviation = data["peer_deviation"].fillna(False).astype(bool)
    data["spike_evidence"] = data["diff_max_abs_5m"].fillna(0).ge(thresholds["spike_jump"])
    data["stuck_evidence"] = data["repeated_count_15m"].fillna(0).ge(thresholds["stuck_repeats_15m"])
    data["dropout_evidence"] = data["missing_ratio_60m"].fillna(0).ge(thresholds["dropout_ratio_60m"])
    data["slow_degradation_evidence"] = (
        data["slope_60m"].fillna(0).abs().ge(thresholds["slow_degradation_slope_60m"])
        & data["std_15m"].fillna(0).ge(thresholds["slow_degradation_std_15m"])
        & peer_deviation
    )
    data["drift_evidence"] = (
        data["slope_60m"].fillna(0).abs().ge(thresholds["drift_slope_60m"]) & peer_deviation
    )
    data["noise_evidence"] = data["std_15m"].fillna(0).ge(thresholds["noise_std_15m"]) & peer_deviation
    rule_fault_columns = [
        "spike_evidence", "stuck_evidence", "dropout_evidence", "slow_degradation_evidence",
        "drift_evidence", "noise_evidence",
    ]
    data["fault_evidence"] = data[rule_fault_columns].any(axis=1) | peer_deviation

    data["environment_abnormal"] = (
        data["diff_max_abs_5m"].fillna(0).ge(thresholds["spike_jump"])
        | data["group_baseline_zscore"].abs().ge(thresholds["environment_group_z"]).fillna(False)
        | data["event_consensus"].fillna(0).ge(thresholds["event_consensus"])
        | ((data["peer_count"] <= 0)
           & data["self_zscore_60m"].abs().ge(thresholds["environment_self_z"]).fillna(False))
    )

    rule_type = np.select(
        [data["dropout_evidence"], data["stuck_evidence"], data["spike_evidence"],
         data["slow_degradation_evidence"], data["drift_evidence"], data["noise_evidence"]],
        ["dropout", "stuck_at", "spike", "slow_degradation", "offset_drift", "noise_increase"],
        default="sensor_anomaly",
    )
    data["predicted_fault_type"] = np.where(data["fault_evidence"], rule_type, "normal")
    data["model_fault_probability"] = np.nan
    data["isolation_score"] = np.nan
    data["isolation_anomaly"] = False
    data["model_fault_type"] = "normal"

    if model_bundle is not None:
        columns = model_bundle["feature_columns"]
        model_missing = set(columns) - set(data.columns)
        if model_missing:
            raise ValueError(f"Trained model requires missing features: {', '.join(sorted(model_missing))}")
        matrix = data[columns]
        classifier = model_bundle["classifier"]
        probabilities = classifier.predict_proba(matrix)
        classes = list(classifier.classes_)
        normal_index = classes.index("none") if "none" in classes else None
        fault_probability = 1 - probabilities[:, normal_index] if normal_index is not None else np.ones(len(data))
        model_labels = np.asarray(classes, dtype=object)[np.argmax(probabilities, axis=1)]
        model_fault = (
            (fault_probability >= thresholds["fault_classifier_probability"]) & (model_labels != "none")
        )
        data["model_fault_probability"] = fault_probability
        data["model_fault_type"] = model_labels

        imputer = model_bundle["isolation_imputer"]
        isolation_forest = model_bundle["isolation_forest"]
        isolation_scores = isolation_forest.decision_function(imputer.transform(matrix))
        isolation_anomaly = isolation_scores < float(model_bundle["isolation_threshold"])
        isolated_anomaly = (
            isolation_anomaly
            & data["group_baseline_zscore"].abs().ge(thresholds["environment_group_z"]).fillna(False)
            & data["event_consensus"].fillna(0).lt(thresholds["event_consensus"])
        )
        data["isolation_score"] = isolation_scores
        data["isolation_anomaly"] = isolation_anomaly
        data["fault_evidence"] = data["fault_evidence"] | model_fault | isolated_anomaly
        data["predicted_fault_type"] = np.where(
            model_fault, model_labels,
            np.where(isolated_anomaly & data["predicted_fault_type"].eq("normal"),
                     "sensor_anomaly", data["predicted_fault_type"]),
        )

    peer_keys = ["run_id", "group_id", "timestamp"]
    peer_summary = data.groupby(peer_keys, sort=False, dropna=False).agg(
        peer_sensor_ids=("sensor_id", lambda values: tuple(values.astype(str))),
        peer_outlier_ids=("sensor_id", lambda values: tuple(
            values[data.loc[values.index, "peer_deviation"].fillna(False).to_numpy()].astype(str)
        )),
        baseline_deviator_ids=("sensor_id", lambda values: tuple(
            values[data.loc[values.index, "group_sensor_deviation"].fillna(False).to_numpy()].astype(str)
        )),
    ).reset_index()
    data = data.merge(peer_summary, on=peer_keys, how="left", validate="many_to_one")

    def peer_agreement(row: pd.Series) -> str:
        others = [sensor for sensor in row["peer_sensor_ids"] if sensor != str(row["sensor_id"])]
        outliers = [sensor for sensor in row["peer_outlier_ids"] if sensor != str(row["sensor_id"])]
        group_deviators = [sensor for sensor in row["baseline_deviator_ids"] if sensor != str(row["sensor_id"])]
        if _truth(row["peer_deviation"]):
            peers_with_median = [sensor for sensor in others if sensor not in outliers]
            return "Peers agreeing with the group median: " + (", ".join(peers_with_median) or "none")
        if _truth(row["group_sensor_deviation"]) or group_deviators:
            return "Peers sharing the daily-baseline change: " + (", ".join(group_deviators) or "none")
        if others:
            return "Peers near the group median: " + ", ".join(others)
        return ""

    def top_features(row: pd.Series) -> list[str]:
        names = []
        mapping = [
            ("spike_evidence", "diff_max_abs_5m"),
            ("stuck_evidence", "repeated_count_15m"),
            ("dropout_evidence", "missing_ratio_60m"),
            ("slow_degradation_evidence", "slope_60m"),
            ("slow_degradation_evidence", "std_15m"),
            ("drift_evidence", "peer_zscore"),
            ("noise_evidence", "std_15m"),
        ]
        for flag, feature in mapping:
            if _truth(row[flag]) and feature not in names:
                names.append(feature)
        if _truth(row["peer_deviation"]) and "peer_zscore" not in names:
            names.append("peer_zscore")
        if _truth(row["environment_abnormal"]):
            if pd.notna(row["event_consensus"]) and row["event_consensus"] >= thresholds["event_consensus"]:
                names.extend(name for name in ("event_consensus", "group_baseline_zscore") if name not in names)
            elif pd.notna(row["group_baseline_zscore"]) and abs(row["group_baseline_zscore"]) >= thresholds["environment_group_z"]:
                names.append("group_baseline_zscore")
            elif row["peer_count"] <= 0:
                names.append("self_zscore_60m")
        if pd.notna(row["model_fault_probability"]) and row["model_fault_probability"] >= thresholds["fault_classifier_probability"]:
            names.append("fault_classifier_probability")
        if _truth(row["isolation_anomaly"]):
            names.append("isolation_score")
        return names[:5]

    data["peer_agreement"] = data.apply(peer_agreement, axis=1)
    data["top_contributing_features"] = data.apply(top_features, axis=1)

    data["alarm_decision"] = np.select(
        [data["environment_abnormal"] & ~data["fault_evidence"],
         data["environment_abnormal"] & data["fault_evidence"], data["fault_evidence"]],
        ["TRUSTED_ALARM", "SENSOR_FAULT_ALERT", "SUPPRESSED_FALSE_ALARM"],
        default="NO_ALARM",
    )

    trust_values = np.empty(len(data), dtype=float)
    for _, indices in data.groupby(["run_id", "sensor_id"], sort=False).groups.items():
        trust_score = float(np.clip(initial_trust, 0, 100))
        for index in indices:
            if _truth(data.at[index, "fault_evidence"]):
                trust_score = max(0.0, trust_score - float(trust_cfg["fault_decay_per_sample"]))
            else:
                trust_score = min(100.0, trust_score + float(trust_cfg["normal_recovery_per_sample"]))
            trust_values[index] = trust_score
    data["trust_score"] = trust_values
    data["maintenance_recommendation"] = np.select(
        [data["trust_score"] <= trust_cfg["replace_below"],
         data["trust_score"] <= trust_cfg["inspect_below"],
         data["trust_score"] <= trust_cfg["recalibrate_below"],
         data["trust_score"] < trust_cfg["ok_at_or_above"]],
        ["Replace", "Inspect", "Recalibrate", "Monitor"], default="OK",
    )
    data["explanation"] = data.apply(lambda row: _reason(row, thresholds), axis=1)
    return data
