"""Explainable baseline for separating process events from sensor faults."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _truth(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _reason(row: pd.Series) -> str:
    evidence: list[str] = []
    if row.get("peer_count", 0) == 0:
        evidence.append("no peer sensors available; using self-consistency features")
    if row["predicted_fault_type"] != "normal":
        evidence.append(f"pattern matches {row['predicted_fault_type']}")
    if pd.notna(row.get("model_fault_probability")) and row["model_fault_probability"] >= 0.65:
        evidence.append(f"fault classifier confidence {row['model_fault_probability']:.0%}")
    if _truth(row.get("isolation_anomaly", False)):
        evidence.append("reading is outside the learned healthy-sensor pattern")
    if pd.notna(row.get("peer_zscore")) and abs(row["peer_zscore"]) >= 2.5:
        evidence.append(f"{row['peer_zscore']:+.1f}σ from peer median")
    if pd.notna(row.get("event_consensus")) and row["event_consensus"] >= 0.5:
        evidence.append(f"{row['event_consensus']:.0%} of the group also deviates from baseline")
    if pd.notna(row.get("group_baseline_zscore")) and abs(row["group_baseline_zscore"]) >= 4.0:
        evidence.append(f"group-history deviation {row['group_baseline_zscore']:+.1f}σ")
    if pd.notna(row.get("missing_ratio_60m")) and row["missing_ratio_60m"] > 0.05:
        evidence.append(f"{row['missing_ratio_60m']:.0%} missing readings in the last hour")
    if pd.notna(row.get("repeated_count_15m")) and row["repeated_count_15m"] >= 5:
        evidence.append(f"{int(row['repeated_count_15m'])} repeated readings in 15 minutes")
    if not evidence:
        evidence.append("no strong sensor fault or group event evidence")
    return "; ".join(evidence)


def decide(features: pd.DataFrame, initial_trust: float = 100.0,
           model_bundle: dict[str, object] | None = None) -> pd.DataFrame:
    """Add fault, environment, alarm, trust, and explanation outputs.

    This transparent baseline is designed to make the prototype usable before
    learned models are introduced. Thresholds are explicit and can be tuned on
    the public validation split only.
    """
    required = {"run_id", "sensor_id", "timestamp", "value", "event_consensus", "peer_zscore",
                "self_zscore_60m", "group_baseline_zscore", "diff_max_abs_5m",
                "repeated_count_15m", "missing_ratio_60m"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(sorted(missing))}")

    data = features.sort_values(["run_id", "sensor_id", "timestamp"]).reset_index(drop=True).copy()
    data["spike_evidence"] = data["diff_max_abs_5m"].fillna(0).ge(4.0)
    data["stuck_evidence"] = data["repeated_count_15m"].fillna(0).ge(5)
    data["dropout_evidence"] = data["missing_ratio_60m"].fillna(0).ge(0.05)
    data["drift_evidence"] = data["slope_60m"].fillna(0).abs().ge(0.025) & data["peer_deviation"].fillna(False)
    data["noise_evidence"] = data["std_15m"].fillna(0).ge(0.9) & data["peer_deviation"].fillna(False)
    data["fault_evidence"] = (
        data[["spike_evidence", "stuck_evidence", "dropout_evidence", "drift_evidence", "noise_evidence"]].any(axis=1)
        | data["peer_deviation"].fillna(False)
    )
    data["environment_abnormal"] = (
        data["diff_max_abs_5m"].fillna(0).ge(4.0)
        | data["group_baseline_zscore"].abs().ge(4.0).fillna(False)
        | data["event_consensus"].fillna(0).ge(0.5)
        | ((data["peer_count"] <= 0) & data["self_zscore_60m"].abs().ge(2.5).fillna(False))
    )

    data["predicted_fault_type"] = np.select(
        [data["dropout_evidence"], data["stuck_evidence"], data["spike_evidence"],
         data["drift_evidence"], data["noise_evidence"]],
        ["dropout", "stuck_at", "spike", "offset_drift", "noise_increase"],
        default="sensor_anomaly",
    )
    data.loc[~data["fault_evidence"], "predicted_fault_type"] = "normal"
    data["model_fault_probability"] = np.nan
    data["isolation_score"] = np.nan
    data["isolation_anomaly"] = False
    data["model_fault_type"] = "normal"
    if model_bundle is not None:
        columns = model_bundle["feature_columns"]
        matrix = data[columns]
        classifier = model_bundle["classifier"]
        probabilities = classifier.predict_proba(matrix)
        classes = list(classifier.classes_)
        normal_idx = classes.index("none") if "none" in classes else None
        fault_probability = 1.0 - probabilities[:, normal_idx] if normal_idx is not None else np.ones(len(data))
        class_index = np.argmax(probabilities, axis=1)
        model_labels = np.asarray(classes, dtype=object)[class_index]
        model_fault = (fault_probability >= 0.65) & (model_labels != "none")
        data["model_fault_probability"] = fault_probability
        data["model_fault_type"] = model_labels

        imputer = model_bundle["isolation_imputer"]
        isolation_forest = model_bundle["isolation_forest"]
        isolation_scores = isolation_forest.decision_function(imputer.transform(matrix))
        isolation_anomaly = isolation_scores < float(model_bundle["isolation_threshold"])
        isolated_anomaly = (
            isolation_anomaly
            & data["group_baseline_zscore"].abs().ge(4.0).fillna(False)
            & data["event_consensus"].fillna(0).lt(0.5)
        )
        data["isolation_score"] = isolation_scores
        data["isolation_anomaly"] = isolation_anomaly
        data["fault_evidence"] = data["fault_evidence"] | model_fault | isolated_anomaly
        rule_label = data["predicted_fault_type"].to_numpy(copy=True)
        data["predicted_fault_type"] = np.where(
            model_fault, model_labels,
            np.where(isolated_anomaly & (rule_label == "normal"), "sensor_anomaly", rule_label),
        )
    data["alarm_decision"] = np.select(
        [data["environment_abnormal"] & ~data["fault_evidence"],
         data["environment_abnormal"] & data["fault_evidence"],
         data["fault_evidence"]],
        ["TRUSTED_ALARM", "SENSOR_FAULT_ALERT", "SUPPRESSED_FALSE_ALARM"],
        default="NO_ALARM",
    )

    trust_values = np.empty(len(data), dtype=float)
    for _, indices in data.groupby(["run_id", "sensor_id"], sort=False).groups.items():
        trust = float(np.clip(initial_trust, 0, 100))
        for idx in indices:
            if _truth(data.at[idx, "fault_evidence"]):
                trust = max(0.0, trust - 2.0)
            else:
                trust = min(100.0, trust + 0.15)
            trust_values[data.index.get_loc(idx)] = trust
    data["trust_score"] = trust_values
    data["maintenance_recommendation"] = pd.cut(
        data["trust_score"], bins=[-0.01, 40, 60, 80, 100],
        labels=["Replace", "Inspect", "Recalibrate", "Monitor"], include_lowest=True,
    ).astype(str)
    data.loc[data["trust_score"] >= 95, "maintenance_recommendation"] = "OK"
    data["explanation"] = data.apply(_reason, axis=1)
    return data
