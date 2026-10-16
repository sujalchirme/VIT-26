"""Evaluate the baseline on a held-out, labelled split; this script never fits models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extract import extract_features
from models.decision import decide
from models.load import load_model_bundle


FAULT_LABELS = ["none", "offset_drift", "gain_error", "spike", "stuck_at",
                "noise_increase", "dropout", "slow_degradation"]


def evaluate(path: Path, model_bundle: dict[str, object] | None = None) -> dict[str, object]:
    readings = pd.read_csv(path, parse_dates=["timestamp"])
    features = extract_features(readings)
    result = decide(features, model_bundle=model_bundle)
    y_true_fault = result["is_fault"].fillna(False).astype(bool)
    y_pred_fault = result["fault_evidence"].fillna(False).astype(bool)
    y_true_type = result["fault_type"].fillna("none").astype(str)
    y_pred_type = result["predicted_fault_type"].replace("normal", "none").astype(str)
    y_true_event = result["is_event"].fillna(False).astype(bool)

    naive_alarm = (result["self_zscore_60m"].abs().ge(2.5).fillna(False)
                   | result["diff_max_abs_5m"].fillna(0).ge(4.0))
    accepted_alarm = result["alarm_decision"].eq("TRUSTED_ALARM")
    naive_false = int((naive_alarm & ~y_true_event).sum())
    system_false = int((accepted_alarm & ~y_true_event).sum())
    false_alarm_reduction = (1.0 - system_false / naive_false) if naive_false else None

    event_groups = result.loc[y_true_event].groupby(
        ["run_id", "group_id", "event_id"], sort=False, dropna=False
    )["environment_abnormal"].any()
    event_sensitivity = float(event_groups.mean()) if len(event_groups) else None
    fault_mask = y_true_fault
    fault_type_f1 = f1_score(y_true_type[fault_mask], y_pred_type[fault_mask],
                             labels=FAULT_LABELS[1:], average="macro", zero_division=0)
    cm = confusion_matrix(y_true_type, y_pred_type, labels=FAULT_LABELS)
    fault_onset_delays = []
    for _, sensor in result.groupby(["run_id", "sensor_id"], sort=False):
        faulty = sensor["is_fault"].fillna(False).astype(bool).to_numpy()
        detected = sensor["fault_evidence"].fillna(False).astype(bool).to_numpy()
        onsets = np.flatnonzero(faulty & ~np.r_[False, faulty[:-1]])
        for onset in onsets:
            end = onset + 1
            while end < len(faulty) and faulty[end]:
                end += 1
            future = np.flatnonzero(detected[onset:end])
            if len(future):
                fault_onset_delays.append(int(future[0]))

    healthy = ~y_true_fault
    faulty_trust = result.loc[y_true_fault, "trust_score"]
    healthy_trust = result.loc[healthy, "trust_score"]
    return {
        "split": str(path),
        "model_mode": "learned_plus_rules" if model_bundle else "rules_only",
        "rows": int(len(result)),
        "fault_detection": {
            "precision": float(precision_score(y_true_fault, y_pred_fault, zero_division=0)),
            "recall": float(recall_score(y_true_fault, y_pred_fault, zero_division=0)),
            "f1": float(f1_score(y_true_fault, y_pred_fault, zero_division=0)),
            "fault_type_macro_f1": float(fault_type_f1),
            "fault_type_labels": FAULT_LABELS,
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
            "classification_report": classification_report(
                y_true_type, y_pred_type, labels=FAULT_LABELS, output_dict=True, zero_division=0
            ),
        },
        "false_alarm_reduction_vs_naive": {
            "naive_non_event_alarms": naive_false,
            "trusted_non_event_alarms": system_false,
            "reduction_fraction": false_alarm_reduction,
        },
        "event_sensitivity": event_sensitivity,
        "genuine_events": int(len(event_groups)),
        "mean_fault_detection_delay_samples": (float(np.mean(fault_onset_delays)) if fault_onset_delays else None),
        "trust_score_mean": {
            "faulty_samples": float(faulty_trust.mean()) if len(faulty_trust) else None,
            "healthy_samples": float(healthy_trust.mean()) if len(healthy_trust) else None,
        },
        "interpretation": "Descriptive simulation metrics; no models are fitted on this split.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, nargs="?", default=Path("data/generated/hidden.csv"))
    parser.add_argument("--output", type=Path, default=Path("eval/report.json"))
    parser.add_argument("--model", type=Path, default=None, help="Optional trained model bundle")
    args = parser.parse_args()
    model_bundle = load_model_bundle(args.model) if args.model else load_model_bundle()
    report = evaluate(args.split, model_bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote evaluation report to {args.output}")


if __name__ == "__main__":
    main()
