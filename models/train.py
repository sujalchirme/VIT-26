"""Train fault and sensor-anomaly models from the public training split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.extract import extract_features
from models.config import load_config
from models.schema import FEATURE_COLUMNS

MODEL_VERSION = 1


def _read_features(path: Path) -> pd.DataFrame:
    readings = pd.read_csv(path, parse_dates=["timestamp"])
    features = extract_features(readings)
    return features


def _balanced_sample(features: pd.DataFrame, target: str, max_per_class: int,
                     seed: int) -> pd.DataFrame:
    groups = []
    for _, group in features.groupby(target, sort=True, dropna=False):
        groups.append(group.sample(n=min(len(group), max_per_class), random_state=seed))
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def train(train_path: Path, output_path: Path, validation_path: Path | None = None,
          max_per_class: int = 25_000, seed: int = 26) -> dict[str, object]:
    train_features = _read_features(train_path)
    train_features["fault_type"] = train_features["fault_type"].fillna("none")
    sampled = _balanced_sample(train_features, "fault_type", max_per_class, seed)
    x_train = sampled[FEATURE_COLUMNS]
    y_train = sampled["fault_type"].astype(str)

    classifier = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("random_forest", RandomForestClassifier(
            n_estimators=160, max_depth=14, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )),
    ])
    classifier.fit(x_train, y_train)

    clean = train_features[
        ~train_features["is_fault"].fillna(False).astype(bool)
        & ~train_features["is_event"].fillna(False).astype(bool)
    ]
    if len(clean) < 100:
        raise ValueError("Need at least 100 clean, non-event samples to fit Isolation Forest")
    if len(clean) > 100_000:
        clean = clean.sample(n=100_000, random_state=seed)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_clean = imputer.fit_transform(clean[FEATURE_COLUMNS])
    isolation_forest = IsolationForest(
        n_estimators=160, contamination="auto", random_state=seed, n_jobs=-1,
    )
    isolation_forest.fit(x_clean)
    normal_quantile = float(load_config()["isolation_forest"]["training_normal_quantile"])
    isolation_threshold = float(np.quantile(isolation_forest.decision_function(x_clean), normal_quantile))

    bundle = {
        "version": MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "classifier": classifier,
        "isolation_imputer": imputer,
        "isolation_forest": isolation_forest,
        "isolation_threshold": isolation_threshold,
        "isolation_training_quantile": normal_quantile,
        "classes": classifier.named_steps["random_forest"].classes_.tolist(),
        "training_rows": int(len(sampled)),
        "clean_rows_for_isolation_forest": int(len(clean)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    report: dict[str, object] = {
        "model_path": str(output_path),
        "training_rows": int(len(sampled)),
        "classes": bundle["classes"],
        "isolation_threshold": isolation_threshold,
    }
    if validation_path:
        validation = _read_features(validation_path)
        validation["fault_type"] = validation["fault_type"].fillna("none")
        prediction = classifier.predict(validation[FEATURE_COLUMNS])
        report["validation_macro_f1"] = float(f1_score(
            validation["fault_type"].astype(str), prediction, average="macro", zero_division=0,
        ))
        report["validation_classification_report"] = classification_report(
            validation["fault_type"].astype(str), prediction, output_dict=True, zero_division=0,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path, nargs="?", default=Path("data/generated/train.csv"))
    parser.add_argument("--validation", type=Path, default=Path("data/generated/validation.csv"))
    parser.add_argument("--output", type=Path, default=Path("models/fault_models.joblib"))
    parser.add_argument("--max-per-class", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()
    validation = args.validation if args.validation.exists() else None
    report = train(args.train, args.output, validation, args.max_per_class, args.seed)
    print(pd.Series(report, dtype=object).to_json(indent=2))


if __name__ == "__main__":
    main()
