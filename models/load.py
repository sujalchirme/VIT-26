"""Load the optional locally trained model artifact."""

from __future__ import annotations

from pathlib import Path

import joblib


DEFAULT_MODEL_PATH = Path(__file__).resolve().with_name("fault_models.joblib")


def load_model_bundle(path: Path = DEFAULT_MODEL_PATH) -> dict[str, object] | None:
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if bundle.get("version") != 1:
        raise ValueError(f"Unsupported sensor model bundle version at {path}")
    return bundle
