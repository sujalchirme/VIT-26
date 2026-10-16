"""FastAPI service for the sensor-trust prototype."""

from __future__ import annotations

from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features.extract import extract_features
from models.decision import decide
from models.load import load_model_bundle


app = FastAPI(title="Sensor Trust API", version="0.1.0")
MODEL_BUNDLE = load_model_bundle()


class Reading(BaseModel):
    timestamp: datetime
    sensor_id: str
    group_id: str = "G1"
    value: float | None
    run_id: str = "live"


class PredictionRequest(BaseModel):
    readings: list[Reading] = Field(min_length=1, max_length=30_000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sensor-trust"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    data = pd.DataFrame([reading.model_dump() for reading in request.readings])
    if data["timestamp"].isna().any():
        raise HTTPException(status_code=422, detail="Every reading needs a valid timestamp")
    try:
        result = decide(extract_features(data), model_bundle=MODEL_BUNDLE)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    latest = result.sort_values("timestamp").groupby("sensor_id", as_index=False).tail(1)
    public_fields = ["timestamp", "run_id", "sensor_id", "group_id", "value", "predicted_fault_type",
                     "environment_abnormal", "fault_evidence", "event_consensus", "peer_count",
                     "trust_score", "maintenance_recommendation", "alarm_decision", "explanation"]
    latest = latest[public_fields].copy()
    latest["timestamp"] = latest["timestamp"].astype(str)
    alarms = latest[latest["alarm_decision"] != "NO_ALARM"]
    return {
        "sensors": latest.astype(object).where(pd.notna(latest), None).to_dict(orient="records"),
        "alarms": alarms.astype(object).where(pd.notna(alarms), None).to_dict(orient="records"),
        "model_mode": "learned_plus_rules" if MODEL_BUNDLE else "rules_only",
        "note": "Trust scores in this request start at 100; submit chronological windows to preserve health history.",
    }
