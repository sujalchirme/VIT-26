"""Streamlit replay dashboard for the sensor-trust prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.generate import Scenario, generate_run  # noqa: E402
from features.extract import extract_features  # noqa: E402
from models.decision import decide  # noqa: E402
from models.load import load_model_bundle  # noqa: E402

MODEL_BUNDLE = load_model_bundle()


FAULT_OPTIONS = {
    "No injected fault": None,
    "Inject offset drift": "offset_drift",
    "Inject gain error": "gain_error",
    "Inject spike": "spike",
    "Inject stuck-at": "stuck_at",
    "Inject noise increase": "noise_increase",
    "Inject dropout": "dropout",
}


@st.cache_data(show_spinner="Generating demo stream and calculating sensor health…")
def load_demo(fault_type: str | None) -> pd.DataFrame:
    faults = (fault_type,) if fault_type else ()
    scenario = Scenario("dashboard_demo", faults, ("propagating", "pulse"))
    readings = generate_run("demo-001", scenario, seed=2601, days=10)
    return decide(extract_features(readings), model_bundle=MODEL_BUNDLE)


def main() -> None:
    st.set_page_config(page_title="Sensor Trust Console", page_icon="📡", layout="wide")
    st.title("Sensor Trust Console")
    st.caption("Separate real environmental events from unreliable sensor readings.")
    st.caption("Detection mode: " + ("trained models + transparent rules" if MODEL_BUNDLE else "transparent rules (train models to enable learned scores)"))

    with st.sidebar:
        st.header("Demo controls")
        selected_injection = st.radio("Fault injection", list(FAULT_OPTIONS), index=0)
        st.caption("Changing this control creates a repeatable synthetic run with the selected fault pattern.")

    result = load_demo(FAULT_OPTIONS[selected_injection])
    timeline = result["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    cursor = st.slider("Replay time", 0, len(timeline) - 1, len(timeline) - 1, step=15)
    replay_time = timeline.iloc[cursor]
    visible = result[result["timestamp"] <= replay_time]
    latest = visible.sort_values("timestamp").groupby("sensor_id", as_index=False).tail(1)

    alert_count = int((latest["alarm_decision"] != "NO_ALARM").sum())
    fault_count = int(latest["fault_evidence"].sum())
    abnormal_count = int(latest["environment_abnormal"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Sensors reporting", len(latest))
    col2.metric("Sensors with abnormal readings", abnormal_count)
    col3.metric("Current alerts", alert_count, help=f"{fault_count} sensor fault indication(s)")

    sensor = st.selectbox("Sensor detail", sorted(latest["sensor_id"].unique()))
    current = latest[latest["sensor_id"] == sensor].iloc[0]
    left, right = st.columns([2, 1])
    with left:
        history = visible[(visible["sensor_id"] == sensor) &
                          (visible["timestamp"] >= replay_time - pd.Timedelta(hours=6))]
        peers = visible[(visible["group_id"] == current["group_id"]) &
                        (visible["timestamp"].isin(history["timestamp"]))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["timestamp"], y=history["value"],
                                 name=sensor, mode="lines", line={"color": "#57b8ff", "width": 2}))
        median = peers.groupby("timestamp")["value"].median().reset_index()
        fig.add_trace(go.Scatter(x=median["timestamp"], y=median["value"],
                                 name="Group median", mode="lines", line={"color": "#f5a742", "dash": "dot"}))
        flagged = history[history["alarm_decision"] != "NO_ALARM"]
        if not flagged.empty:
            fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["value"], name="Flagged",
                                     mode="markers", marker={"color": "#ff5f67", "size": 8}))
        fig.update_layout(title=f"{sensor} vs. group median", xaxis_title="Time (UTC)",
                          yaxis_title="Measurement", height=390, legend={"orientation": "h"})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=float(current["trust_score"]),
            title={"text": f"{sensor} trust"}, number={"suffix": "/100"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#30c48d" if current["trust_score"] >= 80 else "#f5a742"},
                   "steps": [{"range": [0, 60], "color": "#4c242a"},
                             {"range": [60, 80], "color": "#4c4024"},
                             {"range": [80, 100], "color": "#1d4238"}]},
        ))
        gauge.update_layout(height=235, margin={"t": 60, "b": 15, "l": 25, "r": 25})
        st.plotly_chart(gauge, use_container_width=True)
        st.subheader(str(current["alarm_decision"]).replace("_", " ").title())
        st.write(current["explanation"])
        st.write(f"**Health action:** {current['maintenance_recommendation']}")
        if current["peer_count"] == 0:
            st.info("No peer sensors are available. The decision uses this sensor's history only.")

    st.subheader("Current sensor status")
    status_columns = ["sensor_id", "group_id", "value", "predicted_fault_type", "trust_score",
                      "maintenance_recommendation", "alarm_decision", "explanation"]
    st.dataframe(latest[status_columns].sort_values("sensor_id"), use_container_width=True, hide_index=True)

    st.subheader("Recent alarm feed")
    feed = visible[visible["alarm_decision"] != "NO_ALARM"].sort_values("timestamp", ascending=False).head(30)
    if feed.empty:
        st.info("No alarms at the selected replay time.")
    else:
        st.dataframe(feed[["timestamp", "sensor_id", "alarm_decision", "predicted_fault_type",
                           "trust_score", "explanation"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
