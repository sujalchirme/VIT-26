# Sensor Trust, Drift and False-Alarm Detection

This project separates two questions for every sensor reading:

1. Is the measured environmental condition abnormal?
2. Is the sensor producing that reading behaving reliably?

A shared change across related sensors can be a real event. A deviation isolated to one sensor can be a sensor fault. The prototype will combine per-sensor behavior with evidence from peer sensors, then show the reason for each alarm decision.

## Build plan

The implementation follows the project brief in this order:

1. Generate reproducible labelled multi-sensor data.
2. Extract rolling sensor and peer-comparison features (`features/extract.py`).
3. Add anomaly and fault detectors (`models/decision.py` combines interpretable rules with optional trained models).
4. Fuse event consensus and sensor-fault evidence.
5. Track a smoothed sensor trust score and maintenance status.
6. Explain decisions and compare against baselines.
7. Expose inference through FastAPI and a Streamlit/Plotly demo.

## Data generator

`data_gen/generate.py` produces eight sensors in two groups, sampled once per minute. Each run includes a daily cycle, slow trend, sensor offsets, noise, group-level events, and labelled sensor faults. Fault and event labels are independent so a real event can overlap a sensor fault.

The default output has separate train, validation, and hidden-style CSV files. The hidden-style runs use fault/event combinations not used by the public scenarios. Keep these files out of model fitting and threshold selection; use them only for final evaluation. The generator is deterministic for a fixed seed.

```powershell
python -m pip install numpy pandas
python data_gen/generate.py
```

Generated files go under `data/generated/` and are not required to be committed. Options include `--days`, `--public-runs`, `--validation-runs`, `--hidden-runs`, `--seed`, and `--output-dir`.

`features.extract_features(readings)` adds 5/15/60-minute statistics, trend and jump features, repeated-value and missing-data signals, robust peer residuals, and group event-consensus evidence. It uses timestamp windows and keeps separate runs isolated.

## Run the prototype

Install the dependencies, then start the dashboard:

```powershell
python -m pip install -r requirements.txt
streamlit run app/dashboard.py
```

The dashboard includes a replay slider, sensor-to-group comparison, trust gauge, explanation, maintenance status, and selectable fault-injection scenarios. The inference API can be started separately:

```powershell
uvicorn api.main:app --reload
```

It exposes `GET /health` and `POST /predict`. Before training, it uses explicit rules. After training, it combines the models with the same rule traces and reports the active mode.

## Train the learned detectors

Generate the public training and validation files, then fit the Random Forest fault-type classifier and Isolation Forest healthy-sensor model using only the training file. This compact command reproduces the checked-in demo report; omit `--days` for the brief's 30-day default:

```powershell
python data_gen/generate.py --days 10 --public-runs 4 --validation-runs 2 --hidden-runs 2
python models/train.py data/generated/train.csv --validation data/generated/validation.csv
python eval/evaluate.py data/generated/hidden.csv --output eval/prototype_report.json
```

The bundle is saved as `models/fault_models.joblib`. The API, dashboard, and evaluator load it automatically when present. Training prints validation metrics; it does not read the hidden-style file.

## Evaluation

After generating the splits and training the models, run the evaluator against the hidden-style split. It scores predictions only and does not fit or tune on that file:

```powershell
python eval/evaluate.py data/generated/hidden.csv
```

The [prototype report](eval/prototype_report.json) includes fault precision/recall/F1 and type confusion, trusted non-event alarms versus a naive threshold, event sensitivity, fault detection delay, and trust-score separation. In this compact seeded run it detected 8/8 genuine event windows, reduced trusted non-event alarms by 71.5% against the naive threshold, and averaged 2.4 minutes to fault detection. Faulty versus healthy trust averaged 18.0 versus 94.8. These are results on the simulator's held-out combinations, not field performance.

### Data dictionary

| Column | Meaning |
| --- | --- |
| `timestamp` | UTC timestamp at one-minute resolution |
| `run_id` | Independent simulated run; splits are separated by run |
| `sensor_id` | Sensor identifier (`S01` through `S08`) |
| `group_id` | Peer group (`G1` or `G2`) |
| `value` | Simulated measurement; dropout samples are NaN |
| `is_fault` | Whether the sensor is faulty at this timestamp |
| `fault_type` | Fault label, or `none` for healthy readings |
| `is_event` | Whether a genuine environmental event affects this sensor |
| `event_id` | Shared event identifier, or empty outside an event |

## Current scope and limitations

The generator is a simulation for a laptop-friendly prototype, not validated plant data. Event agreement is evidence, not proof: common-mode faults can affect several sensors, and genuine events may affect sensors unevenly. Later fusion logic should expose uncertainty and support a single-sensor fallback instead of treating peer consensus as ground truth.
