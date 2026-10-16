# Sensor Trust, Drift and False-Alarm Detection

This project separates two questions for every sensor reading:

1. Is the measured environmental condition abnormal?
2. Is the sensor producing that reading behaving reliably?

A shared change across related sensors can be a real event. A deviation isolated to one sensor can be a sensor fault. The prototype combines per-sensor behavior with evidence from peer sensors, then shows the reason for each alarm decision.

## Included components

The prototype includes:

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

`features.extract_features(readings)` adds 5/15/60-minute statistics, trend and jump features, repeated-value and missing-data signals, robust peer residuals, and group event-consensus evidence. It compares the current group reading with the nearest same-time reading from the previous day to account for daily seasonality. It uses timestamp windows and keeps separate runs isolated.

## Set up on another computer

Install Python 3.10 or newer and Git, then obtain the project using GitHub Desktop (clone the repository and check out the `sujal` branch) or Git:

```bash
git clone <repository-url>
cd VIT-26
git switch sujal
```

Changes on this computer must be pushed to the shared repository before another computer can clone or pull them. Create a virtual environment and install dependencies from the project folder.

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The virtual environment is local to each computer and is ignored by Git. The dashboard generates its own repeatable demo stream, so it can run without the generated CSV files or a trained model. Start it from the project folder:

```bash
python -m streamlit run app/dashboard.py
```

Open the local URL printed in the terminal (usually `http://localhost:8501`). Keep the terminal open while using the dashboard; press Ctrl+C to stop it. The dashboard includes a replay slider, sensor-to-group comparison, trust gauge, explanation, maintenance status, and selectable fault-injection scenarios.

For a Windows presentation computer, double-click `setup_windows.bat` once while connected to the internet. After setup completes, double-click `run_demo.bat` each time you want to launch the dashboard. A terminal window opens and Streamlit opens the browser; keep the terminal open during the presentation and press Ctrl+C in it to stop the app.

To enable learned model scores on that computer, generate its data and train its own model:

```bash
python data_gen/generate.py --days 10 --public-runs 4 --validation-runs 2 --hidden-runs 2
python models/train.py data/generated/train.csv --validation data/generated/validation.csv
```

The inference API is optional and can be started in a second terminal after activating the same virtual environment:

```bash
python -m uvicorn api.main:app --reload
```

It exposes `GET /health` and `POST /predict`. Before training, it uses explicit rules. After training, it combines the models with the same rule traces and reports the active mode.

For a remote demo, run the API with `uvicorn api.main:app --host 0.0.0.0 --port 8000` and expose it with `ngrok http 8000`. To share the dashboard instead, start Streamlit with `streamlit run app/dashboard.py --server.address 0.0.0.0` and expose port `8501` with ngrok.

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

The [prototype report](eval/prototype_report.json) includes fault precision/recall/F1 and type confusion, trusted non-event alarms versus a naive threshold, event sensitivity, fault detection delay, and trust-score separation. In this compact seeded run it detected 8/8 genuine event windows, reduced trusted non-event alarms by 71.8% against the naive threshold, and averaged 1.96 minutes to fault detection. Faulty versus healthy trust averaged 17.0 versus 94.8. The fault-type macro F1 was 0.69; offset drift remains a weak class, and the spike class has few hidden examples. These are results on the simulator's held-out combinations, not field performance.

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

The prototype now includes the generator, rolling and peer features, rule/model fusion, trust tracking, explanations, maintenance advice, FastAPI endpoints, a Streamlit replay demo, and held-out metrics. Detector, alarm, trust-decay, and maintenance cutoffs are documented in `config.yaml`.

The data is simulated, not validated plant data. The hidden-style evaluation contains only a small number of generated runs and is not evidence of field performance. Event consensus is evidence rather than proof: common-mode faults can affect several sensors, and genuine events may affect sensors unevenly. Trust scores in the API restart at the configured initial value for each request; a production deployment would need persistent per-sensor state. MQTT/OPC-UA ingestion, technician feedback, edge deployment, and real-site calibration are outside this prototype.
