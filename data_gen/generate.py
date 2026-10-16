"""Generate labelled multi-sensor time-series data for the sensor-trust demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLES_PER_DAY = 24 * 60
SENSORS_PER_GROUP = 4


@dataclass(frozen=True)
class Scenario:
    """Fault/event recipe used to keep the evaluation split distinct."""

    name: str
    fault_types: tuple[str, ...]
    event_shapes: tuple[str, ...]
    simultaneous_faults: bool = False


PUBLIC_SCENARIOS = (
    Scenario("public_basic", ("offset_drift", "spike", "stuck_at", "dropout"), ("step", "pulse")),
    Scenario("public_noise_gain_degradation", ("noise_increase", "gain_error", "slow_degradation"), ("ramp", "propagating")),
)

HIDDEN_SCENARIOS = (
    Scenario("hidden_combined", ("offset_drift", "noise_increase", "dropout", "slow_degradation"), ("step", "propagating"), True),
    Scenario("hidden_stuck_event", ("stuck_at", "gain_error", "spike"), ("ramp", "pulse"), True),
)


def generate_run(run_id: str, scenario: Scenario, seed: int, days: int = 30) -> pd.DataFrame:
    """Return long-form readings and independent event/fault ground-truth labels."""
    rng = np.random.default_rng(seed)
    n = days * SAMPLES_PER_DAY
    timestamps = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    sensor_ids = [f"S{i:02d}" for i in range(1, 9)]
    rows: list[pd.DataFrame] = []
    base = 20.0 + 3.0 * np.sin(2 * np.pi * np.arange(n) / SAMPLES_PER_DAY) + 0.00003 * np.arange(n)

    for group_number in range(2):
        group_id = f"G{group_number + 1}"
        event_shapes = list(scenario.event_shapes)
        for local_sensor in range(SENSORS_PER_GROUP):
            sensor_num = group_number * SENSORS_PER_GROUP + local_sensor + 1
            sensor_id = sensor_ids[sensor_num - 1]
            signal = base + group_number * 8 + 0.6 * np.sin(2 * np.pi * np.arange(n) / (SAMPLES_PER_DAY * 7))
            event_mask = np.zeros(n, dtype=bool)
            event_ids = np.full(n, "", dtype=object)
            # Deterministic shared event windows; propagation is represented as lag.
            for event_number, shape in enumerate(event_shapes, start=1):
                width = min(n // 8, 240)
                start = SAMPLES_PER_DAY + (event_number * 5 * SAMPLES_PER_DAY) + group_number * 137
                start = min(start, n - width - 1)
                lag = local_sensor * 3 if shape == "propagating" else 0
                actual_start = min(start + lag, n - width - 1)
                length = max(1, width - lag)
                amplitude = 3.5 if (event_number + group_number) % 2 else -3.5
                idx = np.arange(length)
                if shape == "step":
                    profile = np.full(length, amplitude)
                elif shape == "ramp":
                    profile = np.linspace(0.0, amplitude, length)
                else:
                    profile = amplitude * np.sin(np.linspace(0.0, np.pi, length))
                signal[actual_start:actual_start + length] += profile
                event_mask[actual_start:actual_start + length] = True
                event_ids[actual_start:actual_start + length] = f"event-{event_number + group_number * 10}"

            sensor_offset = (local_sensor - 1.5) * 0.12
            values = signal + sensor_offset + rng.normal(0, 0.18, n)
            is_fault = np.zeros(n, dtype=bool)
            fault_type = np.full(n, "none", dtype=object)
            eligible_faults = list(scenario.fault_types)
            count = 2 if scenario.simultaneous_faults else 1
            chosen_faults = rng.choice(eligible_faults, size=min(count, len(eligible_faults)), replace=False)
            for fault_idx, kind in enumerate(chosen_faults):
                start = min(int(n * (0.25 + 0.04 * ((sensor_num + fault_idx) % 10))), n - 20)
                width = min(180 + sensor_num * 7, max(20, n // 100), n - start)
                end = start + width
                if kind == "offset_drift":
                    values[start:end] += np.linspace(0, rng.uniform(2.0, 5.0), width)
                elif kind == "gain_error":
                    values[start:end] = signal[start:end] * rng.uniform(1.15, 1.4) + (values[start:end] - signal[start:end])
                elif kind == "spike":
                    points = rng.choice(np.arange(start, end), size=min(6, width), replace=False)
                    values[points] += rng.choice([-1.0, 1.0], len(points)) * rng.uniform(6, 10)
                    is_fault[points] = True
                    fault_type[points] = kind
                    continue
                elif kind == "stuck_at":
                    values[start:end] = values[start]
                elif kind == "noise_increase":
                    values[start:end] += rng.normal(0, 0.9, width)
                elif kind == "dropout":
                    values[start:end:5] = np.nan
                elif kind == "slow_degradation":
                    values[start:end] += np.linspace(0, 3.0, width) + rng.normal(0, 0.45, width)
                is_fault[start:end] = True
                fault_type[start:end] = kind

            rows.append(pd.DataFrame({
                "timestamp": timestamps,
                "run_id": run_id,
                "sensor_id": sensor_id,
                "group_id": group_id,
                "value": values.astype(np.float32),
                "is_fault": is_fault,
                "fault_type": fault_type,
                "is_event": event_mask,
                "event_id": event_ids,
            }))

    return pd.concat(rows, ignore_index=True)


def generate_split(split: str, scenarios: tuple[Scenario, ...], runs: int, seed: int, days: int) -> pd.DataFrame:
    frames = []
    for run_number in range(runs):
        scenario = scenarios[run_number % len(scenarios)]
        frames.append(generate_run(f"{split}-{run_number + 1:03d}", scenario,
                                   seed + run_number * 101, days))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--public-runs", type=int, default=4)
    parser.add_argument("--validation-runs", type=int, default=2)
    parser.add_argument("--hidden-runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()
    if args.days < 2 or args.public_runs < 1 or args.validation_runs < 1 or args.hidden_runs < 1:
        parser.error("days must be >= 2 and all run counts must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = generate_split("train", PUBLIC_SCENARIOS, args.public_runs, args.seed, args.days)
    validation = generate_split("validation", PUBLIC_SCENARIOS, args.validation_runs,
                                args.seed + 10_000, args.days)
    hidden = generate_split("hidden", HIDDEN_SCENARIOS, args.hidden_runs, args.seed + 100_000, args.days)
    train.to_csv(args.output_dir / "train.csv", index=False)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    hidden.to_csv(args.output_dir / "hidden.csv", index=False)
    print(f"Wrote {len(train):,} train, {len(validation):,} validation, and "
          f"{len(hidden):,} hidden-style rows to {args.output_dir}")


if __name__ == "__main__":
    main()
