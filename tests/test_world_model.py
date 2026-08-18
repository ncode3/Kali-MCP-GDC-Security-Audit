import pandas as pd

from src.anomaly import ThermalAnomalyDetector
from src.physics import DatacenterState, ThermalPhysicsEngine, simulate_datacenter


def test_simulation_is_deterministic_when_seeded():
    first = simulate_datacenter(steps=8, seed=17)
    second = simulate_datacenter(steps=8, seed=17)

    assert first == second
    assert len(first) == 9
    assert [sample["timestamp"] for sample in first] == list(range(9))


def test_fan_failure_persists_and_reduces_efficiency():
    engine = ThermalPhysicsEngine()
    history = engine.run_simulation(
        steps=5,
        initial_state=DatacenterState(),
        failure_at_step=2,
    )

    assert history[2].fan_failure is False
    assert history[3].fan_failure is True
    assert history[-1].fan_failure is True
    assert history[-1].fan_efficiency < history[2].fan_efficiency


def test_anomaly_detector_reports_critical_samples():
    telemetry = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "temp": [70.0, 85.0, 91.5],
            "datetime_str": ["t0", "t1", "t2"],
        }
    )

    report = ThermalAnomalyDetector().analyze(telemetry, source_file="fixture")

    assert report.risk_assessment == "CRITICAL"
    assert report.anomaly_count == 2
    assert [event.step for event in report.anomalies] == [1, 2]


def test_prediction_requires_enough_samples():
    telemetry = pd.DataFrame({"temp": [65.0] * 9})

    assert ThermalAnomalyDetector().predict_time_to_critical(telemetry) is None
