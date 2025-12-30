"""
AARI World Model: Infrastructure Digital Twin
"""
from .physics import ThermalPhysicsEngine, ThermalConstants, DatacenterState, simulate_datacenter
from .generator import run_simulation, generate_batch_telemetry
from .anomaly import ThermalAnomalyDetector, detect_thermal_anomaly, analyze_telemetry_file

__version__ = "0.1.0"
