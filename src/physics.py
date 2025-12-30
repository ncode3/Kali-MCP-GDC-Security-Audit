#!/usr/bin/env python3
"""
AARI World Model: Datacenter Physics Engine
Models thermal dynamics using Newton's Law of Cooling.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThermalConstants:
    k_cooling: float = 0.15
    k_heating: float = 0.08
    thermal_mass: float = 50.0
    ambient_base: float = 22.0
    ambient_amplitude: float = 5.0
    target_temp: float = 65.0
    critical_temp: float = 85.0
    warning_temp: float = 75.0
    fan_efficiency_nominal: float = 1.0
    fan_degradation_rate: float = 0.02

@dataclass
class DatacenterState:
    timestamp: int = 0
    temperature: float = 65.0
    load: float = 50.0
    fan_efficiency: float = 1.0
    power_draw: float = 0.0
    cooling_active: bool = True
    fan_failure: bool = False
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'temp': round(self.temperature, 2),
            'load': round(self.load, 2),
            'fan_efficiency': round(self.fan_efficiency, 3),
            'power_kw': round(self.power_draw, 2),
            'cooling_active': self.cooling_active,
            'fan_failure': self.fan_failure
        }

class ThermalPhysicsEngine:
    def __init__(self, constants: Optional[ThermalConstants] = None):
        self.C = constants or ThermalConstants()
        self._rng = np.random.default_rng(42)
    
    def seed(self, seed: int):
        self._rng = np.random.default_rng(seed)
    
    def ambient_temperature(self, t: int) -> float:
        phase = (2 * np.pi * t) / 1440
        return self.C.ambient_base + self.C.ambient_amplitude * np.sin(phase)
    
    def compute_load_profile(self, t: int, base_load: float = 50.0) -> float:
        noise = self._rng.normal(0, 5)
        spike = 20 * (self._rng.random() > 0.95)
        return np.clip(base_load + noise + spike, 10, 100)
    
    def cooling_power(self, state: DatacenterState, ambient: float) -> float:
        if not state.cooling_active:
            return 0.0
        delta_t = state.temperature - ambient
        return -self.C.k_cooling * delta_t * state.fan_efficiency
    
    def heat_generation(self, load: float) -> float:
        return self.C.k_heating * load
    
    def power_consumption(self, load: float, temp: float) -> float:
        compute_power = 0.5 + (load / 100) * 4.5
        cooling_overhead = 0.3 + 0.02 * max(0, temp - 60)
        return compute_power * (1 + cooling_overhead)
    
    def step(self, state: DatacenterState, dt: float = 1.0, inject_fan_failure: bool = False) -> DatacenterState:
        new_state = DatacenterState(
            timestamp=state.timestamp + 1,
            cooling_active=state.cooling_active,
            fan_failure=state.fan_failure or inject_fan_failure
        )
        
        if new_state.fan_failure:
            new_state.fan_efficiency = max(0.1, state.fan_efficiency - self.C.fan_degradation_rate * dt)
        else:
            new_state.fan_efficiency = state.fan_efficiency
        
        ambient = self.ambient_temperature(state.timestamp)
        new_state.load = self.compute_load_profile(state.timestamp)
        
        cooling = self.cooling_power(state, ambient)
        heating = self.heat_generation(new_state.load)
        
        dT_dt = (heating + cooling) / self.C.thermal_mass
        new_state.temperature = state.temperature + dT_dt * dt * self.C.thermal_mass
        new_state.power_draw = self.power_consumption(new_state.load, new_state.temperature)
        
        if new_state.temperature >= self.C.critical_temp:
            logger.warning(f"CRITICAL: {new_state.temperature:.1f}°C exceeds {self.C.critical_temp}°C")
        
        return new_state
    
    def run_simulation(self, steps: int = 100, dt: float = 1.0, 
                       initial_state: Optional[DatacenterState] = None,
                       failure_at_step: Optional[int] = None) -> list[DatacenterState]:
        state = initial_state or DatacenterState()
        history = [state]
        
        for step in range(steps):
            inject_failure = (failure_at_step is not None and step == failure_at_step)
            state = self.step(state, dt, inject_fan_failure=inject_failure)
            history.append(state)
        
        return history

def simulate_datacenter(steps: int = 100, failure_mode: bool = False, 
                        failure_step: int = 20, seed: Optional[int] = None) -> list[dict]:
    engine = ThermalPhysicsEngine()
    if seed is not None:
        engine.seed(seed)
    failure_at = failure_step if failure_mode else None
    history = engine.run_simulation(steps=steps, failure_at_step=failure_at)
    return [state.to_dict() for state in history]

if __name__ == "__main__":
    print("AARI World Model: Thermal Physics Demo")
    print("=" * 50)
    normal = simulate_datacenter(steps=50, failure_mode=False, seed=42)
    print(f"\n[Normal] Peak temp: {max(s['temp'] for s in normal):.1f}°C")
    failure = simulate_datacenter(steps=100, failure_mode=True, seed=42)
    print(f"[Failure] Peak temp: {max(s['temp'] for s in failure):.1f}°C")
