#!/usr/bin/env python3
"""
AARI World Model: Telemetry Data Generator
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import json
from .physics import simulate_datacenter

def run_simulation(steps: int = 100, failure_mode: bool = False, failure_step: int = 20,
                   seed: Optional[int] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    history = simulate_datacenter(steps=steps, failure_mode=failure_mode, 
                                   failure_step=failure_step, seed=seed)
    df = pd.DataFrame(history)
    
    base_time = datetime.now() - timedelta(minutes=steps)
    df['datetime'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    df['datetime_str'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['temp_delta'] = df['temp'].diff().fillna(0)
    df['temp_rate'] = df['temp_delta'].rolling(window=5, min_periods=1).mean()
    
    def classify_severity(temp):
        if temp >= 85: return 'CRITICAL'
        elif temp >= 75: return 'WARNING'
        elif temp >= 70: return 'ELEVATED'
        return 'NORMAL'
    
    df['severity'] = df['temp'].apply(classify_severity)
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    
    return df

def generate_batch_telemetry(scenarios: list[dict], output_dir: str = "data/telemetry") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    
    for scenario in scenarios:
        name = scenario.get('name', 'unnamed')
        df = run_simulation(
            steps=scenario.get('steps', 100),
            failure_mode=scenario.get('failure_mode', False),
            failure_step=scenario.get('failure_step', 20),
            seed=scenario.get('seed', 42)
        )
        output_path = f"{output_dir}/{name}.csv"
        df.to_csv(output_path, index=False)
        results[name] = {
            'path': output_path,
            'rows': len(df),
            'peak_temp': float(df['temp'].max()),
            'critical_events': int((df['severity'] == 'CRITICAL').sum())
        }
    
    return results

STANDARD_SCENARIOS = [
    {'name': 'baseline_normal', 'steps': 100, 'failure_mode': False, 'seed': 42},
    {'name': 'fan_failure_early', 'steps': 150, 'failure_mode': True, 'failure_step': 20, 'seed': 42},
    {'name': 'fan_failure_late', 'steps': 150, 'failure_mode': True, 'failure_step': 80, 'seed': 42},
]
