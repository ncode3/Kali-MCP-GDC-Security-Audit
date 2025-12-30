#!/usr/bin/env python3
"""
AARI World Model: Thermal Anomaly Detection
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class AnomalyEvent:
    timestamp: str
    step: int
    temperature: float
    severity: str
    detection_method: str
    description: str
    recommended_action: str
    
    def to_dict(self) -> dict:
        return vars(self)

@dataclass  
class AnalysisReport:
    source_file: str
    analysis_time: str
    total_samples: int
    anomaly_count: int
    anomalies: list
    statistics: dict
    risk_assessment: str
    
    def to_dict(self) -> dict:
        return {
            'source_file': self.source_file,
            'analysis_time': self.analysis_time,
            'total_samples': self.total_samples,
            'anomaly_count': self.anomaly_count,
            'anomalies': [a.to_dict() for a in self.anomalies],
            'statistics': self.statistics,
            'risk_assessment': self.risk_assessment
        }

class ThermalAnomalyDetector:
    def __init__(self, warning_threshold: float = 75.0, critical_threshold: float = 85.0,
                 rate_threshold: float = 2.0, zscore_threshold: float = 2.5):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.rate_threshold = rate_threshold
        self.zscore_threshold = zscore_threshold
    
    def detect_threshold_anomalies(self, df: pd.DataFrame) -> list:
        anomalies = []
        for idx in df[df['temp'] >= self.critical_threshold].index:
            row = df.loc[idx]
            anomalies.append(AnomalyEvent(
                timestamp=str(row.get('datetime_str', idx)),
                step=int(row.get('timestamp', idx)),
                temperature=float(row['temp']),
                severity='CRITICAL',
                detection_method='threshold',
                description=f"Temperature {row['temp']:.1f}°C exceeds critical {self.critical_threshold}°C",
                recommended_action='IMMEDIATE: Emergency cooling required'
            ))
        return anomalies
    
    def predict_time_to_critical(self, df: pd.DataFrame) -> Optional[dict]:
        if len(df) < 10:
            return None
        recent = df.tail(20).copy()
        if 'temp_delta' not in recent.columns:
            recent['temp_delta'] = recent['temp'].diff().fillna(0)
        
        avg_rate = recent['temp_delta'].mean()
        current_temp = recent['temp'].iloc[-1]
        
        if avg_rate <= 0:
            return {'prediction': 'stable_or_cooling', 'current_temp': float(current_temp), 
                    'rate': float(avg_rate), 'time_to_critical': None}
        
        time_to_critical = (self.critical_threshold - current_temp) / avg_rate
        return {
            'prediction': 'heating',
            'current_temp': float(current_temp),
            'rate': float(avg_rate),
            'time_to_critical': float(time_to_critical) if time_to_critical > 0 else 0,
            'urgency': 'HIGH' if time_to_critical < 30 else 'MEDIUM' if time_to_critical < 60 else 'LOW'
        }
    
    def analyze(self, df: pd.DataFrame, source_file: str = "unknown") -> AnalysisReport:
        anomalies = self.detect_threshold_anomalies(df)
        
        statistics = {
            'min_temp': float(df['temp'].min()),
            'max_temp': float(df['temp'].max()),
            'mean_temp': float(df['temp'].mean()),
            'std_temp': float(df['temp'].std()),
            'critical_count': len([a for a in anomalies if a.severity == 'CRITICAL'])
        }
        
        prediction = self.predict_time_to_critical(df)
        if prediction:
            statistics['prediction'] = prediction
        
        risk = 'CRITICAL' if statistics['critical_count'] > 0 else 'LOW'
        
        return AnalysisReport(
            source_file=source_file,
            analysis_time=datetime.now().isoformat(),
            total_samples=len(df),
            anomaly_count=len(anomalies),
            anomalies=anomalies,
            statistics=statistics,
            risk_assessment=risk
        )

def detect_thermal_anomaly(data_path: str, threshold: float = 75.0) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    detector = ThermalAnomalyDetector(warning_threshold=threshold)
    report = detector.analyze(df, source_file=data_path)
    return pd.DataFrame([a.to_dict() for a in report.anomalies])

def analyze_telemetry_file(data_path: str) -> dict:
    df = pd.read_csv(data_path)
    detector = ThermalAnomalyDetector()
    return detector.analyze(df, source_file=data_path).to_dict()
