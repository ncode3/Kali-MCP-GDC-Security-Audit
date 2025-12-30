# Task: Analyze Datacenter Thermal Anomaly

## Objective
Simulate a fan failure, detect anomalies, and generate an incident report.

## Steps
1. Call `run_simulation` with steps=150, inject_failure=true, failure_step=30
2. Call `detect_anomaly` with threshold=75.0
3. Generate report to `reports/thermal-analysis-{timestamp}.md`

## Output
- Peak temperature
- Anomaly count  
- Risk assessment
- Recommended actions
