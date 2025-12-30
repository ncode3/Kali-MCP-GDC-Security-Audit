#!/usr/bin/env python3
"""
AARI Datacenter MCP Server - Exposes simulation as Claude-callable tools
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import run_simulation
from src.anomaly import ThermalAnomalyDetector

class DatacenterMCPServer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.last_simulation_path = None
    
    def handle_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "aari-datacenter", "version": "0.1.0"}
        }
    
    def handle_tools_list(self) -> dict:
        return {"tools": [
            {"name": "run_simulation", "description": "Run thermal simulation",
             "inputSchema": {"type": "object", "properties": {
                 "steps": {"type": "integer", "default": 100},
                 "inject_failure": {"type": "boolean", "default": False},
                 "failure_step": {"type": "integer", "default": 20}
             }}},
            {"name": "detect_anomaly", "description": "Detect thermal anomalies",
             "inputSchema": {"type": "object", "properties": {
                 "threshold": {"type": "number", "default": 75.0}
             }}},
            {"name": "full_analysis", "description": "Complete analysis pipeline",
             "inputSchema": {"type": "object", "properties": {
                 "steps": {"type": "integer", "default": 100},
                 "inject_failure": {"type": "boolean", "default": False}
             }}}
        ]}
    
    def tool_run_simulation(self, params: dict) -> dict:
        import pandas as pd
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.data_dir / f"telemetry_{timestamp}.csv"
        
        df = run_simulation(
            steps=params.get('steps', 100),
            failure_mode=params.get('inject_failure', False),
            failure_step=params.get('failure_step', 20),
            seed=params.get('seed'),
            output_path=str(output_path)
        )
        self.last_simulation_path = str(output_path)
        
        return {
            "success": True, "data_path": str(output_path), "rows": len(df),
            "peak_temp": float(df['temp'].max()), "avg_load": float(df['load'].mean()),
            "critical_events": int((df['temp'] >= 85).sum())
        }
    
    def tool_detect_anomaly(self, params: dict) -> dict:
        import pandas as pd
        data_path = params.get('data_path') or self.last_simulation_path
        if not data_path or not Path(data_path).exists():
            return {"success": False, "error": "No telemetry data. Run simulation first."}
        
        df = pd.read_csv(data_path)
        detector = ThermalAnomalyDetector(warning_threshold=params.get('threshold', 75.0))
        report = detector.analyze(df, source_file=data_path)
        
        return {
            "success": True, "anomaly_count": report.anomaly_count,
            "risk_assessment": report.risk_assessment,
            "statistics": report.statistics
        }
    
    def tool_full_analysis(self, params: dict) -> dict:
        sim = self.tool_run_simulation(params)
        if not sim.get('success'):
            return sim
        anomaly = self.tool_detect_anomaly({'threshold': params.get('threshold', 75.0)})
        return {"simulation": sim, "anomalies": anomaly}
    
    def handle_tools_call(self, params: dict) -> dict:
        handlers = {
            'run_simulation': self.tool_run_simulation,
            'detect_anomaly': self.tool_detect_anomaly,
            'full_analysis': self.tool_full_analysis
        }
        handler = handlers.get(params.get('name'))
        if not handler:
            return {"content": [{"type": "text", "text": json.dumps({"error": "Unknown tool"})}], "isError": True}
        
        result = handler(params.get('arguments', {}))
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False}
    
    def handle_request(self, request: dict) -> dict:
        method = request.get('method', '')
        params = request.get('params', {})
        handlers = {
            'initialize': lambda p: self.handle_initialize(p),
            'tools/list': lambda p: self.handle_tools_list(),
            'tools/call': lambda p: self.handle_tools_call(p)
        }
        handler = handlers.get(method)
        if handler:
            return {"jsonrpc": "2.0", "id": request.get('id'), "result": handler(params)}
        return {"jsonrpc": "2.0", "id": request.get('id'), "error": {"code": -32601, "message": f"Unknown: {method}"}}
    
    def run(self):
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + '\n')
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}) + '\n')
                sys.stdout.flush()

if __name__ == "__main__":
    DatacenterMCPServer().run()
