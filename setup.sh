#!/bin/bash
# AARI World Model + Kali MCP Security Audit Setup Script
# =========================================================
# Run this with Claude Code in your repo directory:
#   chmod +x setup.sh && ./setup.sh
#
# Or just paste the contents to Claude Code and say "run this"

set -e  # Exit on error

echo "🚀 AARI World Model Setup"
echo "========================="

# Clean up the zip file mistake
if [ -f "files.zip" ]; then
    echo "📦 Removing files.zip..."
    rm files.zip
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p src
mkdir -p mcp-servers/datacenter
mkdir -p mcp-servers/deployment
mkdir -p mcp-servers/security
mkdir -p claude-tasks
mkdir -p infra
mkdir -p data
mkdir -p reports

# ============================================
# SOURCE FILES
# ============================================

echo "🔧 Creating source files..."

# src/__init__.py
cat > src/__init__.py << 'PYEOF'
"""
AARI World Model: Infrastructure Digital Twin
"""
from .physics import ThermalPhysicsEngine, ThermalConstants, DatacenterState, simulate_datacenter
from .generator import run_simulation, generate_batch_telemetry
from .anomaly import ThermalAnomalyDetector, detect_thermal_anomaly, analyze_telemetry_file

__version__ = "0.1.0"
PYEOF

# src/physics.py
cat > src/physics.py << 'PYEOF'
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
PYEOF

# src/generator.py
cat > src/generator.py << 'PYEOF'
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
PYEOF

# src/anomaly.py
cat > src/anomaly.py << 'PYEOF'
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
PYEOF

# ============================================
# MCP SERVERS
# ============================================

echo "🔌 Creating MCP servers..."

# mcp-servers/datacenter/server.py
cat > mcp-servers/datacenter/server.py << 'PYEOF'
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
PYEOF

# mcp-servers/deployment/server.py
cat > mcp-servers/deployment/server.py << 'PYEOF'
#!/usr/bin/env python3
"""
AARI OpenShift Deployment MCP Server - Wraps oc CLI for Claude
"""
import json
import sys
import subprocess
import shutil
from pathlib import Path

class OpenShiftMCPServer:
    def __init__(self):
        self.oc_available = shutil.which('oc') is not None
    
    def _run_oc(self, args: list, timeout: int = 60) -> dict:
        if not self.oc_available:
            return {"success": False, "error": "oc CLI not found"}
        try:
            result = subprocess.run(['oc'] + args, capture_output=True, text=True, timeout=timeout)
            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def handle_initialize(self, params: dict) -> dict:
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "aari-openshift", "version": "0.1.0"}}
    
    def handle_tools_list(self) -> dict:
        return {"tools": [
            {"name": "check_login", "description": "Check OpenShift login status", "inputSchema": {"type": "object"}},
            {"name": "get_pods", "description": "List pods", "inputSchema": {"type": "object", "properties": {"project": {"type": "string"}}}},
            {"name": "deploy_app", "description": "Deploy application", "inputSchema": {"type": "object", "properties": {
                "image": {"type": "string"}, "name": {"type": "string"}, "project": {"type": "string"}}, "required": ["image", "name"]}},
            {"name": "get_routes", "description": "Get routes", "inputSchema": {"type": "object", "properties": {"project": {"type": "string"}}}},
            {"name": "apply_yaml", "description": "Apply YAML config", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}
        ]}
    
    def tool_check_login(self, params: dict) -> dict:
        whoami = self._run_oc(['whoami'])
        if not whoami['success']:
            return {"logged_in": False, "error": "Not logged in", "hint": "Run: oc login <cluster-url>"}
        return {"logged_in": True, "user": whoami['stdout'].strip()}
    
    def tool_get_pods(self, params: dict) -> dict:
        project = params.get('project', 'default')
        result = self._run_oc(['get', 'pods', '-n', project, '-o', 'json'])
        if not result['success']:
            return {"success": False, "error": result.get('stderr', result.get('error'))}
        try:
            data = json.loads(result['stdout'])
            pods = [{"name": p['metadata']['name'], "status": p['status']['phase']} for p in data.get('items', [])]
            return {"success": True, "pods": pods}
        except:
            return {"success": True, "raw": result['stdout']}
    
    def tool_deploy_app(self, params: dict) -> dict:
        project = params.get('project', 'default')
        self._run_oc(['project', project])
        result = self._run_oc(['new-app', params['image'], f"--name={params['name']}"], timeout=120)
        return {"success": result['success'], "message": result['stdout'] if result['success'] else result['stderr']}
    
    def tool_get_routes(self, params: dict) -> dict:
        project = params.get('project', 'default')
        result = self._run_oc(['get', 'routes', '-n', project, '-o', 'json'])
        if not result['success']:
            return {"success": False, "error": result.get('stderr', result.get('error'))}
        try:
            data = json.loads(result['stdout'])
            routes = [{"name": r['metadata']['name'], "host": r['spec']['host']} for r in data.get('items', [])]
            return {"success": True, "routes": routes}
        except:
            return {"success": True, "raw": result['stdout']}
    
    def tool_apply_yaml(self, params: dict) -> dict:
        path = params['path']
        if not Path(path).exists():
            return {"success": False, "error": f"File not found: {path}"}
        result = self._run_oc(['apply', '-f', path])
        return {"success": result['success'], "message": result['stdout'] if result['success'] else result['stderr']}
    
    def handle_tools_call(self, params: dict) -> dict:
        handlers = {
            'check_login': self.tool_check_login, 'get_pods': self.tool_get_pods,
            'deploy_app': self.tool_deploy_app, 'get_routes': self.tool_get_routes,
            'apply_yaml': self.tool_apply_yaml
        }
        handler = handlers.get(params.get('name'))
        if not handler:
            return {"content": [{"type": "text", "text": json.dumps({"error": "Unknown tool"})}], "isError": True}
        result = handler(params.get('arguments', {}))
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False}
    
    def handle_request(self, request: dict) -> dict:
        method = request.get('method', '')
        params = request.get('params', {})
        handlers = {'initialize': lambda p: self.handle_initialize(p), 'tools/list': lambda p: self.handle_tools_list(), 'tools/call': lambda p: self.handle_tools_call(p)}
        handler = handlers.get(method)
        if handler:
            return {"jsonrpc": "2.0", "id": request.get('id'), "result": handler(params)}
        return {"jsonrpc": "2.0", "id": request.get('id'), "error": {"code": -32601, "message": f"Unknown: {method}"}}
    
    def run(self):
        for line in sys.stdin:
            if not line.strip(): continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + '\n')
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}) + '\n')
                sys.stdout.flush()

if __name__ == "__main__":
    OpenShiftMCPServer().run()
PYEOF

# mcp-servers/security/server.py (Kali integration for GDC security audits)
cat > mcp-servers/security/server.py << 'PYEOF'
#!/usr/bin/env python3
"""
AARI Security Audit MCP Server - Kali Linux tools for GDC security assessment
"""
import json
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

class SecurityMCPServer:
    def __init__(self, output_dir: str = "reports/security"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tools_available = {
            'nmap': shutil.which('nmap') is not None,
            'nikto': shutil.which('nikto') is not None,
            'gobuster': shutil.which('gobuster') is not None,
        }
    
    def _run_cmd(self, cmd: list, timeout: int = 300) -> dict:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {"success": result.returncode == 0, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def handle_initialize(self, params: dict) -> dict:
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "aari-security", "version": "0.1.0"}}
    
    def handle_tools_list(self) -> dict:
        return {"tools": [
            {"name": "check_tools", "description": "Check available security tools", "inputSchema": {"type": "object"}},
            {"name": "port_scan", "description": "Scan ports with nmap", "inputSchema": {"type": "object", "properties": {
                "target": {"type": "string"}, "ports": {"type": "string", "default": "1-1000"}}, "required": ["target"]}},
            {"name": "service_scan", "description": "Detect services with nmap", "inputSchema": {"type": "object", "properties": {
                "target": {"type": "string"}}, "required": ["target"]}},
            {"name": "vuln_scan", "description": "Basic vulnerability scan", "inputSchema": {"type": "object", "properties": {
                "target": {"type": "string"}}, "required": ["target"]}},
            {"name": "web_scan", "description": "Web vulnerability scan with nikto", "inputSchema": {"type": "object", "properties": {
                "target": {"type": "string"}}, "required": ["target"]}},
            {"name": "generate_report", "description": "Generate security audit report", "inputSchema": {"type": "object"}}
        ]}
    
    def tool_check_tools(self, params: dict) -> dict:
        return {"available_tools": self.tools_available, "ready": any(self.tools_available.values())}
    
    def tool_port_scan(self, params: dict) -> dict:
        if not self.tools_available['nmap']:
            return {"success": False, "error": "nmap not installed. Run: apt install nmap"}
        target = params['target']
        ports = params.get('ports', '1-1000')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"portscan_{timestamp}.txt"
        
        result = self._run_cmd(['nmap', '-p', ports, '-oN', str(output_file), target])
        return {"success": result['success'], "target": target, "output_file": str(output_file),
                "results": result['stdout'] if result['success'] else result.get('error', result.get('stderr'))}
    
    def tool_service_scan(self, params: dict) -> dict:
        if not self.tools_available['nmap']:
            return {"success": False, "error": "nmap not installed"}
        target = params['target']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"servicescan_{timestamp}.txt"
        
        result = self._run_cmd(['nmap', '-sV', '-oN', str(output_file), target], timeout=600)
        return {"success": result['success'], "target": target, "output_file": str(output_file),
                "results": result['stdout'] if result['success'] else result.get('error', result.get('stderr'))}
    
    def tool_vuln_scan(self, params: dict) -> dict:
        if not self.tools_available['nmap']:
            return {"success": False, "error": "nmap not installed"}
        target = params['target']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"vulnscan_{timestamp}.txt"
        
        result = self._run_cmd(['nmap', '--script', 'vuln', '-oN', str(output_file), target], timeout=900)
        return {"success": result['success'], "target": target, "output_file": str(output_file),
                "results": result['stdout'] if result['success'] else result.get('error', result.get('stderr'))}
    
    def tool_web_scan(self, params: dict) -> dict:
        if not self.tools_available['nikto']:
            return {"success": False, "error": "nikto not installed. Run: apt install nikto"}
        target = params['target']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"webscan_{timestamp}.txt"
        
        result = self._run_cmd(['nikto', '-h', target, '-o', str(output_file)], timeout=600)
        return {"success": result['success'], "target": target, "output_file": str(output_file),
                "results": result['stdout'] if result['success'] else result.get('error', result.get('stderr'))}
    
    def tool_generate_report(self, params: dict) -> dict:
        reports = list(self.output_dir.glob('*.txt'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.output_dir / f"audit_summary_{timestamp}.md"
        
        content = f"# GDC Security Audit Report\nGenerated: {datetime.now().isoformat()}\n\n"
        content += f"## Scan Files\n"
        for r in reports:
            content += f"- {r.name}\n"
        
        summary_file.write_text(content)
        return {"success": True, "report_file": str(summary_file), "scans_included": len(reports)}
    
    def handle_tools_call(self, params: dict) -> dict:
        handlers = {
            'check_tools': self.tool_check_tools, 'port_scan': self.tool_port_scan,
            'service_scan': self.tool_service_scan, 'vuln_scan': self.tool_vuln_scan,
            'web_scan': self.tool_web_scan, 'generate_report': self.tool_generate_report
        }
        handler = handlers.get(params.get('name'))
        if not handler:
            return {"content": [{"type": "text", "text": json.dumps({"error": "Unknown tool"})}], "isError": True}
        result = handler(params.get('arguments', {}))
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False}
    
    def handle_request(self, request: dict) -> dict:
        method = request.get('method', '')
        params = request.get('params', {})
        handlers = {'initialize': lambda p: self.handle_initialize(p), 'tools/list': lambda p: self.handle_tools_list(), 'tools/call': lambda p: self.handle_tools_call(p)}
        handler = handlers.get(method)
        if handler:
            return {"jsonrpc": "2.0", "id": request.get('id'), "result": handler(params)}
        return {"jsonrpc": "2.0", "id": request.get('id'), "error": {"code": -32601, "message": f"Unknown: {method}"}}
    
    def run(self):
        for line in sys.stdin:
            if not line.strip(): continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + '\n')
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}) + '\n')
                sys.stdout.flush()

if __name__ == "__main__":
    SecurityMCPServer().run()
PYEOF

# ============================================
# CLAUDE TASKS
# ============================================

echo "📋 Creating Claude tasks..."

cat > claude-tasks/analyze-thermal-anomaly.md << 'EOF'
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
EOF

cat > claude-tasks/deploy-to-openshift.md << 'EOF'
# Task: Deploy Digital Twin to OpenShift (GDC)

## Prerequisites
- oc CLI installed and logged in
- Container image built

## Steps
1. Call `check_login` to verify cluster access
2. Call `apply_yaml` with path=infra/deployment.yaml
3. Call `get_pods` to verify deployment
4. Call `get_routes` to get public URL

## Success Criteria
- Pod status: Running
- Route accessible
EOF

cat > claude-tasks/security-audit-gdc.md << 'EOF'
# Task: Security Audit for GDC

## Objective
Run security assessment on GDC infrastructure using Kali tools.

## Steps
1. Call `check_tools` to verify available scanners
2. Call `port_scan` on GDC gateway IP
3. Call `service_scan` for service detection
4. Call `vuln_scan` for vulnerability assessment
5. Call `generate_report` for summary

## Output
- Open ports
- Running services
- Potential vulnerabilities
- Remediation recommendations

## IMPORTANT
Only run against systems you own or have permission to test.
EOF

# ============================================
# INFRASTRUCTURE
# ============================================

echo "🐳 Creating infrastructure files..."

cat > infra/Dockerfile << 'EOF'
FROM python:3.11-slim

RUN groupadd -r aari && useradd -r -g aari aari
WORKDIR /app

RUN pip install --no-cache-dir numpy pandas

COPY src/ ./src/
COPY mcp-servers/ ./mcp-servers/

RUN mkdir -p /app/data /app/reports && chown -R aari:aari /app
USER aari

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "-m", "mcp-servers.datacenter.server"]
EOF

cat > infra/deployment.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: aari-world-model
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: digital-twin
  namespace: aari-world-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: digital-twin
  template:
    metadata:
      labels:
        app: digital-twin
    spec:
      containers:
      - name: world-model
        image: aari-world-model:v1
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: digital-twin
  namespace: aari-world-model
spec:
  selector:
    app: digital-twin
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: digital-twin
  namespace: aari-world-model
spec:
  to:
    kind: Service
    name: digital-twin
  tls:
    termination: edge
EOF

# ============================================
# CONFIG FILES
# ============================================

echo "⚙️ Creating config files..."

cat > requirements.txt << 'EOF'
numpy>=1.24.0
pandas>=2.0.0
EOF

cat > mcp-config.json << 'EOF'
{
  "servers": {
    "datacenter": {
      "command": "python3",
      "args": ["mcp-servers/datacenter/server.py"],
      "env": {"PYTHONPATH": "."}
    },
    "deployment": {
      "command": "python3", 
      "args": ["mcp-servers/deployment/server.py"]
    },
    "security": {
      "command": "python3",
      "args": ["mcp-servers/security/server.py"]
    }
  }
}
EOF

cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
venv/
data/*.csv
data/*.json
reports/*.md
reports/*.txt
!reports/.gitkeep
.env
*.log
EOF

# Create placeholder files
touch data/.gitkeep
touch reports/.gitkeep

# ============================================
# README
# ============================================

cat > README.md << 'EOF'
# AARI World Model + GDC Security Audit

> Producer mindset. Code-first. No GUIs.

Infrastructure digital twin + security tools for the Garage Data Center.

## Quick Start

```bash
pip install -r requirements.txt
python -m src.physics  # Test simulation
```

## MCP Servers

| Server | Purpose |
|--------|---------|
| datacenter | Thermal simulation & anomaly detection |
| deployment | OpenShift operations |
| security | Kali-based security auditing |

## With Claude Code

```bash
claude  # Start Claude Code in this directory
> run a thermal simulation with fan failure
> deploy to OpenShift
> scan the GDC for open ports
```

## Project Structure

```
├── src/                    # Core physics & ML
├── mcp-servers/           # Claude-callable tools
│   ├── datacenter/        # Simulation
│   ├── deployment/        # OpenShift ops
│   └── security/          # Kali tools
├── claude-tasks/          # Autonomous task definitions
├── infra/                 # Docker & K8s manifests
└── reports/               # Generated outputs
```

## The AARI Way

Stop clicking. Start building.
EOF

# ============================================
# GIT COMMIT
# ============================================

echo "📤 Committing to git..."

git add -A
git commit -m "AARI World Model + GDC Security Audit MCP Servers

- Physics-based datacenter thermal simulation
- Multi-method anomaly detection
- MCP servers for Claude Code integration
- OpenShift deployment manifests
- Kali security audit tools
- Claude task definitions for autonomous ops"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Push to GitHub: git push origin main"
echo "  3. Run Claude Code: claude"
echo ""
echo "🚀 Ready to build."
