# AARI World Model + GDC Security Audit

> **Producer mindset. Code-first. No GUIs.**

A physics-based infrastructure digital twin and security assessment toolkit for the **Garage Data Center (GDC)**. Built for AARI students to learn real infrastructure engineering through AI-native tooling.

---

## What This Is

This repository contains three **MCP (Model Context Protocol) servers** that turn Claude Code into an infrastructure operator:

| Server | What It Does |
|--------|--------------|
| **Datacenter** | Simulates thermal dynamics, detects anomalies, predicts failures |
| **Deployment** | Deploys applications to OpenShift/Kubernetes clusters |
| **Security** | Runs Kali Linux security tools (nmap, nikto) for GDC audits |

**Why MCP instead of Streamlit?**  
Streamlit teaches students to click buttons. MCP teaches students to build AI-native automation. Same pattern used in production systems.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Physics Engine

```bash
python -m src.physics
```

Expected output:
```
AARI World Model: Thermal Physics Demo
==================================================
[Normal] Peak temp: 65.0°C
[Failure] Peak temp: 178.9°C
```

### 3. Run with Claude Code

```bash
# Start Claude Code in this directory
claude

# Then ask it to:
> Run a thermal simulation with fan failure at step 30
> Analyze the telemetry for anomalies
> Deploy the digital twin to OpenShift
```

---

## The Physics

The datacenter simulation models **Newton's Law of Cooling** with failure cascades:

```
dT/dt = (k_heating × load - k_cooling × (T - T_ambient) × efficiency) / thermal_mass
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k_heating` | 0.08 | Heat generated per % load per minute |
| `k_cooling` | 0.15 | Cooling coefficient |
| `T_ambient` | 22°C | Base ambient temperature |
| `critical_temp` | 85°C | Emergency shutdown threshold |
| `fan_degradation` | 0.02/min | Efficiency loss during failure |

### Failure Cascade

When a fan fails at step 20:
1. **Step 20**: Fan efficiency starts degrading (1.0 → 0.98 → 0.96...)
2. **Step 55**: Temperature hits warning threshold (75°C)
3. **Step 70**: Temperature hits critical threshold (85°C)
4. **Step 100**: Runaway to 178°C without intervention

This is why **predictive monitoring matters** - catching the trend at step 30 gives you 40 minutes to respond.

---

## MCP Server Details

### Datacenter Server (`mcp-servers/datacenter/server.py`)

**Tools:**
- `run_simulation` - Execute thermal simulation with configurable parameters
- `detect_anomaly` - Multi-method anomaly detection (threshold, rate-of-change, statistical)
- `full_analysis` - Complete pipeline: simulate → detect → predict

**Example via Claude Code:**
```
> Run simulation with 150 steps, inject fan failure at step 30, then analyze for anomalies above 70°C
```

### Deployment Server (`mcp-servers/deployment/server.py`)

**Tools:**
- `check_login` - Verify OpenShift cluster access
- `get_pods` - List pods in a project
- `deploy_app` - Deploy container image
- `get_routes` - Get public URLs
- `apply_yaml` - Apply Kubernetes manifests

**Prerequisites:**
```bash
# Install OpenShift CLI
brew install openshift-cli  # macOS
# or download from Red Hat

# Login to GDC
oc login https://api.gdc.local:6443 --token=<your-token>
```

### Security Server (`mcp-servers/security/server.py`)

**Tools:**
- `check_tools` - Verify available scanners (nmap, nikto, gobuster)
- `port_scan` - TCP port scanning
- `service_scan` - Service version detection
- `vuln_scan` - Vulnerability assessment
- `web_scan` - Web application scanning
- `generate_report` - Compile audit summary

**Prerequisites:**
```bash
# On Kali Linux or install tools manually
apt install nmap nikto gobuster
```

**⚠️ IMPORTANT:** Only scan systems you own or have explicit permission to test.

---

## Project Structure

```
aari-world-model/
├── src/
│   ├── __init__.py          # Package exports
│   ├── physics.py           # Thermal simulation engine
│   ├── generator.py         # Telemetry data pipeline
│   └── anomaly.py           # Anomaly detection algorithms
│
├── mcp-servers/
│   ├── datacenter/
│   │   └── server.py        # Simulation MCP server
│   ├── deployment/
│   │   └── server.py        # OpenShift MCP server
│   └── security/
│       └── server.py        # Kali tools MCP server
│
├── claude-tasks/
│   ├── analyze-thermal-anomaly.md
│   ├── deploy-to-openshift.md
│   └── security-audit-gdc.md
│
├── infra/
│   ├── Dockerfile           # Multi-stage container build
│   └── deployment.yaml      # OpenShift/K8s manifests
│
├── data/                    # Generated telemetry (gitignored)
├── reports/                 # Generated analysis reports
├── mcp-config.json          # Claude Code MCP configuration
└── requirements.txt         # Python dependencies
```

---

## Claude Tasks

Pre-built task definitions in `claude-tasks/` that Claude Code can execute autonomously:

### `analyze-thermal-anomaly.md`
Simulates a failure scenario and generates an incident report.

### `deploy-to-openshift.md`
Deploys the digital twin to the GDC OpenShift cluster.

### `security-audit-gdc.md`
Runs a security assessment against GDC infrastructure.

**Run a task:**
```bash
claude "Execute the thermal anomaly analysis task"
```

---

## Deploy to GDC

### Build Container

```bash
docker build -t aari-world-model:v1 -f infra/Dockerfile .
```

### Deploy to OpenShift

```bash
# Login to Dave's cluster
oc login https://api.gdc.local:6443

# Apply manifests
oc apply -f infra/deployment.yaml

# Verify
oc get pods -n aari-world-model
oc get route -n aari-world-model
```

### Or via Claude Code

```bash
claude "Deploy the digital twin to OpenShift and verify it's running"
```

---

## For AARI Students

This project teaches:

1. **Physics First** - Understand the math before touching tools
2. **Data Pipelines** - Generate, transform, and analyze telemetry
3. **Anomaly Detection** - Multiple methods, defense in depth
4. **MCP Protocol** - How AI agents invoke tools
5. **Container Deployment** - Docker, Kubernetes, OpenShift
6. **Security Assessment** - Ethical penetration testing

### Learning Path

| Week | Focus | Files |
|------|-------|-------|
| 1 | Physics simulation | `src/physics.py` |
| 2 | Data generation | `src/generator.py` |
| 3 | Anomaly detection | `src/anomaly.py` |
| 4 | MCP development | `mcp-servers/*/server.py` |
| 5 | Container deployment | `infra/` |
| 6 | Security auditing | `mcp-servers/security/` |

---

## The AARI Way

> "The cloud is just someone else's computer. Stop renting. Start building."

This project exists because:
- **AWS** won't teach you infrastructure
- **Bootcamps** won't teach you physics
- **Universities** won't teach you production

AARI teaches all three.

---

## Links

- **AARI Website**: [atlanta-robotics.org](https://atlanta-robotics.org)
- **MCP Protocol**: [modelcontextprotocol.io](https://modelcontextprotocol.io)
- **OpenShift Docs**: [docs.openshift.com](https://docs.openshift.com)

---

## License

Apache 2.0 - Build freely, deploy anywhere.

---

**Questions?** nolan@atlanta-robotics.org
