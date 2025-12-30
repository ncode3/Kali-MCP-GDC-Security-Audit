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
