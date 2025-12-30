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
