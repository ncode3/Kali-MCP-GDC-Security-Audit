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
