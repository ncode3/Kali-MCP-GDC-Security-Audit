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
