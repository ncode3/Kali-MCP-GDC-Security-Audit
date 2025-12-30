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
