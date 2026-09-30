import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from tools import get_tool_schemas, execute_tool
schemas = get_tool_schemas()
print(f"[OK] tools.py - {len(schemas)} tools registered:")
for s in schemas:
    print(f"  - {s['function']['name']}")

from agents import run_orchestrator, AGENT_CONFIGS
print(f"[OK] agents.py - {len(AGENT_CONFIGS)} agents configured:")
for k in AGENT_CONFIGS.keys():
    print(f"  - {k}")

print("[READY] All imports successful!")
