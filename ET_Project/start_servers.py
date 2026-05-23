"""
Start all AutoPrepAI v5 servers.

Launches:
  1. MCP Server (FastMCP) on port 8100
  2. A2A Agent Servers on ports 8201-8209 (9 agents)

Prerequisites:
  - Ollama running: ollama serve
  - Model pulled: ollama pull llama3.2
  - Dependencies: pip install -r requirements.txt
"""

import shutil, subprocess, sys, time, os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def start_server(name, cmd, port):
    """Start a server process."""
    print(f"  Starting {name} on port {port}...")
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    time.sleep(1.5)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        print(f"  [FAIL] {name} failed to start: {stderr[:300]}")
        return None
    print(f"  [OK] {name} running (PID: {proc.pid})")
    return proc


def main():
    print("=" * 60)
    print("  AutoPrepAI v5 -- Starting All Servers")
    print("=" * 60)
    print()

    # Check Ollama
    print("[*] Checking Ollama...")
    try:
        import ollama
        models = ollama.list()
        available = [m.model for m in models.models] if hasattr(models, 'models') else []
        found = any("llama3.2" in m for m in available)
        if found:
            print("  [OK] Ollama running. llama3.2 available.")
        else:
            print(f"  [WARN] Ollama running but llama3.2 not found. Available: {available}")
            print("     Run: ollama pull llama3.2")
    except Exception as e:
        print(f"  [FAIL] Ollama not available: {e}")
        print("     Run: ollama serve")
        return

    print()
    processes = []

    # -- MCP Server --
    print("[*] MCP Server (FastMCP)")
    fastmcp_exe = shutil.which("fastmcp")
    if fastmcp_exe is None:
        print("  [FAIL] fastmcp CLI not found. Run: pip install fastmcp")
    else:
        p = start_server(
            "MCP Tools Server",
            [fastmcp_exe, "run",
             "mcp_server/server.py:mcp",
             "--transport", "http", "--port", "8100"],
            8100,
        )
        if p:
            processes.append(("MCP Server", p))

    time.sleep(2)

    # -- A2A Agents (9 total) --
    print()
    print("[*] A2A Agent Servers")

    agents = [
        ("DataUnderstandingAgent", "a2a_agents.data_understanding:app", 8201),
        ("DataQualityAgent",      "a2a_agents.data_quality:app",       8202),
        ("MissingValueAgent",     "a2a_agents.missing_values:app",     8203),
        ("EncodingAgent",         "a2a_agents.encoding:app",           8204),
        ("FeatureEngineeringAgent", "a2a_agents.feature_engineering:app", 8205),
        ("AutoMLAgent",           "a2a_agents.automl:app",             8206),
        ("EDAAgent",              "a2a_agents.eda:app",                8207),
        ("ReflectionAgent",       "a2a_agents.reflection:app",         8208),
        ("ReportAgent",           "a2a_agents.report:app",             8209),
    ]

    for name, module, port in agents:
        p = start_server(
            name,
            [sys.executable, "-m", "uvicorn", module,
             "--host", "0.0.0.0", "--port", str(port),
             "--log-level", "warning"],
            port,
        )
        if p:
            processes.append((name, p))
        time.sleep(1)

    print()
    print("=" * 60)
    print(f"  [OK] {len(processes)} servers started successfully")
    print()
    print("  MCP Server:  http://localhost:8100/mcp")
    for name, _, port in agents:
        print(f"  {name}: http://localhost:{port}/.well-known/agent.json")
    print()
    print("  Next: streamlit run app.py")
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop all servers.")

    reported_dead = set()
    try:
        while True:
            time.sleep(3)
            alive = 0
            for name, proc in processes:
                if proc.poll() is not None:
                    if name not in reported_dead:
                        reported_dead.add(name)
                        print(f"  [WARN] {name} stopped (exit code: {proc.returncode})")
                else:
                    alive += 1
            if alive == 0 and len(processes) > 0:
                print("\n  All servers stopped. Exiting.")
                break
    except KeyboardInterrupt:
        print("\n\nShutting down all servers...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f"  Stopped {name}")
        print("Done.")


if __name__ == "__main__":
    main()
