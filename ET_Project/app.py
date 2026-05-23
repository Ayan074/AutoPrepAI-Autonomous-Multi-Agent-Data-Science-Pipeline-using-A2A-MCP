"""
AutoPrepAI v5 — Autonomous Multi-Agent Pipeline (Streamlit UI)

Powered by:
  - Ollama (local LLM — llama3.2)
  - FastMCP (real MCP tools server)
  - A2A SDK (real agent-to-agent protocol)

Decision-making: 100% LLM-driven
Execution: 100% deterministic (pandas/sklearn)
"""

import io
import json
import os
import time

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPrepAI — Real LLM Multi-Agent Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# PREMIUM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #f0f0f5;
        --text-secondary: #9ca3af;
        --text-muted: #6b7280;
        --accent-indigo: #6366f1;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --gradient-primary: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
        --radius: 16px;
        --radius-sm: 10px;
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    h1 {
        font-size: 2.2rem !important;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        padding: 18px 20px !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: var(--border-glow) !important;
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    .stButton > button {
        background: var(--gradient-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        padding: 12px 32px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
    }

    .step-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin: 6px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
    }
    .step-card.agent { border-left: 3px solid var(--accent-indigo); }
    .step-card.tool { border-left: 3px solid var(--accent-cyan); }
    .step-card.finish { border-left: 3px solid var(--accent-emerald); }
    .step-card.error { border-left: 3px solid var(--accent-rose); }

    .hero-container { text-align: center; padding: 60px 20px; }
    .hero-title {
        font-size: 3.5rem; font-weight: 900;
        background: var(--gradient-primary);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 10px;
        letter-spacing: -0.04em; line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.15rem; color: var(--text-secondary);
        max-width: 600px; margin: 0 auto; line-height: 1.6;
    }
    .hero-badge {
        display: inline-block; background: rgba(99, 102, 241, 0.12);
        color: var(--accent-indigo); font-size: 0.75rem; font-weight: 700;
        padding: 5px 14px; border-radius: 20px; margin-bottom: 16px;
        letter-spacing: 0.1em; text-transform: uppercase;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .tech-badge {
        display: inline-block; background: rgba(16, 185, 129, 0.12);
        color: var(--accent-emerald); font-size: 0.7rem; font-weight: 700;
        padding: 4px 10px; border-radius: 12px; margin: 2px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: var(--bg-secondary) !important; border-radius: var(--radius-sm) !important; padding: 4px !important; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px !important; color: var(--text-secondary) !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { background: var(--accent-indigo) !important; color: white !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "execution_trace" not in st.session_state:
    st.session_state.execution_trace = []
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def check_ollama_status():
    """Check if Ollama is running and llama3.2 is available."""
    try:
        import ollama
        models = ollama.list()
        available = [m.model for m in models.models] if hasattr(models, 'models') else []
        has_model = any("llama3.2" in m for m in available)
        return {"running": True, "has_model": has_model, "models": available}
    except Exception:
        return {"running": False, "has_model": False, "models": []}


def check_server_status(port: int, name: str):
    """Check if a server is running on a given port."""
    import httpx
    try:
        resp = httpx.get(f"http://localhost:{port}/.well-known/agent.json", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def render_step_card(step: dict):
    """Render a step from the execution trace."""
    action_type = step.get("action_type", "unknown")
    target = step.get("target", "unknown")
    reason = step.get("reason", "")
    confidence = step.get("confidence", 0)
    step_id = step.get("step_id", 0)
    elapsed = step.get("elapsed_s", 0)
    success = step.get("success", True)
    validation = step.get("validation", {})
    val_grade = validation.get("quality_grade", "")

    css_class = {
        "agent_call": "agent",
        "tool_call": "tool",
        "finish": "finish",
    }.get(action_type, "error")

    if not success:
        css_class = "error"

    icon = {"agent_call": "🤖", "tool_call": "🔧", "finish": "✅"}.get(action_type, "❓")
    status_icon = "✅" if success else "❌"
    grade_html = f' | Validation: <b>{val_grade}</b>' if val_grade else ""
    time_html = f' | {elapsed}s' if elapsed else ""

    st.markdown(f"""
    <div class="step-card {css_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:700;color:var(--text-primary);">
                {icon} Step {step_id}: {action_type.upper()} → {target} {status_icon}
            </span>
            <span style="font-size:0.75rem;color:var(--text-muted);">
                confidence: {confidence:.0%}{grade_html}{time_html}
            </span>
        </div>
        <div style="color:var(--text-secondary);font-size:0.8rem;">{reason}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">🧠</div>
        <div style="font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.02em;">
            AutoPrepAI v5
        </div>
        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 4px;">
            100% Real LLM Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech badges
    st.markdown("""
    <div style="text-align:center;margin-bottom:16px;">
        <span class="tech-badge">🦙 Ollama</span>
        <span class="tech-badge">📡 FastMCP</span>
        <span class="tech-badge">🤝 A2A SDK</span>
    </div>
    """, unsafe_allow_html=True)

    # ── System Status ──
    st.markdown("### 🔌 System Status")

    ollama_status = check_ollama_status()
    if ollama_status["running"] and ollama_status["has_model"]:
        st.markdown('<div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#10b981;">✅ Ollama — llama3.2 ready</div>', unsafe_allow_html=True)
    elif ollama_status["running"]:
        st.markdown('<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#f59e0b;">⚠️ Ollama running — run: ollama pull llama3.2</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:rgba(244,63,94,0.1);border:1px solid rgba(244,63,94,0.3);border-radius:8px;padding:8px 12px;font-size:0.8rem;color:#f43f5e;">❌ Ollama not running — run: ollama serve</div>', unsafe_allow_html=True)

    # Agent status (all 9)
    all_agents = [
        ("MCP Server", 8100), ("DataUnderstanding", 8201), ("DataQuality", 8202),
        ("MissingValue", 8203), ("Encoding", 8204), ("FeatureEng", 8205),
        ("AutoML", 8206), ("EDA", 8207), ("Reflection", 8208), ("Report", 8209),
    ]
    agent_statuses = []
    for name, port in all_agents:
        ok = check_server_status(port, name)
        icon = "🟢" if ok else "🔴"
        agent_statuses.append(f"{icon} {name}")
    st.markdown('<div style="font-size:0.72rem;color:var(--text-muted);margin-top:8px;">' + ' &nbsp;|&nbsp; '.join(agent_statuses) + '</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── File Upload ──
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Drop your CSV, Excel, or JSON",
        type=["csv", "xlsx", "xls", "json"],
    )

    if uploaded_file:
        try:
            # ── Robustness: file validation ──
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 50:
                st.error(f"❌ File too large ({file_size_mb:.1f} MB). Maximum is 50 MB.")
                st.stop()

            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            elif name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError(f"Unsupported: {uploaded_file.name}")

            if df.shape[1] > 200:
                st.error(f"❌ Too many columns ({df.shape[1]}). Maximum is 200.")
                st.stop()
            if df.shape[0] < 5:
                st.error(f"❌ Too few rows ({df.shape[0]}). Minimum is 5.")
                st.stop()

            st.session_state.uploaded_df = df
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"✅ **{df.shape[0]:,}** rows × **{df.shape[1]}** cols ({file_size_mb:.1f} MB)")

            # Target column
            st.markdown("### 🎯 Target Column")
            target_options = ["Auto-detect (LLM)"] + list(df.columns)
            target_choice = st.selectbox("Select target", target_options)
            st.session_state["target_choice"] = None if target_choice == "Auto-detect (LLM)" else target_choice

            st.markdown("### 👀 Preview")
            st.dataframe(df.head(5), width=1200, height=200)

        except Exception as e:
            st.error(f"❌ {str(e)}")
            st.session_state.uploaded_df = None

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 10px; font-size: 0.7rem; color: #4b5563;">
        No API keys needed<br>
        100% local LLM via Ollama<br>
        Real MCP + A2A protocols
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
if st.session_state.uploaded_df is None:
    # ── Landing page ──
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🦙 OLLAMA • 📡 FASTMCP • 🤝 A2A SDK</div>
        <div class="hero-title">AutoPrepAI</div>
        <div class="hero-subtitle">
            A 100% real LLM-powered multi-agent pipeline.
            Every decision made by a local LLM. Every agent communicates via
            real A2A protocol. Every tool served via real MCP.
            Zero hardcoded logic.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🦙", "Local LLM", "Ollama llama3.2 — no API keys, no cloud, no cost. Runs on your CPU."),
        ("📡", "Real MCP", "FastMCP server with 6 tools. Real JSON-RPC over HTTP. Not a mock."),
        ("🤝", "Real A2A", "6 agent servers with real Agent Cards. Agent-to-agent HTTP calls."),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border-subtle);
                 border-radius:var(--radius-sm);padding:22px;text-align:center;transition:all 0.3s;">
                <div style="font-size:2rem;margin-bottom:8px;">{icon}</div>
                <div style="font-weight:700;color:var(--text-primary);margin-bottom:4px;">{title}</div>
                <div style="color:var(--text-muted);font-size:0.8rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <br><div style="text-align:center;color:var(--text-muted);font-size:0.9rem;">
        👈 Upload a dataset in the sidebar to begin
    </div>
    """, unsafe_allow_html=True)

else:
    df = st.session_state.uploaded_df
    filename = st.session_state.get("uploaded_filename", "dataset")
    result = st.session_state.get("pipeline_result")

    if result is None:
        # ── Pre-run view ──
        st.markdown("# 🚀 Ready to Process")
        st.markdown(f"**{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", f"{df.shape[0]:,}")
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", f"{df.isnull().sum().sum():,}")
        col4.metric("Duplicates", f"{df.duplicated().sum():,}")
        col5.metric("Memory", f"{df.memory_usage(deep=True).sum()/(1024*1024):.1f} MB")

        st.markdown("---")

        st.markdown("### 🏗️ Pipeline Architecture")
        st.markdown("""
        ```
        ┌──────────────────────────────────────────────────────────────┐
        │              ORCHESTRATOR (LLM Brain) + Memory              │
        │  Dynamic Discovery │ Reflection Loop │ Report Generation    │
        └─────┬──────────────────────────────────┬───────────────────┘
              │ A2A (JSON-RPC)                   │ MCP (FastMCP)
        ┌─────┴────────────────┐          ┌──────┴──────────────┐
        │  9 A2A Agents        │          │  8 MCP Tools        │
        │  (ports 8201-8209)   │          │  (port 8100)        │
        │                      │          │                     │
        │  DataUnderstanding   │  ←A2A→   │  describe_data      │
        │  DataQuality         │          │  check_missing      │
        │  MissingValue        │  ←A2A→   │  correlation        │
        │  Encoding            │          │  detect_outliers    │
        │  FeatureEngineering  │  ←A2A→   │  distribution       │
        │  AutoML (+XGBoost)   │          │  train_model        │
        │  EDA (NEW)           │  ←A2A→   │  evaluate_model     │
        │  Reflection (NEW)    │          │  pipeline_status    │
        │  Report (NEW)        │          │                     │
        └──────────────────────┘          └─────────────────────┘
        ```
        """)

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("🚀  Run Autonomous Pipeline", type="primary"):
                target = st.session_state.get("target_choice", None)

                # Save CSV for orchestrator (as input_data.csv, orchestrator copies to current_data.csv)
                work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir")
                os.makedirs(work_dir, exist_ok=True)
                csv_path = os.path.join(work_dir, "input_data.csv")
                df.to_csv(csv_path, index=False)

                progress = st.progress(0, text="🧠 Initializing orchestrator...")
                trace_container = st.container()

                try:
                    from orchestrator import AutonomousOrchestrator
                    import asyncio

                    orchestrator = AutonomousOrchestrator(model="llama3.2")

                    step_count = [0]
                    def on_step(step_data):
                        event = step_data.get("event", "")
                        if event == "step_decided":
                            step_count[0] += 1
                            step = step_data.get("decision", {})
                            pct = min(step_count[0] / 10, 0.95)
                            progress.progress(pct, text=f"Step {step_count[0]}: {step.get('target', '...')}")

                    orchestrator.on_step(on_step)

                    with st.spinner("🧠 LLM is orchestrating the pipeline autonomously..."):
                        result = asyncio.run(orchestrator.run(csv_path, target_column=target))

                    progress.progress(1.0, text="✅ Pipeline complete!")

                    st.session_state.pipeline_result = result
                    st.session_state.execution_trace = result.get("execution_trace", [])
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Pipeline failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    else:
        # ── Results view ──
        st.markdown("# ✅ Pipeline Complete")

        trace = result.get("execution_trace", [])
        state_data = result.get("state", {})
        llm_status = result.get("llm_status", {})

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Steps", result.get("total_steps", 0))
        c2.metric("LLM Calls", llm_status.get("total_calls", 0))
        c3.metric("Success Rate", f"{llm_status.get('success_rate', 0)}%")
        c4.metric("Agents", len(result.get("discovered_agents", [])))
        c5.metric("Tools", len(result.get("discovered_tools", [])))

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🧠 Trace", "📊 Results", "🔬 EDA",
            "🔄 Reflection", "📋 Report", "📥 Download"
        ])

        with tab1:
            st.markdown("### Step-by-Step LLM Decisions")
            st.markdown("*Every step decided by LLM. Agents discovered dynamically.*")
            # Show discovered agents/tools
            disc_agents = result.get("discovered_agents", [])
            disc_tools = result.get("discovered_tools", [])
            if disc_agents:
                st.markdown(f"**Discovered agents:** {', '.join(disc_agents)}")
            if disc_tools:
                st.markdown(f"**Discovered tools:** {', '.join(disc_tools)}")
            st.markdown("---")

            # ── Observability: Agent Latency Chart ──
            agent_steps = [t for t in trace if t.get('action_type') == 'agent_call']
            if agent_steps:
                st.markdown("#### ⏱️ Agent Latency")
                fig, ax = plt.subplots(figsize=(10, 4))
                names = [t.get('target', '?')[:15] for t in agent_steps]
                times = [t.get('elapsed_s', 0) for t in agent_steps]
                successes = [t.get('success', False) for t in agent_steps]
                colors = ['#10b981' if s else '#f43f5e' for s in successes]
                bars = ax.barh(names, times, color=colors, edgecolor='#ffffff22')
                ax.set_xlabel('Time (seconds)', fontsize=11)
                ax.set_title('Agent Execution Time', fontsize=13, fontweight='bold')
                for bar, t in zip(bars, times):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                           f'{t:.1f}s', va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Observability table
                st.markdown("#### 📊 Agent Breakdown")
                obs_data = []
                for t in agent_steps:
                    preview = t.get('result_preview', '')
                    llm_calls = 0
                    import re
                    m = re.search(r'"llm_calls":\s*(\d+)', preview)
                    if m:
                        llm_calls = int(m.group(1))
                    obs_data.append({
                        'Agent': t.get('target', '?'),
                        'Status': '✅' if t.get('success') else '❌',
                        'Time (s)': t.get('elapsed_s', 0),
                        'LLM Calls': llm_calls,
                        'Confidence': f"{t.get('confidence', 0):.0%}",
                        'Protocol': 'A2A',
                    })
                st.dataframe(pd.DataFrame(obs_data), width=1200, hide_index=True)

            st.markdown("---")
            if trace:
                for step in trace:
                    render_step_card(step)
            else:
                st.info("No execution trace available.")

        with tab2:
            st.markdown("### Pipeline Results")
            # Load AutoML results from disk
            automl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir", "automl_results.json")
            if os.path.exists(automl_path):
                import json as _json
                with open(automl_path) as _f:
                    automl_data = _json.load(_f)
                best = automl_data.get("best_model", {})
                if best:
                    st.markdown(f"### 🏆 Best Model: **{best.get('best_model', 'N/A')}**")
                    st.markdown(f"*{best.get('justification', '')}*")
                    if best.get('warnings'):
                        for w in best['warnings']:
                            st.warning(w)
                all_results = automl_data.get("all_results", [])
                if all_results:
                    results_df = pd.DataFrame([{"Model": r["model"], **r["metrics"]} for r in all_results])
                    st.dataframe(results_df, width=1200, hide_index=True)
            else:
                st.info("No AutoML results yet.")

        with tab3:
            st.markdown("### 🔬 EDA Report")
            eda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir", "eda_report.json")
            if os.path.exists(eda_path):
                import json as _json
                with open(eda_path) as _f:
                    eda_data = _json.load(_f)
                interp = eda_data.get("interpretation", {})
                if interp:
                    health = interp.get("overall_data_health", "?")
                    health_color = {"good": "#10b981", "fair": "#f59e0b", "poor": "#f43f5e", "critical": "#dc2626"}.get(health, "#6b7280")
                    st.markdown(f'<div style="font-size:1.5rem;font-weight:700;color:{health_color};">Data Health: {health.upper()}</div>', unsafe_allow_html=True)
                    for action in interp.get("top_priority_actions", []):
                        st.markdown(f"- ⚡ {action}")
                    st.markdown("---")
                    for finding in interp.get("findings", []):
                        sev = finding.get("severity", "low")
                        sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")
                        st.markdown(f"{sev_icon} **[{finding.get('category', '')}]** {finding.get('description', '')}")
                        if finding.get('recommendation'):
                            st.markdown(f"   → {finding['recommendation']}")
                else:
                    st.json(eda_data.get("eda_facts", {}))

                # ── Visual EDA Charts ──
                st.markdown("---")
                st.markdown("### 📈 Visual Analysis")
                workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir")
                chart_files = [
                    ("eda_correlation.png", "Correlation Heatmap"),
                    ("eda_missing.png", "Missing Values"),
                    ("eda_skewness.png", "Feature Skewness"),
                    ("eda_imbalance.png", "Class Imbalance"),
                ]
                col_a, col_b = st.columns(2)
                for i, (fname, title) in enumerate(chart_files):
                    fpath = os.path.join(workdir, fname)
                    if os.path.exists(fpath):
                        with [col_a, col_b][i % 2]:
                            st.markdown(f"**{title}**")
                            st.image(fpath, use_container_width=True)
            else:
                st.info("No EDA report generated.")

        with tab4:
            st.markdown("### 🔄 Reflection & Self-Healing")
            refl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir", "reflection_log.json")
            if os.path.exists(refl_path):
                import json as _json
                with open(refl_path) as _f:
                    refl_data = _json.load(_f)
                st.markdown(f"**Root Cause:** {refl_data.get('root_cause', 'N/A')}")
                st.markdown(f"**Diagnosis:** {refl_data.get('diagnosis', 'N/A')}")
                st.markdown(f"**Proposed Fix:** {refl_data.get('proposed_fix', 'N/A')}")
                st.markdown(f"**Fix Target Agent:** `{refl_data.get('fix_target_agent', 'N/A')}`")
                st.markdown(f"**Should Retry:** {'Yes' if refl_data.get('should_retry') else 'No'}")
                st.markdown(f"**Confidence:** {refl_data.get('confidence', 0):.0%}")
            elif state_data.get("reflection_done"):
                st.success("Reflection completed — no log file found.")
            else:
                st.info("No reflection needed — model performance was acceptable.")

        with tab5:
            st.markdown("### 📋 Final Report")
            report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir", "final_report.json")
            if os.path.exists(report_path):
                import json as _json
                with open(report_path) as _f:
                    report_data = _json.load(_f)
                narrative = report_data.get("narrative", {})
                if narrative:
                    st.markdown(f"## {narrative.get('title', 'Pipeline Report')}")
                    st.markdown(f"**Overall Confidence:** {narrative.get('overall_confidence', 0):.0%}")
                    for section in narrative.get("sections", []):
                        st.markdown(f"### {section.get('title', '')}")
                        st.markdown(section.get("content", ""))
                        for kp in section.get("key_points", []):
                            st.markdown(f"- {kp}")
                    if narrative.get("risks"):
                        st.markdown("### ⚠️ Risks")
                        for r in narrative["risks"]:
                            st.markdown(f"- {r}")
                    if narrative.get("recommendations"):
                        st.markdown("### 💡 Recommendations")
                        for r in narrative["recommendations"]:
                            st.markdown(f"- {r}")
                else:
                    st.json(report_data)

                # ── Report Download Buttons ──
                st.markdown("---")
                st.markdown("### 📥 Export Report")
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir")

                md_path = os.path.join(workdir, "final_report.md")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as _f:
                        md_data = _f.read()
                    with dl_col1:
                        st.download_button("📄 Download Markdown", md_data.encode("utf-8"),
                                          "autoprep_report.md", "text/markdown")

                html_path = os.path.join(workdir, "final_report.html")
                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as _f:
                        html_data = _f.read()
                    with dl_col2:
                        st.download_button("🌐 Download HTML", html_data.encode("utf-8"),
                                          "autoprep_report.html", "text/html")

                with dl_col3:
                    st.download_button("📊 Download JSON", json.dumps(report_data, indent=2, default=str).encode(),
                                      "autoprep_report.json", "application/json")
            else:
                st.info("No report generated yet.")

        with tab6:
            st.markdown("### Download Processed Data")
            processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir", "current_data.csv")
            if os.path.exists(processed_path):
                processed_df = pd.read_csv(processed_path)
                st.markdown(f"Shape: **{processed_df.shape[0]:,} rows × {processed_df.shape[1]} columns**")
                csv_data = processed_df.to_csv(index=False).encode()
                st.download_button("📥 Download Processed CSV", csv_data, "autoprep_processed.csv", "text/csv")
                st.markdown("### Preview")
                st.dataframe(processed_df.head(20), width=1200)
            # Memory summary
            mem_summary = result.get("memory_summary", "")
            if mem_summary:
                with st.expander("🧠 Pipeline Memory"):
                    st.code(mem_summary)
            # Raw JSON
            with st.expander("⚙️ Full Result JSON"):
                st.json(result)

        st.markdown("---")
        if st.button("🔄 Reset & Run Again"):
            st.session_state.pipeline_result = None
            st.session_state.execution_trace = []
            st.rerun()
