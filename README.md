<p align="center">
  <img src="assets/banner.png" alt="AutoPrepAI" width="100%"/>
</p>

<h1 align="center">🧠 AutoPrepAI v5</h1>
<h3 align="center">Autonomous Multi-Agent Data Science Pipeline</h3>

<p align="center">
  <em>Upload a dataset → LLM orchestrates 9 agents → Get a trained ML model + full report</em><br>
  <em>100% local • No API keys • Runs on CPU via Ollama llama3.2</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LLM-Ollama_llama3.2-FF6F00?style=flat-square"/>
  <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Protocol-A2A_+_MCP-6366f1?style=flat-square"/>
</p>

---

## What is AutoPrepAI?

AutoPrepAI is a fully autonomous data science pipeline where **every decision is made by a local LLM** (Ollama llama3.2). You upload a raw CSV and the system automatically cleans, analyzes, engineers features, trains 6 ML models, and delivers a full explainability report — all without writing a single line of code.

**9 specialized AI agents** communicate through real protocols (A2A + MCP) and are orchestrated by an LLM brain. If model performance is poor, a **Reflection Agent** diagnoses the issue and self-heals the pipeline automatically.

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="AutoPrepAI Architecture" width="85%"/>
</p>

```mermaid
graph TB
    UI["🖥️ Streamlit UI<br/>Premium Dark Theme"]
    ORCH["🧠 Orchestrator<br/>LLM Brain + Memory + Discovery"]
    
    UI --> ORCH
    
    subgraph A2A["🤝 A2A Agent Servers (ports 8201–8209)"]
        DU["📋 DataUnderstanding"]
        DQ["🔍 DataQuality"]
        EDA["📈 EDA"]
        MV["🩹 MissingValue"]
        ENC["🏷️ Encoding"]
        FE["⚙️ FeatureEngineering"]
        AML["🏆 AutoML"]
        REF["🔄 Reflection"]
        REP["📝 Report"]
    end

    subgraph MCP["📡 MCP Tools Server (port 8100)"]
        T1["describe_data"]
        T2["check_missing"]
        T3["correlation_analysis"]
        T4["detect_outliers"]
        T5["distribution_analysis"]
        T6["train_model"]
        T7["evaluate_model"]
        T8["pipeline_status"]
    end

    ORCH -- "A2A Protocol<br/>JSON-RPC over HTTP" --> A2A
    ORCH -- "MCP Protocol<br/>FastMCP" --> MCP
```

---

## How It Works

The pipeline runs in **7 autonomous stages**. The LLM decides **what** to do — deterministic code (pandas/sklearn) handles execution.

```mermaid
flowchart TD
    START(["📂 Upload CSV / Excel / JSON"])
    
    S1["📋 Step 1: Data Understanding<br/>Profile columns, detect target, classify types"]
    S2["🔍 Step 2: Data Quality<br/>Grade quality A–F, flag risky columns"]
    S3["📈 Step 3: EDA<br/>Outliers, correlations, skewness, charts"]
    S4["🩹 Step 4: Missing Values<br/>LLM picks per-column strategy: median, mode, ffill..."]
    S5["🏷️ Step 5: Encoding<br/>LLM picks per-column method: onehot, label, tfidf..."]
    S6["⚙️ Step 6: Feature Engineering<br/>LLM recommends: log, sqrt, interactions, binning"]
    S7["🏆 Step 7: AutoML<br/>Train 6 models, LLM selects the best"]
    
    CHECK{"Score OK?"}
    REFLECT["🔄 Reflection Agent<br/>Diagnose root cause → Fix → Retry"]
    REPORT["📝 Generate Final Report<br/>Markdown + HTML + JSON"]
    DONE(["✅ Pipeline Complete<br/>Processed CSV + Model + Report"])

    START --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> CHECK
    CHECK -- "✅ Yes" --> REPORT --> DONE
    CHECK -- "❌ No" --> REFLECT --> S7

    style START fill:#6366f1,stroke:#818cf8,color:#fff
    style S7 fill:#f59e0b,stroke:#fbbf24,color:#000
    style REFLECT fill:#f43f5e,stroke:#fb7185,color:#fff
    style DONE fill:#10b981,stroke:#34d399,color:#fff
    style REPORT fill:#06b6d4,stroke:#22d3ee,color:#000
```

### Agent Collaboration

Agents talk to each other in real-time using the A2A protocol:

```mermaid
graph LR
    EDA["📈 EDAAgent"] -- "A2A: quality risks?" --> DQ["🔍 DataQualityAgent"]
    AML["🏆 AutoMLAgent"] -- "A2A: feature risks?" --> EDA2["📈 EDAAgent"]
    REF["🔄 ReflectionAgent"] -- "A2A: model weaknesses?" --> AML2["🏆 AutoMLAgent"]

    style EDA fill:#6366f1,color:#fff
    style DQ fill:#06b6d4,color:#fff
    style AML fill:#f59e0b,color:#000
    style EDA2 fill:#6366f1,color:#fff
    style REF fill:#f43f5e,color:#fff
    style AML2 fill:#f59e0b,color:#000
```

### Dataset Versioning

Every successful step saves a snapshot — you can inspect or rollback any stage:

| Stage | File | Description |
|:---:|---|---|
| 0 | `stage_0_raw.csv` | Your original data |
| 3 | `stage_3_missing_fixed.csv` | After imputation |
| 4 | `stage_4_encoded.csv` | After encoding |
| 6 | `stage_6_final.csv` | ML-ready data |
| — | `automl_results.json` | All model metrics |
| — | `eda_report.json` | EDA findings + charts |
| — | `final_report.md` | Report (Markdown) |
| — | `final_report.html` | Report (HTML) |
| — | `pipeline_memory.json` | Full execution log |

---

## How to Run

### Prerequisites

| Tool | Purpose | Install |
|---|---|---|
| **Python 3.10+** | Runtime | [python.org](https://python.org) |
| **Ollama** | Local LLM | [ollama.com](https://ollama.com) |

### Step 1 — Install Dependencies

```bash
git clone https://github.com/yourusername/AutoPrepAI.git
cd AutoPrepAI

python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Step 2 — Setup Ollama

```bash
ollama serve                    # Start Ollama (if not running)
ollama pull llama3.2            # Download model (~2GB, one-time)
```

### Step 3 — Start All Servers

```bash
python start_servers.py
```

> Wait until you see `[OK] 10 servers started successfully`

### Step 4 — Launch the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** → Upload your CSV → Click **"Run Autonomous Pipeline"** → Done! 🎉

---

## Project Structure

```
AutoPrepAI/
├── app.py                      # Streamlit UI
├── orchestrator.py             # LLM orchestrator brain
├── discovery.py                # Dynamic agent/tool discovery
├── memory.py                   # Pipeline memory
├── start_servers.py            # One-command server launcher
├── requirements.txt            # Dependencies
│
├── llm/
│   └── ollama_engine.py        # Ollama LLM engine + Pydantic schemas
│
├── mcp_server/
│   └── server.py               # FastMCP server (8 data tools)
│
├── a2a_agents/
│   ├── base.py                 # Shared agent utilities
│   ├── data_understanding.py   # Dataset profiling
│   ├── data_quality.py         # Quality assessment
│   ├── eda.py                  # EDA + visual charts
│   ├── missing_values.py       # Imputation
│   ├── encoding.py             # Encoding
│   ├── feature_engineering.py  # Feature engineering
│   ├── automl.py               # AutoML + model selection
│   ├── reflection.py           # Self-healing reflection
│   └── report.py               # Explainability report
│
└── workdir/                    # Runtime artifacts (auto-generated)
```

---

<p align="center">
  <strong>Built with 💜 using Ollama • FastMCP • A2A SDK • Streamlit</strong>
</p>
