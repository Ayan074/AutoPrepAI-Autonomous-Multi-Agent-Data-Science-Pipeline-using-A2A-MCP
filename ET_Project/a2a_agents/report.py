"""
A2A Agent: ReportAgent (port 8209)

Skills:
  - generate_report: Final explainable report with LLM narrative

A2A Collaboration:
  - Reads saved artifacts from all other agents
  - LLM generates structured narrative
"""

from __future__ import annotations

import json, logging, os
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import get_llm, read_current_data, get_data_path, ExecutionLog
from llm.ollama_engine import ReportNarrative

logger = logging.getLogger(__name__)

AGENT_CARD = {
    "name": "ReportAgent",
    "description": "Generates final explainable pipeline report with LLM narrative",
    "url": "http://localhost:8209/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {"id": "generate_report", "name": "Generate Report",
         "description": "Create explainable report: preprocessing, EDA, models, risks",
         "tags": ["report", "explainability"], "examples": ["Generate the final report"]},
    ],
}


def _load_json(filename: str) -> dict:
    """Load a JSON artifact from the working directory."""
    path = get_data_path(filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


async def execute_report(execution_trace: list = None,
                         memory_data: dict = None) -> dict:
    """Generate final explainable report."""
    llm = get_llm()
    log = ExecutionLog("ReportAgent")

    df = read_current_data()

    # Load all artifacts
    profile = _load_json("dataset_profile.json")
    quality = _load_json("quality_assessment.json")
    imputation = _load_json("imputation_log.json")
    encoding = _load_json("encoding_log.json")
    fe_log = _load_json("feature_engineering_log.json")
    eda = _load_json("eda_report.json")
    automl = _load_json("automl_results.json")
    reflection = _load_json("reflection_log.json")

    # Build context summary
    context = {
        "dataset_shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
        "target": profile.get("target_column", "unknown"),
        "domain": profile.get("detected_domain", "unknown"),
        "quality_grade": quality.get("assessment", {}).get("quality_grade", "?"),
        "imputation_strategies": len(imputation.get("strategies", [])),
        "encoding_applied": len(encoding) if isinstance(encoding, list) else 0,
        "features_engineered": len(fe_log) if isinstance(fe_log, list) else 0,
        "eda_health": eda.get("interpretation", {}).get("overall_data_health", "?"),
        "models_trained": len(automl.get("all_results", [])),
        "best_model": automl.get("best_model", {}).get("best_model", "none"),
        "best_justification": automl.get("best_model", {}).get("justification", ""),
        "reflection_done": bool(reflection),
        "reflection_cause": reflection.get("root_cause", ""),
        "total_steps": len(execution_trace) if execution_trace else 0,
    }

    # Model metrics summary
    model_lines = []
    for r in automl.get("all_results", []):
        m = r.get("metrics", {})
        model_lines.append(f"  {r['model']}: {json.dumps(m)}")

    prompt = f"""Generate a final explainability report for this ML pipeline.

PIPELINE SUMMARY:
{json.dumps(context, indent=2)}

MODEL RESULTS:
{chr(10).join(model_lines) if model_lines else "No models trained."}

PREPROCESSING:
- Imputation: {json.dumps(imputation.get("strategies", [])[:5], default=str)}
- Encoding: {json.dumps(encoding[:5] if isinstance(encoding, list) else [], default=str)}

EDA FINDINGS:
{json.dumps(eda.get("interpretation", {}).get("findings", [])[:5], default=str)}

REFLECTION:
{json.dumps(reflection, default=str) if reflection else "No reflection needed."}

Generate a structured report with these sections:
1. Data Overview — describe the dataset and domain
2. Preprocessing Decisions — what was done and why
3. EDA Insights — key statistical findings
4. Model Comparison — all models, why best was chosen
5. Risks & Limitations — what could go wrong
6. Recommendations — next steps

Each section needs: title, content paragraph, key_points list.
Also provide overall_confidence (0-1), risks list, recommendations list."""

    narrative = llm.decide(prompt, ReportNarrative)
    if narrative is None:
        log.log("llm_failed", "LLM unavailable", "FAILED")
        return {"error": "LLM unavailable", "context": context, "log": log.to_dict()}

    log.log("report_complete",
            f"Generated {len(narrative.get('sections', []))} sections. "
            f"Confidence: {narrative.get('overall_confidence', 0):.0%}",
            "LLM")

    report = {"narrative": narrative, "context": context, "raw_artifacts": {
        "profile": profile, "quality": quality, "automl": automl,
    }}

    with open(get_data_path("final_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Generate Markdown Report ──
    md_lines = [f"# {narrative.get('title', 'AutoPrepAI Pipeline Report')}\n"]
    md_lines.append(f"**Overall Confidence:** {narrative.get('overall_confidence', 0):.0%}\n")
    md_lines.append(f"**Dataset:** {context.get('dataset_shape', '?')} | **Target:** {context.get('target', '?')} | **Domain:** {context.get('domain', '?')}\n")
    md_lines.append(f"**Best Model:** {context.get('best_model', 'N/A')}\n")
    md_lines.append("---\n")

    for section in narrative.get("sections", []):
        md_lines.append(f"## {section.get('title', '')}\n")
        md_lines.append(f"{section.get('content', '')}\n")
        for kp in section.get("key_points", []):
            md_lines.append(f"- {kp}")
        md_lines.append("")

    if narrative.get("risks"):
        md_lines.append("## ⚠️ Risks\n")
        for r in narrative["risks"]:
            md_lines.append(f"- {r}")
        md_lines.append("")

    if narrative.get("recommendations"):
        md_lines.append("## 💡 Recommendations\n")
        for r in narrative["recommendations"]:
            md_lines.append(f"- {r}")
        md_lines.append("")

    # Model comparison table
    all_results = automl.get("all_results", [])
    if all_results:
        md_lines.append("## 📊 Model Comparison\n")
        metrics_keys = list(all_results[0].get("metrics", {}).keys()) if all_results else []
        header = "| Model | " + " | ".join(metrics_keys) + " |"
        sep = "|---|" + "|".join(["---"] * len(metrics_keys)) + "|"
        md_lines.append(header)
        md_lines.append(sep)
        for r in all_results:
            m = r.get("metrics", {})
            row = f"| {r['model']} | " + " | ".join([f"{m.get(k, 'N/A')}" for k in metrics_keys]) + " |"
            md_lines.append(row)
        md_lines.append("")

    md_lines.append("\n---\n*Generated by AutoPrepAI v5 — Autonomous Multi-Agent Pipeline*")

    md_content = "\n".join(md_lines)
    with open(get_data_path("final_report.md"), "w", encoding="utf-8") as f:
        f.write(md_content)

    # ── Generate HTML Report ──
    html_content = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{narrative.get('title', 'AutoPrepAI Report')}</title>
<style>
body {{ font-family: 'Segoe UI', system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #0f172a; color: #e2e8f0; line-height: 1.7; }}
h1 {{ color: #818cf8; border-bottom: 2px solid #6366f1; padding-bottom: 12px; }}
h2 {{ color: #a5b4fc; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #334155; padding: 10px 14px; text-align: left; }}
th {{ background: #1e293b; color: #818cf8; }}
tr:nth-child(even) {{ background: #1e293b; }}
ul {{ padding-left: 24px; }}
li {{ margin-bottom: 6px; }}
.badge {{ display: inline-block; background: #6366f1; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85rem; margin-right: 8px; }}
.risk {{ color: #f43f5e; }}
.rec {{ color: #10b981; }}
</style></head><body>
"""
    # Convert markdown to simple HTML
    for line in md_content.split("\n"):
        if line.startswith("# "):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith("## "):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith("- "):
            html_content += f"<li>{line[2:]}</li>\n"
        elif line.startswith("|"):
            # Table handling
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(c.replace("-", "") == "" for c in cells):
                continue  # skip separator
            tag = "th" if line == header else "td"
            html_content += "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>\n"
        elif line.startswith("**"):
            html_content += f"<p><strong>{line.replace('**', '')}</strong></p>\n"
        elif line.strip() == "---":
            html_content += "<hr>\n"
        elif line.strip():
            html_content += f"<p>{line}</p>\n"

    html_content += "</body></html>"
    with open(get_data_path("final_report.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    log.log("exports_generated", "Generated MD + HTML report exports", "DETERMINISTIC")

    return {**report, "log": log.to_dict()}


async def agent_card_endpoint(request: Request) -> JSONResponse:
    return JSONResponse(AGENT_CARD)

async def message_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    params = body.get("params", {})
    msg = params.get("message", {})
    parts = msg.get("parts", [])
    text = ""
    for part in parts:
        if part.get("type") == "text":
            text = part["text"]; break

    trace = None
    memory = None
    try:
        d = json.loads(text)
        trace = d.get("execution_trace")
        memory = d.get("memory")
    except (json.JSONDecodeError, TypeError):
        pass

    result = await execute_report(execution_trace=trace, memory_data=memory)
    return JSONResponse({"jsonrpc": "2.0", "id": body.get("id", 1),
        "result": {"artifacts": [{"parts": [{"type": "text",
            "text": json.dumps(result, default=str)}]}]}})

app = Starlette(routes=[
    Route("/.well-known/agent.json", agent_card_endpoint),
    Route("/", message_endpoint, methods=["POST"]),
])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8209)
