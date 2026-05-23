"""
A2A Agent: ReflectionAgent (port 8208)

Skills:
  - analyze_failure: Diagnose why model performance is poor
  - suggest_strategy: Propose a concrete fix + which agent to re-run

ONE reflection loop only. Not recursive.

A2A Collaboration:
  - Calls DataQualityAgent for current data quality state
"""

from __future__ import annotations

import asyncio
import json, logging, os
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import get_llm, read_current_data, get_data_path, call_a2a_agent, ExecutionLog
from llm.ollama_engine import ReflectionAnalysis

logger = logging.getLogger(__name__)

AGENT_CARD = {
    "name": "ReflectionAgent",
    "description": "Analyzes pipeline failures and proposes retry strategies",
    "url": "http://localhost:8208/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {"id": "analyze_failure", "name": "Analyze Failure",
         "description": "Diagnose poor model performance and propose a fix",
         "tags": ["reflection", "diagnosis"], "examples": ["Model accuracy is 0.45, why?"]},
    ],
}


async def execute_reflection(model_results=None, execution_trace=None,
                             failure_history=None, target_column=None,
                             problem_type=None) -> dict:
    """Diagnose pipeline failure and propose ONE fix."""
    llm = get_llm()
    log = ExecutionLog("ReflectionAgent")
    df = read_current_data()

    # A2A: Ask AutoMLAgent for model weaknesses (REAL collaboration)
    model_weaknesses = ""
    try:
        log.log("a2a_request", "Querying AutoMLAgent for model weaknesses...", "A2A")
        model_response = await asyncio.wait_for(
            call_a2a_agent(
                "AutoMLAgent",
                "Provide top model weaknesses and overfitting risks."
            ),
            timeout=120.0,
        )
        if isinstance(model_response, dict):
            best = model_response.get("best_model", {})
            all_results = model_response.get("all_results", [])
            if best:
                model_weaknesses = f"Best model: {best.get('best_model', '?')}, warnings: {best.get('warnings', [])}"
            log.log("a2a_received", f"AutoMLAgent: {model_weaknesses}", "A2A")
    except (asyncio.TimeoutError, Exception) as e:
        log.log("a2a_fallback", f"AutoMLAgent A2A failed ({e}), using disk data", "CACHE")

    # Read quality assessment from disk (fallback context)
    quality_info = {}
    try:
        quality_path = get_data_path("quality_assessment.json")
        if os.path.exists(quality_path):
            with open(quality_path) as f:
                quality_data = json.load(f)
            quality_info = quality_data.get("assessment", {})
            log.log("quality_loaded", f"Quality grade: {quality_info.get('quality_grade', '?')}", "CACHE")
    except Exception:
        pass

    # Load AutoML results if not provided
    if model_results is None:
        automl_path = get_data_path("automl_results.json")
        if os.path.exists(automl_path):
            with open(automl_path) as f:
                model_results = json.load(f)
        else:
            model_results = {}

    all_results = model_results.get("all_results", [])
    best_model = model_results.get("best_model", {})
    perf_lines = []
    for r in all_results:
        m = r.get("metrics", {})
        primary = m.get("f1_weighted") or m.get("r2") or 0
        perf_lines.append(f"  {r['model']}: {primary:.4f}")

    remaining_missing = int(df.isnull().sum().sum())
    remaining_cat = len(df.select_dtypes(include="object").columns)
    n_features = len(df.select_dtypes(include="number").columns)

    issues = []
    if remaining_missing > 0: issues.append(f"{remaining_missing} missing values remain")
    if remaining_cat > 0: issues.append(f"{remaining_cat} unencoded categorical columns")
    if n_features < 3: issues.append(f"Only {n_features} numeric features")
    if len(df) < 50: issues.append(f"Very small dataset ({len(df)} rows)")

    prompt = f"""You are the ReflectionAgent. The pipeline completed but performance may be poor.

MODEL RESULTS:
{chr(10).join(perf_lines) if perf_lines else "No models trained."}
Best: {best_model.get('best_model', 'unknown')}

DATA STATE: {df.shape[0]}×{df.shape[1]}, missing={remaining_missing}, categorical={remaining_cat}, numeric_features={n_features}
Quality grade: {quality_info.get('quality_grade', '?')}
Issues: {json.dumps(issues)}
Failures: {json.dumps(failure_history[:3] if failure_history else [])}

FIX TARGETS (fix_target_agent):
- MissingValueAgent, EncodingAgent, FeatureEngineeringAgent, AutoMLAgent

Diagnose ROOT CAUSE and propose ONE fix. Specify fix_target_agent."""

    analysis = llm.decide(prompt, ReflectionAnalysis)
    if analysis is None:
        log.log("llm_failed", "LLM unavailable", "FAILED")
        return {"error": "LLM unavailable", "log": log.to_dict()}

    log.log("reflection_complete",
            f"Cause: {analysis['root_cause'][:80]}. Fix: {analysis['proposed_fix'][:80]}",
            "LLM")

    with open(get_data_path("reflection_log.json"), "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return {"analysis": analysis, "context": {
        "quality_grade": quality_info.get("quality_grade", "?"),
        "issues": issues, "n_models": len(all_results),
    }, "log": log.to_dict()}


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

    kwargs = {}
    try:
        d = json.loads(text)
        kwargs = {k: d[k] for k in ["model_results","execution_trace","failure_history",
                                      "target_column","problem_type"] if k in d}
    except (json.JSONDecodeError, TypeError):
        pass

    result = await execute_reflection(**kwargs)
    return JSONResponse({"jsonrpc": "2.0", "id": body.get("id", 1),
        "result": {"artifacts": [{"parts": [{"type": "text",
            "text": json.dumps(result, default=str)}]}]}})

app = Starlette(routes=[
    Route("/.well-known/agent.json", agent_card_endpoint),
    Route("/", message_endpoint, methods=["POST"]),
])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8208)
