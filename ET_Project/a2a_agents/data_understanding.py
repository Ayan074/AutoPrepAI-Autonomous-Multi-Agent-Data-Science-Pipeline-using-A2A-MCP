"""
A2A Agent: DataUnderstandingAgent (port 8201)

Skills:
  - profile_dataset: Analyze dataset structure, classify columns, detect target
  - classify_columns: Classify column types using LLM

LLM decides: column types, target column, domain detection
Execution: pandas profiling (deterministic)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import (
    get_llm, read_current_data, save_current_data,
    get_data_path, ExecutionLog, WORKING_DIR,
)
from llm.ollama_engine import DatasetProfile

logger = logging.getLogger(__name__)


# ── Agent Card (served at /.well-known/agent.json) ──
AGENT_CARD = {
    "name": "DataUnderstandingAgent",
    "description": "Analyzes dataset structure, classifies columns, detects target variable and domain",
    "url": "http://localhost:8201/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "profile_dataset",
            "name": "Profile Dataset",
            "description": "Analyze dataset structure, classify columns, detect target column and domain",
            "tags": ["profiling", "eda", "classification"],
            "examples": ["Profile the dataset", "What columns does this dataset have?"],
        },
        {
            "id": "classify_columns",
            "name": "Classify Columns",
            "description": "Classify column types (numerical, categorical, etc.) using LLM reasoning",
            "tags": ["classification", "columns"],
            "examples": ["Classify all columns in the dataset"],
        },
    ],
}


async def execute_profile(data_path: str = None) -> dict:
    """Profile the dataset — ALL decisions made by LLM."""
    llm = get_llm()
    log = ExecutionLog("DataUnderstandingAgent")

    path = data_path or get_data_path()
    df = pd.read_csv(path)

    # ── Build data summary for LLM ──
    column_summaries = []
    for col in df.columns:
        series = df[col]
        summary = {
            "name": col,
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique()),
            "n_missing": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "sample_values": [str(v) for v in series.dropna().head(6).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series):
            summary["mean"] = round(float(series.mean()), 4) if len(series.dropna()) > 0 else None
            summary["std"] = round(float(series.std()), 4) if len(series.dropna()) > 1 else None
            summary["skewness"] = round(float(series.skew()), 4) if len(series.dropna()) > 2 else None
        column_summaries.append(summary)

    prompt = f"""Analyze this dataset and provide a full profile.

Dataset: {df.shape[0]} rows × {df.shape[1]} columns

Columns:
{json.dumps(column_summaries, indent=2)}

For EACH column, classify its type as one of: numerical, categorical, boolean, datetime, text, id.
Also identify:
1. Which column is the target variable (the column to predict)?
2. What domain is this dataset from? (e.g., healthcare, finance, spam detection, etc.)
3. Provide a brief summary of the dataset.

Base your decisions on the dtype, n_unique, sample_values, and column name."""

    # ── LLM decides everything ──
    result = llm.decide(prompt, DatasetProfile)

    if result is None:
        log.log("profile_failed", "LLM unavailable — cannot profile dataset", "FAILED")
        return {"error": "LLM unavailable", "log": log.to_dict()}

    log.log(
        "profile_complete",
        f"Classified {len(result['column_classifications'])} columns. "
        f"Target: {result.get('target_column', 'none')}. "
        f"Domain: {result.get('detected_domain', 'unknown')}.",
        "LLM",
    )

    # ── Save profile to working directory ──
    profile_path = get_data_path("dataset_profile.json")
    with open(profile_path, "w") as f:
        json.dump(result, f, indent=2)

    return {
        "profile": result,
        "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "log": log.to_dict(),
    }


# ═══════════════════════════════════════════════
#  A2A HTTP Endpoints
# ═══════════════════════════════════════════════

async def agent_card_endpoint(request: Request) -> JSONResponse:
    """Serve the Agent Card at /.well-known/agent.json."""
    return JSONResponse(AGENT_CARD)


async def message_endpoint(request: Request) -> JSONResponse:
    """Handle A2A JSON-RPC message/send requests."""
    body = await request.json()
    params = body.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])

    text = ""
    for part in parts:
        if part.get("type") == "text":
            text = part["text"]
            break

    # ── Route to the appropriate skill ──
    if "profile" in text.lower() or "classify" in text.lower() or "understand" in text.lower():
        result = await execute_profile()
    else:
        result = await execute_profile()  # Default skill

    # ── Return A2A JSON-RPC response ──
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "artifacts": [{
                "parts": [{"type": "text", "text": json.dumps(result, default=str)}],
            }],
        },
    })


app = Starlette(routes=[
    Route("/.well-known/agent.json", agent_card_endpoint),
    Route("/", message_endpoint, methods=["POST"]),
])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8201)
