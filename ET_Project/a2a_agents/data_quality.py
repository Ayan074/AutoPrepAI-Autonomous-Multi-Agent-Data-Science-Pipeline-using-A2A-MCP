"""
A2A Agent: DataQualityAgent (port 8202)

Skills:
  - assess_quality: LLM evaluates overall data quality
  - get_risky_columns: Identify columns with significant issues

LLM decides: quality grade, risky columns, priority actions
Execution: pandas analysis (deterministic)

A2A: MissingValueAgent calls this to ask about risky columns.
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

from .base import get_llm, read_current_data, get_data_path, ExecutionLog
from llm.ollama_engine import QualityAssessment

logger = logging.getLogger(__name__)


AGENT_CARD = {
    "name": "DataQualityAgent",
    "description": "Detects data quality issues: missing values, duplicates, type mismatches, outliers",
    "url": "http://localhost:8202/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "assess_quality",
            "name": "Assess Quality",
            "description": "LLM evaluates overall dataset quality and identifies priority actions",
            "tags": ["quality", "assessment"],
            "examples": ["Assess the quality of this dataset"],
        },
        {
            "id": "get_risky_columns",
            "name": "Get Risky Columns",
            "description": "Identify columns with significant quality issues",
            "tags": ["quality", "columns", "risk"],
            "examples": ["Which columns are risky?"],
        },
    ],
}


async def execute_quality_assessment() -> dict:
    """Assess data quality — ALL decisions by LLM."""
    llm = get_llm()
    log = ExecutionLog("DataQualityAgent")

    df = read_current_data()

    # ── Build quality report (deterministic) ──
    quality_facts = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_pct": round(df.duplicated().mean() * 100, 2),
        "total_missing": int(df.isnull().sum().sum()),
        "total_cells": int(df.shape[0] * df.shape[1]),
        "overall_missing_pct": round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        col_report = {
            "dtype": str(series.dtype),
            "n_missing": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "n_unique": int(series.nunique()),
            "is_constant": bool(series.nunique() <= 1),
        }

        # Check for mixed types in object columns
        if series.dtype == object:
            non_null = series.dropna()
            if len(non_null) > 0:
                numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
                col_report["has_mixed_types"] = bool(0 < numeric_count < len(non_null))

        # Check for negative values in numeric columns
        if pd.api.types.is_numeric_dtype(series):
            col_report["has_negatives"] = bool((series < 0).any())
            col_report["skewness"] = round(float(series.skew()), 4) if len(series.dropna()) > 2 else None

        quality_facts["columns"][col] = col_report

    # ── LLM assesses quality ──
    prompt = f"""Assess the data quality of this dataset.

Quality Report:
{json.dumps(quality_facts, indent=2)}

Evaluate the overall quality. Identify:
1. A quality score (0-100) and grade (A-F)
2. Critical issues that MUST be addressed
3. Priority actions (ordered by importance)
4. Risky columns that need special handling during imputation
5. A brief assessment summary"""

    result = llm.decide(prompt, QualityAssessment)

    if result is None:
        log.log("quality_failed", "LLM unavailable — cannot assess quality", "FAILED")
        return {"error": "LLM unavailable", "quality_facts": quality_facts, "log": log.to_dict()}

    log.log(
        "quality_assessed",
        f"Grade: {result['quality_grade']} (Score: {result['quality_score']}). "
        f"Critical issues: {len(result.get('critical_issues', []))}. "
        f"Risky columns: {result.get('risky_columns', [])}.",
        "LLM",
    )

    # ── Save assessment ──
    assessment_path = get_data_path("quality_assessment.json")
    with open(assessment_path, "w") as f:
        json.dump({"assessment": result, "facts": quality_facts}, f, indent=2)

    return {
        "assessment": result,
        "quality_facts": quality_facts,
        "log": log.to_dict(),
    }


# ═══════════════════════════════════════════════
#  A2A Endpoints
# ═══════════════════════════════════════════════

async def agent_card_endpoint(request: Request) -> JSONResponse:
    return JSONResponse(AGENT_CARD)


async def message_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    params = body.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])

    text = ""
    for part in parts:
        if part.get("type") == "text":
            text = part["text"]
            break

    # ── Route based on intent ──
    if "risky" in text.lower() or "risk" in text.lower():
        result = await execute_quality_assessment()
        # Extract just the risky columns for this specific A2A query
        assessment = result.get("assessment", {})
        response = {
            "risky_columns": assessment.get("risky_columns", []),
            "quality_grade": assessment.get("quality_grade", ""),
            "priority_actions": assessment.get("priority_actions", []),
        }
    else:
        result = await execute_quality_assessment()
        response = result

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "artifacts": [{
                "parts": [{"type": "text", "text": json.dumps(response, default=str)}],
            }],
        },
    })


app = Starlette(routes=[
    Route("/.well-known/agent.json", agent_card_endpoint),
    Route("/", message_endpoint, methods=["POST"]),
])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8202)
