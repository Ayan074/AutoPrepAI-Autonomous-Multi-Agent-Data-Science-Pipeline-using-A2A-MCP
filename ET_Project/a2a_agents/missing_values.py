"""
A2A Agent: MissingValueAgent (port 8203)

Skills:
  - impute_missing: LLM decides imputation strategy per column

LLM decides: which strategy (mean/median/mode/interpolate/ffill/zero/unknown)
Execution: pandas fillna/interpolate (deterministic)

A2A: Calls DataQualityAgent to get risky columns BEFORE deciding strategies.
This is REAL agent-to-agent communication via HTTP JSON-RPC.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import (
    get_llm, read_current_data, save_current_data,
    get_data_path, ExecutionLog,
)
from llm.ollama_engine import ImputationPlan

logger = logging.getLogger(__name__)


AGENT_CARD = {
    "name": "MissingValueAgent",
    "description": "Handles missing values via LLM-driven imputation strategies",
    "url": "http://localhost:8203/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "impute_missing",
            "name": "Impute Missing Values",
            "description": "LLM decides optimal imputation strategy per column, then executes deterministically",
            "tags": ["imputation", "missing", "cleaning"],
            "examples": ["Handle all missing values in the dataset"],
        },
    ],
}


async def execute_imputation() -> dict:
    """Handle missing values — LLM decides strategies, pandas executes."""
    llm = get_llm()
    log = ExecutionLog("MissingValueAgent")

    df = read_current_data()
    original_rows = len(df)

    # ── Read risky columns from disk (saved by earlier DataQualityAgent step) ──
    risky_columns = []
    try:
        quality_path = get_data_path("quality_assessment.json")
        if os.path.exists(quality_path):
            with open(quality_path) as f:
                quality_data = json.load(f)
            risky_columns = quality_data.get("assessment", {}).get("risky_columns", [])
            if risky_columns:
                log.log(
                    "quality_loaded",
                    f"Loaded {len(risky_columns)} risky columns from disk: {risky_columns}",
                    "CACHE",
                )
        else:
            log.log("quality_missing", "No quality assessment on disk, proceeding without risky columns", "CACHE")
    except Exception as e:
        log.log("quality_load_error", f"Could not load quality assessment: {e}", "CACHE")

    # ── Build missing value summary for LLM ──
    missing_cols = []
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue

        col_info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": n_missing,
            "missing_pct": round(n_missing / len(df) * 100, 2),
            "n_unique": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
            "is_risky": col in risky_columns,
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if len(series) > 0:
                col_info["mean"] = round(float(series.mean()), 4)
                col_info["median"] = round(float(series.median()), 4)
                col_info["skewness"] = round(float(series.skew()), 4) if len(series) > 2 else 0

        missing_cols.append(col_info)

    if not missing_cols:
        log.log("no_missing", "No missing values found — nothing to impute.", "SKIP")
        return {"strategies_applied": [], "remaining_nan": 0, "log": log.to_dict()}

    # ── LLM decides imputation strategy for ALL columns at once ──
    prompt = f"""Decide the imputation strategy for each column with missing values.

Dataset: {df.shape[0]} rows × {df.shape[1]} columns

Columns with missing values:
{json.dumps(missing_cols, indent=2)}

Risky columns (flagged by DataQualityAgent — use conservative strategies):
{json.dumps(risky_columns)}

For EACH column, decide one strategy: mean, median, mode, interpolate, ffill, zero, or unknown.
- For risky columns, prefer median (numerical) or mode (categorical) — they are robust.
- For skewed numerical data, prefer median over mean.
- For categorical/text data, use mode.
- For time-series data, use ffill or interpolate.

Provide a reason for EACH decision and an overall reasoning summary."""

    plan = llm.decide(prompt, ImputationPlan)

    if plan is None:
        log.log("llm_failed", "LLM unavailable — cannot decide imputation", "FAILED")
        return {"error": "LLM unavailable", "log": log.to_dict()}

    # ── Execute imputation (deterministic) ──
    strategies_applied = []

    for decision in plan["decisions"]:
        col = decision["column"]
        strategy = decision["strategy"]
        reason = decision["reason"]

        if col not in df.columns or df[col].isnull().sum() == 0:
            continue

        n_filled = int(df[col].isnull().sum())

        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
            df[col] = df[col].fillna(fill)
        elif strategy == "interpolate":
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        elif strategy == "ffill":
            df[col] = df[col].ffill().bfill()
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "unknown":
            df[col] = df[col].fillna("UNKNOWN")
        else:
            # If LLM returns something unexpected, use median/mode
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                strategy = "median"
            else:
                mode_val = df[col].mode()
                fill = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
                df[col] = df[col].fillna(fill)
                strategy = "mode"

        log.log(
            f"impute_{strategy}",
            f"Column '{col}': Filled {n_filled} missing with {strategy}. {reason}",
            "LLM",
            [col],
        )
        strategies_applied.append({
            "column": col,
            "strategy": strategy,
            "n_filled": n_filled,
            "reason": reason,
            "decided_by": "LLM",
        })

    # ── Save processed data ──
    save_current_data(df)

    remaining = int(df.isnull().sum().sum())
    log.log(
        "imputation_complete",
        f"Applied {len(strategies_applied)} strategies. Remaining NaN: {remaining}. "
        f"Row retention: {len(df)}/{original_rows} ({len(df)/original_rows*100:.1f}%)",
        "LLM",
    )

    # ── Save imputation log ──
    with open(get_data_path("imputation_log.json"), "w") as f:
        json.dump({"strategies": strategies_applied, "plan": plan}, f, indent=2)

    return {
        "strategies_applied": strategies_applied,
        "remaining_nan": remaining,
        "row_retention_pct": round(len(df) / original_rows * 100, 1),
        "log": log.to_dict(),
    }


async def execute_imputation_with_context(risky_columns: list = None) -> dict:
    """Impute missing values with pre-supplied context (avoids A2A HTTP call).

    Called by the orchestrator with cached DataQualityAgent results.
    The LLM still makes ALL imputation decisions — only the risky column
    context is pre-supplied instead of fetched via HTTP.
    """
    llm = get_llm()
    log = ExecutionLog("MissingValueAgent")

    df = read_current_data()
    original_rows = len(df)
    risky = risky_columns or []

    if risky:
        log.log("context_received", f"Received {len(risky)} risky columns from cache: {risky}", "CACHE")

    # ── Build missing value summary for LLM ──
    missing_cols = []
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing == 0:
            continue
        col_info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "n_missing": n_missing,
            "missing_pct": round(n_missing / len(df) * 100, 2),
            "n_unique": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
            "is_risky": col in risky,
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if len(series) > 0:
                col_info["mean"] = round(float(series.mean()), 4)
                col_info["median"] = round(float(series.median()), 4)
                col_info["skewness"] = round(float(series.skew()), 4) if len(series) > 2 else 0
        missing_cols.append(col_info)

    if not missing_cols:
        log.log("no_missing", "No missing values found.", "SKIP")
        return {"strategies_applied": [], "remaining_nan": 0, "log": log.to_dict()}

    prompt = f"""Decide the imputation strategy for each column with missing values.

Dataset: {df.shape[0]} rows x {df.shape[1]} columns

Columns with missing values:
{json.dumps(missing_cols, indent=2)}

Risky columns (use conservative strategies): {json.dumps(risky)}

For EACH column, decide one strategy: mean, median, mode, interpolate, ffill, zero, or unknown.
- Risky columns: prefer median (numerical) or mode (categorical).
- Skewed numerical: prefer median over mean.
- Categorical/text: use mode.
Provide a reason for EACH decision."""

    plan = llm.decide(prompt, ImputationPlan)
    if plan is None:
        log.log("llm_failed", "LLM unavailable", "FAILED")
        return {"error": "LLM unavailable", "log": log.to_dict()}

    strategies_applied = []
    for decision in plan["decisions"]:
        col = decision["column"]
        strategy = decision["strategy"]
        reason = decision["reason"]
        if col not in df.columns or df[col].isnull().sum() == 0:
            continue
        n_filled = int(df[col].isnull().sum())

        if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN")
        elif strategy == "interpolate":
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
        elif strategy == "ffill":
            df[col] = df[col].ffill().bfill()
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "unknown":
            df[col] = df[col].fillna("UNKNOWN")
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                strategy = "median"
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN")
                strategy = "mode"

        log.log(f"impute_{strategy}", f"'{col}': {n_filled} filled with {strategy}. {reason}", "LLM", [col])
        strategies_applied.append({"column": col, "strategy": strategy, "n_filled": n_filled, "reason": reason, "decided_by": "LLM"})

    save_current_data(df)
    remaining = int(df.isnull().sum().sum())
    log.log("imputation_complete", f"{len(strategies_applied)} strategies. Remaining NaN: {remaining}. Retention: {len(df)/original_rows*100:.1f}%", "LLM")

    return {
        "strategies_applied": strategies_applied,
        "remaining_nan": remaining,
        "row_retention_pct": round(len(df) / original_rows * 100, 1),
        "log": log.to_dict(),
    }


# ═══════════════════════════════════════════════
#  A2A Endpoints
# ═══════════════════════════════════════════════

async def agent_card_endpoint(request: Request) -> JSONResponse:
    return JSONResponse(AGENT_CARD)


async def message_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    result = await execute_imputation()

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
    uvicorn.run(app, host="0.0.0.0", port=8203)
