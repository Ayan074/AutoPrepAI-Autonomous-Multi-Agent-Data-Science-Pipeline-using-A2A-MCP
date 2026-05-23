"""
A2A Agent: FeatureEngineeringAgent (port 8205)

Skills:
  - engineer_features: LLM recommends and applies feature transformations

LLM decides: which transforms (log, sqrt, bin, interaction)
Execution: numpy/pandas transforms (deterministic)
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
    get_data_path, ExecutionLog,
)
from llm.ollama_engine import FeatureEngineeringPlan

logger = logging.getLogger(__name__)


AGENT_CARD = {
    "name": "FeatureEngineeringAgent",
    "description": "Creates new features via LLM-guided transformations: log, sqrt, binning, interactions",
    "url": "http://localhost:8205/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "engineer_features",
            "name": "Engineer Features",
            "description": "LLM recommends feature transformations based on data distributions",
            "tags": ["features", "engineering", "transforms"],
            "examples": ["Create useful features from the dataset"],
        },
    ],
}


async def execute_feature_engineering(target_column: str = None) -> dict:
    """Engineer features — LLM recommends, numpy/pandas executes."""
    llm = get_llm()
    log = ExecutionLog("FeatureEngineeringAgent")

    df = read_current_data()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]

    if len(num_cols) < 1:
        log.log("no_features", "No numerical columns for feature engineering.", "SKIP")
        return {"transforms": [], "log": log.to_dict()}

    # ── Build feature profile for LLM ──
    feature_profile = []
    for col in num_cols[:20]:  # Limit to 20 to keep prompt manageable
        series = df[col].dropna()
        if len(series) < 10:
            continue
        feature_profile.append({
            "column": col,
            "skewness": round(float(series.skew()), 4),
            "std": round(float(series.std()), 4),
            "range": round(float(series.max() - series.min()), 4),
            "n_unique": int(series.nunique()),
            "mean": round(float(series.mean()), 4),
            "all_positive": bool((series > 0).all()),
            "all_non_negative": bool((series >= 0).all()),
        })

    # ── Load correlation data if available ──
    corr_info = ""
    try:
        corr_path = get_data_path("correlations.json")
        if pd.io.common.file_exists(corr_path):
            with open(corr_path) as f:
                corr_data = json.load(f)
            corr_info = f"\nCorrelation pairs (|r| > 0.5):\n{json.dumps(corr_data.get('pairs', [])[:10], indent=2)}"
    except Exception:
        pass

    prompt = f"""Recommend feature engineering transforms for this dataset.

Target column: {target_column or 'unknown'}
Numerical features:
{json.dumps(feature_profile, indent=2)}
{corr_info}

Available transforms:
- log: log1p transform (only for all-positive columns, good for high skewness)
- sqrt: square root (only for non-negative columns)
- square: square of the feature
- bin: quantile binning (5 bins) for high-range features
- interaction: multiply two correlated features (specify interaction_with)

Rules:
- Only recommend transforms with clear statistical justification
- log transforms: only when |skewness| > 2 AND all values positive
- interactions: only for strongly correlated pairs (|r| > 0.7)
- Do NOT over-engineer: max 5 transforms total
- Each transform must include the column name and reason"""

    plan = llm.decide(prompt, FeatureEngineeringPlan)

    if plan is None:
        log.log("llm_failed", "LLM unavailable — cannot decide features", "FAILED")
        return {"error": "LLM unavailable", "log": log.to_dict()}

    # ── Execute transforms (deterministic) ──
    transforms_applied = []

    for transform in plan.get("transforms", [])[:5]:  # Cap at 5
        col = transform["column"]
        t_type = transform["transform"]
        reason = transform.get("reason", "LLM recommendation")

        if col not in df.columns:
            continue

        series = df[col].dropna()

        if t_type == "log" and (series > 0).all():
            new_col = f"{col}_log"
            df[new_col] = np.log1p(df[col])
            log.log("log_transform", f"Created '{new_col}'. {reason}", "LLM", [col, new_col])

        elif t_type == "sqrt" and (series >= 0).all():
            new_col = f"{col}_sqrt"
            df[new_col] = np.sqrt(df[col])
            log.log("sqrt_transform", f"Created '{new_col}'. {reason}", "LLM", [col, new_col])

        elif t_type == "square":
            new_col = f"{col}_sq"
            df[new_col] = df[col] ** 2
            log.log("square_transform", f"Created '{new_col}'. {reason}", "LLM", [col, new_col])

        elif t_type == "bin":
            new_col = f"{col}_binned"
            try:
                df[new_col] = pd.qcut(df[col], q=5, labels=False, duplicates="drop")
                log.log("binning", f"Created '{new_col}' (5 quantile bins). {reason}", "LLM", [col, new_col])
            except Exception:
                continue

        elif t_type == "interaction":
            other_col = transform.get("interaction_with", "")
            if other_col and other_col in df.columns:
                new_col = f"{col}_x_{other_col}"
                df[new_col] = df[col] * df[other_col]
                log.log("interaction", f"Created '{new_col}'. {reason}", "LLM", [col, other_col, new_col])
            else:
                continue
        else:
            continue

        transforms_applied.append({
            "column": col,
            "transform": t_type,
            "reason": reason,
            "decided_by": "LLM",
        })

    # ── Save processed data ──
    save_current_data(df)

    log.log("fe_complete", f"Applied {len(transforms_applied)} transforms. Shape: {df.shape}", "LLM")

    with open(get_data_path("feature_engineering_log.json"), "w") as f:
        json.dump(transforms_applied, f, indent=2)

    return {
        "transforms_applied": transforms_applied,
        "final_shape": list(df.shape),
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
    msg = params.get("message", {})
    parts = msg.get("parts", [])

    text = ""
    for part in parts:
        if part.get("type") == "text":
            text = part["text"]
            break

    target_col = None
    try:
        msg_data = json.loads(text)
        target_col = msg_data.get("target_column")
    except (json.JSONDecodeError, TypeError):
        pass

    result = await execute_feature_engineering(target_column=target_col)

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
    uvicorn.run(app, host="0.0.0.0", port=8205)
