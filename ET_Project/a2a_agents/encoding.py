"""
A2A Agent: EncodingAgent (port 8204)

Skills:
  - encode_categorical: LLM decides encoding strategy per column

LLM decides: encoding type (onehot/label/frequency/binary/tfidf)
Execution: pandas/sklearn encoding (deterministic)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import uvicorn
from sklearn.preprocessing import LabelEncoder
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import (
    get_llm, read_current_data, save_current_data,
    get_data_path, ExecutionLog,
)
from llm.ollama_engine import EncodingPlan

logger = logging.getLogger(__name__)


AGENT_CARD = {
    "name": "EncodingAgent",
    "description": "Encodes categorical columns using LLM-selected strategies",
    "url": "http://localhost:8204/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "encode_categorical",
            "name": "Encode Categorical Columns",
            "description": "LLM decides encoding strategy per column, then executes deterministically",
            "tags": ["encoding", "categorical", "preprocessing"],
            "examples": ["Encode all categorical columns"],
        },
    ],
}


async def execute_encoding(target_column: str = None) -> dict:
    """Encode categorical columns — LLM decides, sklearn/pandas executes."""
    llm = get_llm()
    log = ExecutionLog("EncodingAgent")

    df = read_current_data()

    # ── Load profile to know column types ──
    profile_path = get_data_path("dataset_profile.json")
    profile = {}
    try:
        with open(profile_path) as f:
            profile = json.load(f)
    except FileNotFoundError:
        pass

    # ── Identify categorical/object columns ──
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not object_cols:
        log.log("no_encoding", "No categorical columns to encode.", "SKIP")
        return {"encoding_log": [], "log": log.to_dict()}

    # ── Build column info for LLM ──
    col_infos = []
    for col in object_cols:
        col_infos.append({
            "column": col,
            "n_unique": int(df[col].nunique()),
            "dtype": str(df[col].dtype),
            "sample_values": [str(v) for v in df[col].dropna().unique()[:8].tolist()],
            "is_target": col == target_column,
            "total_rows": len(df),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
        })

    prompt = f"""Decide the encoding strategy for each categorical column.

Dataset: {df.shape[0]} rows × {df.shape[1]} columns
Target column: {target_column or 'unknown'}

Categorical columns to encode:
{json.dumps(col_infos, indent=2)}

For EACH column, choose one encoding: onehot, label, frequency, binary, or tfidf.
- Target column: always use label encoding
- Binary columns (2 unique): use binary
- Low cardinality (≤10 unique): use onehot
- Medium cardinality (11-50): use label
- High cardinality (>50): use frequency
- Text-like columns (long strings, many unique): use tfidf

Provide a reason for each decision."""

    plan = llm.decide(prompt, EncodingPlan)

    if plan is None:
        log.log("llm_failed", "LLM unavailable — cannot decide encoding", "FAILED")
        return {"error": "LLM unavailable", "log": log.to_dict()}

    # ── Execute encoding (deterministic) ──
    encoding_log = []

    for decision in plan["decisions"]:
        col = decision["column"]
        encoding = decision["encoding"]
        reason = decision["reason"]

        if col not in df.columns:
            continue

        n_unique = df[col].nunique()

        if encoding == "binary" or encoding == "label":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            log.log(f"label_encode", f"Column '{col}': Label encoded ({n_unique} classes). {reason}", "LLM", [col])

        elif encoding == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            log.log(f"onehot_encode", f"Column '{col}': One-hot → {len(dummies.columns)} new cols. {reason}", "LLM", [col])

        elif encoding == "frequency":
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).fillna(0).astype(float)
            log.log(f"frequency_encode", f"Column '{col}': Frequency encoded ({n_unique} cats). {reason}", "LLM", [col])

        elif encoding == "tfidf":
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(max_features=100, stop_words="english", min_df=2, max_df=0.95)
                text_data = df[col].fillna("").astype(str)
                tfidf_matrix = tfidf.fit_transform(text_data)
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f"{col}_tfidf_{w}" for w in tfidf.get_feature_names_out()],
                    index=df.index,
                )
                df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)
                log.log(f"tfidf_encode", f"Column '{col}': TF-IDF vectorized ({tfidf_matrix.shape[1]} features). {reason}", "LLM", [col])
            except Exception as e:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                log.log(f"label_encode_fallback", f"Column '{col}': TF-IDF failed ({e}), label encoded.", "LLM", [col])
                encoding = "label"

        encoding_log.append({
            "column": col,
            "encoding": encoding,
            "reason": reason,
            "decided_by": "LLM",
        })

    # ── Save processed data ──
    save_current_data(df)

    log.log("encoding_complete", f"Encoded {len(encoding_log)} columns. Final shape: {df.shape}", "LLM")

    with open(get_data_path("encoding_log.json"), "w") as f:
        json.dump(encoding_log, f, indent=2)

    return {
        "encoding_log": encoding_log,
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

    # Extract target column from message if provided
    target_col = None
    try:
        msg_data = json.loads(text)
        target_col = msg_data.get("target_column")
    except (json.JSONDecodeError, TypeError):
        pass

    result = await execute_encoding(target_column=target_col)

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
    uvicorn.run(app, host="0.0.0.0", port=8204)
