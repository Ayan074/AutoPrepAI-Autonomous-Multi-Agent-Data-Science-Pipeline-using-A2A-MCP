"""
A2A Agent: AutoMLAgent (port 8206)

Skills:
  - train_and_select: Train multiple models, LLM selects the best one

LLM decides:
  - problem type (classification vs regression) — NO hardcoded rules
  - which model is best, and WHY

Models: RandomForest, XGBoost, GradientBoosting, LogisticRegression/Ridge, SVM, DecisionTree

A2A Collaboration:
  - Calls EDAAgent to get data characteristics before training
"""

from __future__ import annotations

import asyncio
import json, logging, os
import numpy as np
import pandas as pd
import uvicorn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import (
    get_llm, read_current_data, get_data_path,
    call_a2a_agent, ExecutionLog,
)
from llm.ollama_engine import ModelSelectionDecision, ProblemTypeDecision

logger = logging.getLogger(__name__)

# Try importing XGBoost (optional but recommended)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Run: pip install xgboost")

RANDOM_STATE = 42

AGENT_CARD = {
    "name": "AutoMLAgent",
    "description": "Trains multiple ML models and uses LLM to select the best one",
    "url": "http://localhost:8206/",
    "version": "2.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "train_and_select",
            "name": "Train and Select Model",
            "description": "Trains 6 models (RF, XGB, GB, LR, SVM, DT), LLM selects best with justification",
            "tags": ["automl", "training", "model-selection"],
            "examples": ["Train models and select the best one"],
        },
    ],
}


def _detect_problem_type_via_llm(llm, df, target_column):
    """LLM decides classification vs regression. NO hardcoded n_unique rules."""
    series = df[target_column]
    sample_vals = [str(v) for v in series.dropna().unique()[:15].tolist()]
    n_unique = int(series.nunique())

    prompt = f"""Decide whether this is a CLASSIFICATION or REGRESSION problem.

Target column: "{target_column}"
dtype: {series.dtype}
n_unique values: {n_unique}
total rows: {len(df)}
sample values: {sample_vals}

Consider:
- The column NAME (semantic meaning)
- The dtype (object/string usually means classification)
- The sample values (labels vs continuous numbers)
- The cardinality relative to dataset size
- Domain knowledge from the column name

Return your decision with reasoning."""

    result = llm.decide(prompt, ProblemTypeDecision)
    if result is not None:
        return result["problem_type"], result["reasoning"]

    # Minimal fallback only if LLM is completely unavailable
    if series.dtype == object or str(series.dtype) == "category":
        return "classification", "LLM unavailable; dtype is object/category"
    return "regression", "LLM unavailable; dtype is numeric, defaulting to regression"


async def execute_automl(target_column: str, problem_type: str = None) -> dict:
    """Train models and let LLM select the best one."""
    llm = get_llm()
    log = ExecutionLog("AutoMLAgent")

    df = read_current_data()

    # ── Auto-detect target column if not found ──
    if target_column not in df.columns:
        for candidate in ["target", "label", "class", "y", "output", "result"]:
            if candidate in df.columns:
                target_column = candidate
                log.log("target_fallback", f"Target '{target_column}' auto-detected", "DETERMINISTIC")
                break
        else:
            target_column = df.columns[-1]
            log.log("target_fallback", f"Using last column '{target_column}' as target", "DETERMINISTIC")

    # ══════════════════════════════════════════
    #  A2A: Ask EDAAgent for feature risks (REAL collaboration)
    # ══════════════════════════════════════════
    eda_health = "unknown"
    try:
        log.log("a2a_request", "Querying EDAAgent for feature risk summary...", "A2A")
        eda_response = await asyncio.wait_for(
            call_a2a_agent(
                "EDAAgent",
                "Provide feature risk summary and imbalance analysis."
            ),
            timeout=120.0,
        )
        if isinstance(eda_response, dict) and "interpretation" in eda_response:
            eda_health = eda_response.get("interpretation", {}).get("overall_data_health", "unknown")
        log.log("a2a_received", f"EDAAgent: data health = {eda_health}", "A2A")
    except (asyncio.TimeoutError, Exception) as e:
        # Fallback: read from disk if A2A fails
        log.log("a2a_fallback", f"A2A failed ({e}), reading EDA from disk", "CACHE")
        try:
            eda_path = get_data_path("eda_report.json")
            if os.path.exists(eda_path):
                with open(eda_path) as f:
                    eda_data = json.load(f)
                eda_health = eda_data.get("interpretation", {}).get("overall_data_health", "unknown")
        except Exception:
            pass

    # ── Prepare data ──
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target if categorical
    le = None
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_column)

    # ══════════════════════════════════════════
    #  LLM decides problem type — NO HARDCODING
    # ══════════════════════════════════════════
    if problem_type not in ("classification", "regression"):
        problem_type, pt_reasoning = _detect_problem_type_via_llm(llm, df, target_column)
        log.log("problem_type_detected", f"LLM decided: {problem_type}. Reason: {pt_reasoning}", "LLM")
    else:
        log.log("problem_type_provided", f"Problem type provided: {problem_type}", "DETERMINISTIC")

    n_unique = int(y.nunique())
    log.log("data_prep", f"Target: {target_column}, Type: {problem_type}, Classes: {n_unique}", "DETERMINISTIC")

    # Select only numeric features
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        return {"error": "No numeric features available. Run EncodingAgent first.", "_success": False}

    # Handle remaining NaN
    X = X.fillna(X.median())
    y = y.fillna(y.mode().iloc[0] if len(y.mode()) > 0 else 0)

    if len(X) < 10:
        return {"error": f"Insufficient data: only {len(X)} rows.", "_success": False}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ── Build model registry (6 models) ──
    if problem_type == "classification":
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "SVM": SVC(random_state=RANDOM_STATE, probability=True),
            "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=RANDOM_STATE,
                use_label_encoder=False, eval_metric="logloss", verbosity=0,
            )
    else:
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "SVR": SVR(),
            "DecisionTree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(
                n_estimators=100, random_state=RANDOM_STATE, verbosity=0,
            )

    # ── Train all models ──
    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Also get train score for overfitting check
            y_train_pred = model.predict(X_train)

            if problem_type == "classification":
                test_metrics = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                    "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                    "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                    "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                }
                train_f1 = round(float(f1_score(y_train, y_train_pred, average="weighted", zero_division=0)), 4)
                test_metrics["train_f1"] = train_f1
                test_metrics["overfit_gap"] = round(train_f1 - test_metrics["f1_weighted"], 4)
            else:
                test_metrics = {
                    "r2": round(float(r2_score(y_test, y_pred)), 4),
                    "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                }
                train_r2 = round(float(r2_score(y_train, y_train_pred)), 4)
                test_metrics["train_r2"] = train_r2
                test_metrics["overfit_gap"] = round(train_r2 - test_metrics["r2"], 4)

            importances = {}
            if hasattr(model, "feature_importances_"):
                for feat, imp in sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])[:10]:
                    importances[feat] = round(float(imp), 4)

            results.append({"model": name, "metrics": test_metrics, "feature_importances": importances})
            log.log(f"train_{name}", f"Trained {name}: {test_metrics}", "DETERMINISTIC")

        except Exception as e:
            log.log(f"train_{name}_failed", f"Failed: {str(e)[:100]}", "ERROR")

    if not results:
        return {"error": "All models failed to train.", "log": log.to_dict(), "_success": False}

    # ── LLM selects the best model ──
    prompt = f"""You trained {len(results)} models for a {problem_type} problem.

Model Results:
{json.dumps(results, indent=2)}

Dataset: {df.shape[0]} rows, {X.shape[1]} features
Target: {target_column}, Problem: {problem_type}
Data health (from EDA): {eda_health}

Select the BEST model. Consider:
1. Primary metric: {'f1_weighted' if problem_type == 'classification' else 'r2'}
2. Overfitting risk (check overfit_gap — high gap = bad)
3. Feature importance distribution

Justify your choice and name a runner-up."""

    selection = llm.decide(prompt, ModelSelectionDecision)
    if selection is None:
        key = "f1_weighted" if problem_type == "classification" else "r2"
        best = max(results, key=lambda r: r["metrics"].get(key, 0))
        selection = {"best_model": best["model"], "justification": "LLM unavailable, selected by best metric",
                     "confidence": 0.5}
        log.log("llm_fallback", f"LLM unavailable, selected {best['model']}", "FALLBACK")
    else:
        log.log("model_selected", f"Best: {selection['best_model']}. {selection['justification']}", "LLM")

    output = {
        "best_model": selection,
        "all_results": results,
        "problem_type": problem_type,
        "target_column": target_column,
        "n_features": X.shape[1],
        "n_classes": n_unique,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "log": log.to_dict(),
    }

    with open(get_data_path("automl_results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


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

    target_column = "target"
    problem_type = None
    try:
        msg_data = json.loads(text)
        target_column = msg_data.get("target_column", target_column)
        problem_type = msg_data.get("problem_type")
    except (json.JSONDecodeError, TypeError):
        pass

    result = await execute_automl(target_column, problem_type)

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
    uvicorn.run(app, host="0.0.0.0", port=8206)
