"""
A2A Agent: EDAAgent (port 8207)

Skills:
  - run_eda: Comprehensive exploratory data analysis
  - detect_anomalies: Statistical anomaly detection

LLM decides: which findings matter, severity, recommendations
Execution: pandas/scipy stats (deterministic)

A2A Collaboration:
  - Calls DataQualityAgent for quality context
  - Calls MCP tools: detect_outliers, distribution_analysis, correlation_analysis
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import uvicorn
from scipy import stats
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .base import (
    get_llm, read_current_data, get_data_path,
    call_a2a_agent, call_mcp_tool, ExecutionLog,
)
from llm.ollama_engine import EDAReport

logger = logging.getLogger(__name__)


AGENT_CARD = {
    "name": "EDAAgent",
    "description": "Performs comprehensive exploratory data analysis with LLM interpretation",
    "url": "http://localhost:8207/",
    "version": "1.0.0",
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "run_eda",
            "name": "Run EDA",
            "description": "Full exploratory analysis: missing, outliers, skewness, correlations, imbalance, multicollinearity",
            "tags": ["eda", "analysis", "statistics"],
            "examples": ["Run exploratory data analysis", "Analyze this dataset"],
        },
        {
            "id": "detect_anomalies",
            "name": "Detect Anomalies",
            "description": "Statistical anomaly detection across all columns",
            "tags": ["anomalies", "outliers"],
            "examples": ["Find anomalies in the data"],
        },
    ],
}


async def execute_eda(target_column: str = None) -> dict:
    """Run comprehensive EDA — stats are deterministic, LLM interprets."""
    llm = get_llm()
    log = ExecutionLog("EDAAgent")

    df = read_current_data()
    csv_path = get_data_path("current_data.csv")

    # ══════════════════════════════════════════
    #  1. A2A: Ask DataQualityAgent for quality risks (REAL collaboration)
    # ══════════════════════════════════════════
    quality_grade = "unknown"
    risky_cols = []
    try:
        log.log("a2a_request", "Querying DataQualityAgent for quality risks...", "A2A")
        quality_response = await asyncio.wait_for(
            call_a2a_agent(
                "DataQualityAgent",
                "Provide quality risks and suspicious columns."
            ),
            timeout=90.0,
        )
        quality_grade = quality_response.get("assessment", {}).get("quality_grade", "unknown")
        risky_cols = quality_response.get("assessment", {}).get("risky_columns", [])
        log.log("a2a_received", f"DataQualityAgent: grade={quality_grade}, risky={risky_cols}", "A2A")
    except (asyncio.TimeoutError, Exception) as e:
        # Fallback: read from disk if A2A fails
        log.log("a2a_fallback", f"A2A failed ({e}), reading from disk", "CACHE")
        try:
            quality_path = get_data_path("quality_assessment.json")
            if os.path.exists(quality_path):
                with open(quality_path) as f:
                    quality_data = json.load(f)
                quality_grade = quality_data.get("assessment", {}).get("quality_grade", "unknown")
                risky_cols = quality_data.get("assessment", {}).get("risky_columns", [])
        except Exception:
            pass

    # ══════════════════════════════════════════
    #  2. MCP: Call outlier detection tool (with timeout protection)
    # ══════════════════════════════════════════
    outlier_summary = {}
    try:
        log.log("mcp_request", "Calling MCP detect_outliers tool...", "MCP")
        outlier_result = await asyncio.wait_for(
            call_mcp_tool("detect_outliers", {"csv_path": csv_path, "method": "iqr"}),
            timeout=60.0,
        )
        outlier_summary = outlier_result.get("columns", {})
        log.log("mcp_received", f"Outlier detection: {len(outlier_summary)} columns analyzed", "MCP")
    except (asyncio.TimeoutError, Exception) as e:
        log.log("mcp_timeout", f"detect_outliers unavailable ({e}), proceeding without outlier data", "MCP")

    # ══════════════════════════════════════════
    #  3. MCP: Call distribution analysis tool (with timeout protection)
    # ══════════════════════════════════════════
    dist_summary = {}
    try:
        log.log("mcp_request", "Calling MCP distribution_analysis tool...", "MCP")
        dist_result = await asyncio.wait_for(
            call_mcp_tool("distribution_analysis", {"csv_path": csv_path}),
            timeout=60.0,
        )
        dist_summary = dist_result.get("columns", {})
        log.log("mcp_received", f"Distribution analysis: {len(dist_summary)} columns analyzed", "MCP")
    except (asyncio.TimeoutError, Exception) as e:
        log.log("mcp_timeout", f"distribution_analysis unavailable ({e}), proceeding without distribution data", "MCP")

    # ══════════════════════════════════════════
    #  4. Deterministic EDA computations
    # ══════════════════════════════════════════

    # Missing value analysis
    missing_analysis = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            missing_analysis[col] = {
                "count": n_missing,
                "pct": round(n_missing / len(df) * 100, 2),
            }

    # Correlation analysis
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    high_corr_pairs = []
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        for i, col_a in enumerate(num_cols):
            for j, col_b in enumerate(num_cols):
                if i >= j:
                    continue
                val = float(corr_matrix.loc[col_a, col_b])
                if abs(val) > 0.7:
                    high_corr_pairs.append({
                        "col_a": col_a, "col_b": col_b,
                        "correlation": round(val, 4),
                    })
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Target imbalance (classification)
    imbalance_info = {}
    if target_column and target_column in df.columns:
        vc = df[target_column].value_counts()
        if len(vc) <= 20:  # likely classification
            imbalance_ratio = round(float(vc.min() / vc.max()), 4) if vc.max() > 0 else 0
            imbalance_info = {
                "class_distribution": {str(k): int(v) for k, v in vc.items()},
                "imbalance_ratio": imbalance_ratio,
                "is_imbalanced": imbalance_ratio < 0.3,
            }

    # Skewness summary
    skewed_cols = []
    for col in num_cols:
        skew = float(df[col].skew()) if len(df[col].dropna()) > 2 else 0
        if abs(skew) > 1:
            skewed_cols.append({"column": col, "skewness": round(skew, 4)})

    # Multicollinearity (VIF approximation via condition number)
    multicollinearity_warning = False
    if len(num_cols) >= 2:
        try:
            numeric_df = df[num_cols].dropna()
            if len(numeric_df) > 10:
                cond_number = float(np.linalg.cond(numeric_df.values))
                multicollinearity_warning = cond_number > 30
        except Exception:
            pass

    # ══════════════════════════════════════════
    #  5. LLM interprets all findings
    # ══════════════════════════════════════════
    eda_facts = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "quality_grade": quality_grade,
        "risky_columns": risky_cols,
        "missing_columns": len(missing_analysis),
        "total_missing": int(df.isnull().sum().sum()),
        "outlier_columns": {k: v["n_outliers"] for k, v in outlier_summary.items() if v.get("n_outliers", 0) > 0},
        "skewed_columns": skewed_cols[:10],
        "high_correlations": high_corr_pairs[:10],
        "target_imbalance": imbalance_info,
        "multicollinearity_warning": multicollinearity_warning,
        "distribution_issues": {
            k: v.get("recommended_transform", "none")
            for k, v in dist_summary.items()
            if v.get("recommended_transform", "none") != "none"
        },
    }

    prompt = f"""Analyze these EDA findings and provide an interpretation.

Dataset: {df.shape[0]} rows × {df.shape[1]} columns
Target column: {target_column or 'unknown'}

EDA Facts:
{json.dumps(eda_facts, indent=2)}

For each finding:
1. Categorize it (missing, outlier, skewness, correlation, imbalance, leakage, multicollinearity, general)
2. Rate severity (low, medium, high, critical)
3. Describe the issue
4. Recommend an action

Also provide:
- overall_data_health (good/fair/poor/critical)
- top_priority_actions (ordered list)
- summary paragraph"""

    interpretation = llm.decide(prompt, EDAReport)

    if interpretation is None:
        log.log("llm_failed", "LLM unavailable — returning raw EDA facts", "FAILED")
        return {"eda_facts": eda_facts, "interpretation": None, "log": log.to_dict()}

    log.log(
        "eda_complete",
        f"Health: {interpretation['overall_data_health']}. "
        f"Findings: {len(interpretation.get('findings', []))}. "
        f"Priority actions: {len(interpretation.get('top_priority_actions', []))}.",
        "LLM",
    )

    # ══════════════════════════════════════════
    #  6. Generate Visual EDA Charts
    # ══════════════════════════════════════════
    charts_generated = []
    try:
        plt.style.use('dark_background')

        # Chart 1: Correlation Heatmap
        if len(num_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_data = df[num_cols[:15]].corr()  # Limit to 15 cols for readability
            sns.heatmap(corr_data, annot=len(num_cols) <= 10, cmap='coolwarm',
                       center=0, fmt='.2f', ax=ax, linewidths=0.5)
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(get_data_path('eda_correlation.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append('eda_correlation.png')

        # Chart 2: Missing Values Chart
        if missing_analysis:
            fig, ax = plt.subplots(figsize=(10, 5))
            cols = list(missing_analysis.keys())[:20]
            pcts = [missing_analysis[c]['pct'] for c in cols]
            bars = ax.barh(cols, pcts, color='#f43f5e', edgecolor='#ffffff22')
            ax.set_xlabel('Missing %', fontsize=12)
            ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
            for bar, pct in zip(bars, pcts):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{pct:.1f}%', va='center', fontsize=9)
            plt.tight_layout()
            fig.savefig(get_data_path('eda_missing.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append('eda_missing.png')

        # Chart 3: Skewness Histogram
        if skewed_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            cols = [s['column'] for s in skewed_cols[:15]]
            skews = [s['skewness'] for s in skewed_cols[:15]]
            colors = ['#f59e0b' if abs(s) > 2 else '#6366f1' for s in skews]
            ax.barh(cols, skews, color=colors, edgecolor='#ffffff22')
            ax.axvline(x=0, color='white', linestyle='--', alpha=0.3)
            ax.set_xlabel('Skewness', fontsize=12)
            ax.set_title('Feature Skewness (|skew| > 1)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            fig.savefig(get_data_path('eda_skewness.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append('eda_skewness.png')

        # Chart 4: Class Imbalance Chart
        if imbalance_info and imbalance_info.get('class_distribution'):
            fig, ax = plt.subplots(figsize=(8, 5))
            dist = imbalance_info['class_distribution']
            classes = list(dist.keys())
            counts = list(dist.values())
            colors_palette = sns.color_palette('Set2', len(classes))
            ax.bar(classes, counts, color=colors_palette, edgecolor='#ffffff22')
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Target Class Distribution ({target_column})',
                        fontsize=14, fontweight='bold')
            for i, (cls, cnt) in enumerate(zip(classes, counts)):
                ax.text(i, cnt + max(counts)*0.02, str(cnt), ha='center', fontsize=10)
            plt.tight_layout()
            fig.savefig(get_data_path('eda_imbalance.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
            charts_generated.append('eda_imbalance.png')

        log.log('charts_generated', f'Generated {len(charts_generated)} EDA charts: {charts_generated}', 'DETERMINISTIC')
    except Exception as e:
        log.log('chart_error', f'Chart generation failed: {e}', 'DETERMINISTIC')

    # Save EDA report
    report = {
        "eda_facts": eda_facts,
        "interpretation": interpretation,
        "outlier_details": outlier_summary,
        "distribution_details": dist_summary,
        "charts": charts_generated,
    }
    with open(get_data_path("eda_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return {**report, "log": log.to_dict()}


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

    # Extract target column
    target_col = None
    try:
        msg_data = json.loads(text)
        target_col = msg_data.get("target_column")
    except (json.JSONDecodeError, TypeError):
        pass

    result = await execute_eda(target_column=target_col)

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
    uvicorn.run(app, host="0.0.0.0", port=8207)
