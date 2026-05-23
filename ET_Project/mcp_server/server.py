"""
AutoPrepAI — Real MCP Server (FastMCP).

Exposes deterministic data tools via the Model Context Protocol.
LLM decides WHICH tools to call and WHY.
Tools EXECUTE deterministically (pandas/sklearn).

Run with:
    fastmcp run mcp_server/server.py:mcp --transport http --port 8100

Or programmatically:
    from fastmcp import Client
    async with Client("http://localhost:8100/mcp") as client:
        result = await client.call_tool("describe_data", {"csv_path": "data.csv"})
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from scipy import stats
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

RANDOM_STATE = 42

mcp = FastMCP("AutoPrepAI Data Tools")


# ═══════════════════════════════════════════════
#  TOOL 1: describe_data
# ═══════════════════════════════════════════════

@mcp.tool
def describe_data(csv_path: str) -> dict:
    """Analyze a CSV dataset and return comprehensive metadata.

    Returns shape, dtypes, basic stats, missing info, sample values,
    duplicate count, and memory usage.
    """
    df = pd.read_csv(csv_path)

    # Basic shape
    n_rows, n_cols = df.shape

    # Column info
    column_info = {}
    for col in df.columns:
        series = df[col]
        info: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique()),
            "n_missing": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().mean() * 100, 2),
            "sample_values": [str(v) for v in series.dropna().head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            info["mean"] = round(float(desc.get("mean", 0)), 4)
            info["std"] = round(float(desc.get("std", 0)), 4)
            info["min"] = round(float(desc.get("min", 0)), 4)
            info["max"] = round(float(desc.get("max", 0)), 4)
            info["median"] = round(float(series.median()), 4)
            info["skewness"] = round(float(series.skew()), 4)
        column_info[col] = info

    return {
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "column_info": column_info,
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }


# ═══════════════════════════════════════════════
#  TOOL 2: check_missing
# ═══════════════════════════════════════════════

@mcp.tool
def check_missing(csv_path: str) -> dict:
    """Check missing values per column.

    Returns per-column missing count, percentage, and severity classification.
    """
    df = pd.read_csv(csv_path)

    missing_report = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        pct = round(n_missing / len(df) * 100, 2) if len(df) > 0 else 0
        if pct > 50:
            severity = "critical"
        elif pct > 10:
            severity = "moderate"
        elif pct > 0:
            severity = "low"
        else:
            severity = "none"

        missing_report[col] = {
            "count": n_missing,
            "percentage": pct,
            "severity": severity,
        }

    total_missing = int(df.isnull().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])

    return {
        "columns": missing_report,
        "total_missing": total_missing,
        "total_cells": total_cells,
        "overall_missing_pct": round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
    }


# ═══════════════════════════════════════════════
#  TOOL 3: correlation_analysis
# ═══════════════════════════════════════════════

@mcp.tool
def correlation_analysis(csv_path: str, threshold: float = 0.5) -> dict:
    """Compute correlation matrix for numeric columns.

    Returns pairs with |correlation| > threshold.
    """
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        return {"pairs": [], "n_numeric_columns": len(num_cols)}

    corr_matrix = df[num_cols].corr()

    pairs = []
    seen = set()
    for i, col_a in enumerate(num_cols):
        for j, col_b in enumerate(num_cols):
            if i >= j:
                continue
            val = float(corr_matrix.loc[col_a, col_b])
            if abs(val) > threshold:
                pair_key = tuple(sorted([col_a, col_b]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": round(val, 4),
                        "strength": "strong" if abs(val) > 0.8 else "moderate",
                    })

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "pairs": pairs[:30],
        "n_numeric_columns": len(num_cols),
        "threshold_used": threshold,
    }


# ═══════════════════════════════════════════════
#  TOOL 4: detect_outliers
# ═══════════════════════════════════════════════

@mcp.tool
def detect_outliers(csv_path: str, method: str = "iqr") -> dict:
    """Detect outliers in numeric columns using IQR or Z-score method.

    Args:
        csv_path: Path to the CSV file.
        method: Detection method — 'iqr' (default) or 'zscore'.

    Returns:
        Per-column outlier counts, percentages, and sample outlier values.
    """
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_report = {}
    total_outliers = 0

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (series < lower) | (series > upper)
        else:  # zscore
            z = np.abs(stats.zscore(series))
            mask = z > 3
            lower, upper = float(series.mean() - 3 * series.std()), float(series.mean() + 3 * series.std())

        n_outliers = int(mask.sum())
        total_outliers += n_outliers

        outlier_report[col] = {
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(series) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "method": method,
            "sample_outliers": [round(float(v), 4) for v in series[mask].head(5).tolist()],
        }

    return {
        "columns": outlier_report,
        "total_outliers": total_outliers,
        "method": method,
        "n_columns_analyzed": len(num_cols),
    }


# ═══════════════════════════════════════════════
#  TOOL 5: distribution_analysis
# ═══════════════════════════════════════════════

@mcp.tool
def distribution_analysis(csv_path: str) -> dict:
    """Analyze distributions of numeric columns.

    Returns skewness, kurtosis, normality test results, and
    recommended transforms for each column.
    """
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    distributions = {}
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 8:
            continue

        skew_val = float(series.skew())
        kurt_val = float(series.kurtosis())

        # Normality test (Shapiro for small samples, D'Agostino for larger)
        is_normal = False
        p_value = 0.0
        try:
            if len(series) <= 5000:
                sample = series.sample(min(len(series), 5000), random_state=42)
                _, p_value = stats.shapiro(sample)
            else:
                _, p_value = stats.normaltest(series)
            is_normal = p_value > 0.05
        except Exception:
            pass

        # Recommend transform
        if abs(skew_val) > 2 and (series > 0).all():
            recommended_transform = "log"
        elif abs(skew_val) > 1:
            recommended_transform = "sqrt" if (series >= 0).all() else "none"
        else:
            recommended_transform = "none"

        distributions[col] = {
            "skewness": round(skew_val, 4),
            "kurtosis": round(kurt_val, 4),
            "is_normal": is_normal,
            "normality_p_value": round(float(p_value), 6),
            "recommended_transform": recommended_transform,
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
        }

    return {
        "columns": distributions,
        "n_columns_analyzed": len(distributions),
    }


# ═══════════════════════════════════════════════
#  TOOL 6: train_model
# ═══════════════════════════════════════════════

MODEL_REGISTRY = {
    "classification": {
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "DecisionTree": DecisionTreeClassifier,
        "SVM": SVC,
    },
    "regression": {
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "DecisionTree": DecisionTreeRegressor,
        "SVR": SVR,
    },
}


@mcp.tool
def train_model(csv_path: str, target_column: str,
                problem_type: str, model_name: str) -> dict:
    """Train a single ML model on the dataset and return metrics.

    Args:
        csv_path: Path to the preprocessed CSV.
        target_column: Name of the target column.
        problem_type: 'classification' or 'regression'.
        model_name: One of the registered model names.

    Returns:
        Training metrics, feature importances, and model details.
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode target if classification + string labels
    le = None
    if problem_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_column)

    # Drop any remaining non-numeric
    X = X.select_dtypes(include=[np.number])
    if X.empty:
        return {"error": "No numeric features available for training."}

    # Fill any remaining NaN
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    registry = MODEL_REGISTRY.get(problem_type, {})
    model_class = registry.get(model_name)
    if model_class is None:
        return {
            "error": f"Unknown model: {model_name}. "
                     f"Available: {list(registry.keys())}"
        }

    try:
        model = model_class(random_state=RANDOM_STATE) if "random_state" in model_class().get_params() else model_class()
    except TypeError:
        model = model_class()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}
    if problem_type == "classification":
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
    else:
        metrics = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }

    # Feature importances
    importances = {}
    if hasattr(model, "feature_importances_"):
        for feat, imp in zip(X.columns, model.feature_importances_):
            importances[feat] = round(float(imp), 4)
    elif hasattr(model, "coef_"):
        coefs = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
        for feat, coef in zip(X.columns, coefs):
            importances[feat] = round(float(abs(coef)), 4)

    return {
        "model": model_name,
        "problem_type": problem_type,
        "metrics": metrics,
        "feature_importances": dict(sorted(importances.items(), key=lambda x: -x[1])[:15]),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "n_features": X.shape[1],
        "target_classes": list(le.classes_) if le else None,
    }


# ═══════════════════════════════════════════════
#  TOOL 7: evaluate_model
# ═══════════════════════════════════════════════

@mcp.tool
def evaluate_model(csv_path: str, target_column: str,
                   model_name: str, problem_type: str,
                   cv_folds: int = 5) -> dict:
    """Evaluate a model using cross-validation for robust performance metrics.

    Args:
        csv_path: Path to the preprocessed CSV.
        target_column: Name of the target column.
        model_name: One of the registered model names.
        problem_type: 'classification' or 'regression'.
        cv_folds: Number of cross-validation folds.

    Returns:
        Cross-validated scores with mean and std.
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if problem_type == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_column)

    X = X.select_dtypes(include=[np.number]).fillna(0)
    if X.empty:
        return {"error": "No numeric features available."}

    registry = MODEL_REGISTRY.get(problem_type, {})
    model_class = registry.get(model_name)
    if model_class is None:
        return {"error": f"Unknown model: {model_name}"}

    try:
        model = model_class(random_state=RANDOM_STATE) if "random_state" in model_class().get_params() else model_class()
    except TypeError:
        model = model_class()

    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    folds = min(cv_folds, len(X))

    try:
        scores = cross_val_score(model, X, y, cv=folds, scoring=scoring)
        return {
            "model": model_name,
            "scoring": scoring,
            "cv_folds": folds,
            "scores": [round(float(s), 4) for s in scores],
            "mean_score": round(float(scores.mean()), 4),
            "std_score": round(float(scores.std()), 4),
        }
    except Exception as e:
        return {"error": f"Cross-validation failed: {str(e)[:200]}"}


# ═══════════════════════════════════════════════
#  TOOL 8: get_pipeline_status (informational)
# ═══════════════════════════════════════════════

@mcp.tool
def get_pipeline_status(working_dir: str) -> dict:
    """Check the current state of the pipeline working directory.

    Returns what files exist (raw data, processed data, models, etc.).
    """
    status = {"working_dir": working_dir, "files": {}}
    if os.path.exists(working_dir):
        for f in os.listdir(working_dir):
            fpath = os.path.join(working_dir, f)
            if os.path.isfile(fpath):
                status["files"][f] = {
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                    "extension": os.path.splitext(f)[1],
                }
    return status


if __name__ == "__main__":
    mcp.run()
