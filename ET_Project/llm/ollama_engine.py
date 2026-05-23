"""
OllamaEngine — Local LLM for AutoPrepAI.

All decision-making flows through this engine.
Execution (pandas/sklearn) remains deterministic.
The LLM only decides WHAT to do, never HOW to execute.

Uses Pydantic schemas for enforced structured JSON output.
Runs on CPU with llama3.2 (3B params).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Literal, Optional, Type

import ollama
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
#  SYSTEM PROMPT — Injected into EVERY LLM call
# ═══════════════════════════════════════════════

SYSTEM_PROMPT = """You are an autonomous AI system controlling a multi-agent data science pipeline.

ROLE: You are the DECISION-MAKER. You decide WHAT should be done.
Execution is handled by deterministic code (pandas, sklearn). You never execute directly.

CORE OBJECTIVE: Produce a high-quality, ML-ready dataset while preserving data integrity.

STRICT RULES:
1. Return ONLY valid JSON matching the requested schema.
2. ALWAYS include "reason" (concise, grounded in data) and "confidence" (0.0-1.0).
3. Base decisions on actual data statistics provided to you. NEVER invent numbers.
4. Preserve at least 80% of original rows. Prefer imputation over dropping.
5. Priority: impute > transform > engineer > drop (last resort).
6. If uncertain, choose the safest option and lower confidence.

SAFE DEFAULTS:
- Numerical missing → median (robust to outliers)
- Categorical missing → mode (most common value)
- Encoding: low cardinality (≤10) → onehot, medium (11-50) → label, high (>50) → frequency
- Outliers: prefer capping over removal
- Scaling: standard for normal, robust if outliers present

CONSISTENCY: Similar columns should get similar treatment. If different, justify why."""


# ═══════════════════════════════════════════════
#  PYDANTIC SCHEMAS — Enforced JSON Output
# ═══════════════════════════════════════════════

class ColumnClassification(BaseModel):
    """Classification of a single column."""
    column: str
    col_type: Literal["numerical", "categorical", "boolean", "datetime", "text", "id"]
    reason: str


class DatasetProfile(BaseModel):
    """LLM's analysis of the full dataset."""
    column_classifications: List[ColumnClassification]
    target_column: Optional[str] = None
    target_reasoning: str = ""
    detected_domain: str = "generic"
    dataset_summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class QualityAssessment(BaseModel):
    """LLM's assessment of data quality."""
    quality_score: int = Field(ge=0, le=100)
    quality_grade: Literal["A", "B", "C", "D", "F"]
    critical_issues: List[str] = []
    priority_actions: List[str] = []
    risky_columns: List[str] = []
    assessment: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ImputationDecision(BaseModel):
    """LLM's decision for imputing a single column."""
    column: str
    strategy: Literal["mean", "median", "mode", "interpolate", "ffill", "zero", "unknown"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class ImputationPlan(BaseModel):
    """LLM's imputation plan for all columns with missing values."""
    decisions: List[ImputationDecision]
    overall_reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class EncodingDecision(BaseModel):
    """LLM's encoding decision for a single column."""
    column: str
    encoding: Literal["onehot", "label", "frequency", "binary", "tfidf"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class EncodingPlan(BaseModel):
    """LLM's encoding plan for all categorical columns."""
    decisions: List[EncodingDecision]
    overall_reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class OutlierDecision(BaseModel):
    """LLM's outlier treatment for a single column."""
    column: str
    action: Literal["cap", "remove", "keep", "log_transform"]
    detection_method: Literal["iqr", "zscore"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class OutlierPlan(BaseModel):
    """LLM's outlier plan for all numerical columns."""
    decisions: List[OutlierDecision]
    overall_reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ScalingDecision(BaseModel):
    """LLM's scaling decision for a single column."""
    column: str
    scaler: Literal["standard", "minmax", "robust", "none"]
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class ScalingPlan(BaseModel):
    """LLM's scaling plan for all numerical columns."""
    decisions: List[ScalingDecision]
    overall_reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class FeatureTransform(BaseModel):
    """A single feature transform recommendation."""
    column: str
    transform: Literal["log", "sqrt", "square", "bin", "interaction"]
    interaction_with: Optional[str] = None
    reason: str


class FeatureEngineeringPlan(BaseModel):
    """LLM's feature engineering recommendations."""
    transforms: List[FeatureTransform] = []
    overall_reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ModelSelectionDecision(BaseModel):
    """LLM's model selection from AutoML results."""
    best_model: str
    justification: str
    runner_up: str = ""
    warnings: List[str] = []
    confidence: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """LLM's validation of the processed dataset."""
    is_ml_ready: bool
    quality_grade: Literal["A", "B", "C", "D", "F"]
    quality_score: int = Field(ge=0, le=100)
    issues_found: List[str] = []
    recommendations: List[str] = []
    assessment: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class OrchestrationStep(BaseModel):
    """LLM's next-step decision for the orchestration loop."""
    step_id: int
    action_type: Literal["agent_call", "tool_call", "finish"]
    target: str
    input: Dict[str, Any] = {}
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)


class A2ACollaborationDecision(BaseModel):
    """LLM decides which peer agents to consult."""
    should_collaborate: bool
    target_agents: List[str] = []
    queries: Dict[str, str] = {}
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ToolSelectionDecision(BaseModel):
    """LLM decides which MCP tools to invoke."""
    use_tools: bool
    tools: List[Dict[str, Any]] = []
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


# ═══════════════════════════════════════════════
#  NEW v5 SCHEMAS — Problem Type, EDA, Reflection, Report
# ═══════════════════════════════════════════════

class ProblemTypeDecision(BaseModel):
    """LLM decides whether this is classification or regression.

    Replaces the hardcoded `if n_unique <= 20 => classification` rule.
    The LLM reasons from column name, dtype, cardinality, sample values.
    """
    problem_type: Literal["classification", "regression"]
    target_column: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)


class EDAFinding(BaseModel):
    """A single EDA finding."""
    category: Literal[
        "missing", "outlier", "skewness", "correlation",
        "imbalance", "leakage", "multicollinearity", "general"
    ]
    severity: Literal["low", "medium", "high", "critical"]
    description: str
    affected_columns: List[str] = []
    recommendation: str = ""


class EDAReport(BaseModel):
    """LLM's interpretation of EDA findings."""
    findings: List[EDAFinding]
    overall_data_health: Literal["good", "fair", "poor", "critical"]
    top_priority_actions: List[str] = []
    summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0)


class ReflectionAnalysis(BaseModel):
    """LLM's diagnosis of why model performance is poor."""
    root_cause: str
    diagnosis: str
    proposed_fix: str
    fix_target_agent: str = ""  # which agent should re-run
    fix_parameters: Dict[str, Any] = {}  # parameters for the retry
    should_retry: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class ReportSection(BaseModel):
    """A single section of the final report."""
    title: str
    content: str
    key_points: List[str] = []


class ReportNarrative(BaseModel):
    """LLM-generated final explainability report."""
    title: str = "AutoPrepAI Pipeline Report"
    sections: List[ReportSection]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    risks: List[str] = []
    recommendations: List[str] = []


# ═══════════════════════════════════════════════
#  OLLAMA ENGINE
# ═══════════════════════════════════════════════

class OllamaEngine:
    """Local LLM engine via Ollama.

    All decisions in the pipeline flow through this engine.
    Uses Pydantic schema enforcement for structured JSON output.
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.is_available = False
        self.call_log: List[Dict[str, Any]] = []
        self._total_calls = 0
        self._successful_calls = 0
        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is running and the model is available."""
        try:
            models = ollama.list()
            available = [m.model for m in models.models] if hasattr(models, 'models') else []
            # Check if our model (or a variant) is available
            for m in available:
                if self.model in m:
                    self.is_available = True
                    logger.info(f"Ollama connected: {self.model} (available: {m})")
                    return
            # Model not pulled yet
            if available:
                logger.warning(
                    f"Model '{self.model}' not found. Available: {available}. "
                    f"Run: ollama pull {self.model}"
                )
            else:
                logger.warning("Ollama running but no models found. Run: ollama pull llama3.2")
            self.is_available = False
        except Exception as e:
            logger.error(f"Ollama not available: {e}. Run: ollama serve")
            self.is_available = False

    def decide(self, prompt: str, schema: Type[BaseModel],
               context: str = "") -> Optional[Dict[str, Any]]:
        """Ask the LLM for a structured decision.

        Args:
            prompt: The decision question with data context.
            schema: Pydantic model defining the expected JSON structure.
            context: Optional additional context prepended to prompt.

        Returns:
            Parsed dict matching the schema, or None on failure.
        """
        if not self.is_available:
            return None

        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        start = time.time()

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                format=schema.model_json_schema(),
                options={"temperature": 0.2, "num_predict": 4096, "num_ctx": 4096},
            )

            raw_text = response["message"]["content"]
            parsed = schema.model_validate_json(raw_text)
            result = parsed.model_dump()

            elapsed = round(time.time() - start, 2)
            self._total_calls += 1
            self._successful_calls += 1

            self.call_log.append({
                "schema": schema.__name__,
                "prompt_preview": prompt[:200],
                "elapsed_s": elapsed,
                "success": True,
                "result_preview": str(result)[:300],
            })

            logger.info(f"LLM decision ({schema.__name__}): {elapsed}s")
            return result

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            self._total_calls += 1

            self.call_log.append({
                "schema": schema.__name__,
                "prompt_preview": prompt[:200],
                "elapsed_s": elapsed,
                "success": False,
                "error": str(e)[:300],
            })

            logger.error(f"LLM call failed ({schema.__name__}): {e}")
            return None

    def decide_text(self, prompt: str, context: str = "") -> Optional[str]:
        """Ask the LLM for free-form text (e.g., report narrative)."""
        if not self.is_available:
            return None

        full_prompt = f"{context}\n\n{prompt}" if context else prompt

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                options={"temperature": 0.3, "num_predict": 2048},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM text call failed: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for UI display."""
        return {
            "provider": "ollama",
            "model": self.model,
            "is_available": self.is_available,
            "total_calls": self._total_calls,
            "successful_calls": self._successful_calls,
            "success_rate": (
                round(self._successful_calls / self._total_calls * 100, 1)
                if self._total_calls > 0 else 0
            ),
        }
