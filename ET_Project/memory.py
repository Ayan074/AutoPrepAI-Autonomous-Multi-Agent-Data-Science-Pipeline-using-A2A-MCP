"""
Lightweight Pipeline Memory — JSON-based execution memory.

Stores:
  - execution_history: all agent/tool calls with results
  - failure_history: all errors with context
  - model_results: all model training outcomes
  - dataset_states: shape/quality snapshots at each stage
  - reasoning_traces: LLM reasoning for key decisions

No vector DBs. No complex storage. Just JSON dicts + disk persistence.
"""

from __future__ import annotations

import json
import logging
import os
import time
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineMemory:
    """Thread-safe in-memory pipeline state with JSON persistence."""

    def __init__(self, working_dir: str = None):
        self._lock = threading.Lock()
        self.working_dir = working_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "workdir"
        )

        self.execution_history: List[Dict[str, Any]] = []
        self.failure_history: List[Dict[str, Any]] = []
        self.model_results: List[Dict[str, Any]] = []
        self.dataset_states: List[Dict[str, Any]] = []
        self.reasoning_traces: List[Dict[str, Any]] = []
        self.reflection_log: List[Dict[str, Any]] = []

    # ── Recording Methods ──

    def record_step(self, step_id: int, action_type: str, target: str,
                    result: dict, success: bool, elapsed_s: float = 0,
                    reason: str = "", confidence: float = 0.0):
        """Record an executed pipeline step."""
        with self._lock:
            entry = {
                "step_id": step_id,
                "action_type": action_type,
                "target": target,
                "success": success,
                "elapsed_s": elapsed_s,
                "reason": reason,
                "confidence": confidence,
                "timestamp": time.time(),
                "result_preview": str(result)[:500],
            }
            self.execution_history.append(entry)

            if not success:
                self.failure_history.append({
                    **entry,
                    "error": result.get("error", "Unknown error"),
                })

    def record_model(self, model_name: str, metrics: dict,
                     problem_type: str, iteration: int = 0):
        """Record a model training result."""
        with self._lock:
            self.model_results.append({
                "model": model_name,
                "metrics": metrics,
                "problem_type": problem_type,
                "iteration": iteration,
                "timestamp": time.time(),
            })

    def record_dataset_state(self, stage: str, rows: int, cols: int,
                             missing: int = 0, categorical: int = 0):
        """Record a dataset shape snapshot."""
        with self._lock:
            self.dataset_states.append({
                "stage": stage,
                "rows": rows,
                "columns": cols,
                "missing_values": missing,
                "categorical_columns": categorical,
                "timestamp": time.time(),
            })

    def record_reasoning(self, agent: str, decision: str,
                         reasoning: str, confidence: float = 0.0):
        """Record an LLM reasoning trace."""
        with self._lock:
            self.reasoning_traces.append({
                "agent": agent,
                "decision": decision,
                "reasoning": reasoning,
                "confidence": confidence,
                "timestamp": time.time(),
            })

    def record_reflection(self, iteration: int, diagnosis: str,
                          proposed_fix: str, outcome: str = "pending"):
        """Record a reflection loop iteration."""
        with self._lock:
            self.reflection_log.append({
                "iteration": iteration,
                "diagnosis": diagnosis,
                "proposed_fix": proposed_fix,
                "outcome": outcome,
                "timestamp": time.time(),
            })

    # ── Query Methods ──

    def get_failure_summary(self) -> str:
        """Return a text summary of all failures for LLM context."""
        with self._lock:
            if not self.failure_history:
                return "No failures recorded."
            lines = []
            for f in self.failure_history[-5:]:  # Last 5 failures
                lines.append(
                    f"- Step {f['step_id']}: {f['target']} failed — {f['error']}"
                )
            return "\n".join(lines)

    def get_model_summary(self) -> str:
        """Return a text summary of all model results for LLM context."""
        with self._lock:
            if not self.model_results:
                return "No models trained yet."
            lines = []
            for m in self.model_results:
                lines.append(
                    f"- {m['model']} (iter {m['iteration']}): {m['metrics']}"
                )
            return "\n".join(lines)

    def get_best_model_score(self) -> Optional[float]:
        """Return the best primary metric score across all models."""
        with self._lock:
            if not self.model_results:
                return None
            scores = []
            for m in self.model_results:
                metrics = m.get("metrics", {})
                # Try classification metric first, then regression
                score = metrics.get("f1_weighted") or metrics.get("r2") or 0
                scores.append(score)
            return max(scores) if scores else None

    def get_context_for_prompt(self) -> str:
        """Return a compact memory summary for LLM decision prompts."""
        with self._lock:
            parts = []

            if self.dataset_states:
                latest = self.dataset_states[-1]
                parts.append(
                    f"Dataset state: {latest['rows']} rows × {latest['columns']} cols, "
                    f"{latest['missing_values']} missing, {latest['categorical_columns']} categorical "
                    f"(stage: {latest['stage']})"
                )

            if self.failure_history:
                parts.append(f"Failures: {len(self.failure_history)} total")
                last = self.failure_history[-1]
                parts.append(f"  Last failure: {last['target']} — {last['error']}")

            if self.model_results:
                best = max(
                    self.model_results,
                    key=lambda m: m['metrics'].get('f1_weighted', m['metrics'].get('r2', 0))
                )
                parts.append(
                    f"Best model: {best['model']} — {best['metrics']}"
                )

            if self.reflection_log:
                last_r = self.reflection_log[-1]
                parts.append(
                    f"Last reflection: {last_r['diagnosis'][:100]}... "
                    f"Fix: {last_r['proposed_fix'][:100]}..."
                )

            return "\n".join(parts) if parts else "No memory recorded yet."

    # ── Persistence ──

    def save_to_disk(self):
        """Persist memory to JSON file."""
        path = os.path.join(self.working_dir, "pipeline_memory.json")
        os.makedirs(self.working_dir, exist_ok=True)
        with self._lock:
            data = {
                "execution_history": self.execution_history,
                "failure_history": self.failure_history,
                "model_results": self.model_results,
                "dataset_states": self.dataset_states,
                "reasoning_traces": self.reasoning_traces,
                "reflection_log": self.reflection_log,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Memory saved to {path}")

    def load_from_disk(self):
        """Load memory from JSON file if it exists."""
        path = os.path.join(self.working_dir, "pipeline_memory.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            with self._lock:
                self.execution_history = data.get("execution_history", [])
                self.failure_history = data.get("failure_history", [])
                self.model_results = data.get("model_results", [])
                self.dataset_states = data.get("dataset_states", [])
                self.reasoning_traces = data.get("reasoning_traces", [])
                self.reflection_log = data.get("reflection_log", [])
            logger.info(f"Memory loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Return full memory as a dict (for report generation)."""
        with self._lock:
            return {
                "execution_history": self.execution_history,
                "failure_history": self.failure_history,
                "model_results": self.model_results,
                "dataset_states": self.dataset_states,
                "reasoning_traces": self.reasoning_traces,
                "reflection_log": self.reflection_log,
            }
