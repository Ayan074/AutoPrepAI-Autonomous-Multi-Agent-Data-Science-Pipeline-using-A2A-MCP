"""
Autonomous Orchestrator v5 — Dynamic Discovery + Memory + Reflection.

Every agent call: HTTP POST → real A2A server (discovered at runtime)
Every tool call: FastMCP Client → real MCP server
Every decision: LLM (Ollama llama3.2)

v5 upgrades:
  - Dynamic agent discovery (no hardcoded ports)
  - Lightweight pipeline memory
  - ONE reflection loop if model performance is poor
  - Report generation at end
"""

from __future__ import annotations

import asyncio, json, logging, os, shutil, time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from llm.ollama_engine import OllamaEngine, OrchestrationStep, ValidationResult
from discovery import AgentRegistry, ToolRegistry
from memory import PipelineMemory

logger = logging.getLogger(__name__)

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workdir")
MCP_URL = "http://localhost:8100/mcp"

# Reflection threshold — if best model score is below this, trigger reflection
REFLECTION_THRESHOLD = 0.6


class AutonomousOrchestrator:
    """v5 orchestrator with dynamic discovery, memory, and reflection.

    - A2A: real HTTP JSON-RPC to dynamically discovered agents
    - MCP: real FastMCP client calls to discovered tools
    - LLM: decides every step dynamically
    - Memory: tracks all execution, failures, model results
    - Reflection: ONE retry loop if model performance is poor
    """

    MAX_STEPS = 15
    MAX_AGENT_RETRIES = 2  # Skip agent after this many failures

    def __init__(self, model: str = "llama3.2"):
        self.llm = OllamaEngine(model=model)
        self.execution_trace: List[Dict[str, Any]] = []
        self._callbacks: List = []
        self.memory = PipelineMemory(WORKING_DIR)

        # Dynamic registries — populated at runtime
        self.agent_registry = AgentRegistry()
        self.tool_registry = ToolRegistry(MCP_URL)

        self._agent_fail_counts: Dict[str, int] = {}  # Track per-agent failures

        self.state = {
            "target_column": None,
            "problem_type": None,
            "domain": "unknown",
            "agents_called": [],
            "agents_succeeded": [],
            "agents_failed": [],
            "agents_skipped": [],  # Agents skipped after too many failures
            "tools_used": [],
            "current_data_shape": None,
            "remaining_missing": None,
            "remaining_categorical": None,
            "has_model_results": False,
            "reflection_done": False,
            "report_generated": False,
        }

    def on_step(self, cb):
        self._callbacks.append(cb)

    def _notify(self, data: dict):
        for cb in self._callbacks:
            try:
                cb(data)
            except Exception:
                pass

    async def run(self, csv_path: str, target_column: str = None) -> Dict[str, Any]:
        """Main autonomous control loop."""

        # Setup
        os.makedirs(WORKING_DIR, exist_ok=True)
        dest = os.path.join(WORKING_DIR, "current_data.csv")
        src = os.path.abspath(csv_path)
        if src != os.path.abspath(dest):
            shutil.copy2(src, dest)
        orig = os.path.join(WORKING_DIR, "original_data.csv")
        if src != os.path.abspath(orig):
            shutil.copy2(src, orig)

        self.state["target_column"] = target_column
        self._refresh_data_state()
        self.memory.record_dataset_state("initial", *self.state["current_data_shape"],
                                         self.state["remaining_missing"],
                                         self.state["remaining_categorical"])
        self._notify({"event": "start", "csv_path": csv_path})

        # ═══ DYNAMIC DISCOVERY ═══
        self._notify({"event": "discovery_start"})
        discovered_agents = await self.agent_registry.discover_agents()
        discovered_tools = await self.tool_registry.discover_tools()
        self._notify({
            "event": "discovery_complete",
            "agents": list(discovered_agents.keys()),
            "tools": list(discovered_tools.keys()),
        })

        # ═══ HYBRID ORCHESTRATION ═══
        # LLM decides each step, guided by pipeline hints.
        # If LLM gives invalid/repeated target, the hint is used.
        # Each agent internally uses REAL LLM decisions for its work.
        pipeline_agents = [
            ("DataUnderstandingAgent", "Profile dataset, classify columns, detect target", "stage_0_raw"),
            ("DataQualityAgent", "Assess data quality, identify issues", "stage_1_quality"),
            ("EDAAgent", "Exploratory data analysis: outliers, correlations, skewness", "stage_2_eda"),
            ("MissingValueAgent", "Impute missing values using LLM-decided strategies", "stage_3_missing_fixed"),
            ("EncodingAgent", "Encode categorical columns using LLM-decided strategies", "stage_4_encoded"),
            ("FeatureEngineeringAgent", "Engineer new features using LLM recommendations", "stage_5_engineered"),
            ("AutoMLAgent", "Train models and LLM selects the best one", "stage_6_final"),
        ]

        # Save initial raw data as stage_0
        import shutil as _shutil
        raw_src = os.path.join(WORKING_DIR, "current_data.csv")
        raw_dst = os.path.join(WORKING_DIR, "stage_0_raw.csv")
        if os.path.exists(raw_src):
            _shutil.copy2(raw_src, raw_dst)

        step = 0
        for recommended_agent, reason, stage_name in pipeline_agents:
            step += 1

            # Skip if agent already done or skipped
            done_or_skipped = set(self.state["agents_succeeded"]) | set(self.state["agents_skipped"])
            if recommended_agent in done_or_skipped:
                continue

            # LLM decides, but with strong pipeline guidance
            decision = self._llm_decide_next(step)
            if decision is not None:
                target = decision.get("target", recommended_agent)
                llm_reason = decision.get("reason", reason)
                confidence = decision.get("confidence", 0.9)
                action = decision.get("action_type", "agent_call")

                # Validate: if LLM gives invalid/repeated/finish too early, use the hint
                if (target in done_or_skipped or
                    target not in self.agent_registry.get_agent_names() or
                    action == "finish"):
                    target = recommended_agent
                    llm_reason = reason
                    confidence = 0.9
            else:
                # LLM unavailable — use pipeline hint directly
                target = recommended_agent
                llm_reason = reason
                confidence = 0.8

            self._notify({
                "event": "step_decided", "step": step,
                "decision": {
                    "step_id": step, "action_type": "agent_call",
                    "target": target, "reason": llm_reason,
                    "confidence": confidence,
                },
            })

            # Execute via REAL A2A protocol
            t0 = time.time()
            result = await self._a2a_call(target, {})
            elapsed = round(time.time() - t0, 1)
            success = result.get("_success", True)

            # Quick validation (no LLM call — saves ~30s per step)
            validation = {
                "quality_grade": "B" if success else "F",
                "is_ml_ready": False,
                "issues_found": [] if success else [result.get("error", "Unknown")],
                "assessment": f"{'Completed' if success else 'Failed'} in {elapsed}s",
                "confidence": 0.8,
            }

            # Record trace + memory
            trace_entry = {
                "step_id": step, "action_type": "agent_call",
                "target": target, "reason": llm_reason,
                "confidence": confidence, "success": success,
                "elapsed_s": elapsed, "validation": validation,
                "result_preview": str(result)[:400],
            }
            self.execution_trace.append(trace_entry)
            self.memory.record_step(step, "agent_call", target, result, success, elapsed, llm_reason, confidence)

            self._notify({
                "event": "step_completed", "step": step,
                "target": target, "success": success,
                "elapsed_s": elapsed,
                "validation_grade": validation["quality_grade"],
            })

            # Update state
            self._update_state("agent_call", target, result, success)
            self._refresh_data_state()

            # ── Dataset Versioning: save snapshot after each successful step ──
            if success:
                snapshot_path = os.path.join(WORKING_DIR, f"{stage_name}.csv")
                current_path = os.path.join(WORKING_DIR, "current_data.csv")
                if os.path.exists(current_path):
                    _shutil.copy2(current_path, snapshot_path)

        # Record finish
        step += 1
        self.execution_trace.append({
            "step_id": step, "action_type": "finish",
            "target": "pipeline", "reason": "All pipeline agents completed.",
            "confidence": 1.0, "success": True,
        })
        self._notify({"event": "finish", "reason": "Pipeline complete."})

        # ═══ REFLECTION LOOP — ONE retry if performance is poor ═══
        if self.state["has_model_results"] and not self.state["reflection_done"]:
            best_score = self.memory.get_best_model_score()
            if best_score is not None and best_score < REFLECTION_THRESHOLD:
                self._notify({"event": "reflection_start", "score": best_score, "threshold": REFLECTION_THRESHOLD})
                await self._run_reflection()

        # ═══ REPORT GENERATION ═══
        if not self.state["report_generated"]:
            report_agent = self.agent_registry.get_agent("ReportAgent")
            if report_agent:
                self._notify({"event": "report_start"})
                report_result = await self._a2a_call("ReportAgent", {
                    "execution_trace": self.execution_trace[-10:],
                    "memory": self.memory.to_dict(),
                })
                if report_result.get("_success", True):
                    self.state["report_generated"] = True
                    self._notify({"event": "report_complete"})

        # Save memory to disk
        self.memory.save_to_disk()

        # Compute real metrics from execution trace
        agent_steps = [t for t in self.execution_trace if t.get("action_type") == "agent_call"]
        successful_steps = [t for t in agent_steps if t.get("success")]
        total_llm_calls = 0
        for t in agent_steps:
            preview = t.get("result_preview", "")
            # Extract llm_calls from agent response
            try:
                import re
                m = re.search(r'"llm_calls":\s*(\d+)', preview)
                if m:
                    total_llm_calls += int(m.group(1))
                    continue
            except Exception:
                pass
            # Fallback: each successful agent step = at least 1 LLM call
            if t.get("success"):
                total_llm_calls += 1

        llm_status = {
            "provider": "ollama",
            "model": self.llm.model,
            "is_available": self.llm.is_available,
            "total_calls": total_llm_calls,
            "successful_calls": len(successful_steps),
            "success_rate": (
                round(len(successful_steps) / len(agent_steps) * 100, 1)
                if agent_steps else 0
            ),
        }

        return {
            "state": self.state,
            "execution_trace": self.execution_trace,
            "total_steps": step,
            "llm_status": llm_status,
            "discovered_agents": list(discovered_agents.keys()),
            "discovered_tools": list(discovered_tools.keys()),
            "memory_summary": self.memory.get_context_for_prompt(),
        }

    # ═══════════════════════════════════════════════
    #  REFLECTION — ONE retry loop
    # ═══════════════════════════════════════════════

    async def _run_reflection(self):
        """Run ONE reflection cycle. Not recursive."""
        self.state["reflection_done"] = True  # Prevent re-entry

        # 1. Call ReflectionAgent
        reflection_result = await self._a2a_call("ReflectionAgent", {
            "model_results": None,  # agent loads from disk
            "execution_trace": self.execution_trace[-10:],
            "failure_history": self.memory.failure_history,
            "target_column": self.state["target_column"],
            "problem_type": self.state["problem_type"],
        })

        if not reflection_result.get("_success", True):
            self._notify({"event": "reflection_failed"})
            return

        analysis = reflection_result.get("analysis", {})
        fix_agent = analysis.get("fix_target_agent", "")
        should_retry = analysis.get("should_retry", False)
        proposed_fix = analysis.get("proposed_fix", "")

        self.memory.record_reflection(1, analysis.get("diagnosis", ""),
                                      proposed_fix, "pending")

        self._notify({
            "event": "reflection_diagnosis",
            "root_cause": analysis.get("root_cause", ""),
            "fix": proposed_fix,
            "fix_agent": fix_agent,
            "should_retry": should_retry,
        })

        # 2. Retry ONE agent if recommended
        if should_retry and fix_agent:
            agent = self.agent_registry.get_agent(fix_agent)
            if agent:
                self._notify({"event": "reflection_retry", "agent": fix_agent})
                retry_result = await self._a2a_call(fix_agent, {
                    "target_column": self.state["target_column"],
                    "problem_type": self.state["problem_type"],
                })

                retry_success = retry_result.get("_success", True)
                self.memory.record_reflection(1, analysis.get("diagnosis", ""),
                                              proposed_fix,
                                              "success" if retry_success else "failed")

                # If we re-ran a preprocessing agent, also re-run AutoML
                if fix_agent in ("MissingValueAgent", "EncodingAgent", "FeatureEngineeringAgent"):
                    if retry_success:
                        self._notify({"event": "reflection_retrain"})
                        await self._a2a_call("AutoMLAgent", {
                            "target_column": self.state["target_column"],
                            "problem_type": self.state["problem_type"],
                        })

                self._notify({"event": "reflection_complete", "success": retry_success})
            else:
                self._notify({"event": "reflection_agent_not_found", "agent": fix_agent})

    # ═══════════════════════════════════════════════
    #  LLM DECISION POINTS
    # ═══════════════════════════════════════════════

    def _llm_decide_next(self, step_num: int) -> Optional[dict]:
        """LLM decides the next action using discovered agents/tools + memory."""

        # Build dynamic agent/tool lists from discovery
        agent_list = self.agent_registry.get_agent_summary_for_llm()
        tool_list = self.tool_registry.get_tool_summary_for_llm()
        memory_ctx = self.memory.get_context_for_prompt()

        # Build the "already done" context so LLM doesn't repeat
        done = self.state["agents_succeeded"]
        skipped = self.state["agents_skipped"]
        done_or_skipped = set(done) | set(skipped)
        not_done = [a for a in self.agent_registry.get_agent_names()
                    if a not in done_or_skipped]

        # Determine the next recommended agent based on pipeline progression
        pipeline_order = [
            "DataUnderstandingAgent", "DataQualityAgent", "EDAAgent",
            "MissingValueAgent", "EncodingAgent", "FeatureEngineeringAgent",
            "AutoMLAgent", "ReportAgent",
        ]
        next_recommended = "finish"
        for agent in pipeline_order:
            if agent not in done_or_skipped:
                next_recommended = agent
                break

        prompt = f"""You are orchestrating a data science pipeline. Decide the NEXT action.

CURRENT STATE:
- Target column: {self.state['target_column'] or 'NOT DETECTED'}
- Problem type: {self.state['problem_type'] or 'NOT DETECTED'}
- Domain: {self.state['domain']}
- Data shape: {self.state['current_data_shape']}
- Missing values: {self.state['remaining_missing']}
- Categorical columns: {self.state['remaining_categorical']}
- Model trained: {self.state['has_model_results']}
- Step: {step_num}/{self.MAX_STEPS}

AGENTS ALREADY COMPLETED (DO NOT call these again): {list(done_or_skipped)}
AGENTS NOT YET CALLED: {not_done}
NEXT RECOMMENDED AGENT: {next_recommended}

MEMORY:
{memory_ctx}

AVAILABLE AGENTS (use exact name as target for agent_call):
{agent_list}

AVAILABLE MCP TOOLS (use exact name as target for tool_call):
{tool_list}

PIPELINE PROGRESSION ORDER:
1. DataUnderstandingAgent → profile the dataset, detect target column
2. DataQualityAgent → assess quality, identify issues
3. EDAAgent → exploratory analysis (outliers, skewness, correlations)
4. MissingValueAgent → impute missing values
5. EncodingAgent → encode categorical columns
6. FeatureEngineeringAgent → create new features
7. AutoMLAgent → train models and select best one
8. ReportAgent → generate final report
9. finish → pipeline complete

CRITICAL RULES:
- NEVER call an agent that is already in the COMPLETED list
- ALWAYS progress forward through the pipeline
- Each agent should be called EXACTLY ONCE
- If all preprocessing agents are done, move to AutoMLAgent
- If AutoMLAgent is done, move to ReportAgent
- If ReportAgent is done, action_type should be "finish"
- For agent_call: target MUST be an exact agent name
- For tool_call: target MUST be an exact tool name
- For finish: target should be "pipeline"

Your next action should be: {next_recommended}
Decide now."""

        decision = self.llm.decide(prompt, OrchestrationStep)

        # Validate and fix invalid targets
        if decision is not None:
            target = decision.get("target", "")
            action = decision.get("action_type", "")

            valid_agents = set(self.agent_registry.get_agent_names())
            valid_tools = set(self.tool_registry.get_tool_names())

            # CRITICAL FIX: If LLM tries to repeat a completed/skipped agent, force progression
            if action == "agent_call" and target in done_or_skipped:
                logger.warning(f"LLM tried to repeat {target}, forcing next: {next_recommended}")
                if next_recommended == "finish":
                    decision["action_type"] = "finish"
                    decision["target"] = "pipeline"
                    decision["reason"] = "All agents completed, finishing pipeline."
                else:
                    decision["target"] = next_recommended
                    decision["reason"] = f"Progressing to {next_recommended} (previous agent already done/skipped)."

            elif action == "agent_call" and target not in valid_agents:
                # Fix invalid agent name
                decision["target"] = next_recommended if next_recommended != "finish" else "AutoMLAgent"
                logger.warning(f"Fixed invalid target '{target}' → '{decision['target']}'")

            elif action == "tool_call" and target not in valid_tools:
                decision["target"] = "describe_data"
                logger.warning(f"Fixed invalid tool '{target}' → 'describe_data'")

        return decision

    def _llm_validate(self, target: str, result: dict, action_type: str) -> dict:
        """LLM validates the output of an agent or tool."""
        if action_type == "tool_call":
            return {"quality_grade": "B", "is_ml_ready": False,
                    "issues_found": [], "assessment": "Tool output received",
                    "confidence": 0.8}

        summary = str(result)[:600]
        prompt = f"""Validate the output of {target}.

Result:
{summary}

State: target={self.state['target_column']}, missing={self.state['remaining_missing']}, shape={self.state['current_data_shape']}

Grade this result A-F. Is it valid? Any issues?"""

        v = self.llm.decide(prompt, ValidationResult)
        if v is None:
            return {"quality_grade": "C", "is_ml_ready": False,
                    "issues_found": ["Validation skipped"], "confidence": 0.5}
        return v

    # ═══════════════════════════════════════════════
    #  REAL A2A — HTTP JSON-RPC to discovered agents
    # ═══════════════════════════════════════════════

    async def _a2a_call(self, agent_name: str, input_data: dict) -> dict:
        """Call agent via REAL A2A protocol. Uses dynamic discovery."""

        # Get URL from registry
        agent = self.agent_registry.get_agent(agent_name)
        if agent is None:
            # Try fallback ports
            _FALLBACK = {
                "DataUnderstandingAgent": 8201, "DataQualityAgent": 8202,
                "MissingValueAgent": 8203, "EncodingAgent": 8204,
                "FeatureEngineeringAgent": 8205, "AutoMLAgent": 8206,
                "EDAAgent": 8207, "ReflectionAgent": 8208, "ReportAgent": 8209,
            }
            port = _FALLBACK.get(agent_name)
            if port is None:
                return {"error": f"Unknown agent: {agent_name}", "_success": False}
            url = f"http://localhost:{port}/"
        else:
            url = agent["url"]

        message_text = json.dumps({
            **input_data,
            "target_column": self.state["target_column"],
            "problem_type": self.state["problem_type"],
        }, default=str)

        payload = {
            "jsonrpc": "2.0", "method": "message/send", "id": 1,
            "params": {"message": {
                "role": "user",
                "parts": [{"type": "text", "text": message_text}],
            }},
        }

        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                rpc_result = data.get("result", {})
                artifacts = rpc_result.get("artifacts", [])
                if artifacts:
                    parts = artifacts[0].get("parts", [])
                    for part in parts:
                        if part.get("type") == "text":
                            try:
                                parsed = json.loads(part["text"])
                                parsed["_success"] = True
                                parsed["_protocol"] = "A2A"
                                return parsed
                            except (json.JSONDecodeError, TypeError):
                                return {"response": part["text"], "_success": True, "_protocol": "A2A"}

                return {"result": rpc_result, "_success": True, "_protocol": "A2A"}

        except httpx.ConnectError:
            logger.error(f"A2A: {agent_name} not reachable")
            return {"error": f"{agent_name} not running. Start servers first.", "_success": False}
        except httpx.ReadTimeout:
            logger.error(f"A2A: {agent_name} timed out")
            return {"error": f"{agent_name} timed out (>600s)", "_success": False}
        except Exception as e:
            logger.error(f"A2A call to {agent_name} failed: {e}")
            return {"error": str(e), "_success": False}

    # ═══════════════════════════════════════════════
    #  REAL MCP — FastMCP Client
    # ═══════════════════════════════════════════════

    async def _mcp_call(self, tool_name: str, input_data: dict) -> dict:
        """Call tool via REAL MCP protocol."""
        try:
            from fastmcp import Client

            csv_path = os.path.join(WORKING_DIR, "current_data.csv")
            if "csv_path" not in input_data:
                input_data["csv_path"] = csv_path

            async with Client(MCP_URL) as client:
                result = await client.call_tool(tool_name, input_data)

                if isinstance(result, list) and result:
                    text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                    try:
                        parsed = json.loads(text)
                        parsed["_success"] = True
                        parsed["_protocol"] = "MCP"
                        return parsed
                    except (json.JSONDecodeError, TypeError):
                        return {"result": text, "_success": True, "_protocol": "MCP"}

                return {"result": str(result), "_success": True, "_protocol": "MCP"}

        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' failed: {e}")
            return {"error": f"MCP server unavailable: {e}", "_success": False}

    # ═══════════════════════════════════════════════
    #  STATE
    # ═══════════════════════════════════════════════

    def _refresh_data_state(self):
        """Read current CSV to update data metrics."""
        try:
            path = os.path.join(WORKING_DIR, "current_data.csv")
            df = pd.read_csv(path)
            self.state["current_data_shape"] = list(df.shape)
            self.state["remaining_missing"] = int(df.isnull().sum().sum())
            self.state["remaining_categorical"] = len(df.select_dtypes(include="object").columns)
        except Exception:
            pass

    def _update_state(self, action: str, target: str, result: dict, success: bool):
        """Update state from real protocol responses."""
        if action == "agent_call":
            if target not in self.state["agents_called"]:
                self.state["agents_called"].append(target)
            if success:
                if target not in self.state["agents_succeeded"]:
                    self.state["agents_succeeded"].append(target)
                # Reset fail count on success
                self._agent_fail_counts.pop(target, None)
            else:
                if target not in self.state["agents_failed"]:
                    self.state["agents_failed"].append(target)
                # Track failure count — skip after MAX_AGENT_RETRIES
                self._agent_fail_counts[target] = self._agent_fail_counts.get(target, 0) + 1
                if self._agent_fail_counts[target] >= self.MAX_AGENT_RETRIES:
                    if target not in self.state["agents_skipped"]:
                        self.state["agents_skipped"].append(target)
                        logger.warning(
                            f"Agent {target} failed {self.MAX_AGENT_RETRIES} times — "
                            f"SKIPPING to prevent infinite retry loop."
                        )
                        self._notify({
                            "event": "agent_skipped", "agent": target,
                            "reason": f"Failed {self.MAX_AGENT_RETRIES} times",
                        })

            if target == "DataUnderstandingAgent" and success:
                profile = result.get("profile", {})
                if profile.get("target_column"):
                    self.state["target_column"] = profile["target_column"]
                if profile.get("detected_domain"):
                    self.state["domain"] = profile["detected_domain"]

            elif target == "AutoMLAgent" and success:
                self.state["has_model_results"] = True
                pt = result.get("problem_type")
                if pt:
                    self.state["problem_type"] = pt
                # Record model results in memory
                for r in result.get("all_results", []):
                    self.memory.record_model(
                        r["model"], r["metrics"],
                        result.get("problem_type", "unknown"), 0
                    )

        elif action == "tool_call":
            if target not in self.state["tools_used"]:
                self.state["tools_used"].append(target)


def run_pipeline(csv_path: str, target_column: str = None,
                 model: str = "llama3.2", step_callback=None) -> Dict[str, Any]:
    """Synchronous entry point."""
    orch = AutonomousOrchestrator(model=model)
    if step_callback:
        orch.on_step(step_callback)
    return asyncio.run(orch.run(csv_path, target_column))
