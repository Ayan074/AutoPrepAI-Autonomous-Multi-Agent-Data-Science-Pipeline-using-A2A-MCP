"""
Base utilities for A2A Agent Servers.

Provides:
 - OllamaEngine integration helper
 - Shared A2A server wiring (AgentCard, DefaultRequestHandler, etc.)
 - Common data I/O (read/write CSV from shared working directory)
 - A2A client helper to call peer agents
 - MCP tool client helper to call MCP tools from agents
 - Dynamic discovery support
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

# Add parent dir to path so we can import llm.ollama_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.ollama_engine import OllamaEngine

logger = logging.getLogger(__name__)


# ── Shared Working Directory ──
# All agents read/write data CSVs from this directory.
WORKING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workdir"
)
os.makedirs(WORKING_DIR, exist_ok=True)


def get_data_path(filename: str = "current_data.csv") -> str:
    """Return the path to a data file in the shared working directory."""
    return os.path.join(WORKING_DIR, filename)


def read_current_data(filename: str = "current_data.csv") -> pd.DataFrame:
    """Read the current pipeline dataframe from disk."""
    path = get_data_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def save_current_data(df: pd.DataFrame, filename: str = "current_data.csv") -> str:
    """Write the current pipeline dataframe to disk. Returns the path."""
    path = get_data_path(filename)
    df.to_csv(path, index=False)
    return path


# ── Shared LLM instance ──
# All agents share one OllamaEngine instance (one connection to Ollama).
_llm_instance: Optional[OllamaEngine] = None


def get_llm(model: str = "llama3.2") -> OllamaEngine:
    """Get or create the shared Ollama engine."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaEngine(model=model)
    return _llm_instance


# ── Dynamic Agent Discovery ──
# Agents discover each other at runtime via agent cards.
# This replaces the old hardcoded A2A_AGENT_PORTS dict.

async def discover_agent_url(agent_name: str, port_range: range = None) -> Optional[str]:
    """Discover a peer agent's URL by scanning ports for its agent card.

    Falls back to well-known ports if discovery fails.
    """
    _range = port_range or range(8201, 8220)

    # Try discovery first
    async with httpx.AsyncClient(timeout=2.0) as client:
        for port in _range:
            try:
                resp = await client.get(
                    f"http://localhost:{port}/.well-known/agent.json"
                )
                if resp.status_code == 200:
                    card = resp.json()
                    if card.get("name") == agent_name:
                        return f"http://localhost:{port}/"
            except (httpx.ConnectError, httpx.ReadTimeout):
                continue
            except Exception:
                continue

    logger.warning(f"Discovery failed for {agent_name}, using fallback ports")

    # Fallback to well-known ports (for reliability)
    _FALLBACK_PORTS = {
        "DataUnderstandingAgent": 8201,
        "DataQualityAgent": 8202,
        "MissingValueAgent": 8203,
        "EncodingAgent": 8204,
        "FeatureEngineeringAgent": 8205,
        "AutoMLAgent": 8206,
        "EDAAgent": 8207,
        "ReflectionAgent": 8208,
        "ReportAgent": 8209,
    }
    port = _FALLBACK_PORTS.get(agent_name)
    return f"http://localhost:{port}/" if port else None


# ── A2A Client Helper ──
# Agents can call each other via HTTP using the A2A protocol.

async def call_a2a_agent(
    target_agent: str,
    message: str,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Call a peer agent via A2A protocol (JSON-RPC over HTTP).

    This is the real A2A inter-agent communication path.
    Agents call each other directly, not via the orchestrator.

    Uses dynamic discovery to find the agent URL.
    """
    url = await discover_agent_url(target_agent)
    if url is None:
        return {"error": f"Agent {target_agent} not discoverable"}

    # A2A JSON-RPC 2.0 message
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "id": 1,
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message}],
            }
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Extract result from JSON-RPC response
            result = data.get("result", {})
            if isinstance(result, dict):
                artifacts = result.get("artifacts", [])
                if artifacts:
                    parts = artifacts[0].get("parts", [])
                    for part in parts:
                        if part.get("type") == "text":
                            text = part["text"]
                            try:
                                return json.loads(text)
                            except (json.JSONDecodeError, TypeError):
                                return {"response": text}
            return result

    except httpx.ConnectError:
        logger.warning(f"A2A: {target_agent} not reachable")
        return {"error": f"Agent {target_agent} not running"}
    except Exception as e:
        logger.error(f"A2A call to {target_agent} failed: {e}")
        return {"error": str(e)}


# ── MCP Tool Client Helper ──
# Agents can call MCP tools directly via the MCP protocol.

MCP_URL = "http://localhost:8100/mcp"


async def call_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Call an MCP tool via the real MCP protocol.

    Agents use this to invoke MCP tools directly (not via orchestrator).
    This is real MCP client communication.
    """
    if arguments is None:
        arguments = {}

    # Default csv_path if not provided
    if "csv_path" not in arguments:
        arguments["csv_path"] = get_data_path("current_data.csv")

    try:
        from fastmcp import Client

        async with Client(MCP_URL) as client:
            result = await client.call_tool(tool_name, arguments)

            if isinstance(result, list) and result:
                text = result[0].text if hasattr(result[0], 'text') else str(result[0])
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return {"result": text}

            return {"result": str(result)}

    except Exception as e:
        logger.error(f"MCP tool '{tool_name}' failed: {e}")
        return {"error": f"MCP call failed: {e}"}


# ── Execution Log ──
# Every agent logs its decisions for traceability.

@dataclass
class ExecutionLog:
    """Log of all decisions and actions taken by an agent."""
    agent_name: str
    entries: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0  # Track real LLM calls

    def log(self, action: str, details: str, decided_by: str = "LLM",
            columns: Optional[List[str]] = None):
        """Add a log entry."""
        if decided_by == "LLM":
            self.llm_calls += 1
        self.entries.append({
            "agent": self.agent_name,
            "action": action,
            "details": details,
            "decided_by": decided_by,
            "columns": columns or [],
        })
        logger.info(f"[{self.agent_name}] {action}: {details}")

    def to_dict(self) -> dict:
        return {
            "entries": self.entries,
            "llm_calls": self.llm_calls,
        }

