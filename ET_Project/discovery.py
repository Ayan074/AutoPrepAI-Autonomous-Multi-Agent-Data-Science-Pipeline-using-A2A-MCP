"""
Dynamic Agent & Tool Discovery — No hardcoded ports.

AgentRegistry:
  - Scans port range, fetches /.well-known/agent.json from each
  - Builds runtime registry of active agents + their skills
  - Replaces all hardcoded A2A_PORTS dicts

ToolRegistry:
  - Connects to MCP server, calls list_tools()
  - Builds runtime registry of available tools + schemas
  - Replaces hardcoded tool lists in orchestrator prompts
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default scan range for A2A agents
DEFAULT_AGENT_PORTS = range(8201, 8220)
DEFAULT_MCP_URL = "http://localhost:8100/mcp"


class AgentRegistry:
    """Dynamically discovers A2A agents by scanning ports for agent cards."""

    def __init__(self, port_range: range = None):
        self.port_range = port_range or DEFAULT_AGENT_PORTS
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._scan_timeout = 2.0  # seconds per port

    async def discover_agents(self) -> Dict[str, Dict[str, Any]]:
        """Scan all ports and build the agent registry.

        Returns:
            Dict of {agent_name: {url, port, skills, description, capabilities}}
        """
        self._agents = {}
        tasks = [
            self._probe_port(port) for port in self.port_range
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict) and result.get("name"):
                self._agents[result["name"]] = result

        logger.info(
            f"Discovery complete: {len(self._agents)} agents found — "
            f"{list(self._agents.keys())}"
        )
        return self._agents

    async def _probe_port(self, port: int) -> Optional[Dict[str, Any]]:
        """Try to fetch an agent card from a single port."""
        url = f"http://localhost:{port}/.well-known/agent.json"
        try:
            async with httpx.AsyncClient(timeout=self._scan_timeout) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    card = resp.json()
                    return {
                        "name": card.get("name", f"agent_{port}"),
                        "url": f"http://localhost:{port}/",
                        "port": port,
                        "description": card.get("description", ""),
                        "skills": card.get("skills", []),
                        "capabilities": card.get("capabilities", {}),
                        "version": card.get("version", "unknown"),
                    }
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass  # Port not active — expected
        except Exception as e:
            logger.debug(f"Probe port {port} failed: {e}")
        return None

    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a discovered agent by name."""
        return self._agents.get(name)

    def get_agent_url(self, name: str) -> Optional[str]:
        """Get the URL of a discovered agent."""
        agent = self._agents.get(name)
        return agent["url"] if agent else None

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Return all discovered agents."""
        return self._agents

    def get_agent_names(self) -> List[str]:
        """Return names of all discovered agents."""
        return list(self._agents.keys())

    def get_agent_summary_for_llm(self) -> str:
        """Build a text summary of available agents for LLM prompts."""
        if not self._agents:
            return "No agents discovered."

        lines = []
        for name, info in self._agents.items():
            skills = [s.get("name", s.get("id", "?")) for s in info.get("skills", [])]
            lines.append(
                f"- {name} (port {info['port']}): {info['description']}. "
                f"Skills: {', '.join(skills)}"
            )
        return "\n".join(lines)

    async def health_check(self, agent_name: str) -> bool:
        """Check if a specific agent is still responding."""
        agent = self._agents.get(agent_name)
        if not agent:
            return False
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(
                    f"{agent['url']}.well-known/agent.json"
                )
                return resp.status_code == 200
        except Exception:
            return False


class ToolRegistry:
    """Dynamically discovers MCP tools from the MCP server."""

    def __init__(self, mcp_url: str = None):
        self.mcp_url = mcp_url or DEFAULT_MCP_URL
        self._tools: Dict[str, Dict[str, Any]] = {}

    async def discover_tools(self) -> Dict[str, Dict[str, Any]]:
        """Connect to MCP server and list all available tools.

        Returns:
            Dict of {tool_name: {description, input_schema}}
        """
        self._tools = {}
        try:
            from fastmcp import Client

            async with Client(self.mcp_url) as client:
                tools = await client.list_tools()
                for tool in tools:
                    name = tool.name if hasattr(tool, 'name') else str(tool)
                    self._tools[name] = {
                        "name": name,
                        "description": (
                            tool.description
                            if hasattr(tool, 'description')
                            else ""
                        ),
                        "input_schema": (
                            tool.inputSchema
                            if hasattr(tool, 'inputSchema')
                            else {}
                        ),
                    }

            logger.info(
                f"MCP discovery: {len(self._tools)} tools found — "
                f"{list(self._tools.keys())}"
            )
        except Exception as e:
            logger.error(f"MCP tool discovery failed: {e}")

        return self._tools

    def get_tool_names(self) -> List[str]:
        """Return names of all discovered tools."""
        return list(self._tools.keys())

    def get_tool_summary_for_llm(self) -> str:
        """Build a text summary of available tools for LLM prompts."""
        if not self._tools:
            return "No MCP tools discovered."

        lines = []
        for name, info in self._tools.items():
            desc = info.get("description", "")[:80]
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Return all discovered tools."""
        return self._tools
