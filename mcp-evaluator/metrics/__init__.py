"""Metrics: `search` (the MCP tool called directly) and `agent` (the agent's search + answer, incl. LLM judge)."""

from metrics.agent import build_agent_metrics
from metrics.search import build_search_metrics

__all__ = ["build_search_metrics", "build_agent_metrics"]
