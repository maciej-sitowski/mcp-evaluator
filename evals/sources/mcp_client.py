"""Thin helper around the official MCP Python client (streamable HTTP transport).

Every function accepts optional HTTP `headers` (e.g. ``{"Authorization": "Bearer ..."}``), so the same
code works against the local mock server and against a real, authenticated MCP server.
"""

import json
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

from mcp import ClientSession

try:  # mcp 1.x: accepts headers directly
    from mcp.client.streamable_http import streamablehttp_client as _legacy_http_client
except ImportError:
    _legacy_http_client = None
try:  # newer releases: bring your own httpx client
    from mcp.client.streamable_http import streamable_http_client as _http_client
except ImportError:
    _http_client = None

Headers = Optional[Dict[str, str]]


class ToolCallError(RuntimeError):
    """The MCP server answered with `isError=true` (or an unusable payload)."""


def wait_for_server(mcp_url: str, timeout_s: float = 60.0, headers: Headers = None) -> None:
    """Wait until the server answers HTTP at all.

    The mock server has a /health route; a real MCP server may not (404) or may require auth (401/403).
    Any HTTP answer means "up"; only connection errors are retried.
    """
    parsed = urlparse(mcp_url)
    health_url = "%s://%s/health" % (parsed.scheme, parsed.netloc)
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(health_url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=3):
                return
        except urllib.error.HTTPError:
            return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError("MCP server at %s not reachable after %ds: %s" % (health_url, timeout_s, last_error))


@asynccontextmanager
async def _connect(mcp_url: str, headers: Headers):
    if _legacy_http_client is not None:
        async with _legacy_http_client(mcp_url, headers=headers or None) as streams:
            yield streams
    else:
        import httpx

        async with httpx.AsyncClient(headers=headers or None, follow_redirects=True, timeout=30.0) as client:
            async with _http_client(mcp_url, http_client=client) as streams:
                yield streams


@asynccontextmanager
async def open_session(mcp_url: str, headers: Headers = None):
    """Yield an initialized `ClientSession`."""
    async with _connect(mcp_url, headers) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """Call a tool and return (payload, latency_seconds). Raises ToolCallError on tool errors."""
    started = time.perf_counter()
    result = await session.call_tool(name, arguments)
    latency = time.perf_counter() - started

    text = " ".join(getattr(c, "text", "") for c in result.content if getattr(c, "type", "") == "text")
    if result.isError:
        raise ToolCallError(text or "tool returned an error")

    payload = getattr(result, "structuredContent", None)
    if payload is None:
        try:
            payload = json.loads(text)
        except ValueError:
            raise ToolCallError("tool returned non-JSON content: %r" % text[:200])
    if isinstance(payload, dict) and set(payload) == {"result"} and isinstance(payload["result"], dict):
        payload = payload["result"]  # unwrap FastMCP's output wrapper, if present
    return payload, latency
