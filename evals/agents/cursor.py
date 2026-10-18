"""Thin wrapper around the Cursor SDK (`cursor-sdk`) used for two things:

  * `run_agent()` - a Cursor agent that answers the customer's question using ONLY the MCP server
    under test. The model, not the golden file, picks the tool and its arguments.
  * `judge()`     - a tool-less Cursor agent that scores a rubric prompt and returns JSON.

Both authenticate with ``CURSOR_API_KEY``. Without a key (or if the key is rejected) `preflight()`
returns a reason and the runner skips the LLM-based metrics.
"""

import tempfile
from typing import Any, Dict, List, Optional, Tuple

from agents.base import (
    AgentRun,
    Usage,
    finish_flow,
    find_payload,
    jsonable,
    parse_json_object,
    with_retries,
)

DEFAULT_MODEL = "composer-2.5"


def _usage(result) -> Usage:
    """RunResult.usage (token counts; Cursor's own dollar figure is not available on every account)."""
    u = getattr(result, "usage", None)
    if u is None:
        return Usage()
    return Usage(
        input_tokens=int(getattr(u, "input_tokens", 0) or 0),
        output_tokens=int(getattr(u, "output_tokens", 0) or 0),
        cache_read_tokens=int(getattr(u, "cache_read_tokens", 0) or 0),
        cache_write_tokens=int(getattr(u, "cache_write_tokens", 0) or 0),
        reasoning_tokens=int(getattr(u, "reasoning_tokens", 0) or 0),
        calls=1,
    )


def preflight(api_key: Optional[str], models: List[str]) -> Optional[str]:
    """Return None when the SDK can be used, otherwise a human-readable reason to skip."""
    if not api_key:
        return "CURSOR_API_KEY is not set"
    try:
        from cursor_sdk import Cursor
    except ImportError:
        return "cursor-sdk is not installed (pip install cursor-sdk)"
    try:
        available = {m.id for m in Cursor.models.list(api_key=api_key)}
    except Exception as exc:  # bad key, no network, ...
        return "could not reach the Cursor API: %s: %s" % (type(exc).__name__, exc)
    unknown = [m for m in models if available and m not in available]
    if unknown:
        return "model(s) not available to this key: %s (available: %s)" % (
            ", ".join(unknown),
            ", ".join(sorted(available)),
        )
    return None


def _unwrap_call(name: str, args: Any) -> Dict[str, Any]:
    """MCP calls are reported by the SDK as one generic tool named `mcp` whose args look like
    {"providerIdentifier": "docs", "toolName": "search_docs", "args": {...}}."""
    args = args if isinstance(args, dict) else {}
    if "toolName" in args:
        inner = args.get("args")
        return {
            "name": str(args["toolName"]),
            "server": args.get("providerIdentifier"),
            "arguments": inner if isinstance(inner, dict) else {},
        }
    return {"name": name, "server": None, "arguments": args}


def _run_agent_once(request: str, *, prompt: str, mcp_url: str, model: str, api_key: str, headers=None) -> "tuple[AgentRun, Optional[BaseException]]":
    """Let a Cursor agent handle `request` with the MCP server as its only tool.

    `flow` is the ordered conversation: user request -> assistant text / tool calls (with the raw MCP
    responses) -> final answer. It is stored in the run file and shown in the dashboard.
    """
    from cursor_sdk import Agent, AgentOptions, HttpMcpServerConfig, LocalAgentOptions

    out = AgentRun()
    out.flow.append({"type": "user", "text": request})
    calls: Dict[str, Dict[str, Any]] = {}
    failure: Optional[BaseException] = None
    try:
        with tempfile.TemporaryDirectory(prefix="eval-agent-") as cwd:
            options = AgentOptions(
                model=model,
                api_key=api_key,
                local=LocalAgentOptions(cwd=cwd),
                mcp_servers={"docs": HttpMcpServerConfig(url=mcp_url, headers=headers or None)},
                tools=["mcp"],  # no shell / file access, only MCP tools
            )
            with Agent.create(options) as agent:
                run = agent.send(prompt)
                for msg in run.messages():
                    kind = getattr(msg, "type", None)
                    if kind == "assistant":
                        # Text arrives as small streamed chunks: glue consecutive ones into one message.
                        text = "".join(
                            getattr(b, "text", "") or "" for b in msg.message.content if getattr(b, "type", "") == "text"
                        )
                        if text:
                            if out.flow[-1]["type"] == "assistant":
                                out.flow[-1]["text"] += text
                            else:
                                out.flow.append({"type": "assistant", "text": text})
                    elif kind == "tool_call":
                        step = calls.get(msg.call_id)
                        if step is None:
                            step = {"type": "tool_call", "name": msg.name, "server": None, "arguments": {},
                                    "status": msg.status, "output": None}
                            calls[msg.call_id] = step
                            out.flow.append(step)
                        if msg.args:
                            step.update(_unwrap_call(msg.name, jsonable(msg.args)))
                        step["status"] = msg.status
                        if msg.status in ("completed", "error") and msg.result is not None:
                            raw = jsonable(msg.result)
                            payload = find_payload(raw)
                            step["output"] = payload if payload is not None else raw
                            step["parsed"] = payload is not None
                result = run.wait()
        out.answer = result.result or ""
        out.usage = _usage(result)
        out.status = str(result.status)
        out.duration_ms = int(result.duration_ms or 0)
        if out.status == "error":
            out.error = "agent run failed"
    except Exception as exc:
        failure = exc
        out.error = "%s: %s" % (type(exc).__name__, exc)
        out.flow.append({"type": "error", "text": out.error})
    finish_flow(out)
    out.tool_calls = list(calls.values())
    return out, failure


def _retryable(exc: BaseException) -> bool:
    return bool(getattr(exc, "is_retryable", False)) or type(exc).__name__ in ("InternalServerError", "RateLimitError", "NetworkError", "APITimeoutError")


def run_agent(request: str, *, prompt: str, mcp_url: str, model: str, api_key: str, headers=None) -> AgentRun:
    out, _ = with_retries(
        lambda: _run_agent_once(request, prompt=prompt, mcp_url=mcp_url, model=model, api_key=api_key, headers=headers),
        _retryable,
    )
    return out


def judge(prompt: str, *, model: str, api_key: str) -> Tuple[Dict[str, Any], Usage]:
    """Ask a tool-less Cursor agent to grade something. The prompt must ask for a JSON object.
    Returns the verdict and what the call consumed."""
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions

    def once():
        try:
            with tempfile.TemporaryDirectory(prefix="eval-judge-") as cwd:
                return Agent.prompt(
                    prompt,
                    AgentOptions(model=model, api_key=api_key, local=LocalAgentOptions(cwd=cwd), tools=[]),
                ), None
        except Exception as exc:
            return None, exc

    result, failure = with_retries(once, _retryable)
    if failure is not None:
        raise failure
    if str(result.status) == "error":
        raise RuntimeError("judge run failed")
    return parse_json_object(result.result), _usage(result)
