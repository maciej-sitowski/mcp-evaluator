"""Thin wrapper around the GitHub Copilot SDK (`github-copilot-sdk`), a drop-in alternative to `cursor.py`.

It exposes the same three functions the runner needs:

  * `preflight()` - can the SDK be used? (token set, runtime starts, model available)
  * `run_agent()` - a Copilot agent that answers the customer's question using ONLY the MCP server
    under test. The model, not the golden file, picks the tool and its arguments.
  * `judge()`     - a tool-less Copilot session that scores a rubric prompt and returns JSON.

Authentication is a GitHub token (``COPILOT_GITHUB_TOKEN``, falling back to ``GH_TOKEN`` / ``GITHUB_TOKEN``):
a fine-grained personal access token with the "Copilot Requests" permission, or an OAuth token of a user
with a Copilot licence. The SDK is async; every call here is synchronous (it runs its own event loop) so
it can be used from the runner exactly like the Cursor backend. `preflight` is called
from the runner's loop, where a nested `asyncio.run` is illegal, so that loop runs
on a worker thread.

Each call starts its own short-lived Copilot runtime with an empty, throw-away home and working directory,
so nothing from your own ``~/.copilot`` (MCP servers, instructions, skills) leaks into the evaluation.
"""

import asyncio
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from agents.base import AgentRun, Usage, finish_flow, find_payload, jsonable, parse_json_object, with_retries

DEFAULT_MODEL = "auto"  # let Copilot pick; pin a model (AGENT_MODEL=gpt-5, ...) for reproducible comparisons
TOKEN_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

TURN_TIMEOUT_S = 180.0  # per prompt: covers the model calls plus the MCP tool call
START_TIMEOUT_S = 60.0

# Tools the runtime adds on its own that are not the MCP server under test; never shown as a search call.
_RUNTIME_TOOLS = {
    "report_intent", "ask_user", "task_complete", "exit_plan_mode", "task", "read_agent", "write_agent",
    "list_agents", "send_inbox", "context_board", "skill",
}


def _plain(value: Any) -> Any:
    """SDK event objects -> plain JSON-safe data (they expose `to_dict()`)."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
        except Exception:
            pass
    return jsonable(value)


def _add_usage(usage: Usage, data) -> None:
    """Fold one `assistant.usage` event (one model call) into `usage`. Copilot bills in AI credits:
    `total_nano_aiu` is a billionth of a credit."""
    usage.calls += 1
    usage.input_tokens += int(data.input_tokens or 0)
    usage.output_tokens += int(data.output_tokens or 0)
    usage.cache_read_tokens += int(data.cache_read_tokens or 0)
    usage.cache_write_tokens += int(data.cache_write_tokens or 0)
    usage.reasoning_tokens += int(data.reasoning_tokens or 0)
    billed = getattr(data, "copilot_usage", None)
    if billed is not None and billed.total_nano_aiu is not None:
        usage.ai_credits = (usage.ai_credits or 0.0) + float(billed.total_nano_aiu) / 1e9


def _client(token: str, home: str, cwd: str):
    from copilot import CopilotClient

    return CopilotClient(
        github_token=token,
        working_directory=cwd,
        base_directory=home,  # COPILOT_HOME: no sessions / config from the real home directory
        log_level="error",
    )


def _session_options(model: str, cwd: str) -> Dict[str, Any]:
    from copilot.session import PermissionHandler

    return dict(
        model=model,
        working_directory=cwd,
        on_permission_request=PermissionHandler.approve_all,
        skip_custom_instructions=True,
        enable_config_discovery=False,
        enable_skills=False,
        enable_session_store=False,
        infinite_sessions={"enabled": False},
    )


def _run(main):
    """Run `main()` (an async function) to completion, even if the caller already has an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(main())

    # Nested asyncio.run() raises. Drive this call on a private loop in a worker thread.
    box: Dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = asyncio.run(main())
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=work, name="copilot-loop")
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


# --------------------------------------------------------------------------- preflight
async def _preflight(token: str, models: List[str]) -> Optional[str]:
    with tempfile.TemporaryDirectory(prefix="eval-copilot-home-") as home, tempfile.TemporaryDirectory(
        prefix="eval-copilot-cwd-"
    ) as cwd:
        client = _client(token, home, cwd)
        try:
            await asyncio.wait_for(client.start(), START_TIMEOUT_S)
            auth = await client.get_auth_status()
            if not auth.isAuthenticated:
                return "the GitHub token was not accepted by Copilot (%s)" % (auth.statusMessage or "not authenticated")
            available = {m.id for m in await client.list_models()}
        finally:
            await client.stop()
    unknown = [m for m in models if m != "auto" and available and m not in available]
    if unknown:
        return "model(s) not available to this token: %s (available: %s)" % (
            ", ".join(unknown),
            ", ".join(sorted(available)),
        )
    return None


def preflight(api_key: Optional[str], models: List[str]) -> Optional[str]:
    """Return None when the SDK can be used, otherwise a human-readable reason to skip."""
    if not api_key:
        return "COPILOT_GITHUB_TOKEN (or GH_TOKEN / GITHUB_TOKEN) is not set"
    try:
        import copilot  # noqa: F401
    except ImportError:
        return "github-copilot-sdk is not installed (pip install github-copilot-sdk)"
    try:
        return _run(lambda: _preflight(api_key, models))
    except Exception as exc:  # no runtime, bad token, no network, ...
        return "could not reach Copilot: %s: %s" % (type(exc).__name__, exc)


# --------------------------------------------------------------------------- agent
async def _run_agent_async(
    request: str, prompt: str, mcp_url: str, model: str, token: str, headers: Optional[Dict[str, str]]
) -> "tuple[AgentRun, Optional[BaseException]]":
    """Let a Copilot agent handle `request` with the MCP server as its only tool.

    `flow` is the ordered conversation: user request -> assistant text / tool calls (with the raw MCP
    responses) -> final answer. It is stored in the run file and shown in the dashboard.
    """
    from copilot import ToolSet
    from copilot.session_events import (
        AssistantMessageData,
        AssistantUsageData,
        ToolExecutionCompleteData,
        ToolExecutionStartData,
    )

    out = AgentRun()
    out.flow.append({"type": "user", "text": request})
    calls: Dict[str, Dict[str, Any]] = {}
    started = time.monotonic()

    def on_event(event) -> None:
        data = event.data
        if isinstance(data, ToolExecutionStartData):
            if not data.mcp_server_name and data.tool_name in _RUNTIME_TOOLS:
                return
            step = {
                "type": "tool_call",
                "name": data.mcp_tool_name or data.tool_name,
                "server": data.mcp_server_name,
                "arguments": data.arguments if isinstance(data.arguments, dict) else {},
                "status": "running",
                "output": None,
            }
            calls[data.tool_call_id] = step
            out.flow.append(step)
        elif isinstance(data, ToolExecutionCompleteData):
            step = calls.get(data.tool_call_id)
            if step is None:
                return
            if data.success and data.result is not None:
                raw = _plain(data.result)
                payload = find_payload(raw)
                step["status"] = "completed"
                step["output"] = payload if payload is not None else raw
                step["parsed"] = payload is not None
            else:
                step["status"] = "error"
                step["output"] = {"error": data.error.message if data.error else "tool call failed"}
                step["parsed"] = False
        elif isinstance(data, AssistantMessageData):
            text = (data.content or "").strip()
            if text:
                out.flow.append({"type": "assistant", "text": text})
        elif isinstance(data, AssistantUsageData):
            _add_usage(out.usage, data)

    failure: Optional[BaseException] = None
    with tempfile.TemporaryDirectory(prefix="eval-copilot-home-") as home, tempfile.TemporaryDirectory(
        prefix="eval-agent-"
    ) as cwd:
        client = _client(token, home, cwd)
        try:
            await asyncio.wait_for(client.start(), START_TIMEOUT_S)
            mcp = {"type": "http", "url": mcp_url, "tools": ["*"]}
            if headers:
                mcp["headers"] = dict(headers)
            async with await client.create_session(
                **_session_options(model, cwd),
                mcp_servers={"docs": mcp},
                available_tools=ToolSet().add_mcp("*"),  # no shell / file access, only MCP tools
            ) as session:
                session.on(on_event)
                final = await session.send_and_wait(prompt, timeout=TURN_TIMEOUT_S)
            if final is not None and getattr(final.data, "content", None):
                out.answer = final.data.content
            else:
                texts = [s["text"] for s in out.flow if s["type"] == "assistant"]
                out.answer = texts[-1] if texts else ""
            out.status = "finished"
        except Exception as exc:
            failure = exc
            out.status = "error"
            out.error = "%s: %s" % (type(exc).__name__, exc)
            out.flow.append({"type": "error", "text": out.error})
        finally:
            try:
                await client.stop()
            except Exception:
                pass
    out.duration_ms = int((time.monotonic() - started) * 1000)
    finish_flow(out)
    out.tool_calls = list(calls.values())
    return out, failure


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(k in text for k in ("rate limit", "429", "500", "502", "503", "504", "overloaded", "capacity", "timed out"))


def _once(request: str, **kwargs):
    try:
        return _run(lambda: _run_agent_async(request, **kwargs))
    except Exception as exc:  # e.g. the event loop could not be created
        out = AgentRun(status="error", error="%s: %s" % (type(exc).__name__, exc))
        out.flow = [{"type": "user", "text": request}, {"type": "error", "text": out.error}]
        return out, exc


def run_agent(request: str, *, prompt: str, mcp_url: str, model: str, api_key: str, headers=None) -> AgentRun:
    out, _ = with_retries(
        lambda: _once(request, prompt=prompt, mcp_url=mcp_url, model=model, token=api_key, headers=headers),
        _retryable,
    )
    return out


# --------------------------------------------------------------------------- judge
async def _judge_async(prompt: str, model: str, token: str) -> Tuple[str, Usage]:
    from copilot.session_events import AssistantUsageData

    usage = Usage()

    def on_event(event) -> None:
        if isinstance(event.data, AssistantUsageData):
            _add_usage(usage, event.data)

    with tempfile.TemporaryDirectory(prefix="eval-copilot-home-") as home, tempfile.TemporaryDirectory(
        prefix="eval-judge-"
    ) as cwd:
        client = _client(token, home, cwd)
        try:
            await asyncio.wait_for(client.start(), START_TIMEOUT_S)
            async with await client.create_session(
                **_session_options(model, cwd),
                available_tools=[],  # the judge only reads the rubric and replies with JSON
            ) as session:
                session.on(on_event)
                final = await session.send_and_wait(prompt, timeout=TURN_TIMEOUT_S)
        finally:
            await client.stop()
    return getattr(getattr(final, "data", None), "content", "") or "", usage


def judge(prompt: str, *, model: str, api_key: str) -> Tuple[Dict[str, Any], Usage]:
    """Ask a tool-less Copilot session to grade something. The prompt must ask for a JSON object.
    Returns the verdict and what the call consumed."""

    def once():
        try:
            return _run(lambda: _judge_async(prompt, model, api_key)), None
        except Exception as exc:
            return None, exc

    result, failure = with_retries(once, _retryable)
    if failure is not None:
        raise failure
    text, usage = result
    return parse_json_object(text), usage
