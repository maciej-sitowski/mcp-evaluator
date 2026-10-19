"""Pieces shared by every agent backend (`cursor.py`, `copilot.py`, ...).

A backend only has to know how to talk to its SDK. Everything that does not depend on the SDK lives
here: the support-assistant prompt, the `AgentRun` result record, finding the MCP tool's JSON payload
in whatever wrapper the SDK used, retries with backoff, and parsing the judge's JSON reply.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Placeholders: {product} {tool} {tools} {arguments_hint} {request}. A product can override it in
# product.json ("assistant_prompt") to match how your real assistant is instructed. {tool} is the
# bare name of the tool the direct check calls; {tools} is the phrase listing every tool this run allows.
AGENT_PROMPT = """You are a support assistant for {product}.
Answer the customer's question using ONLY the documentation returned by {tools} of the MCP server.
{arguments_hint}If the documentation does not cover the question, say so plainly instead of guessing.
Keep the answer short.

Customer: {request}"""


def agent_prompt(
    template, product: str, tool: str, default_arguments: Dict[str, Any], request: str, tools: Optional[List[str]] = None
) -> str:
    names = [n for n in (tools or [tool]) if n] or [tool]
    if len(names) == 1:
        phrase = "the `%s` tool" % names[0]
    else:
        phrase = "these tools: %s" % ", ".join("`%s`" % n for n in names)
    hint = ""
    if default_arguments:
        hint = "When you call the tool, always pass these arguments: %s.\n" % json.dumps(default_arguments)
    return (template or AGENT_PROMPT).format(
        product=product, tool=names[0], tools=phrase, arguments_hint=hint, request=request
    )


@dataclass
class Usage:
    """What a model call consumed. Tokens are always filled in when the SDK reports them; `ai_credits` only
    when the provider bills in credits itself (GitHub Copilot). Turning this into dollars is `run_evals.py`'s job."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0  # part of output_tokens, shown for information only
    ai_credits: Optional[float] = None  # as reported by the provider; None = not reported
    calls: int = 0  # model calls (agent turns / judge replies) behind these numbers

    def add(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls
        if other.ai_credits is not None:
            self.ai_credits = (self.ai_credits or 0.0) + other.ai_credits
        return self

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def known(self) -> bool:
        """False when the SDK reported nothing at all (so a cost of 0 would be a lie)."""
        return self.total_tokens > 0 or self.ai_credits is not None


@dataclass
class AgentRun:
    answer: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # {"name", "arguments", "output", ...}
    flow: List[Dict[str, Any]] = field(default_factory=list)  # ordered user / assistant / tool_call steps
    status: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    usage: Usage = field(default_factory=Usage)


def find_payload(value: Any, depth: int = 0) -> Optional[Dict[str, Any]]:
    """Find the tool's JSON payload (a dict with a `results` list) inside whatever wrapper the SDK used:
    plain dicts, MCP content lists, JSON encoded in strings, nested several levels deep."""
    if depth > 8:
        return None
    if isinstance(value, str):
        try:
            return find_payload(json.loads(value), depth + 1)
        except ValueError:
            return None
    if isinstance(value, dict):
        if isinstance(value.get("results"), list):
            return value
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        for item in value:
            found = find_payload(item, depth + 1)
            if found is not None:
                return found
    return None


def jsonable(value: Any) -> Any:
    """Make an SDK value safe to store in the run file."""
    try:
        return json.loads(json.dumps(value, default=lambda o: getattr(o, "__dict__", str(o))))
    except (TypeError, ValueError):
        return str(value)


def finish_flow(out: AgentRun) -> None:
    """Tidy up the conversation once the run is over: trim texts and make sure the final answer is its last step."""
    for step in out.flow:
        if step["type"] == "assistant":
            step["text"] = step["text"].strip()
    last = out.flow[-1]
    if out.answer.strip() and not (last["type"] == "assistant" and last["text"] == out.answer.strip()):
        out.flow.append({"type": "assistant", "text": out.answer.strip()})


def with_retries(
    fn: Callable[[], Tuple[Any, Optional[BaseException]]],
    retryable: Callable[[BaseException], bool],
    attempts: int = 4,
):
    """Call `fn()` -> (value, failure); retry transient failures (e.g. "Server at capacity") with backoff."""
    for attempt in range(1, attempts + 1):
        value, failure = fn()
        if failure is None or not retryable(failure) or attempt == attempts:
            return value, failure
        time.sleep(min(float(getattr(failure, "retry_after", None) or 0) or 3 * 2 ** (attempt - 1), 60))
    return value, failure


def parse_json_object(text: str) -> Dict[str, Any]:
    """First JSON object in a model reply (tolerates ```json fences and chatter around it)."""
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text or "") if ch == "{"):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("judge did not return a JSON object: %r" % (text or "")[:200])
