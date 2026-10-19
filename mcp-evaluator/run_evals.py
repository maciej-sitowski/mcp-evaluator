"""Run the product goldens against the MCP server and store the results as JSON.

Products live in ``products/<id>/`` (docs/, product.json). Golden sets live in ``goldens/<id>/`` next to
this file. For every golden:
  1. call the real MCP tool over HTTP with fixed arguments,
  2. wrap the response in a DeepEval `LLMTestCase`,
  3. score it with the metrics in `metrics/search.py`,
  4. (optional, needs an agent provider's key: CURSOR_API_KEY or a GitHub token for Copilot) let an agent
     answer the question through the MCP server and grade it with `metrics/agent.py`.
     Pick the provider with AGENT_PROVIDER=cursor|copilot (see `agents/__init__.py`).

Each run is saved to ``<results-dir>/runs/<run_id>.json``; the dashboard reads those files.
No DeepEval / Confident AI account is involved.
"""

# =========================================================================== CONFIG
# The one place to choose the agent and the products. Precedence for each setting:
#   command-line flag  >  environment variable (handy for a one-off run)  >  the value below.

# Who plays the support assistant: "cursor" or "copilot" (providers live in agents/__init__.py).
AGENT_PROVIDER = "cursor"
# Who grades it. None = the same provider as the agent; set e.g. "cursor" to keep one fixed judge.
JUDGE_PROVIDER = None
# Models. None = the provider's default (cursor: composer-2.5, copilot: auto). Model ids differ per
# provider, so check them when you switch AGENT_PROVIDER.
AGENT_MODEL = None
JUDGE_MODEL = None

# Folder with one sub-folder per product (product.json, docs/). A relative path is resolved from the
# repo root. Inside Docker the repo's `products/` folder is mounted at /products, which is where
# "products" lands; to use a folder elsewhere, change the `./products:/products` mounts in
# docker-compose.yml as well. Golden sets are not in that folder: they live in goldens/<id>/ here.
PRODUCTS_DIR = "products"

# What you changed for this run. Stored on the run file and shown in the history table, so a score
# change can be matched to the change that produced it. Keys and values are yours; leave the dict
# empty to store nothing. Override for one run with --param key=value (repeatable), or replace the
# whole dict with RUN_PARAMS='{"search": "v2", "change": "shorter prompt"}'.
RUN_PARAMS = {
    "search": "v1",
    # "change": "baseline",
}

# LLM cost (shown per run and per case in the dashboard). A call is priced from the credits the provider
# reports (GitHub Copilot) or, failing that, from its tokens times PRICING (Cursor reports tokens only).
# Both are API list prices: what you are really charged depends on your plan's allowance and discounts.
AI_CREDIT_USD = 0.01  # 1 GitHub AI credit = $0.01 (https://docs.github.com/copilot/reference/copilot-billing/models-and-pricing)
# USD per 1M tokens, by provider and model id. A model that is not listed still shows its tokens, its cost is "n/a".
# Cursor prices: https://cursor.com/docs/models-and-pricing
PRICING = {
    "cursor": {
        "composer-2.5": {"input": 0.50, "output": 2.50, "cache_read": 0.20, "cache_write": 0.50},
    },
}
# =========================================================================== /CONFIG

# Must be set before deepeval is imported: no telemetry, no update prompts.
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "NO")

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import urllib.request

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase, ToolCall

from agents import PROVIDERS, get_provider
from agents.base import Usage, agent_prompt
from cost import case_cost, cost_record, fmt_usd, summarize_cost
from metrics import build_agent_metrics, build_search_metrics
from sources.mcp_client import call_tool, open_session, wait_for_server
from sources.products import DEFAULT_QUERY_ARGUMENT, Product, load_products, normalize_patterns

HERE = Path(__file__).parent


def default_products_dir() -> str:
    """$PRODUCTS_DIR, else PRODUCTS_DIR from the CONFIG section; relative paths start at the repo root."""
    path = Path(os.environ.get("PRODUCTS_DIR") or PRODUCTS_DIR).expanduser()
    return str(path if path.is_absolute() else HERE.parent / path)


# --------------------------------------------------------------------------- goldens
def build_goldens(product: Product, all_ids: List[str]) -> List[Golden]:
    goldens = []
    for g in product.goldens:
        # The golden's own arguments win over the product's defaults; with none, the question is the query.
        arguments = {**product.default_arguments, **g.get("arguments", {DEFAULT_QUERY_ARGUMENT: g["input"]})}
        expected = normalize_patterns(g.get("expected_docs", []), product.id, all_ids)
        forbidden = normalize_patterns(g.get("forbidden_docs", []), product.id, all_ids)
        reference = g.get("reference_answer")
        goldens.append(
            Golden(
                input=g["input"],
                expected_output=reference or (("Includes: " + ", ".join(expected)) if expected else "No results"),
                additional_metadata={
                    "id": g["id"],
                    "product": product.id,
                    "product_name": product.name,
                    "tool_name": product.tool,
                    "arguments": arguments,
                    "expected_ids": expected,
                    "forbidden_ids": forbidden,
                    "reference_answer": reference,
                    "must_include": g.get("must_include", []),
                },
            )
        )
    return goldens


# --------------------------------------------------------------------------- one case
def metric_record(metric) -> Dict[str, Any]:
    return {
        "score": metric.score,
        "threshold": metric.threshold,
        "success": bool(metric.success),
        "reason": metric.reason,
    }


def _context(results: List[Dict[str, Any]]) -> List[str]:
    return ["%s: %s - %s" % (r.get("id"), r.get("section", ""), r.get("text", "")) for r in results if isinstance(r, dict)]


async def run_golden(session, product: Product, golden: Golden, llm: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = golden.additional_metadata
    arguments = meta["arguments"]
    tool = meta["tool_name"]

    payload: Dict[str, Any] = {}
    latency_s = 0.0
    error = None
    try:
        payload, latency_s = await call_tool(session, tool, arguments)
        actual_output = json.dumps(payload)
    except Exception as exc:  # tool/transport error -> the case fails, the run continues
        error = "%s: %s" % (type(exc).__name__, exc)
        actual_output = "ERROR " + error

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
        retrieval_context=_context(payload.get("results", [])),
        tools_called=[ToolCall(name=tool, input_parameters=arguments, output=payload or error)],
        completion_time=latency_s,
        additional_metadata=meta,
    )

    metric_results = {}
    for metric in build_search_metrics():
        metric.measure(test_case)
        metric_results[metric.__name__] = metric_record(metric)

    agent_record = None
    cost = None
    if llm:
        agent_record, cost = await run_agent_metrics(product, golden, llm, metric_results)

    return {
        "product": meta["product"],
        "product_name": meta["product_name"],
        "id": meta["id"],
        "input": golden.input,
        "arguments": arguments,
        "expected_ids": meta["expected_ids"],
        "forbidden_ids": meta["forbidden_ids"],
        "reference_answer": meta["reference_answer"],
        "must_include": meta["must_include"],
        "returned_ids": [r["id"] for r in payload.get("results", [])],
        "latency_ms": round(latency_s * 1000, 2),
        "error": error,
        "passed": all(m["success"] for m in metric_results.values()),
        "metrics": metric_results,
        "agent": agent_record,
        "cost": cost,
    }


async def run_agent_metrics(
    product: Product, golden: Golden, llm: Dict[str, Any], metric_results: Dict[str, Any]
) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """Let the agent (Cursor or Copilot) answer the golden's `input` itself, then grade it (incl. LLM-as-a-judge)."""
    tool = golden.additional_metadata["tool_name"]
    prompt = agent_prompt(product.assistant_prompt, product.name, tool, product.default_arguments, golden.input)
    # The SDK is synchronous; run it in a worker thread so the MCP session stays responsive.
    agent = await asyncio.to_thread(
        llm["agent"]["backend"].run_agent,
        golden.input,
        prompt=prompt,
        mcp_url=llm["mcp_url"],
        model=llm["agent"]["model"],
        api_key=llm["agent"]["api_key"],
        headers=llm["headers"],
    )
    meta = dict(golden.additional_metadata)
    meta["agent_tool_calls"] = agent.tool_calls
    meta["agent_error"] = agent.error
    first = next((c for c in agent.tool_calls if str(c["name"]).endswith(tool)), None)
    results = (first or {}).get("output")
    results = results.get("results", []) if isinstance(results, dict) else []
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=agent.answer,
        retrieval_context=_context(results),
        tools_called=[
            ToolCall(name=c["name"], input_parameters=c["arguments"], output=c["output"]) for c in agent.tool_calls
        ],
        additional_metadata=meta,
    )

    judge_usage = Usage()  # all judge replies of this case; the metrics run one after the other

    def judge_fn(judge_prompt: str) -> Dict[str, Any]:
        verdict, used = llm["judge"]["backend"].judge(
            judge_prompt, model=llm["judge"]["model"], api_key=llm["judge"]["api_key"]
        )
        judge_usage.add(used)
        return verdict

    for metric in build_agent_metrics(judge_fn):
        await asyncio.to_thread(metric.measure, test_case)
        metric_results[metric.__name__] = metric_record(metric)

    def priced(role: str, usage: Usage) -> Dict[str, Any]:
        cfg = llm[role]
        return cost_record(usage, cfg["provider"], cfg["model"], llm["pricing"], llm["credit_usd"])

    cost = case_cost(priced("agent", agent.usage), priced("judge", judge_usage))
    record = {
        "answer": agent.answer,
        "tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in agent.tool_calls],
        "flow": agent.flow,
        "returned_ids": [r.get("id") for r in results],
        "duration_ms": agent.duration_ms,
        "error": agent.error,
    }
    return record, cost


# --------------------------------------------------------------------------- aggregation
def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def _rate(passed: int, total: int) -> float:
    return round(passed / total, 4) if total else 0.0


def _metric_stats(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for name in cases[0]["metrics"]:
        results = [c["metrics"][name] for c in cases]
        stats[name] = {
            "avg_score": round(sum(r["score"] for r in results) / len(results), 4),
            "pass_rate": _rate(sum(r["success"] for r in results), len(results)),
        }
    return stats


def summarize(cases: List[Dict[str, Any]], products: List[Product]) -> Dict[str, Any]:
    passed = sum(c["passed"] for c in cases)

    by_product: Dict[str, Dict[str, Any]] = {}
    for p in products:
        mine = [c for c in cases if c["product"] == p.id]
        if not mine:
            continue
        ok = sum(c["passed"] for c in mine)
        by_product[p.id] = {
            "name": p.name,
            "color": p.color,
            "cases_total": len(mine),
            "cases_passed": ok,
            "pass_rate": _rate(ok, len(mine)),
            "metrics": _metric_stats(mine),
            "cost": summarize_cost(mine),
        }

    latencies = [c["latency_ms"] for c in cases if not c["error"]]
    return {
        "cases_total": len(cases),
        "cases_passed": passed,
        "pass_rate": _rate(passed, len(cases)),
        "metrics": _metric_stats(cases),
        "products": by_product,
        "cost": summarize_cost(cases),  # None without the LLM judge
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50": round(_percentile(latencies, 50), 2),
            "p95": round(_percentile(latencies, 95), 2),
        },
    }


def print_report(run: Dict[str, Any]) -> None:
    print()
    print("%-14s %-28s %-6s %s" % ("PRODUCT", "CASE", "STATUS", "FAILED METRICS"))
    print("-" * 90)
    for c in run["cases"]:
        failed = [n for n, m in c["metrics"].items() if not m["success"]]
        print("%-14s %-28s %-6s %s" % (c["product"], c["id"], "PASS" if c["passed"] else "FAIL", ", ".join(failed)))
    s = run["summary"]
    print("-" * 90)
    params = " | " + " ".join("%s=%s" % kv for kv in run["params"].items()) if run.get("params") else ""
    print("Run %s | engine=%s | label=%r%s" % (run["id"], run["engine"], run["label"], params))
    print("Cases passed: %d/%d (%.0f%%)" % (s["cases_passed"], s["cases_total"], s["pass_rate"] * 100))
    for pid, p in s["products"].items():
        print("  %-14s %d/%d passed%s" % (
            pid, p["cases_passed"], p["cases_total"],
            " | cost %s" % fmt_usd(p["cost"]["total_usd"]) if p.get("cost") else "",
        ))
    for name, m in s["metrics"].items():
        print("  %-22s avg score %.3f | pass rate %3.0f%%" % (name, m["avg_score"], m["pass_rate"] * 100))
    print("  latency ms: p50=%.1f p95=%.1f" % (s["latency_ms"]["p50"], s["latency_ms"]["p95"]))
    if s.get("cost"):
        c = s["cost"]
        print("  LLM cost: %s%s (agent %s, judge %s) | %s per case | %s tokens" % (
            fmt_usd(c["total_usd"]), "" if c["complete"] else "+", fmt_usd(c["agent_usd"]), fmt_usd(c["judge_usd"]),
            fmt_usd(c["avg_case_usd"]), format(c["total_tokens"], ",")))
        if not c["complete"]:
            print("    (+ some calls could not be priced: add the model to PRICING in run_evals.py)")


# --------------------------------------------------------------------------- main
def build_headers(args: argparse.Namespace) -> Dict[str, str]:
    """HTTP headers for the MCP server (auth). Values are never printed or stored in the run file."""
    headers: Dict[str, str] = {}
    token = (os.environ.get("MCP_AUTH_HEADER") or "").strip()
    if token:
        headers["Authorization"] = token
    for item in args.mcp_header:
        name, sep, value = item.partition(":")
        if not sep or not name.strip():
            raise SystemExit("--mcp-header must look like 'Name: value', got %r" % item)
        headers[name.strip()] = value.strip()
    return headers


async def run(args: argparse.Namespace) -> int:
    headers = build_headers(args)
    products = load_products(Path(args.products_dir), HERE / "goldens", args.products, args.cases)
    all_ids = [p.id for p in products]
    wait_for_server(args.mcp_url, headers=headers)
    if headers:
        print("MCP request headers: %s" % ", ".join(sorted(headers)))

    llm, llm_status = setup_llm(args, headers)
    print(llm_status)

    cases: List[Dict[str, Any]] = []
    async with open_session(args.mcp_url, headers=headers) as session:
        tools = [t.name for t in (await session.list_tools()).tools]
        for product in products:
            if product.tool not in tools:
                raise SystemExit(
                    "Tool %r (product %s) not exposed by the server (found: %s)" % (product.tool, product.id, tools)
                )
            for golden in build_goldens(product, all_ids):
                cases.append(await run_golden(session, product, golden, llm))
                print("  done %s/%s" % (product.id, golden.additional_metadata["id"]), flush=True)

    now = datetime.now(timezone.utc)
    run_record = {
        "id": now.strftime("%Y%m%dT%H%M%SZ"),
        "timestamp": now.isoformat(timespec="seconds"),
        "label": args.label,
        "params": resolve_run_params(args),
        "engine": detect_engine(args.mcp_url),
        "mcp_url": args.mcp_url,
        "tools": tools,
        "llm_judge": {
            "enabled": bool(llm),
            "status": llm_status,
            "agent_provider": llm["agent"]["provider"] if llm else None,
            "agent_model": llm["agent"]["model"] if llm else None,
            "judge_provider": llm["judge"]["provider"] if llm else None,
            "judge_model": llm["judge"]["model"] if llm else None,
        },
        "summary": summarize(cases, products),
        "cases": cases,
    }

    runs_dir = Path(args.results_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    target = runs_dir / ("%s.json" % run_record["id"])
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    tmp.replace(target)  # atomic: the dashboard never sees half-written files

    print_report(run_record)
    print("\nSaved %s" % target)

    if args.strict and run_record["summary"]["cases_passed"] < run_record["summary"]["cases_total"]:
        return 1
    return 0


def setup_llm(args: argparse.Namespace, headers: Dict[str, str]):
    """(config, message). Config is None when the LLM-based metrics are skipped."""
    if args.no_llm:
        return None, "LLM judge: skipped (--no-llm)"
    roles = {}
    for role, provider_name, model in (
        ("agent", args.agent_provider, args.agent_model),
        ("judge", args.judge_provider or args.agent_provider, args.judge_model),
    ):
        provider = get_provider(provider_name)
        roles[role] = {
            "provider": provider.name,
            "label": provider.label,
            "model": model or provider.default_model,
            "api_key": provider.api_key(),
            "backend": provider.load(),
        }
    # Check each provider once, with every model it is asked to serve.
    for name in sorted({r["provider"] for r in roles.values()}):
        mine = [r for r in roles.values() if r["provider"] == name]
        reason = mine[0]["backend"].preflight(mine[0]["api_key"], sorted({r["model"] for r in mine}))
        if reason:
            return None, "LLM judge: skipped, %s: %s. Deterministic metrics only." % (mine[0]["label"], reason)
    config = {
        "agent": roles["agent"],
        "judge": roles["judge"],
        "mcp_url": args.mcp_url,
        "headers": headers,
        "pricing": PRICING,
        "credit_usd": AI_CREDIT_USD,
    }
    return config, "LLM judge: enabled (agent=%s/%s, judge=%s/%s)" % (
        roles["agent"]["provider"], roles["agent"]["model"], roles["judge"]["provider"], roles["judge"]["model"],
    )


def resolve_run_params(args: argparse.Namespace) -> Dict[str, str]:
    """RUN_PARAMS from CONFIG, replaced by $RUN_PARAMS when set, then each --param key=value on top."""
    raw = (os.environ.get("RUN_PARAMS") or "").strip()
    if raw:
        try:
            base = json.loads(raw)
        except ValueError as exc:
            raise SystemExit("RUN_PARAMS must be a JSON object, e.g. {\"search\": \"v2\"} (%s)" % exc)
        if not isinstance(base, dict):
            raise SystemExit("RUN_PARAMS must be a JSON object, got %s" % type(base).__name__)
    else:
        base = dict(RUN_PARAMS)
    for item in args.param:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise SystemExit("--param must look like 'key=value', got %r" % item)
        base[key] = value.strip()
    out: Dict[str, str] = {}
    for key, value in base.items():
        name = str(key).strip()
        if not name or value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
        elif isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, (int, float)):
            text = str(value)
        else:
            text = json.dumps(value)
        if text:
            out[name] = text
    return out


def detect_engine(mcp_url: str) -> str:
    """Which search engine the mock server is running (reported by its /health route)."""
    parsed = urlparse(mcp_url)
    try:
        with urllib.request.urlopen("%s://%s/health" % (parsed.scheme, parsed.netloc), timeout=3) as resp:
            return json.loads(resp.read().decode()).get("engine", "unknown")
    except Exception:
        return "unknown"


def _env_list(name: str) -> List[str]:
    """Comma-separated environment variable -> list ("" / unset -> [])."""
    return [v.strip() for v in (os.environ.get(name) or "").split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcp-url", default=os.environ.get("MCP_URL", "http://localhost:8000/mcp"))
    p.add_argument("--mcp-header", action="append", default=[], metavar="'Name: value'",
                   help="extra HTTP header for the MCP server (repeatable). MCP_AUTH_HEADER sets Authorization.")
    p.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "/results"))
    p.add_argument("--products-dir", default=default_products_dir())
    p.add_argument("--label", default=os.environ.get("RUN_LABEL", ""), help="free-text note stored in the run file")
    p.add_argument("--param", action="append", default=[], metavar="key=value",
                   help="parameter stored with this run and shown in the history table (repeatable). "
                        "RUN_PARAMS='{\"k\": \"v\"}' replaces the RUN_PARAMS dict in the CONFIG section.")
    p.add_argument("--product", dest="products", action="append", default=_env_list("PRODUCT"), metavar="ID",
                   help="only run this product id (repeatable; or PRODUCT=id1,id2)")
    p.add_argument("--case", dest="cases", action="append", default=_env_list("CASE"), metavar="ID",
                   help="only run this golden id (repeatable; or CASE=id1,id2)")
    p.add_argument("--agent-provider", choices=sorted(PROVIDERS),
                   default=os.environ.get("AGENT_PROVIDER") or AGENT_PROVIDER,
                   help="who plays the support assistant: cursor (CURSOR_API_KEY) or copilot (COPILOT_GITHUB_TOKEN)")
    p.add_argument("--judge-provider", choices=sorted(PROVIDERS), default=os.environ.get("JUDGE_PROVIDER") or JUDGE_PROVIDER,
                   help="who grades the assistant (default: same as --agent-provider)")
    p.add_argument("--agent-model", default=os.environ.get("AGENT_MODEL") or AGENT_MODEL,
                   help="model for the assistant (default: the provider's default)")
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL") or JUDGE_MODEL,
                   help="model for the judge (default: the judge provider's default)")
    p.add_argument("--no-llm", action="store_true", help="skip the agent + LLM-judge metrics even if a key is set")
    p.add_argument("--strict", action="store_true", help="exit with code 1 if any case fails (for CI)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
