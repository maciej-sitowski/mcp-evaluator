"""LLM cost of a run: tokens -> dollars, and the sums the run file and the dashboard show.

Every model call reports a `Usage` (see `agents/base.py`). A call is priced like this:

  1. the provider billed it in credits itself (GitHub Copilot: AI credits)  -> credits x AI_CREDIT_USD, "reported"
  2. else the model is in the PRICING table (set in `run_evals.py`)           -> tokens x price per 1M,  "estimated"
  3. else it is not priced: the tokens are still recorded and the cost is None.

Costs are API list prices. What you are actually charged depends on your plan (included allowances, discounts).
"""

from typing import Any, Dict, Iterable, List, Optional

from agents.base import Usage

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_PRICE_KEYS = {  # Usage field -> key in a PRICING entry
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_tokens": "cache_read",
    "cache_write_tokens": "cache_write",
}


def cost_record(
    usage: Usage, provider: str, model: str, pricing: Dict[str, Dict[str, Dict[str, float]]], credit_usd: float
) -> Dict[str, Any]:
    """One priced model role (agent or judge) of one case."""
    record: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "calls": usage.calls,
        **{f: getattr(usage, f) for f in TOKEN_FIELDS},
        "reasoning_tokens": usage.reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "ai_credits": round(usage.ai_credits, 6) if usage.ai_credits is not None else None,
        "cost_usd": None,
        "cost_source": None,
    }
    prices = (pricing.get(provider) or {}).get(model)
    if usage.ai_credits is not None:
        record["cost_usd"] = usage.ai_credits * credit_usd
        record["cost_source"] = "reported"
    elif usage.known and prices:
        record["cost_usd"] = sum(getattr(usage, f) * float(prices.get(k, 0.0)) for f, k in _PRICE_KEYS.items()) / 1e6
        record["cost_source"] = "estimated"
    if record["cost_usd"] is not None:
        record["cost_usd"] = round(record["cost_usd"], 8)
    record["known"] = usage.known
    return record


def combine(records: Iterable[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Sum role records (or case costs' totals). None when there is nothing to sum."""
    records = [r for r in records if r and r.get("known", True)]
    if not records:
        return None
    priced = [r["cost_usd"] for r in records if r.get("cost_usd") is not None]
    total: Dict[str, Any] = {f: sum(r.get(f, 0) for r in records) for f in TOKEN_FIELDS}
    total["reasoning_tokens"] = sum(r.get("reasoning_tokens", 0) for r in records)
    total["total_tokens"] = sum(r.get("total_tokens", 0) for r in records)
    total["calls"] = sum(r.get("calls", 0) for r in records)
    total["cost_usd"] = round(sum(priced), 8) if priced else None
    # False when some call could not be priced (then the cost is a lower bound)
    total["complete"] = all(r.get("cost_usd") is not None and r.get("complete", True) for r in records)
    total["known"] = True
    return total


def case_cost(agent: Dict[str, Any], judge: Dict[str, Any]) -> Dict[str, Any]:
    """The `cost` block of a case: agent, judge and their total."""
    return {"agent": agent, "judge": judge, "total": combine([agent, judge])}


def summarize_cost(cases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Run / product level cost: totals, agent vs judge split, average per case. None when no case has a cost."""
    with_cost = [c["cost"] for c in cases if c.get("cost") and c["cost"].get("total")]
    if not with_cost:
        return None
    total = combine(c["total"] for c in with_cost)
    agent = combine(c["agent"] for c in with_cost)
    judge = combine(c["judge"] for c in with_cost)
    out = {
        "total_usd": total["cost_usd"],
        "agent_usd": agent["cost_usd"] if agent else None,
        "judge_usd": judge["cost_usd"] if judge else None,
        "avg_case_usd": round(total["cost_usd"] / len(with_cost), 8) if total["cost_usd"] is not None else None,
        "complete": total["complete"],
        "cases": len(with_cost),
        "calls": total["calls"],
        **{f: total[f] for f in TOKEN_FIELDS},
        "total_tokens": total["total_tokens"],
        "agent_tokens": agent["total_tokens"] if agent else 0,
        "judge_tokens": judge["total_tokens"] if judge else 0,
    }
    return out


def fmt_usd(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "$%.4f" % value if value < 1 else "$%.2f" % value
