"""Deterministic metrics for the documentation-search MCP tool (`search_docs`), called with fixed arguments.

No LLM judge and no API key needed. For the metrics that score the agent's own search and answer see `agent.py`.
"""

import os
from typing import List

from deepeval.metrics import BaseMetric

from metrics.base import DeterministicMetric, case_meta, found_refs, matches, result_ids, tool_arguments, tool_payload

REQUIRED_RESULT_FIELDS = ("id", "product", "doc", "section", "text", "score")
DEFAULT_LIMIT = 5


class RecallAtK(DeterministicMetric):
    """Fraction of the expected doc references present in the returned results.

    If no docs are expected, the score is 1.0 only when nothing is returned.
    """

    metric_name = "Recall@K"
    default_threshold = 1.0

    def compute(self, test_case):
        expected = case_meta(test_case).get("expected_ids", [])
        returned = result_ids(test_case)
        if not expected:
            ok = not returned
            return (1.0 if ok else 0.0), (
                "No results, as expected" if ok else "Expected no results but got %s" % returned
            )
        found = found_refs(returned, expected)
        missing = [i for i in expected if i not in found]
        reason = "Found %d/%d expected docs" % (len(found), len(expected))
        if missing:
            reason += "; missing %s" % missing
        return len(found) / len(expected), reason


class FirstHitRank(DeterministicMetric):
    """Reciprocal rank (MRR) of the first expected doc in the results."""

    metric_name = "MRR"
    default_threshold = 0.5

    def compute(self, test_case):
        expected = case_meta(test_case).get("expected_ids", [])
        if not expected:
            return 1.0, "Nothing expected; ranking not applicable"
        for rank, rid in enumerate(result_ids(test_case), start=1):
            if any(matches(rid, p) for p in expected):
                return 1.0 / rank, "First expected doc (%s) found at rank %d" % (rid, rank)
        return 0.0, "No expected doc in the results"


class ForbiddenAbsent(DeterministicMetric):
    """1.0 when none of the forbidden doc references are returned."""

    metric_name = "ForbiddenAbsent"
    default_threshold = 1.0

    def compute(self, test_case):
        forbidden = case_meta(test_case).get("forbidden_ids", [])
        if not forbidden:
            return 1.0, "No forbidden docs defined"
        returned = result_ids(test_case)
        leaked = [r for r in returned if any(matches(r, p) for p in forbidden)]
        if leaked:
            return 1.0 - len(leaked) / len(returned), "Forbidden docs returned: %s" % leaked
        return 1.0, "No forbidden docs returned"


class ProductIsolation(DeterministicMetric):
    """Every returned result must belong to the product that was searched (no cross-product leaks)."""

    metric_name = "ProductIsolation"
    default_threshold = 1.0

    def compute(self, test_case):
        wanted = tool_arguments(test_case).get("product") or case_meta(test_case).get("product")
        results = tool_payload(test_case).get("results", [])
        if not results:
            return 1.0, "No results to check"
        foreign = [r["id"] for r in results if r.get("product") != wanted]
        if foreign:
            return 1.0 - len(foreign) / len(results), "Results from other products: %s" % foreign
        return 1.0, "All %d results belong to %s" % (len(results), wanted)


class ResponseContract(DeterministicMetric):
    """Response shape: required fields, `limit` respected, scores sorted descending."""

    metric_name = "ResponseContract"
    default_threshold = 1.0

    def compute(self, test_case):
        payload = tool_payload(test_case)
        results = payload.get("results")
        problems = []
        if not isinstance(results, list):
            return 0.0, "`results` is missing or not a list"
        for key in ("query", "engine", "total_matches"):
            if key not in payload:
                problems.append("missing top-level key %r" % key)
        for r in results:
            missing = [f for f in REQUIRED_RESULT_FIELDS if f not in r]
            if missing:
                problems.append("%s missing fields %s" % (r.get("id", "?"), missing))
        limit = tool_arguments(test_case).get("limit", DEFAULT_LIMIT)
        if len(results) > limit:
            problems.append("returned %d results, limit is %d" % (len(results), limit))
        scores = [r.get("score", 0) for r in results]
        if scores != sorted(scores, reverse=True):
            problems.append("results are not sorted by score")
        if problems:
            return 0.0, "; ".join(problems)
        return 1.0, "Response matches the contract"


class Latency(DeterministicMetric):
    """Tool call latency. Passes when it is at or under the threshold (ms)."""

    metric_name = "Latency"

    def __init__(self, threshold_ms: float = None):
        if threshold_ms is None:
            threshold_ms = float(os.environ.get("LATENCY_THRESHOLD_MS", "500"))
        super().__init__(threshold=1.0)
        self.threshold_ms = threshold_ms

    def compute(self, test_case):
        latency_ms = (test_case.completion_time or 0.0) * 1000
        score = 1.0 if latency_ms <= self.threshold_ms else self.threshold_ms / latency_ms
        return score, "%.1f ms (threshold %.0f ms)" % (latency_ms, self.threshold_ms)


def build_search_metrics() -> List[BaseMetric]:
    """Fresh metric instances (metrics keep per-case state, so don't share them)."""
    return [
        RecallAtK(),
        FirstHitRank(),
        ForbiddenAbsent(),
        ProductIsolation(),
        ResponseContract(),
        Latency(),
    ]
