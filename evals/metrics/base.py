"""Shared plumbing for every metric: reading a test case, matching doc references, and the base class.

Each test case carries:
  * actual_output         - JSON string with the tool's response payload (or the agent's final answer)
  * tools_called[0]       - the tool name + the arguments it was called with
  * completion_time       - tool call latency in seconds
  * additional_metadata   - {"expected_ids": [...], "forbidden_ids": [...], "product": ...}
                            expected/forbidden entries are doc references, see `matches()`

Metrics subclass `deepeval.metrics.BaseMetric` (through `DeterministicMetric`), so they can be used with
`deepeval.evaluate(...)`, `assert_test(...)` or - as `run_evals.py` does - by calling `.measure()` directly.
"""

import json
from typing import Any, Dict, List, Tuple

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


def tool_payload(test_case: LLMTestCase) -> Dict[str, Any]:
    try:
        return json.loads(test_case.actual_output)
    except (TypeError, ValueError):
        raise ValueError("tool call failed: %s" % (test_case.actual_output or "no output"))


def result_ids(test_case: LLMTestCase) -> List[str]:
    return [r["id"] for r in tool_payload(test_case).get("results", [])]


def matches(result_id: str, pattern: str) -> bool:
    """Does a returned chunk id (``product-one/pricing.md#plans``) satisfy a golden's doc reference?

    * ``product-one/pricing.md#plans`` - exactly that section
    * ``product-one/pricing.md``       - any section of that file
    * ``product-two/*``                - anything under the prefix
    """
    if pattern.endswith("*"):
        return result_id.startswith(pattern[:-1])
    return result_id == pattern or result_id.startswith(pattern + "#")


def found_refs(returned: List[str], patterns: List[str]) -> List[str]:
    """The doc references (patterns) that at least one returned id satisfies."""
    return [p for p in patterns if any(matches(r, p) for r in returned)]


def case_meta(test_case: LLMTestCase) -> Dict[str, Any]:
    return test_case.additional_metadata or {}


def tool_arguments(test_case: LLMTestCase) -> Dict[str, Any]:
    calls = test_case.tools_called or []
    return dict(calls[0].input_parameters or {}) if calls else {}


class DeterministicMetric(BaseMetric):
    """Shared plumbing: subclasses only implement `compute()`."""

    metric_name = "Metric"
    default_threshold = 1.0

    def __init__(self, threshold: float = None):
        self.threshold = self.default_threshold if threshold is None else threshold
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.error = None
        # Attributes some DeepEval versions read from every metric.
        self.strict_mode = False
        self.async_mode = False
        self.include_reason = True
        self.evaluation_model = None
        self.verbose_mode = False

    def compute(self, test_case: LLMTestCase) -> Tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        try:
            self.score, self.reason = self.compute(test_case)
            self.error = None
        except Exception as exc:  # a broken tool response is a failed case, not a crash
            self.score, self.reason, self.error = 0.0, str(exc), str(exc)
        self.score = round(float(self.score), 4)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self):
        return self.metric_name
