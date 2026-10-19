"""Agent-in-the-loop metrics, including LLM-as-a-judge.

These score a test case whose `actual_output` is the final answer of a Cursor agent that chose the
tool and its arguments itself (see `agents.cursor.run_agent`). The judge metrics send a rubric to a
second, tool-less Cursor agent (`agents.cursor.judge`) that replies with ``{"score": 0-10, "reason": ...}``.

Grouped by what they measure (the dashboard shows the same three groups):

  Correctness - is what the agent did and said right?
    AgentToolUse        deterministic  did it call the search tool and get a usable payload?
    AgentRecall         deterministic  did its search return the expected docs?
    AnswerFacts         deterministic  does the answer contain the golden's `must_include` facts?
    Judge:Arguments     LLM judge      did it translate the question into correct tool arguments?
    Judge:Correctness   LLM judge      is the answer right (vs. the reference answer / retrieved docs)?
  Relevancy - is it about what the customer asked?
    Judge:Relevance         LLM judge  are the doc sections the tool returned relevant to the question?
    Judge:AnswerRelevancy   LLM judge  does the final answer address the question?
  Quality - is it a good answer?
    Judge:Quality       LLM judge      clear, concise, helpful, well-toned?

Goldens may carry a `reference_answer` (ground truth written by the product owner) and `must_include`
(short facts that must appear in the answer). Both are optional.
"""

import json
import re
from typing import Any, Callable, Dict, List, Tuple

from deepeval.test_case import LLMTestCase

from metrics.base import DeterministicMetric, case_meta, found_refs

JUDGE_THRESHOLD = 0.7

# The judge gets the model's raw data, so keep it short.
_MAX_RESULTS = 8
_MAX_TEXT = 700

JUDGE_PREAMBLE = """You are a strict evaluator for a support assistant that answers questions from product documentation.
Grade the item below against the rubric. Reply with ONLY a JSON object, no markdown:
{"score": <integer 0-10>, "reason": "<one or two sentences>"}
"""

ARGUMENTS_RUBRIC = """Rubric - correctness of the tool arguments (10 = perfect, 0 = useless):
The tool takes `query` (keywords), `product` (whose documentation to search) and `limit`.
Are the arguments a correct translation of the customer's question? `product` must be the product the
customer is asking about (given below). `query` should keep the meaningful keywords of the question
and must not invent terms the customer never used. If the question is about something the product
probably does not document, searching for it anyway is acceptable."""

CORRECTNESS_RUBRIC = """Rubric - correctness of the final answer (10 = perfect, 0 = useless):
Judge only factual correctness. If a reference answer is given it is the ground truth: the answer must agree
with it (wording may differ) and include every required fact. Otherwise judge against the retrieved documentation.
Anything stated that is neither in the reference nor in the retrieved documentation is an error (invented features,
numbers, steps). If the documentation does not cover the question, the correct answer says so plainly instead of
guessing. Style does not matter here."""

RELEVANCE_RUBRIC = """Rubric - relevance of the retrieved documentation (10 = perfect, 0 = useless):
Are the documentation sections returned by the tool relevant to the customer's question, with the best match first?
If the product's documentation does not cover the question, an empty result list (or only clearly tangential
hits that the assistant ignores) is the CORRECT outcome and scores 10. An empty list for a question the
documentation plausibly covers scores low."""

ANSWER_RELEVANCY_RUBRIC = """Rubric - relevancy of the final answer (10 = perfect, 0 = useless):
Does the answer address exactly what the customer asked, staying on topic? Penalise ignoring part of the
question, padding with unrelated content, or answering about the wrong product. If the question is outside what the
product documents, a short answer that says so (and points to something useful) is ideal."""

QUALITY_RUBRIC = """Rubric - quality of the final answer (10 = perfect, 0 = useless):
Judge writing and helpfulness, not factual accuracy. A good answer is clear, concise (a few short lines),
easy to scan, polite and natural in tone, and gives the customer something actionable (the exact steps or
settings path, or a sensible next step). Penalise rambling, repetition, jargon, awkward formatting or a robotic tone."""


def _tool_name(test_case: LLMTestCase) -> str:
    return case_meta(test_case).get("tool_name", "search_docs")


def _tool_names(test_case: LLMTestCase) -> List[str]:
    meta = case_meta(test_case)
    names = meta.get("tool_names")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return [_tool_name(test_case)]


def _agent_call(test_case: LLMTestCase) -> Dict[str, Any]:
    tools = _tool_names(test_case)
    for call in case_meta(test_case).get("agent_tool_calls", []):
        name = str(call.get("name", ""))
        if any(name == tool or name.endswith(tool) for tool in tools):
            return call
    return {}


def _all_results(call: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = call.get("output")
    results = output.get("results", []) if isinstance(output, dict) else []
    return [r for r in results if isinstance(r, dict)]


def _docs(call: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"id": r.get("id"), "section": r.get("section"), "text": str(r.get("text", ""))[:_MAX_TEXT]}
        for r in _all_results(call)[:_MAX_RESULTS]
    ]


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _product_line(test_case: LLMTestCase) -> str:
    meta = case_meta(test_case)
    return "Product the customer is asking about: %s (id: %s)" % (meta.get("product_name", "?"), meta.get("product", "?"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


class AgentToolUse(DeterministicMetric):
    """1.0 when the agent called the search tool and the call returned a parsable payload."""

    metric_name = "AgentToolUse"
    default_threshold = 1.0

    def compute(self, test_case):
        agent_error = case_meta(test_case).get("agent_error")
        if agent_error:
            return 0.0, "Agent run failed: %s" % agent_error
        tool = ", ".join(_tool_names(test_case))
        call = _agent_call(test_case)
        if not call:
            names = [c.get("name") for c in case_meta(test_case).get("agent_tool_calls", [])]
            return 0.0, "Agent never called %s (tools called: %s)" % (tool, names or "none")
        if not isinstance(call.get("output"), dict) or "results" not in call["output"]:
            return 0.0, "Tool call returned no usable results payload"
        return 1.0, "Agent called %s with %s" % (tool, json.dumps(call.get("arguments")))


class AgentRecall(DeterministicMetric):
    """Like Recall@K, but for the search the agent ran itself with the arguments it chose."""

    metric_name = "AgentRecall"
    default_threshold = 1.0

    def compute(self, test_case):
        if case_meta(test_case).get("agent_error"):
            return 0.0, "Agent run failed: %s" % case_meta(test_case)["agent_error"]
        call = _agent_call(test_case)
        if not call:
            return 0.0, "Agent never called %s" % ", ".join(_tool_names(test_case))
        returned = [r.get("id") for r in _all_results(call)]
        expected = case_meta(test_case).get("expected_ids", [])
        if not expected:
            ok = not returned
            return (1.0 if ok else 0.0), (
                "No results, as expected" if ok else "Expected no results but the agent's search returned %s" % returned
            )
        found = found_refs(returned, expected)
        missing = [i for i in expected if i not in found]
        reason = "Agent's search found %d/%d expected docs" % (len(found), len(expected))
        if missing:
            reason += "; missing %s" % missing
        return len(found) / len(expected), reason


class AnswerFacts(DeterministicMetric):
    """Share of the golden's `must_include` facts that appear (case-insensitively) in the answer."""

    metric_name = "AnswerFacts"
    default_threshold = 1.0

    def compute(self, test_case):
        facts = case_meta(test_case).get("must_include", [])
        if not facts:
            return 1.0, "No required facts defined"
        if case_meta(test_case).get("agent_error"):
            return 0.0, "Agent run failed: %s" % case_meta(test_case)["agent_error"]
        answer = _norm(test_case.actual_output)
        found = [f for f in facts if _norm(f) in answer]
        missing = [f for f in facts if f not in found]
        reason = "Answer contains %d/%d required facts" % (len(found), len(facts))
        if missing:
            reason += "; missing %s" % missing
        return len(found) / len(facts), reason


class _JudgeMetric(DeterministicMetric):
    """Subclasses build the prompt; this class runs the judge and turns its 0-10 verdict into 0-1."""

    default_threshold = JUDGE_THRESHOLD
    rubric = ""

    def __init__(self, judge: Callable[[str], Dict[str, Any]], threshold: float = None):
        super().__init__(threshold)
        self._judge = judge

    def build_prompt(self, test_case: LLMTestCase) -> str:
        raise NotImplementedError

    def compute(self, test_case) -> Tuple[float, str]:
        if case_meta(test_case).get("agent_error"):
            return 0.0, "Skipped judge: agent run failed (%s)" % case_meta(test_case)["agent_error"]
        verdict = self._judge(JUDGE_PREAMBLE + "\n" + self.rubric + "\n\n" + self.build_prompt(test_case))
        try:
            raw = float(verdict["score"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("judge verdict has no numeric score: %r" % (verdict,))
        return min(max(raw, 0.0), 10.0) / 10.0, "%s/10 - %s" % (
            ("%g" % raw), str(verdict.get("reason", "")).strip() or "no reason given"
        )


class JudgeArguments(_JudgeMetric):
    metric_name = "Judge:Arguments"
    rubric = ARGUMENTS_RUBRIC

    def build_prompt(self, test_case):
        call = _agent_call(test_case)
        return "%s\n\nCustomer question:\n%s\n\nArguments the assistant passed to %s:\n%s" % (
            _product_line(test_case), test_case.input, _tool_name(test_case),
            _dump(call.get("arguments") if call else None),
        )


class JudgeCorrectness(_JudgeMetric):
    metric_name = "Judge:Correctness"
    rubric = CORRECTNESS_RUBRIC

    def build_prompt(self, test_case):
        meta = case_meta(test_case)
        reference = meta.get("reference_answer")
        facts = meta.get("must_include")
        parts = [_product_line(test_case), "Customer question:\n%s" % test_case.input]
        if reference:
            parts.append("Reference answer (ground truth):\n%s" % reference)
        if facts:
            parts.append("Required facts:\n%s" % _dump(facts))
        parts.append("Documentation the assistant retrieved:\n%s" % _dump(_docs(_agent_call(test_case))))
        parts.append("Assistant's final answer:\n%s" % test_case.actual_output)
        return "\n\n".join(parts)


class JudgeRelevance(_JudgeMetric):
    metric_name = "Judge:Relevance"
    rubric = RELEVANCE_RUBRIC

    def build_prompt(self, test_case):
        call = _agent_call(test_case)
        return "%s\n\nCustomer question:\n%s\n\nArguments used: %s\n\nDocumentation returned (best first):\n%s" % (
            _product_line(test_case), test_case.input,
            _dump(call.get("arguments") if call else None), _dump(_docs(call)),
        )


class JudgeAnswerRelevancy(_JudgeMetric):
    metric_name = "Judge:AnswerRelevancy"
    rubric = ANSWER_RELEVANCY_RUBRIC

    def build_prompt(self, test_case):
        return "%s\n\nCustomer question:\n%s\n\nAssistant's final answer:\n%s" % (
            _product_line(test_case), test_case.input, test_case.actual_output)


class JudgeQuality(_JudgeMetric):
    metric_name = "Judge:Quality"
    rubric = QUALITY_RUBRIC

    def build_prompt(self, test_case):
        return "Customer question:\n%s\n\nAssistant's final answer:\n%s" % (test_case.input, test_case.actual_output)


def build_agent_metrics(judge: Callable[[str], Dict[str, Any]]) -> List[DeterministicMetric]:
    """Fresh instances; `judge` maps a prompt to the parsed JSON verdict. Order = Correctness, Relevancy, Quality."""
    return [
        AgentToolUse(),
        AgentRecall(),
        AnswerFacts(),
        JudgeArguments(judge),
        JudgeCorrectness(judge),
        JudgeRelevance(judge),
        JudgeAnswerRelevancy(judge),
        JudgeQuality(judge),
    ]
