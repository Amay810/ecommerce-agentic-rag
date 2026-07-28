"""Evidence-aware variants of the executable LLM policy."""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Any

from .domain import AgentAction, AgentObservation
from .evidence import render_evidence_ledger, verify_answer
from .llm_policy import LLMPolicy


EVIDENCE_INSTRUCTIONS = """
EVIDENCE CONTRACT:
- Use only the EVIDENCE LEDGER for product, order, inventory, eligibility and policy facts.
- Put each supporting evidence id immediately after the fact, for example [E3].
- A sentence with several facts must cite every supporting id separately: [E2][E7].
- Never use evidence ranges such as [E1-E5], and never cite an id that is absent or supports another value.
- User-provided budgets and identifiers may be repeated without citation; they do not prove business facts.
- If evidence is insufficient, explicitly say so; do not fill gaps from memory.
""".strip()


def _diagnostic_count(result: dict[str, Any]) -> int:
    return (len(result.get("unsupported_high_risk_claims") or [])
            + len(result.get("citation_diagnostics") or [])
            + len(result.get("omitted_required_facts") or []))


def _compact_repair_reason(result: dict[str, Any]) -> dict[str, Any]:
    """Exclude freshness, full verifier JSON and unrelated successful checks."""
    return {
        "hard": {
            "invalid_evidence_ids": result.get("invalid_citations") or [],
            "contradictions": result.get("contradicted_claims") or [],
            "cited_oppositions": result.get("citation_oppositions") or [],
        },
        "diagnostic": {
            "unsupported": result.get("unsupported_high_risk_claims") or [],
            "citations": result.get("citation_diagnostics") or [],
            "omitted_required_facts": result.get("omitted_required_facts") or [],
        },
    }


def _relevant_evidence(result: dict[str, Any], ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids: set[str] = set()
    fields: set[str] = set(result.get("omitted_required_facts") or [])
    for group in ("contradicted_claims", "citation_oppositions", "citation_failures"):
        for row in result.get(group) or []:
            ids.update(row.get("evidence_ids") or [])
            if row.get("field"):
                fields.add(str(row["field"]))
    selected = [row for row in ledger if str(row.get("evidence_id")) in ids
                or any(str(row.get("field", "")).startswith(field.rstrip("*")) for field in fields)]
    if not selected:
        failure_text = " ".join(str(item.get("sentence") or item.get("claim") or "")
                                for group in ("unsupported_high_risk_claims", "citation_diagnostics")
                                for item in result.get(group) or [])
        tokens = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", failure_text.lower()))
        selected = sorted(ledger, key=lambda row: len(tokens & set(re.findall(
            r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}",
            f"{row.get('field', '')} {row.get('value', '')} {row.get('text', '')}".lower()))), reverse=True)
    return selected[:12]


class EvidenceGroundedPolicy(LLMPolicy):
    uses_evidence = True

    def __init__(self, *args: Any, repair: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.repair = repair
        self.last_verification_spans: list[dict[str, Any]] = []
        self.last_repair_spans: list[dict[str, Any]] = []

    @property
    def variant_name(self) -> str:
        return "evidence_verify_repair" if self.repair else "evidence_verify"

    def _build_prompt(self, observation: AgentObservation) -> str:
        prompt = super()._build_prompt(observation)
        if not observation.evidence_ledger:
            return prompt
        return f"{prompt}\n\n{EVIDENCE_INSTRUCTIONS}\n\nEVIDENCE LEDGER:\n{render_evidence_ledger(observation.evidence_ledger)}"

    def _verify(self, action: AgentAction, observation: AgentObservation, *, phase: str) -> dict[str, Any]:
        result = verify_answer(action.content, observation.evidence_ledger,
                               require_citations=bool(observation.evidence_ledger),
                               user_messages=observation.history)
        span = {"phase": phase, "answer": action.content, **result,
                "passed": bool(result["hard_verification_pass"]),
                "diagnostic_count": _diagnostic_count(result)}
        self.last_verification_spans.append(span)
        return span

    def act(self, observation: AgentObservation) -> AgentAction:
        self.last_verification_spans = []
        self.last_repair_spans = []
        action = LLMPolicy.act(self, observation)
        initial_trace = copy.deepcopy(self.last_trace)
        if action.action_type != "final_answer" or action.requires_user_response:
            return action

        initial = self._verify(action, observation, phase="initial")
        hard_failed = not initial["hard_verification_pass"]
        has_diagnostics = _diagnostic_count(initial) > 0
        if not hard_failed and (not self.repair or not has_diagnostics):
            return action
        if hard_failed and not self.repair:
            self.last_trace = {**initial_trace, "evidence_variant": self.variant_name,
                               "verification": copy.deepcopy(initial)}
            return AgentAction.handoff("answer_verification_failed")

        compact_reason = _compact_repair_reason(initial)
        relevant = _relevant_evidence(initial, observation.evidence_ledger)
        repair_message = (
            "Rewrite the rejected answer as one concise final_answer JSON action. "
            "Correct only the listed failures, cite each fact with separate [E#] ids, "
            "do not add new facts, and do not use citation ranges.\n"
            f"FAILED ANSWER: {action.content}\n"
            f"REASONS: {compact_reason}"
        )
        repair_observation = replace(
            observation, current_message=repair_message,
            history=[],
            evidence_ledger=relevant,
        )
        repaired = LLMPolicy.act(self, repair_observation)
        repair_trace = copy.deepcopy(self.last_trace)
        span: dict[str, Any] = {
            "attempt": 1, "original_answer": action.content, "requested_reason": compact_reason,
            "relevant_evidence_ids": [row.get("evidence_id") for row in relevant],
            "action": {"action_type": repaired.action_type, "tool_name": repaired.tool_name,
                       "arguments": repaired.arguments, "content": repaired.content,
                       "requires_user_response": repaired.requires_user_response},
            "llm": repair_trace, "hard_recovery": False, "diagnostic_improvement": False,
        }
        self.last_repair_spans.append(span)

        if repaired.action_type != "final_answer" or repaired.requires_user_response:
            span.update(passed=False, adopted=False, failure="repair_did_not_return_terminal_answer")
            self.last_trace = {**initial_trace, "evidence_variant": self.variant_name,
                               "verification": copy.deepcopy(initial), "repair_llm": repair_trace,
                               "repair_resolution": "failed"}
            return AgentAction.handoff("answer_repair_failed") if hard_failed else action

        repaired_verification = self._verify(repaired, observation, phase="repair")
        unsupported_not_worse = len(repaired_verification["unsupported_high_risk_claims"]) <= len(
            initial["unsupported_high_risk_claims"])
        diagnostic_improvement = _diagnostic_count(repaired_verification) < _diagnostic_count(initial)
        hard_recovery = hard_failed and repaired_verification["hard_verification_pass"]
        adopted = bool(repaired_verification["hard_verification_pass"] and unsupported_not_worse and (
            hard_recovery or diagnostic_improvement))
        span.update(verification=copy.deepcopy(repaired_verification), passed=adopted, adopted=adopted,
                    hard_recovery=hard_recovery and adopted,
                    diagnostic_improvement=diagnostic_improvement and adopted)
        self.last_trace = {**initial_trace, "evidence_variant": self.variant_name,
                           "verification": copy.deepcopy(initial), "repair_llm": repair_trace,
                           "repair_resolution": "adopted" if adopted else "rejected"}
        if adopted:
            return repaired
        return AgentAction.handoff("answer_repair_failed") if hard_failed else action
