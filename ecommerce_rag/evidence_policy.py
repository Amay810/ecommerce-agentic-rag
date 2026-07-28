"""Evidence-aware variants of the executable LLM policy."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .domain import AgentAction, AgentObservation
from .evidence import render_evidence_ledger, verify_answer
from .llm_policy import LLMPolicy


EVIDENCE_INSTRUCTIONS = """
EVIDENCE CONTRACT:
- For factual claims, use only the EVIDENCE LEDGER below.
- Cite the supporting evidence id immediately after the claim, for example [E3].
- Never cite an evidence id that does not support the exact value or state.
- If evidence is insufficient, say so or hand off; do not fill gaps from memory.
""".strip()


class EvidenceGroundedPolicy(LLMPolicy):
    """LLMPolicy with deterministic verification and optional one-shot repair.

    ``repair=False`` implements the ``evidence_verify`` ablation.  ``repair=True``
    implements ``evidence_verify_repair``.  The original :class:`LLMPolicy`
    remains the untouched ``base`` variant.
    """

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
        return (
            f"{prompt}\n\n{EVIDENCE_INSTRUCTIONS}\n\nEVIDENCE LEDGER:\n"
            f"{render_evidence_ledger(observation.evidence_ledger)}"
        )

    @staticmethod
    def _runtime_pass(result: dict[str, Any]) -> bool:
        # Non-factual refusals or handoff notices can legitimately have no
        # evidence. Once evidence is applicable, both fact support and binding
        # are required for evidence-aware variants.
        return (not result["applicable"]) or (
            result["answer_fact_pass"] and result["citation_binding_pass"]
        )

    def _verify(self, action: AgentAction, observation: AgentObservation, *, phase: str) -> dict[str, Any]:
        result = verify_answer(
            action.content,
            observation.evidence_ledger,
            require_citations=bool(observation.evidence_ledger),
        )
        span = {"phase": phase, "answer": action.content, **result, "passed": self._runtime_pass(result)}
        self.last_verification_spans.append(span)
        return span

    def act(self, observation: AgentObservation) -> AgentAction:
        self.last_verification_spans = []
        self.last_repair_spans = []
        action = LLMPolicy.act(self, observation)
        initial_trace = copy.deepcopy(self.last_trace)

        # User questions such as verification code and confirmation are not a
        # terminal answer and therefore are not answer-verification targets.
        if action.action_type != "final_answer" or action.requires_user_response:
            return action

        initial_verification = self._verify(action, observation, phase="initial")
        if initial_verification["passed"]:
            return action

        if not self.repair:
            self.last_trace = {**initial_trace, "evidence_variant": self.variant_name,
                               "verification": copy.deepcopy(initial_verification)}
            return AgentAction.handoff("answer_verification_failed")

        repair_message = (
            "The deterministic evidence verifier rejected your proposed final answer. "
            "Produce one corrected final_answer JSON action. Use only the ledger, cite "
            "the exact supporting [E#] after each factual claim, and remove unsupported "
            f"or contradicted claims. Verifier result: {initial_verification}"
        )
        repair_observation = replace(
            observation,
            current_message=repair_message,
            history=[
                *copy.deepcopy(observation.history),
                {"role": "assistant", "content": action.content, "action": "rejected_final_answer"},
                {"role": "verifier", "content": repair_message},
            ],
        )
        repaired = LLMPolicy.act(self, repair_observation)
        repair_trace = copy.deepcopy(self.last_trace)
        repair_span: dict[str, Any] = {
            "attempt": 1,
            "original_answer": action.content,
            "requested_reason": initial_verification,
            "action": {
                "action_type": repaired.action_type,
                "tool_name": repaired.tool_name,
                "arguments": repaired.arguments,
                "content": repaired.content,
                "requires_user_response": repaired.requires_user_response,
            },
            "llm": repair_trace,
        }
        self.last_repair_spans.append(repair_span)

        if repaired.action_type != "final_answer" or repaired.requires_user_response:
            repair_span["passed"] = False
            repair_span["failure"] = "repair_did_not_return_terminal_answer"
            self.last_trace = {**initial_trace, "evidence_variant": self.variant_name,
                               "verification": copy.deepcopy(initial_verification),
                               "repair_llm": repair_trace, "repair_resolution": "failed"}
            return AgentAction.handoff("answer_repair_failed")

        repaired_verification = self._verify(repaired, observation, phase="repair")
        repair_span["verification"] = copy.deepcopy(repaired_verification)
        repair_span["passed"] = repaired_verification["passed"]
        self.last_trace = {**initial_trace, "evidence_variant": self.variant_name,
                           "verification": copy.deepcopy(initial_verification),
                           "repair_llm": repair_trace,
                           "repair_resolution": "passed" if repaired_verification["passed"] else "failed"}
        return repaired if repaired_verification["passed"] else AgentAction.handoff("answer_repair_failed")
