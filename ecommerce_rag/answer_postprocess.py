"""Answer-only postprocessing for frozen trajectories.

The postprocessor consumes a draft and evidence after planning has ended.  It
cannot call tools, mutate state, or turn a base handoff into an answer.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .claim_verifier import classify_claim
from .evidence import render_evidence_ledger
from .llm_policy import Generation
from .verifier import split_sentences


GROUNDING_SYSTEM = """You revise a completed customer-support draft using only the supplied evidence.
Return plain answer text, not JSON. Preserve the user's language. Do not call tools, request new information,
or claim that an operation occurred unless the evidence says so. Cite factual statements with individual [E#] ids.
If evidence is insufficient, state the limitation rather than inventing a fact."""
GROUNDING_PROMPT_SHA256 = hashlib.sha256(GROUNDING_SYSTEM.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PostprocessResult:
    mode: str
    eligible: bool
    ineligible_reason: str | None
    draft_answer: str
    final_answer: str
    changed: bool
    verification: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    truncated: bool = False
    latency_ms: float = 0.0
    generation_config_hash: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerPostprocessor:
    MODES = frozenset({"shadow", "terminal_grounded"})

    def __init__(self, generate: Callable[[str, str], Generation] | None = None,
                 *, generation_config: dict[str, Any] | None = None):
        self.generate = generate
        self.generation_config = generation_config or {}

    @staticmethod
    def eligibility(trajectory: dict[str, Any], grade: dict[str, Any]) -> str | None:
        actions = trajectory.get("actions") or []
        if any(action.get("action_type") == "handoff" for action in actions):
            return "base_handoff"
        terminal = actions[-1] if actions else None
        if not terminal or terminal.get("action_type") != "final_answer" or terminal.get("requires_user_response"):
            return "no_terminal_draft"
        if not str(trajectory.get("final_answer") or "").strip():
            return "no_terminal_draft"
        if not trajectory.get("evidence_ledger"):
            return "no_evidence"
        if not grade.get("answer_fact_applicable", False):
            return "not_fact_applicable"
        return None

    @staticmethod
    def _verify(answer: str, ledger: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        claims = split_sentences(answer) or ([answer.strip()] if answer.strip() else [])
        return [classify_claim(claim, ledger, user_messages=messages, citation_required=True).to_dict()
                | {"claim": claim} for claim in claims]

    def process(self, draft_answer: str, evidence_ledger: list[dict[str, Any]],
                user_messages: list[dict[str, Any]], mode: str) -> PostprocessResult:
        if mode not in self.MODES:
            raise ValueError(f"unsupported postprocess mode: {mode}")
        started = time.perf_counter()
        if mode == "shadow":
            verification = self._verify(draft_answer, evidence_ledger, user_messages)
            return PostprocessResult(mode, True, None, draft_answer, draft_answer, False,
                                     verification=verification,
                                     latency_ms=(time.perf_counter() - started) * 1000)
        if self.generate is None:
            raise RuntimeError("terminal_grounded requires a generator")
        config_hash = stable_hash(self.generation_config)
        if not self.generation_config or not config_hash:
            raise RuntimeError("generation configuration is required")
        user = (f"DRAFT ANSWER:\n{draft_answer}\n\nEVIDENCE LEDGER:\n"
                f"{render_evidence_ledger(evidence_ledger)}\n\nRewrite the final answer.")
        try:
            generated = self.generate(GROUNDING_SYSTEM, user)
            if isinstance(generated, str):
                generated = Generation(text=generated)
        except Exception as exc:  # fail closed; caller must fail the run
            return PostprocessResult(mode, True, None, draft_answer, "", False,
                                     latency_ms=(time.perf_counter() - started) * 1000,
                                     generation_config_hash=config_hash,
                                     error=f"{type(exc).__name__}: {exc}")
        final = generated.text.strip()
        error = "generation_truncated" if generated.truncated else "empty_generation" if not final else None
        verification = self._verify(final, evidence_ledger, user_messages) if final else []
        return PostprocessResult(
            mode, True, None, draft_answer, final, final != draft_answer,
            verification=verification, raw_output=generated.text,
            prompt_tokens=generated.prompt_tokens, completion_tokens=generated.completion_tokens,
            finish_reason=generated.finish_reason, truncated=generated.truncated,
            latency_ms=(time.perf_counter() - started) * 1000,
            generation_config_hash=config_hash, error=error,
        )
