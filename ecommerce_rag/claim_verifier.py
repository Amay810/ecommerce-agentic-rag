"""Claim-level verifier used by offline calibration and answer postprocessing.

This module deliberately does not participate in agent planning.  Its labels
describe the relationship between one claim and the evidence available in the
frozen trajectory; ``unsupported`` means only that the ledger is insufficient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

from .evidence import (
    EVIDENCE_CITATION,
    _explicit_contradiction,
    _facts,
    _fact_supported,
    _row_supports_sentence,
    extract_user_context,
)


FACT_STATUSES = frozenset({"supported", "contradicted", "unsupported", "not_factual", "unknown"})
CITATION_STATUSES = frozenset({"correct", "incorrect", "missing", "not_required", "unknown"})
_ENTITY = re.compile(r"\b(?:P[0-9]{5}|O[0-9]{6}|POL[0-9]{3})\b", re.I)
_PRICE_RELATION = re.compile(
    r"(?P<price>[0-9]+(?:\.[0-9]+)?)\s*元[^0-9]{0,8}"
    r"(?P<relation>超过|高于|大于|不超过|低于|小于|以内)"
    r"[^0-9]{0,10}(?P<budget>[0-9]+(?:\.[0-9]+)?)\s*元"
)


@dataclass(frozen=True)
class ClaimVerification:
    fact_status: str
    citation_status: str
    factual: bool
    automatic_decision: bool
    hard_failure: bool
    reasons: list[str]
    cited_evidence_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _subject_rows(claim: str, ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entities = {value.upper() for value in _ENTITY.findall(claim)}
    if not entities:
        return ledger
    selected = [row for row in ledger if any(
        entity in str(row.get(key, "")).upper()
        for entity in entities for key in ("source_id", "value", "text")
    )]
    return selected or ledger


def _range_contradiction(claim: str, budgets: list[float]) -> bool:
    match = _PRICE_RELATION.search(claim)
    if match:
        price, budget = float(match.group("price")), float(match.group("budget"))
        relation = match.group("relation")
        expected = price > budget if relation in {"超过", "高于", "大于"} else price <= budget
        if not expected:
            return True
    amounts = [float(value) for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*元", claim)]
    prices = [value for value in amounts if value not in budgets]
    if budgets and prices:
        price, budget = prices[-1], budgets[-1]
        if re.search(r"超出预算|超过预算", claim) and price <= budget:
            return True
        if re.search(r"未达到[^。！？]{0,12}预算要求[^。！？]{0,12}(?:不符合|不推荐)", claim) and price <= budget:
            return True
    return False


def classify_claim(
    claim: str,
    ledger: list[dict[str, Any]],
    *,
    user_messages: list[dict[str, Any]] | None = None,
    citation_required: bool = False,
) -> ClaimVerification:
    """Classify one atomic claim without changing any agent action.

    ``unknown`` is an abstention.  It never becomes a hard failure.  Only an
    explicit contradiction or an invalid/opposing citation is hard-failing.
    """
    text = claim.strip()
    by_id = {str(row.get("evidence_id")): row for row in ledger}
    cited = [f"E{value}" for value in EVIDENCE_CITATION.findall(text)]
    missing = [eid for eid in cited if eid not in by_id]
    cited_rows = [by_id[eid] for eid in cited if eid in by_id]
    clean = EVIDENCE_CITATION.sub("", text).strip()
    facts = _facts(clean)
    entities = _ENTITY.findall(clean)
    factual = bool(facts or entities or re.search(
        r"价格|库存|状态|日期|期限|退货|保修|支持|兼容|重量|尺寸|型号|停产|预算", clean, re.I
    ))
    if not text:
        return ClaimVerification("unknown", "unknown", False, False, False,
                                 ["empty_claim"], cited)
    if not factual:
        return ClaimVerification("not_factual", "not_required", False, True, False,
                                 ["no_factual_signal"], cited)

    subject_rows = _subject_rows(clean, ledger)
    user_context = extract_user_context(user_messages)
    comparison_text = clean
    for budget in user_context.get("budgets", []):
        comparison_text = re.sub(rf"(?<![0-9]){budget:g}(?:\.0+)?\s*元", "", comparison_text)
    contradiction = _explicit_contradiction(comparison_text, subject_rows)
    if re.search(r"未送达", clean) and any(
        row.get("field") == "order.status" and str(row.get("value")).lower() == "delivered"
        for row in subject_rows
    ):
        contradiction = {"reason": "delivered_evidence_opposes_undelivered_claim"}
    range_conflict = _range_contradiction(clean, user_context.get("budgets", []))
    if contradiction or range_conflict:
        fact_status = "contradicted"
        reasons = ["explicit_contradiction" if contradiction else "range_logic_contradiction"]
    elif facts and all(_fact_supported(fact, subject_rows, user_context) for fact in facts):
        fact_status, reasons = "supported", ["all_structured_facts_supported"]
    elif facts:
        fact_status, reasons = "unsupported", ["ledger_has_no_support"]
    elif re.search(r"(?:所有产品|\bP[0-9]{5}\b)[^。！？]{0,40}(?:支持|兼容|具备|容量|重量|尺寸)", clean, re.I):
        fact_status, reasons = "unsupported", ["unsupported_product_attribute"]
    elif not ledger and re.search(r"无法找到|没有找到|不存在|未检索到", clean):
        fact_status, reasons = "unsupported", ["no_search_evidence_for_no_answer"]
    elif any(_row_supports_sentence(clean, row) for row in subject_rows):
        fact_status, reasons = "supported", ["lexical_evidence_binding"]
    else:
        fact_status, reasons = "unknown", ["claim_not_deterministically_bindable"]

    if missing:
        citation_status = "incorrect"
        reasons.append("invalid_evidence_id")
    elif cited_rows:
        cited_opposition = _explicit_contradiction(clean, cited_rows)
        if cited_opposition:
            citation_status = "incorrect"
            reasons.append("cited_evidence_supports_opposite")
        elif any(_row_supports_sentence(clean, row) for row in cited_rows):
            citation_status = "correct"
        else:
            citation_status = "incorrect"
            reasons.append("citation_does_not_bind")
    else:
        citation_status = "missing" if citation_required else "not_required"

    hard = fact_status == "contradicted" or bool(missing) or "cited_evidence_supports_opposite" in reasons
    return ClaimVerification(
        fact_status=fact_status,
        citation_status=citation_status,
        factual=True,
        automatic_decision=fact_status != "unknown",
        hard_failure=hard,
        reasons=reasons,
        cited_evidence_ids=cited,
    )
