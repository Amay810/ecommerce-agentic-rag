"""Shared evidence ledger and deterministic answer verification.

The ledger is derived exclusively from tool results that the policy has already
seen.  It contains no task labels or gold answers.  Runtime verification and
offline grading both call :func:`verify_answer`, preventing two implementations
from drifting apart.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from . import freshness
from .verifier import split_sentences


EVIDENCE_CITATION = re.compile(r"\[E([1-9][0-9]*)\]")
_ID_FACT = re.compile(r"\b(?:P[0-9]{5}|O[0-9]{6}|POL[0-9]{3})\b", re.I)
_DATE_FACT = re.compile(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b")
_NUMERIC_FACT = re.compile(
    r"(?<![A-Za-z0-9])[0-9]+(?:\.[0-9]+)?\s*(?:元|天|日|个工作日|工作日|小时|分钟|件|个|%|mAh|Pa|kg|g|ml)",
    re.I,
)
_STATE_TERMS = {
    "pending", "processed", "delivered", "cancelled", "requested",
    "available", "out_of_stock", "已交付", "已送达", "待处理", "处理中",
    "已取消", "缺货", "有货", "可退货", "不可退货", "符合退货条件",
    "不符合退货条件", "已停产", "未停产",
}


def _text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _source(prefix: str, row: dict[str, Any], fallback: str) -> str:
    doc_id = row.get("doc_id")
    if doc_id:
        return str(doc_id)
    preferred = {
        "product": ("product_id", "order_id", "handoff_id"),
        "order": ("order_id", "product_id", "handoff_id"),
        "policy": ("policy_id", "product_id", "order_id"),
    }.get(prefix, ("product_id", "order_id", "handoff_id"))
    for key in preferred:
        if row.get(key):
            return f"{prefix}:{row[key]}"
    return fallback


def evidence_from_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    call_id: str,
    *,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Normalize one successful tool result into stable, atomic evidence rows."""
    if not isinstance(result, dict) or not result.get("ok", False):
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(source_id: str, kind: str, field: str, value: Any, *, text: str | None = None,
            updated_at: str | None = None) -> None:
        if value is None:
            return
        key = (source_id, field, _text(value))
        if key in seen:
            return
        seen.add(key)
        rows.append({
            "evidence_id": f"E{start_index + len(rows)}",
            "source_id": source_id,
            "tool_call_id": call_id,
            "tool_name": tool_name,
            "kind": kind,
            "field": field,
            "value": value,
            "text": text if text is not None else _text(value),
            "confidence": 1.0,
            "updated_at": updated_at,
        })

    def add_product(product: dict[str, Any], *, prefix: str = "product") -> None:
        source_id = _source("product", product, f"tool:{call_id}")
        for field in ("product_id", "title", "category", "price", "inventory", "discontinued", "updated_at"):
            if field in product:
                add(source_id, f"{prefix}_field", f"product.{field}", product.get(field),
                    updated_at=product.get("updated_at"))

    def add_order(order: dict[str, Any]) -> None:
        source_id = _source("order", order, f"tool:{call_id}")
        for field in (
            "order_id", "product_id", "status", "ordered_at", "delivered_at",
            "opened", "quality_issue", "inventory_status", "return_status", "version",
        ):
            if field in order:
                add(source_id, "order_field", f"order.{field}", order.get(field))

    if tool_name == "search_catalog":
        for item in result.get("items", []):
            if isinstance(item, dict):
                add_product(item, prefix="catalog_candidate")
    elif tool_name == "get_product":
        product = result.get("product") or {}
        if isinstance(product, dict):
            add_product(product)
            source_id = _source("product", product, f"tool:{call_id}")
            for index, passage in enumerate(result.get("evidence", []), 1):
                add(source_id, "product_passage", f"product.evidence.{index}", passage, text=str(passage))
    elif tool_name == "compare_products":
        for wrapped in result.get("products", []):
            if not isinstance(wrapped, dict):
                continue
            product = wrapped.get("product") or {}
            if isinstance(product, dict):
                add_product(product, prefix="comparison_field")
                source_id = _source("product", product, f"tool:{call_id}")
                for index, passage in enumerate(wrapped.get("evidence", []), 1):
                    add(source_id, "product_passage", f"product.evidence.{index}", passage, text=str(passage))
    elif tool_name == "get_policy":
        canonical = arguments.get("policy_type")
        for policy in result.get("policies", []):
            if not isinstance(policy, dict):
                continue
            source_id = _source("policy", policy, f"tool:{call_id}")
            add(source_id, "policy_field", "policy.category", canonical)
            add(source_id, "policy_field", "policy.title", policy.get("title"))
            add(source_id, "policy_clause", "policy.text", policy.get("text"), text=policy.get("text"),
                updated_at=policy.get("updated_at"))
    elif tool_name == "get_order":
        order = result.get("order") or {}
        if isinstance(order, dict):
            add_order(order)
    elif tool_name == "check_return_eligibility":
        order = result.get("order") or {}
        source_id = _source("order", order if isinstance(order, dict) else {}, f"order:{arguments.get('order_id', call_id)}")
        for field in ("eligible", "reason", "days_since_delivery"):
            if field in result:
                add(source_id, "return_eligibility", f"return.{field}", result.get(field))
        if isinstance(order, dict):
            add_order(order)
    elif tool_name == "create_return_request":
        source_id = f"order:{result.get('order_id') or arguments.get('order_id') or call_id}"
        for field in ("changed", "order_id", "return_status"):
            if field in result:
                add(source_id, "return_transition", f"return.{field}", result.get(field))

    return rows


def render_evidence_ledger(ledger: list[dict[str, Any]], *, max_chars: int = 8000) -> str:
    """Compact prompt representation with stable evidence identifiers."""
    lines = []
    for row in ledger:
        text = str(row.get("text", "")).replace("\n", " ")
        lines.append(
            f"[{row.get('evidence_id')}] source={row.get('source_id')} "
            f"field={row.get('field')} value={_text(row.get('value'))} text={text}"
        )
    rendered = "\n".join(lines)
    return rendered if len(rendered) <= max_chars else rendered[:max_chars] + "…[evidence truncated]"


def freshness_from_ledger(answer: str, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Adapt ledger sources to the existing freshness diagnostic interface."""
    products: dict[str, dict[str, Any]] = {}
    policies: dict[str, dict[str, Any]] = {}
    for row in ledger:
        source_id = str(row.get("source_id", ""))
        target = policies if source_id.startswith("policy:") else products if source_id.startswith("product:") else None
        if target is None:
            continue
        item = target.setdefault(source_id, {"source_id": source_id, "default_updated_at": None})
        if row.get("updated_at"):
            item["default_updated_at"] = row["updated_at"]
    return freshness.assess(
        {"products": list(products.values()), "policies": list(policies.values())},
        intent="evidence_grounded_agent", answer=answer,
    )


def _sentences(answer: str) -> list[str]:
    # Models often put a citation immediately after terminal punctuation. Keep
    # that citation bound to the preceding sentence before using the shared
    # sentence splitter, otherwise ``事实。[E1]`` becomes an uncited fact plus a
    # discarded citation-only fragment.
    normalized = re.sub(r"([。！？.!?])(\s*(?:\[E[1-9][0-9]*\])+)", r"\2\1", answer)
    sentences = split_sentences(normalized)
    return sentences or ([answer.strip()] if answer.strip() else [])


def _facts(text: str) -> list[str]:
    facts = _ID_FACT.findall(text) + _DATE_FACT.findall(text) + _NUMERIC_FACT.findall(text)
    lowered = text.lower()
    # Prefer the longest non-overlapping state phrase so “不符合退货条件” does
    # not also emit the opposite substring “符合退货条件”.
    occupied: list[tuple[int, int]] = []
    for term in sorted(_STATE_TERMS, key=len, reverse=True):
        for match in re.finditer(re.escape(term.lower()), lowered):
            span = match.span()
            if any(not (span[1] <= used[0] or span[0] >= used[1]) for used in occupied):
                continue
            facts.append(term)
            occupied.append(span)
    return list(dict.fromkeys(x.strip() for x in facts if x.strip()))


def _corpus(rows: Iterable[dict[str, Any]]) -> str:
    return "\n".join(
        f"{row.get('source_id', '')} {row.get('field', '')} {_text(row.get('value'))} {row.get('text', '')}"
        for row in rows
    ).lower()


def _fact_supported(fact: str, rows: Iterable[dict[str, Any]]) -> bool:
    rows = list(rows)
    fact_l = re.sub(r"\s+", "", fact.lower())
    corpus = re.sub(r"\s+", "", _corpus(rows))
    if fact_l in corpus:
        return True
    amount = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)元", fact_l)
    if amount:
        expected = float(amount.group(1))
        for row in rows:
            if row.get("field") == "product.price" and isinstance(row.get("value"), (int, float)):
                if float(row["value"]) == expected:
                    return True
    aliases = {
        "已送达": "delivered", "已交付": "delivered", "待处理": "pending",
        "处理中": "processed", "已取消": "cancelled", "缺货": "out_of_stock",
        "有货": "available", "可退货": "true", "符合退货条件": "true",
        "不可退货": "false", "不符合退货条件": "false",
    }
    return aliases.get(fact_l, "\0") in corpus


def _explicit_contradiction(sentence: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    lowered = sentence.lower()
    checks = (
        ("已停产", "product.discontinued", True),
        ("未停产", "product.discontinued", False),
        ("不符合退货条件", "return.eligible", False),
        ("不可退货", "return.eligible", False),
        ("符合退货条件", "return.eligible", True),
        ("可退货", "return.eligible", True),
    )
    for phrase, field, claimed in checks:
        if phrase not in lowered:
            continue
        if phrase == "符合退货条件" and "不符合退货条件" in lowered:
            continue
        if phrase == "可退货" and "不可退货" in lowered:
            continue
        values = [row.get("value") for row in rows if row.get("field") == field]
        if values and claimed not in values:
            return {"sentence": sentence, "claim": phrase, "field": field,
                    "evidence_values": values, "reason": "explicit_contradiction"}

    state_fields = {
        "order.status": {"pending", "processed", "delivered", "cancelled"},
        "order.inventory_status": {"available", "out_of_stock"},
        "return.return_status": {"requested", "already_requested"},
    }
    for field, vocabulary in state_fields.items():
        claimed_values = [value for value in vocabulary if value in lowered]
        evidence_values = [str(row.get("value")).lower() for row in rows if row.get("field") == field]
        if claimed_values and evidence_values and not any(value in evidence_values for value in claimed_values):
            return {"sentence": sentence, "claim": claimed_values[0], "field": field,
                    "evidence_values": evidence_values, "reason": "explicit_contradiction"}

    # A cited policy clause that contains a different duration makes a wrong
    # duration a contradiction, not merely an unsupported number.
    claimed_durations = re.findall(r"[0-9]+(?:\.[0-9]+)?\s*(?:天|日|工作日|小时|分钟)", sentence)
    if claimed_durations:
        policy_rows = [row for row in rows if str(row.get("field", "")).startswith("policy.")]
        for duration in claimed_durations:
            if policy_rows and not _fact_supported(duration, policy_rows):
                evidence_durations = []
                for row in policy_rows:
                    evidence_durations.extend(re.findall(
                        r"[0-9]+(?:\.[0-9]+)?\s*(?:天|日|工作日|小时|分钟)", str(row.get("text", ""))))
                if evidence_durations:
                    return {"sentence": sentence, "claim": duration, "field": "policy.text",
                            "evidence_values": evidence_durations, "reason": "explicit_contradiction"}
    return None


def _value_mentioned(value: Any, answer: str) -> bool:
    rendered = _text(value).lower()
    lowered = answer.lower()
    if rendered and rendered in lowered:
        return True
    aliases = {
        "delivered": ("已送达", "已交付", "已签收"),
        "pending": ("待处理", "待付款"),
        "processed": ("处理中", "已处理"),
        "cancelled": ("已取消",),
        "requested": ("已提交退货", "退货申请已提交", "已申请退货"),
        "true": ("符合退货条件", "可以退货", "可退货"),
        "false": ("不符合退货条件", "不能退货", "不可退货"),
    }
    return any(alias in lowered for alias in aliases.get(rendered, ()))


def _row_supports_sentence(sentence: str, row: dict[str, Any]) -> bool:
    clean = EVIDENCE_CITATION.sub("", sentence)
    if _value_mentioned(row.get("value"), clean):
        return True
    facts = _facts(clean)
    if facts and all(_fact_supported(fact, [row]) for fact in facts):
        return True
    # Deterministic lexical support for non-numeric policy/product prose. Four
    # contiguous CJK characters are specific enough to reject a naked citation
    # while still permitting short extractive paraphrases such as “电子发票”.
    evidence_text = re.sub(r"\s+", "", str(row.get("text", "")).lower())
    for sequence in re.findall(r"[\u4e00-\u9fff]{4,}", clean):
        for index in range(len(sequence) - 3):
            if sequence[index:index + 4] in evidence_text:
                return True
    return False


def verify_answer(
    answer: str,
    ledger: list[dict[str, Any]],
    *,
    expectations: dict[str, Any] | None = None,
    require_citations: bool = False,
) -> dict[str, Any]:
    """Verify high-risk claims using exact, deterministic evidence checks.

    This deliberately does not judge tone, persuasiveness, or generic prose.
    Unknown prose remains outside the hard gate instead of being guessed at with
    keywords or embedding similarity.
    """
    expectations = expectations or {}
    by_id = {str(row.get("evidence_id")): row for row in ledger}
    invalid_citations: list[str] = []
    unsupported: list[dict[str, Any]] = []
    contradicted: list[dict[str, Any]] = []
    citation_failures: list[dict[str, Any]] = []
    cited_any = False

    for sentence in _sentences(answer):
        citation_ids = [f"E{x}" for x in EVIDENCE_CITATION.findall(sentence)]
        cited_any = cited_any or bool(citation_ids)
        missing_ids = [eid for eid in citation_ids if eid not in by_id]
        invalid_citations.extend(missing_ids)
        cited_rows = [by_id[eid] for eid in citation_ids if eid in by_id]
        facts = _facts(sentence)
        # Factual correctness is measured against the full ledger. Whether the
        # selected citation supports that otherwise-correct fact is a separate
        # binding metric below.
        contradiction = _explicit_contradiction(sentence, ledger)
        if contradiction:
            contradicted.append(contradiction)
        for fact in facts:
            if not _fact_supported(fact, ledger):
                unsupported.append({"sentence": sentence, "claim": fact, "reason": "not_in_evidence"})
            if citation_ids and not _fact_supported(fact, cited_rows):
                citation_failures.append({"sentence": sentence, "claim": fact,
                                          "evidence_ids": citation_ids, "reason": "citation_does_not_support_claim"})
        if citation_ids and not facts and cited_rows and not any(
                _row_supports_sentence(sentence, row) for row in cited_rows):
            citation_failures.append({"sentence": sentence, "claim": None,
                                      "evidence_ids": citation_ids, "reason": "citation_has_no_lexical_support"})

    required = list(expectations.get("required_fact_keys", []))
    omitted: list[str] = []
    covered = 0
    for key in required:
        matching = [row for row in ledger if row.get("field") == key]
        key_covered = False
        for row in matching:
            if _value_mentioned(row.get("value"), answer):
                key_covered = True
                break
            eid = str(row.get("evidence_id"))
            if any(f"[{eid}]" in sentence and _row_supports_sentence(sentence, row)
                   for sentence in _sentences(answer)):
                key_covered = True
                break
        if key_covered:
            covered += 1
        else:
            omitted.append(key)

    forbidden_values = expectations.get("forbidden_values", {})
    for field, values in forbidden_values.items():
        for value in values:
            if _text(value).lower() in answer.lower():
                contradicted.append({"sentence": answer, "claim": _text(value), "field": field,
                                     "reason": "forbidden_expected_value"})

    coverage = covered / len(required) if required else None
    applicable = bool(ledger or required or forbidden_values)
    fact_pass = applicable and not unsupported and not contradicted and not omitted
    citation_ok = not invalid_citations and not citation_failures and (cited_any or not require_citations)
    return {
        "applicable": applicable,
        "answer_fact_pass": fact_pass,
        "citation_binding_pass": citation_ok,
        "required_evidence_coverage": coverage,
        "unsupported_high_risk_claims": unsupported,
        "contradicted_claims": contradicted,
        "omitted_required_facts": omitted,
        "invalid_citations": sorted(set(invalid_citations)),
        "citation_failures": citation_failures,
        "cited_evidence_ids": sorted(set(EVIDENCE_CITATION.findall(answer))),
        "require_citations": require_citations,
        # Diagnostic only in Phase A. It is intentionally excluded from
        # ``answer_fact_pass`` until calibrated against the dynamic data policy.
        "freshness": freshness_from_ledger(answer, ledger),
    }
