# -*- coding: utf-8 -*-
"""SupportCase: the auditable, persistable record of a single customer-support turn.

Each call to the agent produces a SupportCase instead of a throwaway dict: query,
routed intent, cited evidence, verification results (grounding / citation / consistency),
a confidence score, and a versioned product/policy snapshot. This is the foundation of
the memory flywheel (failure writeback, KB-gap mining, case reuse).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import config

# verdicts that count as a verification problem (mirror verifier.consistency_check)
_BAD_VERDICTS = {"矛盾", "资料外"}


def build_snapshot(chunks: list[dict]) -> dict:
    """Versioned product/policy snapshot, deduped by doc_id.

    `version` is reserved (None) for future use; `default_updated_at` is carried from the
    chunk's `updated_at` (set by data_loader from the source data) so the freshness
    guardrail can check staleness without a schema change.
    """
    products: dict[str, dict] = {}
    policies: dict[str, dict] = {}
    for c in chunks:
        doc_id = c.get("doc_id")
        if c.get("source_type") == "policy":
            policies.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "policy_type": c.get("category"),
                    "version": None,
                    "default_updated_at": c.get("updated_at"),
                },
            )
        else:
            products.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": c.get("title"),
                    "price": c.get("price"),
                    "inventory": c.get("inventory"),
                    "version": None,
                    "default_updated_at": c.get("updated_at"),
                },
            )
    return {"products": list(products.values()), "policies": list(policies.values())}


def make_case_id(query: str, trace: list[str], ts: str) -> str:
    """Stable, traceable id: sc_{utc_compact_ts}_{hash10}.

    The hash is over query + trace so the same turn maps to the same id across the
    SQLite store, the JSONL mirror and any export.
    """
    compact = ts.translate(str.maketrans("", "", "-:.")).replace("+0000", "").replace("Z", "")
    digest = hashlib.sha1((query + "".join(trace)).encode("utf-8")).hexdigest()[:10]
    return f"sc_{compact}_{digest}"


@dataclass
class SupportCase:
    case_id: str
    ts: str
    query: str
    intent: str
    action: str
    evidence: list[dict] = field(default_factory=list)
    grounding_ratio: float | None = None
    citation_ok: bool | None = None
    consistency_verdict: str | None = None
    confidence: float = 0.0
    snapshot: dict = field(default_factory=lambda: {"products": [], "policies": []})
    answer: str | None = None
    trace: list[str] = field(default_factory=list)
    needs_review: bool = False
    freshness: dict | None = None  # freshness guardrail verdict (Step 3); None if not assessed

    # columns that stay scalar in SQLite; the rest are JSON-encoded by to_row()
    _JSON_FIELDS = ("evidence", "snapshot", "trace", "freshness")

    @staticmethod
    def compute_needs_review(
        action: str,
        grounding_ratio: float | None,
        citation_ok: bool | None,
        consistency_verdict: str | None,
    ) -> bool:
        if action in ("handoff", "caution"):
            return True
        if grounding_ratio is not None and grounding_ratio < config.GROUNDING_MIN_RATIO:
            return True
        if citation_ok is False:
            return True
        if consistency_verdict in _BAD_VERDICTS:
            return True
        return False

    @classmethod
    def from_agent_result(cls, result: dict) -> "SupportCase":
        ts = datetime.now(timezone.utc).isoformat()
        query = result.get("query", "")
        trace = result.get("trace", []) or []
        chunks = result.get("chunks", []) or []

        # evidence + citation index (first-seen doc_id order mirrors retriever.format_context)
        doc_citation: dict[str, int] = {}
        evidence = []
        for c in chunks:
            doc_id = c.get("doc_id")
            if doc_id not in doc_citation:
                doc_citation[doc_id] = len(doc_citation) + 1
            evidence.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": doc_id,
                    "source_type": c.get("source_type"),
                    "title": c.get("title"),
                    "score": c.get("score"),
                    "dense_sim": c.get("dense_sim"),
                    "citation_index": doc_citation[doc_id],
                }
            )

        confidence = max((c.get("dense_sim") or 0.0 for c in chunks), default=0.0)

        grounding = result.get("grounding") or {}
        grounding_ratio = grounding.get("ratio") if grounding else None
        citations = result.get("citations") or {}
        citation_ok = citations.get("ok") if citations else None
        consistency = result.get("consistency") or {}
        consistency_verdict = consistency.get("verdict") if consistency else None
        action = result.get("action", "")

        return cls(
            case_id=make_case_id(query, trace, ts),
            ts=ts,
            query=query,
            intent=result.get("intent", ""),
            action=action,
            evidence=evidence,
            grounding_ratio=grounding_ratio,
            citation_ok=citation_ok,
            consistency_verdict=consistency_verdict,
            confidence=float(confidence),
            snapshot=build_snapshot(chunks),
            answer=result.get("answer"),
            trace=list(trace),
            needs_review=cls.compute_needs_review(
                action, grounding_ratio, citation_ok, consistency_verdict
            ),
            freshness=result.get("freshness"),
        )

    def to_row(self) -> dict:
        """Flat dict for SQLite: list/dict fields are JSON-encoded into *_json columns."""
        data = asdict(self)
        row = {k: v for k, v in data.items() if k not in self._JSON_FIELDS}
        row["needs_review"] = int(self.needs_review)
        for f in self._JSON_FIELDS:
            row[f"{f}_json"] = json.dumps(data[f], ensure_ascii=False)
        return row

    @classmethod
    def column_names(cls) -> list[str]:
        scalar = [f for f in cls.__dataclass_fields__ if f not in cls._JSON_FIELDS]
        return scalar + [f"{f}_json" for f in cls._JSON_FIELDS]
