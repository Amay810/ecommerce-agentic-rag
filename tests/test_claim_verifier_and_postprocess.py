from __future__ import annotations

from ecommerce_rag.answer_postprocess import AnswerPostprocessor
from ecommerce_rag.claim_verifier import classify_claim
from ecommerce_rag.llm_policy import Generation
from scripts.prepare_verifier_calibration import HOLDOUT_IDS, SMOKE_IDS, challenge_rows


def ledger() -> list[dict]:
    return [
        {"evidence_id": "E1", "source_id": "product:P00001", "field": "product.product_id",
         "value": "P00001", "text": "P00001"},
        {"evidence_id": "E2", "source_id": "product:P00001", "field": "product.price",
         "value": 99.0, "text": "99.0"},
        {"evidence_id": "E3", "source_id": "order:O000001", "field": "order.status",
         "value": "delivered", "text": "已送达"},
    ]


def test_claim_statuses_keep_unsupported_separate_from_contradiction() -> None:
    assert classify_claim("P00001 的价格为99元。[E2]", ledger(), citation_required=True).fact_status == "supported"
    wrong = classify_claim("P00001 的价格为199元。", ledger())
    assert wrong.fact_status == "contradicted" and wrong.hard_failure
    absent = classify_claim("P00001 的电池容量为9000 mAh。", ledger())
    assert absent.fact_status == "unsupported" and not absent.hard_failure


def test_unknown_and_not_factual_never_hard_fail() -> None:
    unknown = classify_claim("该设计更适合日常使用。", ledger())
    assert unknown.fact_status in {"unknown", "not_factual"}
    assert not unknown.hard_failure


def test_user_budget_is_not_misread_as_product_price() -> None:
    result = classify_claim(
        "我找到一个不高于101元的产品，价格为99元。", ledger(),
        user_messages=[{"role": "user", "content": "预算不高于101元"}],
    )
    assert result.fact_status == "supported"


def test_delivered_evidence_opposes_undelivered_claim() -> None:
    result = classify_claim("订单 O000001 已超过37天未送达。", ledger())
    assert result.fact_status == "contradicted"


def test_wrong_citation_is_separate_from_supported_fact() -> None:
    result = classify_claim("P00001 的价格为99元。[E3]", ledger(), citation_required=True)
    assert result.fact_status == "supported"
    assert result.citation_status == "incorrect"


def test_shadow_never_changes_answer() -> None:
    result = AnswerPostprocessor().process("价格为99元。[E2]", ledger(), [], "shadow")
    assert result.final_answer == result.draft_answer
    assert not result.changed and result.raw_output is None


def test_terminal_grounded_records_fixed_generation_metadata() -> None:
    def generate(system: str, user: str) -> Generation:
        assert "DRAFT ANSWER" in user and "EVIDENCE LEDGER" in user
        return Generation("P00001 的价格为99元。[E2]", "stop", 20, 10, False)

    processor = AnswerPostprocessor(generate, generation_config={"do_sample": False, "temperature": 0})
    result = processor.process("旧答案", ledger(), [], "terminal_grounded")
    assert result.raw_output == result.final_answer
    assert result.generation_config_hash and not result.error and not result.truncated


def test_handoff_is_ineligible_and_must_pass_through() -> None:
    trajectory = {"actions": [{"action_type": "handoff"}], "final_answer": "已转人工", "evidence_ledger": ledger()}
    assert AnswerPostprocessor.eligibility(trajectory, {"answer_fact_applicable": True}) == "base_handoff"


def test_frozen_holdout_and_smoke_are_disjoint() -> None:
    assert len(HOLDOUT_IDS) == len(set(HOLDOUT_IDS)) == 16
    assert len(SMOKE_IDS) == len(set(SMOKE_IDS)) == 12
    assert not set(HOLDOUT_IDS) & set(SMOKE_IDS)


def test_challenge_shape_and_review_status() -> None:
    rows = challenge_rows()
    assert len(rows) == len({row["challenge_id"] for row in rows}) == 150
    assert all(row["human_review_status"] == "assistant_prefilled_pending_user_confirmation" for row in rows)
    assert sum(row["gold_fact_status"] == "unsupported" for row in rows) == 25
