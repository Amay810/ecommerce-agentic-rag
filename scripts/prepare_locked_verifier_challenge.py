"""Create locked_v2 inputs without invoking the verifier.

Run this only after the verifier-v2 code commit exists.  The generated human
labels are assistant-prefilled review aids and remain non-frozen until the user
confirms all 150 rows and ``freeze_verifier_challenge`` records their hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from ecommerce_rag.claim_verifier import verifier_config_hash


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_SOURCE = ROOT / "ecommerce_rag" / "claim_verifier.py"
EVALUATOR_SOURCE = ROOT / "scripts" / "evaluate_verifier_challenge.py"
FAMILIES = ("numeric_unit", "negation_polarity", "entity_binding", "budget_range",
            "state_date", "citation_binding")
SEED = 20260817


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip()


def evidence(eid: str, source: str, field: str, value: Any, text: str | None = None) -> dict[str, Any]:
    return {
        "evidence_id": eid, "source_id": source, "tool_call_id": "locked-v2-call",
        "tool_name": "locked_challenge_fixture", "field": field, "value": value,
        "text": str(value) if text is None else text,
    }


def case_kind(index: int) -> str:
    if index <= 15:
        return "target_error"
    if index <= 18:
        return "supported_original"
    if index <= 21:
        return "supported_paraphrase"
    if index <= 23:
        return "equivalent_unit_control"
    return "multi_entity_number_distractor_control"


def locked_rows() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(1, 26):
            target = index <= 15
            product_number = 20000 + family_index * 200 + index
            product = f"P{product_number:05d}"
            other = f"P{product_number + 90:05d}"
            price = 230 + family_index * 37 + index * 3
            ledger = [
                evidence("E1", f"product:{product}", "product.product_id", product),
                evidence("E2", f"product:{product}", "product.price", price),
            ]
            messages: list[dict[str, str]] = []
            citation_required = family == "citation_binding"
            citation = "not_required"
            if family == "numeric_unit":
                if target and index <= 10:
                    claim, fact = f"{product} 的价格是 {price + 13} 元。", "contradicted"
                elif target:
                    claim, fact = f"{product} 的重量是 {700 + index} 克。", "unsupported"
                elif index in {22, 23}:
                    claim, fact = f"{product} 售价 {price * 100} 分。[E2]", "supported"
                    citation = "correct"
                else:
                    claim, fact = f"{product} 的售价为 {price} 元。[E2]", "supported"
                    citation = "correct"
            elif family == "negation_polarity":
                ledger.append(evidence("E3", f"product:{product}", "product.discontinued", False, "未停产"))
                if target and index <= 10:
                    claim, fact = f"{product} 已经停产。", "contradicted"
                elif target:
                    claim, fact = f"{product} 支持卫星通信。", "unsupported"
                else:
                    claim, fact = (f"{product} 目前仍在售。[E3]" if index >= 19 else f"{product} 未停产。[E3]"), "supported"
                    citation = "correct"
            elif family == "entity_binding":
                ledger.extend([
                    evidence("E3", f"product:{other}", "product.product_id", other),
                    evidence("E4", f"product:{other}", "product.price", price + 80),
                ])
                if target and index <= 10:
                    claim, fact, citation = f"{product} 的价格是 {price + 80} 元。[E4]", "contradicted", "incorrect"
                elif target:
                    claim, fact = f"{product} 的屏幕刷新率是 {120 + index}Hz。", "unsupported"
                else:
                    distraction = f"；{other} 的价格是 {price + 80} 元[E4]" if index >= 24 else ""
                    claim, fact, citation = f"{product} 的价格是 {price} 元[E2]{distraction}。", "supported", "correct"
            elif family == "budget_range":
                budget = price + 35
                messages = [{"role": "user", "content": f"预算上限是 {budget} 元，请不要超出。"}]
                if target and index <= 10:
                    claim, fact = f"{product} 售价 {price} 元，高于 {budget} 元预算。[E2]", "contradicted"
                elif target:
                    claim, fact = f"{product} 的会员价是 {price - 20} 元。", "unsupported"
                else:
                    wording = "在预算以内" if index >= 19 else f"不超过 {budget} 元预算"
                    claim, fact = f"{product} 售价 {price} 元，{wording}。[E2]", "supported"
                citation = "correct" if "[E2]" in claim else "not_required"
            elif family == "state_date":
                order = f"O{700000 + family_index * 100 + index:06d}"
                day = 3 + index % 20
                date = f"2026-06-{day:02d}"
                ledger = [
                    evidence("E1", f"order:{order}", "order.order_id", order),
                    evidence("E2", f"order:{order}", "order.status", "delivered", "已送达"),
                    evidence("E3", f"order:{order}", "order.delivered_at", date),
                ]
                if target and index <= 10:
                    claim, fact, citation = f"订单 {order} 尚未送达，送达日期为2026-07-{day:02d}。[E2][E3]", "contradicted", "incorrect"
                elif target:
                    claim, fact = f"订单 {order} 的退款已到账。", "unsupported"
                else:
                    wording = date if index < 22 else f"2026年6月{day}日"
                    claim, fact, citation = f"订单 {order} 已送达，日期为{wording}。[E2][E3]", "supported", "correct"
            else:
                ledger.extend([
                    evidence("E3", f"product:{other}", "product.product_id", other),
                    evidence("E4", f"product:{other}", "product.price", price + 80),
                ])
                if target:
                    claim, fact, citation = f"{product} 的价格是 {price} 元。[E4]", "supported", "incorrect"
                else:
                    suffix = f" 另一个商品 {other} 售价 {price + 80} 元[E4]。" if index >= 24 else ""
                    claim, fact, citation = f"{product} 的价格是 {price} 元。[E2]{suffix}", "supported", "correct"
            # Stable random nonce makes the locked text/IDs distinct without affecting labels.
            nonce = rng.randrange(100000, 999999)
            rows.append({
                "challenge_id": f"vc2_{family}_{index:02d}_{nonce}", "family": family,
                "case_kind": case_kind(index), "claim_text": claim,
                "evidence_ledger": ledger, "user_messages": messages,
                "citation_required": citation_required,
                "gold_fact_status": fact, "gold_citation_status": citation,
                "human_fact_status": fact, "human_citation_status": citation,
                "human_review_status": "assistant_prefilled_pending_user_confirmation",
                "review_notes": "待用户独立核对；确认前禁止运行 verifier。",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("locked_v2 generation requires a clean verifier-v2 worktree")
    verifier_commit = source_commit(VERIFIER_SOURCE)
    evaluator_commit = source_commit(EVALUATOR_SOURCE)
    rows = locked_rows()
    output = args.output_dir / "verifier_challenge_locked_v2.jsonl"
    manifest_path = args.output_dir / "verifier_challenge_locked_v2_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite locked_v2 inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                      encoding="utf-8")
    manifest = {
        "schema_version": 2, "name": "verifier_challenge_locked_v2", "dataset_role": "locked",
        "seed": SEED, "row_count": 150,
        "families": {family: 25 for family in FAMILIES},
        "target_error_per_family": 15, "supported_control_per_family": 10,
        "control_types": ["supported_original", "supported_paraphrase", "equivalent_unit_control",
                          "multi_entity_number_distractor_control"],
        "jsonl_sha256": sha256(output), "labels_frozen": False,
        "verifier_invoked_during_generation": False,
        "verifier_code_commit": verifier_commit, "verifier_source_sha256": sha256(VERIFIER_SOURCE),
        "verifier_config_hash": verifier_config_hash(),
        "evaluator_code_commit": evaluator_commit, "evaluator_source_sha256": sha256(EVALUATOR_SOURCE),
        "reuse_policy": "one formal locked evaluation only; verifier changes require locked_v3",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({"rows": len(rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
