# -*- coding: utf-8 -*-
"""Compile and gate the cancel_pending v0 blueprint slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecommerce_rag.retail_task_compiler import (
    RetailTaskCompiler,
    coverage_from_blueprints,
)
from ecommerce_rag.retail_task_compiler.blueprint import canonical_hash
from ecommerce_rag.retail_task_compiler.contamination import load_test_signatures


def _demo_entities() -> list[dict]:
    """Synthetic entities for offline compiler smoke; not τ³ DB rows."""
    return [
        {
            "user": {
                "user_id": "demo_user_001",
                "email": "demo_user_001@example.com",
                "first_name": "Demo",
                "last_name": "User",
                "zip": "10001",
            },
            "order": {"order_id": "#W9000001", "status": "pending"},
            "auth_mode": "email",
            "reason": "no longer needed",
        },
        {
            "user": {
                "user_id": "demo_user_002",
                "email": "demo_user_002@example.com",
                "first_name": "Casey",
                "last_name": "Lee",
                "zip": "94107",
            },
            "order": {"order_id": "#W9000002", "status": "pending"},
            "auth_mode": "name_zip",
            "reason": "ordered by mistake",
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signatures",
        type=Path,
        default=Path("docs/tau3_retail_test40_structure_signatures.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/retail_task_compiler_v0_cancel_pending_report.json"),
    )
    args = parser.parse_args(argv)

    signatures = load_test_signatures(args.signatures)
    compiler = RetailTaskCompiler(test_signatures=signatures)
    db_snapshot_hash = canonical_hash({"kind": "demo_offline_db", "n": 2})

    results = []
    accepted = []
    for index, entity in enumerate(_demo_entities()):
        result = compiler.compile_cancel_pending(
            task_id=f"rtc_v0_cancel_{index:03d}",
            user=entity["user"],
            order=entity["order"],
            auth_mode=entity["auth_mode"],
            reason=entity["reason"],
            db_snapshot_hash=db_snapshot_hash,
        )
        results.append(result.to_dict())
        if result.accepted:
            accepted.append(result.blueprint)

    coverage = coverage_from_blueprints(accepted)
    report = {
        "generator_version": "retail_task_compiler.v0.cancel_pending",
        "attempted": len(results),
        "accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "contamination_count": sum(
            1 for item in results if item["contamination"]["contaminated"]
        ),
        "coverage": coverage.to_dict(),
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.output}: accepted={report['accepted']} "
        f"rejected={report['rejected']} contamination={report['contamination_count']}"
    )
    return 0 if report["rejected"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
