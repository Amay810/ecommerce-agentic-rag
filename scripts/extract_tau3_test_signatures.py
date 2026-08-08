# -*- coding: utf-8 -*-
"""Extract frozen structural signatures for τ³ Retail test 40."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ecommerce_rag.retail_task_compiler.constants import TAU2_COMMIT
from ecommerce_rag.retail_task_compiler.contamination import extract_test_signatures
from ecommerce_rag.tau3_retail_v1 import validate_tau2_checkout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau-root",
        type=Path,
        default=Path(r"E:\cv_codex\external\tau2-bench"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/tau3_retail_test40_structure_signatures.json"),
    )
    args = parser.parse_args(argv)

    validate_tau2_checkout(args.tau_root)
    retail = args.tau_root / "data" / "tau2" / "domains" / "retail"
    tasks = json.loads((retail / "tasks.json").read_text(encoding="utf-8"))
    splits = json.loads((retail / "split_tasks.json").read_text(encoding="utf-8"))
    payload = extract_test_signatures(
        tasks=tasks,
        test_ids=splits["test"],
        tau2_commit=TAU2_COMMIT,
    )
    if payload["count"] != 40:
        raise SystemExit(f"expected 40 test signatures, got {payload['count']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({payload['count']} signatures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
