"""Prefill evidence for retrieval gold review without claiming human sign-off."""

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--products", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    products = {f"product:{row['id']}": row for row in read_jsonl(args.products)}
    benchmark = {row["id"]: row for row in read_jsonl(args.benchmark)}
    with args.audit.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    extra = ["assistant_gold_doc_ids", "assistant_is_answerable", "assistant_check",
             "assistant_evidence", "assistant_notes", "human_confirmed"]
    fields = list(rows[0]) + [name for name in extra if name not in rows[0]]
    for row in rows:
        spec = benchmark.get(row["id"], {})
        proposed = [value.strip() for value in row.get("proposed_gold_doc_ids", "").split("|") if value.strip()]
        evidence, checks = [], []
        for doc_id in proposed:
            product = products.get(doc_id)
            if not product:
                checks.append(False)
                evidence.append(f"{doc_id}: MISSING FROM CORPUS")
                continue
            ceiling = (spec.get("constraints") or {}).get("max_price")
            price_ok = ceiling is None or (product.get("price") is not None and float(product["price"]) <= float(ceiling))
            checks.append(price_ok)
            evidence.append(
                f"{doc_id}: title={product.get('title')}; price={product.get('price')}; "
                f"brand={product.get('attributes', {}).get('brand_or_store', '')}; category={product.get('category', '')}; "
                f"budget_ok={price_ok}"
            )
        answerable = bool(proposed)
        row.update({
            "assistant_gold_doc_ids": "|".join(proposed),
            "assistant_is_answerable": str(answerable).lower(),
            "assistant_check": "supported" if checks and all(checks) else ("no_answer_candidate" if not proposed else "needs_correction"),
            "assistant_evidence": " || ".join(evidence),
            "assistant_notes": "Codex-assisted prefill only; confirm alternative matching SKUs and sign human_confirmed.",
            "human_confirmed": "",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "supported": sum(row["assistant_check"] == "supported" for row in rows),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
