"""Build the local Dense + BM25 + RRF retrieval index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ecommerce_rag import config
from ecommerce_rag.retrieval_index import (
    build_chunks,
    build_index,
    load_policies,
    load_products,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--products", type=Path, default=config.PRODUCT_DATA_PATH)
    parser.add_argument("--policies", type=Path, default=config.POLICY_DATA_PATH)
    parser.add_argument("--embed-model", default=config.EMBED_MODEL)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    products = load_products(args.products)
    policies = load_policies(args.policies)
    chunks, parents = build_chunks(products, policies)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "products": len(products),
                    "policies": len(policies),
                    "chunks": len(chunks),
                    "parents": len(parents),
                    "source_types": dict(Counter(chunk["source_type"] for chunk in chunks)),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(
        json.dumps(
            build_index(
                args.output_dir,
                args.products,
                args.policies,
                args.embed_model,
                args.batch_size,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
