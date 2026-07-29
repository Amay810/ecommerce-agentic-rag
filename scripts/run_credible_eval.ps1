$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m scripts.generate_hidden_tasks --db ecommerce_rag/data/agent_env_v2.db --output ecommerce_rag/data/harness_tasks_v2.jsonl
python -m scripts.generate_retrieval_eval_v2 --products ecommerce_rag/data/amazon_products_5k.jsonl --output ecommerce_rag/data/retrieval_eval_v2_300.jsonl
python -m ecommerce_rag.harness run --tasks ecommerce_rag/data/harness_tasks_v2.jsonl --db ecommerce_rag/data/agent_env_v2.db --store logs/harness_v2_rule.sqlite --index ecommerce_rag/index_5000 --policy rule --split locked --repeats 3 --output docs/harness_v2_rule_locked_pass3.json
python -m ecommerce_rag.benchmark --testset ecommerce_rag/data/retrieval_eval_v2_300.jsonl --index ecommerce_rag/index_5000 --split locked --constraints --output docs/retrieval_v2_locked_raw.json
python -m ecommerce_rag.rl_gate --tasks ecommerce_rag/data/harness_tasks_v2.jsonl --store logs/harness_v2_rule.sqlite --output docs/agent_rl_gate_v2.json
python -m unittest discover -s tests -v
