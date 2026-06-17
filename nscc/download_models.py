# -*- coding: utf-8 -*-
"""Pre-fetch HuggingFace models on a node WITH internet (e.g. NSCC login node).

NSCC compute nodes are offline. Run this on the login node first so the models
land in ~/.cache/huggingface/hub; the PBS job then loads them with
HF_HUB_OFFLINE=1. Downloading is network I/O (not heavy compute), so it is safe
on the login node. Do NOT run model inference here — that belongs in a PBS job.

Usage (login node):
    ~/miniforge3/bin/conda run -n parkinson python nscc/download_models.py
"""

from huggingface_hub import snapshot_download

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # embedder
    "BAAI/bge-reranker-base",  # cross-encoder reranker (zh/en)
]

if __name__ == "__main__":
    for repo in MODELS:
        path = snapshot_download(repo)
        print(f"OK {repo} -> {path}")
