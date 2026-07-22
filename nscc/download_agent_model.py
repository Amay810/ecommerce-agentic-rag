"""Cache the local next-action model on an internet-enabled NSCC login node."""

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    path = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
    print(path)
