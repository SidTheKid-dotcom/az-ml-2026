"""Agent J: download Qwen/Qwen3-Reranker-4B at a pinned revision into DEST (argv[1])."""
import os, sys, time
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
from huggingface_hub import snapshot_download
REV = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
t = time.time()
p = snapshot_download("Qwen/Qwen3-Reranker-4B", revision=REV, local_dir=sys.argv[1], max_workers=8)
print("done", p, round(time.time() - t, 1), "s", flush=True)
