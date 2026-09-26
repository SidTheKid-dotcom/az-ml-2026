import os, time
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
from huggingface_hub import snapshot_download, model_info
t = time.time()
try:
    info = model_info("Qwen/Qwen3-Reranker-4B")
    print("sha", info.sha, "card license", (info.card_data or {}).get("license") if info.card_data else None, flush=True)
except Exception as e:
    print("model_info failed", repr(e), flush=True)
p = snapshot_download("Qwen/Qwen3-Reranker-4B", local_dir="/workspace/az-ml-new/models/Qwen3-Reranker-4B", max_workers=8)
print("done", p, round(time.time() - t, 1), "s", flush=True)
