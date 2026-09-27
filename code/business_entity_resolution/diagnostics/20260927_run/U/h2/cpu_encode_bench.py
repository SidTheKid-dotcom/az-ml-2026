"""Benchmark CPU encoding of R's pre-tokenized train_pool texts with e5_small_ft200 (fp32 vs bf16), 28 threads.
Also compares bf16 / fp32 embeddings (cosine) on the same texts. Label-free.
"""
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

ROOT = Path("/home/ubuntu/run27/U")
torch.set_num_threads(28)
ids = np.load(ROOT / "rtokens/train_pool.ids.npy", mmap_mode="r")
off = np.load(ROOT / "rtokens/train_pool.off.npy")
N = 16384
rows = np.arange(0, 10_000_000, 10_000_000 // N)[:N]
lens = (off[rows + 1] - off[rows]).astype(np.int64)
print("mean tokens", lens.mean())
model = AutoModel.from_pretrained(str(ROOT / "rmodel_full"), attn_implementation="sdpa").eval()


def enc(dtype, bs=256):
    order = np.argsort(lens)
    out = np.zeros((N, 384), np.float32)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=(dtype == "bf16")):
        for s in range(0, N, bs):
            r = order[s:s + bs]
            L = int(lens[r].max())
            batch = np.ones((len(r), L), np.int64)
            mask = np.zeros((len(r), L), np.int64)
            for j, i in enumerate(r):
                a, b = off[rows[i]], off[rows[i] + 1]
                batch[j, :b - a] = ids[a:b]
                mask[j, :b - a] = 1
            h = model(input_ids=torch.from_numpy(batch), attention_mask=torch.from_numpy(mask)).last_hidden_state.float()
            m = torch.from_numpy(mask).unsqueeze(-1).float()
            p = (h * m).sum(1) / m.sum(1)
            out[r] = torch.nn.functional.normalize(p, dim=1).numpy()
    return out, N / (time.time() - t0)


e32, s32 = enc("fp32")
print(f"fp32: {s32:.0f} texts/s")
e16, s16 = enc("bf16")
print(f"bf16: {s16:.0f} texts/s; mean cos(bf16, fp32) {float((e32 * e16).sum(1).mean()):.6f}, "
      f"min {float((e32 * e16).sum(1).min()):.6f}")
