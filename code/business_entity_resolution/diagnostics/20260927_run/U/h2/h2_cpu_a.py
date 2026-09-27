"""HOLDOUT-2 step A on CPU (label-free), used only if no GPU arrives: same outputs as h2_gpu.py (gpu_out layout).

    python h2_cpu_a.py encode SHARD NSHARDS THREADS   # bf16 (AMX) mean-pooled e5_small_ft200 embeddings of the raw pool
    python h2_cpu_a.py search THREADS                 # forward top-50 for HOLDOUT-2 S1s; joint reverse top-20

Encoding: R's pre-tokenized texts (same tokenizer/truncation), mean pooling + L2 in float32, stored float16
(bf16 vs fp32 cosine 0.99995 measured on 16k texts). Forward: float32 dot, top-50 per HOLDOUT-2 S1 (same country).
Reverse: R's stored top-20 arena S1s per record (GPU run) merged with the HOLDOUT-2 S1s whose cosine to the record
beats the 20th arena cosine -> joint top-20 (joint S1 index: arena rows then HOLDOUT-2 rows), as h2_gpu.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/run27/U")
OUT = ROOT / "h2" / "cpu_a"
N_ARENA = 2_072_821
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def encode_rows(model, ids, off, rows, torch, max_tokens=24576):
    lens = (off[rows + 1] - off[rows]).astype(np.int64)
    order = np.argsort(lens, kind="stable")
    out = np.zeros((len(rows), 384), np.float16)
    i, done, last = 0, 0, time.time()
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        while i < len(order):
            L0 = int(lens[order[i]])
            bs = max(1, min(1024, max_tokens // max(L0, 1)))
            sel = order[i:i + bs]
            L = int(lens[sel].max())
            starts = off[rows[sel]]
            col = np.arange(L)
            valid = col[None, :] < lens[sel][:, None]
            batch = np.where(valid, ids[np.where(valid, starts[:, None] + col[None, :], 0)], 1).astype(np.int64)
            h = model(input_ids=torch.from_numpy(batch), attention_mask=torch.from_numpy(valid.astype(np.int64)))
            h = h.last_hidden_state.float()
            m = torch.from_numpy(valid).unsqueeze(-1).float()
            p = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)
            out[sel] = p.numpy().astype(np.float16)
            i += len(sel)
            done += len(sel)
            if time.time() - last > 60:
                last = time.time()
                log(f"  {done:,}/{len(rows):,} ({done / (last - T0):,.0f}/s)")
    return out


def encode(shard, nshards, threads):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(threads)
    OUT.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(str(ROOT / "rmodel_full"), attn_implementation="sdpa").eval()
    if shard == -1:  # HOLDOUT-2 S1s + 2,000 arena S1s for the fidelity check
        t = ROOT / "h2" / "tokens"
        ids, off = np.load(t / "h2_s1.ids.npy", mmap_mode="r"), np.load(t / "h2_s1.off.npy")
        chk = np.arange(0, N_ARENA, N_ARENA // 2000)[:2000]
        rows = np.r_[np.arange(N_ARENA, len(off) - 1), chk]
        np.save(OUT / "s1_rows.npy", rows)
        np.save(OUT / "s1_emb.npy", encode_rows(model, ids, off, rows, torch))
        log("S1 embeddings saved")
        return
    ids = np.load(ROOT / "rtokens/train_pool.ids.npy", mmap_mode="r")
    off = np.load(ROOT / "rtokens/train_pool.off.npy")
    n = len(off) - 1
    rows = np.arange(shard, n, nshards)
    np.save(OUT / f"pool_emb_{shard}.npy", encode_rows(model, ids, off, rows, torch))
    log(f"pool shard {shard} saved ({len(rows)} rows)")


def search(threads):
    import torch
    torch.set_num_threads(threads)
    pcc = np.load(ROOT / "rtokens/train_pool.cc.npy")
    n_pool = len(pcc)
    shards = sorted(OUT.glob("pool_emb_*.npy"))
    ns = len(shards)
    pool = np.zeros((n_pool, 384), np.float16)
    for f in shards:
        k = int(f.stem.split("_")[-1])
        pool[np.arange(k, n_pool, ns)] = np.load(f)
    s1_rows, s1_emb = np.load(OUT / "s1_rows.npy"), np.load(OUT / "s1_emb.npy").astype(np.float32)
    scc = np.load(ROOT / "h2/tokens/h2_s1.cc.npy")
    h_mask = s1_rows >= N_ARENA
    h_rows, h_emb = s1_rows[h_mask] - N_ARENA, s1_emb[h_mask]
    c_rows, c_emb = s1_rows[~h_mask], s1_emb[~h_mask]
    n_h2 = int(h_mask.sum())
    g = ROOT / "h2" / "gpu_out_cpu"
    g.mkdir(parents=True, exist_ok=True)
    fwd_idx = np.full((n_h2, 50), -1, np.int32)
    fwd_score = np.full((n_h2, 50), np.nan, np.float32)
    chk_idx = np.full((len(c_rows), 50), -1, np.int32)
    rev_idx_a = np.load(ROOT / "rsrc/gpu/train_rev_idx.npy")
    rev_sc_a = np.load(ROOT / "rsrc/gpu/train_rev_score.npy").astype(np.float32)
    rev_idx = rev_idx_a.copy()
    rev_score = rev_sc_a.copy()
    for code in (0, 1):
        p_rows = np.flatnonzero(pcc == code)
        P = torch.from_numpy(pool[p_rows].astype(np.float32))
        hq = np.flatnonzero(scc[N_ARENA + h_rows] == code)
        cq = np.flatnonzero(scc[c_rows] == code)
        for qidx, emb, di, ds in ((hq, h_emb, fwd_idx, fwd_score), (cq, c_emb, chk_idx, None)):
            for s in range(0, len(qidx), 128):
                q = torch.from_numpy(emb[qidx[s:s + 128]])
                v, i = torch.topk(q @ P.T, 50, dim=1)
                di[qidx[s:s + 128]] = p_rows[i.numpy()]
                if ds is not None:
                    ds[qidx[s:s + 128]] = v.numpy()
        log(f"country {code}: forward done")
        # reverse: HOLDOUT-2 S1s that enter each record's joint top-20
        H = torch.from_numpy(h_emb[hq])
        kth = rev_sc_a[p_rows, -1]
        for s in range(0, len(p_rows), 20000):
            pr = p_rows[s:s + 20000]
            sim = (P[s:s + 20000] @ H.T).numpy()
            thr = np.nan_to_num(kth[s:s + 20000], nan=-1.0)
            hit_r, hit_c = np.nonzero(sim > thr[:, None])
            if len(hit_r) == 0:
                continue
            bounds = np.flatnonzero(np.r_[True, hit_r[1:] != hit_r[:-1], True])
            for a0, b0 in zip(bounds[:-1], bounds[1:]):
                r = hit_r[a0]
                sel = hit_c[a0:b0]
                rec = pr[r]
                cand_s = np.r_[rev_sc_a[rec], sim[r, sel]]
                cand_i = np.r_[rev_idx_a[rec].astype(np.int64), N_ARENA + h_rows[hq[sel]]]
                o = np.argsort(-cand_s, kind="stable")[:20]
                rev_idx[rec] = cand_i[o].astype(np.int32)
                rev_score[rec] = cand_s[o]
        log(f"country {code}: reverse merge done")
        del P, H
    np.save(g / "h2_fwd_idx.npy", fwd_idx)
    np.save(g / "h2_fwd_score.npy", fwd_score)
    np.save(g / "h2_rev_idx.npy", rev_idx)
    np.save(g / "h2_rev_score.npy", rev_score.astype(np.float16))
    np.save(g / "check_arena_rows.npy", c_rows)
    np.save(g / "check_arena_fwd_idx.npy", chk_idx)
    (g / "gpu_report.json").write_text(json.dumps({"mode": "CPU fallback (bf16 AMX encode, fp32 dot)",
                                                   "h2_s1": n_h2, "pool": int(n_pool),
                                                   "reverse": "R's GPU arena top-20 merged with HOLDOUT-2 cosines",
                                                   "labels_read": False}, indent=1) + "\n")
    log("CPU step A outputs saved")


if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        search(int(sys.argv[2]))
