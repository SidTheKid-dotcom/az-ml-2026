"""HOLDOUT-2 final check, GPU step A (label-free): joint dense retrieval with the frozen encoder (e5_small_ft200).

Uses Agent R's `reverse_retrieval` encoder/search functions. S1 set = the 2,072,821 arena S1s followed by the 50,000
HOLDOUT-2 S1s (tokens/h2_s1.*); record set = the full raw train pool (R's train_pool tokens). Per country:
  * forward top-50 records for the HOLDOUT-2 S1s only, scored as P2 did (float32 dot of float16 unit vectors),
    with every cosine kept (v1's retrieval_score / rank features need them);
  * reverse top-20 S1s (arena + HOLDOUT-2) for every raw record (R's reverse rule).
Writes OUT/h2_fwd_idx.npy (50000x50 raw pool row), h2_fwd_score.npy (float32), h2_rev_idx.npy (n_pool x 20 joint S1
row), h2_rev_score.npy (float16), gpu_report.json.

    python h2_gpu.py --r-code DIR --model DIR --tokens DIR --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-code", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--rev-chunk", type=int, default=2048)
    ap.add_argument("--fwd-chunk", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=131072)
    ap.add_argument("--n-h2", type=int, default=50_000)
    ap.add_argument("--check-arena", type=int, default=20_000, help="arena S1s (fixed stride) whose forward top-50 "
                    "is recomputed for a fidelity check against the stored P2 pairs")
    a = ap.parse_args()
    sys.path.insert(0, a.r_code)
    import torch
    from ber.reverse_retrieval import _load_model, blocked_topk, encode_fast, log

    torch.backends.cuda.matmul.allow_tf32 = False
    tok, out = Path(a.tokens), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = _load_model(a.model, "cuda")
    emb, cc = {}, {}
    for side in ("s1", "pool"):
        ids = np.load(tok / f"h2_{side}.ids.npy")
        off = np.load(tok / f"h2_{side}.off.npy")
        cc[side] = np.load(tok / f"h2_{side}.cc.npy")
        log(f"encoding h2_{side}: {len(off) - 1:,}")
        emb[side] = encode_fast(model, ids, off, "cuda", a.max_tokens)
        del ids, off
    del model
    torch.cuda.empty_cache()
    n_s1, n_pool = len(cc["s1"]), len(cc["pool"])
    h_lo = n_s1 - a.n_h2
    rev_idx = np.full((n_pool, a.k), -1, np.int32)
    rev_score = np.full((n_pool, a.k), np.nan, np.float32)
    fwd_idx = np.full((a.n_h2, 50), -1, np.int32)
    fwd_score = np.full((a.n_h2, 50), np.nan, np.float32)
    chk = np.arange(0, h_lo, max(1, h_lo // max(a.check_arena, 1)))[:a.check_arena]
    chk_idx = np.full((len(chk), 50), -1, np.int32)
    chk_pos = {int(r): j for j, r in enumerate(chk)}
    for code in sorted(set(np.unique(cc["s1"][h_lo:])) & set(np.unique(cc["pool"]))):
        s_rows = np.flatnonzero(cc["s1"] == code)
        p_rows = np.flatnonzero(cc["pool"] == code)
        s_t, p_t = torch.from_numpy(s_rows).cuda(), torch.from_numpy(p_rows).cuda()
        S, P = emb["s1"][s_t], emb["pool"][p_t]
        v, i = blocked_topk(P, S, a.k, a.rev_chunk)  # reverse: record -> joint S1s (R's rule)
        rev_score[p_rows, :v.shape[1]] = v.cpu().numpy()
        rev_idx[p_rows, :v.shape[1]] = s_t[i.long()].int().cpu().numpy()
        del v, i
        h_rows = s_rows[s_rows >= h_lo]
        c_rows = s_rows[np.isin(s_rows, chk)]
        q_rows = np.concatenate([h_rows, c_rows])
        Q = emb["s1"][torch.from_numpy(q_rows).cuda()].float()
        Pt = P.float().T.contiguous()
        for s in range(0, len(q_rows), a.fwd_chunk):  # forward: float32 dot, as P2
            sim = Q[s:s + a.fwd_chunk] @ Pt
            v, i = torch.topk(sim, 50, dim=1, largest=True, sorted=True)
            rows = q_rows[s:s + a.fwd_chunk]
            vi, ii = v.cpu().numpy(), p_t[i.long()].int().cpu().numpy()
            h = rows >= h_lo
            fwd_score[rows[h] - h_lo] = vi[h]
            fwd_idx[rows[h] - h_lo] = ii[h]
            for j in np.flatnonzero(~h):
                chk_idx[chk_pos[int(rows[j])]] = ii[j]
            del sim, v, i
        del S, P, Q, Pt
        torch.cuda.empty_cache()
        log(f"country {code}: joint S1 {len(s_rows):,} (HOLDOUT-2 {len(h_rows):,}) x records {len(p_rows):,} "
            f"done ({time.time() - t0:.0f}s)")
    np.save(out / "h2_fwd_idx.npy", fwd_idx)
    np.save(out / "h2_fwd_score.npy", fwd_score)
    np.save(out / "h2_rev_idx.npy", rev_idx)
    np.save(out / "h2_rev_score.npy", rev_score.astype(np.float16))
    np.save(out / "check_arena_rows.npy", chk)
    np.save(out / "check_arena_fwd_idx.npy", chk_idx)
    rep = {"joint_s1": n_s1, "h2_rows": [h_lo, n_s1], "pool": n_pool, "k_reverse": a.k,
           "forward_score": "float32 dot of fp16 unit vectors (P2 procedure)", "seconds": round(time.time() - t0, 1),
           "labels_read": False}
    (out / "gpu_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    log("H2_GPU_A_DONE")


if __name__ == "__main__":
    main()
