"""L: rival-aware listwise (record-centric) cross-encoder.

For every record r the input is ``[CLS] record [SEP] cand_1 [SEP] ... cand_k [SEP]`` with the k <= K claimant
S1s of r (top-K by the v1 base probability; order shuffled in training). Each candidate segment is mean-pooled;
a head scores (candidate, record) pairs and a "none" option; softmax over {candidates, none}.
Output: P(owner = s | r) for every candidate row (NaN for rows outside the record's top-K).

Arena protocol (labels of B_REST/SLICE never read for training; HOLDOUT-2 S1s are not in the arena, asserted):
* claimants = arena candidate rows (U ``prep/arena_cand``) of A and B_REST S1s (SLICE S1s removed everywhere);
* TRAIN groups = records whose non-SLICE claimants are all A S1s (so no B_REST record/rival set is ever trained
  on); target = the A owner if it is among the top-K claimants, else "none" (ownership read only for A owners);
* B_REST groups = records with at least one B_REST claimant (A claimants stay in the set as text-only rivals).
Test: all test candidate rows (U ``prep/test``), all countries.

    python -m ber.listwise_ce prep-arena  (cpu)
    python -m ber.listwise_ce prep-test   (cpu)
    python -m ber.listwise_ce train --data D --out O   (gpu)
    python -m ber.listwise_ce score --data D --ckpt C --split brest|test --out O   (gpu)
    python -m ber.listwise_ce evaluate --scores F --out O   (cpu)
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np

T0 = time.time()
U = Path("/home/ubuntu/run27/U")
L = Path("/home/ubuntu/run27/L")
K = 8
TMAX = 40


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- prep (CPU)

def _tok_chunk(args):
    tok_path, texts, tmax = args
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    tok.no_padding()
    tok.no_truncation()
    encs = tok.encode_batch(texts, add_special_tokens=False)
    lens = np.fromiter((min(len(e.ids), tmax) for e in encs), np.int32, len(encs))
    flat = np.fromiter(itertools.chain.from_iterable(e.ids[:tmax] for e in encs), np.int32, int(lens.sum()))
    return flat, lens


def tokenize(texts, tok_path, procs, tmax=TMAX, chunk=50_000):
    from multiprocessing import get_context

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    jobs = [(tok_path, texts[s:s + chunk], tmax) for s in range(0, len(texts), chunk)]
    with get_context("fork").Pool(procs) as pool:
        res = pool.map(_tok_chunk, jobs, chunksize=1)
    flat = np.concatenate([r[0] for r in res]) if res else np.zeros(0, np.int32)
    lens = np.concatenate([r[1] for r in res]) if res else np.zeros(0, np.int32)
    off = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    return flat, off


def read_texts(path, rows=None, source_marker=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["entity_id", "business_name", "business_address"])
    if rows is not None:
        t = t.take(pa.array(rows))
    names = t.column("business_name").to_pylist()
    addrs = t.column("business_address").to_pylist()
    if source_marker:
        ids = t.column("entity_id").to_pylist()
        return [f"{i[:2]}: {n or ''} | {a or ''}" for i, n, a in zip(ids, names, addrs)]
    return [f"{n or ''} | {a or ''}" for n, a in zip(names, addrs)]


def build_groups(left, right, base, keep, k=K):
    """Top-k claimants (by base desc) per record among rows ``keep``. Returns group arrays (row index = input row)."""
    rows = np.flatnonzero(keep)
    order = rows[np.lexsort((rows, -base[rows], right[rows]))]
    r = right[order]
    start = np.r_[True, r[1:] != r[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    rank = np.arange(len(order)) - first[gid]
    n_claim = np.diff(np.r_[first, len(order)]).astype(np.int32)
    G = len(first)
    g_rec = r[first].astype(np.int64)
    g_row = np.full((G, k), -1, np.int64)
    sel = rank < k
    g_row[gid[sel], rank[sel]] = order[sel]
    return g_rec, g_row, n_claim


def save_tokens(out, prefix, flat, off):
    np.save(out / f"{prefix}_flat.npy", flat)
    np.save(out / f"{prefix}_off.npy", off)


def extra_brest_records():
    """Arena pool rows claimed by a B_REST S1 through Agent R reverse pairs or Agent U route pairs (v2 claimants)."""
    import pyarrow.parquet as pq

    recs = []
    for f in ("r_reverse_pairs_b_rest.parquet", "u_route_pairs_brest.parquet"):
        r = pq.read_table(L / "data/v2" / f, columns=["right_idx"]).column("right_idx").to_numpy()
        recs.append(r[r >= 0].astype(np.int64))
    return np.unique(np.concatenate(recs))


def prep_arena(a):
    out = Path(a.prep_out) if a.prep_out else L / "data/arena"
    out.mkdir(parents=True, exist_ok=True)
    tok_path = str(Path(a.tok) / "tokenizer.json")
    part = np.load(U / "data/arena/finalize_v1/s1_partition.npy")
    import pyarrow.parquet as pq

    s1_ids = pq.read_table(U / "data/arena/reference.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    h2 = set(Path(L / "data/h2_reserved_ids.txt").read_text().split())
    assert len(h2) == 50000
    inter = sum(1 for i in s1_ids if i in h2)
    log(f"HOLDOUT-2 ids in arena reference: {inter}")
    assert inter == 0, "HOLDOUT-2 S1s present in arena"
    del s1_ids
    left = np.load(U / "prep/arena_cand/left.npy").astype(np.int64)
    right = np.load(U / "prep/arena_cand/right.npy").astype(np.int64)
    cand_rows = np.load(U / "prep/arena_cand/cand_rows.npy")
    assert np.all(np.diff(cand_rows) > 0)
    base = np.asarray(np.load(L / "data/base_probabilities.npy", mmap_mode="r")[cand_rows], dtype=np.float32)
    keep = part[left] != 2  # SLICE removed everywhere
    g_rec, g_row, n_claim = build_groups(left, right, base, keep)
    G = len(g_rec)
    valid = g_row >= 0
    g_cand = np.where(valid, left[np.maximum(g_row, 0)], -1)
    cpart = np.where(valid, part[np.maximum(g_cand, 0)], 255)
    # full-claimant partition flags (not just top-K)
    rows = np.flatnonzero(keep)
    n_rec = int(right.max()) + 1
    has_b = np.zeros(n_rec, bool)
    has_b[right[rows[part[left[rows]] == 1]]] = True
    has_a = np.zeros(n_rec, bool)
    has_a[right[rows[part[left[rows]] == 0]]] = True
    if a.v2_exclude:
        has_b[extra_brest_records()] = True  # never train on a record that a B_REST S1 claims via R/U pairs
    g_has_b = has_b[g_rec]
    g_train = has_a[g_rec] & ~g_has_b
    log(f"groups {G}; train-eligible {int(g_train.sum())}; B_REST {int(g_has_b.sum())}; "
        f"n_claim>K {int((n_claim > K).sum())}")
    # targets for train groups: A owner position among top-K else none (slot K)
    owner = np.load(U / "data/arena/labels/rec_owner.npy")[g_rec].astype(np.int64)
    owner_a = np.where((owner >= 0) & (part[np.maximum(owner, 0)] == 0), owner, -1)
    owner_a[~g_train] = -1  # ownership of non-train groups never used
    hit = (g_cand == owner_a[:, None]) & (owner_a[:, None] >= 0)
    target = np.where(hit.any(1), hit.argmax(1), K).astype(np.int8)
    tr = np.flatnonzero(g_train)
    n_top = valid.sum(1)
    stats = {"groups": int(G), "train_groups": int(len(tr)), "brest_groups": int(g_has_b.sum()),
             "train_by_n": {int(n): int(c) for n, c in zip(*np.unique(n_top[tr], return_counts=True))},
             "train_target_none_by_n": {int(n): float((target[tr][n_top[tr] == n] == K).mean())
                                        for n in np.unique(n_top[tr])},
             "train_owner_a_outside_topk": int(((owner_a[tr] >= 0) & (target[tr] == K)).sum())}
    # B_REST coverage: B_REST rows (brest_prep order) inside a top-K group
    brest_pair_rows = np.load(U / "prep/brest/pair_rows.npy")
    pos = np.searchsorted(cand_rows, brest_pair_rows)
    assert np.all(cand_rows[pos] == brest_pair_rows)
    cand2brest = np.full(len(left), -1, np.int64)
    cand2brest[pos] = np.arange(len(pos))
    g_out = np.where(valid, cand2brest[np.maximum(g_row, 0)], -1)
    covered = np.zeros(len(pos), bool)
    covered[g_out[g_out >= 0]] = True
    blab = np.load(U / "prep/brest/label.npy")
    stats["brest_rows"] = int(len(pos))
    stats["brest_rows_in_topk"] = int(covered.sum())
    stats["brest_true_rows_outside_topk"] = int((blab & ~covered).sum())
    stats["brest_true_rows"] = int(blab.sum())
    log(json.dumps(stats))
    # token tables: every arena S1 and every record of a train/B_REST group
    s1_texts = read_texts(U / "data/arena/reference.parquet")
    flat, off = tokenize(s1_texts, tok_path, a.procs)
    del s1_texts
    save_tokens(out, "s1", flat, off)
    log(f"S1 tokens: {len(off) - 1} texts, mean len {np.diff(off).mean():.1f}, truncated share "
        f"{(np.diff(off) >= TMAX).mean():.4f}")
    use = g_train | g_has_b
    recs = np.unique(g_rec[use])
    rec_texts = read_texts(U / "data/arena/pool.parquet", rows=recs, source_marker=True)
    flat, off = tokenize(rec_texts, tok_path, a.procs)
    del rec_texts
    save_tokens(out, "rec", flat, off)
    rec_map = np.full(n_rec, -1, np.int64)
    rec_map[recs] = np.arange(len(recs))
    log(f"record tokens: {len(recs)} texts, mean len {np.diff(off).mean():.1f}")
    np.save(out / "rec_map.npy", rec_map)
    np.save(out / "g_rec.npy", g_rec)
    np.save(out / "g_cand.npy", g_cand.astype(np.int32))
    np.save(out / "g_out.npy", g_out)  # brest_prep row per slot (-1: not a B_REST row)
    np.save(out / "g_train.npy", g_train)
    np.save(out / "g_brest.npy", g_has_b)
    np.save(out / "g_target.npy", target)
    np.save(out / "g_nclaim.npy", n_claim)
    (out / "prep_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    log("arena prep done")


def prep_test(a):
    out = Path(a.prep_out) if a.prep_out else L / "data/test"
    out.mkdir(parents=True, exist_ok=True)
    tok_path = str(Path(a.tok) / "tokenizer.json")
    d = U / "prep/test"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    base = np.load(d / "base.npy").astype(np.float32)
    g_rec, g_row, n_claim = build_groups(left, right, base, np.ones(len(left), bool))
    valid = g_row >= 0
    g_cand = np.where(valid, left[np.maximum(g_row, 0)], -1)
    stats = {"rows": int(len(left)), "groups": int(len(g_rec)), "rows_in_topk": int(valid.sum()),
             "n_claim>K": int((n_claim > K).sum()),
             "by_n": {int(n): int(c) for n, c in zip(*np.unique(valid.sum(1), return_counts=True))}}
    log(json.dumps(stats))
    s1_texts = read_texts(U / "data/test/dense_test/reference.parquet")
    flat, off = tokenize(s1_texts, tok_path, a.procs)
    del s1_texts
    save_tokens(out, "s1", flat, off)
    recs = np.unique(g_rec)
    rec_texts = read_texts(U / "data/test/dense_test/pool.parquet", rows=recs, source_marker=True)
    flat, off = tokenize(rec_texts, tok_path, a.procs)
    del rec_texts
    save_tokens(out, "rec", flat, off)
    n_rec = int(right.max()) + 1
    rec_map = np.full(n_rec, -1, np.int64)
    rec_map[recs] = np.arange(len(recs))
    np.save(out / "rec_map.npy", rec_map)
    np.save(out / "g_rec.npy", g_rec)
    np.save(out / "g_cand.npy", g_cand.astype(np.int32))
    np.save(out / "g_out.npy", g_row)  # test_prep row per slot
    np.save(out / "g_nclaim.npy", n_claim)
    (out / "prep_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    log("test prep done")


# ----------------------------------------------------------------------------- batching

class Tokens:
    def __init__(self, d):
        d = Path(d)
        self.s1_flat = np.load(d / "s1_flat.npy")
        self.s1_off = np.load(d / "s1_off.npy")
        self.rec_flat = np.load(d / "rec_flat.npy")
        self.rec_off = np.load(d / "rec_off.npy")
        self.rec_map = np.load(d / "rec_map.npy")
        self.g_rec = np.load(d / "g_rec.npy")
        self.g_cand = np.load(d / "g_cand.npy")
        self.n = (self.g_cand >= 0).sum(1)

    def seq_len(self, groups):
        rm = self.rec_map[self.g_rec[groups]]
        ln = 2 + (self.rec_off[rm + 1] - self.rec_off[rm])
        c = self.g_cand[groups]
        cl = np.where(c >= 0, self.s1_off[np.maximum(c, 0) + 1] - self.s1_off[np.maximum(c, 0)] + 1, 0)
        return ln + cl.sum(1)

    def batch(self, groups, rng=None, cls=0, sep=2, pad=1):
        """Token ids, token types, attention mask, segment ids (-1 = CLS/pad, 0 = record, j = slot j-1) and slot
        permutation (slot j of the output holds original candidate perm[j])."""
        B = len(groups)
        seqs, segs, perms = [], [], []
        for g in groups:
            rm = self.rec_map[self.g_rec[g]]
            rec = self.rec_flat[self.rec_off[rm]:self.rec_off[rm + 1]]
            cands = self.g_cand[g]
            n = int((cands >= 0).sum())
            perm = rng.permutation(n) if rng is not None else np.arange(n)
            parts = [np.array([cls], np.int32), rec, np.array([sep], np.int32)]
            sp = [np.array([-1], np.int16), np.zeros(len(rec) + 1, np.int16)]
            for j, o in enumerate(perm):
                s = cands[o]
                t = self.s1_flat[self.s1_off[s]:self.s1_off[s + 1]]
                parts += [t, np.array([sep], np.int32)]
                sp.append(np.full(len(t) + 1, j + 1, np.int16))
            seqs.append(np.concatenate(parts))
            segs.append(np.concatenate(sp))
            perms.append(perm)
        Lm = max(len(s) for s in seqs)
        ids = np.full((B, Lm), pad, np.int64)
        seg = np.full((B, Lm), -1, np.int64)
        for i, (s, sg) in enumerate(zip(seqs, segs)):
            ids[i, :len(s)] = s
            seg[i, :len(s)] = sg
        att = (seg != -1).astype(np.int64)
        att[:, 0] = 1
        tt = (seg >= 1).astype(np.int64)
        return ids, tt, att, seg, perms


def make_batches(tokens, groups, budget, rng=None, max_b=512):
    """Token-budget batches of groups with similar length (sorted; shuffled batch order when rng given)."""
    ln = tokens.seq_len(groups)
    if rng is not None:
        noise = rng.random(len(groups)) * 8  # jitter so batches differ between epochs
        order = np.argsort(ln + noise, kind="stable")
    else:
        order = np.argsort(ln, kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, ln[i])
        if cur and (m * (len(cur) + 1) > budget or len(cur) >= max_b):
            batches.append(groups[np.array(cur)])
            cur, m = [], ln[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(groups[np.array(cur)])
    if rng is not None:
        rng.shuffle(batches)
    return batches


# ----------------------------------------------------------------------------- model (GPU)

def build_model(path):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class LW(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(path, attn_implementation="sdpa")
            H = self.enc.config.hidden_size
            self.pair = nn.Sequential(nn.Linear(4 * H, H), nn.GELU(), nn.Linear(H, 1))
            self.none = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, 1))

        def forward(self, ids, tt, att, seg):
            if self.enc.config.type_vocab_size < 2:
                tt = torch.zeros_like(tt)
            h = self.enc(input_ids=ids, attention_mask=att, token_type_ids=tt).last_hidden_state
            oh = torch.nn.functional.one_hot(seg.clamp(min=0), K + 1).to(h.dtype) * (seg >= 0).unsqueeze(-1).to(h.dtype)
            cnt = oh.sum(1)  # B, K+1
            pooled = torch.einsum("blk,blh->bkh", oh, h) / cnt.clamp(min=1).unsqueeze(-1)
            hr, hc = pooled[:, 0], pooled[:, 1:]
            hr_e = hr.unsqueeze(1).expand_as(hc)
            lc = self.pair(torch.cat([hc, hr_e, hc * hr_e, (hc - hr_e).abs()], -1)).squeeze(-1).float()
            lc = lc.masked_fill(cnt[:, 1:] == 0, -1e4)
            ln = self.none(torch.cat([h[:, 0], hr], -1)).float()
            return torch.cat([lc, ln], 1)  # B, K+1 (slot K = none)

    return LW()


def to_dev(arrs, dev):
    import torch

    return [torch.from_numpy(x).to(dev, non_blocking=True) for x in arrs]


def train(a):
    import threading
    import queue

    import torch

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tokens = Tokens(a.data)
    g_train = np.load(Path(a.data) / "g_train.npy")
    target_all = np.load(Path(a.data) / "g_target.npy")
    tr = np.flatnonzero(g_train)
    n = tokens.n[tr]
    # sample: all multi-claimant train groups (capped) + a share of single-claimant groups
    multi = tr[n >= 2]
    single = tr[n == 1]
    if len(multi) > a.max_multi:
        multi = rng.choice(multi, a.max_multi, replace=False)
    heavy = multi[tokens.n[multi] >= 5]
    multi = np.concatenate([multi] + [heavy] * a.dup_heavy)  # test has ~8x more 8-claimant groups than train
    single = rng.choice(single, min(len(single), a.n_single), replace=False)
    groups = np.concatenate([multi, single])
    log(f"train groups {len(groups)} (multi {len(multi)}, single {len(single)}); none share "
        f"{(target_all[groups] == K).mean():.4f}")
    np.save(out / "train_groups.npy", groups)
    batches = make_batches(tokens, groups, a.budget, rng=rng)
    total = len(batches) * a.epochs
    log(f"{len(batches)} batches/epoch, {total} steps")
    dev = torch.device(a.device)
    model = build_model(a.model).to(dev)
    enc_params = list(model.enc.parameters())
    head_params = list(model.pair.parameters()) + list(model.none.parameters())
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.lr * 10}],
                            weight_decay=0.01, fused=True)
    warm = min(500, total // 20)
    budget_s = a.max_minutes * 60
    t_start = time.time()

    def progress(s):  # linear decay over whichever ends first: the planned steps or the time budget
        return max(s / total, (time.time() - t_start) / budget_s)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 1 - progress(s)))

    q = queue.Queue(maxsize=16)

    def producer():
        prng = np.random.default_rng(a.seed + 1)
        for ep in range(a.epochs):
            bl = batches if ep == 0 else make_batches(tokens, groups, a.budget, rng=prng)
            for b in bl:
                ids, tt, att, seg, perms = tokens.batch(b, rng=prng)
                tgt = np.full(len(b), K, np.int64)
                for i, (g, p) in enumerate(zip(b, perms)):
                    t = target_all[g]
                    if t < K:
                        tgt[i] = int(np.flatnonzero(p == t)[0])
                q.put((ids, tt, att, seg, tgt))
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    model.train()
    step, t_last, seen, loss_acc, acc_acc, cnt = 0, time.time(), 0, 0.0, 0.0, 0
    while True:
        item = q.get()
        if item is None:
            break
        ids, tt, att, seg, tgt = to_dev(item, dev)
        with torch.autocast(a.device, dtype=torch.bfloat16):
            logits = model(ids, tt, att, seg)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        seen += len(tgt)
        loss_acc += float(loss.detach()) if step % 50 == 0 else 0.0
        if step % 50 == 0:
            acc_acc += float((logits.argmax(1) == tgt).float().mean())
            cnt += 1
        if step % 200 == 0:
            el = time.time() - t_last
            log(f"step {step}/{total} loss {loss_acc / max(cnt, 1):.4f} acc {acc_acc / max(cnt, 1):.4f} "
                f"groups/s {seen / el:.0f} lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
            t_last, seen, loss_acc, acc_acc, cnt = time.time(), 0, 0.0, 0.0, 0
        if step % a.save_every == 0:
            torch.save(model.state_dict(), out / "model.pt")
        if progress(step) >= 1.0:
            log(f"schedule complete at step {step} ({progress(step):.3f})")
            break
    torch.save(model.state_dict(), out / "model.pt")
    json.dump({"steps": step, "total_planned": total, "groups": int(len(groups)), "args": vars(a),
               "elapsed_s": round(time.time() - T0, 1)}, open(out / "train_report.json", "w"), indent=2)
    log("train done")


def score(a):
    import threading
    import queue

    import torch

    tokens = Tokens(a.data)
    d = Path(a.data)
    if a.subset:
        sel = np.load(d / f"groups_{a.subset}.npy")
    elif a.split == "brest":
        sel = np.flatnonzero(np.load(d / "g_brest.npy"))
    elif a.split == "trainval":
        sel = np.load(Path(a.ckpt).parent / "train_groups.npy")[:a.limit]
    else:
        sel = np.arange(len(tokens.g_rec))
    if a.limit and a.split != "trainval":
        sel = sel[:a.limit]
    batches = make_batches(tokens, sel, a.budget)
    log(f"scoring {len(sel)} groups in {len(batches)} batches")
    dev = torch.device(a.device)
    model = build_model(a.model).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    probs = np.full((len(tokens.g_rec), K + 1), np.nan, np.float32)
    q = queue.Queue(maxsize=16)

    def producer():
        for b in batches:
            q.put((b, tokens.batch(b)[:4]))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    done, t1 = 0, time.time()
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                break
            b, arrs = item
            with torch.autocast(a.device, dtype=torch.bfloat16):
                logits = model(*to_dev(arrs, dev))
            probs[b] = torch.softmax(logits.float(), 1).cpu().numpy()
            done += len(b)
            if done // 200_000 != (done - len(b)) // 200_000:
                log(f"{done}/{len(sel)} groups, {done / (time.time() - t1):.0f} groups/s")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.split}_{a.subset}" if a.subset else a.split
    np.save(out / f"probs_{tag}.npy", probs)
    np.save(out / f"groups_{tag}.npy", sel)
    log(f"scored {len(sel)} groups in {time.time() - t1:.0f}s")


# ----------------------------------------------------------------------------- export / evaluation (CPU)

def export(a):
    d = Path(a.data)
    out = Path(a.out)
    probs = np.load(out / f"probs_{a.split}.npy")
    sel = np.load(out / f"groups_{a.split}.npy")
    g_out = np.load(d / "g_out.npy")
    g_cand = np.load(d / "g_cand.npy")
    n_rows = len(np.load(U / ("prep/brest/left.npy" if a.split == "brest" else "prep/test/left.npy"), mmap_mode="r"))
    p = np.full(n_rows, np.nan, np.float32)
    pn = np.full(n_rows, np.nan, np.float32)
    gsize = np.full(n_rows, np.nan, np.float32)
    go = g_out[sel]
    gi, sj = np.nonzero(go >= 0)
    rows = go[gi, sj]
    p[rows] = probs[sel][gi, sj]
    pn[rows] = probs[sel][gi, K]
    gsize[rows] = (g_cand[sel] >= 0).sum(1)[gi]
    # rival: best other candidate probability in the group
    pr = np.nan_to_num(probs[sel][:, :K], nan=-1.0)
    top1 = pr.max(1)
    srt = np.sort(pr, 1)
    top2 = srt[:, -2]
    mine = pr[gi, sj]
    rival = np.where(mine >= top1[gi], top2[gi], top1[gi])
    rv = np.full(n_rows, np.nan, np.float32)
    rv[rows] = np.maximum(rival, 0.0)
    X = np.column_stack([p, pn, rv, gsize]).astype(np.float32)
    path = out / f"{a.split}_lw_features.npy"
    np.save(path, X)
    cols = ["lw_p", "lw_p_none", "lw_rival_p", "lw_group_size"]
    col_sha = {}
    for j, c in enumerate(cols):
        cp = out / f"{a.split}_{c}.npy"
        np.save(cp, np.ascontiguousarray(X[:, j]))
        col_sha[cp.name] = sha256(cp)
    ncl = np.load(d / "g_nclaim.npy")[sel]
    info = {"columns": cols, "rows": int(n_rows),
            "row_order": "U brest_prep" if a.split == "brest" else "U test_prep",
            "rows_scored": int(np.isfinite(p).sum()), "nan_meaning": "row outside its record's top-8 claimants",
            "meaning": {"lw_p": "P(owner = s | record) from a softmax over the record's top-8 claimant S1s + none",
                        "lw_p_none": "P(none | record) of the row's record", "lw_rival_p": "max P of the other claimants",
                        "lw_group_size": "claimants in the model input (<= 8); density-sensitive, not recommended"},
            "claimant_sets": ("top-8 by v1 base probability among all candidate S1s of the record; arena: A and B_REST "
                              "S1s (SLICE removed), test: all test S1s"),
            "records_scored": int(len(sel)), "records_truncated_at_8": int((ncl > K).sum()),
            "share_records_truncated": float((ncl > K).mean()),
            "model": "intfloat/multilingual-e5-small (MIT) fine-tuned on arena A records whose claimants are all A",
            "sha256": sha256(path), "column_files_sha256": col_sha}
    (out / f"{a.split}_lw_features.json").write_text(json.dumps(info, indent=2) + "\n")
    log(json.dumps(info))


def evaluate(a):
    import pyarrow.parquet as pq

    from ber.arena_decision import component_bootstrap, evaluate as ev, one_owner

    out = Path(a.out)
    d = U / "prep/brest"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    label = np.load(d / "label.npy")
    comp = np.load(U / "data/arena/finalize_v1/component_ids.npy")
    part = np.load(U / "data/arena/finalize_v1/s1_partition.npy")
    true_count = np.load(U / "data/arena/labels/true_count.npy")
    country = np.asarray(pq.read_table(U / "data/arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    X = np.load(out / "brest_lw_features.npy")
    p = np.nan_to_num(X[:, 0], nan=0.0).astype(np.float64)
    pop = part == 1
    half = (comp[left] % 2).astype(np.int64)
    s1_half = (comp % 2).astype(np.int64)

    def rank_within(group, score):
        order = np.lexsort((np.arange(len(score)), -score, group))
        g = group[order]
        start = np.r_[True, g[1:] != g[:-1]]
        idx = np.arange(len(g))
        first = np.maximum.accumulate(np.where(start, idx, 0))
        rank = np.empty(len(g), dtype=np.int64)
        rank[order] = idx - first
        return rank

    def cap(acc, sc, k=11):
        rows = np.flatnonzero(acc)
        acc = acc.copy()
        acc[rows[rank_within(left[rows], sc[rows]) >= k]] = False
        return acc

    oof = np.load(d / "oof_c2tree.npy")
    thr0 = np.where(half == 1, 0.75, 0.725)
    base_acc = cap(one_owner(oof >= thr0, left, right, oof), oof)
    base_res = ev(base_acc, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per_base = base_res.pop("per_s1")
    log(f"pair tree baseline {base_res['macro_f05']:.6f}")
    grid = {0: {}, 1: {}}
    thrs = [round(t, 3) for t in np.arange(0.2, 0.96, 0.025)]
    for t in thrs:
        acc = cap(one_owner(p >= t, left, right, p), p)
        per = ev(acc, left, label, true_count, return_per_s1=True)["per_s1"]
        for h in (0, 1):
            grid[h][t] = float(per[pop & (s1_half == h)].mean())
    pick = {h: max(grid[h], key=grid[h].get) for h in (0, 1)}
    final = np.zeros(len(left), bool)
    for h in (0, 1):
        t = pick[1 - h]
        final |= (p >= t) & (half == h)
    final = cap(one_owner(final, left, right, p), p)
    res = ev(final, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per = res.pop("per_s1")
    boot = component_bootstrap(per_base, per, comp, n=a.bootstrap, seed=20260927, population=pop)
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy")
    slim = lambda r: dict({k: r[k] for k in keys}, by_country={c: {k: v[k] for k in ("macro_f05", "precision",
                                                                                      "recall", "false_links")}
                                                            for c, v in r["by_country"].items()})
    rep = {"question": "L listwise CE standalone on B_REST (A-trained; threshold per half picked on the other half)",
           "pair_tree_baseline": slim(base_res), "lw": slim(res), "lw_minus_pair_tree": boot,
           "picks": pick, "grid": grid, "labels_read": "B_REST only (evaluation)"}
    (out / "brest_report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    np.save(out / "per_s1_lw.npy", per)
    log(f"LW standalone {res['macro_f05']:.6f} vs pair tree {base_res['macro_f05']:.6f}; {boot}")


def kcheck(a):
    """K-lite A/B on B_REST: binary LightGBM (K-robust params) on x_base + x_q + K export (33 cols), with vs
    without the LW features; 2-fold component-parity cross-fit; threshold per half picked on the other half."""
    import lightgbm as lgb
    import pyarrow.parquet as pq

    from ber.arena_decision import component_bootstrap, evaluate as ev, one_owner

    out = Path(a.out)
    d = U / "prep/brest"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    label = np.load(d / "label.npy")
    comp = np.load(U / "data/arena/finalize_v1/component_ids.npy")
    part = np.load(U / "data/arena/finalize_v1/s1_partition.npy")
    true_count = np.load(U / "data/arena/labels/true_count.npy")
    country = np.asarray(pq.read_table(U / "data/arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    pop = part == 1
    half = (comp[left] % 2).astype(np.int64)
    s1_half = (comp % 2).astype(np.int64)
    X0 = np.hstack([np.load(d / "x_base.npy"), np.load(d / "x_q.npy"), np.load(L / "data/brest_k_features.npy")])
    variants = {"klite": X0}
    if not a.baseline_only:
        lw = np.load(out / "brest_lw_features.npy")[:, :3]  # lw_p, lw_p_none, lw_rival_p (group size: density-sensitive)
        variants["klite_lw"] = np.hstack([X0, lw])
    params = {"learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "verbose": -1, "seed": 0,
              "objective": "binary", "num_threads": a.threads}

    def rank_within(group, score):
        order = np.lexsort((np.arange(len(score)), -score, group))
        g = group[order]
        start = np.r_[True, g[1:] != g[:-1]]
        idx = np.arange(len(g))
        first = np.maximum.accumulate(np.where(start, idx, 0))
        rank = np.empty(len(g), dtype=np.int64)
        rank[order] = idx - first
        return rank

    def cap(acc, sc, k=11):
        rows = np.flatnonzero(acc)
        acc = acc.copy()
        acc[rows[rank_within(left[rows], sc[rows]) >= k]] = False
        return acc

    thrs = [round(t, 3) for t in np.arange(0.5, 0.951, 0.025)]
    report, pers = {}, {}
    for name, X in variants.items():
        oof_path = out / f"oof_{name}.npy"
        if oof_path.exists():
            p = np.load(oof_path).astype(np.float64)
        else:
            p = np.zeros(len(left))
            for h in (0, 1):
                tr, te = np.flatnonzero(half == h), np.flatnonzero(half != h)
                m = lgb.train(params, lgb.Dataset(X[tr], label[tr].astype(np.float32)), num_boost_round=a.rounds)
                p[te] = m.predict(X[te], num_threads=a.threads)
                if name == "klite_lw" and h == 0:
                    gain = m.feature_importance("gain")
                    report["lw_gain_share_half0"] = float(gain[-3:].sum() / gain.sum())
                    report["lw_gain_rank_half0"] = [int((gain > g).sum()) + 1 for g in gain[-3:]]
                log(f"{name} half {h} fitted")
            np.save(oof_path, p.astype(np.float32))
        grid = {0: {}, 1: {}}
        for t in thrs:
            acc = cap(one_owner(p >= t, left, right, p), p)
            per = ev(acc, left, label, true_count, return_per_s1=True)["per_s1"]
            for h in (0, 1):
                grid[h][t] = float(per[pop & (s1_half == h)].mean())
        pick = {h: max(grid[h], key=grid[h].get) for h in (0, 1)}
        final = np.zeros(len(left), bool)
        for h in (0, 1):
            final |= (p >= pick[1 - h]) & (half == h)
        final = cap(one_owner(final, left, right, p), p)
        res = ev(final, left, label, true_count, population=pop, country=country, return_per_s1=True)
        pers[name] = res.pop("per_s1")
        report[name] = {k: res[k] for k in ("macro_f05", "precision", "recall", "false_links", "singleton_accuracy")}
        report[name]["by_country"] = {c: v["macro_f05"] for c, v in res["by_country"].items()}
        report[name]["picks"] = pick
        log(f"{name}: {res['macro_f05']:.6f}")
    if len(pers) == 2:
        report["lw_minus_klite"] = component_bootstrap(pers["klite"], pers["klite_lw"], comp, n=a.bootstrap,
                                                       seed=20260927, population=pop)
        log(json.dumps(report["lw_minus_klite"]))
    (out / "kcheck_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")



# ----------------------------------------------------------------------------- v2: claimant sets + R reverse / U route pairs

V2_SRC = {"brest": ["brest_prep", "r_reverse_pairs_b_rest", "u_route_pairs_brest"],
          "test": ["test_prep", "r_reverse_pairs_test", "u_route_pairs_test"]}


def select_v2(rec, score, is_new, k=K, reserve=2):
    """Per record: dense rows ranked by score, new rows ranked by score; keep_new = min(n_new, max(reserve, k - n_dense)),
    keep_dense = min(n_dense, k - keep_new). Returns kept row mask."""
    order = np.lexsort((-score, is_new, rec))
    r, nw = rec[order], is_new[order]
    key_start = np.r_[True, (r[1:] != r[:-1]) | (nw[1:] != nw[:-1])]
    first = np.maximum.accumulate(np.where(key_start, np.arange(len(r)), 0))
    rank = np.arange(len(r)) - first
    rstart = np.r_[True, r[1:] != r[:-1]]
    gid = np.cumsum(rstart) - 1
    n_new = np.bincount(gid, weights=nw.astype(float)).astype(np.int64)
    n_all = np.bincount(gid).astype(np.int64)
    n_d = n_all - n_new
    keep_new = np.minimum(n_new, np.maximum(reserve, k - n_d))
    keep_d = np.minimum(n_d, k - keep_new)
    lim = np.where(nw, keep_new[gid], keep_d[gid])
    keep = np.zeros(len(rec), bool)
    keep[order[rank < lim]] = True
    return keep


def prep_v2(a):
    import pyarrow.parquet as pq

    split = a.split
    out = L / f"data/v2_{split}"
    out.mkdir(parents=True, exist_ok=True)
    tok_path = str(Path(a.tok) / "tokenizer.json")
    v2 = L / "data/v2"
    raw_name = "train_pool" if split == "brest" else "test_pool"
    raw_ids = pq.read_table(v2 / f"{raw_name}.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    id2raw = {e: i for i, e in enumerate(raw_ids)}
    if split == "brest":
        part = np.load(U / "data/arena/finalize_v1/s1_partition.npy")
        c2r = pq.read_table(U / "data/arena/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
        left = np.load(U / "prep/arena_cand/left.npy").astype(np.int64)
        right = np.load(U / "prep/arena_cand/right.npy").astype(np.int64)
        cand_rows = np.load(U / "prep/arena_cand/cand_rows.npy")
        base = np.asarray(np.load(L / "data/base_probabilities.npy", mmap_mode="r")[cand_rows], dtype=np.float32)
        pos = np.searchsorted(cand_rows, np.load(U / "prep/brest/pair_rows.npy"))
        out0 = np.full(len(left), -1, np.int64)
        out0[pos] = np.arange(len(pos))
        keep0 = part[left] != 2
        L0, R0, S0, O0 = left[keep0], c2r[right[keep0]], base[keep0], out0[keep0]
        ra = pq.read_table(v2 / "r_reverse_pairs_a.parquet", columns=["left_idx", "raw_row_idx", "cosine"])
        rb = pq.read_table(v2 / "r_reverse_pairs_b_rest.parquet", columns=["left_idx", "raw_row_idx", "cosine"])
        ub = pq.read_table(v2 / "u_route_pairs_brest.parquet", columns=["left_idx", "right_idx", "p_u8f"])
        parts = [(L0, R0, S0, np.zeros(len(L0), np.int8), O0),
                 (ra.column("left_idx").to_numpy().astype(np.int64), ra.column("raw_row_idx").to_numpy().astype(np.int64),
                  ra.column("cosine").to_numpy().astype(np.float32), np.full(len(ra), 1, np.int8), np.full(len(ra), -1, np.int64)),
                 (rb.column("left_idx").to_numpy().astype(np.int64), rb.column("raw_row_idx").to_numpy().astype(np.int64),
                  rb.column("cosine").to_numpy().astype(np.float32), np.full(len(rb), 2, np.int8), np.arange(len(rb))),
                 (ub.column("left_idx").to_numpy().astype(np.int64), c2r[ub.column("right_idx").to_numpy()].astype(np.int64),
                  ub.column("p_u8f").to_numpy().astype(np.float32), np.full(len(ub), 3, np.int8), np.arange(len(ub)))]
        # src 1 = R A rows (context only); outputs: src0 -> brest_prep row, src2 -> R b_rest row, src3 -> U row
        assert np.all(part[parts[1][0]] == 0) and np.all(part[parts[2][0]] == 1) and np.all(part[parts[3][0]] == 1)
        s1_ref = U / "data/arena/reference.parquet"
    else:
        d = U / "prep/test"
        c2r = pq.read_table(U / "data/test/dense_test/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
        left = np.load(d / "left.npy").astype(np.int64)
        right = np.load(d / "right.npy").astype(np.int64)
        base = np.load(d / "base.npy").astype(np.float32)
        rt = pq.read_table(v2 / "r_reverse_pairs_test.parquet", columns=["left_idx", "raw_row_idx", "cosine"])
        ut = pq.read_table(v2 / "u_route_pairs_test.parquet", columns=["left_idx", "rec_id", "p_u8f"])
        urec = np.fromiter((id2raw[e] for e in ut.column("rec_id").to_pylist()), np.int64, len(ut))
        parts = [(left, c2r[right].astype(np.int64), base, np.zeros(len(left), np.int8), np.arange(len(left))),
                 (rt.column("left_idx").to_numpy().astype(np.int64), rt.column("raw_row_idx").to_numpy().astype(np.int64),
                  rt.column("cosine").to_numpy().astype(np.float32), np.full(len(rt), 2, np.int8), np.arange(len(rt))),
                 (ut.column("left_idx").to_numpy().astype(np.int64), urec, ut.column("p_u8f").to_numpy().astype(np.float32),
                  np.full(len(ut), 3, np.int8), np.arange(len(ut)))]
        s1_ref = U / "data/test/dense_test/reference.parquet"
    chk = np.random.default_rng(0).choice(len(c2r), 2000, replace=False)
    pool_ids = pq.read_table(U / ("data/arena/pool.parquet" if split == "brest" else "data/test/dense_test/pool.parquet"),
                             columns=["entity_id"]).column("entity_id").to_pylist()
    assert all(raw_ids[c2r[i]] == pool_ids[i] for i in chk), "raw_row_idx mapping mismatch"
    del pool_ids
    Lf = np.concatenate([p[0] for p in parts])
    Rf = np.concatenate([p[1] for p in parts])
    Sf = np.concatenate([p[2] for p in parts])
    Cf = np.concatenate([p[3] for p in parts])
    Of = np.concatenate([p[4] for p in parts])
    # dedupe (s1, record) keeping the first source
    key = Rf * 4_000_000 + Lf
    _, first = np.unique(key, return_index=True)
    dup = len(key) - len(first)
    sel = np.sort(first)
    Lf, Rf, Sf, Cf, Of = Lf[sel], Rf[sel], Sf[sel], Cf[sel], Of[sel]
    if split == "brest":
        has_b = np.zeros(int(Rf.max()) + 1, bool)
        has_b[Rf[part[Lf] == 1]] = True
        m = has_b[Rf]
        Lf, Rf, Sf, Cf, Of = Lf[m], Rf[m], Sf[m], Cf[m], Of[m]
    keep = select_v2(Rf, Sf, Cf > 0)
    n_rows_src = {int(c): int((Cf == c).sum()) for c in np.unique(Cf)}
    kept_src = {int(c): int((keep & (Cf == c)).sum()) for c in np.unique(Cf)}
    Lk, Rk, Sk, Ck, Ok = Lf[keep], Rf[keep], Sf[keep], Cf[keep], Of[keep]
    order = np.lexsort((-Sk, Ck > 0, Rk))
    Lk, Rk, Ck, Ok = Lk[order], Rk[order], Ck[order], Ok[order]
    start = np.r_[True, Rk[1:] != Rk[:-1]]
    gid = np.cumsum(start) - 1
    firsti = np.flatnonzero(start)
    rank = np.arange(len(Rk)) - firsti[gid]
    G = len(firsti)
    g_rec = Rk[firsti]
    g_cand = np.full((G, K), -1, np.int32)
    g_src = np.full((G, K), -1, np.int8)
    g_out = np.full((G, K), -1, np.int64)
    g_cand[gid, rank] = Lk
    g_src[gid, rank] = Ck
    g_out[gid, rank] = Ok
    n_claim = np.bincount(np.searchsorted(g_rec, Rf), minlength=G).astype(np.int32)
    stats = {"split": split, "rows_by_src": n_rows_src, "kept_by_src": kept_src, "duplicates_dropped": int(dup),
             "groups": int(G), "records_truncated": int((n_claim > K).sum())}
    if split == "brest":
        blab = np.load(U / "prep/brest/label.npy")
        rbl = pq.read_table(v2 / "r_reverse_pairs_b_rest.parquet", columns=["label"]).column("label").to_numpy()
        tk0 = Ok[(Ck == 0) & (Ok >= 0)]
        stats["brest_true_dense_in_groups"] = int(blab[tk0].sum())
        stats["r_true_in_groups"] = int(rbl[Ok[Ck == 2]].sum())
    log(json.dumps(stats))
    if split == "test":
        s1_texts = read_texts(s1_ref)
        flat, off = tokenize(s1_texts, tok_path, a.procs)
        del s1_texts
    else:
        flat, off = np.load(L / "data/arena_b/s1_flat.npy"), np.load(L / "data/arena_b/s1_off.npy")
    save_tokens(out, "s1", flat, off)
    recs = np.unique(g_rec)
    texts = pq.read_table(v2 / f"{raw_name}.parquet", columns=["text"]).column("text")
    import pyarrow as pa
    tx = texts.take(pa.array(recs)).to_pylist()
    rec_texts = [f"{raw_ids[r][:2]}: {t[9:] if t.startswith('passage: ') else t}" for r, t in zip(recs, tx)]
    del tx, texts
    flat, off = tokenize(rec_texts, tok_path, a.procs)
    save_tokens(out, "rec", flat, off)
    rec_map = np.full(len(raw_ids), -1, np.int64)
    rec_map[recs] = np.arange(len(recs))
    np.save(out / "rec_map.npy", rec_map)
    np.save(out / "g_rec.npy", g_rec)
    np.save(out / "g_cand.npy", g_cand)
    np.save(out / "g_src.npy", g_src)
    np.save(out / "g_out.npy", g_out)
    np.save(out / "g_nclaim.npy", n_claim)
    np.save(out / "g_brest.npy", np.ones(G, bool))
    cc = pq.read_table(v2 / f"{raw_name}.parquet", columns=["cc"]).column("cc").to_numpy()[g_rec]
    np.save(out / "groups_first.npy", np.flatnonzero(cc != 2))  # US/India records (scored first)
    np.save(out / "groups_second.npy", np.flatnonzero(cc == 2))  # France records
    (out / "prep_report.json").write_text(json.dumps(stats, indent=2) + "\n")
    log("v2 prep done")


def export_v2(a):
    import pyarrow.parquet as pq

    split = a.split
    d = L / f"data/v2_{split}"
    out = Path(a.out)
    if (out / f"probs_{split}.npy").exists():
        probs = np.load(out / f"probs_{split}.npy")
        sel = np.load(out / f"groups_{split}.npy")
    else:  # merge subset passes (first = US/India, second = France)
        tags = [t for t in ("first", "second") if (out / f"groups_{split}_{t}.npy").exists()]
        probs = np.load(out / f"probs_{split}_{tags[0]}.npy")
        sel = np.load(out / f"groups_{split}_{tags[0]}.npy")
        for t in tags[1:]:
            s2 = np.load(out / f"groups_{split}_{t}.npy")
            probs[s2] = np.load(out / f"probs_{split}_{t}.npy", mmap_mode="r")[s2]
            sel = np.concatenate([sel, s2])
        log(f"merged subsets {tags}")
    g_out, g_src, g_cand = np.load(d / "g_out.npy")[sel], np.load(d / "g_src.npy")[sel], np.load(d / "g_cand.npy")[sel]
    pr = probs[sel]
    n_src = [len(np.load(U / f"prep/{'brest' if split == 'brest' else 'test'}/left.npy", mmap_mode="r"))] + \
        [pq.ParquetFile(L / "data/v2" / f"{f}.parquet").metadata.num_rows for f in V2_SRC[split][1:]]
    src_codes = [0, 2, 3]
    p8 = np.nan_to_num(pr[:, :K], nan=-1.0)
    top1 = p8.max(1)
    top2 = np.sort(p8, 1)[:, -2]
    gsize = (g_cand >= 0).sum(1)
    info = {"columns": ["lw_p", "lw_p_none", "lw_rival_p", "lw_group_size"], "files": {}}
    for name, code, n in zip(V2_SRC[split], src_codes, n_src):
        gi, sj = np.nonzero((g_src == code) & (g_out >= 0))
        rows = g_out[gi, sj]
        X = np.full((n, 4), np.nan, np.float32)
        mine = pr[gi, sj]
        X[rows, 0] = mine
        X[rows, 1] = pr[gi, K]
        X[rows, 2] = np.maximum(np.where(mine >= top1[gi], top2[gi], top1[gi]), 0.0)
        X[rows, 3] = gsize[gi]
        tag = f"{split}_{name}" if code else split
        path = out / f"{tag}_lw_features.npy"
        np.save(path, X)
        col_sha = {}
        for j, c in enumerate(info["columns"]):
            cp = out / f"{tag}_{c}.npy"
            np.save(cp, np.ascontiguousarray(X[:, j]))
            col_sha[cp.name] = sha256(cp)
        info["files"][tag] = {"rows": int(n), "row_order": name if code else f"U {name}", "rows_scored": int(len(rows)),
                              "sha256": sha256(path), "column_files_sha256": col_sha}
    info["claimant_sets"] = ("top-8 per record (raw record index) over dense candidate rows (v1 base order) + Agent R reverse "
                             "pairs (cosine order) + Agent U route pairs (p_u8f order); new pairs get >= 2 reserved slots; "
                             "arena: A and B_REST S1s (SLICE removed; R A-pairs as context), test: all")
    info["nan_meaning"] = "row not in its record's top-8"
    (out / f"{split}_v2_features.json").write_text(json.dumps(info, indent=2) + "\n")
    log(json.dumps(info))


def evaluate_v2(a):
    """Standalone B_REST on dense rows only (comparable with v1) and on dense + R + U rows."""
    import pyarrow.parquet as pq

    from ber.arena_decision import evaluate as ev, one_owner, component_bootstrap

    out = Path(a.out)
    d = U / "prep/brest"
    comp = np.load(U / "data/arena/finalize_v1/component_ids.npy")
    part = np.load(U / "data/arena/finalize_v1/s1_partition.npy")
    true_count = np.load(U / "data/arena/labels/true_count.npy")
    country = np.asarray(pq.read_table(U / "data/arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    c2r = pq.read_table(U / "data/arena/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    owner = np.load(U / "data/arena/labels/rec_owner.npy")
    v2 = L / "data/v2"
    l0, r0, y0 = np.load(d / "left.npy").astype(np.int64), c2r[np.load(d / "right.npy")], np.load(d / "label.npy")
    rb = pq.read_table(v2 / "r_reverse_pairs_b_rest.parquet", columns=["left_idx", "raw_row_idx", "label"])
    ub = pq.read_table(v2 / "u_route_pairs_brest.parquet", columns=["left_idx", "right_idx"])
    l1, r1, y1 = rb.column("left_idx").to_numpy().astype(np.int64), rb.column("raw_row_idx").to_numpy(), rb.column("label").to_numpy()
    uri = ub.column("right_idx").to_numpy()
    l2, r2 = ub.column("left_idx").to_numpy().astype(np.int64), c2r[uri]
    y2 = owner[uri] == l2
    p0 = np.nan_to_num(np.load(out / "brest_lw_p.npy"), nan=0.0)
    p1 = np.nan_to_num(np.load(out / "brest_r_reverse_pairs_b_rest_lw_p.npy"), nan=0.0)
    p2 = np.nan_to_num(np.load(out / "brest_u_route_pairs_brest_lw_p.npy"), nan=0.0)
    pop = part == 1
    s1_half = (comp % 2).astype(np.int64)

    def rank_within(group, score):
        order = np.lexsort((np.arange(len(score)), -score, group))
        g = group[order]
        st = np.r_[True, g[1:] != g[:-1]]
        idx = np.arange(len(g))
        f = np.maximum.accumulate(np.where(st, idx, 0))
        rk = np.empty(len(g), dtype=np.int64)
        rk[order] = idx - f
        return rk

    def run(left, rec, label, p):
        half = s1_half[left]

        def dec(t):
            acc = one_owner(p >= t, left, rec, p)
            rows = np.flatnonzero(acc)
            acc[rows[rank_within(left[rows], p[rows]) >= 11]] = False
            return acc
        grid = {0: {}, 1: {}}
        for t in [round(x, 3) for x in np.arange(0.5, 0.951, 0.025)]:
            per = ev(dec(t), left, label, true_count, return_per_s1=True)["per_s1"]
            for h in (0, 1):
                grid[h][t] = float(per[pop & (s1_half == h)].mean())
        pick = {h: max(grid[h], key=grid[h].get) for h in (0, 1)}
        acc = (dec(pick[1]) & (half == 0)) | (dec(pick[0]) & (half == 1))
        res = ev(acc, left, label, true_count, population=pop, country=country, return_per_s1=True)
        per = res.pop("per_s1")
        slim = {k: res[k] for k in ("macro_f05", "precision", "recall", "false_links", "singleton_accuracy")}
        slim["by_country"] = {c: v["macro_f05"] for c, v in res["by_country"].items()}
        slim["picks"] = pick
        return slim, per
    rep = {}
    rep["v2_dense_rows"], per_d = run(l0, r0, y0, p0)
    L_ = np.concatenate([l0, l1, l2]); R_ = np.concatenate([r0, r1, r2])
    rep["v2_dense_r_u_rows"], per_all = run(L_, R_, np.concatenate([y0, y1, y2]), np.concatenate([p0, p1, p2]))
    per_v1 = np.load(L / "out/run1/per_s1_lw.npy")
    rep["v2_dense_minus_v1"] = component_bootstrap(per_v1, per_d, comp, n=a.bootstrap, seed=20260927, population=pop)
    rep["v2_all_minus_v2_dense"] = component_bootstrap(per_d, per_all, comp, n=a.bootstrap, seed=20260927, population=pop)
    rep["route_rows_true"] = int(y2.sum())
    (out / "brest_v2_report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    log(json.dumps(rep, default=float))


def kcheck2(a):
    """K-lite + v1 LW columns vs K-lite + v2 LW columns (brest_prep rows)."""
    import lightgbm as lgb
    import pyarrow.parquet as pq

    from ber.arena_decision import component_bootstrap, evaluate as ev, one_owner

    out = Path(a.out)
    d = U / "prep/brest"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    label = np.load(d / "label.npy")
    comp = np.load(U / "data/arena/finalize_v1/component_ids.npy")
    part = np.load(U / "data/arena/finalize_v1/s1_partition.npy")
    true_count = np.load(U / "data/arena/labels/true_count.npy")
    pop = part == 1
    half = (comp[left] % 2).astype(np.int64)
    s1_half = (comp % 2).astype(np.int64)
    X0 = np.hstack([np.load(d / "x_base.npy"), np.load(d / "x_q.npy"), np.load(L / "data/brest_k_features.npy")])
    v1 = np.load(L / "out/run1/brest_lw_features.npy")[:, :3]
    v2 = np.load(out / "brest_lw_features.npy")[:, :3]
    params = {"learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "verbose": -1, "seed": 0,
              "objective": "binary", "num_threads": a.threads}

    def rank_within(group, score):
        order = np.lexsort((np.arange(len(score)), -score, group))
        g = group[order]
        st = np.r_[True, g[1:] != g[:-1]]
        idx = np.arange(len(g))
        f = np.maximum.accumulate(np.where(st, idx, 0))
        rk = np.empty(len(g), dtype=np.int64)
        rk[order] = idx - f
        return rk

    def dec(p, t):
        acc = one_owner(p >= t, left, right, p)
        rows = np.flatnonzero(acc)
        acc[rows[rank_within(left[rows], p[rows]) >= 11]] = False
        return acc
    rep, pers = {}, {}
    for name, X in (("klite_lw_v1", None), ("klite_lw_v2", np.hstack([X0, v2])), ("klite_lw_v1v2", np.hstack([X0, v1, v2]))):
        path = (L / "out/run1/oof_klite_lw.npy") if name == "klite_lw_v1" else out / f"oof_{name}.npy"
        if path.exists():
            p = np.load(path).astype(np.float64)
        else:
            p = np.zeros(len(left))
            for h in (0, 1):
                tr, te = np.flatnonzero(half == h), np.flatnonzero(half != h)
                m = lgb.train(params, lgb.Dataset(X[tr], label[tr].astype(np.float32)), num_boost_round=500)
                p[te] = m.predict(X[te], num_threads=a.threads)
            np.save(path, p.astype(np.float32))
        grid = {0: {}, 1: {}}
        for t in [round(x, 3) for x in np.arange(0.6, 0.951, 0.025)]:
            per = ev(dec(p, t), left, label, true_count, return_per_s1=True)["per_s1"]
            for h in (0, 1):
                grid[h][t] = float(per[pop & (s1_half == h)].mean())
        pick = {h: max(grid[h], key=grid[h].get) for h in (0, 1)}
        acc = (dec(p, pick[1]) & (half == 0)) | (dec(p, pick[0]) & (half == 1))
        res = ev(acc, left, label, true_count, population=pop, return_per_s1=True)
        pers[name] = res.pop("per_s1")
        rep[name] = {k: res[k] for k in ("macro_f05", "precision", "recall", "false_links")}
        log(f"{name}: {res['macro_f05']:.6f}")
    rep["v2_minus_v1"] = component_bootstrap(pers["klite_lw_v1"], pers["klite_lw_v2"], comp, n=a.bootstrap, seed=20260927, population=pop)
    rep["v1v2_minus_v1"] = component_bootstrap(pers["klite_lw_v1"], pers["klite_lw_v1v2"], comp, n=a.bootstrap, seed=20260927, population=pop)
    (out / "kcheck2_report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    log(json.dumps(rep, default=float))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prep-arena", "prep-test", "train", "score", "export", "evaluate", "kcheck",
                                       "prep-v2", "export-v2", "evaluate-v2", "kcheck2"])
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--tok", default=str(L / "models/me5s"))
    p.add_argument("--prep-out", default="")
    p.add_argument("--v2-exclude", action="store_true")
    p.add_argument("--data", default=str(L / "data/arena"))
    p.add_argument("--model", default=str(L / "models/me5s"))
    p.add_argument("--out", default=str(L / "out/run1"))
    p.add_argument("--ckpt", default="")
    p.add_argument("--split", default="brest")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--subset", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--budget", type=int, default=24576, help="padded tokens per batch")
    p.add_argument("--max-multi", type=int, default=1_000_000)
    p.add_argument("--n-single", type=int, default=300_000)
    p.add_argument("--dup-heavy", type=int, default=2, help="extra copies of train groups with >= 5 claimants")
    p.add_argument("--max-minutes", type=float, default=90)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--baseline-only", action="store_true")
    a = p.parse_args()
    {"prep-arena": prep_arena, "prep-test": prep_test, "train": train, "score": score, "export": export,
     "evaluate": evaluate, "kcheck": kcheck, "prep-v2": prep_v2, "export-v2": export_v2, "evaluate-v2": evaluate_v2,
     "kcheck2": kcheck2}[a.command](a)


if __name__ == "__main__":
    main()
