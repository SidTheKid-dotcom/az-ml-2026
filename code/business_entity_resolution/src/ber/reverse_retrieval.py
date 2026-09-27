"""Reverse dense retrieval (record -> nearest same-country S1s) with the frozen retrieval encoder.

Forward retrieval gives each S1 its dense top-50 records. A true record can miss that list when the S1's
list is crowded (templated names, long chains) or when no S1 retrieved the record at all. Reverse retrieval
asks every record for its top-k S1s instead; pairs not already in the forward top-50 or the Track R routes
are new candidates.

Stages (one file; each stage runs where its inputs are):
  texts     (workstation) build the four text sets exactly as the original retrieval did:
            "query: {name} | {address}" for S1s, "passage: {name} | {address}" for records, raw TSVs read
            with dtype=str, keep_default_na=False, file order preserved (S2 then S3).
  tokenize  (CPU box) tokenize with the checkpoint's tokenizer.json, truncation 128, special tokens added.
  gpu       (H100) mean-pooled fp16 encoding (== sentence-transformers Pooling(mean)+Normalize), exact
            country-blocked cosine search in both directions: reverse top-K S1s per record and forward
            top-50 records per S1 (the latter only for a fidelity check against the stored forward pairs
            and for the S1's 50th forward cosine).
  post      (CPU) drop pairs already in forward top-50 / routes, measure on B_REST (B_REST labels only),
            export parquet.

No labels are read before `post`; `post` reads ground truth for B_REST S1s only (SLICE / HOLDOUT-2 never).
No IDs, row order or country enter any feature except the country block itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

COUNTRIES = {"US": 0, "India": 1, "France": 2}
COLS = ["entity_id", "business_name", "business_address", "country"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------- texts
def _texts(frame, prefix: str) -> list:
    # identical to ber.full_embeddings.embedding_texts
    return (prefix + frame.business_name.fillna("").astype(str) + " | "
            + frame.business_address.fillna("").astype(str)).tolist()


def _cc(series) -> np.ndarray:
    bad = set(series.unique()) - set(COUNTRIES)
    if bad:
        raise ValueError(f"unexpected country values {sorted(bad)[:5]}")
    return series.map(COUNTRIES).to_numpy(np.int8)


def stage_texts(args) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw, arena, out = Path(args.raw), Path(args.arena), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, ids, texts, cc):
        pq.write_table(pa.table({"entity_id": pa.array(ids, pa.string()), "text": pa.array(texts, pa.string()),
                                 "cc": pa.array(cc, pa.int8())}), out / f"{name}.parquet", compression="zstd")
        log(f"{name}: {len(ids):,} rows")

    # train S1 = arena reference (already read from raw with keep_default_na=False, raw S1 order)
    ref = pd.read_parquet(arena / "reference.parquet", columns=COLS)
    write("train_s1", ref.entity_id.tolist(), _texts(ref, "query: "), _cc(ref.country))
    del ref
    # test S1 = raw test_source1 (as ber.full_dense_inference read it); must equal dense_test reference order
    ref = pd.read_csv(raw / "test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False, usecols=COLS)
    dref = pd.read_parquet(args.test_reference, columns=["entity_id"])
    if not np.array_equal(ref.entity_id.to_numpy(), dref.entity_id.to_numpy()):
        raise ValueError("raw test S1 order != dense_test reference order")
    write("test_s1", ref.entity_id.tolist(), _texts(ref, "query: "), _cc(ref.country))
    del ref, dref
    for split in ("train", "test"):
        ids, texts, cc = [], [], []
        for src in (2, 3):
            for frame in pd.read_csv(raw / f"{split}/{split}_source{src}.tsv", sep="\t", dtype=str,
                                     keep_default_na=False, usecols=COLS, chunksize=500_000):
                ids.extend(frame.entity_id.tolist())
                texts.extend(_texts(frame, "passage: "))
                cc.append(_cc(frame.country))
        write(f"{split}_pool", ids, texts, np.concatenate(cc))
        del ids, texts, cc


# -------------------------------------------------------------------------------------------- tokenize
def stage_tokenize(args) -> None:
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=args.max_length)
    src, out = Path(args.texts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.sets:
        t0 = time.time()
        table = pq.read_table(src / f"{name}.parquet", columns=["text", "cc"])
        cc = table.column("cc").to_numpy().astype(np.int8)
        texts = table.column("text")
        n = len(cc)
        lengths = np.empty(n, np.int16)
        pieces = []
        step = 200_000
        for s in range(0, n, step):
            enc = tok.encode_batch(texts.slice(s, step).to_pylist(), add_special_tokens=True)
            lengths[s:s + len(enc)] = [len(e.ids) for e in enc]
            pieces.append(np.fromiter((i for e in enc for i in e.ids), np.int32))
        ids = np.concatenate(pieces)
        offsets = np.zeros(n + 1, np.int64)
        np.cumsum(lengths, out=offsets[1:])
        assert offsets[-1] == len(ids)
        np.save(out / f"{name}.ids.npy", ids)
        np.save(out / f"{name}.off.npy", offsets)
        np.save(out / f"{name}.cc.npy", cc)
        log(f"{name}: {n:,} texts, {len(ids):,} tokens, mean len {lengths.mean():.1f}, "
            f"max {lengths.max()}, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------- gpu
def _load_model(model_dir: str, device: str):
    import torch
    from transformers import AutoModel

    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32  # CPU only for smoke tests
    try:
        model = AutoModel.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval()


def _fast_batches(ids, off, order, lengths, max_tokens, pad_id):
    """Vectorised padded batch builder (avoids a Python loop per row)."""
    n = len(order)
    i = 0
    while i < n:
        L0 = int(lengths[order[i]])
        bs = max(1, min(8192, max_tokens // max(L0, 1)))
        rows = order[i:i + bs]
        L = int(lengths[rows].max())
        starts = off[rows]
        lens = lengths[rows]
        col = np.arange(L)
        valid = col[None, :] < lens[:, None]
        gather = np.where(valid, starts[:, None] + col[None, :], 0)
        batch = np.where(valid, ids[gather], pad_id).astype(np.int64)
        yield rows, batch, valid.astype(np.int64)
        i += len(rows)


def encode_fast(model, ids, off, device, max_tokens=131072, pad_id=1):
    import torch
    import threading
    import queue

    n = len(off) - 1
    lengths = np.diff(off).astype(np.int64)
    order = np.argsort(lengths, kind="stable")
    dim = model.config.hidden_size
    result = torch.empty((n, dim), dtype=torch.float16, device=device)
    q: "queue.Queue" = queue.Queue(maxsize=8)
    pin = str(device).startswith("cuda")

    def producer():
        for item in _fast_batches(ids, off, order, lengths, max_tokens, pad_id):
            q.put(tuple(torch.from_numpy(x).pin_memory() if pin else torch.from_numpy(x) for x in item))
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    t0 = time.time()
    done = 0
    last = t0
    with torch.inference_mode():
        while True:
            item = q.get()
            if item is None:
                break
            rows, bt, mt = (x.to(device, non_blocking=True) for x in item)
            hidden = model(input_ids=bt, attention_mask=mt).last_hidden_state
            m = mt.unsqueeze(-1).float()
            pooled = (hidden.float() * m).sum(1) / m.sum(1).clamp(min=1e-9)
            result[rows] = torch.nn.functional.normalize(pooled, p=2, dim=1).to(torch.float16)
            done += len(rows)
            if time.time() - last > 60:
                last = time.time()
                log(f"  {done:,}/{n:,} ({done / (last - t0):,.0f}/s)")
    th.join()
    if pin:
        torch.cuda.synchronize()
    log(f"encoded {n:,} in {time.time() - t0:.0f}s ({n / max(time.time() - t0, 1e-9):,.0f}/s)")
    return result


def blocked_topk(queries, keys, k: int, chunk: int):
    """Exact top-k by cosine (fp16 matmul, fp32 accumulate by cuBLAS) for fp16 unit vectors on GPU."""
    import torch

    nq = queries.shape[0]
    k = min(k, keys.shape[0])
    scores = torch.empty((nq, k), dtype=torch.float32, device=queries.device)
    idx = torch.empty((nq, k), dtype=torch.int32, device=queries.device)
    cpu = queries.device.type == "cpu"
    kt = keys.T.contiguous().float() if cpu else keys.T.contiguous()
    for s in range(0, nq, chunk):
        q = queries[s:s + chunk]
        sim = (q.float() if cpu else q) @ kt
        v, i = torch.topk(sim, k, dim=1, largest=True, sorted=True)
        scores[s:s + chunk] = v.float()
        idx[s:s + chunk] = i.int()
        del sim
    return scores, idx


def stage_gpu(args) -> None:
    import torch

    tok, out, dev = Path(args.tokens), Path(args.out), args.device
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _load_model(args.model, dev)
    report = {"model": args.model, "k_reverse": args.k, "k_forward": args.k_forward, "splits": {}}
    for split in args.splits:
        t_split = time.time()
        emb, cc = {}, {}
        for side in ("s1", "pool"):
            name = f"{split}_{side}"
            ids = np.load(tok / f"{name}.ids.npy")
            off = np.load(tok / f"{name}.off.npy")
            cc[side] = np.load(tok / f"{name}.cc.npy")
            log(f"encoding {name}: {len(off) - 1:,}")
            emb[side] = encode_fast(model, ids, off, dev, args.max_tokens)
            del ids, off
            if args.save_embeddings:
                np.save(out / f"{name}.emb.npy", emb[side].cpu().numpy())
        n_s1, n_pool = len(cc["s1"]), len(cc["pool"])
        rev_idx = np.full((n_pool, args.k), -1, np.int32)
        rev_score = np.full((n_pool, args.k), np.nan, np.float32)
        fwd_idx = np.full((n_s1, args.k_forward), -1, np.int32)
        fwd_score = np.full((n_s1, args.k_forward), np.nan, np.float32)
        t0 = time.time()
        for code in sorted(set(np.unique(cc["s1"])) & set(np.unique(cc["pool"]))):
            s_rows = np.flatnonzero(cc["s1"] == code)
            p_rows = np.flatnonzero(cc["pool"] == code)
            s_t = torch.from_numpy(s_rows).to(dev)
            p_t = torch.from_numpy(p_rows).to(dev)
            S = emb["s1"][s_t]
            P = emb["pool"][p_t]
            v, i = blocked_topk(P, S, args.k, args.rev_chunk)          # reverse: record -> S1
            kk = v.shape[1]
            rev_score[p_rows, :kk] = v.cpu().numpy()
            rev_idx[p_rows, :kk] = s_t[i.long()].int().cpu().numpy()
            del v, i
            v, i = blocked_topk(S, P, args.k_forward, args.fwd_chunk)  # forward: S1 -> record
            kk = v.shape[1]
            fwd_score[s_rows, :kk] = v.cpu().numpy()
            fwd_idx[s_rows, :kk] = p_t[i.long()].int().cpu().numpy()
            del v, i, S, P
            torch.cuda.empty_cache()
            log(f"{split} country {code}: S1 {len(s_rows):,} x records {len(p_rows):,} done "
                f"({time.time() - t0:.0f}s)")
        np.save(out / f"{split}_rev_idx.npy", rev_idx)
        np.save(out / f"{split}_rev_score.npy", rev_score.astype(np.float16) if args.fp16_scores else rev_score)
        np.save(out / f"{split}_fwd_idx.npy", fwd_idx)
        np.save(out / f"{split}_fwd_score.npy", fwd_score[:, [0, -1]])  # top-1 and 50th cosine
        report["splits"][split] = {"s1": n_s1, "pool": n_pool, "seconds": time.time() - t_split}
        del emb
        torch.cuda.empty_cache()
        (out / f"gpu_report_{'_'.join(args.splits)}.json").write_text(json.dumps(report, indent=1))
    log("GPU stage complete")


def stage_check(args) -> None:
    """Fidelity: our tokenizer+pooling vs sentence-transformers on a sample (run on the GPU box)."""
    import torch
    import pyarrow.parquet as pq
    from sentence_transformers import SentenceTransformer

    tok = Path(args.tokens)
    rng = np.random.default_rng(0)
    res = {}
    st = SentenceTransformer(args.model, device=args.device, local_files_only=True)
    st.max_seq_length = 128
    if str(args.device).startswith("cuda"):
        st.half()
    st.eval()
    model = _load_model(args.model, args.device)
    for name in args.sets:
        texts = pq.read_table(Path(args.texts) / f"{name}.parquet", columns=["text"]).column("text")
        ids = np.load(tok / f"{name}.ids.npy", mmap_mode="r")
        off = np.load(tok / f"{name}.off.npy")
        sel = np.sort(rng.choice(len(off) - 1, size=min(args.n, len(off) - 1), replace=False))
        sample = texts.take(sel).to_pylist()
        hf = st.tokenize(sample)["input_ids"]
        mism = 0
        for j, r in enumerate(sel):
            mine = np.asarray(ids[off[r]:off[r + 1]])
            theirs = hf[j][hf[j] != st.tokenizer.pad_token_id].numpy()
            mism += int(not np.array_equal(mine, theirs))
        sub_off = np.zeros(len(sel) + 1, np.int64)
        np.cumsum(np.diff(off)[sel], out=sub_off[1:])
        sub_ids = np.concatenate([np.asarray(ids[off[r]:off[r + 1]]) for r in sel])
        mine_e = encode_fast(model, sub_ids, sub_off, args.device).float().cpu().numpy()
        ref_e = st.encode(sample, batch_size=512, convert_to_numpy=True, normalize_embeddings=True)
        cos = (mine_e * ref_e).sum(1)
        res[name] = {"n": len(sel), "token_mismatch": mism, "cos_min": float(cos.min()),
                     "cos_mean": float(cos.mean())}
        log(f"check {name}: {res[name]}")
    Path(args.out).write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ post
def _in_rows(table: np.ndarray, rows: np.ndarray, values: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """values[i] in table[rows[i]] for a 2-D table (row-wise membership), chunked."""
    out = np.zeros(len(rows), bool)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = (table[rows[s:s + chunk]] == values[s:s + chunk, None]).any(1)
    return out


def _in_keys(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.zeros(len(keys), bool)
    pos = np.searchsorted(sorted_keys, keys)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    return sorted_keys[pos] == keys


def _oracle(found: np.ndarray, true: np.ndarray, cap: int = 11) -> np.ndarray:
    """Best-possible per-S1 F0.5: predict exactly the retrieved true links (at most `cap`)."""
    f = np.minimum(found, cap).astype(np.float64)
    t = true.astype(np.float64)
    return np.where(t == 0, 1.0, np.where(f == 0, 0.0, 5 * f / np.maximum(4 * f + t, 1e-12)))


def _rev_flat(rev_idx: np.ndarray, rev_score: np.ndarray, kmax: int):
    n, k = rev_idx.shape
    k = min(k, kmax)
    rec = np.repeat(np.arange(n, dtype=np.int64), k)
    s1 = rev_idx[:, :k].reshape(-1).astype(np.int64)
    rank = np.tile(np.arange(1, k + 1, dtype=np.int8), n)
    score = rev_score[:, :k].reshape(-1).astype(np.float32)
    top1 = np.repeat(rev_score[:, 0].astype(np.float32), k)
    ok = s1 >= 0
    return rec[ok], s1[ok], rank[ok], score[ok], top1[ok]


def _rules(rank, orphan, gap) -> dict:
    """Candidate pruning rules (label-free inputs only): reverse rank, orphan record (no S1 holds it in its
    stored forward top-50), gap = S1's 50th forward cosine - pair cosine."""
    r1 = rank <= 1
    return {
        "rank1": r1,
        "rank2": rank <= 2,
        "rank3": rank <= 3,
        "rank5": rank <= 5,
        "rank1|orphan<=3": r1 | (orphan & (rank <= 3)),
        "rank1|orphan<=5": r1 | (orphan & (rank <= 5)),
        "rank1|orphan<=10": r1 | (orphan & (rank <= 10)),
        "rank1|orphan<=20": r1 | (orphan & (rank <= 20)),
        "rank2|orphan<=10": (rank <= 2) | (orphan & (rank <= 10)),
        "rank1|gap<=0.01&rank<=5": r1 | ((gap <= 0.01) & (rank <= 5)),
        "rank1|orphan<=10|gap<=0.01&rank<=5": r1 | (orphan & (rank <= 10)) | ((gap <= 0.01) & (rank <= 5)),
        "rank1|orphan<=10|gap<=0.02&rank<=3": r1 | (orphan & (rank <= 10)) | ((gap <= 0.02) & (rank <= 3)),
    }


def _country_array(codes: np.ndarray):
    import pyarrow as pa

    names = [k for k, _ in sorted(COUNTRIES.items(), key=lambda kv: kv[1])]
    return pa.DictionaryArray.from_arrays(pa.array(codes.astype(np.int32)), pa.array(names)).cast(pa.string())


def stage_post(args) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    res, out, arena = Path(args.results), Path(args.out), Path(args.arena)
    out.mkdir(parents=True, exist_ok=True)
    report = {"k_max": args.k_max, "export_rule": {"rule": args.export_rule, "rank_max": args.export_rank_max,
                                                    "min_cosine": args.export_min_cos}}
    texts = Path(args.texts)

    def ids_of(name):
        return pq.read_table(texts / f"{name}.parquet", columns=["entity_id"]).column("entity_id").to_numpy(
            zero_copy_only=False)

    # ------------------------------------------------------------------ arena (train) side
    if "train" in args.splits:
        log("train: loading arena keys")
        part = np.load(arena / "finalize_v1/s1_partition.npy")
        n_s1 = len(part)
        ref = pq.read_table(arena / "v1/reference.parquet", columns=["entity_id", "country"])
        s1_ids = ref.column("entity_id").to_numpy(zero_copy_only=False)
        s1_country = np.asarray(ref.column("country").to_pylist(), dtype=object)
        s1_cc = np.array([COUNTRIES[c] for c in s1_country], np.int8)
        if not np.array_equal(s1_ids, ids_of("train_s1")):
            raise ValueError("train_s1 text order != arena reference order")
        rec_ids = ids_of("train_pool")
        n_raw = len(rec_ids)
        compact_raw = pq.read_table(arena / "v1/pool.parquet", columns=["raw_row_idx"]).column(0).to_numpy()
        raw_to_compact = np.full(n_raw, -1, np.int64)
        raw_to_compact[compact_raw] = np.arange(len(compact_raw))
        _smp = np.random.default_rng(1).choice(len(compact_raw), 100_000, replace=False)
        _cid = pq.read_table(arena / "v1/pool.parquet", columns=["entity_id"]).column(0).take(_smp).to_pylist()
        if list(rec_ids[compact_raw[_smp]]) != _cid:
            raise ValueError("arena pool raw_row_idx does not match raw train pool order")
        t = pq.read_table(arena / "v1/pairs.parquet", columns=["left_idx", "right_raw_idx"])
        left = t.column("left_idx").to_numpy()
        if len(left) != n_s1 * 50 or not np.array_equal(left[::50], np.arange(n_s1)) or \
                not np.array_equal(left[49::50], np.arange(n_s1)):
            raise ValueError("arena pairs are not 50 per S1 in reference order")
        fwd = t.column("right_raw_idx").to_numpy().astype(np.int32).reshape(n_s1, 50)
        del t, left
        # label-free record context: how many S1s hold the record in their stored forward top-50
        claimants = np.bincount(fwd.reshape(-1), minlength=n_raw).astype(np.int32)

        # truth for A and B_REST S1s only (SLICE never; HOLDOUT-2 S1s are not in the arena)
        log("train: reading ground truth for A/B_REST S1s")
        gt = pd.read_csv(Path(args.raw) / "train/train_ground_truth.tsv", sep="\t", dtype=str,
                         keep_default_na=False)
        srow = pd.Index(s1_ids).get_indexer(gt.source1_entity_id.to_numpy())
        keep = srow >= 0
        keep[keep] = part[srow[keep]] != 2
        gt, srow = gt.loc[keep], srow[keep]
        lists = gt.matched_entity_ids.str.split(",")
        lens = lists.str.len().to_numpy()
        flat = np.concatenate(lists.to_numpy()) if len(lists) else np.array([], object)
        t_s1 = np.repeat(srow, lens)
        nonempty = flat != ""
        t_s1, flat = t_s1[nonempty], flat[nonempty]
        t_rec = pd.Index(rec_ids).get_indexer(flat)
        if (t_rec < 0).any():
            raise ValueError(f"{int((t_rec < 0).sum())} true records not in the raw train pool")
        true_count = np.bincount(t_s1, minlength=n_s1)
        arena_tc = np.load(arena / "labels/true_count.npy")
        chk = part != 2
        if not np.array_equal(true_count[chk], arena_tc[chk].astype(true_count.dtype)):
            raise ValueError("ground-truth counts differ from arena true_count")
        true_keys = np.sort(t_s1.astype(np.int64) * n_raw + t_rec)
        del gt, lists, flat

        # routes (B_REST only), keyed by entity ids
        rt = pq.read_table(args.routes_brest, columns=["s1_id", "rec_id"])
        r_s1 = pd.Index(s1_ids).get_indexer(rt.column("s1_id").to_numpy(zero_copy_only=False))
        r_rec = pd.Index(rec_ids).get_indexer(rt.column("rec_id").to_numpy(zero_copy_only=False))
        if (r_s1 < 0).any() or (r_rec < 0).any():
            raise ValueError("route ids not found")
        route_keys = np.unique(r_s1.astype(np.int64) * n_raw + r_rec)

        # ---- baseline found per S1 (B_REST)
        in_fwd_true = _in_rows(fwd, t_s1, t_rec)
        in_route_true = _in_keys(route_keys, t_s1.astype(np.int64) * n_raw + t_rec)
        brest = part == 1

        def summarize(found_per_s1, label):
            rows = {}
            for c in ("ALL", "US", "India"):
                m = brest if c == "ALL" else brest & (s1_cc == COUNTRIES[c])
                T = true_count[m].sum()
                rows[c] = {"recall": float(found_per_s1[m].sum() / T),
                           "oracle_f05": float(_oracle(found_per_s1[m], true_count[m]).mean()),
                           "s1": int(m.sum()), "true_links": int(T), "found": int(found_per_s1[m].sum())}
            log(f"  {label}: " + "  ".join(f"{c} R={v['recall']:.5f} F*={v['oracle_f05']:.5f}"
                                          for c, v in rows.items()))
            return rows

        f_dense = np.bincount(t_s1[in_fwd_true], minlength=n_s1)
        f_route = np.bincount(t_s1[in_fwd_true | in_route_true], minlength=n_s1)
        report["b_rest"] = {"dense_top50": summarize(f_dense, "dense top-50"),
                            "dense+routes": summarize(f_route, "dense+routes")}

        # ---- reverse pairs
        log("train: flattening reverse top-k")
        rev_idx = np.load(res / "train_rev_idx.npy", mmap_mode="r")
        rev_score = np.load(res / "train_rev_score.npy", mmap_mode="r")
        rec, s1, rank, score, top1 = _rev_flat(rev_idx, rev_score, args.k_max)
        keep = part[s1] != 2  # SLICE S1 pairs dropped entirely
        rec, s1, rank, score, top1 = rec[keep], s1[keep], rank[keep], score[keep], top1[keep]
        in_fwd = _in_rows(fwd, s1, rec.astype(np.int32))
        keys = s1 * n_raw + rec
        in_route = _in_keys(route_keys, keys)
        new = ~in_fwd & ~in_route
        report["train_reverse_all"] = {"pairs": int(len(s1)), "in_forward_top50": int(in_fwd.sum()),
                                       "in_routes": int((in_route & ~in_fwd).sum()), "new": int(new.sum())}
        rec, s1, rank, score, top1, keys = rec[new], s1[new], rank[new], score[new], top1[new], keys[new]
        label = _in_keys(true_keys, keys)
        del in_fwd, in_route, new
        fwd_sc = np.load(res / "train_fwd_score.npy", mmap_mode="r")
        s1_fwd50 = np.asarray(fwd_sc[:, -1], np.float32)

        # ---- measurement grid on B_REST
        is_b = part[s1] == 1
        grid = {}
        for kmax in [1, 2, 3, 5, 10, 20]:
            if kmax > args.k_max:
                continue
            for mincos in [None, 0.80, 0.85, 0.90]:
                m = is_b & (rank <= kmax)
                if mincos is not None:
                    m &= score >= mincos
                f = f_route + np.bincount(s1[m & label], minlength=n_s1)
                name = f"rank<={kmax}" + ("" if mincos is None else f",cos>={mincos}")
                grid[name] = {"new_pairs": int(m.sum()), "new_true": int((m & label).sum()),
                              "pairs_per_b_rest_s1": float(m.sum() / brest.sum()),
                              "true_rate": float((m & label).sum() / max(m.sum(), 1)),
                              "metrics": summarize(f, name)}
        report["b_rest"]["dense+routes+reverse"] = grid
        # per-rank true rates and the out-of-compact-pool share
        per_rank = {}
        for r in range(1, min(args.k_max, 20) + 1):
            m = is_b & (rank == r)
            per_rank[r] = {"pairs": int(m.sum()), "true": int((m & label).sum()),
                           "true_outside_compact_pool": int((m & label & (raw_to_compact[rec] < 0)).sum()),
                           "true_record_no_fwd_claimant": int((m & label & (claimants[rec] == 0)).sum())}
        report["b_rest"]["per_rank_new"] = per_rank
        # missed links that the reverse search (k_max) still does not reach
        found_any = in_fwd_true | in_route_true | _in_keys(np.sort(keys[label]), t_s1.astype(np.int64) * n_raw + t_rec)
        bt = brest[t_s1]
        report["b_rest"]["still_missed_links"] = int((bt & ~found_any).sum())

        # ---- pruning rules on B_REST
        orphan = claimants[rec] == 0
        gap = s1_fwd50[s1] - score
        rules = _rules(rank, orphan, gap)
        rule_eval = {}
        for name, mask in rules.items():
            m = is_b & mask
            f = f_route + np.bincount(s1[m & label], minlength=n_s1)
            rule_eval[name] = {"new_pairs": int(m.sum()), "new_true": int((m & label).sum()),
                               "pairs_per_b_rest_s1": float(m.sum() / brest.sum()),
                               "true_rate": float((m & label).sum() / max(m.sum(), 1)),
                               "metrics": summarize(f, name)}
        report["b_rest"]["rules"] = rule_eval
        # ---- export (A and B_REST, rule applied)
        if args.export_rule:
            m_exp = rules[args.export_rule].copy()
        else:
            m_exp = rank <= args.export_rank_max
        if args.export_min_cos is not None:
            m_exp &= score >= args.export_min_cos
        for pname, code in (("b_rest", 1), ("a", 0)):
            m = m_exp & (part[s1] == code)
            frame = pa.table({
                "s1_id": pa.array(s1_ids[s1[m]], pa.string()), "rec_id": pa.array(rec_ids[rec[m]], pa.string()),
                "left_idx": s1[m].astype(np.int64), "right_idx": raw_to_compact[rec[m]],
                "raw_row_idx": rec[m].astype(np.int64),
                "country": _country_array(s1_cc[s1[m]]),
                "rev_rank": rank[m].astype(np.int8), "cosine": score[m],
                "rec_top1_cos": top1[m], "s1_fwd50_cos": s1_fwd50[s1[m]],
                "rec_fwd_claimants": claimants[rec[m]], "label": label[m]})
            path = out / f"r_reverse_pairs_{pname}.parquet"
            pq.write_table(frame, path, compression="zstd")
            report[f"export_{pname}"] = {"rows": frame.num_rows, "true": int(label[m].sum()),
                                         "path": path.name, "sha256": sha256(path)}
            log(f"exported {path.name}: {frame.num_rows:,} rows, {int(label[m].sum()):,} true")
        # fidelity: recomputed forward top-50 vs stored forward top-50 (sample of S1s)
        fwd_new = np.load(res / "train_fwd_idx.npy", mmap_mode="r")
        rng = np.random.default_rng(0)
        smp = np.sort(rng.choice(n_s1, 20000, replace=False))
        a_ = np.asarray(fwd_new[smp])
        b_ = fwd[smp]
        overlap = np.mean([len(np.intersect1d(x, y)) / 50 for x, y in zip(a_, b_)])
        report["fidelity_train_forward_top50_overlap"] = float(overlap)
        log(f"fidelity: forward top-50 overlap (recomputed vs stored) = {overlap:.5f}")
        del rec, s1, rank, score, top1, keys, label, fwd

    # ------------------------------------------------------------------ test side
    if "test" in args.splits:
        log("test: loading keys")
        tref = pq.read_table(args.test_dir + "/reference.parquet", columns=["entity_id", "country"])
        s1_ids = tref.column("entity_id").to_numpy(zero_copy_only=False)
        s1_country = np.asarray(tref.column("country").to_pylist(), dtype=object)
        s1_cc = np.array([COUNTRIES[c] for c in s1_country], np.int8)
        if not np.array_equal(s1_ids, ids_of("test_s1")):
            raise ValueError("test_s1 order != dense_test reference order")
        n_s1 = len(s1_ids)
        rec_ids = ids_of("test_pool")
        n_raw = len(rec_ids)
        dense_raw = pq.read_table(args.test_dir + "/pool.parquet", columns=["raw_row_idx"]).column(0).to_numpy()
        raw_to_dense = np.full(n_raw, -1, np.int64)
        raw_to_dense[dense_raw] = np.arange(len(dense_raw))
        _smp = np.random.default_rng(1).choice(len(dense_raw), 100_000, replace=False)
        _did = pq.read_table(args.test_dir + "/pool.parquet", columns=["entity_id"]).column(0).take(_smp).to_pylist()
        if list(rec_ids[dense_raw[_smp]]) != _did:
            raise ValueError("dense_test pool raw_row_idx does not match raw test pool order")
        left = np.load(args.test_dir + "/left.npy", mmap_mode="r")
        if len(left) != n_s1 * 50 or not np.array_equal(left[::50], np.arange(n_s1)):
            raise ValueError("test index is not 50 per S1 in reference order")
        fwd = dense_raw[np.load(args.test_dir + "/right.npy")].astype(np.int32).reshape(n_s1, 50)
        claimants = np.bincount(fwd.reshape(-1), minlength=n_raw).astype(np.int32)
        rt = pq.read_table(args.routes_test, columns=["s1_id", "rec_id"])
        r_s1 = pd.Index(s1_ids).get_indexer(rt.column("s1_id").to_numpy(zero_copy_only=False))
        r_rec = pd.Index(rec_ids).get_indexer(rt.column("rec_id").to_numpy(zero_copy_only=False))
        if (r_s1 < 0).any() or (r_rec < 0).any():
            raise ValueError("test route ids not found")
        route_keys = np.unique(r_s1.astype(np.int64) * n_raw + r_rec)
        all_route_keys = np.array([], np.int64)
        if args.routes_test_all:
            d = Path(args.routes_test_all)
            la, ra = np.load(d / "left.npy"), np.load(d / "right.npy")
            set_raw = pq.read_table(d / "pool.parquet", columns=["raw_row_idx"]).column(0).to_numpy()
            all_route_keys = np.unique(la.astype(np.int64) * n_raw + set_raw[ra])
        rev_idx = np.load(res / "test_rev_idx.npy", mmap_mode="r")
        rev_score = np.load(res / "test_rev_score.npy", mmap_mode="r")
        rec, s1, rank, score, top1 = _rev_flat(rev_idx, rev_score, args.k_max)
        in_fwd = _in_rows(fwd, s1, rec.astype(np.int32))
        keys = s1 * n_raw + rec
        in_route = _in_keys(route_keys, keys)
        new = ~in_fwd & ~in_route
        report["test_reverse_all"] = {"pairs": int(len(s1)), "in_forward_top50": int(in_fwd.sum()),
                                      "in_routes": int((in_route & ~in_fwd).sum()), "new": int(new.sum())}
        rec, s1, rank, score, top1, keys = rec[new], s1[new], rank[new], score[new], top1[new], keys[new]
        in_all_routes = _in_keys(all_route_keys, keys)
        fwd_sc = np.load(res / "test_fwd_score.npy", mmap_mode="r")
        s1_fwd50 = np.asarray(fwd_sc[:, -1], np.float32)
        cnt = {}
        pair_cc = s1_cc[s1]
        outside = raw_to_dense[rec] < 0
        for kmax in [1, 2, 3, 5, 10, 20]:
            if kmax > args.k_max:
                continue
            m = rank <= kmax
            cnt[f"rank<={kmax}"] = {c: {"new_pairs": int((m & (pair_cc == COUNTRIES[c])).sum()),
                                        "per_s1": float((m & (pair_cc == COUNTRIES[c])).sum()
                                                        / (s1_cc == COUNTRIES[c]).sum()),
                                        "to_records_outside_dense_pool": int(
                                            (m & (pair_cc == COUNTRIES[c]) & outside).sum())}
                                    for c in ("US", "India", "France")}
        report["test_new_pairs"] = cnt
        orphan = claimants[rec] == 0
        gap = s1_fwd50[s1] - score
        rules = _rules(rank, orphan, gap)
        report["test_rules"] = {name: {c: {"new_pairs": int((mask & (pair_cc == COUNTRIES[c])).sum()),
                                           "per_s1": float((mask & (pair_cc == COUNTRIES[c])).sum()
                                                           / (s1_cc == COUNTRIES[c]).sum())}
                                       for c in ("US", "India", "France")} for name, mask in rules.items()}
        m = rules[args.export_rule].copy() if args.export_rule else rank <= args.export_rank_max
        if args.export_min_cos is not None:
            m &= score >= args.export_min_cos
        frame = pa.table({
            "s1_id": pa.array(s1_ids[s1[m]], pa.string()), "rec_id": pa.array(rec_ids[rec[m]], pa.string()),
            "left_idx": s1[m].astype(np.int64), "right_idx": raw_to_dense[rec[m]],
            "raw_row_idx": rec[m].astype(np.int64),
            "country": _country_array(pair_cc[m]),
            "rev_rank": rank[m].astype(np.int8), "cosine": score[m],
            "rec_top1_cos": top1[m], "s1_fwd50_cos": s1_fwd50[s1[m]],
            "rec_fwd_claimants": claimants[rec[m]], "in_track_r_all_routes": in_all_routes[m]})
        path = out / "r_reverse_pairs_test.parquet"
        pq.write_table(frame, path, compression="zstd")
        report["export_test"] = {"rows": frame.num_rows, "path": path.name, "sha256": sha256(path),
                                 "by_country": {c: int((pair_cc[m] == COUNTRIES[c]).sum()) for c in ("US", "India", "France")}}
        log(f"exported {path.name}: {frame.num_rows:,} rows")
        fwd_new = np.load(res / "test_fwd_idx.npy", mmap_mode="r")
        rng = np.random.default_rng(0)
        smp = np.sort(rng.choice(n_s1, 20000, replace=False))
        overlap = np.mean([len(np.intersect1d(x, y)) / 50 for x, y in zip(np.asarray(fwd_new[smp]), fwd[smp])])
        report["fidelity_test_forward_top50_overlap"] = float(overlap)
        log(f"fidelity test: forward top-50 overlap = {overlap:.5f}")
    (out / f"r_report_{'_'.join(args.splits)}.json").write_text(json.dumps(report, indent=1, default=str))
    log("post complete")


# ------------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("texts")
    a.add_argument("--raw", required=True)
    a.add_argument("--arena", required=True)
    a.add_argument("--test-reference", required=True)
    a.add_argument("--out", required=True)
    a = sub.add_parser("tokenize")
    a.add_argument("--model", required=True)
    a.add_argument("--texts", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--max-length", type=int, default=128)
    a.add_argument("--sets", nargs="+", default=["train_s1", "test_s1", "train_pool", "test_pool"])
    a = sub.add_parser("gpu")
    a.add_argument("--model", required=True)
    a.add_argument("--tokens", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--splits", nargs="+", default=["train", "test"])
    a.add_argument("--k", type=int, default=20)
    a.add_argument("--k-forward", type=int, default=50)
    a.add_argument("--rev-chunk", type=int, default=4096)
    a.add_argument("--fwd-chunk", type=int, default=512)
    a.add_argument("--max-tokens", type=int, default=131072)
    a.add_argument("--device", default="cuda")
    a.add_argument("--fp16-scores", action="store_true")
    a.add_argument("--save-embeddings", action="store_true")
    a = sub.add_parser("check")
    a.add_argument("--model", required=True)
    a.add_argument("--tokens", required=True)
    a.add_argument("--texts", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--sets", nargs="+", default=["train_s1", "test_pool"])
    a.add_argument("--n", type=int, default=4000)
    a.add_argument("--device", default="cuda")
    a = sub.add_parser("post")
    a.add_argument("--results", required=True)
    a.add_argument("--texts", required=True)
    a.add_argument("--arena", required=True)
    a.add_argument("--raw", required=True, help="dir containing train/train_ground_truth.tsv")
    a.add_argument("--routes-brest", required=True)
    a.add_argument("--routes-test", required=True)
    a.add_argument("--routes-test-all", help="u8src/test dir (left/right/pool.parquet): all Track R routes")
    a.add_argument("--test-dir", required=True, help="dir with dense_test reference/pool parquet + index left/right")
    a.add_argument("--out", required=True)
    a.add_argument("--splits", nargs="+", default=["train", "test"])
    a.add_argument("--k-max", type=int, default=20)
    a.add_argument("--export-rank-max", type=int, default=3)
    a.add_argument("--export-min-cos", type=float)
    a.add_argument("--export-rule", help="name from _rules(); overrides --export-rank-max")
    args = p.parse_args()
    {"texts": stage_texts, "tokenize": stage_tokenize, "gpu": stage_gpu, "check": stage_check,
     "post": stage_post}[args.stage](args)


if __name__ == "__main__":
    main()
