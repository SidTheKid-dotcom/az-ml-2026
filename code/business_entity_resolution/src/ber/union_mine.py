"""Key-branch union retrieval beyond the dense E5 top-50 (arena B_REST and test).

`mine` (label-free): every S1 of the set is matched against the *full* S2/S3
pool of its split (raw TSV rows), country-blocked, through eight exact-key
branches — normalized name, harmonized phonetic name, address digit sequence,
address digit set, core name + digit sequence (composite key), core name +
digit set, first core token + digit sequence, first core token + digit set.
A branch posting larger than its cap is skipped for that S1 (counted). Pairs
already in the dense set are removed. Each remaining pair gets a cheap rank
score, 0.7·WRatio(core names) + 0.3·token_set_ratio(addresses) + 0.01 per
branch hit, and the best ``--keep`` per S1 are saved with their rank.

`evaluate` (B_REST only): reads the training truth for B_REST S1 IDs only
(other payloads are never split), checks it reproduces ``label.npy`` on the
dense B_REST pairs, and reports per k: extra pairs, new true links, recall
ceiling and oracle macro F0.5 delta with a component bootstrap.

`export`: a scorable set for one k — reference.parquet, pool.parquet (new
records only), left.npy, right.npy, tree.npy of ones (the gate is bypassed),
and pairs.parquet (raw rows, rank, score, branch bits). No truth is exported.
True links are never injected: candidates come from text keys only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
BRANCHES = ("name", "phonetic", "digit_seq", "digit_set", "core_seq", "core_set", "tok_seq", "tok_set")
CAPS = {"name": 64, "phonetic": 64, "digit_seq": 64, "digit_set": 32,
        "core_seq": 256, "core_set": 256, "tok_seq": 128, "tok_set": 128}
SHIFT = np.int64(24)  # raw pool rows < 2**24
ARENA = Path("/workspace/az-ml-new/arena")

_G: dict = {}


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def read_raw(path: Path) -> pd.DataFrame:
    """Same parser settings as arena_build.read_raw / v1's full dense pool."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=RECORD_COLUMNS)[RECORD_COLUMNS]


def load_pool(raw_dir: Path, split: str) -> pd.DataFrame:
    frames = [read_raw(raw_dir / f"{split}_source{s}.tsv") for s in (2, 3)]
    pool = pd.concat(frames, ignore_index=True)
    if pool.entity_id.duplicated().any():
        raise ValueError("Duplicate pool entity IDs")
    return pool


def _key_worker(bounds):
    from ber.v2_lexical_mining import text_keys
    start, stop = bounds
    names, addrs, countries = _G["names"], _G["addrs"], _G["countries"]
    n = stop - start
    cols = {b: [""] * n for b in BRANCHES}
    core, addr = [""] * n, [""] * n
    for i in range(n):
        k = text_keys(names[start + i], addrs[start + i], countries[start + i])
        c = k.country
        first = k.core.split()[0] if k.core else ""
        core[i], addr[i] = k.core or k.name, k.address
        if k.name:
            cols["name"][i] = f"{c}\x1f{k.name}"
        if k.phonetic:
            cols["phonetic"][i] = f"{c}\x1f{k.phonetic}"
        if k.digit_seq:
            cols["digit_seq"][i] = f"{c}\x1f{k.digit_seq}"
            if k.core:
                cols["core_seq"][i] = f"{c}\x1f{k.core}\x1f{k.digit_seq}"
                cols["tok_seq"][i] = f"{c}\x1f{first}\x1f{k.digit_seq}"
        if k.digit_set:
            cols["digit_set"][i] = f"{c}\x1f{k.digit_set}"
            if k.core:
                cols["core_set"][i] = f"{c}\x1f{k.core}\x1f{k.digit_set}"
                cols["tok_set"][i] = f"{c}\x1f{first}\x1f{k.digit_set}"
    hashes = {}
    for b in BRANCHES:
        values = np.asarray(cols[b], dtype=object)
        h = pd.util.hash_array(values, categorize=False).astype(np.uint64)
        h[values == ""] = 0
        h[h == 0] = 0  # 0 is the empty-key sentinel
        hashes[b] = h
    return start, core, addr, hashes


def compute_keys(frame: pd.DataFrame, workers: int, chunk: int = 100_000) -> dict:
    _G["names"] = frame.business_name.tolist()
    _G["addrs"] = frame.business_address.tolist()
    _G["countries"] = frame.country.tolist()
    n = len(frame)
    core = np.empty(n, dtype=object)
    addr = np.empty(n, dtype=object)
    hashes = {b: np.zeros(n, dtype=np.uint64) for b in BRANCHES}
    bounds = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]
    with mp.get_context("fork").Pool(workers) as pool:
        for start, c, a, h in pool.imap_unordered(_key_worker, bounds):
            stop = start + len(c)
            core[start:stop] = c
            addr[start:stop] = a
            for b in BRANCHES:
                hashes[b][start:stop] = h[b]
    _G.clear()
    return {"core": core, "addr": addr, "hashes": hashes}


def branch_join(hs: np.ndarray, hp: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """All (s1, pool) pairs with equal nonzero hash; S1s whose posting exceeds ``cap`` are skipped."""
    valid = np.flatnonzero(hp != 0)
    order = valid[np.argsort(hp[valid], kind="stable")]
    sorted_h = hp[order]
    lo = np.searchsorted(sorted_h, hs, side="left")
    hi = np.searchsorted(sorted_h, hs, side="right")
    count = (hi - lo).astype(np.int64)
    count[hs == 0] = 0
    posting = count.copy()
    over = count > cap
    stats = {"s1_with_key": int((hs != 0).sum()), "s1_hit": int(((count > 0) & ~over).sum()),
             "s1_over_cap": int(over.sum())}
    count[over] = 0
    s1 = np.repeat(np.arange(len(hs), dtype=np.int64), count)
    offsets = np.repeat(lo - np.cumsum(count) + count, count) + np.arange(int(count.sum()), dtype=np.int64)
    raw = order[offsets].astype(np.int64)
    stats["pairs"] = int(len(s1))
    stats["posting"] = posting
    return s1, raw, stats


def mine(args) -> dict:
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw)
    if args.set in ("b_rest", "a"):
        partition = np.load(ARENA / "finalize_v1" / "s1_partition.npy")
        code = 1 if args.set == "b_rest" else 0  # B_REST or training partition A; SLICE (2) never
        s1_arena = np.flatnonzero(partition == code).astype(np.int64)
        reference = pd.read_parquet(ARENA / "v1" / "reference.parquet", columns=RECORD_COLUMNS)
        reference = reference.iloc[s1_arena].reset_index(drop=True)
        reference["arena_idx"] = s1_arena
        local = np.full(len(partition), -1, dtype=np.int64)
        local[s1_arena] = np.arange(len(s1_arena))
        table = pq.read_table(ARENA / "v1" / "pairs.parquet", columns=["left_idx", "right_raw_idx"])
        left = table.column("left_idx").to_numpy().astype(np.int64)
        rraw = table.column("right_raw_idx").to_numpy().astype(np.int64)
        del table
        keep = local[left] >= 0
        dense_keys = (local[left[keep]] << SHIFT) | rraw[keep]
        del left, rraw, keep
        split = "train"
        compact = pd.read_parquet(ARENA / "v1" / "pool.parquet", columns=["entity_id", "raw_row_idx"])
    else:
        test_inputs = Path(args.test_inputs)
        reference = pd.read_parquet(test_inputs / "dense_test" / "reference.parquet", columns=RECORD_COLUMNS)
        compact = pd.read_parquet(test_inputs / "dense_test" / "pool.parquet", columns=["entity_id", "raw_row_idx"])
        left = np.load(test_inputs / "index" / "left.npy").astype(np.int64)
        right = np.load(test_inputs / "index" / "right.npy").astype(np.int64)
        dense_keys = (left << SHIFT) | compact.raw_row_idx.to_numpy()[right]
        del left, right
        split = "test"
    dense_keys = np.unique(dense_keys)
    log(f"{args.set}: {len(reference):,} S1s, {len(dense_keys):,} dense pairs")
    pool = load_pool(raw_dir, split)
    if len(pool) >= 1 << 24:
        raise ValueError("pool too large for the pair key")
    raw_ids = compact.raw_row_idx.to_numpy()
    if not np.array_equal(pool.entity_id.to_numpy()[raw_ids], compact.entity_id.to_numpy()):
        raise ValueError("Raw pool rows do not reproduce the compact dense pool IDs")
    log(f"full {split} pool: {len(pool):,} raw rows (compact dense pool {len(compact):,} verified)")
    del compact
    t0 = time.time()
    ref_keys = compute_keys(reference, args.workers)
    pool_keys = compute_keys(pool, args.workers)
    log(f"keys in {time.time() - t0:.0f}s")
    all_keys, all_bits, branch_stats = [], [], {}
    postings = np.zeros((len(reference), len(BRANCHES)), dtype=np.int32)
    for bit, branch in enumerate(BRANCHES):
        s1, raw, stats = branch_join(ref_keys["hashes"][branch], pool_keys["hashes"][branch], CAPS[branch])
        postings[:, bit] = np.minimum(stats.pop("posting"), np.iinfo(np.int32).max)
        all_keys.append((s1 << SHIFT) | raw)
        all_bits.append(np.full(len(s1), 1 << bit, dtype=np.uint8))
        branch_stats[branch] = stats
        log(f"{branch}: {stats}")
    keys = np.concatenate(all_keys)
    bits = np.concatenate(all_bits)
    del all_keys, all_bits
    order = np.argsort(keys, kind="stable")
    keys, bits = keys[order], bits[order]
    first = np.ones(len(keys), dtype=bool)
    first[1:] = keys[1:] != keys[:-1]
    starts = np.flatnonzero(first)
    bits = np.bitwise_or.reduceat(bits, starts)
    keys = keys[starts]
    total_union = len(keys)
    pos = np.searchsorted(dense_keys, keys)
    pos[pos >= len(dense_keys)] = len(dense_keys) - 1
    in_dense = dense_keys[pos] == keys
    keys, bits = keys[~in_dense], bits[~in_dense]
    log(f"key union {total_union:,} pairs; {int(in_dense.sum()):,} already dense; {len(keys):,} new")
    s1 = (keys >> SHIFT).astype(np.int64)
    raw = (keys & ((1 << 24) - 1)).astype(np.int64)
    del keys
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    t0 = time.time()
    name = cpdist(ref_keys["core"][s1], pool_keys["core"][raw], scorer=fuzz.WRatio,
                  workers=args.workers, dtype=np.float32)
    address = cpdist(ref_keys["addr"][s1], pool_keys["addr"][raw], scorer=fuzz.token_set_ratio,
                     workers=args.workers, dtype=np.float32)
    extra = {}
    if args.features:
        for col, a, b, scorer in (("name_ratio", "core", "core", fuzz.ratio),
                                  ("name_tset", "core", "core", fuzz.token_set_ratio),
                                  ("name_partial", "core", "core", fuzz.partial_ratio),
                                  ("address_ratio", "addr", "addr", fuzz.ratio),
                                  ("address_tsort", "addr", "addr", fuzz.token_sort_ratio)):
            extra[col] = cpdist(ref_keys[a][s1], pool_keys[b][raw], scorer=scorer, workers=args.workers,
                                dtype=np.float32)
        lens = np.frompyfunc(len, 1, 1)
        s_nl, p_nl = lens(ref_keys["core"]).astype(np.int16), lens(pool_keys["core"]).astype(np.int16)
        s_al, p_al = lens(ref_keys["addr"]).astype(np.int16), lens(pool_keys["addr"]).astype(np.int16)
        extra["s1_name_len"], extra["rec_name_len"] = s_nl[s1], p_nl[raw]
        extra["s1_addr_len"], extra["rec_addr_len"] = s_al[s1], p_al[raw]
    hits = np.unpackbits(bits[:, None], axis=1).sum(axis=1).astype(np.float32)
    score = (0.7 * name + 0.3 * address) / 100 + 0.01 * hits
    log(f"rank scores in {time.time() - t0:.0f}s")
    order = np.lexsort((raw, -score, s1))
    s1, raw, score, bits, name, address = (x[order] for x in (s1, raw, score, bits, name, address))
    extra = {k: v[order] for k, v in extra.items()}
    first = np.ones(len(s1), dtype=bool)
    first[1:] = s1[1:] != s1[:-1]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(s1)), 0))
    rank = (np.arange(len(s1)) - group_start).astype(np.int32)
    keep = rank < args.keep if args.keep > 0 else np.ones(len(rank), dtype=bool)
    frame = pd.DataFrame({"s1": s1[keep].astype(np.int32), "raw": raw[keep].astype(np.int32),
                          "rank": rank[keep], "score": score[keep].astype(np.float32),
                          "name_sim": name[keep].astype(np.float32), "address_sim": address[keep].astype(np.float32),
                          "bits": bits[keep], **{k: v[keep] for k, v in extra.items()}})
    frame.to_parquet(out / "new_pairs.parquet", index=False)
    np.save(out / "s1_postings.npy", postings)
    reference.to_parquet(out / "reference.parquet", index=False)
    per_k = {k: int((frame["rank"] < k).sum()) for k in (1, 2, 3, 5, 10) if k <= args.keep or args.keep <= 0}
    report = {"set": args.set, "s1": int(len(reference)), "pool_rows": int(len(pool)),
              "dense_pairs": int(len(dense_keys)), "branch_caps": CAPS, "branches": branch_stats,
              "key_union_pairs": int(total_union), "key_union_in_dense": int(in_dense.sum()),
              "new_pairs_all": int(len(s1)), "new_pairs_kept": int(len(frame)), "keep": args.keep,
              "new_pairs_by_k": per_k, "s1_with_new": int(frame.s1.nunique()),
              "truth_read": False, "positive_injection": False, "seconds": time.time() - started}
    (out / "mine.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({k: report[k] for k in ("new_pairs_all", "new_pairs_by_k", "s1_with_new", "seconds")}))
    return report


def _oracle(tp, t):
    tp, t = np.asarray(tp, np.float64), np.asarray(t, np.float64)
    den = 4 * tp + t
    return np.where(t == 0, 1.0, np.divide(5 * tp, den, out=np.zeros_like(den), where=den > 0))


def b_rest_truth(raw_dir: Path, reference: pd.DataFrame) -> dict[str, int]:
    """Target entity ID -> local B_REST index; payloads of other S1s are never split."""
    index = pd.Series(np.arange(len(reference)), index=reference.entity_id.to_numpy())
    wanted = set(reference.entity_id)
    owners: dict[str, int] = {}
    with (raw_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("Unexpected training ground-truth header")
        for line in stream:
            source_id, _, payload = line.rstrip("\r\n").partition("\t")
            if source_id not in wanted:
                continue
            position = int(index[source_id])
            for target in payload.split(",") if payload else []:
                owners[target] = position
    return owners


def evaluate_cmd(args) -> dict:
    from ber.arena_decision import component_bootstrap
    mined = Path(args.mined)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(mined / "reference.parquet", columns=["entity_id", "country", "arena_idx"])
    pairs = pd.read_parquet(mined / "new_pairs.parquet")
    arena_idx = reference.arena_idx.to_numpy()
    partition = np.load(ARENA / "finalize_v1" / "s1_partition.npy")
    if not np.all(partition[arena_idx] == 1):
        raise SystemExit("mined set is not B_REST")
    components = np.load(ARENA / "finalize_v1" / "component_ids.npy")
    true_count_all = np.load(ARENA / "labels" / "true_count.npy").astype(np.int64)
    population = partition == 1
    n_all = len(partition)
    owners = b_rest_truth(Path(args.raw), reference)
    pool_ids = pd.concat([pd.read_csv(Path(args.raw) / f"train_source{s}.tsv", sep="\t", dtype=str,
                                      keep_default_na=False, usecols=["entity_id"]) for s in (2, 3)],
                         ignore_index=True).entity_id
    owner_raw = pool_ids.map(owners).fillna(-1).to_numpy().astype(np.int64)
    # Check the truth map reproduces the saved dense labels on B_REST pairs.
    table = pq.read_table(ARENA / "v1" / "pairs.parquet", columns=["left_idx", "right_raw_idx"])
    left = table.column("left_idx").to_numpy().astype(np.int64)
    rraw = table.column("right_raw_idx").to_numpy().astype(np.int64)
    del table
    rows = np.flatnonzero(population[left])
    label = np.asarray(np.load(ARENA / "labels" / "label.npy", mmap_mode="r")[rows])
    local = np.full(n_all, -1, dtype=np.int64)
    local[arena_idx] = np.arange(len(arena_idx))
    mine_label = owner_raw[rraw[rows]] == local[left[rows]]
    mismatch = int((mine_label != label).sum())
    if mismatch:
        raise SystemExit(f"truth map disagrees with label.npy on {mismatch} dense B_REST pairs")
    if int(sum(1 for _ in owners)) != int(true_count_all[arena_idx].sum()):
        raise SystemExit("truth targets do not match true_count on B_REST")
    dense_tp = np.bincount(left[rows][label], minlength=n_all)
    del left, rraw, rows, label, mine_label
    new_label = owner_raw[pairs.raw.to_numpy()] == pairs.s1.to_numpy()
    np.save(out / "new_label.npy", new_label)
    base = _oracle(dense_tp, true_count_all)
    t_brest = int(true_count_all[population].sum())
    country = reference.country.to_numpy()
    table_rows = []
    for k in args.ks:
        m = pairs["rank"].to_numpy() < k
        add = np.bincount(arena_idx[pairs.s1.to_numpy()[m & new_label]], minlength=n_all)
        union = _oracle(dense_tp + add, true_count_all)
        boot = component_bootstrap(base, union, components, n=args.bootstrap, seed=20260926, population=population)
        by_country = {}
        for c in np.unique(country):
            cm = country == c
            cmask = cm[pairs.s1.to_numpy()]
            by_country[str(c)] = {"extra_pairs": int((m & cmask).sum()), "new_true": int((m & new_label & cmask).sum())}
        row = {"k": k, "extra_pairs": int(m.sum()), "new_true_links": int((m & new_label).sum()),
               "new_true_per_1000": float(1000 * (m & new_label).sum() / max(1, m.sum())),
               "recall_dense": float(dense_tp[population].sum() / t_brest),
               "recall_union": float((dense_tp + add)[population].sum() / t_brest),
               "oracle_dense": float(base[population].mean()), "oracle_union": float(union[population].mean()),
               "oracle_delta": boot["delta"], "ci": [boot["ci_low"], boot["ci_high"]], "by_country": by_country}
        table_rows.append(row)
        log(json.dumps({key: row[key] for key in ("k", "extra_pairs", "new_true_links", "new_true_per_1000",
                                                     "recall_union", "oracle_delta", "ci")}))
    # Which branches carry the found links (k = max).
    m = pairs["rank"].to_numpy() < max(args.ks)
    branch = {b: {"pairs": int((m & ((pairs.bits.to_numpy() >> i) & 1).astype(bool)).sum()),
                  "true": int((m & new_label & ((pairs.bits.to_numpy() >> i) & 1).astype(bool)).sum())}
              for i, b in enumerate(BRANCHES)}
    report = {"scope": "arena B_REST (SLICE and HOLDOUT-2 unread); oracle = accept exactly the retrieved true links",
              "b_rest_s1": int(population.sum()), "b_rest_true_links": t_brest,
              "truth_check_dense_pairs_mismatch": mismatch, "k_table": table_rows, "branches_at_max_k": branch,
              "bootstrap": {"unit": "arena claim component", "replicates": args.bootstrap, "seed": 20260926}}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def export_cmd(args) -> dict:
    mined, out = Path(args.mined), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(mined / "reference.parquet")
    pairs = pd.read_parquet(mined / "new_pairs.parquet")
    pairs = pairs[pairs["rank"] < args.k].reset_index(drop=True)
    split = "train" if args.set == "b_rest" else "test"
    pool = load_pool(Path(args.raw), split)
    raw_rows = np.unique(pairs.raw.to_numpy())
    new_pool = pool.iloc[raw_rows].reset_index(drop=True)
    new_pool["raw_row_idx"] = raw_rows.astype(np.int64)
    right = np.searchsorted(raw_rows, pairs.raw.to_numpy()).astype(np.int32)
    left = pairs.s1.to_numpy().astype(np.int32)
    reference.to_parquet(out / "reference.parquet", index=False)
    new_pool.to_parquet(out / "pool.parquet", index=False)
    np.save(out / "left.npy", left)
    np.save(out / "right.npy", right)
    np.save(out / "tree.npy", np.ones(len(left), dtype=np.float32))
    pairs.to_parquet(out / "pairs.parquet", index=False)
    if args.label is not None:
        lab = np.load(args.label)
        full = pd.read_parquet(mined / "new_pairs.parquet", columns=["rank"])["rank"].to_numpy() < args.k
        np.save(out / "eval_label.npy", lab[full])  # merge-time use only; the scorer never reads it
    report = {"set": args.set, "k": args.k, "pairs": int(len(left)), "reference": int(len(reference)),
              "pool_records": int(len(new_pool)), "s1_with_pairs": int(pairs.s1.nunique()),
              "gate_bypass": "tree.npy is all ones", "labels_in_scorer_inputs": False}
    (out / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--set", choices=("a", "b_rest", "test"), required=True)
    m.add_argument("--features", action="store_true", help="Extra pair similarities for the learned ranker")
    m.add_argument("--raw", required=True, help="Dir with {train,test}_source{2,3}.tsv")
    m.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
    m.add_argument("--workers", type=int, default=8)
    m.add_argument("--keep", type=int, default=10, help="Top pairs per S1 kept (0 keeps all)")
    m.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--mined", required=True)
    e.add_argument("--raw", required=True)
    e.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    e.add_argument("--bootstrap", type=int, default=2000)
    e.add_argument("--output", required=True)
    x = sub.add_parser("export")
    x.add_argument("--set", choices=("b_rest", "test"), required=True)
    x.add_argument("--mined", required=True)
    x.add_argument("--raw", required=True)
    x.add_argument("--k", type=int, required=True)
    x.add_argument("--label", help="evaluate's new_label.npy (B_REST only), subset saved for merge")
    x.add_argument("--output", required=True)
    args = parser.parse_args()
    {"mine": mine, "evaluate": evaluate_cmd, "export": export_cmd}[args.command](args)


if __name__ == "__main__":
    main()
