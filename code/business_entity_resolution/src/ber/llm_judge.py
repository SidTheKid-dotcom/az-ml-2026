"""Agent J: Qwen3-Reranker-4B LoRA pair judge trained on hard partition-A pairs, scored on uncertain pairs only.

Commands (CPU prep on the EC2 box; GPU train/score with ``ber.qwen4b_train``)::

    python -m ber.llm_judge build-train --arena ARENA --texts TEXTS --reverse-a R_A.parquet \
        --fr FR_V3 --model TOKENIZER_DIR --out DIR
    python -m ber.llm_judge build-score --keys KEYS.parquet --side train|test --texts TEXTS \
        --model TOKENIZER_DIR --out DIR

``build-train`` writes the ``ber.qwen_train`` materialized layout (tokens/offsets/labels/weights +
manifest, batch 128, step ``s`` = rows ``[s*128, (s+1)*128)``), so ``ber.qwen4b_train train`` runs on it
unchanged. Training pairs (partition A only; B_REST / SLICE / HOLDOUT-2 never read):

* arena A dense pairs stratified by v1 blend and label (hard positives: low blend; hard negatives:
  high blend; some easy pairs of both classes), plus empty-address and native-script (non-Latin record
  name) pairs with blend >= 0.01 or label 1;
* Agent R's partition-A reverse pairs: every positive and a random sample of negatives;
* Frenchified partition-A pairs (``arena_fr_v3`` world 1, option A): real positives/decoys and the
  typed synthetic kinds (legal-form swap + house-number delta, decoy word + number shift, siblings,
  same-number legal add/drop copies).

US/India text is Agent R's raw text (``name | address`` from the raw TSVs, prefix stripped) looked up by
entity ID, i.e. the same source used for scoring; French text is the Frenchified arena's
``name | address``. Labels are only used as training targets. Weights are uniform.

``build-score`` tokenizes any keyed (``s1_id``, ``rec_id``) pair list in the ``qwen_train`` score layout
(``pair_index`` = row order, ``total_pairs`` = rows), so ``ber.qwen4b_train score`` writes
``scores.npy`` aligned with ``pairs.parquet``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import qwen_train as qt

MODEL_ID = "Qwen/Qwen3-Reranker-4B"
BATCH = qt.BATCH
T0 = time.time()


def log(message) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def plain(names, addresses) -> np.ndarray:
    """``name | address`` with nulls as empty strings (C2 / Qwen plain view)."""
    import pandas as pd

    n = pd.Series(names, dtype=object).fillna("").astype(str).to_numpy(dtype=object)
    a = pd.Series(addresses, dtype=object).fillna("").astype(str).to_numpy(dtype=object)
    return np.asarray([f"{x} | {y}" for x, y in zip(n, a)], dtype=object)


class TextLookup:
    """Entity ID -> raw ``name | address`` text from Agent R's text parquets (prefix stripped)."""

    def __init__(self, texts_dir, side: str):
        import pandas as pd
        import pyarrow.parquet as pq

        self.side = side
        self.maps = {}
        for kind, prefix in (("s1", "query: "), ("pool", "passage: ")):
            path = Path(texts_dir) / f"{side}_{kind}.parquet"
            table = pq.read_table(path, columns=["entity_id", "text"]).to_pandas()
            text = table.text.to_numpy(dtype=object)
            bad = int(sum(1 for t in text[:100000] if not t.startswith(prefix)))
            if bad:
                raise ValueError(f"{path}: {bad} texts lack the {prefix!r} prefix")
            cut = len(prefix)
            self.maps[kind] = pd.Series([t[cut:] for t in text], index=table.entity_id.to_numpy(), dtype=object)
            if not self.maps[kind].index.is_unique:
                raise ValueError(f"{path}: duplicate entity IDs")
            log(f"text lookup {side}_{kind}: {len(table):,} entities")

    def get(self, kind: str, ids) -> np.ndarray:
        values = self.maps[kind].reindex(np.asarray(ids, dtype=object)).to_numpy(dtype=object)
        missing = int(sum(1 for v in values if not isinstance(v, str)))
        if missing:
            raise ValueError(f"{missing} {kind} IDs have no text")
        return values


def tokenize_frame(tokenizer, left_text, right_text):
    """Tokenize each distinct left/right text once and assemble prompts (``qwen_train`` format)."""
    import pandas as pd

    lcode, luniq = pd.factorize(pd.Series(left_text, dtype=object), sort=False)
    rcode, runiq = pd.factorize(pd.Series(right_text, dtype=object), sort=False)
    return qt.tokenize_pairs(tokenizer, list(luniq), list(runiq), lcode.astype(np.int64), rcode.astype(np.int64))


def _choose(rng, pool_rows, n):
    if len(pool_rows) <= n:
        return pool_rows
    return rng.choice(pool_rows, n, replace=False)


def build_train(args) -> dict:
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arena = Path(args.arena)
    summary = json.loads((arena / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") not in (0, None):
        raise ValueError("Arena labels include HOLDOUT-2 rows")
    table = pq.read_table(arena / "v1" / "pairs.parquet", columns=["left_idx", "right_idx"])
    left = table.column("left_idx").to_numpy().astype(np.int64)
    right = table.column("right_idx").to_numpy().astype(np.int64)
    del table
    blend = np.load(arena / "v1" / "probabilities.npy", mmap_mode="r")
    label_all = np.load(arena / "labels" / "label.npy", mmap_mode="r")
    partition = np.load(arena / "finalize_v1" / "s1_partition.npy")
    if len(blend) != len(left) or len(label_all) != len(left):
        raise ValueError("Arena arrays are not aligned")
    rows_a = np.flatnonzero(partition[left] == 0)
    b = np.asarray(blend[rows_a], dtype=np.float32)
    y = np.asarray(label_all[rows_a]).astype(bool)
    log(f"arena pairs {len(left):,}; partition A pairs {len(rows_a):,}; positives {int(y.sum()):,}")
    reference = pd.read_parquet(arena / "v1" / "reference.parquet", columns=["entity_id", "business_address"])
    pool_t = pq.read_table(arena / "v1" / "pool.parquet", columns=["entity_id", "business_name", "business_address"])
    rec_empty = pc.fill_null(pc.equal(pc.utf8_trim_whitespace(pc.fill_null(pool_t.column("business_address"), "")), ""),
                             True).to_numpy(zero_copy_only=False)
    native = pc.fill_null(pc.match_substring_regex(pool_t.column("business_name"),
                                                   r"[^\x{0000}-\x{024F}\x{1E00}-\x{1EFF}\x{2000}-\x{206F}]"),
                          False).to_numpy(zero_copy_only=False)
    pool_ids = pool_t.column("entity_id").to_numpy(zero_copy_only=False)
    del pool_t
    s1_empty = reference.business_address.fillna("").astype(str).str.strip().eq("").to_numpy()
    s1_ids = reference.entity_id.to_numpy(dtype=object)
    log(f"records: empty address {int(rec_empty.sum()):,}, native-script name {int(native.sum()):,}")

    picks, kinds = [], []

    def add(mask, n, name):
        chosen = _choose(rng, np.flatnonzero(mask), n)
        picks.append(chosen)
        kinds.append(np.full(len(chosen), name, dtype=object))
        log(f"  {name}: {len(chosen):,} of {int(mask.sum()):,}")

    q = args.scale
    pos_bins = ((0, .1, 25_000), (.1, .5, 20_000), (.5, .9, 15_000), (.9, 1.01, 20_000))
    neg_bins = ((.5, 1.01, 50_000), (.2, .5, 36_000), (.05, .2, 45_000), (.01, .05, 30_000),
                (0, .01, 12_000))  # v1 blend has no mass in (0.002, 0.01)
    for lo, hi, n in pos_bins:
        add(y & (b >= lo) & (b < hi), int(n * q), f"arena_pos_b{lo}-{min(hi, 1)}")
    for lo, hi, n in neg_bins:
        add(~y & (b >= lo) & (b < hi), int(n * q), f"arena_neg_b{lo}-{min(hi, 1)}")
    gate = y | (b >= .01)
    empty = rec_empty[right[rows_a]] | s1_empty[left[rows_a]]
    add(gate & empty, int(15_000 * q), "arena_empty_address")
    add(gate & native[right[rows_a]], int(20_000 * q), "arena_native_script")
    sel = np.concatenate(picks)
    kind_arr = np.concatenate(kinds)
    sel, first = np.unique(sel, return_index=True)
    kind_arr = kind_arr[first]
    pair_rows = rows_a[sel]
    arena_frame = pd.DataFrame({
        "source": "arena_A", "kind": kind_arr, "s1_id": s1_ids[left[pair_rows]],
        "rec_id": pool_ids[right[pair_rows]], "label": y[sel].astype(np.int8),
        "blend": b[sel], "arena_pair": pair_rows})
    log(f"arena A sample {len(arena_frame):,}, positive rate {arena_frame.label.mean():.3f}")

    rev = pd.read_parquet(args.reverse_a, columns=["s1_id", "rec_id", "left_idx", "label"])
    if (partition[rev.left_idx.to_numpy()] != 0).any():
        raise ValueError("Reverse A file holds non-A S1s")
    rpos = np.flatnonzero(rev.label.to_numpy())
    rneg = _choose(rng, np.flatnonzero(~rev.label.to_numpy()), int(30_000 * q))
    rsel = np.concatenate([rpos, rneg])
    rev_frame = pd.DataFrame({"source": "reverse_A", "kind": np.where(rev.label.to_numpy()[rsel], "rev_pos", "rev_neg"),
                              "s1_id": rev.s1_id.to_numpy()[rsel], "rec_id": rev.rec_id.to_numpy()[rsel],
                              "label": rev.label.to_numpy()[rsel].astype(np.int8), "blend": np.float32(-1),
                              "arena_pair": -1})
    log(f"reverse A: {len(rpos):,} positives + {len(rneg):,} negatives")

    texts = TextLookup(args.texts, "train")
    us_in = pd.concat([arena_frame, rev_frame], ignore_index=True)
    us_in = us_in.drop_duplicates(["s1_id", "rec_id"]).reset_index(drop=True)
    us_in["left_text"] = texts.get("s1", us_in.s1_id)
    us_in["right_text"] = texts.get("pool", us_in.rec_id)
    # fidelity: R raw text vs arena reference text for sampled S1s
    ref_full = pd.read_parquet(arena / "v1" / "reference.parquet", columns=["entity_id", "business_name", "business_address"])
    sample = ref_full.sample(20000, random_state=1)
    same = (plain(sample.business_name, sample.business_address) == texts.get("s1", sample.entity_id)).mean()
    log(f"R text == arena plain text on 20k S1s: {same:.4f}")
    del texts, ref_full

    fr = Path(args.fr)
    ft = pq.read_table(fr / "arena" / "pairs.parquet", columns=["left_idx", "right_idx", "world", "kind"])
    fl, fr_r = ft.column("left_idx").to_numpy().astype(np.int64), ft.column("right_idx").to_numpy().astype(np.int64)
    world, fkind = ft.column("world").to_numpy(), ft.column("kind").to_numpy()
    flabel = np.load(fr / "labels" / "label.npy").astype(bool)
    fblend = np.load(fr / "arena" / "v1_blend.npy")
    fpart = np.load(fr / "s1_partition.npy")
    frrows = (world == 1) & (fpart[fl] == 0)
    names = ["real_positive", "real_decoy", "syn_legal_swap_delta", "syn_decoy_word_shift", "syn_sibling_own_copy",
             "syn_sibling_vs_original_records", "syn_original_vs_sibling_copies",
             "syn_same_number_legal_added_or_dropped_copy"]
    quota = {0: 35_000, 1: 25_000, 2: 12_000, 3: 12_000, 4: 4_000, 5: 6_000, 6: 6_000, 7: 8_000}
    fpicks, fkinds = [], []
    for k, n in quota.items():
        mask = frrows & (fkind == k)
        if k == 1:  # real decoys: half from blend >= 0.1, half uniform
            hi = _choose(rng, np.flatnonzero(mask & (fblend >= .1)), int(n * q) // 2)
            rest = _choose(rng, np.setdiff1d(np.flatnonzero(mask), hi), int(n * q) - len(hi))
            chosen = np.concatenate([hi, rest])
        else:
            chosen = _choose(rng, np.flatnonzero(mask), int(n * q))
        fpicks.append(chosen)
        fkinds.append(np.full(len(chosen), "fr_" + names[k], dtype=object))
        log(f"  fr {names[k]}: {len(chosen):,} of {int(mask.sum()):,}, positive rate {flabel[chosen].mean():.3f}")
    fsel = np.concatenate(fpicks)
    fref = pd.read_parquet(fr / "arena" / "reference.parquet", columns=["entity_id", "business_name", "business_address"])
    fpool = pd.read_parquet(fr / "arena" / "pool.parquet", columns=["entity_id", "business_name", "business_address"])
    li, ri = fl[fsel], fr_r[fsel]
    fr_frame = pd.DataFrame({"source": "french_A", "kind": np.concatenate(fkinds),
                             "s1_id": fref.entity_id.to_numpy(dtype=object)[li],
                             "rec_id": fpool.entity_id.to_numpy(dtype=object)[ri],
                             "label": flabel[fsel].astype(np.int8), "blend": fblend[fsel].astype(np.float32),
                             "arena_pair": fsel,
                             "left_text": plain(fref.business_name.to_numpy(dtype=object)[li],
                                                fref.business_address.to_numpy(dtype=object)[li]),
                             "right_text": plain(fpool.business_name.to_numpy(dtype=object)[ri],
                                                 fpool.business_address.to_numpy(dtype=object)[ri])})
    frame = pd.concat([us_in, fr_frame], ignore_index=True)
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    frame = frame.iloc[: len(frame) - len(frame) % BATCH].reset_index(drop=True)
    frame["pair_row"] = np.arange(len(frame), dtype=np.int64)
    frame["step"] = frame.pair_row // BATCH
    frame.to_parquet(out / "train_pairs.parquet", index=False)
    log(f"training pairs {len(frame):,}; positive rate {frame.label.mean():.3f}; "
        f"by source {frame.source.value_counts().to_dict()}")
    tokenizer = qt.load_tokenizer(args.model)
    tokens, offsets = tokenize_frame(tokenizer, frame.left_text.tolist(), frame.right_text.tolist())
    labels = frame.label.to_numpy().astype(np.int8)
    weights = np.ones(len(frame), dtype=np.float32)
    qt.save_npy(out / "tokens.npy", tokens)
    qt.save_npy(out / "offsets.npy", offsets)
    qt.save_npy(out / "labels.npy", labels)
    qt.save_npy(out / "weights.npy", weights)
    lengths = np.diff(offsets)
    manifest = {
        "kind": "train", "state": "complete", "model_id": MODEL_ID, "tokenizer_dir": str(Path(args.model).resolve()),
        **qt.token_meta(tokenizer), "text_view": "raw 'name | address' (R texts for US/India, fr_v3 for French)",
        "seed": args.seed, "batch_size": BATCH, "steps": len(frame) // BATCH, "pairs": len(frame),
        "positive_fraction": float(labels.mean()),
        "kinds": {str(k): int(v) for k, v in frame.kind.value_counts().items()},
        "sources": {str(k): int(v) for k, v in frame.source.value_counts().items()},
        "token_length": {"mean": float(lengths.mean()), "p50": float(np.percentile(lengths, 50)),
                         "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max())},
        "files": {name: qt.sha256_file(out / name) for name in
                  ("tokens.npy", "offsets.npy", "labels.npy", "weights.npy", "train_pairs.parquet")},
        "partition_sha256": qt.sha256_file(arena / "finalize_v1" / "s1_partition.npy"),
        "labels_read": "partition-A arena labels, R reverse-A labels, fr_v3 world-1 partition-A labels",
        "holdout2_rows_read": 0, "scale": args.scale}
    qt.json_dump(out / "manifest.json", manifest)
    log(json.dumps({k: manifest[k] for k in ("pairs", "positive_fraction", "sources", "token_length")}))
    return manifest


def build_score(args) -> dict:
    import pandas as pd

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = pd.read_parquet(args.keys)
    if keys.duplicated(["s1_id", "rec_id"]).any():
        raise ValueError("Duplicate (s1_id, rec_id) keys")
    texts = TextLookup(args.texts, args.side)
    left_text = texts.get("s1", keys.s1_id)
    right_text = texts.get("pool", keys.rec_id)
    del texts
    keys.to_parquet(out / "pairs.parquet", index=False)
    tokenizer = qt.load_tokenizer(args.model)
    tokens, offsets = tokenize_frame(tokenizer, list(left_text), list(right_text))
    n = len(keys)
    qt.save_npy(out / "tokens.npy", tokens)
    qt.save_npy(out / "offsets.npy", offsets)
    qt.save_npy(out / "pair_index.npy", np.arange(n, dtype=np.int64))
    lengths = np.diff(offsets)
    manifest = {"kind": "test" if args.side == "test" else "arena", "state": "complete", "model_id": MODEL_ID,
                **qt.token_meta(tokenizer), "gate": None, "total_pairs": int(n), "scored_pairs": int(n),
                "keys": str(Path(args.keys).resolve()), "keys_sha256": qt.sha256_file(args.keys), "side": args.side,
                "alignment": "row k of scores.npy = row k of pairs.parquet",
                "token_length": {"mean": float(lengths.mean()), "p99": float(np.percentile(lengths, 99)),
                                 "max": int(lengths.max())},
                "files": {name: qt.sha256_file(out / name) for name in
                          ("tokens.npy", "offsets.npy", "pair_index.npy", "pairs.parquet")},
                "labels_read": False}
    qt.json_dump(out / "manifest.json", manifest)
    log(json.dumps({k: manifest[k] for k in ("kind", "total_pairs", "token_length")}))
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build-train")
    p.add_argument("--arena", required=True, help="dir with v1/, labels/, finalize_v1/")
    p.add_argument("--texts", required=True, help="Agent R texts dir ({train,test}_{s1,pool}.parquet)")
    p.add_argument("--reverse-a", required=True)
    p.add_argument("--fr", required=True, help="arena_fr_v3 dir")
    p.add_argument("--model", required=True, help="tokenizer dir")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260927)
    p = sub.add_parser("build-score")
    p.add_argument("--keys", required=True)
    p.add_argument("--side", choices=("train", "test"), required=True)
    p.add_argument("--texts", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-train":
        build_train(args)
    else:
        build_score(args)


if __name__ == "__main__":
    main()
