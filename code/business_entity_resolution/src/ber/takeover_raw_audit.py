"""CPU-only raw-data accounting for the 27 Sep takeover.

The only label authority is the supplied TSV, filtered by an arena-ID allowlist
BEFORE decoding the second column. SLICE and non-arena truth payloads are never
parsed. Accounting truth is never inserted into a candidate set.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import unicodedata

# Set before numpy/Arrow imports, including when this is run as a standalone file.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inspect(a):
    roots = [a.arena, a.data, a.texts, a.u / "nat", a.u / "nat/gt",
             a.u / "prep/brest_u8t", a.texts.parent / "out",
             Path("/opt/az-ml-new/data/raw/dataset/train")]
    result = {"directories": {}, "parquet": {}, "metadata": {}}
    for p in roots:
        result["directories"][str(p)] = (
            [{"name": x.name, "bytes": x.stat().st_size if x.is_file() else None}
             for x in sorted(p.iterdir())] if p.exists() else "missing")
    paths = [a.arena / "v1/reference.parquet", a.arena / "v1/pool.parquet",
             a.arena / "v1/pairs.parquet", a.u / "data/arena/pool.parquet",
             a.texts / "train_s1.parquet", a.texts / "train_pool.parquet",
             a.texts.parent / "out/r_reverse_pairs_a.parquet", a.data / "train.parquet"]
    for p in paths:
        if p.exists():
            pf = pq.ParquetFile(p)
            result["parquet"][str(p)] = {"rows": pf.metadata.num_rows, "schema": str(pf.schema_arrow)}
    for p in [a.data / "train_complete.json", a.data / "brest_complete.json",
              a.u / "prep/brest_u8t/meta.json", a.u / "prep/brest_u8t/metadata.json",
              a.arena / "labels/labels_summary.json"]:
        if p.exists():
            result["metadata"][str(p)] = json.loads(p.read_text())
    result["cgroup"] = {str(p): p.read_text().strip() for p in
                        [Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/memory.max")]
                        if p.exists()}
    save_json(a.out / "inspection.json", result)
    log("inspection", **result)


def source_evidence(a):
    """One-core local streaming verification; no truth file is opened here."""
    result = {}
    sources = [1] if a.command == "s1-evidence" else [2, 3]
    for split in ("train", "test"):
        for source in sources:
            path = a.raw / split / f"{split}_source{source}.tsv"
            ids, content = hashlib.sha256(), hashlib.sha256()
            count, samples, cc = 0, [], Counter()
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                assert reader.fieldnames == ["entity_id", "business_name", "business_address", "country"]
                for i, row in enumerate(reader):
                    assert None not in row and all(v is not None for v in row.values())
                    ids.update((row["entity_id"] + "\n").encode())
                    values = [row[k] for k in reader.fieldnames]
                    content.update((json.dumps(values, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                    cc[row["country"]] += 1
                    if i < 3 or i % 32749 == 0:
                        samples.append({"source_row": i, **row})
                    count += 1
            result[f"{split}_s{source}"] = {"path": str(path), "bytes": path.stat().st_size,
                "rows": count, "ordered_id_sha256": ids.hexdigest(),
                "canonical_record_sha256": content.hexdigest(), "country_counts": dict(cc), "samples": samples}
            log("source_evidence", source=f"{split}_s{source}", rows=count, samples=len(samples))
    save_json(a.out / ("raw_s1_evidence.json" if sources == [1] else "raw_pool_evidence.json"), result)


def field_exceptions(a):
    """Recover field boundaries that a concatenated raw text cannot encode."""
    rows, offset = [], 0
    for source in (2, 3):
        path = a.raw / "train" / f"train_source{source}.tsv"
        with path.open(encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
                if " | " in row["business_name"] or " | " in row["business_address"]:
                    rows.append({"raw_row_idx": offset + i, **row})
            offset += i + 1
    save_json(a.out / "raw_field_exceptions.json", {"rows": rows, "pool_rows": offset,
              "definition": "Raw name or address contains the literal concatenation separator; fields copied from original TSV"})
    log("raw_field_exceptions", rows=len(rows), pool_rows=offset)


def verify_s1_samples(a):
    evidence = json.loads((a.out / "raw_s1_evidence.json").read_text())
    checks = {}
    for split in ("train", "test"):
        sample = {x["entity_id"]: x for x in evidence[f"{split}_s1"]["samples"]}
        ids, hit, cc_map = [], 0, {}
        pf = pq.ParquetFile(a.texts / f"{split}_s1.parquet")
        for batch in pf.iter_batches(batch_size=65536):
            d = batch.to_pydict()
            ids.extend(d["entity_id"])
            for sid, text, cc in zip(d["entity_id"], d["text"], d["cc"]):
                if sid not in sample:
                    continue
                raw = sample[sid]
                assert text == "query: " + raw["business_name"] + " | " + raw["business_address"], sid
                if raw["country"] in cc_map:
                    assert cc_map[raw["country"]] == cc
                cc_map[raw["country"]] = cc
                hit += 1
        assert hit > 30
        if split == "train":
            arena_ids = pq.read_table(a.arena / "v1/reference.parquet", columns=["entity_id"])["entity_id"].to_pylist()
            assert ids == arena_ids
        else:
            h = hashlib.sha256()
            for sid in ids:
                h.update((sid + "\n").encode())
            assert h.hexdigest() == evidence["test_s1"]["ordered_id_sha256"]
        checks[split] = {"rows": len(ids), "source_sample_matches": hit, "source_sample_mismatches": 0,
                         "raw_country_to_cc": cc_map,
                         "id_order_verified_against": "arena reference" if split == "train" else "entire raw test S1"}
    return checks


def read_selected_truth(a):
    """Read S1s first, then allowlisted raw labels, before any target text."""
    if (a.out / "truth_read.json").exists():
        raise FileExistsError("Raw truth has already been filtered; reuse its accounting files")
    sample_checks = verify_s1_samples(a)
    ref_ids = pq.read_table(a.arena / "v1/reference.parquet", columns=["entity_id"])["entity_id"].to_pylist()
    part = np.load(a.arena / "finalize_v1/s1_partition.npy", allow_pickle=False)
    train = np.load(a.data / "train_s1.npy", allow_pickle=False)
    screen = np.load(a.data / "screen_s1.npy", allow_pickle=False)
    assert len(ref_ids) == len(part) == len(set(ref_ids))
    assert len(train) == len(np.unique(train)) == 200000 and np.all(part[train] == 0)
    assert len(screen) == len(np.unique(screen)) == 30000 and np.all(part[screen] == 1)
    selected = np.sort(np.r_[train, np.flatnonzero(part == 1)])
    assert len(selected) == 721846
    allowed = {ref_ids[i].encode(): int(i) for i in selected}
    counts = np.full(len(part), -1, dtype=np.int16)
    left, target, payload_hash = [], [], hashlib.sha256()
    skipped, lines = 0, 0
    with a.gt.open("rb") as f:
        header = f.readline().rstrip(b"\r\n")
        assert header == b"source1_entity_id\tmatched_entity_ids"
        for line in f:
            # Do not decode/strip/split the excluded payload. Only find the first
            # delimiter and interpret the source ID; discard unselected bytes.
            tab = line.find(b"\t")
            assert tab > 0
            i = allowed.get(line[:tab])
            lines += 1
            if i is None:
                skipped += 1
                continue
            assert part[i] in (0, 1) and counts[i] == -1
            payload_hash.update(line)
            payload = line[tab + 1:].rstrip(b"\r\n").decode("utf-8")
            recs = payload.split(",") if payload else []
            assert len(recs) == len(set(recs))
            assert all(r.startswith(("S2-", "S3-")) for r in recs)
            counts[i] = len(recs)
            left.extend([i] * len(recs))
            target.extend(recs)
    assert np.all(counts[selected] >= 0)
    assert len(target) == len(set(target)), "A raw target has more than one selected owner"
    pq.write_table(pa.table({"left_idx": np.asarray(left, np.int32), "rec_id": target}),
                   a.out / "selected_truth_ids.parquet", compression="zstd")
    pq.write_table(pa.table({"left_idx": selected.astype(np.int32),
                            "s1_id": [ref_ids[i] for i in selected], "partition": part[selected],
                            "true_count": counts[selected]}), a.out / "selected_true_count.parquet", compression="zstd")
    report = {"ground_truth_path": str(a.gt), "ground_truth_bytes": a.gt.stat().st_size,
              "raw_rows_scanned_for_source_ID_only": lines, "selected_payload_rows_parsed": len(selected),
              "unselected_payload_rows_skipped_without_decoding": skipped, "excluded_payloads_parsed": 0,
              "holdout2_payloads_parsed": 0, "slice_payloads_parsed": 0,
              "selected_raw_line_sha256": payload_hash.hexdigest(), "raw_true_links": len(target),
              "A": {"s1": len(train), "true_links": int(counts[train].sum()),
                    "singletons": int((counts[train] == 0).sum())},
              "B_REST": {"s1": int((part == 1).sum()), "true_links": int(counts[part == 1].sum()),
                         "singletons": int((counts[part == 1] == 0).sum())},
              "screen": {"s1": len(screen), "partition": "B_REST", "true_links": int(counts[screen].sum()),
                         "singletons": int((counts[screen] == 0).sum())},
              "s1_source_verification": sample_checks,
              "identities": {"partition_sha256": digest(a.arena / "finalize_v1/s1_partition.npy"),
                             "train_s1_sha256": digest(a.data / "train_s1.npy"),
                             "screen_s1_sha256": digest(a.data / "screen_s1.npy")}}
    save_json(a.out / "truth_read.json", report)
    log("truth_read", **report)


def verified_pool(a):
    """Stream S2-then-S3 projections, checking full ID order and raw samples."""
    evidence = json.loads((a.out / "raw_pool_evidence.json").read_text())
    checks, retained = {}, []
    countries = {"US": 0, "India": 1, "France": 2}
    for split in ("train", "test"):
        lengths = [evidence[f"{split}_s{s}"]["rows"] for s in (2, 3)]
        samples = {}
        for source, offset in ((2, 0), (3, lengths[0])):
            for row in evidence[f"{split}_s{source}"]["samples"]:
                samples[offset + row["source_row"]] = row
        sample_rows = np.asarray(sorted(samples), np.int64)
        hs = [hashlib.sha256(), hashlib.sha256()]
        n, matched = 0, 0
        code_counts = np.zeros((2, 3), np.int64)
        for batch in pq.ParquetFile(a.texts / f"{split}_pool.parquet").iter_batches(batch_size=65536):
            d = batch.to_pydict()
            stop = n + len(batch)
            for source, lo, hi in ((0, 0, lengths[0]), (1, lengths[0], sum(lengths))):
                b, e = max(0, lo - n), min(len(batch), hi - n)
                if b < e:
                    hs[source].update(("\n".join(d["entity_id"][b:e]) + "\n").encode())
                    code_counts[source] += np.bincount(d["cc"][b:e], minlength=3)
            for j in sample_rows[np.searchsorted(sample_rows, n):np.searchsorted(sample_rows, stop)]:
                raw, k = samples[j], int(j - n)
                assert d["entity_id"][k] == raw["entity_id"]
                assert d["text"][k] == "passage: " + raw["business_name"] + " | " + raw["business_address"]
                assert d["cc"][k] == countries[raw["country"]]
                matched += 1
            if split == "train":
                retained.append(batch)
            n = stop
        assert n == sum(lengths) and matched == len(samples)
        for i, source in enumerate((2, 3)):
            raw = evidence[f"{split}_s{source}"]
            assert hs[i].hexdigest() == raw["ordered_id_sha256"]
            assert all(code_counts[i, code] == raw["country_counts"].get(country, 0)
                       for country, code in countries.items())
        checks[split] = {"rows": n, "s2_rows": lengths[0], "s3_rows": lengths[1],
                         "ordered_id_sha256_by_source": [h.hexdigest() for h in hs],
                         "source_sample_matches": matched, "source_sample_mismatches": 0,
                         "full_ID_order_and_country_counts_match_raw": True,
                         "full_text_equality_tested": False}
        log("pool_verified", split=split, **checks[split])
    table = pa.Table.from_batches(retained)
    assert pc.all(pc.starts_with(table["text"], pattern="passage: ")).as_py()
    plain = pc.replace_substring(table["text"], pattern="passage: ", replacement="", max_replacements=1)
    delimiter_counts = pc.count_substring(plain, pattern=" | ")
    bad_delimiters = pc.sum(pc.cast(pc.not_equal(delimiter_counts, 1), pa.int64())).as_py()
    fixes = json.loads((a.out / "raw_field_exceptions.json").read_text())
    assert fixes["pool_rows"] == len(table) and len(fixes["rows"]) == bad_delimiters
    bad_rows = pc.indices_nonzero(pc.not_equal(delimiter_counts, 1)).to_numpy()
    assert np.array_equal(bad_rows, [r["raw_row_idx"] for r in fixes["rows"]])
    for raw, cached in zip(fixes["rows"], table.take(pa.array(bad_rows)).to_pylist()):
        assert raw["entity_id"] == cached["entity_id"]
        assert cached["text"] == "passage: " + raw["business_name"] + " | " + raw["business_address"]
    checks["train"]["texts_with_ambiguous_field_separator"] = bad_delimiters
    checks["train"]["ambiguous_field_boundaries_recovered_from_raw_TSV"] = bad_delimiters
    return table, checks


def contains(sorted_keys, keys):
    if not len(sorted_keys):
        return np.zeros(len(keys), bool)
    ix = np.searchsorted(sorted_keys, keys)
    return (ix < len(sorted_keys)) & (sorted_keys[np.minimum(ix, len(sorted_keys) - 1)] == keys)


def pair_keys(left, raw_right, n_pool):
    return np.asarray(left, np.int64) * n_pool + raw_right


def pool_mapping(path, raw_ids):
    mapping, checked = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=262144, columns=["entity_id", "raw_row_idx"]):
        indices = batch.column("raw_row_idx").to_numpy()
        ids = batch.column("entity_id").to_numpy(zero_copy_only=False)
        assert np.all((indices >= 0) & (indices < len(raw_ids)))
        assert np.array_equal(raw_ids[indices], ids)
        mapping.append(indices.astype(np.int32))
        checked += len(batch)
    result = np.concatenate(mapping)
    assert len(np.unique(result)) == len(result)
    return result, {"path": str(path), "rows": checked, "all_ID_to_raw_index_mismatches": 0}


def entity_score(tp, pred, truth):
    den = 4 * pred + truth
    return np.where(den == 0, 1., np.divide(5. * tp, den, out=np.zeros(len(den)), where=den > 0))


def score_summary(tp, pred, truth, mask, country):
    def one(m):
        singles = m & (truth == 0)
        nt, npred, ngt = (int(v[m].sum()) for v in (tp, pred, truth))
        return {"s1": int(m.sum()), "macro_f05": float(entity_score(tp[m], pred[m], truth[m]).mean()),
                "precision": nt / npred if npred else None, "link_recall": nt / ngt if ngt else None,
                "true_links": ngt, "true_positive_links": nt, "predicted_links": npred,
                "false_links": npred - nt, "missed_true_links": ngt - nt,
                "singletons": int(singles.sum()), "singleton_false_merges": int((pred[singles] > 0).sum()),
                "singleton_accuracy": float((pred[singles] == 0).mean()) if singles.any() else None,
                "non_singletons_with_zero_true_predictions": int((m & (truth > 0) & (tp == 0)).sum())}
    out = one(mask)
    out["by_country"] = {str(c): one(mask & (country == c)) for c in np.unique(country[mask])}
    return out


def decode(left, right, prob, threshold):
    rows = np.flatnonzero(prob >= threshold)
    if not len(rows):
        return np.zeros(len(left), bool)
    # Same one-owner / lower-left tie break and cap-11 as the saved U8t decoder.
    order = np.lexsort((rows, left[rows], -prob[rows].astype(np.float64), right[rows]))
    rows = rows[order]
    rows = rows[np.r_[True, right[rows][1:] != right[rows][:-1]]]
    order = np.lexsort((rows, -prob[rows].astype(np.float64), left[rows]))
    ranked = rows[order]
    group = left[ranked]
    first = np.maximum.accumulate(np.where(np.r_[True, group[1:] != group[:-1]], np.arange(len(group)), 0))
    keep = ranked[np.arange(len(group)) - first < 11]
    out = np.zeros(len(left), bool)
    out[keep] = True
    return out


NATIVE = re.compile(r"[\u0900-\u0dff\u0f00-\u109f\u1780-\u17ff]")
SPACE = re.compile(r"[\W_]+", re.UNICODE)
DIGITS = re.compile(r"\d+")


def normal(text):
    return SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


class TextSide:
    def __init__(self, names, addresses):
        self.name = np.asarray(names, object)
        self.address = np.asarray(addresses, object)
        self.nn = np.asarray([normal(x) for x in names], object)
        self.na = np.asarray([normal(x) for x in addresses], object)
        self.native_name = np.fromiter((bool(NATIVE.search(x)) for x in names), bool, len(names))
        self.native_addr = np.fromiter((bool(NATIVE.search(x)) for x in addresses), bool, len(names))
        self.empty = np.fromiter((not x.strip() for x in addresses), bool, len(names))
        self.digit = np.asarray([(m.group().lstrip("0") or "0") if (m := DIGITS.search(x)) else ""
                                 for x in addresses], object)


def split_raw_text(texts, raw_rows, fixes):
    names, addresses = [], []
    for text, raw in zip(texts, raw_rows):
        if int(raw) in fixes:
            names.append(fixes[int(raw)]["business_name"])
            addresses.append(fixes[int(raw)]["business_address"])
        else:
            plain = text.removeprefix("passage: ")
            assert text.startswith("passage: ") and plain.count(" | ") == 1
            name, _, address = plain.partition(" | ")
            names.append(name)
            addresses.append(address)
    return np.asarray(names, object), np.asarray(addresses, object)


def feature_frame(left, right, ref, pool, country, source):
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    n = len(left)
    ln, rn, la, ra = ref.nn[left], pool.nn[right], ref.na[left], pool.na[right]
    ratio = lambda fn, x, y: np.fromiter((fn(a, b) / 100 for a, b in zip(x, y)), np.float32, n)
    name_ratio = ratio(fuzz.ratio, ln, rn)
    addr_ratio = ratio(fuzz.ratio, la, ra)
    addr_tsr = ratio(fuzz.token_set_ratio, la, ra)
    name_tsr = ratio(fuzz.token_set_ratio, ln, rn)
    name_edit = np.fromiter((Levenshtein.distance(a, b, score_cutoff=2) for a, b in zip(ln, rn)), np.int8, n)
    addr_edit = np.fromiter((Levenshtein.distance(a, b, score_cutoff=2) for a, b in zip(la, ra)), np.int8, n)
    both_address = ~ref.empty[left] & ~pool.empty[right]
    digit_equal = (ref.digit[left] == pool.digit[right]) & (ref.digit[left] != "")
    digit_conflict = ((ref.digit[left] != pool.digit[right]) & (ref.digit[left] != "") & (pool.digit[right] != ""))
    same_address = both_address & (la == ra)
    data = {"native_name": ref.native_name[left] | pool.native_name[right],
            "native_address": ref.native_addr[left] | pool.native_addr[right],
            "right_address_empty": pool.empty[right], "left_address_empty": ref.empty[left],
            "raw_name_exact": ref.name[left] == pool.name[right], "normalized_name_exact": ln == rn,
            "normalized_address_exact": same_address, "first_digit_group_equal": digit_equal,
            "first_digit_group_conflict": digit_conflict,
            "lexically_dissimilar_name_exact_address": (name_ratio < .5) & same_address,
            "lexically_dissimilar_name_matching_address": (name_ratio < .5) & both_address & (addr_tsr >= .9) & ~digit_conflict,
            "name_typo_1to2_edits": (name_edit > 0) & (name_edit <= 2),
            "address_typo_1to2_edits": both_address & (addr_edit > 0) & (addr_edit <= 2),
            "near_name_same_address": (name_ratio >= .85) & (ln != rn) & same_address,
            "exact_name_empty_address": (ln == rn) & pool.empty[right],
            "name_ratio": name_ratio, "name_token_set_ratio": name_tsr,
            "address_ratio": addr_ratio, "address_token_set_ratio": addr_tsr,
            "country": country, "source": source}
    return pd.DataFrame(data)


def feature_summary(frame):
    def one(f):
        booleans = {k: {"count": int(f[k].sum()), "fraction": float(f[k].mean()) if len(f) else None}
                    for k in f if f[k].dtype == bool}
        quantiles = {k: {str(q): float(v) for q, v in zip((0, .1, .5, .9, 1), np.quantile(f[k], (0, .1, .5, .9, 1)))}
                     for k in ("name_ratio", "name_token_set_ratio", "address_ratio", "address_token_set_ratio") if len(f)}
        return {"pairs": len(f), "flags": booleans, "quantiles": quantiles}
    result = one(frame)
    result["by_source"] = {str(k): one(v) for k, v in frame.groupby("source", sort=True)}
    result["by_country"] = {str(k): one(v) for k, v in frame.groupby("country", sort=True)}
    return result


def route_report(keys, dense, owner, train_keys, n_pool):
    unique = np.unique(keys)
    left, right = unique // n_pool, unique % n_pool
    y = owner[right] == left
    novel = ~contains(dense, unique)
    not_packed = ~contains(train_keys, unique)
    return {"rows": len(keys), "unique_pairs": len(unique), "duplicate_pairs": len(keys) - len(unique),
            "true_pairs_raw": int(y.sum()), "false_pairs_raw": int((~y).sum()),
            "beyond_dense_pairs": int(novel.sum()), "beyond_dense_true": int((novel & y).sum()),
            "outside_current_training_pack_true": int((not_packed & y).sum()),
            "outside_current_training_pack_false": int((not_packed & ~y).sum())}


def audit(a):
    started = time.monotonic()
    if (a.out / "report.json").exists():
        raise FileExistsError("Completed audit exists; reuse it instead of repeating")
    guard = json.loads((a.out / "truth_read.json").read_text())
    assert guard["excluded_payloads_parsed"] == 0
    ref = pd.read_parquet(a.arena / "v1/reference.parquet").fillna("")
    part = np.load(a.arena / "finalize_v1/s1_partition.npy", allow_pickle=False)
    train_s1 = np.load(a.data / "train_s1.npy", allow_pickle=False)
    screen = np.load(a.data / "screen_s1.npy", allow_pickle=False)
    counts_frame = pd.read_parquet(a.out / "selected_true_count.parquet")
    counts = np.full(len(ref), -1, np.int32)
    counts[counts_frame.left_idx] = counts_frame.true_count
    selected = counts >= 0
    is_train = np.zeros(len(ref), bool)
    is_train[train_s1] = True
    brest = part == 1
    assert np.all(selected == (brest | is_train))
    assert np.array_equal(np.asarray(ref.entity_id)[counts_frame.left_idx], counts_frame.s1_id)
    raw_pool, verification = verified_pool(a)
    raw_ids = raw_pool["entity_id"].to_numpy(zero_copy_only=False)
    raw_index = pd.Index(raw_ids)
    assert raw_index.is_unique
    n_pool = len(raw_ids)
    truth = pd.read_parquet(a.out / "selected_truth_ids.parquet")
    truth_left = truth.left_idx.to_numpy()
    truth_right = raw_index.get_indexer(truth.rec_id)
    assert np.all(truth_right >= 0)
    raw_owner = np.full(n_pool, -1, np.int32)
    raw_owner[truth_right] = truth_left
    assert np.array_equal(np.bincount(truth_left, minlength=len(ref))[selected], counts[selected])
    raw_cc = raw_pool["cc"].to_numpy()
    ref_cc = np.asarray(ref.country.map({"US": 0, "India": 1}), np.int8)
    assert np.all(ref_cc[truth_left] == raw_cc[truth_right])
    pool_map, verification["legacy_pool"] = pool_mapping(a.arena / "v1/pool.parquet", raw_ids)
    modern_map, verification["modern_pool"] = pool_mapping(a.u / "data/arena/pool.parquet", raw_ids)
    assert np.array_equal(pool_map, modern_map)
    modern_ref_ids = pq.read_table(a.u / "data/arena/reference.parquet", columns=["entity_id"])["entity_id"].to_numpy()
    assert np.array_equal(modern_ref_ids, ref.entity_id)
    extras = json.loads((a.u / "prep/brest_u8t/extra_rec_ids.json").read_text())
    extra_raw = raw_index.get_indexer(extras)
    assert np.all(extra_raw >= 0) and len(extras) == len(set(extras))
    modern_map = np.r_[modern_map, extra_raw].astype(np.int32)
    assert len(np.unique(modern_map)) == len(modern_map)
    verification["modern_extra_records"] = len(extras)
    verification["modern_right_code_count"] = len(modern_map)
    save_json(a.out / "source_verification.json", {"s1": guard["s1_source_verification"], "pool": verification,
        "scope": "All raw TSVs streamed locally. Full pool ID order/country counts and distributed raw text samples verified; full text equality not claimed."})

    # Write all screen truth before the heavier diagnostics. Zero-link businesses
    # live in a separate population table, never as synthetic candidate pairs.
    screen_mask = np.zeros(len(ref), bool)
    screen_mask[screen] = True
    take = screen_mask[truth_left]
    screen_truth = pa.table({"left_idx": truth_left[take].astype(np.int32),
        "right_idx": truth_right[take].astype(np.int32), "s1_id": np.asarray(ref.entity_id)[truth_left[take]],
        "rec_id": raw_ids[truth_right[take]], "true_count": counts[truth_left[take]].astype(np.int16)})
    pq.write_table(screen_truth, a.out / "screen_truth.parquet", compression="zstd")
    pq.write_table(pa.table({"left_idx": screen.astype(np.int32), "s1_id": np.asarray(ref.entity_id)[screen],
        "true_count": counts[screen].astype(np.int16)}), a.out / "screen_true_count.parquet", compression="zstd")
    screen_manifest = {**guard["screen"], "purpose": "METRIC ACCOUNTING ONLY; never inject into retrieval",
        "left_idx": "arena v1 reference order = R/texts/train_s1.parquet order",
        "right_idx": "original RAW train_source2 order then train_source3; R/texts/train_pool.parquet order",
        "pool_rows": n_pool, "s2_rows": verification["train"]["s2_rows"],
        "pair_rows": screen_truth.num_rows, "population_file": "screen_true_count.parquet",
        "zero_link_businesses_have_no_pair_row": True,
        "files": {f: {"sha256": digest(a.out / f), "bytes": (a.out / f).stat().st_size}
                  for f in ("screen_truth.parquet", "screen_true_count.parquet")}}
    save_json(a.out / "screen_truth_manifest.json", screen_manifest)
    log("screen_truth_ready", **screen_manifest)

    world = np.load(a.data / "brest_world.npz", allow_pickle=False)
    # Do not materialize its full true_count array: it also stores excluded S1s.
    # Only the raw selected truth above supplies denominators for this audit.
    wl, wr = world["left"], modern_map[world["right"]]
    assert np.all(brest[wl]) and np.array_equal(world["population"].astype(bool), brest)
    assert np.array_equal(world["partition"], part)
    y = raw_owner[wr] == wl
    assert np.array_equal(y, world["label"].astype(bool)), "U8t labels disagree with raw selected truth"
    wkeys = pair_keys(wl, wr, n_pool)
    sorted_world = np.sort(wkeys)
    assert np.all(sorted_world[1:] != sorted_world[:-1])
    meta = json.loads((a.u / "prep/brest_u8t/meta.json").read_text())
    assert meta["rows"] == len(wl) and meta["n_rec"] == len(modern_map)
    route_sizes = {"dense_retained": len(wl) - sum(meta[k] for k in ("route_rows", "reverse_rows", "nat_rows")),
                   "routes": meta["route_rows"], "reverse": meta["reverse_rows"], "native": meta["nat_rows"]}
    assert list(route_sizes.values()) == [3895977, 689474, 452167, 28785]
    log("world_raw_labels_verified", rows=len(wl), positives=int(y.sum()), route_sizes=route_sizes)

    # Read candidate indices only, never the old all-partition label arrays.
    dense_parts, dense_n = [], np.zeros(len(ref), np.int32)
    dense_tp = np.zeros(len(ref), np.int32)
    for batch in pq.ParquetFile(a.arena / "v1/pairs.parquet").iter_batches(
            batch_size=524288, columns=["left_idx", "right_idx", "right_raw_idx"]):
        l = batch.column("left_idx").to_numpy()
        m = selected[l]
        if not m.any():
            continue
        l = l[m]
        raw = batch.column("right_raw_idx").to_numpy()[m]
        compact = batch.column("right_idx").to_numpy()[m]
        assert np.array_equal(pool_map[compact], raw)
        dense_parts.append(pair_keys(l, raw, n_pool))
        dense_n += np.bincount(l, minlength=len(ref)).astype(np.int32)
        dense_tp += np.bincount(l[raw_owner[raw] == l], minlength=len(ref)).astype(np.int32)
    dense = np.concatenate(dense_parts)
    del dense_parts
    dense.sort()
    assert np.all(dense[1:] != dense[:-1]) and np.all(dense_n[selected] == 50)
    in_dense = contains(dense, wkeys)
    truth_keys = pair_keys(truth_left, truth_right, n_pool)
    truth_in_dense = contains(dense, truth_keys)
    truth_in_union = contains(sorted_world, truth_keys)
    union_tp = np.bincount(wl[y], minlength=len(ref))
    assert np.all(union_tp[brest] <= counts[brest])
    country = ref.country.to_numpy()
    retrieval = {"scope": "same complete B_REST; raw labels; full 10,320,219-target search pool",
        "dense_top50": score_summary(dense_tp, dense_tp, counts, brest, country),
        "current_U8t_union": score_summary(union_tp, union_tp, counts, brest, country),
        "dense_candidates": int(dense_n[brest].sum()), "union_candidates": len(wl),
        "union_pairs_not_in_full_dense_top50": int((~in_dense).sum()),
        "union_only_true_links": int((y & ~in_dense).sum()),
        "dense_true_links_not_in_current_union": int((brest[truth_left] & truth_in_dense & ~truth_in_union).sum()),
        "union_S1_with_no_candidates": int((brest & (np.bincount(wl, minlength=len(ref)) == 0)).sum()),
        "by_candidate_source": {}, "cumulative_union_oracles": {}}
    start, cumulative_tp = 0, np.zeros(len(ref), np.int32)
    for route, rows in route_sizes.items():
        stop = start + rows
        ym = y[start:stop]
        if route == "dense_retained":
            assert in_dense[start:stop].all()
        else:
            expected_true = meta[{"routes": "route_true", "reverse": "reverse_true", "native": "nat_true"}[route]]
            assert int(ym.sum()) == expected_true
        cumulative_tp += np.bincount(wl[start:stop][ym], minlength=len(ref)).astype(np.int32)
        retrieval["by_candidate_source"][route] = {"rows": rows, "raw_true": int(ym.sum()),
            "raw_false": int((~ym).sum()), "in_full_dense_top50": int(in_dense[start:stop].sum())}
        retrieval["cumulative_union_oracles"][route] = score_summary(cumulative_tp, cumulative_tp, counts, brest, country)
        start = stop
    prob = world["baseline_p"]
    assert np.all(np.isfinite(prob)) and np.all((prob >= 0) & (prob <= 1))
    accept = decode(wl, wr, prob, np.float32(.725))
    pred_n = np.bincount(wl[accept], minlength=len(ref))
    pred_tp = np.bincount(wl[accept & y], minlength=len(ref))
    retrieval["saved_U8t_fixed_0.725"] = score_summary(pred_tp, pred_n, counts, brest, country)
    report_path = a.u / "u4/u8t__base_q_coll_ed_route_l1v2_rev3_nat/report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        picks = saved["thresholds"]["per_fold_threshold"]
        folds = world["folds"][wl]
        thresholds = np.asarray([picks[str(k)] for k in range(5)], np.float32)[folds]
        ca = decode(wl, wr, prob, thresholds)
        retrieval["saved_U8t_cross_threshold"] = score_summary(
            np.bincount(wl[ca & y], minlength=len(ref)), np.bincount(wl[ca], minlength=len(ref)), counts, brest, country)
        retrieval["saved_U8t_cross_threshold"]["saved_thresholds_reused_without_tuning"] = picks
        retrieval["saved_U8t_cross_threshold"]["delta_vs_stored_report"] = (
            retrieval["saved_U8t_cross_threshold"]["macro_f05"] - saved["result"]["macro_f05"])
        retrieval["current_scorer_provenance"] = {"protocol": saved["protocol"], "variant": saved["variant"],
            "fit_rows_cap": saved["fit_rows_cap"], "oof_sha256": saved["oof_sha256"],
            "source": str(report_path), "interpretation": "U8t is a union-scored OOF tree; this is distinct from the A-only dense-positive neural training pack."}
    f, oracle = retrieval["saved_U8t_fixed_0.725"]["macro_f05"], retrieval["current_U8t_union"]["macro_f05"]
    retrieval["fixed_baseline_loss_decomposition"] = {"total_loss": 1 - f, "current_union_retrieval_loss": 1 - oracle,
        "retrieved_pair_scoring_and_decision_gap": oracle - f,
        "retrieval_fraction_of_current_total_macro_loss": (1 - oracle) / (1 - f),
        "definition": "(1 - current union perfect-classification oracle) / (1 - saved fixed baseline); descriptive bound, not an intervention estimate"}
    save_json(a.out / "retrieval.json", retrieval)
    log("retrieval", **{k: retrieval[k] for k in ("dense_top50", "current_U8t_union", "fixed_baseline_loss_decomposition")})

    train = pd.read_parquet(a.data / "train.parquet")
    tl = train.left_idx.to_numpy()
    tr = pool_map[train.right_idx.to_numpy()]
    assert np.all(is_train[tl]) and np.all(part[tl] == 0)
    assert np.array_equal(np.asarray(ref.entity_id)[tl], train.s1_id)
    assert np.array_equal(raw_ids[tr], train.rec_id)
    ty = raw_owner[tr] == tl
    assert np.array_equal(ty, train.label.astype(bool)) and np.array_equal(counts[tl], train.true_count)
    tk = np.sort(pair_keys(tl, tr, n_pool))
    assert np.all(tk[1:] != tk[:-1]) and contains(dense, tk).all()
    train_truth = is_train[truth_left]
    packed_truth = contains(tk, truth_keys)
    missing = train_truth & ~packed_truth
    assert ty.sum() == 678887 and missing.sum() == 11387
    assert np.array_equal(packed_truth[train_truth], truth_in_dense[train_truth])
    assert np.array_equal(train.left_text, np.asarray(ref.business_name + " | " + ref.business_address)[tl])
    actual_train_text = pc.replace_substring(raw_pool["text"].take(pa.array(tr)), "passage: ", "", max_replacements=1).to_numpy()
    assert np.array_equal(train.right_text, actual_train_text)
    del actual_train_text
    training = {"s1": len(train_s1), "pairs": len(train), "raw_true_links": int(train_truth.sum()),
        "training_positives_verified_against_raw": int(ty.sum()), "training_negatives_verified_against_raw": int((~ty).sum()),
        "missing_raw_true_links": int(missing.sum()), "missing_fraction": float(missing.sum() / train_truth.sum()),
        "S1_with_any_missing_positive": int(len(np.unique(truth_left[missing]))),
        "all_dense_retrieved_A_positives_in_pack": True, "all_training_pairs_in_dense_top50": True,
        "raw_true_count_zero_training_S1": int((counts[train_s1] == 0).sum()),
        "non_singleton_training_S1_with_zero_retrieved_positives": int(((counts > 0) & is_train & (dense_tp == 0)).sum()),
        "source_of_truth": "allowlisted raw train_ground_truth.tsv; all pack labels and IDs rechecked",
        "train_sha256": digest(a.data / "train.parquet"), "available_A_routes": {}}
    routes_a = {}
    reverse_path = a.texts.parent / "out/r_reverse_pairs_a.parquet"
    rv = pd.read_parquet(reverse_path, columns=["left_idx", "raw_row_idx", "s1_id", "rec_id"])
    assert np.all(part[rv.left_idx] == 0)
    assert np.array_equal(np.asarray(ref.entity_id)[rv.left_idx], rv.s1_id)
    assert np.array_equal(raw_ids[rv.raw_row_idx], rv.rec_id)
    rv = rv[is_train[rv.left_idx]]
    routes_a["existing_R_reverse"] = pair_keys(rv.left_idx.to_numpy(), rv.raw_row_idx.to_numpy(), n_pool)
    nat_path = a.u / "nat/arena_candidates.parquet"
    # The historical 'true' column was populated for B_REST only (A is false in
    # that column). Never consume it; independently label selected A pairs below.
    nat = pd.read_parquet(nat_path, columns=["s1", "raw", "rank", "s1_id", "rec_id"])
    nat = nat[is_train[nat.s1]]
    assert np.array_equal(np.asarray(ref.entity_id)[nat.s1], nat.s1_id)
    assert np.array_equal(raw_ids[nat.raw], nat.rec_id)
    for name, subset in (("existing_native_rank0", nat[nat["rank"] == 0]), ("existing_native_all_saved_ranks", nat)):
        routes_a[name] = pair_keys(subset.s1.to_numpy(), subset.raw.to_numpy(), n_pool)
    routes_a["existing_reverse_plus_native_rank0"] = np.union1d(routes_a["existing_R_reverse"], routes_a["existing_native_rank0"])
    for name, keys in routes_a.items():
        training["available_A_routes"][name] = route_report(keys, dense, raw_owner, tk, n_pool)
    dictionary = json.loads((a.u / "nat/dictionary.json").read_text())
    training["native_dictionary_provenance"] = {k: v for k, v in dictionary.items() if k != "dictionary"}
    training["native_candidate_label_column_warning"] = "Ignored stored true column: generation code labels B_REST only, so selected A labels were recomputed from raw truth."
    training["previous_reverse_augmentation"] = {"novel": False, "user_reported_path": "/dev/shm/c2v2/aug on H100",
        "CPU_R_reverse_artifact_verified": str(reverse_path), "H100_artifact_overlap_checked": False,
        "scope": "Missing from this new pack does not mean never used by previous models."}

    # Only materialize target strings that are used by the bounded feature audit.
    union_only = ~in_dense
    used_raw = np.unique(np.r_[tr, truth_right, wr[union_only]])
    used_table = raw_pool.take(pa.array(used_raw))
    texts = used_table["text"].to_pylist()
    fixes = {r["raw_row_idx"]: r for r in json.loads((a.out / "raw_field_exceptions.json").read_text())["rows"]}
    names, addresses = split_raw_text(texts, used_raw, fixes)
    rs = TextSide(names, addresses)
    fs = TextSide(ref.business_name.to_numpy(), ref.business_address.to_numpy())
    raw_to_side = np.full(n_pool, -1, np.int32)
    raw_to_side[used_raw] = np.arange(len(used_raw), dtype=np.int32)
    del texts, names, addresses, used_table
    source = np.where(np.arange(n_pool) < verification["train"]["s2_rows"], "S2", "S3")
    groups = {"train_positive": (tl[ty], tr[ty]), "train_negative": (tl[~ty], tr[~ty]),
        "union_only_B_REST_true": (wl[union_only & y], wr[union_only & y]),
        "union_only_B_REST_false": (wl[union_only & ~y], wr[union_only & ~y]),
        "missing_A_true": (truth_left[missing], truth_right[missing]),
        "current_union_missed_B_REST_true": (truth_left[brest[truth_left] & ~truth_in_union], truth_right[brest[truth_left] & ~truth_in_union])}
    summaries, examples = {}, {}
    missing_features = None
    for name, (l, r) in groups.items():
        ff = feature_frame(l, raw_to_side[r], fs, rs, country[l], source[r])
        summaries[name] = feature_summary(ff)
        if name == "missing_A_true":
            missing_features = ff
        examples[name] = []
        for flag in ("native_name", "right_address_empty", "lexically_dissimilar_name_matching_address", "name_typo_1to2_edits"):
            for pos in np.flatnonzero(ff[flag].to_numpy())[:3]:
                ri = raw_to_side[r[pos]]
                examples[name].append({"flag": flag, "s1_id": ref.entity_id.iloc[l[pos]], "rec_id": raw_ids[r[pos]],
                    "left_name": fs.name[l[pos]], "left_address": fs.address[l[pos]],
                    "right_name": rs.name[ri], "right_address": rs.address[ri],
                    "name_ratio": float(ff.name_ratio.iloc[pos]), "address_token_set_ratio": float(ff.address_token_set_ratio.iloc[pos])})
        log("feature_group", group=name, pairs=len(l))
    definitions = {"normalization": "Unicode NFKC + casefold + non-word/underscore to spaces; no transliteration",
        "native_name": "Indic/selected South-Asian Unicode ranges in either raw name; not merely non-ASCII",
        "lexically_dissimilar_name_matching_address": "name ratio <0.5; both addresses nonempty; address token-set >=0.9; no conflicting first digit group; alias proxy, not a semantic alias label",
        "name_typo_1to2_edits": "normalized name Levenshtein distance 1 or 2; typo-like, not proof of an actual typo",
        "near_name_same_address": "different normalized names with ratio >=0.85 and identical nonempty normalized address",
        "flags_overlap": True, "quantiles": "all pairs in each group, not a sample",
        "union_only": "U8t pairs absent from actual full legacy dense top-50, not merely appended file rows"}
    save_json(a.out / "feature_shift.json", {"definitions": definitions, "groups": summaries,
        "B_REST_usage": "diagnostics only; no training labels or training records generated from B_REST"})
    save_json(a.out / "examples.json", examples)
    training["missing_types_overlapping"] = summaries["missing_A_true"]
    categories = np.full(int(missing.sum()), "other", object)
    for category, flag in (("typo_like", "name_typo_1to2_edits"), ("alias_proxy", "lexically_dissimilar_name_matching_address"),
                           ("empty_address", "right_address_empty"), ("native_script", "native_name")):
        categories[missing_features[flag].to_numpy()] = category
    training["missing_primary_types"] = dict(Counter(categories))
    training["primary_type_precedence"] = "native_script > empty_address > alias_proxy > typo_like > other"
    missing_frame = pd.DataFrame({"left_idx": truth_left[missing], "right_idx": truth_right[missing],
        "s1_id": np.asarray(ref.entity_id)[truth_left[missing]], "rec_id": raw_ids[truth_right[missing]],
        "source": source[truth_right[missing]], "primary_type": categories})
    missing_frame.to_parquet(a.out / "missing_A_truth_accounting.parquet", index=False)
    save_json(a.out / "training_signal.json", training)

    # Observable ambiguity: same S1 and identical raw target name/address/country/
    # source, but distinct target IDs have different raw labels. Group over the
    # entire current candidate world, not only the feature-distribution subset.
    all_wraw, inverse = np.unique(wr, return_inverse=True)
    raw_wtext = raw_pool["text"].take(pa.array(all_wraw)).to_numpy(zero_copy_only=False)
    raw_rnames, raw_raddresses = split_raw_text(raw_wtext, all_wraw, fixes)
    observable = pd.MultiIndex.from_arrays([raw_rnames, raw_raddresses, raw_cc[all_wraw], source[all_wraw]])
    text_codes, _ = pd.factorize(observable, sort=False)
    groups_key = pair_keys(wl, text_codes[inverse], int(text_codes.max()) + 1)
    ambiguity_rows = pd.DataFrame({"key": groups_key, "left": wl, "y": y.astype(np.int8),
                                  "accept": accept, "raw": wr})
    ag = ambiguity_rows.groupby("key", sort=False).agg(rows=("y", "size"), true=("y", "sum"), left=("left", "first"))
    mixed = ag[(ag.true > 0) & (ag.true < ag.rows)]
    mixed_keys = mixed.index.to_numpy()
    mixed_pair = np.isin(groups_key, mixed_keys)
    wempty = np.fromiter((not address.strip() for address in raw_raddresses), bool, len(raw_raddresses))[inverse]
    s1_names = pd.MultiIndex.from_arrays([country, ref.business_name])
    name_codes, _ = pd.factorize(s1_names, sort=False)
    mult = np.bincount(name_codes)[name_codes]
    same_raw_name = np.asarray(ref.business_name)[wl] == raw_rnames[inverse]
    empty_name = wempty & same_raw_name
    ambiguity = {"scope": "B_REST raw labels; current complete U8t candidate world; S1 name multiplicity over the 2,072,821 arena inputs (not all original S1s)",
        "identical_observable_pair_conflicts": {"groups": len(mixed), "S1": int(mixed.left.nunique()),
            "candidate_pairs": int(mixed_pair.sum()), "true_pairs": int(y[mixed_pair].sum()),
            "false_pairs": int((~y[mixed_pair]).sum()),
            "unavoidable_pair_classification_errors_for_same_output_per_text_group": int(np.minimum(mixed.true, mixed.rows - mixed.true).sum()),
            "empty_address_pairs": int((mixed_pair & wempty).sum()),
            "definition": "Same S1; raw target name,address,country,source identical; target IDs differ. Labels disagree.",
            "limitation": "Empirical ambiguity for a pair-text-only deterministic classifier. IDs, wider graph/context and unseen information are not ruled out; this is not a universal impossibility or macro-F0.5 bound."},
        "empty_address_exact_raw_name": {},
        "raw_singletons": {"A_selected": int((counts[train_s1] == 0).sum()), "B_REST": int((counts[brest] == 0).sum()),
                           "baseline_false_merges": int((pred_n[brest & (counts == 0)] > 0).sum())}}
    for name, mask in (("unique_among_arena_S1", empty_name & (mult[wl] == 1)),
                       ("shared_among_arena_S1", empty_name & (mult[wl] > 1))):
        ambiguity["empty_address_exact_raw_name"][name] = {"pairs": int(mask.sum()), "true": int(y[mask].sum()),
            "true_fraction": float(y[mask].mean()) if mask.any() else None, "accepted": int((mask & accept).sum()),
            "accepted_false": int((mask & accept & ~y).sum())}
    save_json(a.out / "ambiguity.json", ambiguity)

    elapsed = time.monotonic() - started
    result = {"scope": "200000 selected A plus complete B_REST; no excluded ground-truth payload parsed",
        "raw_training_signal": {k: training[k] for k in ("raw_true_links", "training_positives_verified_against_raw", "missing_raw_true_links", "missing_fraction", "missing_primary_types")},
        "retrieval": {k: retrieval[k] for k in ("dense_top50", "current_U8t_union", "saved_U8t_fixed_0.725", "fixed_baseline_loss_decomposition")},
        "screen_truth": screen_manifest, "available_A_routes": training["available_A_routes"],
        "minimal_next_experiment": {"hypothesis": "The new A pack misses informative positive/negative support from existing independent routes.",
            "changed_variable": "A training-pair support: add raw-verified existing reverse + native rank0 pairs absent from the dense-only pack; retain same A businesses, model, seed and training budget",
            "comparison": "Fixed B_REST screen, full competing pool and screen_true_count denominators; compare macro F0.5, false links/singletons and union-only true recall, then full B_REST if promising",
            "no_execution_here": True, "not_novel": "Reverse augmentation already existed; compare this exact pack, do not claim a system-first union scorer."},
        "limitations": ["Descriptive audit on reused development B_REST, not hidden test or fresh holdout accuracy.",
            "Raw TSVs were streamed; full pool ID order verified, text equality checked on distributed samples rather than every record.",
            "Alias/typo categories are explicit lexical proxies and overlap.",
            "Historical H100 reverse-augmentation membership was not inspected; current-pack absence is not system-wide absence.",
            "No GPU model was run and no training gain is measured by this audit.",
            "Only B_REST world labels were consumed; full-world stored true_count and legacy label/rec_owner arrays were not used."],
        "runtime": {"seconds": elapsed, "max_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "threads": 4, "arrow_io_threads": 2, "numpy": np.__version__, "pyarrow": pa.__version__},
        "code_sha256": digest(Path(__file__)),
        "files": {f: {"sha256": digest(a.out / f), "bytes": (a.out / f).stat().st_size} for f in
                  ("retrieval.json", "training_signal.json", "feature_shift.json", "ambiguity.json", "source_verification.json")}}
    save_json(a.out / "report.json", result)
    log("audit_complete", seconds=elapsed, missing_training_positive=int(missing.sum()),
        union_oracle=oracle, fixed_baseline=f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["inspect", "s1-evidence", "pool-evidence", "field-exceptions", "truth", "audit"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arena", type=Path, default=Path("/opt/az-ml-new/artifacts/arena"))
    p.add_argument("--u", type=Path, default=Path("/home/ubuntu/run27/U"))
    p.add_argument("--data", type=Path, default=Path("/home/ubuntu/takeover_20260927_1000/data"))
    p.add_argument("--texts", type=Path, default=Path("/home/ubuntu/run27/R/texts"))
    p.add_argument("--raw", type=Path, default=Path("data/raw/dataset"))
    p.add_argument("--gt", type=Path, default=Path("/home/ubuntu/run27/U/nat/gt/train_ground_truth.tsv"))
    a = p.parse_args()
    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    a.out.mkdir(parents=True, exist_ok=True)
    {"inspect": inspect, "s1-evidence": source_evidence, "pool-evidence": source_evidence,
     "field-exceptions": field_exceptions, "truth": read_selected_truth, "audit": audit}[a.command](a)


if __name__ == "__main__":
    main()
