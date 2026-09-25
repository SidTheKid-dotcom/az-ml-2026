"""PLAN_v2 A0 name-only ownership rule on text files: normalization, submission rewrite, pricing, TUNE check.

Empty-address records are almost always owned. For an S1 whose normalized name
is unique among its country's S1s, every empty-address candidate with exactly
that normalized name is added (the exact-name owner wins the record); links to
empty-address records whose name is shared by >= 2 S1s are dropped. The array
rule itself is :func:`ber.arena_decision.name_only_rule`.

Commands::

    python -m ber.name_only tune --tune-dir .../v2_p1_decision --train-dir .../train --report tune.json
    python -m ber.name_only test --base matching.tsv --candidate candidate_pairs.tsv \
        --test-dir .../test --out-dir output/u5_name_only [--tune-report tune.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from ber.arena_decision import entity_f05, evaluate, name_only_rule, one_owner

SUFFIX_TOKENS = frozenset(
    "the inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl sas sasu sa sci "
    "eurl gmbh".split())
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
EMPTY_ADDRESS_TOKENS = ("<NULL>", "##")
TEST_ROWS = 1_732_544
MATCHING_HEADER = "source1_entity_id\tmatched_entity_ids"
CANDIDATE_HEADER = "source1_entity_id\tcandidate_entity_ids"
# Per-link probabilities measured on TUNE (PLAN_v2 section 1).
DEFAULT_PROBABILITIES = {"add": 0.978, "drop_mult2": 0.469, "drop_mult3plus": 0.028, "owner_removed": 1 - 0.978}


def norm_name(value):
    """NFKD -> ASCII -> lowercase -> [a-z0-9 ] -> tokens minus legal-form/suffix tokens."""
    text = unicodedata.normalize("NFKD", "" if value is None else str(value)).encode("ascii", "ignore").decode().lower()
    return " ".join(w for w in _NON_ALNUM.sub(" ", text).split() if w not in SUFFIX_TOKENS)


def is_empty_address(value):
    """Missing, whitespace-only, or stripped value equal to "<NULL>" or "##"."""
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    return not stripped or stripped in EMPTY_ADDRESS_TOKENS


def name_multiplicity(countries, names):
    """Per row: number of rows sharing (country, normalized name)."""
    frame = pd.DataFrame({"c": np.asarray(countries, dtype=object), "n": np.asarray(names, dtype=object)})
    return frame.groupby(["c", "n"], sort=False)["n"].transform("size").to_numpy(dtype=np.int64)


def _read_options(columns):
    return dict(parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                                     include_columns=list(columns)))


def read_source(path, columns):
    return pacsv.read_csv(path, **_read_options(columns))


def empty_address_names(paths, keep_ids=None):
    """Map entity_id -> normalized name for every empty-address record (streamed, low memory)."""
    out = {}
    columns = ("entity_id", "business_name", "business_address")
    for path in paths:
        reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=64 << 20), **_read_options(columns))
        for batch in reader:
            ids = batch.column("entity_id").to_pylist()
            addresses = batch.column("business_address").to_pylist()
            names = batch.column("business_name").to_pylist()
            for entity, address, name in zip(ids, addresses, names):
                if is_empty_address(address) and (keep_ids is None or entity in keep_ids):
                    out[entity] = norm_name(name)
    return out


def iter_id_rows(path, header):
    """Yield (source1_id, [ids]) from a results-style TSV; tolerates CRLF, checks the exact header."""
    with open(path, encoding="utf-8", newline="") as handle:
        first = handle.readline().rstrip("\r\n")
        if first != header:
            raise ValueError(f"{path}: header {first!r} != {header!r}")
        for line in handle:
            s1, tab, rest = line.rstrip("\r\n").partition("\t")
            if not tab:
                raise ValueError(f"{path}: malformed row {line[:80]!r}")
            yield s1, (rest.split(",") if rest else [])


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _poisson_binomial(probabilities):
    dist = np.ones(1)
    for p in probabilities:
        dist = np.convolve(dist, [1.0 - p, p])
    return dist


def expected_row_delta(kept, p_removed, p_added):
    """Expected per-S1 F0.5 change when links are removed and added (other kept links assumed true).

    ``p_removed``/``p_added`` are the probabilities that each removed/added link
    is true; T = kept + true removed + true added. For a single change this is
    exactly the manual's per-link value at the row's list length.
    """
    removed, added = _poisson_binomial(p_removed), _poisson_binomial(p_added)
    t_r = np.arange(len(removed))[:, None]
    t_a = np.arange(len(added))[None, :]
    total = kept + t_r + t_a
    before = entity_f05(kept + t_r, kept + len(p_removed), total)
    after = entity_f05(kept + t_a, kept + len(p_added), total)
    return float((removed[:, None] * added[None, :] * (after - before)).sum())


def per_link_value(k, p_true, action):
    """Manual section 4 expected value of one change on a list of length k (other links correct)."""
    if action == "add":
        return p_true / (5 * k + 1) - (1 - p_true) * 4 / (5 * k + 4)
    if k < 1:
        raise ValueError("cannot remove from an empty list")
    return -p_true / (5 * k - 4) + (1 - p_true) * 4 / (5 * k - 1)


def _drop_probability(mult, probabilities):
    return probabilities["drop_mult2"] if mult == 2 else probabilities["drop_mult3plus"]


def price_rows(rows_kept, removed, added, probabilities):
    """Sum of exact row expectations and of per-link manual values (base list length)."""
    exact = per_link = 0.0
    for row, kept in rows_kept.items():
        rem, add = removed.get(row, []), added.get(row, [])
        p_rem = [probabilities[kind] if kind != "drop" else _drop_probability(m, probabilities) for kind, m in rem]
        p_add = [probabilities["add"]] * len(add)
        exact += expected_row_delta(kept, p_rem, p_add)
        k = kept + len(rem)
        per_link += sum(per_link_value(k, p, "remove") for p in p_rem) + sum(per_link_value(k, p, "add") for p in p_add)
    return exact, per_link


def _log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ----------------------------------------------------------------------------- TUNE reproduction

def tune_check(tune_dir, train_dir, threshold=0.775):
    """Reproduce v1 0.968992, one-owner 0.969029 and the name-only rule +0.003962 on saved TUNE pairs."""
    tune_dir, train_dir = Path(tune_dir), Path(train_dir)
    cols = ["source1_entity_id", "candidate_entity_id", "right_idx", "original_blend_probability",
            "is_truth_for_scoring_only", "baseline_accept"]
    pairs = pd.read_parquet(tune_dir / "tune_pairs.parquet", columns=cols)
    refs = pd.read_parquet(tune_dir / "tune_references.parquet", columns=["source1_entity_id", "country", "true_count"])
    s1_idx = pd.Index(refs.source1_entity_id).get_indexer(pairs.source1_entity_id)
    if (s1_idx < 0).any():
        raise ValueError("tune pair S1 missing from tune references")
    s1 = read_source(train_dir / "train_source1.tsv", ("entity_id", "business_name", "country")).to_pandas()
    s1["nn"] = [norm_name(x) for x in s1.business_name]
    s1["mult"] = name_multiplicity(s1.country, s1.nn)
    s1 = s1.set_index("entity_id")
    ref_nn = s1.nn.reindex(refs.source1_entity_id).to_numpy()
    ref_mult = s1["mult"].reindex(refs.source1_entity_id).to_numpy()
    empties = empty_address_names([train_dir / f"train_source{i}.tsv" for i in (2, 3)],
                                  keep_ids=set(pairs.candidate_entity_id))
    rec_nn = pairs.candidate_entity_id.map(empties)
    rec_empty = rec_nn.notna().to_numpy()
    exact = rec_empty & (rec_nn.to_numpy() == ref_nn[s1_idx])
    mult = ref_mult[s1_idx].astype(np.int64)
    label = pairs.is_truth_for_scoring_only.to_numpy()
    blend = pairs.original_blend_probability.to_numpy()
    rec_idx = pairs.right_idx.to_numpy()
    true_count = refs.true_count.to_numpy()
    base = blend >= threshold
    common = dict(s1_idx=s1_idx, label=label, true_count=true_count, country=refs.country.to_numpy())
    res = {"threshold": threshold, "baseline_matches_saved_flag": bool((base == pairs.baseline_accept.to_numpy()).all())}
    base_eval = evaluate(base, **common)
    res["v1"] = base_eval
    owner = one_owner(base, s1_idx, rec_idx, blend)
    removed = base & ~owner
    res["one_owner"] = evaluate(owner, **common)
    res["one_owner"].update(removed_links=int(removed.sum()), removed_true_links=int((removed & label).sum()),
                            contested_records=int(len(np.unique(rec_idx[removed]))))
    for owner_wins in (False, True):
        new, info = name_only_rule(base, s1_idx, rec_idx, rec_empty, exact, mult, owner_wins=owner_wins)
        ev = evaluate(new, **common)
        key = f"name_only_owner_wins_{str(owner_wins).lower()}"
        res[key] = {"macro_f05": ev["macro_f05"], "delta_vs_v1": ev["macro_f05"] - base_eval["macro_f05"],
                    "precision": ev["precision"], "recall": ev["recall"], "false_links": ev["false_links"],
                    "by_country": ev["by_country"], "added": int(info["added"].sum()),
                    "dropped": int(info["dropped"].sum()), "owner_removed": int(info["owner_removed"].sum()),
                    "conflicted_records": info["conflicted_records"]}
    new, info = name_only_rule(base, s1_idx, rec_idx, rec_empty, exact, mult, owner_wins=False)
    empty_name = ref_nn[s1_idx] == ""

    def precision(mask):
        return {"pairs": int(mask.sum()), "precision": float(label[mask].mean()) if mask.any() else None}

    ee = rec_empty & exact
    res["subset_precision"] = {
        "all_exact_unique": precision(ee & (mult == 1)),
        "all_exact_mult2": precision(ee & (mult == 2)),
        "all_exact_mult3plus": precision(ee & (mult >= 3)),
        "added_links": precision(info["added"]),
        "dropped_links_mult2": precision(info["dropped"] & (mult == 2)),
        "dropped_links_mult3plus": precision(info["dropped"] & (mult >= 3)),
        "dropped_links_empty_norm_name": precision(info["dropped"] & empty_name),
        "exact_pairs_empty_norm_name": precision(ee & empty_name),
    }
    # Owner-change probability: the other S1 held a record that its exact-unique owner claims.
    uniq_recs = np.unique(rec_idx[ee & (mult == 1)])
    held_by_other = base & ~(ee & (mult == 1)) & np.isin(rec_idx, uniq_recs)
    res["subset_precision"]["held_by_other_s1_of_exact_unique_records"] = precision(held_by_other)
    return res


# ----------------------------------------------------------------------------- test file

def build_test_file(base_path, candidate_path, test_dir, out_dir, probabilities=None, alt_probabilities=None,
                    validator=None, run_official=True, expected_rows=TEST_ROWS):
    probabilities = dict(DEFAULT_PROBABILITIES if probabilities is None else probabilities)
    test_dir, out_dir = Path(test_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _log("reading test Source 1")
    s1 = read_source(test_dir / "test_source1.tsv", ("entity_id", "business_name", "country"))
    s1_ids = s1.column("entity_id").to_pylist()
    countries = np.asarray(s1.column("country").to_pylist(), dtype=object)
    s1_nn = [norm_name(x) for x in s1.column("business_name").to_pylist()]
    del s1
    mult = name_multiplicity(countries, s1_nn)
    n = len(s1_ids)
    if len(set(s1_ids)) != n:
        raise ValueError("duplicate test Source 1 IDs")
    _log(f"{n} S1s; reading empty-address records")
    empties = empty_address_names([test_dir / f"test_source{i}.tsv" for i in (2, 3)])
    _log(f"{len(empties)} empty-address records; reading base file")
    base_rows = []
    for i, (entity, ids) in enumerate(iter_id_rows(base_path, MATCHING_HEADER)):
        if i >= n or entity != s1_ids[i]:
            raise ValueError(f"base file row {i + 2} is not test_source1 row {i + 2}")
        base_rows.append(ids)
    if len(base_rows) != n:
        raise ValueError("base file row count differs from test_source1")
    _log("streaming candidates for empty-address exact-name records")
    pair_s1, pair_id, pair_base = [], [], []
    for i, (entity, cands) in enumerate(iter_id_rows(candidate_path, CANDIDATE_HEADER)):
        if i >= n or entity != s1_ids[i]:
            raise ValueError(f"candidate row {i + 2} is not test_source1 row {i + 2}")
        base_ids = base_rows[i]
        cand_set = set(cands)
        if not set(base_ids) <= cand_set:
            raise ValueError(f"base row {i + 2} has IDs outside its candidate list")
        for rid in base_ids:
            pair_s1.append(i), pair_id.append(rid), pair_base.append(True)
        name = s1_nn[i]
        in_base = set(base_ids)
        for rid in cands:
            if rid not in in_base and empties.get(rid) == name:
                pair_s1.append(i), pair_id.append(rid), pair_base.append(False)
    _log(f"{len(pair_s1)} pairs in the decision table")
    pair_s1 = np.asarray(pair_s1, dtype=np.int64)
    pair_id = np.asarray(pair_id, dtype=object)
    pair_base = np.asarray(pair_base, dtype=bool)
    rec_idx, _ = pd.factorize(pair_id)
    rec_nn = pd.Series(pair_id).map(empties)
    rec_empty = rec_nn.notna().to_numpy()
    s1_nn_arr = np.asarray(s1_nn, dtype=object)
    exact = rec_empty & (rec_nn.to_numpy() == s1_nn_arr[pair_s1])
    pair_mult = mult[pair_s1]
    if len(np.unique(np.stack([pair_s1, rec_idx]), axis=1)[0]) != len(pair_s1):
        raise ValueError("duplicate (S1, record) pair in the decision table")
    new, info = name_only_rule(pair_base, pair_s1, rec_idx, rec_empty, exact, pair_mult, owner_wins=True)

    # Rebuild rows: kept base IDs in base order, then additions in candidate order.
    new_rows = list(base_rows)
    changed_rows = np.unique(pair_s1[new != pair_base])
    if np.any(np.diff(pair_s1) < 0):
        raise AssertionError("decision table is not in row order")
    starts = np.searchsorted(pair_s1, changed_rows, side="left")
    stops = np.searchsorted(pair_s1, changed_rows, side="right")
    for row, a, b in zip(changed_rows, starts, stops):
        keep, from_base = new[a:b], pair_base[a:b]
        ids = pair_id[a:b]
        new_rows[row] = list(ids[keep & from_base]) + list(ids[keep & ~from_base])
    out_path = out_dir / "matching_results.tsv"
    _log(f"writing {out_path}")
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(MATCHING_HEADER + "\n")
        for entity, ids in zip(s1_ids, new_rows):
            handle.write(f"{entity}\t{','.join(ids)}\n")

    # Counts and pricing, by country.
    removed, added, kept = {}, {}, {}
    for row in changed_rows:
        kept[int(row)] = 0
    for flag, kind in ((info["dropped"], "drop"), (info["owner_removed"], "owner_removed")):
        for row, m in zip(pair_s1[flag], pair_mult[flag]):
            removed.setdefault(int(row), []).append((kind, int(m)))
    for row in pair_s1[info["added"]]:
        added.setdefault(int(row), []).append(("add", 1))
    kept_counts = np.bincount(pair_s1[pair_base & new], minlength=n)
    for row in kept:
        kept[row] = int(kept_counts[row])
    base_len = np.fromiter((len(r) for r in base_rows), dtype=np.int64, count=n)
    new_len = np.fromiter((len(r) for r in new_rows), dtype=np.int64, count=n)
    empty_name = s1_nn_arr == ""
    groups = {"France": countries == "France", "US_India": countries != "France"}
    for c in np.unique(countries):
        groups[str(c)] = countries == c
    by_group = {}
    for g, row_mask in groups.items():
        pm = row_mask[pair_s1]
        rows_g = {r: kept[r] for r in kept if row_mask[r]}
        exact_pts, link_pts = price_rows(rows_g, removed, added, probabilities)
        entry = {
            "s1_rows": int(row_mask.sum()),
            "links_added": int((info["added"] & pm).sum()),
            "links_dropped": int((info["dropped"] & pm).sum()),
            "links_dropped_mult2": int((info["dropped"] & pm & (pair_mult == 2)).sum()),
            "links_dropped_mult3plus": int((info["dropped"] & pm & (pair_mult >= 3)).sum()),
            "links_dropped_empty_norm_name": int((info["dropped"] & pm & empty_name[pair_s1]).sum()),
            "owner_changes": int((info["owner_removed"] & pm).sum()),
            "rows_changed": int(row_mask[changed_rows].sum()),
            "rows_newly_empty": int((row_mask & (base_len > 0) & (new_len == 0)).sum()),
            "rows_newly_non_empty": int((row_mask & (base_len == 0) & (new_len > 0)).sum()),
            "links_before": int(base_len[row_mask].sum()),
            "links_after": int(new_len[row_mask].sum()),
            "longest_list_before": int(base_len[row_mask].max()) if row_mask.any() else 0,
            "longest_list_after": int(new_len[row_mask].max()) if row_mask.any() else 0,
            "expected_entity_points_row_exact": exact_pts,
            "expected_entity_points_per_link_sum": link_pts,
            "predicted_public_delta_row_exact": exact_pts / expected_rows,
            "predicted_public_delta_per_link_sum": link_pts / expected_rows,
        }
        if alt_probabilities:
            alt_exact, _ = price_rows(rows_g, removed, added, alt_probabilities)
            entry["predicted_public_delta_tune_conditional_probabilities"] = alt_exact / expected_rows
        by_group[g] = entry
    _log("strict validation")
    strict = strict_validate(out_path, candidate_path, test_dir / "test_source1.tsv", expected_rows)
    report = {
        "rule": "PLAN_v2 A0 name-only ownership rule on top of upload #3 (v1 + one-owner)",
        "output": str(out_path), "sha256": file_sha256(out_path), "bytes": out_path.stat().st_size,
        "inputs": {"base": str(base_path), "base_sha256": file_sha256(base_path),
                   "candidate": str(candidate_path), "candidate_sha256": file_sha256(candidate_path),
                   "test_dir": str(test_dir)},
        "probabilities": probabilities, "alt_probabilities_tune_conditional": alt_probabilities,
        "decision_table_pairs": int(len(pair_s1)), "empty_address_records": len(empties),
        "conflicted_records_skipped": info["conflicted_records"],
        "contested_ids": strict["contested_ids"],
        "by_group": by_group,
        "strict_validation": strict,
        "pricing_note": ("row_exact = exact expectation per changed S1 under independent Bernoulli links, other "
                         "links assumed true (equals the manual per-link value when a row has one change); "
                         "per_link_sum = sum of manual per-link values at the base list length. Divided by "
                         f"{expected_rows}; assumes the public subset mirrors the full test set."),
    }
    if run_official:
        _log("official validator")
        report["official_validator"] = run_official_validator(out_path, test_dir, out_dir, validator)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log("done")
    return report


def strict_validate(matching, candidate, source1, expected_rows=TEST_ROWS):
    """Row order/count, header, LF-only, no duplicate IDs per row, subset of candidates, no contested IDs."""
    from ber.validate_release import stream_check

    counts = dict(rows_checked=0, matched_links=0, candidate_links=0, empty_match_rows=0)
    stream_check(matching, candidate, source1, expected_rows, counts)
    raw = Path(matching).read_bytes()
    seen, contested, longest = set(), 0, 0
    for _, ids in iter_id_rows(matching, MATCHING_HEADER):
        longest = max(longest, len(ids))
        for rid in ids:
            if rid in seen:
                contested += 1
            seen.add(rid)
    result = {"passed": contested == 0 and b"\r" not in raw and raw.endswith(b"\n"), "counts": counts,
              "contested_ids": contested, "crlf_bytes": raw.count(b"\r"), "ends_with_newline": raw.endswith(b"\n"),
              "longest_list": longest, "distinct_matched_ids": len(seen)}
    if not result["passed"]:
        raise ValueError(f"strict validation failed: {result}")
    return result


def run_official_validator(matching, test_dir, out_dir, validator=None):
    """Unchanged official validator with --check-ids.

    ``--candidate`` is not passed: its candidate cross-check holds 86.6M IDs as
    Python sets (~9 GB), beyond this machine's budget; the subset check is done
    by :func:`strict_validate`, and the candidate file is the v1 file that
    already passed. Run from ``out_dir`` so the default candidate path is absent.
    """
    validator = Path(validator or Path(test_dir).parents[1] / "utils" / "validate_submission.py")
    command = ["python3", str(validator), "--matching", str(Path(matching).resolve()),
               "--test-dir", str(Path(test_dir).resolve()), "--check-ids"]
    done = subprocess.run(command, cwd=out_dir, capture_output=True, text=True, check=False)
    lines = done.stdout.splitlines()
    return {"command": command, "cwd": str(out_dir), "exit_code": done.returncode, "stdout": lines,
            "stderr": done.stderr.splitlines(), "validator_sha256": file_sha256(validator),
            "note": ("--candidate omitted: the validator's candidate sets need ~9.1 GiB (measured 0.52 GiB per "
                     "100k rows), over the 5 GB local budget. Matches-subset-of-candidates, row order and "
                     "contested IDs are enforced by strict_validation; the candidate file is the v1 file "
                     "(sha256 fc0ec7a4...) that passed the full validator with --candidate --check-ids."),
            "passed": done.returncode == 0 and any(line.startswith("PASS") for line in lines)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune")
    tune.add_argument("--tune-dir", required=True)
    tune.add_argument("--train-dir", required=True)
    tune.add_argument("--report", required=True)
    test = sub.add_parser("test")
    for name in ("base", "candidate", "test-dir", "out-dir"):
        test.add_argument(f"--{name}", required=True)
    test.add_argument("--tune-report", help="tune check JSON; its conditional precisions give a sensitivity price")
    test.add_argument("--validator")
    args = parser.parse_args(argv)
    if args.command == "tune":
        result = tune_check(args.tune_dir, args.train_dir)
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: result[k] for k in result if k.startswith("name_only")}, indent=1))
        return 0
    alt = None
    if args.tune_report:
        sp = json.loads(Path(args.tune_report).read_text())["subset_precision"]
        alt = {"add": sp["added_links"]["precision"], "drop_mult2": sp["dropped_links_mult2"]["precision"],
               "drop_mult3plus": sp["dropped_links_mult3plus"]["precision"],
               "owner_removed": sp["held_by_other_s1_of_exact_unique_records"]["precision"]
               if sp["held_by_other_s1_of_exact_unique_records"]["pairs"] else DEFAULT_PROBABILITIES["owner_removed"]}
    report = build_test_file(args.base, args.candidate, args.test_dir, args.out_dir, alt_probabilities=alt,
                             validator=args.validator)
    print(json.dumps({"sha256": report["sha256"], "by_group": report["by_group"],
                      "official_passed": report.get("official_validator", {}).get("passed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
