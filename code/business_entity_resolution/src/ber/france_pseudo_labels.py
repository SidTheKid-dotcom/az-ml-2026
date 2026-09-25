"""Label-free, multi-view France pseudo-labels on the test candidates (PLAN_v2 section 5).

Views that must agree for a pseudo-positive (P+) pair (French S1, record):
  * exact key (normalized name + address digits) equal, and the S1 key unique among
    French S1 keys;
  * street-token Jaccard >= 0.5 (French-normalized street tokens);
  * v1 blend >= 0.9;
  * no other S1 claims the record with v1 blend >= 0.5;
  * the record address is not empty.
Pseudo-negatives (P-):
  * ``rival_of_pplus``: every other French S1 holding a P+ record among its v1
    candidates (one owner per record);
  * ``template_same_city_number_diff_street``: candidate pairs whose names share a
    template (equal normalized names, or equal length >= 2 tokens differing in
    exactly one token), whose S1 city is among the record's admin phrases, whose
    address digit groups are equal as multisets (same house number), and whose
    street tokens are disjoint and dissimilar (Jaccard 0, sorted-token ratio < 0.6,
    both non-empty).
A pair that qualifies as both P+ and P- is dropped (counted). 10% of French S1s are a
pseudo-holdout (sha256(entity_id) mod 10 == 0); a pair is ``holdout`` when its S1
or (for rival negatives) the P+ owner is held out.

These labels come from the unlabeled supplied test text and the v1 scores; they are
not French ground truth, and P+ is by construction a set v1 already accepts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .forensic_features import (EMPTY, StreetVocabulary, address_kinds, default_vocabulary,
                                digit_strings, exact_keys, fold_ascii, forensic_features,
                                normalize_names, s1_multiplicities, template_similar)

COUNTRY = "France"
N_PAIRS = 86_627_200
P_BLEND = 0.9
RIVAL_BLEND = 0.5
STREET_JACCARD = 0.5
DIFF_STREET_RATIO = 0.6
HOLDOUT_MOD = 10
V1_THRESHOLD = 0.775
SCORE_FILES = ("probabilities.npy", "base_probabilities.npy", "scored_pairs.parquet")


def holdout_flags(entity_ids) -> np.ndarray:
    return np.fromiter((int(hashlib.sha256(e.encode()).hexdigest()[:12], 16) % HOLDOUT_MOD == 0
                         for e in entity_ids), dtype=bool, count=len(entity_ids))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_score_artifacts(scores: Path) -> dict[str, str]:
    """Verify the three aligned v1 artifacts against their copied hash manifest."""
    manifest = scores / "source.sha256"
    if not manifest.is_file():
        raise ValueError(f"missing score artifact manifest: {manifest}")
    expected = {}
    for line in manifest.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or Path(parts[1]).name != parts[1]:
            raise ValueError("invalid source.sha256 entry")
        expected[parts[1]] = parts[0].lower()
    if set(expected) != set(SCORE_FILES):
        raise ValueError(f"source.sha256 must identify exactly {SCORE_FILES}")
    for name in SCORE_FILES:
        path = scores / name
        if not path.is_file() or _sha256(path) != expected[name]:
            raise ValueError(f"score artifact hash mismatch: {name}")
    return expected


def verify_dense_artifacts(dense: Path, reference_rows: int, pool_rows: int) -> dict:
    """Check the completed dense bundle's label-free and pair-alignment contract."""
    complete_path = dense / "complete.json"
    if not complete_path.is_file():
        raise ValueError(f"missing dense completion marker: {complete_path}")
    complete = json.loads(complete_path.read_text())
    metadata = complete.get("metadata", {})
    config = complete.get("config", {})
    expected = {
        "state": complete.get("state") == "complete",
        "labels_accessed": config.get("labels_accessed") is False,
        "positive_candidate_injection": config.get("positive_candidate_injection") is False,
        "references": metadata.get("references") == reference_rows,
        "pool_records": metadata.get("pool_records") == pool_rows,
        "candidate_pairs": metadata.get("candidate_pairs") == N_PAIRS,
        "pairs_unique": metadata.get("pairs_unique") is True,
        "pair_order": metadata.get("pair_order") == ["left_idx", "right_idx"],
    }
    failed = [name for name, valid in expected.items() if not valid]
    if failed:
        raise ValueError(f"dense artifact contract failed: {', '.join(failed)}")
    return complete


def load_french_pool(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Pool positions (right_idx) of French records and their text."""
    file = pq.ParquetFile(path)
    positions, frames, offset = [], [], 0
    for group in range(file.num_row_groups):
        country = file.read_row_group(group, columns=["country"]).column(0)
        mask = pc.equal(country, COUNTRY)
        hits = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        if len(hits):
            table = file.read_row_group(group, columns=["entity_id", "business_name", "business_address"])
            frames.append(table.filter(mask).to_pandas())
            positions.append(hits + offset)
        offset += len(country)
    if not positions:
        empty = pd.DataFrame(columns=["entity_id", "business_name", "business_address"])
        return np.empty(0, dtype=np.int64), empty
    return np.concatenate(positions), pd.concat(frames, ignore_index=True)


def stream_french_pairs(pairs: Path, blend: np.ndarray, tree: np.ndarray, french_left: np.ndarray,
                        npool: int, expected_pairs: int = N_PAIRS) -> tuple[pd.DataFrame, np.ndarray]:
    """French-S1 pairs with scores; per-record count of claims >= RIVAL_BLEND over all pairs."""
    file = pq.ParquetFile(pairs)
    if (blend.ndim != 1 or tree.ndim != 1 or file.metadata.num_rows != expected_pairs
            or len(blend) != expected_pairs or len(tree) != expected_pairs):
        raise ValueError("scored pairs and score arrays are not the complete aligned v1 set")
    claims = np.zeros(npool, dtype=np.int32)
    parts, offset = [], 0
    for group in range(file.num_row_groups):
        table = file.read_row_group(group, columns=["left_idx", "right_idx"])
        left = table.column(0).to_numpy()
        right = table.column(1).to_numpy()
        b = np.asarray(blend[offset:offset + len(left)])
        t = np.asarray(tree[offset:offset + len(left)])
        if ((left < 0).any() or (left >= len(french_left)).any()
                or (right < 0).any() or (right >= npool).any()):
            raise ValueError("scored pair index outside reference or pool bounds")
        if (not np.isfinite(b).all() or not np.isfinite(t).all()
                or (b < 0).any() or (b > 1).any() or (t < 0).any() or (t > 1).any()):
            raise ValueError("score arrays must contain finite probabilities in [0, 1]")
        claims += np.bincount(right[b >= RIVAL_BLEND], minlength=npool).astype(np.int32)
        keep = french_left[left]
        if keep.any():
            rows = np.flatnonzero(keep)
            parts.append(pd.DataFrame({
                "pair_row": (rows + offset).astype(np.int64), "left_idx": left[rows], "right_idx": right[rows],
                "v1_blend": b[rows], "v1_tree": t[rows]}))
        offset += len(left)
    if offset != expected_pairs:
        raise ValueError("pair stream lost rows")
    columns = ["pair_row", "left_idx", "right_idx", "v1_blend", "v1_tree"]
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)), claims


def build(args) -> dict:
    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    vocab = StreetVocabulary.load(args.vocab) if args.vocab else default_vocabulary()
    cv = vocab[COUNTRY]
    score_hashes = verify_score_artifacts(args.scores)

    ref = pd.read_parquet(args.dense / "reference.parquet",
                          columns=["entity_id", "business_name", "business_address", "country"])
    french = (ref.country == COUNTRY).to_numpy()
    s1_idx = np.flatnonzero(french)
    s1 = ref.loc[french, ["entity_id", "business_name", "business_address"]].reset_index(drop=True)
    del ref
    mult = s1_multiplicities(s1.business_name, s1.business_address, COUNTRY)
    s1["norm_name"], s1["key"], s1["key_count"] = mult.norm_name, mult.exact_key, mult.key_count
    s1["holdout"] = holdout_flags(s1.entity_id.tolist())
    s1_digits = digit_strings(fold_ascii(s1.business_address)).to_pylist()
    s1["digit_set"] = [" ".join(sorted(d.split())) for d in s1_digits]
    s1_admin = [cv.parse(t)[1] for t in fold_ascii(s1.business_address, keep_commas=True).to_pylist()]
    s1["city"] = [next(iter(sorted(cv.cities(a))), "") for a in s1_admin]
    local_s1 = np.full(len(french), -1, dtype=np.int64)
    local_s1[s1_idx] = np.arange(len(s1_idx))
    print(f"S1 France {len(s1)}; unique keys {(s1.key_count == 1).sum()}; "
          f"with city {(s1.city != '').sum()} ({time.monotonic() - started:.0f}s)", flush=True)

    pool_pos, rec = load_french_pool(args.dense / "pool.parquet")
    npool = pq.ParquetFile(args.dense / "pool.parquet").metadata.num_rows
    verify_dense_artifacts(args.dense, len(french), npool)
    if s1.entity_id.duplicated().any() or rec.entity_id.duplicated().any():
        raise ValueError("dense reference or pool contains duplicate entity IDs")
    local_rec = np.full(npool, -1, dtype=np.int64)
    local_rec[pool_pos] = np.arange(len(pool_pos))
    rec["norm_name"] = normalize_names(rec.business_name)
    rec["key"] = exact_keys(rec.business_name, rec.business_address, normalized_names=rec.norm_name.to_numpy())
    rec["kind"] = address_kinds(rec.business_address)
    rec_digits = digit_strings(fold_ascii(rec.business_address)).to_pylist()
    rec["digit_set"] = [" ".join(sorted(d.split())) for d in rec_digits]
    print(f"French pool records {len(rec)} ({time.monotonic() - started:.0f}s)", flush=True)

    blend = np.load(args.scores / "probabilities.npy", mmap_mode="r")
    tree = np.load(args.scores / "base_probabilities.npy", mmap_mode="r")
    pairs, claims = stream_french_pairs(args.scores / "scored_pairs.parquet", blend, tree, french, npool)
    pairs["s1"] = local_s1[pairs.left_idx.to_numpy()]
    pairs["r"] = local_rec[pairs.right_idx.to_numpy()]
    if (pairs.s1 < 0).any() or (pairs.r < 0).any():
        raise ValueError("a French S1 pair points outside the French pool")
    pairs["claims_ge_05"] = claims[pairs.right_idx.to_numpy()]
    print(f"French pairs {len(pairs)} ({time.monotonic() - started:.0f}s)", flush=True)

    # --- P+ -----------------------------------------------------------------------
    codes, _ = pd.factorize(pd.concat([s1.key, rec.key], ignore_index=True))
    s1_kid, rec_kid = codes[:len(s1)], codes[len(s1):]
    ps, pr = pairs.s1.to_numpy(), pairs.r.to_numpy()
    key_match = (s1_kid[ps] == rec_kid[pr]) & (s1.key.to_numpy()[ps] != "")
    unique_key = key_match & (s1.key_count.to_numpy()[ps] == 1)
    cand = pairs[unique_key].copy()
    feats = forensic_features(s1.business_name.to_numpy()[cand.s1], s1.business_address.to_numpy()[cand.s1],
                              rec.business_name.to_numpy()[cand.r], rec.business_address.to_numpy()[cand.r],
                              COUNTRY, vocab)
    cand["street_token_jaccard"] = feats.street_token_jaccard.to_numpy()
    cand["name_token_jaccard"] = feats.name_token_jaccard.to_numpy()
    cand["address_kind"] = feats.address_kind.to_numpy()
    own_claim = (cand.v1_blend >= RIVAL_BLEND).astype(np.int32)
    steps = {"key_match_pairs": int(key_match.sum()), "unique_key_match_pairs": int(len(cand))}
    m = (cand.street_token_jaccard >= STREET_JACCARD).to_numpy().copy()
    steps["street_jaccard_ge_0.5"] = int(m.sum())
    m &= (cand.v1_blend >= P_BLEND).to_numpy()
    steps["and_blend_ge_0.9"] = int(m.sum())
    m &= ((cand.claims_ge_05 - own_claim) == 0).to_numpy()
    steps["and_no_rival_ge_0.5"] = int(m.sum())
    m &= (cand.address_kind != EMPTY).to_numpy()
    steps["and_record_address_not_empty"] = int(m.sum())
    pos = cand[m].copy()
    if pos.r.duplicated().any():
        raise ValueError("a record has two P+ owners; exact-key uniqueness failed")
    pos["label"] = np.int8(1)
    pos["reasons"] = "unique_exact_key;street_jaccard>=0.5;blend>=0.9;no_rival>=0.5;address_not_empty"
    pos["owner_s1"] = pos.s1
    print(f"P+ {len(pos)} ({time.monotonic() - started:.0f}s)", flush=True)

    # --- P- rivals --------------------------------------------------------------------
    owner = np.full(len(rec), -1, dtype=np.int64)
    owner[pos.r.to_numpy()] = pos.s1.to_numpy()
    rival_mask = (owner[pr] >= 0) & (owner[pr] != ps)
    rivals = pairs[rival_mask].copy()
    rivals["owner_s1"] = owner[rivals.r.to_numpy()]
    rivals["label"] = np.int8(0)
    rivals["reasons"] = "rival_of_pplus_owner"

    # --- P- template / same city / same number / different street --------------------
    s1_ds, rec_ds = s1.digit_set.to_numpy(), rec.digit_set.to_numpy()
    same_digits = (s1_ds[ps] == rec_ds[pr]) & (s1_ds[ps] != "")
    sub = pairs[same_digits].copy()
    s1_names, rec_names = s1.norm_name.to_numpy(), rec.norm_name.to_numpy()
    templ = np.fromiter((template_similar(a, b) for a, b in
                         zip(s1_names[sub.s1.to_numpy()], rec_names[sub.r.to_numpy()])), dtype=bool, count=len(sub))
    sub = sub[templ].copy()
    s1_city = s1.city.to_numpy()
    rec_fold = fold_ascii(rec.business_address.to_numpy()[sub.r.to_numpy()], keep_commas=True).to_pylist()
    in_city = np.fromiter((c != "" and c in cv.parse(t)[1] for c, t in
                           zip(s1_city[sub.s1.to_numpy()], rec_fold)), dtype=bool, count=len(sub))
    sub = sub[in_city].copy()
    tf = forensic_features(s1.business_name.to_numpy()[sub.s1], s1.business_address.to_numpy()[sub.s1],
                           rec.business_name.to_numpy()[sub.r], rec.business_address.to_numpy()[sub.r],
                           COUNTRY, vocab)
    diff_street = ((tf.street_token_jaccard == 0) & (tf.street_token_ratio < DIFF_STREET_RATIO)).to_numpy()
    templates = sub[diff_street].copy()
    templates["label"] = np.int8(0)
    templates["reasons"] = "template_same_city_number_diff_street"
    templates["owner_s1"] = owner[templates.r.to_numpy()]
    template_steps = {"same_digit_multiset_pairs": int(same_digits.sum()), "and_template_name": int(templ.sum()),
                      "and_same_city": int(in_city.sum()), "and_different_street": int(diff_street.sum())}

    # --- merge ---------------------------------------------------------------------
    negatives = pd.concat([rivals, templates], ignore_index=True)
    both = negatives.pair_row.duplicated(keep=False).to_numpy()
    negatives.loc[both, "reasons"] = "rival_of_pplus_owner;template_same_city_number_diff_street"
    negatives = negatives.drop_duplicates("pair_row")
    conflict = negatives.pair_row.isin(pos.pair_row)
    dropped_conflicts = int(conflict.sum())
    negatives = negatives[~conflict]
    labels = pd.concat([pos, negatives], ignore_index=True)
    lab_s1, lab_r = labels.s1.to_numpy(), labels.r.to_numpy()
    feats = forensic_features(s1.business_name.to_numpy()[lab_s1], s1.business_address.to_numpy()[lab_s1],
                              rec.business_name.to_numpy()[lab_r], rec.business_address.to_numpy()[lab_r],
                              COUNTRY, vocab)
    s1_hold = s1.holdout.to_numpy()
    owner_s1 = labels.owner_s1.to_numpy().astype(np.int64)
    owner_hold = np.where(owner_s1 >= 0, s1_hold[np.maximum(owner_s1, 0)], False)
    result = pd.DataFrame({
        "s1_entity_id": s1.entity_id.to_numpy()[lab_s1],
        "record_entity_id": rec.entity_id.to_numpy()[lab_r],
        "label": labels.label.to_numpy().astype(np.int8),
        "reasons": labels.reasons.to_numpy(),
        "holdout": s1_hold[lab_s1] | owner_hold,
        "s1_holdout": s1_hold[lab_s1],
        "owner_entity_id": np.where(owner_s1 >= 0, s1.entity_id.to_numpy()[np.maximum(owner_s1, 0)], ""),
        "v1_blend": labels.v1_blend.to_numpy().astype(np.float32),
        "v1_tree": labels.v1_tree.to_numpy().astype(np.float32),
        "left_idx": labels.left_idx.to_numpy().astype(np.int32),
        "right_idx": labels.right_idx.to_numpy().astype(np.int32),
        "pair_row": labels.pair_row.to_numpy().astype(np.int64),
        "exact_key_equal": feats.exact_key_equal.to_numpy(),
        "name_token_jaccard": feats.name_token_jaccard.to_numpy(),
        "street_token_jaccard": feats.street_token_jaccard.to_numpy(),
        "street_token_ratio": feats.street_token_ratio.to_numpy(),
        "digit_relation": feats.digit_relation.to_numpy(),
    }).sort_values("pair_row", kind="stable").reset_index(drop=True)
    if result.duplicated(["s1_entity_id", "record_entity_id"]).any():
        raise ValueError("duplicate labeled pairs")
    result.to_parquet(out / "pseudo_labels.parquet", index=False)

    counts = summarize(result, s1, pairs, steps, template_steps, dropped_conflicts)
    counts["elapsed_seconds"] = round(time.monotonic() - started, 1)
    counts["inputs"] = {"scores": str(args.scores), "dense": str(args.dense),
                        "vocab": str(args.vocab or "default resources/forensic_vocab.json"),
                        "score_sha256": score_hashes}
    (out / "counts.json").write_text(json.dumps(counts, indent=2, allow_nan=False) + "\n")
    write_audit(result, s1, rec, local_s1, local_rec, out, args.seed)
    return counts


def summarize(result: pd.DataFrame, s1: pd.DataFrame, pairs: pd.DataFrame, steps: dict,
              template_steps: dict, dropped_conflicts: int) -> dict:
    pos, neg = result[result.label == 1], result[result.label == 0]
    hold = result.holdout.to_numpy()
    def reject_rate(frame: pd.DataFrame) -> float | None:
        return float((frame.v1_blend < V1_THRESHOLD).mean()) if len(frame) else None

    counts = {
        "french_s1": int(len(s1)), "french_s1_holdout": int(s1.holdout.sum()),
        "french_candidate_pairs": int(len(pairs)),
        "p_plus_funnel": steps, "p_minus_template_funnel": template_steps,
        "p_plus": int(len(pos)), "p_plus_s1": int(pos.s1_entity_id.nunique()),
        "p_plus_records": int(pos.record_entity_id.nunique()),
        "p_minus": int(len(neg)),
        "p_minus_rival": int(neg.reasons.str.contains("rival_of_pplus_owner").sum()),
        "p_minus_template": int(neg.reasons.str.contains("template_same_city").sum()),
        "p_minus_both_reasons": int(neg.reasons.str.contains(";").sum()),
        "dropped_p_plus_p_minus_conflicts": dropped_conflicts,
        "holdout": {"p_plus": int((pos.holdout).sum()), "p_minus": int((neg.holdout).sum()),
                    "pairs": int(hold.sum())},
        "p_plus_per_s1": {str(k): int(v) for k, v in
                          pos.groupby("s1_entity_id").size().value_counts().sort_index().items()},
        # v1 on its own pseudo-labels: P+ needs blend >= 0.9 so recall is 1 by construction.
        "v1_at_0.775_on_pseudo_labels": {
            "p_plus_accept_rate": float((pos.v1_blend >= V1_THRESHOLD).mean()) if len(pos) else None,
            "p_minus_reject_rate": reject_rate(neg),
            "p_minus_rival_reject_rate": reject_rate(neg[neg.reasons.str.contains("rival")]),
            "p_minus_template_reject_rate": reject_rate(neg[neg.reasons.str.contains("template")]),
            "holdout_p_minus_reject_rate": reject_rate(neg[neg.holdout]),
        },
        "caveat": ("Pseudo-labels from unlabeled test text and v1 scores; not French accuracy. "
                   "P+ requires v1 blend >= 0.9, so it cannot measure v1 recall."),
    }
    return counts


def write_audit(result, s1, rec, local_s1, local_rec, out: Path, seed: int, n: int = 200):
    s1_text = s1.set_index("entity_id")[["business_name", "business_address"]]
    rec_text = rec.set_index("entity_id")[["business_name", "business_address"]]
    for label, name in ((1, "audit_pos.txt"), (0, "audit_neg.txt")):
        part = result[result.label == label]
        sample = part.sample(n=min(n, len(part)), random_state=seed)
        lines = [f"# {name}: {len(sample)} random of {len(part)} (seed {seed}); label={label}", ""]
        for i, row in enumerate(sample.itertuples(index=False), 1):
            a = s1_text.loc[row.s1_entity_id]
            b = rec_text.loc[row.record_entity_id]
            lines += [f"[{i}] {row.reasons}  blend={row.v1_blend:.3f} tree={row.v1_tree:.3f} "
                      f"streetJ={row.street_token_jaccard:.2f} streetR={row.street_token_ratio:.2f} "
                      f"nameJ={row.name_token_jaccard:.2f} key={row.exact_key_equal} holdout={row.holdout}",
                      f"    S1  {row.s1_entity_id}: {a.business_name} | {a.business_address}",
                      f"    REC {row.record_entity_id}: {b.business_name} | {b.business_address}"]
            if row.owner_entity_id and row.owner_entity_id != row.s1_entity_id:
                o = s1_text.loc[row.owner_entity_id]
                lines.append(f"    OWN {row.owner_entity_id}: {o.business_name} | {o.business_address}")
            lines.append("")
        (out / name).write_text("\n".join(lines) + "\n")


def main():
    root = Path("/home/sidvish/Documents/Work/Personal/az-ml-new")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=root / "artifacts/v1_test_scores")
    parser.add_argument("--dense", type=Path, default=root / "artifacts/v1_test_scores/dense_test")
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=root / "output/france_pseudo")
    parser.add_argument("--seed", type=int, default=20260926)
    counts = build(parser.parse_args())
    print(json.dumps({k: counts[k] for k in ("p_plus", "p_plus_s1", "p_minus", "p_minus_rival",
                                             "p_minus_template", "holdout", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
