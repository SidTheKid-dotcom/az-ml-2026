"""F1: the S1 public file (0.978287) with one French change. French candidate pairs whose core name
(minus legal forms and corporate suffix words) is equal and whose address has the same digits and
similar text, with no corporate suffix word added, get plain-blend probability max(p, 0.99) before
one owner + name-only + cap. On arena B_REST US/India this class is 99.85-99.99% true (legal form
same/added/dropped/changed). US/India rows are untouched. No labels are read.
Parity: rebuilding the S1 file without the boost must reproduce its SHA256."""
import hashlib, json, re, subprocess, sys, unicodedata
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from rapidfuzz import fuzz
from ber.pair_tree_c2 import (COLUMNS, c2_blend, c2_on_rows, decode, v1_cache, write_tsv, validate_tsv,
                              by_country, sha256, log)
LEGAL = set("inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl sas sasu sa sci eurl ei snc gmbh opc lp pllc pc".split())
SUFFIX = set("group groupe holdings holding fils sons france india usa international cie associes developpement distribution".split())
FILLER = set("et and the".split())
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()
def name_info(s):
    t = norm(s); return (frozenset(x for x in t if x not in LEGAL | SUFFIX | FILLER), frozenset(x for x in t if x in SUFFIX))
def digits(s): return tuple(str(int(d)) for d in re.findall(r"\d+", s or ""))
def rule(ln, la, rn, ra):
    (lc, ls), (rc, rs) = ln, rn
    if not lc or lc != rc or (rs - ls): return False
    ld = digits(la)
    return bool(ld) and ld == digits(ra) and fuzz.token_set_ratio(" ".join(norm(la)), " ".join(norm(ra))) >= 70
S1_SHA = "559f37b9b66f4e18fc28c0b25ae60d08543c386e0bdb32c619990b4c9cbd1a78"
W = Path("/opt/az-ml-new/agents/pair_tree_c2"); T = Path("/opt/az-ml-new/test_inputs")
S1RUN = Path("/opt/az-ml-new/runs/s1_pair_tree_40k"); TS = Path("/opt/az-ml-new/runs/c2_step040000/test_scores")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/az-ml-new/france_legal/f1")
OUT.mkdir(parents=True, exist_ok=True)
rep = json.loads((S1RUN / "report.json").read_text()); pw, pt = rep["plain_c2_name_only"]["weight"], rep["plain_c2_name_only"]["threshold"]
ref_table = pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id"] + COLUMNS)
pool_table = pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id"] + COLUMNS)
s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object); rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
country = np.asarray(ref_table.column("country").to_pylist(), dtype=object); n_s1, n_rec = len(s1_ids), len(rec_ids)
ref, pool = ref_table.select(COLUMNS).to_pandas(), pool_table.select(COLUMNS).to_pandas()
cache = v1_cache(None, None, None, None, None, None, W / "step005000/test_v1_cache.npz")
left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64); n_pairs = int(cache["n"][2])
c2 = c2_on_rows(TS, cache["cand_rows"], cache["base"], must_score=np.ones(len(left), bool), n_pairs=n_pairs)
plain_prob = c2_blend(np.where(np.isnan(c2), -1.0, c2), cache["base"], pw)
mult = cache["s1_mult"][left]
prob = np.load(S1RUN / "probabilities_candidates.npy")
accept = decode(prob, rep["threshold"], rep["variant"], left, right, cache["rec_empty"], cache["name_exact"], mult)
plain = decode(plain_prob, pt, "model_then_name_only", left, right, cache["rec_empty"], cache["name_exact"], mult)
france = country[left] == "France"
la = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); ra = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
pair_keys = np.sort(la * n_rec + ra); del la, ra
rec_index = {v: i for i, v in enumerate(rec_ids)}
# parity with the uploaded S1 file
par = OUT / "parity/matching_results.tsv"
write_tsv(par, s1_ids, rec_ids, left, right, np.where(france, plain, accept), np.where(france, plain_prob, prob))
if sha256(par) != S1_SHA:
    raise SystemExit(f"parity FAILED: {sha256(par)} != {S1_SHA}")
log("parity OK: rebuilt S1 file matches 0.978287 upload"); par.unlink()
# French rule pairs
sel = np.flatnonzero(france & (cache["base"] >= 0.3))
NI = {}
def ni(df, i, key):
    k = (key, i)
    if k not in NI: NI[k] = name_info(df.business_name[i])
    return NI[k]
hit = np.array([rule(ni(ref, i, 0), ref.business_address[i] or "", ni(pool, j, 1), pool.business_address[j] or "")
                for i, j in zip(left[sel], right[sel])], dtype=bool)
rule_rows = sel[hit]
log(f"French rule pairs: {len(rule_rows)} of {len(sel)} French rows with tree >= 0.3; already accepted {int(plain[rule_rows].sum())}")
plain_prob2 = plain_prob.copy(); plain_prob2[rule_rows] = np.maximum(plain_prob2[rule_rows], 0.99)
plain2 = decode(plain_prob2, pt, "model_then_name_only", left, right, cache["rec_empty"], cache["name_exact"], mult)
final = np.where(france, plain2, accept); final_prob = np.where(france, plain_prob2, prob)
path = OUT / "matching_results.tsv"
lengths = write_tsv(path, s1_ids, rec_ids, left, right, final, final_prob)
check = validate_tsv(path, s1_ids, rec_index, pair_keys, n_rec)
diff = {"french_added": int((plain2 & ~plain & france).sum()), "french_removed": int((plain & ~plain2 & france).sum()),
        "french_added_rule_pairs": int((plain2 & ~plain)[rule_rows].sum()), "non_french_changed": int(((final != np.where(france, plain, accept)) & ~france).sum())}
report = {"file": str(path), "sha256": sha256(path), "validation": check, "by_country": by_country(country, lengths),
          "vs_s1_upload": diff, "rule_pairs": int(len(rule_rows)), "base_file_sha256": S1_SHA, "labels_read": False}
(OUT / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
print(json.dumps(report, indent=1, default=float))
