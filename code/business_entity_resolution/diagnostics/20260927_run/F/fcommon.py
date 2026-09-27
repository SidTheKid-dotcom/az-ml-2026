"""Agent F (27 Sep run): shared loading + F1 French name normalization.

F1 normalization (French rows only): drop standalone French legal-form tokens (sarl sas sasu sa sci eurl ei snc;
case/accent/dot/punctuation-insensitive) anywhere in the name, then strip trailing copy suffixes (cie, services,
associes/associés, "et fils", "& fils"; a connector "et"/"&" before cie is dropped with it) only at the end.
Addresses are never touched. If nothing would remain, the name is kept. No labels are read.
"""
import re, sys, unicodedata
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

from ber.pair_tree_c2 import COLUMNS, c2_blend, c2_on_rows, decode, v1_cache, log

FR_LEGAL = frozenset("sarl sas sasu sa sci eurl ei snc".split())
TRAIL = frozenset("cie services associes".split())
TOK = re.compile(r"[^\W_]+(?:\.[^\W_]+)*\.?")

S1_SHA = "559f37b9b66f4e18fc28c0b25ae60d08543c386e0bdb32c619990b4c9cbd1a78"
S1_FILE = Path("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv")
QWEN_FILE = Path("/workspace/az-ml-new/qwen/files_test_step_003250_c2min0.01/model_us_india_c2_france/matching_results.tsv")
W = Path("/opt/az-ml-new/agents/pair_tree_c2"); T = Path("/opt/az-ml-new/test_inputs")
S1RUN = Path("/opt/az-ml-new/runs/s1_pair_tree_40k"); TS = Path("/opt/az-ml-new/runs/c2_step040000/test_scores")
OUT = Path("/workspace/az-ml-new/run27/F")


def key(tok):
    return unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode().lower().replace(".", "")


def _clean(s):
    s = re.sub(r"\(\s*\)", " ", s)
    s = re.sub(r"\s+([,;:)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"(\s*[-–]\s*){2,}", " - ", s)
    s = re.sub(r"(\s*,\s*){2,}", ", ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"^[\s,;:&+/\-–]+", "", s)
    s = re.sub(r"[\s,;:&+/\-–]+$", "", s)
    return s


def normalize_name(name):
    """Return (new_name, dropped_keys). Unchanged names are returned as the identical string."""
    if not name:
        return name, ()
    toks = [(m.start(), m.end(), key(m.group())) for m in TOK.finditer(name)]
    if not toks:
        return name, ()
    drop = [k in FR_LEGAL for _, _, k in toks]
    keep = [i for i in range(len(toks)) if not drop[i]]
    while keep:
        last = keep[-1]; k = toks[last][2]
        if k in TRAIL:
            drop[last] = True; keep.pop()
            if keep and toks[keep[-1]][2] == "et":
                drop[keep[-1]] = True; keep.pop()
            continue
        if k == "fils" and len(keep) >= 2:
            prev = keep[-2]
            if toks[prev][2] == "et":
                drop[last] = drop[prev] = True; keep.pop(); keep.pop(); continue
            if "&" in name[toks[prev][1]:toks[last][0]]:
                drop[last] = True; keep.pop(); continue
        break
    if not any(drop) or not keep:
        return name, ()
    out, pos = [], 0
    for (a, b, _), d in zip(toks, drop):
        if d:
            out.append(name[pos:a]); pos = b
    out.append(name[pos:])
    new = _clean(" ".join(p for p in out))
    if not new or new == name:
        return name, ()
    return new, tuple(k for (_, _, k), d in zip(toks, drop) if d)


def load(need_accept=True):
    """Everything the #5 rule needs, on cache candidate rows (as france_legal_file.py)."""
    import json
    rep = json.loads((S1RUN / "report.json").read_text()); pw, pt = rep["plain_c2_name_only"]["weight"], rep["plain_c2_name_only"]["threshold"]
    ref_table = pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id"] + COLUMNS)
    pool_table = pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id"] + COLUMNS)
    d = {"rep": rep, "pw": pw, "pt": pt, "ref_table": ref_table, "pool_table": pool_table}
    d["s1_ids"] = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object)
    d["rec_ids"] = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
    d["country"] = np.asarray(ref_table.column("country").to_pylist(), dtype=object)
    d["pool_country"] = np.asarray(pool_table.column("country").to_pylist(), dtype=object)
    cache = v1_cache(None, None, None, None, None, None, W / "step005000/test_v1_cache.npz")
    d["cache"] = cache
    left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
    d["left"], d["right"], d["base"] = left, right, cache["base"]
    d["n_pairs"] = int(cache["n"][2])
    d["c2"] = c2_on_rows(TS, cache["cand_rows"], cache["base"], must_score=np.ones(len(left), bool), n_pairs=d["n_pairs"])
    d["mult"] = cache["s1_mult"][left]
    d["france"] = d["country"][left] == "France"
    if need_accept:
        prob = np.load(S1RUN / "probabilities_candidates.npy")
        d["prob"] = prob
        d["accept"] = decode(prob, rep["threshold"], rep["variant"], left, right, cache["rec_empty"], cache["name_exact"], d["mult"])
    return d


def plain_of(d, c2):
    plain_prob = c2_blend(np.where(np.isnan(c2), -1.0, c2), d["base"], d["pw"])
    plain = decode(plain_prob, d["pt"], "model_then_name_only", d["left"], d["right"], d["cache"]["rec_empty"],
                   d["cache"]["name_exact"], d["mult"])
    return plain_prob, plain
