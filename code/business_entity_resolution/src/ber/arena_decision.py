"""Decision layer on cached arrays: scoring, components, ownership rules and expected-F0.5 decoding.

Every function works on flat pair arrays (one row per S1-record candidate pair):
``s1_idx`` (int, 0..n_s1-1), ``rec_idx`` (int record key), float scores and an
optional boolean ``label``. Per-S1 arrays (``true_count``, country, component)
are indexed by ``s1_idx``. No text is read here.

Official metric per S1: F0.5 = 5*TP / (4*k + T); T=0 and k=0 scores 1, T=0 and
k>0 scores 0. ``true_count`` must count every true link of the S1, including
links retrieval never returned, so misses lower the score.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

__all__ = [
    "entity_f05", "evaluate", "component_bootstrap", "claim_graph_components", "one_owner",
    "name_only_rule", "break_even", "expected_f05_decode", "iterated_best_response",
]


def _as_int(values):
    return np.asarray(values, dtype=np.int64)


def entity_f05(tp, k, true_count):
    """Vectorized official per-S1 F0.5 with the singleton convention."""
    tp, k, t = (np.asarray(x, dtype=np.float64) for x in (tp, k, true_count))
    denominator = 4.0 * k + t
    score = np.divide(5.0 * tp, denominator, out=np.zeros(np.broadcast(tp, k, t).shape), where=denominator > 0)
    return np.where((t == 0) & (k == 0), 1.0, score)


def _group_means(values, codes, mask):
    out = {}
    codes = np.asarray(codes)
    for code in np.unique(codes[mask]):
        m = mask & (codes == code)
        out[str(code)] = {"s1": int(m.sum()), "macro_f05": float(values[m].mean())}
    return out


def evaluate(accept_mask, s1_idx, label, true_count, *, population=None, country=None, rec_idx=None,
             rec_kind=None, return_per_s1=False):
    """Macro F0.5 over every S1 in ``population`` (default: all ``len(true_count)`` S1s).

    S1s without any candidate pair still count (k=0). ``true_count`` includes
    unretrieved links. Returns macro F0.5, micro precision/recall, false links,
    singleton accuracy, per-country macro F0.5 (``country`` per S1) and a
    link-level breakdown by record kind (``rec_kind`` indexed by ``rec_idx``).
    """
    accept = np.asarray(accept_mask, dtype=bool)
    s1 = _as_int(s1_idx)
    label = np.asarray(label, dtype=bool)
    t = _as_int(true_count)
    n = len(t)
    k = np.bincount(s1[accept], minlength=n)
    tp = np.bincount(s1[accept & label], minlength=n)
    if np.any(tp > t):
        raise ValueError("true_count is smaller than the accepted true links of some S1")
    per_s1 = entity_f05(tp, k, t)
    pop = np.ones(n, dtype=bool) if population is None else np.asarray(population, dtype=bool)
    in_pop_pair = pop[s1]
    total_tp, total_k, total_t = int(tp[pop].sum()), int(k[pop].sum()), int(t[pop].sum())
    singles = pop & (t == 0)
    result = {
        "s1": int(pop.sum()),
        "macro_f05": float(per_s1[pop].mean()) if pop.any() else float("nan"),
        "precision": total_tp / total_k if total_k else 0.0,
        "recall": total_tp / total_t if total_t else 0.0,
        "predicted_links": total_k,
        "true_links": total_t,
        "true_positive_links": total_tp,
        "false_links": total_k - total_tp,
        "retrieved_true_links": int((label & in_pop_pair).sum()),
        "singletons": int(singles.sum()),
        "singleton_accuracy": float((k[singles] == 0).mean()) if singles.any() else None,
        "empty_predictions": int((pop & (k == 0)).sum()),
    }
    if country is not None:
        result["by_country"] = _group_means(per_s1, country, pop)
    if rec_kind is not None:
        kind = np.asarray(rec_kind)[_as_int(rec_idx)]
        by_kind = {}
        for code in np.unique(kind[in_pop_pair]):
            m = in_pop_pair & (kind == code)
            acc, lab = accept & m, label & m
            by_kind[str(code)] = {
                "pairs": int(m.sum()), "accepted": int(acc.sum()), "true_accepted": int((acc & label).sum()),
                "false_accepted": int((acc & ~label).sum()), "retrieved_true": int(lab.sum()),
                "precision": float(label[acc].mean()) if acc.any() else None,
                "retrieved_recall": float(accept[lab].mean()) if lab.any() else None,
            }
        result["by_record_kind"] = by_kind
    if return_per_s1:
        result["per_s1"] = per_s1
    return result


def component_bootstrap(per_s1_scores_a, per_s1_scores_b, component_ids, n=2000, seed=0, population=None):
    """Paired delta (b - a) of macro F0.5 with a 95% percentile CI, resampling whole components.

    Each replicate draws components with replacement; the macro delta is the
    S1-weighted mean of per-S1 differences in the drawn components.
    """
    a = np.asarray(per_s1_scores_a, dtype=np.float64)
    b = np.asarray(per_s1_scores_b, dtype=np.float64)
    comp = _as_int(component_ids)
    if population is not None:
        pop = np.asarray(population, dtype=bool)
        a, b, comp = a[pop], b[pop], comp[pop]
    _, comp = np.unique(comp, return_inverse=True)
    n_comp = int(comp.max()) + 1 if comp.size else 0
    diff_sum = np.bincount(comp, weights=b - a, minlength=n_comp)
    size = np.bincount(comp, minlength=n_comp).astype(np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n)
    chunk = max(1, int(2e7 // max(n_comp, 1)))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        counts = np.stack([np.bincount(rng.integers(0, n_comp, n_comp), minlength=n_comp) for _ in range(stop - start)])
        deltas[start:stop] = (counts @ diff_sum) / (counts @ size)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float((b - a).mean()), "ci_low": float(lo), "ci_high": float(hi),
            "bootstrap_mean": float(deltas.mean()), "p_le_zero": float((deltas <= 0).mean()),
            "components": n_comp, "s1": int(len(a)), "replicates": int(n), "seed": seed}


def claim_graph_components(s1_idx, rec_idx, blend, threshold, n_s1=None):
    """Component id per S1: S1s sharing a record where both pairs have blend >= threshold are joined.

    For each record the qualifying claimants are chained (c0-c1, c1-c2, ...),
    which yields the same connected components as a clique with far fewer
    edges; connected components then run on the sparse S1-S1 graph.
    """
    s1 = np.asarray(s1_idx)
    n = int(s1.max()) + 1 if n_s1 is None else int(n_s1)
    keep = np.asarray(blend) >= threshold
    s = s1[keep].astype(np.int64)
    r = np.asarray(rec_idx)[keep]
    order = np.lexsort((s, r))
    s, r = s[order], r[order]
    same = r[1:] == r[:-1]
    left, right = s[:-1][same], s[1:][same]
    distinct = left != right
    left, right = left[distinct], right[distinct]
    graph = sparse.coo_matrix((np.ones(len(left), dtype=np.int8), (left, right)), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def one_owner(accept_mask, s1_idx, rec_idx, score):
    """Keep only the highest-score accepted claim per record (ties: lower s1_idx, then lower pair row)."""
    accept = np.asarray(accept_mask, dtype=bool)
    rows = np.flatnonzero(accept)
    if rows.size == 0:
        return accept.copy()
    r = np.asarray(rec_idx)[rows]
    order = np.lexsort((rows, np.asarray(s1_idx)[rows], -np.asarray(score, dtype=np.float64)[rows], r))
    r_sorted = r[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = r_sorted[1:] != r_sorted[:-1]
    out = np.zeros_like(accept)
    out[rows[order[first]]] = True
    return out


def name_only_rule(accept_mask, s1_idx, rec_idx, rec_empty, name_exact, name_mult, *, owner_wins=True):
    """PLAN_v2 A0 on pair arrays; returns ``(new_accept, info)``.

    ``rec_empty``: the record's address is empty; ``name_exact``: the record's
    normalized name equals the S1's; ``name_mult``: number of S1s in the S1's
    country with that normalized name (per pair).

    - ADD every empty-address exact-name pair with multiplicity 1, unless two
      or more distinct S1s have such a pair to the same record (then skip it).
    - DROP accepted empty-address exact-name pairs with multiplicity >= 2.
    - ``owner_wins``: a record that was ADDed is removed from every other S1
      that currently accepts it.
    """
    accept = np.asarray(accept_mask, dtype=bool)
    s1 = _as_int(s1_idx)
    rec = _as_int(rec_idx)
    mult = _as_int(name_mult)
    exact_empty = np.asarray(rec_empty, dtype=bool) & np.asarray(name_exact, dtype=bool)
    unique = exact_empty & (mult == 1)
    pair_rec, pair_s1 = rec[unique], s1[unique]
    distinct = np.unique(np.stack([pair_rec, pair_s1]), axis=1) if unique.any() else np.empty((2, 0), dtype=np.int64)
    recs, counts = np.unique(distinct[0], return_counts=True)
    conflicted = recs[counts > 1]
    conflict = unique & np.isin(rec, conflicted)
    added = unique & ~conflict & ~accept
    dropped = exact_empty & (mult >= 2) & accept
    new = (accept & ~dropped) | added
    owner_removed = np.zeros_like(accept)
    if owner_wins and added.any():
        owned = np.unique(rec[added])
        owner = np.full(len(owned), -1, dtype=np.int64)
        owner[np.searchsorted(owned, rec[added])] = s1[added]
        pos = np.searchsorted(owned, rec)
        pos_clip = np.minimum(pos, len(owned) - 1)
        hit = (owned[pos_clip] == rec) & new
        owner_removed = hit & (owner[pos_clip] != s1)
        new &= ~owner_removed
    info = {"added": added, "dropped": dropped, "owner_removed": owner_removed, "conflict_pairs": conflict,
            "conflicted_records": int(len(conflicted))}
    return new, info
