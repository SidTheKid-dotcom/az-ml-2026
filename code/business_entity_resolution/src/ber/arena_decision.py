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


def _summary(per_s1, k, tp, t, mask):
    total_tp, total_k, total_t = int(tp[mask].sum()), int(k[mask].sum()), int(t[mask].sum())
    singles = mask & (t == 0)
    return {
        "s1": int(mask.sum()),
        "macro_f05": float(per_s1[mask].mean()) if mask.any() else float("nan"),
        "precision": total_tp / total_k if total_k else 0.0,
        "recall": total_tp / total_t if total_t else 0.0,
        "predicted_links": total_k,
        "true_links": total_t,
        "true_positive_links": total_tp,
        "false_links": total_k - total_tp,
        "singletons": int(singles.sum()),
        "singleton_accuracy": float((k[singles] == 0).mean()) if singles.any() else None,
        "empty_predictions": int((mask & (k == 0)).sum()),
    }


def evaluate(accept_mask, s1_idx, label, true_count, *, population=None, country=None, rec_idx=None,
             rec_kind=None, return_per_s1=False):
    """Macro F0.5 over every S1 in ``population`` (default: all ``len(true_count)`` S1s).

    S1s without any candidate pair still count (k=0). ``true_count`` includes
    unretrieved links. Returns macro F0.5, micro precision/recall, false links,
    singleton accuracy, the same summary per country (``country`` per S1) and a
    link-level breakdown by record kind (``rec_kind`` indexed by ``rec_idx``).
    """
    accept = np.asarray(accept_mask, dtype=bool)
    s1 = np.asarray(s1_idx)
    label = np.asarray(label, dtype=bool)
    t = _as_int(true_count)
    n = len(t)
    k = np.bincount(s1[accept], minlength=n)
    tp = np.bincount(s1[accept & label], minlength=n)
    if len(k) > n:
        raise ValueError("s1_idx exceeds len(true_count)")
    if np.any(tp > t):
        raise ValueError("true_count is smaller than the accepted true links of some S1")
    per_s1 = entity_f05(tp, k, t)
    pop = np.ones(n, dtype=bool) if population is None else np.asarray(population, dtype=bool)
    in_pop_pair = pop[s1]
    result = _summary(per_s1, k, tp, t, pop)
    result["retrieved_true_links"] = int((label & in_pop_pair).sum())
    if country is not None:
        codes = np.asarray(country)
        result["by_country"] = {str(c): _summary(per_s1, k, tp, t, pop & (codes == c)) for c in np.unique(codes[pop])}
    if rec_kind is not None:
        kind = np.asarray(rec_kind)[np.asarray(rec_idx)]
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


def claim_graph_components(s1_idx, rec_idx, blend, threshold, n_s1=None, chunk=1 << 24):
    """Component id per S1: S1s sharing a record where both pairs have blend >= threshold are joined.

    Qualifying pairs are sorted by one int64 key (record, S1); consecutive
    claimants of a record are chained (c0-c1, c1-c2, ...), which gives the same
    components as a clique with far fewer edges. Connected components then run
    on the sparse S1-S1 graph. Peak memory is about 8 bytes per qualifying pair
    plus 8 bytes per edge. S1s with no qualifying pair are their own component.
    """
    s1 = np.asarray(s1_idx)
    n = int(s1.max()) + 1 if n_s1 is None else int(n_s1)
    keep = np.asarray(blend) >= threshold
    key = np.asarray(rec_idx)[keep].astype(np.int64)
    key *= n
    key += s1[keep]
    del keep
    key.sort()
    lefts, rights = [], []
    for start in range(0, max(len(key) - 1, 0), chunk):
        block = key[start:start + chunk + 1]
        rec = block // n
        same = rec[1:] == rec[:-1]
        who = (block - rec * n).astype(np.int32)
        left, right = who[:-1][same], who[1:][same]
        distinct = left != right
        lefts.append(left[distinct])
        rights.append(right[distinct])
    del key
    left = np.concatenate(lefts) if lefts else np.empty(0, dtype=np.int32)
    right = np.concatenate(rights) if rights else np.empty(0, dtype=np.int32)
    # int32 weights: duplicate edges are summed and must never wrap to an explicit zero.
    graph = sparse.csr_matrix((np.ones(len(left), dtype=np.int32), (left, right)), shape=(n, n))
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


# ----------------------------------------------------------------------------- expected-F0.5 decoding

def break_even(k):
    """Probability above which adding a link to a correct list of length k pays: add-wrong / (add-true + add-wrong)."""
    k = np.asarray(k, dtype=np.float64)
    wrong = 4.0 / (5.0 * k + 4.0)
    return wrong / (1.0 / (5.0 * k + 1.0) + wrong)


def _poisson_pmf(lam, tol=1e-18):
    """Poisson(lam) pmf truncated once terms fall below ``tol`` (omitted mass < 1e-17 for lam < 1)."""
    if lam <= 0:
        return np.ones(1)
    pmf = [np.exp(-lam)]
    u = 0
    while True:
        u += 1
        nxt = pmf[-1] * lam / u
        if nxt < tol and u > lam:
            return np.asarray(pmf)
        pmf.append(nxt)


def _conv_bernoulli(dist, p):
    """Distribution of S + B with B ~ Bernoulli(p) per row; ``dist`` is (G, m), returns (G, m + 1)."""
    g, m = dist.shape
    out = np.empty((g, m + 1))
    out[:, :m] = dist * (1.0 - p)[:, None]
    out[:, m] = 0.0
    out[:, 1:] += dist * p[:, None]
    return out


def _conv_fixed(dist, pmf):
    g, m = dist.shape
    out = np.zeros((g, m + len(pmf) - 1))
    for u, weight in enumerate(pmf):
        out[:, u:u + m] += dist * weight
    return out


def _decode_block(probs, n_select, pmf, max_k, tie=1e-12):
    """Best prefix length per row of ``probs`` (G, n), rows sorted with selectable candidates first.

    Value of prefix k: E[5X/(4k+X+Y)], X ~ PoissonBinomial(first k),
    Y ~ PoissonBinomial(the rest) + Poisson; k=0 scores P(T=0). A longer
    prefix must beat the best shorter one by more than ``tie``.
    """
    g, n = probs.shape
    top = min(n, max_k)
    suffix = [None] * (top + 1)
    dist = np.ones((g, 1))
    if top == n:
        suffix[n] = dist
    for j in range(n - 1, -1, -1):
        dist = _conv_bernoulli(dist, probs[:, j])
        if j <= top:
            suffix[j] = dist
    best_e = suffix[0][:, 0] * pmf[0]
    best_k = np.zeros(g, dtype=np.int64)
    prefix = np.ones((g, 1))
    for k in range(1, top + 1):
        prefix = _conv_bernoulli(prefix, probs[:, k - 1])
        rest = _conv_fixed(suffix[k], pmf)
        x = np.arange(k + 1, dtype=np.float64)[:, None]
        y = np.arange(rest.shape[1], dtype=np.float64)[None, :]
        weight = 5.0 * x / (4.0 * k + x + y)
        value = np.einsum("gx,gx->g", prefix, rest @ weight.T)
        better = (k <= n_select) & (value > best_e + tie)
        best_e = np.where(better, value, best_e)
        best_k = np.where(better, k, best_k)
        suffix[k] = None
    return best_k, best_e


def _s1_groups(s1, n_s1):
    counts = np.bincount(s1, minlength=n_s1)
    starts = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])
    order = None if s1.size < 2 or bool(np.all(s1[1:] >= s1[:-1])) else np.argsort(s1, kind="stable")
    return order, starts, counts


def expected_f05_decode(p, s1_idx, lam=0.055, max_k=15, *, n_s1=None, selectable=None, chunk_cells=8_000_000):
    """Bayes-optimal prefix per S1 under independent Bernoulli(p) candidates.

    Candidates of each S1 are sorted by p (ties: lower pair row) and the prefix
    length k in 0..min(n, max_k) maximizing E[5*TP/(4k+T)] is chosen, with
    T = TP + true links among the S1's other candidates + Poisson(lam)
    unretrieved links; k=0 scores P(T=0). Pairs with ``selectable`` False cannot
    be predicted but still count toward T. Exact Poisson-binomial DP,
    vectorized over S1s with the same candidate count.

    Returns ``(accept, info)`` with per-S1 ``info['k']`` and
    ``info['expected_f05']`` (S1s without candidates: k=0, exp(-lam)).
    """
    p = np.asarray(p)
    s1 = np.asarray(s1_idx)
    n_s1 = (int(s1.max()) + 1 if s1.size else 0) if n_s1 is None else int(n_s1)
    pmf = _poisson_pmf(lam)
    order, starts, counts = _s1_groups(s1, n_s1)
    sel = None if selectable is None else np.asarray(selectable, dtype=bool)
    accept = np.zeros(len(p), dtype=bool)
    k_out = np.zeros(n_s1, dtype=np.int16)
    e_out = np.full(n_s1, np.exp(-lam))
    for n in np.unique(counts[counts > 0]):
        n = int(n)
        rows = np.flatnonzero(counts == n)
        step = max(1, chunk_cells // ((n + len(pmf)) * (min(n, max_k) + 2)))
        offsets = np.arange(n)
        for start in range(0, len(rows), step):
            chunk = rows[start:start + step]
            idx = starts[chunk][:, None] + offsets
            if order is not None:
                idx = order[idx]
            probs = p[idx].astype(np.float64)
            if sel is None:
                key, n_select = probs, np.full(len(chunk), n)
            else:
                allowed = sel[idx]
                key, n_select = np.where(allowed, probs, -1.0), allowed.sum(axis=1)
            o = np.argsort(-key, axis=1, kind="stable")
            idx = np.take_along_axis(idx, o, axis=1)
            probs = np.take_along_axis(probs, o, axis=1)
            best_k, best_e = _decode_block(probs, n_select, pmf, max_k)
            accept[idx[offsets[None, :] < best_k[:, None]]] = True
            k_out[chunk] = best_k
            e_out[chunk] = best_e
    return accept, {"k": k_out, "expected_f05": e_out}


def _resolve_contests(accept, p, s1, rec, n_s1):
    """Pairs to ban so each contested record keeps at most its best claimant, and only above break-even."""
    rows = np.flatnonzero(accept)
    banned = np.zeros(len(accept), dtype=bool)
    stats = {"contested_records": 0, "kept": 0, "dropped_from_all": 0, "banned_links": 0}
    if rows.size == 0:
        return banned, stats
    prob = np.asarray(p, dtype=np.float64)[rows]
    r = np.asarray(rec)[rows]
    order = np.lexsort((rows, s1[rows], -prob, r))
    rows, r, prob = rows[order], r[order], prob[order]
    first = np.ones(len(rows), dtype=bool)
    first[1:] = r[1:] != r[:-1]
    group = np.cumsum(first) - 1
    contested = np.bincount(group)[group] >= 2
    if not contested.any():
        return banned, stats
    list_len = np.bincount(s1[accept], minlength=n_s1)
    best = np.flatnonzero(first & contested)
    keep = prob[best] >= break_even(list_len[s1[rows[best]]] - 1)
    keep_row = np.zeros(len(rows), dtype=bool)
    keep_row[best[keep]] = True
    banned[rows[contested & ~keep_row]] = True
    stats.update(contested_records=int(len(best)), kept=int(keep.sum()), dropped_from_all=int((~keep).sum()),
                 banned_links=int(banned.sum()))
    return banned, stats


def iterated_best_response(p, s1_idx, rec_idx, lam=0.055, max_k=15, max_rounds=5, *, n_s1=None,
                           chunk_cells=8_000_000):
    """Expected-F0.5 decoding with one owner per record.

    Decode every S1; for each record claimed by 2+ S1s keep only the claimant
    with the highest p (ties: lower s1_idx, then pair row) if p >= break_even(k),
    k = that S1's list length without the record; otherwise drop the record
    from every claimant. Losing pairs become unselectable (they still count
    toward T) and the affected S1s are re-decoded. Repeat until no contest
    remains or ``max_rounds``; contests left after the last round are resolved
    once more without re-decoding, so the output always has one owner.
    """
    s1 = np.asarray(s1_idx)
    rec = np.asarray(rec_idx)
    p = np.asarray(p)
    n_s1 = (int(s1.max()) + 1 if s1.size else 0) if n_s1 is None else int(n_s1)
    accept, _ = expected_f05_decode(p, s1, lam, max_k, n_s1=n_s1, chunk_cells=chunk_cells)
    selectable = np.ones(len(accept), dtype=bool)
    history, converged = [], False
    for round_no in range(1, max_rounds + 1):
        banned, stats = _resolve_contests(accept, p, s1, rec, n_s1)
        if not banned.any():
            converged = True
            break
        selectable &= ~banned
        accept &= ~banned
        affected = np.unique(s1[banned])
        sub = np.flatnonzero(np.isin(s1, affected))
        redecoded, _ = expected_f05_decode(p[sub], s1[sub], lam, max_k, n_s1=n_s1, selectable=selectable[sub],
                                           chunk_cells=chunk_cells)
        stats.update(round=round_no, redecoded_s1=int(len(affected)),
                     links_changed_by_redecode=int((accept[sub] != redecoded).sum()))
        accept[sub] = redecoded
        history.append(stats)
    final = None
    if not converged:
        banned, final = _resolve_contests(accept, p, s1, rec, n_s1)
        converged = not banned.any()
        accept &= ~banned
    return accept, {"rounds": history, "converged_without_forced_resolution": converged,
                    "final_forced_resolution": final, "k": np.bincount(s1[accept], minlength=n_s1)}
