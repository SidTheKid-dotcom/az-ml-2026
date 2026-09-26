"""Track F step 1: sibling S1s (same core name at the same address) in test and train.

Core name = name tokens minus legal forms, corporate suffix words and fillers (accent-folded,
lower-cased, dots removed; the france_legal_file.py definition). Two address keys:
  exact: the set of normalized address components (same address up to component order);
  loose: address digits + the set of digit-free components (same digits and city/region).
Test: share of S1s in sibling groups (size >= 2) per country, and whether members differ only
in legal form. Train: arena partitions A and B_REST only (SLICE and HOLDOUT-2 labels are never
read; HOLDOUT-2 is not in the arena), record owners from arena rec_owner: are sibling records
split across members, and does the record's exact legal form pick its owner?
"""
import json, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd

LEGAL = set("inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl sas sasu sa sci eurl ei snc gmbh opc lp pllc pc".split())
SUFFIX = set("group groupe holdings holding fils sons france india usa international cie associes developpement distribution".split())
FILLER = set("et and the".split())
SYN = {"incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt"}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def keys(name, address):
    t = norm(name)
    core = " ".join(sorted(set(x for x in t if x not in LEGAL | SUFFIX | FILLER)))
    legal = " ".join(sorted(set(SYN.get(x, x) for x in t if x in LEGAL)))
    comps = [" ".join(norm(c)) for c in (address or "").split(",")]
    comps = [c for c in comps if c]
    exact = "|".join(sorted(comps))
    digits = " ".join(str(int(d)) for d in re.findall(r"\d+", address or ""))
    loose = digits + "#" + "|".join(sorted(c for c in comps if not re.search(r"\d", c)))
    return core, legal, exact, loose, bool(digits)


def frame(df):
    k = [keys(n, a) for n, a in zip(df.business_name.fillna(""), df.business_address.fillna(""))]
    out = pd.DataFrame(k, columns=["core", "legal", "exact", "loose", "has_digits"], index=df.index)
    out["country"] = df.country.values
    out["name"] = df.business_name.fillna("").values
    out["address"] = df.business_address.fillna("").values
    return out


def groups(f, key):
    ok = (f.core != "") & (f[key].str.len() > 1)
    g = f[ok].groupby(["core", key]).ngroup()
    size = g.map(g.value_counts())
    gid = pd.Series(-1, index=f.index); gid[ok] = g
    sz = pd.Series(1, index=f.index); sz[ok] = size
    return gid.values, sz.values


def test_stats(f):
    out = {}
    for key in ("exact", "loose"):
        gid, sz = groups(f, key)
        res = {}
        for c in sorted(f.country.unique()):
            m = (f.country == c).values
            sib = m & (sz >= 2)
            # among sibling groups: do members differ in legal form? are some exact duplicates (same legal form)?
            sub = f[sib].assign(gid=gid[sib])
            per = sub.groupby("gid").agg(n=("legal", "size"), n_legal=("legal", "nunique"))
            res[c] = {"s1": int(m.sum()), "in_sibling_groups": int(sib.sum()), "share": float(sib.mean() / m.mean()),
                      "groups": int(len(per)), "group_size_hist": {int(k): int(v) for k, v in per.n.value_counts().sort_index().items()},
                      "groups_all_distinct_legal": int((per.n_legal == per.n).sum()),
                      "groups_with_repeated_legal": int((per.n_legal < per.n).sum())}
        out[key] = res
    return out, gid, sz


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    test = pd.read_parquet("/opt/az-ml-new/test_inputs/dense_test/reference.parquet")
    ft = frame(test)
    report["test"], gid, sz = test_stats(ft)
    fr = (ft.country == "France").values & (sz >= 2)
    ex = ft[fr].assign(gid=gid[fr]).sort_values(["gid"]).head(60)
    report["test_france_examples"] = [f"{r.gid}: {r.name} | {r.address}" for r in ex.itertuples()]
    # full-canonicalization merge risk: French sibling pairs whose legal forms map to the same class
    cls = {"sarl": "llc", "eurl": "llc", "sas": "corp", "sasu": "corp", "sa": "inc", "sci": "lp", "snc": "partnership"}
    def canon_legal(l): return " ".join(sorted(set(cls.get(x, x) for x in l.split())))
    for key in ("exact", "loose"):
        g2, s2 = groups(ft, key)
        m = (ft.country == "France").values & (s2 >= 2)
        sub = ft[m].assign(gid=g2[m], cl=ft[m].legal.map(canon_legal))
        per = sub.groupby("gid").agg(n=("cl", "size"), ncl=("cl", "nunique"), nl=("legal", "nunique"))
        report[f"test_france_{key}_groups_merged_by_full_canon"] = int(((per.ncl < per.nl)).sum())
        report[f"test_france_{key}_s1_in_groups_merged_by_full_canon"] = int(per.n[per.ncl < per.nl].sum())
    del test
    # ---------------- train (A + B_REST only)
    ref = pd.read_parquet("/workspace/az-ml-new/arena/v1/reference.parquet")
    part = np.load("/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy")
    keep = part <= 1  # 0 = A, 1 = B_REST; SLICE (2) excluded
    fa = frame(ref)
    fa["keep"] = keep
    fk = fa[keep]
    report["train_A_BREST"], gid_k, sz_k = test_stats(fk)
    pool = pd.read_parquet("/workspace/az-ml-new/arena/v1/pool.parquet", columns=["business_name", "business_address"])
    owner = np.load("/workspace/az-ml-new/arena/labels/rec_owner.npy")
    owned = owner >= 0
    owned &= np.where(owned, keep[np.maximum(owner, 0)], False)  # only owners in A/B_REST
    rec_rows = np.flatnonzero(owned)
    rec_owner = owner[rec_rows]
    # sibling group of each kept S1 (index into ref rows)
    for key in ("exact", "loose"):
        g_all = np.full(len(ref), -1); s_all = np.ones(len(ref), int)
        g, s = groups(fk, key)
        g_all[np.flatnonzero(keep)] = g; s_all[np.flatnonzero(keep)] = s
        in_group = s_all[rec_owner] >= 2
        rr, ro = rec_rows[in_group], rec_owner[in_group]
        members = defaultdict(list)
        for i in np.flatnonzero(s_all >= 2):
            members[g_all[i]].append(i)
        legal_s1 = fa.legal.values
        rec_legal = [" ".join(sorted(set(SYN.get(x, x) for x in norm(n) if x in LEGAL))) for n in pool.business_name.values[rr]]
        stats = Counter()
        per_country = defaultdict(Counter)
        for r_leg, o in zip(rec_legal, ro):
            c = fa.country.values[o]
            sibs = members[g_all[o]]
            match = [i for i in sibs if legal_s1[i] == r_leg]
            k = ("owner_legal_equal" if legal_s1[o] == r_leg else "owner_legal_differs")
            if len(match) == 1:
                k2 = "unique_legal_match_is_owner" if match[0] == o else "unique_legal_match_is_other_sibling"
            elif len(match) == 0:
                k2 = "no_sibling_has_record_legal"
            else:
                k2 = "several_siblings_share_record_legal"
            per_country[c][k] += 1; per_country[c][k2] += 1; per_country[c]["records"] += 1
        # how records are split: records per member, members with zero records
        cnt = np.bincount(ro, minlength=len(ref))
        split = {}
        for c in ("US", "India"):
            mem = np.flatnonzero((s_all >= 2) & (fa.country.values == c))
            split[c] = {"members": int(len(mem)), "mean_records": float(cnt[mem].mean()) if len(mem) else None,
                        "members_with_0_records": int((cnt[mem] == 0).sum()),
                        "all_s1_mean_records": float(np.bincount(rec_owner, minlength=len(ref))[keep & (fa.country.values == c)].mean())}
        report[f"train_{key}_record_ownership"] = {c: dict(v) for c, v in per_country.items()}
        report[f"train_{key}_split"] = split
    exs = []
    g, s = groups(fk, "exact")
    sub = fk[s >= 2].assign(gid=g[s >= 2]).sort_values("gid").head(30)
    report["train_examples"] = [f"{r.gid}: {r.name} | {r.address} | {r.country}" for r in sub.itertuples()]
    (out_dir / "sibling_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
