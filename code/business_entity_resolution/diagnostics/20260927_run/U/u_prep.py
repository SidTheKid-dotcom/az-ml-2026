"""Agent U prep (27 Sep): compact B_REST + test feature matrices for the U4 combined tree and the audits.

Runs on the pod with the frozen code paths (pair_tree_c2.load_arena, qwen_ensemble.load_arena_q,
c2_features/q_features on test) so the features are byte-identical to the shipped trees.
Writes /workspace/az-ml-new/run27/U/prep/{brest,test}/*.npy + manifest.json. Labels: B_REST rows only.
Parity checks: saved OOF of the C2 tree / Qwen tree / collective tree reproduce 0.985511 / 0.985900 /
0.985971 with their saved picks; the S1 final model reproduces the saved test probabilities.

    python u_prep.py brest|test|parity
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.pair_tree_c2 import VARIANTS, accept_per_row, c2_features, c2_on_rows, proba, v1_cache
from ber.qwen_ensemble import load_arena_q, q_features, q_on_rows

OUT = Path("/workspace/az-ml-new/run27/U/prep")
C2_BREST = "/opt/az-ml-new/runs/c2_step040000/b_rest_scores"
C2_TEST = "/opt/az-ml-new/runs/c2_step040000/test_scores"
Q_BREST = "/workspace/az-ml-new/qwen/brest_step_003250_c2min0.01"
Q_TEST = "/workspace/az-ml-new/qwen/test_step_003250_c2min0.01"
CACHE = "/opt/az-ml-new/agents/pair_tree_c2/step005000"
W_C2 = Path("/opt/az-ml-new/agents/pair_tree_c2/step040000")
W_Q = Path("/workspace/az-ml-new/qwen/ens_brest_step_003250_c2min0.01")
W_COLL = Path("/opt/az-ml-new/agents/pair_tree_c2/eureka_collective")
TEST_INPUTS = Path("/opt/az-ml-new/test_inputs")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def save(d, name, arr):
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", arr)


def brest():
    args = argparse.Namespace(arena="/workspace/az-ml-new/arena/v1", labels="/workspace/az-ml-new/arena/labels",
                              finalize="/workspace/az-ml-new/arena/finalize_v1", scores=C2_BREST, qwen_scores=Q_BREST,
                              work=str(OUT / "scratch"), cache=CACHE, extended=False)
    data = load_arena_q(args)
    arena = data["arena"]
    if not np.array_equal(np.load(W_C2 / "b_rest_pair_rows.npy"), data["pair_rows"]):
        raise SystemExit("pair rows differ from the C2 tree work dir")
    if not np.array_equal(np.load(W_Q / "b_rest_pair_rows.npy"), data["pair_rows"]):
        raise SystemExit("pair rows differ from the Qwen tree work dir")
    xc = np.load(W_COLL / "b_rest_collective.npy")
    if len(xc) != len(data["label"]):
        raise SystemExit("collective rows differ")
    d = OUT / "brest"
    save(d, "x_base", data["x"].astype(np.float32))
    save(d, "x_q", data["xq"].astype(np.float32))
    save(d, "x_coll", xc.astype(np.float32))
    save(d, "label", data["label"])
    save(d, "left", arena.left.astype(np.int64))
    save(d, "right", arena.right.astype(np.int64))
    save(d, "pair_rows", data["pair_rows"])
    save(d, "rec_empty", arena.rec_empty)
    save(d, "name_exact", arena.name_exact)
    save(d, "mult", arena.mult)
    save(d, "true_count", arena.true_count)
    save(d, "population", arena.population)
    save(d, "component", data["component"])
    save(d, "partition", data["partition"])
    cu, cc = np.unique(arena.country.astype(str), return_inverse=True)
    save(d, "country_code", cc.astype(np.int8))
    save(d, "base", data["base"])
    save(d, "blend", data["blend"])
    for name, w in (("oof_c2tree", W_C2), ("oof_qtree", W_Q), ("oof_coll", W_COLL)):
        save(d, name, np.load(w / "oof.npy"))
    (d / "meta.json").write_text(json.dumps({"countries": list(cu), "n_s1": data["n_s1"], "n_rec": data["n_rec"],
                                             "rows": int(len(data["label"])), "q_filter": data["q_filter"],
                                             "shift": data["shift"], "q_shift": data["q_shift"]}, default=float) + "\n")
    log(f"brest saved: {len(data['label'])} rows")
    parity(d, arena)


def parity(d=OUT / "brest", arena=None):
    from ber.pair_tree_c2 import Arena
    if arena is None:
        ld = lambda n: np.load(d / f"{n}.npy", allow_pickle=False)  # noqa: E731
        cu = json.loads((d / "meta.json").read_text())["countries"]
        arena = Arena(ld("left"), ld("right"), ld("label"), ld("true_count"), ld("rec_empty"), ld("name_exact"),
                      ld("mult"), ld("population"), np.asarray(cu)[ld("country_code")])
    comp = np.load(d / "component.npy")
    half_row = (comp[arena.left] % 2).astype(np.int8)
    out = {}
    for name, w, key in (("c2tree", W_C2, "picks"), ("qtree", W_Q, None), ("coll", W_COLL, None)):
        oof = np.load(d / f"oof_{name}.npy")
        if key:
            picks = json.loads((w / "crossfit_state.json").read_text())["picks"]
        elif name == "qtree":
            picks = json.loads((w / "report.json").read_text())["tree_q_picks"]
        else:
            picks = json.loads((w / "report.json").read_text())["crossfit_picks"]
        thr = np.where(half_row == 1, picks["0"]["model"]["threshold"], picks["1"]["model"]["threshold"])
        out[name] = arena.score(accept_per_row(arena, oof, thr, "model"))["macro_f05"]
        log(f"parity {name}: {out[name]:.6f}")
    (d / "parity.json").write_text(json.dumps(out) + "\n")


def test():
    d = OUT / "test"
    ref = pq.read_table(TEST_INPUTS / "dense_test" / "reference.parquet", columns=["entity_id", "country"])
    country = np.asarray(ref.column("country").to_pylist(), dtype=object)
    n_s1 = len(country)
    n_rec = pq.ParquetFile(TEST_INPUTS / "dense_test" / "pool.parquet").metadata.num_rows
    cache = v1_cache(None, None, None, None, None, None, Path(CACHE) / "test_v1_cache.npz")
    left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
    n_pairs = int(cache["n"][2])
    everything = np.ones(len(left), bool)
    c2 = c2_on_rows(C2_TEST, cache["cand_rows"], cache["base"], must_score=everything, n_pairs=n_pairs)
    q, flt = q_on_rows(Q_TEST, C2_TEST, cache["cand_rows"], cache["base"], everything, n_pairs)
    x2 = c2_features(left, right, c2, cache["base"], n_s1, n_rec)
    xq = q_features(left, right, c2, q, cache["base"], n_s1, n_rec)
    x_base = np.hstack([cache["v1x"], x2]).astype(np.float32)
    save(d, "x_base", x_base)
    save(d, "x_q", xq.astype(np.float32))
    save(d, "left", left)
    save(d, "right", right)
    save(d, "cand_rows", cache["cand_rows"])
    save(d, "rec_empty", cache["rec_empty"])
    save(d, "name_exact", cache["name_exact"])
    save(d, "mult", cache["s1_mult"][left])
    save(d, "base", cache["base"])
    cu, cc = np.unique(country.astype(str), return_inverse=True)
    save(d, "country_code", cc.astype(np.int8))
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(W_C2 / "final_model.cbm"))
    rows = np.random.default_rng(0).choice(len(left), 200_000, replace=False)
    p = proba(model, x_base[rows])
    saved = np.load("/opt/az-ml-new/runs/s1_pair_tree_40k/probabilities_candidates.npy", mmap_mode="r")
    diff = float(np.abs(p - saved[rows]).max())
    (d / "meta.json").write_text(json.dumps({"countries": list(cu), "n_s1": n_s1, "n_rec": int(n_rec),
                                             "rows": int(len(left)), "q_filter": flt,
                                             "s1_final_model_max_abs_diff_200k": diff}) + "\n")
    log(f"test saved: {len(left)} rows; S1 model parity max|diff| {diff:.2e}")


if __name__ == "__main__":
    {"brest": brest, "test": test, "parity": parity}[sys.argv[1]]()
