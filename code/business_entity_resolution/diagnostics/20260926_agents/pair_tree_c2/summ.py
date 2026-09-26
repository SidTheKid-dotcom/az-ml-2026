import json, sys
r = json.load(open(sys.argv[1]))
def row(name, d):
    bc = d.get("by_country", {})
    fm = d.get("false_merge", {})
    print(f"{name:40s} F={d['macro_f05']:.6f} P={d['precision']:.5f} R={d['recall']:.5f} FL={d['false_links']} "
          f"links={d['predicted_links']} sing={d['singleton_accuracy']:.4f} fm={fm.get('rate', 0):.4f} ({fm.get('false_merges')}) "
          + " ".join(f"{c}={v['macro_f05']:.6f}" for c, v in bc.items()))
for k, v in r["baselines"].items():
    if isinstance(v, dict) and "macro_f05" in v:
        row(k, v)
    elif k == "c2_grid":
        print("c2_grid picks", v["picks"], "full", v["full_b_rest_pick"])
    else:
        print(k, v)
for k, v in r["crossfit"].items():
    row("xfit " + k, v)
print("picks", r["crossfit_picks"])
print("full oof thr", r["full_oof_threshold"], r["full_oof_f05_at_threshold"])
for k, v in r["bootstrap"].items():
    print(f"{k:55s}", {a: round(b, 6) for a, b in v.items() if a in ("delta", "ci_low", "ci_high", "p_le_zero")})
print("pop", r["population"])
print("shift", r["c2_rival_shift"])
imp = r["feature_importance"]["0"]
print([(k, round(v, 1)) for k, v in sorted(imp.items(), key=lambda x: -x[1])[:12]])
print("elapsed", r["elapsed_seconds"])
g = r["full_oof_grid"]["model"]
print({k: round(v, 6) for k, v in g.items() if 0.4 <= float(k) <= 0.9})
