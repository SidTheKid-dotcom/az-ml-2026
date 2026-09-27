"""Agent J: print a compact markdown summary of eval_brest / adv_check JSONs (for the README)."""
import json
import sys

ev = json.load(open(sys.argv[1]))
print("| rows | n | true | U p AUC | llm AUC | avg-logit AUC |")
print("|---|---|---|---|---|---|")
for k, v in ev["auc"].items():
    print(f"| {k} | {v['rows']:,} | {v['true']:,} | {v['u_p']:.4f} | {v['llm']:.4f} | {v['avg_logit']:.4f} |")
s = ev["stack_auc_uncertain"]
print(f"\nStack AUC on scored rows: U p {s['u_p']:.4f}, U-only restack {s['stack_u_only']:.4f}, U + llm restack {s['stack_u_llm']:.4f}.")
m = ev["matched_threshold"]
print(f"Matched count at U p >= 0.70 ({m['accepted']:,} rows): U precision {m['u_precision']:.4f} / recall-in-band "
      f"{m['u_recall_in_band']:.4f}; llm alone precision {m['llm_precision']:.4f} / recall-in-band {m['llm_recall_in_band']:.4f}.")
k = ev["k_lite"]
print("\n| K-lite (all 521,846 B_REST S1s) | macro F0.5 | US | India | precision | recall | false links | singleton acc |")
print("|---|---|---|---|---|---|---|---|")
for n in ("baseline_u", "stack_u_only", "stack_u_llm"):
    r = k[n]
    print(f"| {n} | {r['macro_f05']:.6f} | {r['f05_US']:.6f} | {r['f05_India']:.6f} | {r['precision']:.5f} | {r['recall']:.5f} | "
          f"{r['false_links']:,} | {r['singleton_accuracy']:.4f} |")
for n in ("bootstrap_llm_vs_u_only", "bootstrap_llm_vs_baseline_u", "bootstrap_u_only_vs_baseline_u"):
    b = k[n]
    print(f"- {n.replace('bootstrap_', '')}: {b['delta']:+.6f} [{b['ci_low']:+.6f}, {b['ci_high']:+.6f}]")
if len(sys.argv) > 2:
    adv = json.load(open(sys.argv[2]))
    print("\n| adversarial (B_REST vs test) | rows B_REST | rows test | AUC raw llm | AUC raw U p | AUC llm within U-p decile | accept/1k B_REST | accept/1k test |")
    print("|---|---|---|---|---|---|---|---|")
    for key, v in adv["by_country_rowtype"].items():
        w = v["auc_within_u_decile_mean"]
        print(f"| {key} | {v['rows_brest']:,} | {v['rows_test']:,} | {v['auc_raw']:.3f} | {v['auc_u_p_raw']:.3f} | "
              f"{(w if w is not None else float('nan')):.3f} | {v['accept_per_1k_brest']:.1f} | {v['accept_per_1k_test']:.1f} |")
