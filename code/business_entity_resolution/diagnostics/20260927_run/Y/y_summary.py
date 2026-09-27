"""Agent Y: one-screen summary of the Laya C2-FR results (B_REST monitor, Frenchified TUNE, French rows at 0.10/0.70,
U-pair scores, family agreement). Usage: y_summary.py N  (N = laya_step_XXXXXX). Prints markdown; reads JSON only."""
import json, sys
from pathlib import Path
Y = Path("/home/ubuntu/run27/Y"); R = Y / "results"; N = sys.argv[1]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=6):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


out = []
ev = load(R / f"eval_{N}.json")
if ev:
    b = ev["B_REST"]
    out.append("## B_REST monitor (20,000 US/India S1s, real labels; 0.9*model + 0.1*tree >= 0.10, one owner)")
    out.append("| model | F0.5 | P | R | false links | singleton acc | US | India |")
    out.append("|---|---|---|---|---|---|---|---|")
    for k, name in (("candidate", "Laya C2-FR"), ("baseline", "stored C2 40k")):
        x = b[k]
        out.append(f"| {name} | {f(x['macro_f05'])} | {f(x['precision'], 5)} | {f(x['recall'], 5)} | {x['false_links']} | "
                   f"{f(x['singleton_accuracy'], 4)} | {f(x['by_country']['US']['macro_f05'])} | {f(x['by_country']['India']['macro_f05'])} |")
    d = b.get("candidate_minus_baseline", {})
    out.append(f"\nDelta (paired per S1) {f(d.get('mean'))} +- {f(d.get('se'))} (se); gate-2 flag {b.get('gate2_pass')} (reported, not gated).")
    t = ev["TUNE_frenchified"]["candidate"]
    r5 = t["rule5_0.9c2_0.1proxytree_ge_0.10"]
    out.append("\n## Frenchified TUNE world (synthetic, labels by construction; not French accuracy)")
    out.append(f"rule5 F0.5 {f(r5['macro_f05'])} P {f(r5['precision'], 5)} R {f(r5['recall'], 5)}; best rule5 threshold {t['rule5_best_threshold']}")
    out.append("mean score by kind: " + json.dumps({k: (round(v, 4) if v is not None else None) for k, v in t["mean_score_by_kind"].items()}))
for tag in ("laya_t010_", "laya_t070_"):
    dg = load(R / f"{tag}diagnostics.json")
    if not dg:
        continue
    out.append(f"\n## French rows {tag} (rule {dg['rule']['weight']}*model + {1 - dg['rule']['weight']:.1f}*tree >= {dg['rule']['threshold']})")
    out.append(f"links {dg['file']['links']} (S1 file {dg['signatures_old_s1_file']['links']}); added {dg['links_added']}, removed {dg['links_removed']}, "
               f"S1 rows changed {dg['s1_rows_changed']}; empty French rows {dg['signatures_new']['empty_french_s1']} (S1 file {dg['signatures_old_s1_file']['empty_french_s1']})")
    for k in ("falsifier_added", "falsifier_removed", "falsifier_kept"):
        x = dg[k]
        out.append(f"- {k}: n {x['n']}, delta-shift {f(x['shift_rate'], 4)}, decoy word {f(x['decoy_word_rate'], 4)}, either {f(x['either_rate'], 4)}, even shift {f(x['even_shift_2_12_rate'], 4)}")
    pa = dg["probe_anchor"]
    if isinstance(pa, dict):
        for k, x in pa.items():
            out.append(f"- {k}: n {x['n']}, mean model {f(x['mean_c2fr'], 4)} (C2 40k {f(x['mean_c2_40k'], 4)}), still accepted {f(x['still_accepted_new'], 4)}")
    out.append(f"- file sha256 {dg['file']['sha256']}, rows {dg['file']['french_rows']}, max list {dg['file']['max_list']}, records unique {dg['file']['records_unique']}")
ag = load(R / "agreement.json")
if ag:
    out.append("\n## Link agreement with the other French families (label-free)")
    out.append("links: " + json.dumps(ag["links"]))
    out.append("probe acceptance: " + json.dumps({k: {p: round(v, 4) for p, v in d.items()} for k, d in ag["probe_accept"].items()}))
    out.append("additions vs S1 file and their support: " + json.dumps(ag["additions_vs_s1_file_support"]))
print("\n".join(out))
