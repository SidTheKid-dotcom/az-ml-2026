"""Regenerate the experiment ledger from completed, inspectable reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts')
    parser.add_argument('--additional-artifacts', action='append', default=[],
                        help='Additional completed-report roots; never replace historical rows with a partial checkout')
    parser.add_argument('--output', default='learnings')
    args = parser.parse_args()
    rows = []
    setups = ('quick_v1', 'phonetic_v1', 'e5_small', 'bge_m3',
              'e5_small_ft200', 'e5_cross200', 'e5_ft_tabular_sweep',
              'hard_tabular_sweep', 'hard_residual_sweep', 'hard_country_transfer',
              'hard_ce_v1', 'hard_ce_residual_comparison', 'hard_bge_ce600',
              'hard_dense_sweep', 'hard_union_sweep', 'hard_union_bge_ce600',
              'medium_dense_sweep', 'medium_union_sweep', 'medium_dense_residual_sweep',
              'medium_union_residual_sweep', 'medium_ce_v1',
               'medium_dense_bge_ce600', 'medium_dense_bge_ce1800',
               'ce_shortlist_bge_v1_20260925', 'ce_shortlist_bge_gate050_20260925')
    for setup in setups:
        paths = [Path(root) / setup / 'results.json' for root in
                 (args.artifacts, *args.additional_artifacts)]
        available = [path for path in paths if path.is_file()]
        if not available:
            continue
        path = available[0]
        report = json.loads(path.read_text())
        if any(json.loads(other.read_text()) != report for other in available[1:]):
            raise ValueError(f'Conflicting completed reports for {setup}')
        if 'comparisons' in report:
            # Baseline and gated baseline repeat an already counted experiment.
            comparisons = [r for r in report['comparisons'] if r['cross_encoder_weight'] > 0]
        else:
            comparisons = report.get('results', [report])
        for record in comparisons:
            score = record.get('tune',record.get('transfer_tune'))
            if score is None:
                continue
            scope = ('full pool' if setup.startswith(('hard_','medium_','ce_shortlist_')) else 'sampled pool')
            count = score.get('reference_count', report.get('paired_tune', {}).get('reference_count'))
            scope += f"; {count} tuning references"
            if 'fit_country' in record:
                scope += f"; transfer from {record['fit_country']}"
            precision = score.get('micro_precision', score.get('precision'))
            recall = score.get('micro_recall', score.get('recall'))
            threshold = score.get('threshold', report.get('train_calibration', {}).get('choice', {}).get('threshold'))
            if None in (count, precision, recall, threshold):
                raise ValueError(f'Incomplete completed score in {path}')
            rows.append(dict(setup=setup, model=record.get('model', record.get('variant', setup)),
                macro_f05=score['macro_f05'], precision=precision,
                recall=recall, threshold=threshold,
                singleton_accuracy=score['singleton_accuracy'],
                false_positive_links=score.get('false_positive_links', score.get('false_links',
                    round(score['predicted_links'] * (1 - precision)))),
                scope=scope,
                source=f'artifacts/{setup}/results.json'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    previous = output / 'experiments.json'
    if previous.exists():
        missing_or_changed = [row for row in json.loads(previous.read_text()) if row not in rows]
        if missing_or_changed:
            raise ValueError(f'Refusing to erase or change {len(missing_or_changed)} historical ledger rows')
    (output/'experiments.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (output/'experiments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Completed model comparisons', '',
        f'{len(rows)} scored configurations, including ensembles/blend weights and hyperparameter variants. '
        'Repeated baseline evaluations and failed runs are not counted. These are not independent model families.', '',
        '**Development scores do not establish submission readiness.** '
        'The initial model fell from 0.981500 to 0.832719 on the same 1,000 businesses when evaluated '
        'against 9,290,818 possible targets. Subsequent hard-pool experiments use all 10,320,219 original training targets. '
        'A fresh final holdout takes priority over additional tuning gains.', '']
    for prefix,title in [('full pool','Full-pool development'),('sampled pool','Small-pool screening')]:
        subset=[r for r in rows if r['scope'].startswith(prefix)]
        if not subset:
            continue
        lines += [f'## {title}', '',
            '| Setup | Matcher | Macro F0.5 | Precision | Recall | False links | Scope |',
            '| --- | --- | ---: | ---: | ---: | ---: | --- |']
        for row in sorted(subset,key=lambda r:r['macro_f05'],reverse=True):
            lines.append(f"| {row['setup']} | {row['model']} | {row['macro_f05']:.6f} | "
                f"{row['precision']:.6f} | {row['recall']:.6f} | {row['false_positive_links']} | {row['scope']} |")
        lines.append('')
    lines += ['', 'Small-pool rows use 2,400 tuning businesses and 201,536 targets. Full-pool rows indicate '
        'their reference population; different populations and transfer checks are not direct model comparisons. '
        'Thresholds use only TRAIN calibration or TUNE labels as stated in each report. '
        'The official test set has hidden labels; no entire-test F0.5 or F1 is available.', '',
        'Keep for independent validation: full-pool E5 plus lexical/phonetic retrieval, '
        'numeric-conflict features, and tree/BGE cross-encoder blends. Negative class weights '
        'depend on the candidate set: weight 1 won the initial lexical tree, while weight 4 won '
        'the quick-reference union tree. Prefer measured blends over the pure reranker in paired tests; '
        'heuristic-only matching and simple tree averaging remain weak leading options.', '',
        'BGE600 on the lexical hard pool improved on MiniLM300 by 0.006166 at matched 19,200 '
        'pair presentations; paired 95% tune interval [0.002883, 0.009499]. '
        'Union candidates plus 50% BGE reached 0.969812 versus the union tree 0.956008. '
        'These are tuning results; larger-data and frozen holdout comparisons remain necessary. '
        'The earlier residual-tree 25% CE near-tie had 31 fewer false links than its 50% blend. '
        'Original-tree 50%-minus-25% CE has a tune-only paired 95% interval of [-0.002536, 0.002939]. '
        'Retrieval-only ablations, the frozen full-pool stress test, and anchor-threshold checks are '
        'documented in EXPERIMENT_DECISIONS.md rather than counted as new model configurations.', '',
        'The first E5 fine-tuning attempt failed on non-tensor tokenizer metadata. Tensor-only device transfer '
        'fixed it, and the successful retry is recorded above. No result is attributed to the failed run.', '']
    (output/'scoreboard.md').write_text('\n'.join(lines))
    print(f'Recorded {len(rows)} completed configurations')


if __name__ == '__main__':
    main()
