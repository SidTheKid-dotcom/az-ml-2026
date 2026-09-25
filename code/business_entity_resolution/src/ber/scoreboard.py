"""Regenerate the experiment ledger from completed, inspectable reports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HISTORICAL_COUNT = 107
DECISION_SETUP = 'v2_p1_decision'
TREE_SCREEN_SETUP = 'v2_tree_dense_compat'


def historical_rows(artifacts):
    rows = []
    setups = ('quick_v1', 'phonetic_v1', 'e5_small', 'bge_m3',
              'e5_small_ft200', 'e5_cross200', 'e5_ft_tabular_sweep',
              'hard_tabular_sweep', 'hard_residual_sweep', 'hard_country_transfer',
              'hard_ce_v1', 'hard_ce_residual_comparison', 'hard_bge_ce600',
              'hard_dense_sweep', 'hard_union_sweep', 'hard_union_bge_ce600',
              'medium_dense_sweep', 'medium_union_sweep', 'medium_dense_residual_sweep',
              'medium_union_residual_sweep', 'medium_ce_v1',
              'medium_dense_bge_ce600', 'medium_dense_bge_ce1800')
    for setup in setups:
        path = artifacts / setup / 'results.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if 'comparisons' in report:
            # Baseline and gated baseline repeat an already counted experiment.
            comparisons = [r for r in report['comparisons'] if r['cross_encoder_weight'] > 0]
        else:
            comparisons = report.get('results', [report])
        for record in comparisons:
            score = record.get('tune',record.get('transfer_tune'))
            if score is None:
                continue
            scope = ('full pool' if setup.startswith(('hard_','medium_')) else 'sampled pool')
            scope += f"; {score['reference_count']} tuning references"
            if 'fit_country' in record:
                scope += f"; transfer from {record['fit_country']}"
            rows.append(dict(setup=setup, model=record.get('model', record.get('variant', setup)),
                macro_f05=score['macro_f05'], precision=score['micro_precision'],
                recall=score['micro_recall'], threshold=score['threshold'],
                singleton_accuracy=score['singleton_accuracy'],
                false_positive_links=score.get('false_positive_links',
                    round(score['predicted_links'] * (1 - score['micro_precision']))),
                scope=scope,
                source=str(path)))
    return rows


def preserve_historical_rows(rows, output):
    if len(rows) != HISTORICAL_COUNT:
        raise ValueError(f'Found {len(rows)} historical scores, expected {HISTORICAL_COUNT}. '
                         'Use --artifacts with the complete original repository artifacts; '
                         'refusing to overwrite the ledger with a partial scan.')
    prior = output / 'experiments.json'
    if not prior.exists():
        return rows
    old = [row for row in json.loads(prior.read_text())
           if row['setup'] not in (DECISION_SETUP, TREE_SCREEN_SETUP)]
    if len(old) != HISTORICAL_COUNT:
        raise ValueError(f'Existing ledger has {len(old)} historical scores, expected {HISTORICAL_COUNT}.')
    keyed = {(row['setup'], row['model'], row['scope']): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError('Duplicate historical setup/model/scope in artifact reports.')
    for row in old:
        key = row['setup'], row['model'], row['scope']
        found = keyed.get(key)
        if found is None or {k: v for k, v in found.items() if k != 'source'} != {
                k: v for k, v in row.items() if k != 'source'}:
            raise ValueError(f'Historical score missing or changed: {key}. Refusing a partial ledger.')
    # The old source paths remain stable; the reports above have just been re-read
    # from --artifacts and checked against every historical score.
    return old


def decision_rows(report_dir):
    paths = {name: report_dir / name for name in (
        'decision_review.json', 'experiment_config.json', 'results.json',
        'source_balance_ablation.json')}
    review = json.loads(paths['decision_review.json'].read_text())
    for name in ('experiment_config.json', 'results.json', 'source_balance_ablation.json'):
        digest = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        if digest != review['artifact_sha256'][name]:
            raise ValueError(f'P1 report checksum mismatch: {paths[name]}')
    config = json.loads(paths['experiment_config.json'].read_text())
    results = json.loads(paths['results.json'].read_text())
    ablation = json.loads(paths['source_balance_ablation.json'].read_text())
    baseline = review['baseline']
    if (results['baseline'] != baseline or ablation['baseline'] != baseline
            or config['tune_references'] != baseline['reference_count']
            or config['tune_candidate_pairs'] != results['retrieval_ceiling']['pairs']):
        raise ValueError('P1 inputs disagree on baseline, population, or candidate pairs.')
    assignment = next(row for row in results['assignment']
                      if row['variant'] == 'assignment_delta_0.00')
    variants = (
        ('assignment_delta_0.00', assignment['score'], config['baseline_threshold'],
         paths['results.json']),
        ('calibrated_expected_f05_no_source_oof',
         ablation['calibrated_rule_without_source_bonus'], None,
         paths['source_balance_ablation.json']),
        ('calibrated_expected_f05_no_source_plus_assignment_0',
         review['no_source_rule_plus_pure_assignment'], None,
         paths['decision_review.json']),
        ('calibrated_expected_f05_nested_source_plus_assignment_0',
         review['nested_source_rule_plus_pure_assignment'], None,
         paths['decision_review.json']),
    )
    rows = []
    for name, score, threshold, source in variants:
        if (score['reference_count'] != baseline['reference_count']
                or score['predicted_links'] != score['true_positive_links'] + score['false_positive_links']
                or score['true_links'] != score['true_positive_links'] + score['missed_true_links']):
            raise ValueError(f'Inconsistent P1 score: {name}')
        calibration = 'v1 threshold' if threshold is not None else 'OOF calibration'
        rows.append(dict(setup=DECISION_SETUP, model=name,
            macro_f05=score['macro_f05'], precision=score['micro_precision'],
            recall=score['micro_recall'], threshold=threshold,
            singleton_accuracy=score['singleton_accuracy'],
            false_positive_links=score['false_positive_links'],
            scope=f"full pool; {score['reference_count']} reused medium TUNE references; "
                  f'fixed candidates; {calibration}; decision only', source=str(source)))
    return rows


def tree_screen_row(report_dir):
    complete = json.loads((report_dir / 'complete.json').read_text())
    report_path = report_dir / 'results.json'
    report_bytes = report_path.read_bytes()
    config_path = report_dir / 'experiment_config.json'
    report = json.loads(report_bytes)
    config = json.loads(config_path.read_text())
    score = report['evaluation']['tune']
    train = config['train_population']
    if (complete['state'] != 'complete'
            or hashlib.sha256(report_bytes).hexdigest() != complete['results_sha256']
            or hashlib.sha256(config_path.read_bytes()).hexdigest() != report['experiment_config_sha256']
            or complete['tree_checkpoint_sha256'] != report['tree_bundle']['sha256']
            or config['mode'] != 'screen' or config['positive_candidate_injection']
            or report['holdout_labels_used'] or train['train_tune_overlap']
            or train['tune_references'] != score['reference_count']
            or config['full_competing_target_pool'] != 10320219
            or report['tune_preparation']['candidate_pairs'] != report['evaluation']['retrieval_ceiling']['pairs']
            or score['predicted_links'] != score['true_positive_links'] + score['false_positive_links']):
        raise ValueError('Dense-only tree screen is incomplete or inconsistent; refusing to ledger it.')
    return dict(setup=TREE_SCREEN_SETUP,
        model=f"catboost_{train['train_references']}_train_dense_only_screen",
        macro_f05=score['macro_f05'], precision=score['micro_precision'],
        recall=score['micro_recall'], threshold=score['threshold'],
        singleton_accuracy=score['singleton_accuracy'],
        false_positive_links=score['false_positive_links'],
        scope=f"full pool; {score['reference_count']} reused medium TUNE references; "
              f"{train['train_references']} sampled medium TRAIN; dense-only compatibility screen",
        source=str(report_path))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts',
                        help='Complete historical reports (original repo when run from v2).')
    parser.add_argument('--p1-report', default='artifacts/v2_p1_decision')
    parser.add_argument('--tree-screen', default='artifacts/v2_tree_screen/dense_compat_final')
    parser.add_argument('--output', default='learnings')
    args = parser.parse_args(argv)
    output = Path(args.output)
    historical = preserve_historical_rows(historical_rows(Path(args.artifacts)), output)
    decisions = decision_rows(Path(args.p1_report))
    tree_screen = tree_screen_row(Path(args.tree_screen))
    rows = historical + decisions + [tree_screen]
    output.mkdir(parents=True, exist_ok=True)
    (output/'experiments.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (output/'experiments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Completed model and decision comparisons', '',
        f'{len(historical)} historical scored model configurations, including ensembles/blend weights and '
        f'hyperparameter variants; 1 dense-only CatBoost compatibility screen; '
        f'{len(decisions)} P1 decision-layer comparisons on saved v1 scores. '
        'Decision variants are not new models. Repeated baselines and failed runs are not counted.', '',
        '**Development scores do not establish submission readiness.** '
        'The initial model fell from 0.981500 to 0.832719 on the same 1,000 businesses when evaluated '
        'against 9,290,818 possible targets. Subsequent hard-pool experiments use all 10,320,219 original training targets. '
        'HOLDOUT-2 is reserved and unopened for v2 selection.', '',
        '## V2 P1 decision layer — reused TUNE only', '',
        'Same 12,000 medium TUNE references and 600,000 dense candidates retrieved against the full '
        '10,320,219-target pool. The saved v1 threshold baseline is 0.968992 / 494 false links; '
        'calibrated variants use entity-disjoint cross-fitting within TUNE and choose a per-entity '
        'prefix (null scalar threshold in CSV/JSON). No fresh holdout or new model training.', '',
        '| Decision variant | Macro F0.5 | Precision | Recall | False links | Singleton accuracy |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in decisions:
        lines.append(f"| {row['model']} | {row['macro_f05']:.6f} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['false_positive_links']} | {row['singleton_accuracy']:.6f} |")
    lines += ['', 'No-source rule + pure assignment improves paired TUNE macro F0.5 by 0.001107 '
        '(95% entity-bootstrap interval [0.000378, 0.001860]); the nested source bonus adds only '
        '0.000122 ([-0.000118, 0.000337]) and 12 more false links. Keep the higher-precision '
        'no-source candidate for a fresh holdout. The retrieved composite-key override adds zero '
        'accepted links and is not a duplicate scored row.', '',
        '## V2 P3 dense-only tree compatibility screen', '',
        'This is a scored CatBoost model fitted on only 1,200 medium TRAIN S1s; it is not a full-scale '
        'P3 tree or a union-candidate matcher. The 12,000 TUNE S1s were retrieved from all 10,320,219 targets.', '',
        '| Setup | Matcher | Macro F0.5 | Precision | Recall | False links | Scope |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |',
        f"| {tree_screen['setup']} | {tree_screen['model']} | {tree_screen['macro_f05']:.6f} | "
        f"{tree_screen['precision']:.6f} | {tree_screen['recall']:.6f} | "
        f"{tree_screen['false_positive_links']} | {tree_screen['scope']} |", '',
        '## P0 runtime and compatibility checks (not model scores)', '',
        'Scratch had 15.09 GiB free after cleanup; the bounded GPU-kind supervisor '
        'and workstation CatBoost/parquet compatibility smokes passed. '
        'Selective BF16 with FP32 rescoring within ±0.04 of the fixed threshold preserves the '
        'fresh FP32 accept set on 2,000 probe pairs and all 12,000 TUNE references (43,511 CE pairs; '
        '1,065 rescored), taking 34.11s versus 106.53s. This fresh FP32 run differs by one '
        'accepted link from the original saved v1 score; the runtime comparison is not a model gain. '
        'Batch256 reaches 1,414–1,474 pairs/s on the fixed benchmark, below the 1,700 adoption gate; '
        'retain batch128. GPU reports: `/workspace/az-ml-new/v2/amp_selective/report.json`; '
        'local runner/CatBoost smoke: `artifacts/v2_p0_runner/`.', '',
        'Historical report paths in CSV/JSON resolve under `--artifacts` '
        f'(`{Path(args.artifacts).resolve()}` for this regeneration).', '']
    for prefix,title in [('full pool','Full-pool development'),('sampled pool','Small-pool screening')]:
        subset=[r for r in historical if r['scope'].startswith(prefix)]
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
        'Thresholds use tuning labels only. The official test set has hidden labels; no entire-test F0.5 or F1 is available.', '',
        'Keep for independent validation: full-pool E5 plus lexical/phonetic retrieval, '
        'numeric-conflict features, and tree/BGE cross-encoder blends. Negative class weights '
        'depend on the candidate set: weight 1 won the initial lexical tree, while weight 4 won '
        'the quick-reference union tree. Prefer measured blends over the pure reranker in paired tests; '
        'heuristic-only matching and simple tree averaging remain weak leading options.', '',
        'BGE600 on the lexical hard pool improved on MiniLM300 by 0.006166 at matched 19,200 '
        'pair presentations; paired 95% tune interval [0.002883, 0.009499]. '
        'Union candidates plus 50% BGE reached 0.969812 versus the union tree 0.956008. '
        'These are tuning results; larger-data and fresh v2 holdout comparisons remain necessary. '
        'The earlier residual-tree 25% CE near-tie had 31 fewer false links than its 50% blend. '
        'Original-tree 50%-minus-25% CE has a tune-only paired 95% interval of [-0.002536, 0.002939]. '
        'P2 retrieval oracles, P4 cluster/ByT5 text probes, the frozen full-pool stress test, and anchor checks are '
        'documented in EXPERIMENT_DECISIONS.md rather than counted as new model configurations.', '',
        'The first E5 fine-tuning attempt failed on non-tensor tokenizer metadata. Tensor-only device transfer '
        'fixed it, and the successful retry is recorded above. No result is attributed to the failed run.', '']
    (output/'scoreboard.md').write_text('\n'.join(lines))
    print(f'Recorded {len(historical)} historical models + 1 tree compatibility model + '
          f'{len(decisions)} P1 decision-layer comparisons ({len(rows)} rows)')


if __name__ == '__main__':
    main()
