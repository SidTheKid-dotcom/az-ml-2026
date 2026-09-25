"""Guard the historical ledger and the provenance of P1 decision rows."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ber.scoreboard import decision_rows, main, tree_screen_row


class ScoreboardTests(unittest.TestCase):
    def test_incomplete_checkout_cannot_erase_the_historical_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'learnings'
            output.mkdir()
            prior = output / 'experiments.json'
            prior.write_text('[{"setup": "historical", "macro_f05": 0.9}]\n')
            markdown = output / 'scoreboard.md'
            markdown.write_text('historical ledger\n')

            with self.assertRaisesRegex(ValueError, 'partial scan'):
                main(['--artifacts', str(root / 'missing'), '--output', str(output)])
            self.assertEqual(prior.read_text(), '[{"setup": "historical", "macro_f05": 0.9}]\n')
            self.assertEqual(markdown.read_text(), 'historical ledger\n')
            self.assertFalse((output / 'experiments.csv').exists())

    def test_p1_rows_use_verified_crossfit_reports_not_repeated_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score = dict(reference_count=3, macro_f05=0.9, micro_precision=0.95,
                         micro_recall=0.85, predicted_links=2, true_positive_links=1,
                         false_positive_links=1, true_links=2, missed_true_links=1,
                         singleton_accuracy=1.0)
            config = dict(tune_references=3, tune_candidate_pairs=50, baseline_threshold=0.775)
            results = dict(baseline=score, retrieval_ceiling=dict(pairs=50),
                           assignment=[dict(variant='assignment_delta_0.00', score=score)])
            ablation = dict(baseline=score, calibrated_rule_without_source_bonus={**score,
                                                                                    'macro_f05': 0.91})
            for name, report in [('experiment_config.json', config), ('results.json', results),
                                 ('source_balance_ablation.json', ablation)]:
                (root / name).write_text(json.dumps(report))
            hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                      for name in ('experiment_config.json', 'results.json',
                                   'source_balance_ablation.json')}
            review = dict(baseline=score, artifact_sha256=hashes,
                          no_source_rule_plus_pure_assignment={**score, 'macro_f05': 0.92},
                          nested_source_rule_plus_pure_assignment={**score, 'macro_f05': 0.93})
            (root / 'decision_review.json').write_text(json.dumps(review))

            rows = decision_rows(root)
            self.assertEqual([row['macro_f05'] for row in rows], [0.9, 0.91, 0.92, 0.93])
            self.assertEqual([row['threshold'] for row in rows], [0.775, None, None, None])
            self.assertTrue(all('TUNE' in row['scope'] for row in rows))
            (root / 'source_balance_ablation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                decision_rows(root)

    def test_dense_tree_screen_requires_completed_scoped_model_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(mode='screen', positive_candidate_injection=False,
                          full_competing_target_pool=10320219,
                          train_population=dict(train_references=1200, tune_references=3,
                                                train_tune_overlap=0))
            config_path = root / 'experiment_config.json'
            config_path.write_text(json.dumps(config))
            score = dict(reference_count=3, macro_f05=0.8, micro_precision=0.9,
                         micro_recall=0.7, threshold=0.65, singleton_accuracy=1.0,
                         true_positive_links=2, false_positive_links=1, predicted_links=3)
            report = dict(experiment_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                          tree_bundle=dict(sha256='checkpoint'), holdout_labels_used=False,
                          tune_preparation=dict(candidate_pairs=150),
                          evaluation=dict(tune=score, retrieval_ceiling=dict(pairs=150)))
            report_path = root / 'results.json'
            report_path.write_text(json.dumps(report))
            (root / 'complete.json').write_text(json.dumps(dict(state='complete',
                results_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                tree_checkpoint_sha256='checkpoint')))

            row = tree_screen_row(root)
            self.assertEqual(row['macro_f05'], 0.8)
            self.assertIn('1200 sampled medium TRAIN; dense-only compatibility screen', row['scope'])
            report['evaluation']['tune']['macro_f05'] = 0.95
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'incomplete or inconsistent'):
                tree_screen_row(root)


if __name__ == '__main__':
    unittest.main()
