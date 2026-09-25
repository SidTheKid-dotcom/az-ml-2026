"""Release packaging must reject incomplete or modified submissions."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from ber.full_embeddings import checkpoint_identity, file_digest
from package_submission import build, checked_release


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.code = self.repo / 'code/business_entity_resolution'
        self.release = self.repo / 'release'
        self.models = self.repo / 'models'
        self.release.mkdir()
        for name in ['src/example.py', 'tests/example.py', 'README.md', 'requirements.txt', 'requirements-gpu.txt']:
            self.write(self.code / name, 'example\n')
        source = self.code / 'src/example.py'
        tree = self.models / 'medium_dense_bge_ce1800/original_gate_and_tree.joblib'
        encoder = self.models / 'e5_small_ft200'
        cross = self.models / 'medium_dense_bge_ce1800/checkpoint'
        self.write(tree, 'tree')
        for model in (encoder, cross):
            self.write(model / 'model.safetensors', 'fixture weights')
        self.pipeline = self.repo / 'docs/frozen_dense_pipeline.json'
        self.write_json(self.pipeline, {
            'tree_checkpoint_sha256': file_digest(tree),
            'cross_encoder_sha256': checkpoint_identity(cross)['sha256'],
            'retrieval': {'checkpoint_sha256': checkpoint_identity(encoder)['sha256']},
            'source_sha256': {'code/business_entity_resolution/src/example.py': file_digest(source)},
        })
        validator = self.repo / 'data/raw/utils/validate_submission.py'
        self.write(validator, 'unchanged fixture validator')
        self.write_json(self.repo / 'docs/source_integrity.json', {'files': [
            {'path': 'data/raw/utils/validate_submission.py', 'sha256': file_digest(validator)}]})
        for name in ['docs/gpu_runtime.json', 'docs/checkpoints.json', 'learnings/first_submission_holdout.json']:
            self.write_json(self.repo / name, {})
        self.write(self.repo / 'problem-statement/models.yaml', 'models: []\n')
        self.write(self.repo / 'docs/submission_methodology.md', '# Methodology\n\n**Draft status:** pending\n')
        self.write(self.release / 'matching_results.tsv', 'source1_entity_id\tmatched_entity_ids\nS1-1\tS2-2\n')
        self.write(self.release / 'candidate_pairs.tsv', 'source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-2,S3-3\n')
        self.write(self.release / 'frozen_pipeline.json', self.pipeline.read_text())
        self.write(self.release / 'validation.validator.log', 'PASS\n')
        self.write_json(self.release / 'validation.json', {
            'passed': True, 'stream_complete': True,
            'counts': {'rows_checked': 1732544, 'candidate_links': 86627200},
            'original_validator': {'passed_without_warnings': True}})
        self.write_json(self.release / 'report.json', {})
        self.write_json(self.release / 'complete.json', {'state': 'complete',
            'report': {'benchmark_only': False, 'pipeline_sha256': file_digest(self.pipeline)}})
        self.write_json(self.release / 'ready.json', {
            'ready_for_submission': True, 'source1_rows': 1732544,
            'pipeline_sha256': file_digest(self.pipeline),
            'file_sha256': {p.name: file_digest(p) for p in self.release.iterdir()}})
        self.args = argparse.Namespace(repo=self.repo, release=self.release, models_root=self.models,
            destination=self.repo / 'bundles', team_name='Techquilla Shots', team_members='Fixture member')

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def write_json(self, path, value):
        self.write(path, json.dumps(value))

    def test_pending_release_is_rejected(self):
        (self.release / 'ready.json').unlink()
        with self.assertRaises(FileNotFoundError):
            checked_release(self.release, self.pipeline)

    def test_changed_prediction_is_rejected(self):
        with (self.release / 'matching_results.tsv').open('a') as stream:
            stream.write('S1-2\tS3-9\n')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            checked_release(self.release, self.pipeline)

    def test_partial_coverage_is_rejected(self):
        path = self.release / 'validation.json'
        report = json.loads(path.read_text())
        report['counts']['rows_checked'] = 50000
        self.write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'validation is required'):
            checked_release(self.release, self.pipeline)

    def test_changed_checkpoint_is_rejected(self):
        self.write(self.models / 'e5_small_ft200/model.safetensors', 'other weights')
        with self.assertRaisesRegex(ValueError, 'model identity mismatch'):
            build(self.args)

    def test_bundle_layout_and_bytes_and_no_overwrite(self):
        # Small synthetic files stand in for outputs with trusted validator reports;
        # full-data validation itself is tested separately in test_validate_release.
        with patch('package_submission.subprocess.run', return_value=SimpleNamespace(stdout='fixturecommit\n')):
            archive = build(self.args)
            self.assertEqual(archive.name, 'Techquilla Shots_submission.zip')
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
                    self.assertEqual(bundle.read('output/' + name), (self.release / name).read_bytes())
                self.assertIn('code/business_entity_resolution/models/e5_small_ft200/model.safetensors', bundle.namelist())
                self.assertIn('code/business_entity_resolution/resources/validate_submission.py', bundle.namelist())
                self.assertIn('Team Name:** Techquilla Shots', bundle.read('Documentation_template.md').decode())
                manifest = json.loads(bundle.read('package_manifest.json'))
                self.assertEqual(manifest['pipeline_sha256'], file_digest(self.pipeline))
            self.assertTrue(archive.with_suffix('.zip.sha256').is_file())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                build(self.args)


if __name__ == '__main__':
    unittest.main()
