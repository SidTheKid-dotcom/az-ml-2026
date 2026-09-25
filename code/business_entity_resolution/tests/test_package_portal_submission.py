"""The uploadable ZIP must preserve validated outputs and the exact folder tree."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from package_portal_submission import build, digest


class PortalPackageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Path(temp.name)
        self.release = self.repo / 'release'
        self.release.mkdir()
        code = self.repo / 'code/business_entity_resolution'
        self.write(code / 'src/ber/example.py', 'VALUE = 1\n')
        self.write(code / 'README.md', 'Reproduce from supplied data.\n')
        self.write(code / 'requirements-gpu.txt', 'numpy==2.1.2\n')
        self.write(code / 'resources/validate_submission.py', '# supplied\n')
        for name in ('checkpoints.json', 'gpu_runtime.json', 'source_integrity.json'):
            self.write(self.repo / 'docs' / name, '{}\n')
        self.write(self.repo / 'problem-statement/models.yaml', 'models: []\n')
        self.write(self.repo / 'docs/submission_methodology.md', '**Team Name:** Techquilla Shots\n')
        matching = self.release / 'matching_results.tsv'
        candidate = self.release / 'candidate_pairs.tsv'
        self.write(matching, 'source1_entity_id\tmatched_entity_ids\nS1-1\tS2-2\n')
        self.write(candidate, 'source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-2,S3-3\n')
        self.write(self.release / 'frozen_pipeline.json', '{}\n')
        self.write(self.release / 'validation.json', json.dumps({
            'passed': True, 'stream_complete': True, 'errors': [],
            'counts': {'rows_checked': 1_732_544},
            'original_validator': {'passed_without_warnings': True},
            'file_sha256': {str(matching): digest(matching), str(candidate): digest(candidate)},
        }))
        self.write(self.release / 'ready.json', json.dumps({
            'ready_for_submission': True, 'source1_rows': 1_732_544,
            'pipeline_sha256': digest(self.release / 'frozen_pipeline.json'),
            'file_sha256': {name: digest(self.release / name) for name in
                            ('matching_results.tsv', 'candidate_pairs.tsv', 'validation.json')},
        }))
        self.args = argparse.Namespace(repo=self.repo, release=self.release,
            destination=self.repo / 'portal', team_name='Techquilla Shots')

    @staticmethod
    def write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def test_exact_tree_and_validated_bytes(self):
        archive = build(self.args)
        self.assertEqual(archive.name, 'Techquilla Shots_submission.zip')
        self.assertLess(archive.stat().st_size, 500_000_000)
        with ZipFile(archive) as bundle:
            self.assertEqual(set(bundle.namelist()), {
                'output/matching_results.tsv', 'output/candidate_pairs.tsv',
                'code/business_entity_resolution/README.md',
                'code/business_entity_resolution/requirements.txt',
                'code/business_entity_resolution/src/ber/example.py',
                'code/business_entity_resolution/src/resources/checkpoints.json',
                'code/business_entity_resolution/src/resources/frozen_dense_pipeline.json',
                'code/business_entity_resolution/src/resources/gpu_runtime.json',
                'code/business_entity_resolution/src/resources/models.yaml',
                'code/business_entity_resolution/src/resources/source_integrity.json',
                'code/business_entity_resolution/src/resources/validate_submission.py',
                'Documentation_template.md',
            })
            self.assertEqual(bundle.read('output/matching_results.tsv'),
                             (self.release / 'matching_results.tsv').read_bytes())
            self.assertEqual(bundle.read('output/candidate_pairs.tsv'),
                             (self.release / 'candidate_pairs.tsv').read_bytes())
        with self.assertRaises(FileExistsError):
            build(self.args)

    def test_changed_output_is_rejected(self):
        with (self.release / 'matching_results.tsv').open('a') as stream:
            stream.write('S1-2\tS3-4\n')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            build(self.args)


if __name__ == '__main__':
    unittest.main()
