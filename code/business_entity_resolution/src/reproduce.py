"""Replay the selected model on every test record from an arbitrary local checkout."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parents[1]


def resource(name: str, fallback: Path) -> Path:
    packaged = CODE / 'resources' / name
    return packaged if packaged.exists() else fallback


def verify_sources(config: dict, code: Path = CODE) -> None:
    from ber.full_embeddings import file_digest
    prefix = 'code/business_entity_resolution/'
    for name, expected in config['source_sha256'].items():
        if not name.startswith(prefix):
            raise ValueError(f'Unexpected frozen source path: {name}')
        path = code / name.removeprefix(prefix)
        if file_digest(path) != expected:
            raise ValueError(f'Frozen source differs: {path}')


def verify_models(config: dict, models: Path) -> tuple[Path, Path, Path]:
    from ber.full_embeddings import checkpoint_identity, file_digest
    encoder = models / 'e5_small_ft200'
    tree = models / 'medium_dense_bge_ce1800/original_gate_and_tree.joblib'
    cross = models / 'medium_dense_bge_ce1800/checkpoint'
    if file_digest(tree) != config['tree_checkpoint_sha256']:
        raise ValueError('Selected tree identity mismatch')
    for path, expected in [(encoder, config['retrieval']['checkpoint_sha256']),
                           (cross, config['cross_encoder_sha256'])]:
        if checkpoint_identity(path)['sha256'] != expected:
            raise ValueError(f'Selected model identity mismatch: {path}')
    return encoder, tree, cross


def commands(dataset: Path, work: Path, encoder: Path, tree: Path,
             pipeline: Path, validator: Path) -> list[list[str]]:
    test = dataset / 'test'
    embeddings, candidates, output = (work / name for name in ('embeddings', 'candidates', 'output'))
    return [
        ['ber.full_embeddings', '--model', str(encoder), '--inputs',
         str(test / 'test_source2.tsv'), str(test / 'test_source3.tsv'),
         '--expected-rows', '4887273', '5082316', '--output', str(embeddings),
         '--batch-size', '512', '--chunk-size', '50000', '--max-length', '128', '--threads', '2'],
        ['ber.full_dense_candidates', '--query-input', str(test / 'test_source1.tsv'),
         '--expected-query-rows', '1732544', '--embeddings', str(embeddings / 'pool_embeddings.npy'),
         '--model', str(encoder), '--output', str(candidates), '--top-k', '50',
         '--batch-size', '512', '--query-batch-size', '512', '--pool-batch-size', '50000',
         '--threads', '2', '--device', 'cuda', '--backend', 'torch'],
        ['ber.predict_streaming', '--data', str(candidates), '--pairs', str(candidates / 'pairs.parquet'),
         '--checkpoint', str(tree), '--pipeline', str(pipeline), '--output', str(output),
         '--workers', '4', '--reference-batch-size', '1000', '--expected-references', '1732544',
         '--reuse-sorted-pairs', '--ce-threads', '2'],
        ['ber.validate_release', '--matching', str(output / 'matching_results.tsv'),
         '--candidate', str(output / 'candidate_pairs.tsv'), '--test-dir', str(test),
         '--validator', str(validator), '--report', str(output / 'validation.json')],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Supplied dataset/ containing test/')
    parser.add_argument('--work-dir', type=Path, required=True, help='New fast local disk directory (30GB+ free)')
    parser.add_argument('--models-root', type=Path, default=CODE / 'models')
    parser.add_argument('--pipeline', type=Path, default=resource('frozen_dense_pipeline.json', REPO / 'docs/frozen_dense_pipeline.json'))
    parser.add_argument('--validator', type=Path, default=resource('validate_submission.py', REPO / 'data/raw/utils/validate_submission.py'))
    parser.add_argument('--dry-run', action='store_true', help='Print commands without opening data/weights or launching jobs')
    args = parser.parse_args()
    dataset, work, models = (p.resolve() for p in (args.dataset, args.work_dir, args.models_root))
    pipeline = work / 'relocated_pipeline.json'
    encoder = models / 'e5_small_ft200'
    tree = models / 'medium_dense_bge_ce1800/original_gate_and_tree.joblib'
    plan = commands(dataset, work, encoder, tree, pipeline, args.validator.resolve())
    if args.dry_run:
        for command in plan:
            print(shlex.join([sys.executable, '-m', *command]))
        return
    # A separate work directory protects active runs and makes accidental reuse explicit.
    if work.exists() and any(work.iterdir()):
        raise ValueError('Use an empty work directory; inspect completed artifacts before rerunning')
    from ber.full_embeddings import file_digest
    config = json.loads(args.pipeline.read_text())
    verify_sources(config)
    encoder, tree, cross = verify_models(config, models)
    for source in (1, 2, 3):
        if not (dataset / 'test' / f'test_source{source}.tsv').is_file():
            raise ValueError(f'Missing test source {source}')
    expected_validator = json.loads(resource('source_integrity.json', REPO / 'docs/source_integrity.json').read_text())
    for row in expected_validator['files']:
        if '/dataset/test/' in row['path']:
            path = dataset / 'test' / Path(row['path']).name
            if file_digest(path) != row['sha256']:
                raise ValueError(f'Supplied test data identity mismatch: {path.name}')
    digest = next(row['sha256'] for row in expected_validator['files'] if row['path'].endswith('/validate_submission.py'))
    if file_digest(args.validator) != digest:
        raise ValueError('Supplied official validator changed')
    work.mkdir(parents=True, exist_ok=True)
    config['relocated_from_pipeline_sha256'] = file_digest(args.pipeline)
    config['tree_checkpoint'] = str(tree)
    config['cross_encoder_checkpoint'] = str(cross)
    config['retrieval']['path'] = str(encoder)
    pipeline.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(CODE / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    for command in plan:
        print(shlex.join([sys.executable, '-u', '-m', *command]), flush=True)
        subprocess.run([sys.executable, '-u', '-m', *command], env=env, check=True)
    report = json.loads((work / 'output/validation.json').read_text())
    if not report['passed']:
        raise ValueError('Output validation failed')
    print(f'Validated all test references: {work / "output"}. No leaderboard score computed.')


if __name__ == '__main__':
    main()
