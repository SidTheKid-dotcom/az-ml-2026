"""Build a self-contained final competition ZIP only from a verified v1 release."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

from ber.full_embeddings import file_digest
from reproduce import verify_models, verify_sources


def checked_release(release: Path, pipeline: Path) -> dict:
    ready = json.loads((release / 'ready.json').read_text())
    validation = json.loads((release / 'validation.json').read_text())
    complete = json.loads((release / 'complete.json').read_text())
    if (ready.get('ready_for_submission') is not True or ready.get('source1_rows') != 1732544
            or ready.get('pipeline_sha256') != file_digest(pipeline)):
        raise ValueError('Release is not ready for this exact frozen pipeline')
    if (not validation.get('passed') or not validation.get('stream_complete')
            or validation['counts']['rows_checked'] != 1732544
            or validation['counts']['candidate_links'] != 86627200
            or not validation['original_validator']['passed_without_warnings']):
        raise ValueError('Full official and strict validation is required')
    if (complete.get('state') != 'complete' or complete['report'].get('benchmark_only')
            or complete['report']['pipeline_sha256'] != ready['pipeline_sha256']):
        raise ValueError('Prediction completion does not match selected procedure')
    for name in ('matching_results.tsv', 'candidate_pairs.tsv', 'validation.json',
                 'validation.validator.log', 'frozen_pipeline.json', 'report.json', 'complete.json'):
        if file_digest(release / name) != ready['file_sha256'].get(name):
            raise ValueError(f'Validated release bytes changed: {name}')
    return ready


def build(args) -> Path:
    repo, release, models = (Path(p).resolve() for p in (args.repo, args.release, args.models_root))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.team_name):
        raise ValueError('Use a filename-safe team name (letters, digits, underscore, hyphen)')
    code = repo / 'code/business_entity_resolution'
    pipeline = repo / 'docs/frozen_dense_pipeline.json'
    ready = checked_release(release, pipeline)
    config = json.loads(pipeline.read_text())
    verify_sources(config, code)
    encoder, tree, cross = verify_models(config, models)
    integrity = json.loads((repo / 'docs/source_integrity.json').read_text())
    validator = code / 'resources/validate_submission.py'
    if not validator.is_file():
        validator = repo / 'data/raw/utils/validate_submission.py'
    expected = next(r['sha256'] for r in integrity['files'] if r['path'].endswith('/validate_submission.py'))
    if file_digest(validator) != expected:
        raise ValueError('Official validator identity differs from the supplied archive')
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f'{args.team_name}_submission.zip'
    temporary = archive.with_suffix('.zip.tmp')
    if archive.exists() or temporary.exists():
        raise ValueError('Archive already exists; use a new destination rather than overwriting history')
    entries: dict[str, Path] = {}
    prefix = 'code/business_entity_resolution/'
    for directory in ('src', 'tests'):
        for path in sorted((code / directory).rglob('*.py')):
            if '__pycache__' not in path.parts:
                entries[prefix + str(path.relative_to(code))] = path
    for path in [code / 'README.md', *sorted(code.glob('requirements*.txt')), *sorted(code.glob('run_*.sh'))]:
        entries[prefix + path.name] = path
    for path in (encoder, cross):
        for child in sorted(path.rglob('*')):
            if child.is_file() and '__pycache__' not in child.parts:
                entries[prefix + 'models/' + str(child.relative_to(models))] = child
    entries[prefix + 'models/' + str(tree.relative_to(models))] = tree
    for name in ('frozen_dense_pipeline.json', 'gpu_runtime.json', 'source_integrity.json', 'checkpoints.json'):
        entries[prefix + 'resources/' + name] = repo / 'docs' / name
    entries[prefix + 'resources/models.yaml'] = repo / 'problem-statement/models.yaml'
    entries[prefix + 'resources/validate_submission.py'] = validator
    entries[prefix + 'resources/holdout_score.json'] = repo / 'learnings/first_submission_holdout.json'
    for name in ('ready.json', 'validation.json', 'validation.validator.log', 'report.json', 'complete.json'):
        entries[prefix + 'resources/release/' + name] = release / name
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        entries['output/' + name] = release / name
    methodology = (repo / 'docs/submission_methodology.md').read_text()
    methodology = re.sub(r'^\*\*Draft status:\*\*.*$',
        '**Release status:** All 1,732,544 test references passed official and strict validation. '
        'This verifies format and coverage; the official score remains platform-provided.', methodology, flags=re.M)
    methodology = methodology.replace('final matching/TSV validation is running.', 'final matching and TSV validation passed.')
    methodology = methodology.replace('Complete official-test prediction and validation are running; release claims will be updated only from verified artifacts.',
                                      'Complete official-test prediction and validation passed; both final files are included unchanged.')
    methodology = methodology.replace('the official validator report and final file hashes will be appended after verification.',
                                      'validator reports and file hashes are included under `code/business_entity_resolution/resources/release/`.')
    methodology = re.sub(r'^\*\*Team Name:\*\*.*\n?', '', methodology, flags=re.M)
    header = f'**Team Name:** {args.team_name}\n'
    if args.team_members:
        header += f'**Team Members:** {args.team_members}\n'
    methodology = header + '\n' + methodology.lstrip()
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, capture_output=True, check=True).stdout.strip()
    manifest = {'git_commit': commit, 'pipeline_sha256': ready['pipeline_sha256'],
                'official_test_score': None, 'files': {}}
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
            for name, path in sorted(entries.items()):
                before = file_digest(path)
                bundle.write(path, name)
                if file_digest(path) != before:
                    raise ValueError(f'File changed during packaging: {path}')
                manifest['files'][name] = {'sha256': before, 'bytes': path.stat().st_size}
            bundle.writestr('Documentation_template.md', methodology)
            manifest['files']['Documentation_template.md'] = {
                'sha256': hashlib.sha256(methodology.encode()).hexdigest(), 'bytes': len(methodology.encode())}
            bundle.writestr('package_manifest.json', json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(temporary) as bundle:
            for name, identity in manifest['files'].items():
                digest = hashlib.sha256()
                with bundle.open(name) as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        digest.update(chunk)
                if digest.hexdigest() != identity['sha256'] or bundle.getinfo(name).file_size != identity['bytes']:
                    raise ValueError(f'Packaged bytes differ: {name}')
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    archive.with_suffix('.zip.sha256').write_text(f'{file_digest(archive)}  {archive.name}\n')
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--models-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--team-name', required=True)
    parser.add_argument('--team-members', default='', help='Optional real member names; omitted when unspecified')
    print(build(parser.parse_args()))


if __name__ == '__main__':
    main()
