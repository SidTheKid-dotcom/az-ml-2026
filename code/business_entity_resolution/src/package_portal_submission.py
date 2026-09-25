"""Build the organizer's under-500 MB ZIP from a completed, validated release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZIP_BZIP2, ZipFile


LIMIT_BYTES = 500_000_000
EXPECTED_ROWS = 1_732_544


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def validated_release(release: Path) -> dict:
    ready_path = release / 'ready.json'
    report_path = release / 'validation.json'
    ready = json.loads(ready_path.read_text())
    report = json.loads(report_path.read_text())
    if ready.get('ready_for_submission') is not True or ready.get('source1_rows') != EXPECTED_ROWS:
        raise ValueError('Release is not ready for every official test S1')
    if (report.get('passed') is not True or report.get('stream_complete') is not True
            or report.get('counts', {}).get('rows_checked') != EXPECTED_ROWS
            or report.get('errors') or report.get('original_validator', {}).get('passed_without_warnings') is not True):
        raise ValueError('Official and strict all-row validation must pass')
    for name in ('matching_results.tsv', 'candidate_pairs.tsv', 'validation.json'):
        expected = ready.get('file_sha256', {}).get(name)
        if not expected or digest(release / name) != expected:
            raise ValueError(f'Validated release bytes changed: {name}')
    report_hashes = report.get('file_sha256', {})
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        values = [value for path, value in report_hashes.items() if Path(path).name == name]
        if len(values) != 1 or values[0] != ready['file_sha256'][name]:
            raise ValueError(f'Validator did not check the current {name}')
    pipeline = release / 'frozen_pipeline.json'
    if not pipeline.is_file() or digest(pipeline) != ready.get('pipeline_sha256'):
        raise ValueError('Release pipeline identity differs from the ready marker')
    return ready


def build(args: argparse.Namespace) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9 _-]{0,78}[A-Za-z0-9])?', args.team_name):
        raise ValueError('Team name must be safe as a filename')
    repo, release, destination = (Path(path).resolve() for path in (args.repo, args.release, args.destination))
    ready = validated_release(release)
    code = repo / 'code/business_entity_resolution'
    methodology = (repo / 'docs/submission_methodology.md').read_text(encoding='utf-8')
    if f'**Team Name:** {args.team_name}' not in methodology or '[Your Team Name]' in methodology:
        raise ValueError('Fill the methodology with the exact team name first')

    files: dict[str, Path] = {
        'output/matching_results.tsv': release / 'matching_results.tsv',
        'output/candidate_pairs.tsv': release / 'candidate_pairs.tsv',
        'code/business_entity_resolution/README.md': code / 'README.md',
        'code/business_entity_resolution/requirements.txt': code / 'requirements-gpu.txt',
        'Documentation_template.md': repo / 'docs/submission_methodology.md',
    }
    for source in sorted((code / 'src').rglob('*.py')):
        if '__pycache__' not in source.parts:
            files['code/business_entity_resolution/' + str(source.relative_to(code))] = source
    resources = {
        'checkpoints.json': repo / 'docs/checkpoints.json',
        'frozen_dense_pipeline.json': release / 'frozen_pipeline.json',
        'gpu_runtime.json': repo / 'docs/gpu_runtime.json',
        'models.yaml': repo / 'problem-statement/models.yaml',
        'source_integrity.json': repo / 'docs/source_integrity.json',
        'validate_submission.py': code / 'resources/validate_submission.py',
    }
    for name, path in resources.items():
        files['code/business_entity_resolution/src/resources/' + name] = path
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError('A required output, source, dependency or resource file is missing')
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f'{args.team_name}_submission.zip'
    temporary = destination / (archive.name + '.tmp')
    if archive.exists() or temporary.exists():
        raise FileExistsError('Use a fresh destination; never overwrite a submission')
    expected = {name: digest(path) for name, path in files.items()}
    try:
        with ZipFile(temporary, 'x', compression=ZIP_BZIP2, compresslevel=9, allowZip64=True) as bundle:
            for name, path in sorted(files.items()):
                bundle.write(path, name)
        if temporary.stat().st_size > LIMIT_BYTES:
            raise ValueError(f'Portal ZIP exceeds {LIMIT_BYTES} bytes')
        with ZipFile(temporary) as bundle:
            if set(bundle.namelist()) != set(files) or len(bundle.namelist()) != len(files):
                raise ValueError('ZIP layout contains missing, extra or duplicate entries')
            for name, wanted in expected.items():
                sha = hashlib.sha256()
                with bundle.open(name) as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                        sha.update(block)
                if sha.hexdigest() != wanted:
                    raise ValueError(f'Archived bytes differ: {name}')
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            if expected['output/' + name] != ready['file_sha256'][name]:
                raise ValueError(f'Output changed while packaging: {name}')
        temporary.replace(archive)
        archive.with_suffix('.zip.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
        return archive
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--team-name', required=True)
    print(build(parser.parse_args()))


if __name__ == '__main__':
    main()
