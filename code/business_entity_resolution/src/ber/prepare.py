"""Safely extract supplied files and build fixed, entity-disjoint experiment splits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def extract(archive, destination):
    destination = Path(destination)
    with ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename
            if not name.startswith('student_resource/') or member.is_dir():
                continue
            relative = Path(name).relative_to('student_resource')
            if '..' in relative.parts or relative.name == '.DS_Store':
                continue
            target = destination / relative
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            log(f'Extracting {relative}')
            with z.open(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    return destination / 'dataset'


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t')


def reservoir(path, count, seed, exclude=None):
    rng, chosen, seen = random.Random(seed), [], 0
    for row in rows(path):
        if exclude and row['entity_id'] in exclude:
            continue
        seen += 1
        if len(chosen) < count:
            chosen.append(row)
        else:
            j = rng.randrange(seen)
            if j < count:
                chosen[j] = row
    return chosen


def prepare(dataset, destination, count=12000, distractors=160000, seed=20260925, exclude_reference_file=None):
    dataset, destination = Path(dataset), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    log(f'Sampling {count} reference businesses')
    excluded = set()
    if exclude_reference_file:
        excluded = set(pd.read_parquet(exclude_reference_file, columns=['entity_id']).entity_id)
    sampled = reservoir(dataset / 'train/train_source1.tsv', count, seed, exclude=excluded)
    reference = pd.DataFrame(sampled).fillna('')
    reference['split'] = ''
    rng = random.Random(seed + 1)
    for country, group in reference.groupby('country', sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        train_end, tune_end = int(.60 * len(indices)), int(.80 * len(indices))
        reference.loc[indices[:train_end], 'split'] = 'train'
        reference.loc[indices[train_end:tune_end], 'split'] = 'tune'
        reference.loc[indices[tune_end:], 'split'] = 'eval'
    selected = set(reference.entity_id)
    truth = {}
    for row in rows(dataset / 'train/train_ground_truth.tsv'):
        if row['source1_entity_id'] in selected:
            truth[row['source1_entity_id']] = [x for x in row['matched_entity_ids'].split(',') if x]
    if set(truth) != selected:
        raise ValueError('Missing truth for selected references')
    positives = {entity for ids in truth.values() for entity in ids}
    log(f'Reading source pools; preserving all {len(positives)} true matches')
    pool_rows = []
    for source in (2, 3):
        rng = random.Random(seed + source)
        negatives, seen, positive_rows = [], 0, []
        cap = distractors // 2 + (distractors % 2 if source == 2 else 0)
        for row in rows(dataset / f'train/train_source{source}.tsv'):
            if row['entity_id'] in positives:
                positive_rows.append(row)
                continue
            seen += 1
            if len(negatives) < cap:
                negatives.append(row)
            else:
                j = rng.randrange(seen)
                if j < cap:
                    negatives[j] = row
        log(f'Source {source}: {len(positive_rows)} matches and {len(negatives)} distractors')
        pool_rows.extend(positive_rows)
        pool_rows.extend(negatives)
    random.Random(seed + 11).shuffle(pool_rows)
    pool = pd.DataFrame(pool_rows).fillna('')
    if pool.entity_id.duplicated().any():
        raise ValueError('Duplicate IDs in candidate pool')
    if positives - set(pool.entity_id):
        raise ValueError('Positive training records missing from pool')
    all_ids = sorted(selected)
    digest = hashlib.sha256('\n'.join(all_ids).encode()).hexdigest()
    metadata = {
        'seed': seed, 'references': len(reference), 'pool_records': len(pool),
        'true_links': len(positives), 'distractors_requested': distractors,
        'reference_id_sha256': digest,
        'excluded_prior_references': len(excluded),
        'split_counts': reference.groupby(['split', 'country']).size().to_dict(),
        'limitations': ['Sampled candidate pool is easier than the full release.',
                        'France has no training labels and is not measured by this validation.',
                        'Use tuning split for model selection; eval is a separate entity-disjoint check.'],
    }
    metadata['split_counts'] = {f'{a}/{b}': int(v) for (a, b), v in metadata['split_counts'].items()}
    reference.to_parquet(destination / 'reference.parquet', index=False)
    pool.to_parquet(destination / 'pool.parquet', index=False)
    (destination / 'truth.json').write_text(json.dumps(truth), encoding='utf-8')
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    log(f'Prepared {destination}: {json.dumps(metadata)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('data/quick'))
    parser.add_argument('--references', type=int, default=12000)
    parser.add_argument('--distractors', type=int, default=160000)
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--exclude-reference-file', type=Path)
    args = parser.parse_args()
    dataset = args.dataset or extract(args.archive, args.raw_dir)
    prepare(dataset, args.output, args.references, args.distractors, args.seed, args.exclude_reference_file)


if __name__ == '__main__':
    main()
