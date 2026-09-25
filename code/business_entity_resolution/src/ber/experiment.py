"""Run comparable retrieval, matching, threshold, and held-out experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import generate_candidates
from .features import build_features
from .metrics import candidate_report, pair_labels, score_pairs, tune_threshold
from .models import fit_model, predict_proba
from .prepare import log


def json_write(path, data):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_data(directory):
    directory = Path(directory)
    reference = pd.read_parquet(directory / 'reference.parquet').fillna('')
    pool = pd.read_parquet(directory / 'pool.parquet').fillna('')
    truth = {k: set(v) for k, v in json.loads((directory / 'truth.json').read_text()).items()}
    return reference, pool, truth


def run(args):
    started = time.time()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference, pool, truth = load_data(args.data)
    config = {'top_k': args.top_k, 'channels': tuple(args.channels.split(',')),
              'ngram_range': (args.ngram_min, args.ngram_max), 'max_features': args.max_features,
              'n_jobs': args.threads, 'batch_size': args.batch_size,
              'min_similarity': args.min_similarity, 'exact_name_cap': args.exact_name_cap}
    metadata = json.loads((Path(args.data) / 'metadata.json').read_text())
    source_hash = hashlib.sha256(b''.join(
        (Path(__file__).parent / name).read_bytes()
        for name in ('candidates.py', 'normalization.py', 'features.py')
    )).hexdigest()
    if args.pairs:
        config['provided_pairs_sha256'] = hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()
    cache_key = hashlib.sha256(json.dumps({'config': config, 'data': metadata, 'code': source_hash}, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(args.cache) / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'pairs.parquet').exists():
        log('Loading cached candidates')
        pairs = pd.read_parquet(cache / 'pairs.parquet')
    else:
        log(f'Generating candidates: {config}')
        pairs = pd.read_parquet(args.pairs) if args.pairs else generate_candidates(reference, pool, **config)
        pairs.to_parquet(cache / 'pairs.parquet', index=False)
    log(f'{len(pairs):,} candidates; measuring retrieval')
    report_splits = ('train', 'tune', 'eval') if args.evaluate else ('train', 'tune')
    retrieval = {split: candidate_report(reference, pool, pairs, truth, reference.split.eq(split))
                 for split in report_splits}
    json_write(output / 'retrieval.json', retrieval)
    log(f'Tuning retrieval: {retrieval["tune"]}')
    if (cache / 'features.parquet').exists():
        log('Loading cached pair features')
        X = pd.read_parquet(cache / 'features.parquet')
    else:
        log('Building pair features')
        X = build_features(reference, pool, pairs)
        for column in ('phonetic_score', 'phonetic_only'):
            if column in pairs:
                X[column] = pairs[column].to_numpy(dtype=np.float32)
        X.to_parquet(cache / 'features.parquet', index=False)
    y = pair_labels(reference, pool, pairs, truth)
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    split_values = reference.split.to_numpy()[left]
    train_idx = np.flatnonzero(split_values == 'train')
    tune_idx = np.flatnonzero(split_values == 'tune')
    rng = np.random.default_rng(args.seed)
    # Sampling is training-only. All candidates remain in tuning/evaluation.
    if len(train_idx) > args.max_train_pairs:
        positives = train_idx[y[train_idx] == 1]
        negatives = train_idx[y[train_idx] == 0]
        cap = max(1, args.max_train_pairs - len(positives))
        negatives = rng.choice(negatives, min(len(negatives), cap), replace=False)
        train_idx = np.sort(np.r_[positives, negatives])
    log(f'Train {len(train_idx):,} pairs ({int(y[train_idx].sum()):,} matches); {len(X.columns)} features')
    results = []
    probabilities = {}
    winner = None
    model_specs = args.models.split(',')
    for kind in model_specs:
        model_started = time.time()
        log(f'Fitting {kind} on {args.device}')
        kwargs = {'seed': args.seed}
        if kind.startswith('catboost'):
            kwargs.update(iterations=args.iterations, depth=(8 if kind.endswith('deep') else 6),
                          thread_count=args.threads,
                          eval_set=(X.iloc[tune_idx], y[tune_idx]), verbose=100)
        elif kind == 'histgb':
            kwargs.update(max_iter=args.iterations)
        family = 'catboost' if kind.startswith('catboost') else kind
        model = fit_model(X.iloc[train_idx], y[train_idx], kind=family, device=args.device, **kwargs)
        probs = predict_proba(model, X)
        probabilities[kind] = probs
        threshold, trace = tune_threshold(reference, pool, pairs, probs, truth, reference.split.eq('tune'))
        tune = score_pairs(reference, pool, pairs, probs, truth, threshold, reference.split.eq('tune'))
        record = {'model': kind, 'tune': tune, 'elapsed_seconds': time.time()-model_started,
                  'train_pairs': len(train_idx), 'retrieval_cache': str(cache.resolve())}
        if args.evaluate:
            record['eval'] = score_pairs(reference, pool, pairs, probs, truth, threshold, reference.split.eq('eval'))
        results.append(record)
        log(json.dumps(record))
        joblib.dump({'model': model, 'threshold': threshold, 'feature_columns': list(X.columns),
                     'candidate_config': config, 'data_metadata': metadata, 'record': record}, output / f'{kind}.joblib')
        np.save(output / f'{kind}_probabilities.npy', probs)
        json_write(output / f'{kind}_thresholds.json', trace)
        if winner is None or tune['macro_f05'] > winner['tune']['macro_f05']:
            winner = record
        json_write(output / 'results.json', {'results': results, 'selected_by_tune': winner,
                    'retrieval': retrieval, 'data': metadata, 'elapsed_seconds': time.time()-started})
    # An ensemble is evaluated as its own candidate; selection remains tuning-only.
    trees = [k for k in probabilities if k.startswith('catboost')]
    if len(trees) > 1:
        probs = np.mean([probabilities[k] for k in trees], axis=0)
        threshold, trace = tune_threshold(reference, pool, pairs, probs, truth, reference.split.eq('tune'))
        record = {'model': 'catboost_ensemble', 'components': trees,
                  'tune': score_pairs(reference, pool, pairs, probs, truth, threshold, reference.split.eq('tune'))}
        if args.evaluate:
            record['eval'] = score_pairs(reference, pool, pairs, probs, truth, threshold, reference.split.eq('eval'))
        results.append(record)
        np.save(output / 'catboost_ensemble_probabilities.npy', probs)
        json_write(output / 'catboost_ensemble_thresholds.json', trace)
        if record['tune']['macro_f05'] > winner['tune']['macro_f05']:
            winner = record
    summary = {'results': results, 'selected_by_tune': winner, 'retrieval': retrieval,
               'data': metadata, 'elapsed_seconds': time.time()-started}
    json_write(output / 'results.json', summary)
    log(f'Winner selected by tuning split: {winner["model"]}')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/quick')
    p.add_argument('--output', default='artifacts/quick_v1')
    p.add_argument('--cache', default='artifacts/cache')
    p.add_argument('--models', default='heuristic,logistic,catboost,catboost_deep')
    p.add_argument('--device', choices=['CPU','GPU'], default='CPU')
    p.add_argument('--iterations', type=int, default=400)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--channels', default='name,address')
    p.add_argument('--ngram-min', type=int, default=3)
    p.add_argument('--ngram-max', type=int, default=4)
    p.add_argument('--max-features', type=int, default=100000)
    p.add_argument('--min-similarity', type=float, default=.08)
    p.add_argument('--exact-name-cap', type=int, default=100)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-train-pairs', type=int, default=300000)
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--pairs', help='Use an existing candidate union instead of regenerating')
    p.add_argument('--evaluate', action='store_true', help='Explicitly score development eval; off by default')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
