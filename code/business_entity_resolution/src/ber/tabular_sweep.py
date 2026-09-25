"""Fast false-merge-focused matcher sweep on a fixed mined candidate set."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from .experiment import load_data, json_write
from .features import build_features
from .numeric_features import build_numeric_features
from .metrics import pair_labels, candidate_report, tune_threshold, score_pairs
from .models import fit_model, predict_proba
from .prepare import log


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference, pool, truth = load_data(args.data)
    # This sweep never loads or evaluates final-holdout labels.
    valid = reference.split.isin(['train', 'tune']).to_numpy()
    permitted_ids = set(reference.loc[valid, 'entity_id'])
    truth = {k:v for k,v in truth.items() if k in permitted_ids}
    pairs = pd.read_parquet(args.pairs)
    retain_pairs = valid[pairs.left_idx.to_numpy()]
    original_pair_count = len(pairs)
    pairs = pairs.loc[retain_pairs].reset_index(drop=True)
    if args.features:
        X = pd.read_parquet(args.features)
        if len(X) != original_pair_count:
            raise ValueError('Cached feature rows do not match original candidate rows')
        X = X.loc[retain_pairs].reset_index(drop=True)
    else:
        X = build_features(reference, pool, pairs)
        for name in ['phonetic_score','phonetic_only','dense_score','dense_only',
                     'lexical_scores_missing','lexical_retrieval_score']:
            if name in pairs:
                X[name] = pairs[name].to_numpy(dtype=np.float32)
    numeric = build_numeric_features(reference, pool, pairs)
    X.to_parquet(output/'base_features.parquet',index=False)
    numeric.to_parquet(output/'numeric_features.parquet',index=False)
    y = pair_labels(reference,pool,pairs,truth)
    split = reference.split.to_numpy()[pairs.left_idx.to_numpy()]
    train, tune = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'tune')
    retrieval = candidate_report(reference,pool,pairs,truth,reference.split.eq('tune'))
    log(f'Tuning full candidate retrieval: {retrieval}')
    baselines = []
    combined_features = pd.concat([X,numeric],axis=1)
    for checkpoint in args.baseline_checkpoints:
        bundle = joblib.load(checkpoint)
        columns = bundle['feature_columns']
        missing = set(columns) - set(combined_features)
        if missing:
            raise ValueError(f'Baseline requires missing features: {sorted(missing)}')
        baseline_probabilities = predict_proba(bundle['model'], combined_features[columns])
        frozen = score_pairs(reference,pool,pairs,baseline_probabilities,truth,bundle['threshold'],reference.split.eq('tune'))
        new_threshold, _ = tune_threshold(reference,pool,pairs,baseline_probabilities,truth,reference.split.eq('tune'))
        retuned = score_pairs(reference,pool,pairs,baseline_probabilities,truth,new_threshold,reference.split.eq('tune'))
        baseline_record = {'checkpoint':str(checkpoint),'frozen':frozen,'retuned_on_tune':retuned}
        baselines.append(baseline_record)
        log(json.dumps(baseline_record))
    json_write(output/'baselines.json',baselines)
    del combined_features
    reports = []
    winner = None
    for use_numeric in (False, True):
        features = pd.concat([X,numeric],axis=1) if use_numeric else X
        for depth in args.depths:
            for negative_weight in args.negative_weights:
                label=f'd{depth}_neg{negative_weight:g}_numeric{int(use_numeric)}'
                started=time.time()
                log(f'Training {label}: {len(train)} pairs')
                gpu_options = {'gpu_ram_part':args.gpu_ram_part} if args.device == 'GPU' else {}
                model = fit_model(features.iloc[train], y[train], kind='catboost', device=args.device,
                    seed=args.seed, iterations=args.iterations, depth=depth, thread_count=args.threads,
                    class_weights=[negative_weight,1.0], eval_set=(features.iloc[tune],y[tune]), verbose=False, **gpu_options)
                probabilities=predict_proba(model,features)
                threshold, trace=tune_threshold(reference,pool,pairs,probabilities,truth,reference.split.eq('tune'))
                score=score_pairs(reference,pool,pairs,probabilities,truth,threshold,reference.split.eq('tune'))
                predicted=(probabilities>=threshold)&(split=='tune')
                score['false_positive_links']=int((predicted & (y==0)).sum())
                record={'model':label,'tune':score,'numeric_features':use_numeric,'depth':depth,
                        'negative_weight':negative_weight,'elapsed_seconds':time.time()-started}
                reports.append(record)
                log(json.dumps(record))
                if winner is None or score['macro_f05'] > winner['tune']['macro_f05']:
                    winner=record
                    joblib.dump({'model':model,'threshold':threshold,'feature_columns':list(features.columns),
                                 'numeric_features':use_numeric,'record':record},output/'best.joblib')
                    np.save(output/'best_probabilities.npy',probabilities)
                    json_write(output/'best_thresholds.json',trace)
                json_write(output/'results.json',{'results':reports,'selected_by_tune':winner,
                           'candidate_report':retrieval,'baselines':baselines})
                del model
                gc.collect()
    pairs.to_parquet(output/'pairs.parquet',index=False)
    return reports


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--pairs',required=True)
    p.add_argument('--features')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='GPU',choices=['CPU','GPU'])
    p.add_argument('--iterations',type=int,default=500)
    p.add_argument('--threads',type=int,default=6)
    p.add_argument('--depths',type=int,nargs='+',default=[6,8])
    p.add_argument('--negative-weights',type=float,nargs='+',default=[1,2])
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--gpu-ram-part',type=float,default=.22)
    p.add_argument('--baseline-checkpoints',nargs='*',default=[])
    run(p.parse_args())

if __name__=='__main__':
    main()
