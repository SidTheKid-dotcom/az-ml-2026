"""Apply one frozen matcher, writing both required TSVs and optional validation.

Feature batches retain every candidate for each reference. This command never
fits a model, changes a threshold, or uses truth to change predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .experiment import json_write
from .features import build_features
from .numeric_features import build_numeric_features, NAMES as NUMERIC_NAMES
from .residual_features import build_residual_features, NAMES as RESIDUAL_NAMES
from .models import predict_proba
from .metrics import candidate_report, score_pairs, pair_labels
from .uncertainty import paired_bootstrap
from .prepare import log


def feature_batch(reference, pool, pairs, columns):
    features = build_features(reference, pool, pairs)
    for column in ('phonetic_score', 'phonetic_only', 'dense_score', 'dense_only',
                   'lexical_scores_missing', 'lexical_retrieval_score'):
        if column in pairs:
            features[column] = pairs[column].to_numpy(dtype=np.float32)
    if set(columns) & set(NUMERIC_NAMES):
        features = pd.concat([features, build_numeric_features(reference,pool,pairs)], axis=1)
    if set(columns) & set(RESIDUAL_NAMES):
        features = pd.concat([features,build_residual_features(reference,pool,pairs)],axis=1)
    missing = set(columns) - set(features)
    if missing:
        raise ValueError(f'Candidate data lacks required model features: {sorted(missing)}')
    return features[columns]


def run(args):
    started = time.time()
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(data/'reference.parquet').fillna('')
    pool = pd.read_parquet(data/'pool.parquet').fillna('')
    if reference.entity_id.duplicated().any() or pool.entity_id.duplicated().any():
        raise ValueError('Reference and pool IDs must each be unique')
    bundle = joblib.load(args.checkpoint)
    threshold = float(bundle['threshold'])
    anchor_threshold = float(bundle.get('anchor_threshold',0))
    pairs = pd.read_parquet(args.pairs)
    if pairs.duplicated(['left_idx','right_idx']).any():
        raise ValueError('Duplicate candidate pairs would corrupt scoring')
    if len(pairs) and (pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
                      or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool)):
        raise ValueError('Candidate indices do not match the supplied records')
    pairs = pairs.sort_values(['left_idx','right_idx'],kind='stable').reset_index(drop=True)
    selected = np.ones(len(reference),dtype=bool)
    if args.split:
        if 'split' not in reference:
            raise ValueError('--split requires reference split labels')
        selected = reference.split.eq(args.split).to_numpy()
        if not selected.any():
            raise ValueError('Requested reference split is empty')
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    pool_ids = pool.entity_id.to_numpy()
    probabilities = np.lib.format.open_memmap(output/'probabilities.npy',mode='w+',
                                              dtype=np.float32,shape=(len(pairs),))
    probabilities[:] = np.nan
    match_path, candidate_path = output/'matching_results.tsv', output/'candidate_pairs.tsv'
    with match_path.open('w',newline='') as mstream, candidate_path.open('w',newline='') as cstream:
        matches, candidates = csv.writer(mstream, delimiter='\t', lineterminator='\n'), csv.writer(cstream, delimiter='\t', lineterminator='\n')
        matches.writerow(['source1_entity_id','matched_entity_ids'])
        candidates.writerow(['source1_entity_id','candidate_entity_ids'])
        for start in range(0,len(reference),args.reference_batch_size):
            end = min(start+args.reference_batch_size,len(reference))
            begin, stop = np.searchsorted(left,[start,end])
            batch = pairs.iloc[begin:stop]
            if len(batch):
                values = predict_proba(bundle['model'],feature_batch(reference,pool,batch,bundle['feature_columns']))
                if not np.isfinite(values).all():
                    raise ValueError('Non-finite matcher probabilities')
                if anchor_threshold>0:
                    unique,inverse=np.unique(batch.left_idx.to_numpy(),return_inverse=True)
                    maxima=np.full(len(unique),-1.,dtype=np.float32)
                    np.maximum.at(maxima,inverse,values)
                    values=np.where(maxima[inverse]>=anchor_threshold,values,-1.)
                probabilities[begin:stop] = values
            for i in range(start,end):
                if not selected[i]:
                    continue
                first,last = np.searchsorted(left,[i,i+1])
                rights = pairs.right_idx.iloc[first:last].to_numpy(dtype=np.int64)
                ids = pool_ids[rights]
                accepted = np.asarray(probabilities[first:last],dtype=np.float64) >= threshold
                matches.writerow([reference.iloc[i].entity_id,','.join(sorted(ids[accepted]))])
                candidates.writerow([reference.iloc[i].entity_id,','.join(sorted(ids))])
            if start % (args.reference_batch_size*10) == 0:
                log(f'Frozen prediction: {end:,}/{len(reference):,} references')
    probabilities.flush()
    # Sorted pairs accompany the probability array, preserving exact row alignment.
    pairs.to_parquet(output/'scored_pairs.parquet',index=False)
    report = {'checkpoint':str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'threshold':threshold,'anchor_threshold':anchor_threshold,'threshold_changed':False,'reference_count':int(selected.sum()),
        'candidate_pairs':int(selected[left].sum()),'split':args.split,
        'elapsed_seconds':time.time()-started}
    if args.score:
        supplied = json.loads((data/'truth.json').read_text())
        ids = set(reference.loc[selected,'entity_id'])
        if ids - supplied.keys():
            raise ValueError('Truth must explicitly cover every scored reference, including singletons')
        truth = {key:set(value) for key,value in supplied.items() if key in ids}
        report['score'] = score_pairs(reference,pool,pairs,probabilities,truth,threshold,selected)
        report['retrieval'] = candidate_report(reference,pool,pairs,truth,selected)
        accepted = np.asarray(probabilities,dtype=np.float64) >= threshold
        positive = pair_labels(reference,pool,pairs,truth).astype(bool)
        predicted_count = np.bincount(left[accepted],minlength=len(reference))
        tp = np.bincount(left[accepted & positive],minlength=len(reference))
        true_count = np.array([len(truth.get(entity,set())) for entity in reference.entity_id])
        denominator = 4*predicted_count+true_count
        f05 = np.divide(5.*tp,denominator,out=np.zeros(len(reference)),where=denominator>0)
        f05[(predicted_count==0)&(true_count==0)] = 1
        f1den = predicted_count+true_count
        f1 = np.divide(2.*tp,f1den,out=np.zeros(len(reference)),where=f1den>0)
        f1[(predicted_count==0)&(true_count==0)] = 1
        entities = pd.DataFrame({'source1_entity_id':reference.entity_id,'country':reference.country,
            'true_links':true_count,'predicted_links':predicted_count,'true_positive_links':tp,
            'false_positive_links':predicted_count-tp,'false_negative_links':true_count-tp,
            'macro_f05':f05,'macro_f1':f1}).loc[selected]
        entities.to_csv(output/'entities.tsv',sep='\t',index=False)
        report['score']['macro_f1'] = float(f1[selected].mean())
        report['score']['false_positive_links'] = int((predicted_count-tp)[selected].sum())
        report['score']['false_negative_links'] = int((true_count-tp)[selected].sum())
        if selected.sum()>1:
            report['macro_f05_uncertainty'] = paired_bootstrap(f05[selected],n_bootstrap=10000)
    json_write(output/'report.json',report)
    log(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--pairs',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split')
    parser.add_argument('--score',action='store_true')
    parser.add_argument('--reference-batch-size',type=int,default=1000)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
