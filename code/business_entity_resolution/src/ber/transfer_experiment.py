"""Country-transfer stress test using full-pool hard negatives.

Each matcher is fitted on one country's TRAIN references and its threshold is
chosen on that same country's TUNE references. The other country's TUNE set is
then scored at that frozen threshold. This diagnoses transfer; it is not a
measurement of France or a replacement for the untouched final holdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd

from .experiment import load_data, json_write
from .metrics import pair_labels, score_pairs, tune_threshold
from .models import fit_model, predict_proba
from .prepare import log


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--base-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--iterations',type=int,default=650)
    args=p.parse_args()
    reference,pool,truth=load_data(args.data)
    base,output=Path(args.base_dir),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_parquet(base/'pairs.parquet')
    if not reference.split.isin(['train','tune']).all():
        raise ValueError('Transfer test accepts only train/tune references')
    X=pd.concat([pd.read_parquet(base/'base_features.parquet'),
                 pd.read_parquet(base/'numeric_features.parquet')],axis=1)
    if len(X)!=len(pairs):
        raise ValueError('Feature/candidate row count mismatch')
    y=pair_labels(reference,pool,pairs,truth)
    rows=pairs.left_idx.to_numpy(dtype=np.int64)
    reports=[]
    for country in sorted(reference.country.unique()):
        started=time.time()
        train=(reference.split.eq('train') & reference.country.eq(country)).to_numpy()
        tune=(reference.split.eq('tune') & reference.country.eq(country)).to_numpy()
        other=(reference.split.eq('tune') & ~reference.country.eq(country)).to_numpy()
        train_rows,tune_rows=np.flatnonzero(train[rows]),np.flatnonzero(tune[rows])
        log(f'Fitting transfer matcher on {country}: {train.sum():,} references')
        # Fixed before observing transfer scores; not selected on the other country.
        model=fit_model(X.iloc[train_rows],y[train_rows],kind='catboost',device='GPU',
            seed=20260925,iterations=args.iterations,depth=8,class_weights=[2.,1.],
            thread_count=args.threads,gpu_ram_part=.22,
            eval_set=(X.iloc[tune_rows],y[tune_rows]),verbose=False)
        probabilities=predict_proba(model,X)
        threshold,trace=tune_threshold(reference,pool,pairs,probabilities,truth,tune)
        report={'fit_country':country,'source_tune':score_pairs(reference,pool,pairs,probabilities,truth,threshold,tune),
                'transfer_tune':score_pairs(reference,pool,pairs,probabilities,truth,threshold,other),
                'elapsed_seconds':time.time()-started}
        reports.append(report)
        log(json.dumps(report))
        joblib.dump({'model':model,'threshold':threshold,'feature_columns':list(X.columns),
                     'numeric_features':True,'record':report},output/f'{country}.joblib')
        json_write(output/'results.json',{'results':reports,'eval_used':False,
            'limitation':'Country-transfer development stress only; France has no training labels.'})


if __name__=='__main__':
    main()
