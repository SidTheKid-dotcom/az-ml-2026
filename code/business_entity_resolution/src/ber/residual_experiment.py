"""Add distinctive-name evidence to the selected full-pool matcher."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .experiment import load_data,json_write
from .residual_features import build_residual_features
from .metrics import pair_labels,score_pairs,tune_threshold
from .models import fit_model,predict_proba
from .prepare import log


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--base-dir',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    base,output=Path(args.base_dir),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    reference,pool,truth=load_data(args.data)
    pairs=pd.read_parquet(base/'pairs.parquet')
    if not reference.split.isin(['train','tune']).all():
        raise ValueError('This experiment accepts train/tune only')
    bundle=joblib.load(base/'best.joblib')
    original=pd.read_parquet(base/'base_features.parquet')
    numeric=pd.read_parquet(base/'numeric_features.parquet')
    residual=build_residual_features(reference,pool,pairs)
    features=pd.concat([original,numeric],axis=1)[bundle['feature_columns']]
    features=pd.concat([features,residual],axis=1)
    pd.concat([original,residual],axis=1).to_parquet(output/'base_features.parquet',index=False)
    numeric.to_parquet(output/'numeric_features.parquet',index=False)
    y=pair_labels(reference,pool,pairs,truth)
    split=reference.split.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)]
    train,tune=np.flatnonzero(split=='train'),np.flatnonzero(split=='tune')
    reports=[];winner=None
    for weight in [1.,2.,4.]:
        log(f'Name residual matcher: negative_weight={weight}')
        model=fit_model(features.iloc[train],y[train],kind='catboost',device='GPU',
            seed=20260925,iterations=650,depth=8,class_weights=[weight,1.],
            thread_count=2,gpu_ram_part=.22,eval_set=(features.iloc[tune],y[tune]),verbose=False)
        probabilities=predict_proba(model,features)
        threshold,trace=tune_threshold(reference,pool,pairs,probabilities,truth,reference.split.eq('tune'))
        score=score_pairs(reference,pool,pairs,probabilities,truth,threshold,reference.split.eq('tune'))
        score['false_positive_links']=int(((probabilities>=threshold)&(y==0)&(split=='tune')).sum())
        record={'model':f'residual_d8_neg{weight:g}','tune':score,'negative_weight':weight}
        reports.append(record);log(json.dumps(record))
        if winner is None or score['macro_f05']>winner['tune']['macro_f05']:
            winner=record
            joblib.dump({'model':model,'threshold':threshold,'feature_columns':list(features),
                'numeric_features':bundle.get('numeric_features',True),'record':record},output/'best.joblib')
            np.save(output/'best_probabilities.npy',probabilities)
            json_write(output/'best_thresholds.json',trace)
        json_write(output/'results.json',{'results':reports,'selected_by_tune':winner,
            'baseline':bundle['record'],'eval_used':False})
    pairs.to_parquet(output/'pairs.parquet',index=False)


if __name__=='__main__':
    main()
