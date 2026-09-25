"""Compare fixed matcher scores with a reference-level abstention rule on tune."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from .experiment import load_data, json_write
from .metrics import pair_labels, score_pairs, tune_threshold


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--base-dir',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    base,output=Path(args.base_dir),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    reference,pool,truth=load_data(args.data)
    pairs=pd.read_parquet(base/'pairs.parquet')
    probabilities=np.load(base/'best_probabilities.npy')
    if len(probabilities)!=len(pairs):
        raise ValueError('Candidate/probability row mismatch')
    left=pairs.left_idx.to_numpy(dtype=np.int64)
    if not reference.iloc[np.unique(left)].split.isin(['train','tune']).all():
        raise ValueError('This sweep accepts train/tune candidates only')
    maxima=np.full(len(reference),-1.,dtype=np.float32)
    np.maximum.at(maxima,left,probabilities)
    labels=pair_labels(reference,pool,pairs,truth)
    mask=reference.split.eq('tune').to_numpy()
    reports=[]
    winner=None
    for anchor in [0,.25,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99]:
        effective=np.where(maxima[left]>=anchor,probabilities,-1.)
        threshold,trace=tune_threshold(reference,pool,pairs,effective,truth,mask)
        score=score_pairs(reference,pool,pairs,effective,truth,threshold,mask)
        score['false_positive_links']=int(((effective>=threshold)&(labels==0)&mask[left]).sum())
        report={'anchor_threshold':anchor,'pair_threshold':threshold,'tune':score}
        reports.append(report)
        # Prefer the simpler policy on exact score ties.
        if winner is None or score['macro_f05']>winner['tune']['macro_f05']:
            winner=report
            bundle=joblib.load(base/'best.joblib')
            bundle.update(threshold=threshold,anchor_threshold=anchor,decision_record=report)
            joblib.dump(bundle,output/'best.joblib')
            np.save(output/'best_effective_probabilities.npy',effective)
            json_write(output/'best_thresholds.json',trace)
    result={'results':reports,'selected_by_tune':winner,'eval_used':False,
            'definition':'Return no matches for a reference whose maximum pair score is below the anchor threshold.'}
    json_write(output/'results.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
