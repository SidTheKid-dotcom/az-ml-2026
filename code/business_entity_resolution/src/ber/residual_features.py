"""Compare the name tokens left after removing words shared by both records.

This separates a one-letter change in a three-letter business identifier from
a one-letter typo in a long, generic word such as 'technologies'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from .features import _core_name, _phonetic
from .normalization import normalize_text

NAMES = ['residual_left_log_tokens','residual_right_log_tokens','residual_left_log_chars',
    'residual_right_log_chars','residual_ratio','residual_token_sort','residual_phonetic_ratio',
    'residual_jaro','residual_initials_equal','residual_both_present','residual_left_only',
    'residual_right_only','residual_shorter_max_word_log','residual_left_shared_char_fraction',
    'residual_right_shared_char_fraction','residual_best_word_similarity']


def build_residual_features(reference,pool,pairs):
    lu,lm=np.unique(pairs.left_idx.to_numpy(dtype=np.int64),return_inverse=True)
    ru,rm=np.unique(pairs.right_idx.to_numpy(dtype=np.int64),return_inverse=True)
    def parse(frame,positions):
        return [set(_core_name(normalize_text(value)).split())
                for value in frame.business_name.iloc[positions].fillna('')]
    lefts,rights=parse(reference,lu),parse(pool,ru)
    output=np.zeros((len(pairs),len(NAMES)),dtype=np.float32)
    for i,(l,r) in enumerate(zip(lm,rm)):
        left,right=lefts[l],rights[r]
        shared=left & right
        lw,rw=sorted(left-shared),sorted(right-shared)
        ls,rs=' '.join(lw),' '.join(rw)
        both=bool(ls and rs)
        exact=bool(left and right and not ls and not rs)
        lp,rp=_phonetic(ls),_phonetic(rs)
        output[i]=(
            np.log1p(len(lw)),np.log1p(len(rw)),np.log1p(len(ls)),np.log1p(len(rs)),
            fuzz.ratio(ls,rs)/100 if both else float(exact),
            fuzz.token_sort_ratio(ls,rs)/100 if both else float(exact),
            fuzz.ratio(lp,rp)/100 if lp and rp else float(exact),
            JaroWinkler.normalized_similarity(ls,rs) if both else float(exact),
            both and ''.join(w[0] for w in lw)==''.join(w[0] for w in rw),
            both,bool(ls and not rs),bool(rs and not ls),
            np.log1p(min(max(map(len,lw),default=0),max(map(len,rw),default=0))),
            sum(map(len,shared))/max(sum(map(len,left)),1),
            sum(map(len,shared))/max(sum(map(len,right)),1),
            max((fuzz.ratio(a,b)/100 for a in lw for b in rw),default=float(exact)),
        )
    return pd.DataFrame(output,index=pairs.index,columns=NAMES)
