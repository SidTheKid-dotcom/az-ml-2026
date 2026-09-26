"""Score rival negatives for an already sealed France-rule selection.

Reads selected_pair_rows.npy written before labels were opened. Does not
change the selection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
OUT = Path(os.environ.get("AUDIT_OUT", "/opt/az-ml-new/runs/france_transfer_audit"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))


def main() -> None:
    summary = json.loads((ART / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("refusing HOLDOUT-2 labels")
    chosen = np.load(OUT / "selected_pair_rows.npy")
    left = np.load(INDEX / "left.npy", mmap_mode="r")
    right = np.load(INDEX / "right.npy", mmap_mode="r")
    label = np.load(ART / "labels" / "label.npy", mmap_mode="r")
    n_rec = int(right.max()) + 1
    owner = np.full(n_rec, -1, np.int32)
    owner_row = np.full(n_rec, -1, np.int64)
    sel_right = np.asarray(right[chosen])
    sel_left = np.asarray(left[chosen])
    if len(np.unique(sel_right)) != len(chosen):
        raise SystemExit("selection has two owners for one record")
    owner[sel_right] = sel_left
    owner_row[sel_right] = chosen
    rival = (owner[right] >= 0) & (np.asarray(left) != owner[right])
    rival_rows = np.flatnonzero(rival)
    rival_label = np.asarray(label[rival_rows], dtype=bool)
    # A true rival means the sealed owner is not the true owner of that record.
    report = {
        "selected_pairs": int(len(chosen)),
        "selected_precision": float(np.asarray(label[chosen], dtype=bool).mean()),
        "rival_pairs": int(len(rival_rows)),
        "rival_true_links": int(rival_label.sum()),
        "rival_negative_precision": float((~rival_label).mean()) if len(rival_label) else None,
        "records_whose_true_owner_was_rejected": int(rival_label.sum()),
        "holdout2_rows_read": 0,
        "slice_scored": False,
    }
    (OUT / "rivals.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
