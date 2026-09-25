"""P3.2/P3.3: full-pool TRAIN-only BGE reranker with a bounded noise ablation.

The P2 handoff is an *atomic* ready JSON, written only after dense mining and
the TRAIN truth join finish. It has state=complete, split=train,
dense_mining_complete=true, train_truth_join_complete=true,
positive_candidate_injection=false, full_pool_records=10320219,
train_reference_id_sha256, and files {reference, pool, candidate_pairs,
train_truth_pairs}. Each file entry has path, sha256 and rows. Paths may be
relative to the ready JSON. The TRAIN-only reference/pool files contain the
usual entity_id, business_name, business_address, country columns; reference
also has split=train. train_truth_pairs.parquet contains EVERY real TRAIN link
as left_idx/right_idx, including links missed by retrieval. The completed,
unmodified candidate_pairs.parquet has left_idx/right_idx, label or
training_label (0/1), dense_rank (0 means no dense hit), and either the
same_name/same_digits flags or P2's hard_negative_priority (0 is a near twin).
The truth is used to verify this join and as a training pool, NEVER to insert
an edge in retrieval or in TUNE. All indices refer to those two record files.

TUNE is a separate completed full-target-pool data bundle. The existing
full_dense_medium layout and the P2 12k medium union plus --tune-pairs context
file are supported. --tune-base-dir selects an aligned completed tabular tree;
otherwise --tune-scores/--tune-score-pairs/--tune-tree-checkpoint provide the
aligned P3.1 tree. Full TRAIN requires independent P3.1 tree scores and an
identical pair-order file; P2's ready marker stays immutable. The real launch
requires the sealed HOLDOUT-2
reservation, exactly 12,000 saved TUNE S1s, and 10,320,219 competing targets.
--smoke relaxes only these scale/count gates and marks every result as synthetic.

Examples (use a separate durable --checkpoint-root for a full run):
  python -m ber.v2_ce_train --train-ready P2/ready.json \
    --sealed-reservation P2/holdout2 --train-tree-checkpoint P3/tree/tree_bundle.joblib \
    --output /tmp/az-ml-v2/ce_screen/tree_scores --device cpu --amp fp32 --materialize-train-scores
  python -m ber.v2_ce_train --train-ready P2/ready.json \
    --sealed-reservation P2/holdout2 --tune-data P2/tune \
    --tune-pairs P2/tune/context.parquet \
    --tune-scores P3/tree/tune_probabilities.npy --tune-score-pairs P3/tree/tune_pairs.parquet \
    --tune-tree-checkpoint P3/tree/tree_bundle.joblib \
    --train-scores /tmp/az-ml-v2/ce_screen/tree_scores/train_probabilities.npy \
    --train-score-pairs P2/candidate_pairs.parquet \
    --train-tree-checkpoint P3/tree/tree_bundle.joblib \
    --checkpoint-root /workspace/az-ml-v2/ce --output /tmp/az-ml-v2/ce_screen --prepare-only
  python -m ber.v2_ce_train [same inputs] --max-steps 10000 --microbatch 32 --launch-full
  python -m ber.v2_ce_train [same inputs] --max-steps 20000 --microbatch 32 --resume --launch-full

Neither preparation nor training reads HOLDOUT-2 match labels. No full run is
permitted without the P2 completion marker, verified identities and ID seal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .experiment import json_write
from .hf_experiment import record_texts
from .metrics import pair_labels
from .uncertainty import paired_bootstrap


MODEL_ID = "BAAI/bge-reranker-v2-m3"
FULL_POOL_ROWS = 10_320_219
TUNE_IDS_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
PAIR_DTYPE = np.dtype([("left", "<u4"), ("right", "<u4"), ("channels", "u1")])
CHANNELS = {"tree": 1, "dense": 2, "name_digits": 4}
RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
_DIGITS = re.compile(r"\d+")
_LATIN_ACCENTS = {"a": "à", "e": "é", "i": "î", "o": "ô", "u": "ù", "c": "ç"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def id_sha256(ids) -> str:
    """The sealed reservation's newline-joined sorted-ID convention, no final LF."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _file_entry(ready: Path, item: dict, name: str) -> Path:
    if not all(key in item for key in ("path", "sha256", "rows")):
        raise ValueError(f"P2 {name} needs an exact path, SHA256 and row count")
    path = Path(item["path"])
    path = path if path.is_absolute() else ready.parent / path
    if not path.is_file() or file_sha256(path) != item["sha256"]:
        raise ValueError(f"P2 {name} is absent or its bytes changed: {path}")
    if pq.ParquetFile(path).metadata.num_rows != item["rows"]:
        raise ValueError(f"P2 {name} row count differs from its complete marker")
    return path


def validate_ready(path: Path, *, full_pool_rows: int = FULL_POOL_ROWS) -> tuple[dict, dict[str, Path]]:
    """Do not accept an in-progress miner, partial truth join or sampled target pool."""
    path = Path(path)
    ready = json.loads(path.read_text(encoding="utf-8"))
    required = {"state": "complete", "split": "train", "dense_mining_complete": True,
                "train_truth_join_complete": True, "positive_candidate_injection": False,
                "full_pool_records": full_pool_rows}
    for key, expected in required.items():
        if type(ready.get(key)) is not type(expected) or ready[key] != expected:
            raise ValueError(f"P2 ready marker must attest {key}={expected!r}")
    if not isinstance(ready.get("train_reference_id_sha256"), str):
        raise ValueError("P2 ready marker needs the TRAIN ID digest")
    if ready.get("truth_used_in_retrieval", False) is not False:
        raise ValueError("P2 retrieval must have fixed candidate edges before the TRAIN truth join")
    files = ready.get("files", {})
    names = ("reference", "pool", "candidate_pairs", "train_truth_pairs")
    if not all(name in files for name in names):
        raise ValueError("P2 TRAIN references, targets, candidate join and complete true links are required")
    return ready, {name: _file_entry(path, files[name], name) for name in names}


def sealed_ids(path: Path, *, smoke: bool = False) -> tuple[set[str], str]:
    """Read only ID-only reservation files; never open any holdout truth."""
    path = Path(path)
    report = json.loads((path / "reservation.json").read_text(encoding="utf-8"))
    if report.get("verification", {}).get("labels_read") is not False:
        raise ValueError("HOLDOUT-2 reservation must attest that no labels were read")
    rule, held = report["future_raw_train_tune"], report["holdout2"]
    if not smoke and held.get("count") != 50_000:
        raise ValueError("Expected the fixed 50,000-S1 HOLDOUT-2 reservation")
    forbidden_path = path / rule["excluded_id_file"]
    holdout_path = path / held["id_file"]
    raw = forbidden_path.read_bytes()
    reserved_raw = holdout_path.read_bytes()
    forbidden = raw.decode("utf-8").splitlines()
    reserved = reserved_raw.decode("utf-8").splitlines()
    if (not raw.endswith(b"\n") or not reserved_raw.endswith(b"\n")
            or forbidden != sorted(set(forbidden)) or reserved != sorted(set(reserved))
            or any(not entity.startswith("S1-") for entity in forbidden)
            or any(not entity.startswith("S1-") for entity in reserved)
            or len(forbidden) != rule["excluded_count"] or len(reserved) != held["count"]
            or hashlib.sha256(raw).hexdigest() != rule["excluded_file_sha256"]
            or hashlib.sha256(reserved_raw).hexdigest() != held["id_file_sha256"]
            or id_sha256(forbidden) != rule["excluded_sorted_id_sha256"]
            or id_sha256(reserved) != held["sorted_id_sha256"]
            or not set(reserved) <= set(forbidden)):
        raise ValueError("HOLDOUT-2 exclusion IDs or reservation digests differ")
    return set(forbidden), file_sha256(path / "reservation.json")


def _records(path: Path, *, reference: bool) -> pd.DataFrame:
    columns = (*RECORD_COLUMNS, "split") if reference else RECORD_COLUMNS
    frame = pd.read_parquet(path, columns=list(columns))
    if (frame[list(RECORD_COLUMNS)].isna().any().any() or frame.entity_id.eq("").any()
            or frame.entity_id.duplicated().any()):
        raise ValueError(f"Null records or duplicate entity IDs: {path}")
    if reference and (frame.split.isna().any() or not frame.split.eq("train").all()):
        raise ValueError("P2 TRAIN records must contain only TRAIN S1s")
    return frame


def _pair_key(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)


def _indices(frame: pd.DataFrame, nleft: int, nright: int, what: str) -> tuple[np.ndarray, np.ndarray]:
    left = frame.left_idx.to_numpy()
    right = frame.right_idx.to_numpy()
    if (len(left) == 0 or not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(right.dtype, np.integer)
            or left.min() < 0 or right.min() < 0 or left.max() >= nleft or right.max() >= nright
            or nleft >= 2**32 or nright >= 2**32):
        raise ValueError(f"{what} indices must be nonempty, valid uint32 positions in the supplied records")
    return left.astype(np.uint32), right.astype(np.uint32)


def _checked_tune_references(directory: Path, forbidden: set[str], *, smoke: bool,
                             full_pool_rows: int, pairs_override: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Verify full-pool retrieval and fixed TUNE *before* ever loading tune truth."""
    directory = Path(directory)
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if complete.get("state") != "complete":
        raise ValueError("TUNE retrieval is not committed")
    legacy = (directory / "metadata.json").is_file()
    pair_path = Path(pairs_override) if pairs_override else directory / "pairs.parquet"
    if legacy:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if complete.get("metadata") != metadata or pair_path.resolve() != (directory / "pairs.parquet").resolve():
            raise ValueError("Legacy TUNE pairs/metadata do not match the completed miner")
        total = metadata.get("full_pool_records", metadata.get("mining_full_pool_records"))
        if complete.get("config", {}).get("labels_used_for_mining", False) is not False:
            raise ValueError("TUNE mining used labels")
        count = metadata.get("candidate_pairs")
        kind = "completed_legacy_dense_or_union"
    else:
        # P2's fixed 12k medium-TUNE union has no legacy metadata.json. Its
        # context-scored pair file and original/raw union are separate, hashed
        # artifacts; all three identities must match before reading TUNE truth.
        if pairs_override is None:
            raise ValueError("P2 TUNE union requires --tune-pairs for its completed score-context Parquet")
        total = complete.get("full_target_pool_rows")
        if (complete.get("truth_used_in_retrieval") is not False or
                complete.get("positive_candidate_injection") is not False or
                not (directory / "candidate_pairs.parquet").is_file()):
            raise ValueError("P2 TUNE union is not a label-free completed retrieval artifact")
        original = directory / "candidate_pairs.parquet"
        raw = directory / "candidate_pairs_raw.parquet"
        raw_entry = complete.get("raw_pairs", {})
        if (not raw.is_file() or file_sha256(raw) != raw_entry.get("sha256")
                or pq.ParquetFile(original).metadata.num_rows != raw_entry.get("rows")):
            raise ValueError("P2 TUNE original and raw candidate sets differ")
        context_path = pair_path.with_suffix(".context.json")
        context = json.loads(context_path.read_text(encoding="utf-8"))
        if (context.get("state") != "complete" or context.get("positive_candidate_injection") is not False
                or context.get("inputs", {}).get("pairs", {}).get("sha256") != file_sha256(original)
                or context.get("pair_file", {}).get("sha256") != file_sha256(pair_path)
                or pq.ParquetFile(pair_path).metadata.num_rows != raw_entry.get("rows")):
            raise ValueError("P2 TUNE score-context pairs differ from the completed full-pool union")
        count = raw_entry["rows"]
        kind = "completed_p2_tune_union_with_scored_context"
    injected = metadata.get("positive_candidate_injection") if legacy else complete.get("positive_candidate_injection")
    if type(total) is not int or total != full_pool_rows or injected is not False:
        raise ValueError("TUNE must use label-free retrieval against the complete competing target pool")
    reference = pd.read_parquet(directory / "reference.parquet").fillna("")
    if (reference.entity_id.eq("").any() or reference.entity_id.duplicated().any()
            or set(reference.split) - {"train", "tune"}):
        raise ValueError("TUNE bundle must contain unique TRAIN/TUNE S1s only")
    reference_sha = id_sha256(reference.entity_id.tolist())
    if ((legacy and metadata.get("reference_id_sha256") != reference_sha)
            or (not legacy and complete.get("reference_ids_sha256") != reference_sha)):
        raise ValueError("TUNE reference population differs from completed mining")
    tune = reference.loc[reference.split.eq("tune")]
    expected = TUNE_IDS_SHA256 if not smoke else None
    if not len(tune) or (not smoke and (len(tune) != 12_000 or id_sha256(tune.entity_id.tolist()) != expected)):
        raise ValueError("Expected the fixed 12,000 entity-disjoint full-pool TUNE S1s")
    if not set(tune.entity_id) <= forbidden:
        raise ValueError("TUNE IDs are not covered by the sealed TRAIN exclusion")
    counts = {f"{split}/{country}": int(n) for (split, country), n
              in reference.groupby(["split", "country"]).size().items()}
    if (legacy and counts != metadata.get("split_counts")) or (not legacy and complete.get("references") != len(reference)):
        raise ValueError("TUNE country/split counts changed since mining")
    return reference, {"tune_id_sha256": id_sha256(tune.entity_id.tolist()),
                       "tune_references": len(tune), "full_pool_records": total,
                       "reference_id_sha256": reference_sha, "candidate_pairs": count,
                       "dataset_kind": kind, "pair_source": str(pair_path.resolve()),
                       "pair_file_sha256": file_sha256(pair_path),
                       "complete_file_sha256": file_sha256(directory / "complete.json")}


def _pair_order_sha256(path: Path) -> str:
    """Hash every positional edge in order, independent of Parquet row groups."""
    digest = hashlib.sha256()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=131_072, columns=["left_idx", "right_idx"]):
        left, right = (batch.column(i).to_numpy() for i in (0, 1))
        if (not len(left) or not np.issubdtype(left.dtype, np.integer)
                or not np.issubdtype(right.dtype, np.integer) or left.min() < 0 or right.min() < 0
                or left.max() >= 2**32 or right.max() >= 2**32):
            raise ValueError(f"Pair-order file has invalid positional indices: {path}")
        digest.update(_pair_key(left, right).tobytes())
    return digest.hexdigest()


def _full_tree_identity(checkpoint: Path, *, train_ready: Path | None = None,
                        tune_pair_sha: str | None = None, score_pairs: Path | None = None,
                        scores: Path | None = None) -> dict:
    """Require the selected P3.1 checkpoint and its completed scored pair order."""
    checkpoint = Path(checkpoint)
    identity = {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)}
    if checkpoint.name != "tree_bundle.joblib":
        return identity  # Explicit external score/edge hashes still bind a non-P3.1 tree.
    directory = checkpoint.parent
    complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "experiment_config.json").read_text(encoding="utf-8"))
    results = directory / "results.json"
    if (complete.get("state") != "complete" or complete.get("tree_checkpoint_sha256") != identity["sha256"]
            or complete.get("results_sha256") != file_sha256(results)
            or config.get("mode") != "full"):
        raise ValueError("P3.1 selected full TRAIN tree is incomplete or its checkpoint changed")
    if train_ready is not None and (config.get("input_train", {}).get("ready", {}).get("sha256")
                                   != file_sha256(train_ready)):
        raise ValueError("P3.1 TRAIN tree was fitted on another P2 ready marker")
    if tune_pair_sha is not None and (config.get("input_tune", {}).get("pairs", {}).get("sha256")
                                      != tune_pair_sha):
        raise ValueError("P3.1 TUNE tree did not use this completed P2 union/context pair file")
    if score_pairs is not None:
        evidence = json.loads(results.read_text(encoding="utf-8"))
        pair_sha = file_sha256(score_pairs)
        probability_sha = file_sha256(scores)
        if (pair_sha != file_sha256(directory / "tune_pairs.parquet")
                or probability_sha != file_sha256(directory / "tune_probabilities.npy")
                or evidence.get("tune_preparation", {}).get("pairs", {}).get("sha256") != pair_sha
                or evidence.get("evaluation", {}).get("tune_pair_probabilities", {}).get("sha256") != probability_sha):
            raise ValueError("P3.1 saved TUNE probabilities or pair-order file changed after scoring")
    identity["complete_sha256"] = file_sha256(directory / "complete.json")
    return identity


def _train_score_source(args, ready: dict, files: dict[str, Path]) -> tuple[np.ndarray | None, dict]:
    paths = [getattr(args, name, None) for name in ("train_scores", "train_score_pairs", "train_tree_checkpoint")]
    if not any(paths):
        if not args.smoke:
            raise ValueError("Full CE needs independent P3.1 TRAIN tree scores, pair-order file and selected checkpoint")
        return None, {"kind": "embedded_synthetic_smoke_probabilities"}
    if not all(paths):
        raise ValueError("TRAIN tree scores need --train-scores, --train-score-pairs and --train-tree-checkpoint together")
    score_path, pair_path, tree_path = map(Path, paths)
    if not args.smoke and tree_path.name != "tree_bundle.joblib":
        raise ValueError("Full CE requires the completed P3.1 selected TRAIN tree checkpoint")
    count = ready["files"]["candidate_pairs"]["rows"]
    scores = np.load(score_path, mmap_mode="r")
    if (scores.shape != (count,) or not np.issubdtype(scores.dtype, np.floating)
            or pq.ParquetFile(pair_path).metadata.num_rows != count
            or not tree_path.is_file()):
        raise ValueError("P3.1 TRAIN score vector/pair order must cover every completed P2 candidate")
    candidate_order = _pair_order_sha256(files["candidate_pairs"])
    score_order = (candidate_order if pair_path.resolve() == files["candidate_pairs"].resolve()
                   else _pair_order_sha256(pair_path))
    if candidate_order != score_order:
        raise ValueError("TRAIN tree probability order differs from the complete P2 candidate join")
    checkpoint = _full_tree_identity(tree_path, train_ready=Path(args.train_ready) if not args.smoke else None)
    materialization_sha = None
    if not args.smoke:
        marker_path = score_path.parent / "complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        stored = marker.get("identity", {})
        if (marker.get("state") != "complete" or marker.get("pairs") != count
                or marker.get("scores_sha256") != file_sha256(score_path)
                or marker.get("pair_order_sha256") != candidate_order
                or Path(marker.get("score_pairs_path", "")).resolve() != files["candidate_pairs"].resolve()
                or stored.get("p2_ready_sha256") != file_sha256(Path(args.train_ready))
                or stored.get("tree_checkpoint", {}).get("sha256") != checkpoint["sha256"]):
            raise ValueError("Full CE requires the completed, aligned P3.1 TRAIN tree materialization")
        materialization_sha = file_sha256(marker_path)
    return scores, {"kind": "separate_aligned_train_tree_scores", "scores_path": str(score_path.resolve()),
                    "scores_sha256": file_sha256(score_path), "score_pairs_path": str(pair_path.resolve()),
                    "score_pairs_sha256": file_sha256(pair_path), "pair_order_sha256": candidate_order,
                    "tree_checkpoint": checkpoint, "materialization_complete_sha256": materialization_sha}


def _original_pair_batches(path: Path, nreference: int, npool: int, *, reference_batch: int = 1_000,
                           read_batch: int = 100_000, max_pairs: int = 200_000):
    """Keep P2's exact pair order while batching complete S1 candidate groups."""
    if min(reference_batch, read_batch, max_pairs) < 1:
        raise ValueError("Pair reader and complete-reference batch limits must be positive")
    parquet = pq.ParquetFile(path)
    stream = iter(parquet.iter_batches(batch_size=read_batch, use_threads=False))
    previous_left = -1

    def next_frame():
        nonlocal previous_left
        for batch in stream:
            frame = batch.to_pandas()
            if frame.empty:
                continue
            left, right = _indices(frame, nreference, npool, "P2 candidate stream")
            if left[0] < previous_left or np.any(left[1:] < left[:-1]):
                raise ValueError("P2 candidate file must be globally ordered by S1")
            previous_left = int(left[-1])
            return frame
        return None

    current, offset = next_frame(), 0
    consumed = 0
    for start in range(0, nreference, reference_batch):
        end = min(start + reference_batch, nreference)
        pieces, count = [], 0
        while current is not None:
            stop = int(np.searchsorted(current.left_idx.to_numpy(dtype=np.int64), end, side="left"))
            if stop > offset:
                pieces.append(current.iloc[offset:stop])
                count += stop - offset
                if count > max_pairs:
                    raise ValueError("Complete S1 group batch exceeded the explicit pair bound")
            offset = stop
            if offset == len(current):
                current, offset = next_frame(), 0
            else:
                break
        if not pieces:
            continue  # The S1 interval is still accounted for; it has zero candidates.
        pairs = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0].reset_index(drop=True)
        if pairs.duplicated(["left_idx", "right_idx"]).any():
            raise ValueError("Duplicate P2 candidate edge in a complete reference group")
        consumed += len(pairs)
        yield start, end, pairs
    if current is not None or consumed != parquet.metadata.num_rows:
        raise ValueError("P2 pair stream ended early or has unconsumed candidate edges")


def _score_tree_batch(model, columns: list[str], reference: pd.DataFrame, pool: pd.DataFrame,
                      pairs: pd.DataFrame, start: int, end: int, threads: int) -> np.ndarray:
    """Use P3.1's exact sorted-group features, then restore original P2 row order."""
    from .predict import feature_batch

    if (pairs.empty or pairs.left_idx.min() < start or pairs.left_idx.max() >= end
            or not np.array_equal(reference.country.to_numpy()[pairs.left_idx.to_numpy(dtype=np.int64)],
                                  pool.country.to_numpy()[pairs.right_idx.to_numpy(dtype=np.int64)])):
        raise ValueError("Tree scoring requires complete, country-blocked S1 candidate groups")
    order = np.lexsort((pairs.right_idx.to_numpy(dtype=np.int64), pairs.left_idx.to_numpy(dtype=np.int64)))
    sorted_pairs = pairs.iloc[order].reset_index(drop=True)
    rights, inverse = np.unique(sorted_pairs.right_idx.to_numpy(dtype=np.int64), return_inverse=True)
    local = sorted_pairs.copy()
    local["left_idx"] = local.left_idx.to_numpy(dtype=np.int64) - start
    local["right_idx"] = inverse
    X = feature_batch(reference.iloc[start:end][list(RECORD_COLUMNS)].reset_index(drop=True),
                      pool.iloc[rights][list(RECORD_COLUMNS)].reset_index(drop=True), local, columns)
    if list(X.columns) != columns or X.shape != (len(pairs), len(columns)) or not np.isfinite(X.to_numpy()).all():
        raise ValueError("P3.1 ordered candidate features changed or include nonfinite values")
    probabilities = np.asarray(model.predict_proba(X, thread_count=threads))
    if probabilities.shape != (len(pairs), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Selected tree did not score every TRAIN candidate in the same feature order")
    values = probabilities[:, 1].astype(np.float32)[np.argsort(order)]
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Selected tree emitted invalid TRAIN probabilities")
    return values


def materialize_train_tree_scores(args) -> dict:
    """CPU-only P3.1 inference on ALL P2 TRAIN edges, with no truth or TUNE read."""
    if args.smoke or not args.train_tree_checkpoint or Path(args.train_tree_checkpoint).name != "tree_bundle.joblib":
        raise ValueError("Full TRAIN score materialization needs the completed P3.1 selected tree")
    import joblib

    started = time.monotonic()
    ready, files = validate_ready(args.train_ready)
    forbidden, sealed_sha = sealed_ids(args.sealed_reservation)
    checkpoint = _full_tree_identity(Path(args.train_tree_checkpoint), train_ready=Path(args.train_ready))
    reference = _records(files["reference"], reference=True)
    pool = _records(files["pool"], reference=False)
    if (id_sha256(reference.entity_id.tolist()) != ready["train_reference_id_sha256"]
            or set(reference.entity_id) & forbidden):
        raise ValueError("P2 TRAIN score input includes a sealed/non-TRAIN S1")
    from .v2_tree_train import feature_columns

    parquet = pq.ParquetFile(files["candidate_pairs"])
    columns = feature_columns(parquet.schema_arrow.names)
    bundle = joblib.load(args.train_tree_checkpoint)
    model = bundle["model"]
    if list(bundle["feature_columns"]) != columns or list(model.feature_names_) != columns:
        raise ValueError("P3.1 tree feature order differs from the full P2 candidate schema")
    output = Path(args.output)
    marker_path = output / "complete.json"
    identity = {"p2_ready_sha256": file_sha256(Path(args.train_ready)),
                "candidate_pairs_sha256": ready["files"]["candidate_pairs"]["sha256"],
                "sealed_reservation_sha256": sealed_sha, "tree_checkpoint": checkpoint,
                "ordered_features": columns, "source_sha256": file_sha256(Path(__file__)),
                "reference_batch": args.score_reference_batch, "pair_read_batch": args.score_pair_read_batch,
                "max_pairs_per_batch": args.score_max_pairs_per_batch, "threads": args.threads}
    score_path = output / "train_probabilities.npy"
    if marker_path.exists():
        old = json.loads(marker_path.read_text(encoding="utf-8"))
        if (old.get("state") != "complete" or old.get("identity") != identity
                or old.get("scores_sha256") != file_sha256(score_path)):
            raise ValueError("Previous TRAIN tree scores are incomplete or belong to different sources")
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError("Uncommitted TRAIN tree scoring output needs inspection")
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "train_probabilities.npy.tmp"
    scores = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                       shape=(parquet.metadata.num_rows,))
    digest = hashlib.sha256()
    offset = 0
    try:
        for start, end, pairs in _original_pair_batches(files["candidate_pairs"], len(reference), len(pool),
                                                         reference_batch=args.score_reference_batch,
                                                         read_batch=args.score_pair_read_batch,
                                                         max_pairs=args.score_max_pairs_per_batch):
            values = _score_tree_batch(model, columns, reference, pool, pairs, start, end, args.threads)
            scores[offset:offset + len(pairs)] = values
            digest.update(_pair_key(pairs.left_idx.to_numpy(dtype=np.uint32),
                                    pairs.right_idx.to_numpy(dtype=np.uint32)).tobytes())
            offset += len(pairs)
            if offset % 1_000_000 < len(pairs):
                print(json.dumps({"scored_train_candidates": offset, "total": len(scores)}), flush=True)
        if offset != len(scores):
            raise ValueError("Selected tree failed to score every completed P2 TRAIN candidate")
        scores.flush()
        del scores
        temporary.replace(score_path)
        report = {"state": "complete", "scope": "TRAIN tree gate only; no HOLDOUT-2 or TUNE labels read",
                  "identity": identity, "pairs": offset, "pair_order_sha256": digest.hexdigest(),
                  "score_pairs_path": str(files["candidate_pairs"].resolve()),
                  "scores_path": str(score_path.resolve()), "scores_sha256": file_sha256(score_path),
                  "seconds": time.monotonic() - started}
        json_write(marker_path, report)
        return report
    finally:
        if temporary.exists() and "scores" in locals():
            scores.flush()  # An interrupted .tmp stays for inspection; it is never a completion marker.


def prepare(args) -> dict:
    """Stream the complete P2 join once; retain every TRAIN true link, never inject TUNE edges."""
    pool_rows = FULL_POOL_ROWS if not args.smoke else args.smoke_full_pool_rows
    ready, files = validate_ready(args.train_ready, full_pool_rows=pool_rows)
    forbidden, sealed_sha = sealed_ids(args.sealed_reservation, smoke=args.smoke)
    tune_ref, tune_meta = _checked_tune_references(args.tune_data, forbidden, smoke=args.smoke,
                                                   full_pool_rows=pool_rows,
                                                   pairs_override=getattr(args, "tune_pairs", None))
    references = _records(files["reference"], reference=True)
    pool = _records(files["pool"], reference=False)
    train_ids = set(references.entity_id)
    if (len(references) != ready["files"]["reference"]["rows"]
            or len(pool) != ready["files"]["pool"]["rows"]
            or id_sha256(train_ids) != ready["train_reference_id_sha256"]
            or train_ids & forbidden or train_ids & set(tune_ref.entity_id)):
        raise ValueError("P2 TRAIN IDs/counts differ, overlap TUNE or include sealed S1s")
    if set(references.country) - {"India", "US"}:
        raise ValueError("TRAIN labels must originate from the supplied US/India data")
    if pool.entity_id.isin(references.entity_id).any():
        raise ValueError("Source 1 IDs cannot appear in the target pool")
    tree_scores, score_source = _train_score_source(args, ready, files)
    identity = {"schema": 1, "scope": "compatibility_smoke" if args.smoke else "full_pool_tune",
                "p2_ready_sha256": file_sha256(Path(args.train_ready)),
                "p2_input_files": {name: {**ready["files"][name], "resolved_path": str(file.resolve())}
                                   for name, file in files.items()},
                "train_reference_id_sha256": ready["train_reference_id_sha256"],
                "sealed_reservation_sha256": sealed_sha, "tune": tune_meta,
                "full_pool_records": pool_rows, "positive_candidate_injection": False,
                "tree_score_source": score_source,
                "hard_negative_selection": {"tree_gate": args.tree_gate, "dense_top_k": args.dense_top_k,
                                            "shares": {"tree": .5, "dense": .25, "name_digits": .25}}}
    output = Path(args.output)
    prepared_path = output / "prepared.json"
    if prepared_path.is_file():
        old = json.loads(prepared_path.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise ValueError("Output belongs to another TRAIN, TUNE, or sealed split")
        if file_sha256(output / "positive_pairs.npy") != old["positive_sha256"] or file_sha256(
                output / "negative_pairs.npy") != old["negative_sha256"]:
            raise ValueError("Prepared sampling pools have changed")
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError("Uncommitted or unrelated CE output; use a new --output")
    truth = pq.read_table(files["train_truth_pairs"], columns=["left_idx", "right_idx"]).to_pandas()
    pos_left, pos_right = _indices(truth, len(references), len(pool), "TRAIN true link")
    if not np.array_equal(references.country.to_numpy()[pos_left], pool.country.to_numpy()[pos_right]):
        raise ValueError("TRAIN positive links must keep the same country")
    truth_keys = np.sort(_pair_key(pos_left, pos_right))
    if len(truth_keys) > 1 and np.any(truth_keys[1:] == truth_keys[:-1]):
        raise ValueError("Duplicate TRAIN true links")
    candidates = pq.ParquetFile(files["candidate_pairs"])
    columns = candidates.schema_arrow.names
    label_column = "label" if "label" in columns else "training_label" if "training_label" in columns else None
    direct_twins = {"same_name", "same_digits"} <= set(columns)
    if (not {"left_idx", "right_idx", "dense_rank"} <= set(columns) or label_column is None
            or (not direct_twins and "hard_negative_priority" not in columns)
            or (tree_scores is None and "tree_probability" not in columns)):
        raise ValueError("P2 join needs indices, a truth-joined label, dense_rank, and a near-twin flag; "
                         "TRAIN tree probabilities must be explicit or independently aligned")
    read_columns = ["left_idx", "right_idx", label_column, "dense_rank"]
    read_columns += ["same_name", "same_digits"] if direct_twins else ["hard_negative_priority"]
    if tree_scores is None:
        read_columns.append("tree_probability")
    if candidates.metadata.num_rows < 1:
        raise ValueError("Empty P2 candidate join")
    output.mkdir(parents=True, exist_ok=True)
    keys_path, negative_tmp = output / "candidate_keys.tmp", output / "negative_pairs.tmp"
    keys = np.memmap(keys_path, dtype=np.uint64, mode="w+", shape=(candidates.metadata.num_rows,))
    negatives = np.memmap(negative_tmp, dtype=PAIR_DTYPE, mode="w+", shape=(candidates.metadata.num_rows,))
    stats = Counter()
    position = nnegative = 0
    try:
        for batch in candidates.iter_batches(batch_size=131_072, columns=read_columns):
            frame = batch.to_pandas()
            left, right = _indices(frame, len(references), len(pool), "TRAIN candidate")
            if not np.array_equal(references.country.to_numpy()[left], pool.country.to_numpy()[right]):
                raise ValueError("TRAIN candidate pairs cross country boundaries")
            key = _pair_key(left, right)
            labels = frame[label_column].to_numpy()
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("P2 TRAIN join labels must be binary")
            found = np.searchsorted(truth_keys, key)
            true = (found < len(truth_keys)) & (truth_keys[np.minimum(found, len(truth_keys) - 1)] == key)
            if not np.array_equal(true, labels.astype(bool)):
                raise ValueError("P2 candidate labels disagree with the complete TRAIN truth join")
            score = (frame.tree_probability.to_numpy(dtype=np.float64) if tree_scores is None else
                     np.asarray(tree_scores[position:position + len(key)], dtype=np.float64))
            dense = frame.dense_rank.to_numpy()
            if (not np.isfinite(score).all() or (score < 0).any() or (score > 1).any()
                    or not np.issubdtype(dense.dtype, np.integer) or (dense < 0).any()):
                raise ValueError("Tree probabilities or dense ranks are invalid")
            if direct_twins:
                for name in ("same_name", "same_digits"):
                    if frame[name].isna().any() or not frame[name].isin([True, False]).all():
                        raise ValueError(f"P2 hard-negative {name} must be a complete boolean column")
                near_twin = frame.same_name.to_numpy(dtype=bool) & frame.same_digits.to_numpy(dtype=bool)
            else:
                priority = frame.hard_negative_priority.to_numpy()
                if (not np.issubdtype(priority.dtype, np.integer) or (priority < -1).any()
                        or (priority > 9).any()):
                    raise ValueError("P2 near-twin priority must be a bounded integer")
                near_twin = priority == 0  # P2: exact core name AND address digit set.
            channels = ((score >= args.tree_gate).astype(np.uint8) * CHANNELS["tree"]
                        | ((dense > 0) & (dense <= args.dense_top_k)).astype(np.uint8) * CHANNELS["dense"]
                        | near_twin.astype(np.uint8) * CHANNELS["name_digits"])
            eligible = ~true & (channels != 0)
            count = int(eligible.sum())
            negatives["left"][nnegative:nnegative + count] = left[eligible]
            negatives["right"][nnegative:nnegative + count] = right[eligible]
            negatives["channels"][nnegative:nnegative + count] = channels[eligible]
            keys[position:position + len(key)] = key
            position += len(key)
            nnegative += count
            stats["retrieved_train_true_links"] += int(true.sum())
            stats["unmatched_candidates"] += int((~true).sum())
        if position != candidates.metadata.num_rows or not nnegative:
            raise ValueError("Incomplete P2 pair stream or no hard negatives")
        keys.flush()
        keys.sort()
        for start in range(1, position, 1_000_000):
            stop = min(start + 1_000_000, position)
            if np.any(keys[start:stop] == keys[start - 1:stop - 1]):
                raise ValueError("Duplicate retrieved candidate pairs")
        for name, flag in CHANNELS.items():
            stats[f"{name}_hard_negatives"] = int(np.count_nonzero(negatives["channels"][:nnegative] & flag))
            if not stats[f"{name}_hard_negatives"] and not args.smoke:
                raise ValueError(f"P2 TRAIN join has no {name} hard negatives")
        positive_file, negative_file = output / "positive_pairs.npy", output / "negative_pairs.npy"
        positives_array = np.column_stack((pos_left, pos_right))
        np.save(str(positive_file) + ".tmp", positives_array)
        Path(str(positive_file) + ".tmp.npy").replace(positive_file)
        compact = np.lib.format.open_memmap(str(negative_file) + ".tmp", mode="w+", dtype=PAIR_DTYPE,
                                            shape=(nnegative,))
        for start in range(0, nnegative, 1_000_000):
            stop = min(start + 1_000_000, nnegative)
            compact[start:stop] = negatives[start:stop]
        compact.flush()
        del compact
        Path(str(negative_file) + ".tmp").replace(negative_file)
        report = {"identity": identity, "positive_pool": len(truth_keys),
                  "hard_negative_pool": nnegative, "candidate_rows": position,
                  "missed_train_truth_links": len(truth_keys) - stats["retrieved_train_true_links"],
                  "counts": dict(stats), "positive_sha256": file_sha256(positive_file),
                  "negative_sha256": file_sha256(negative_file),
                  "train_truth_coverage": "all eligible TRAIN links, including retrieval misses; training only",
                  "note": "40k*128=5.12M positive slots, fewer than the full TRAIN positive pool; exposure is measured."}
        json_write(prepared_path, report)
        return report
    finally:
        del keys, negatives
        keys_path.unlink(missing_ok=True)
        negative_tmp.unlink(missing_ok=True)


class PositiveCycle:
    """Without-replacement presentations within each epoch, exactly resumable."""

    def __init__(self, size: int, seed: int, *, epoch: int = 0, offset: int = 0):
        if size <= 0 or seed < 0 or epoch < 0 or not 0 <= offset <= size:
            raise ValueError("Invalid positive sampling state")
        self.size, self.seed, self.epoch, self.offset = size, seed, epoch, offset
        self.order = self._order()

    def _order(self) -> np.ndarray:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(self.size)

    def take(self, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("Positive presentation count must be positive")
        chunks = []
        while count:
            if self.offset == self.size:
                self.epoch += 1
                self.offset = 0
                self.order = self._order()
            size = min(count, self.size - self.offset)
            chunks.append(self.order[self.offset:self.offset + size])
            self.offset += size
            count -= size
        return np.concatenate(chunks)

    def state(self) -> dict:
        return {"epoch": self.epoch, "offset": self.offset, "size": self.size, "seed": self.seed}


def hard_negative_indices(negative: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.flatnonzero(negative["channels"] & flag) for key, flag in CHANNELS.items()}


def sample_hard_negatives(pools: dict[str, np.ndarray], rng: np.random.Generator, count: int) -> np.ndarray:
    """50% tree gate, 25% dense, 25% same-name/same-digit near-twins."""
    shares = {"tree": count // 2, "dense": count // 4, "name_digits": count - count // 2 - count // 4}
    if any(len(pools[name]) == 0 and n for name, n in shares.items()):
        raise ValueError("All required hard-negative channels need examples")
    return rng.permutation(np.concatenate([rng.choice(pools[name], n, replace=True)
                                           for name, n in shares.items() if n]))


def name_noise_class(clean: str, noisy: str) -> str:
    """Classify observed TRAIN-linked names; no plan percentages are hardcoded."""
    from rapidfuzz.distance import Levenshtein

    from .normalization import normalize_text

    left, right = normalize_text(clean), normalize_text(noisy)
    if clean != noisy and any(ord(char) > 127 for char in noisy):
        return "script_or_accent"
    if left == right:
        return "exact_after_normalization"
    if sorted(left.split()) == sorted(right.split()):
        return "reorder_only"
    distance = Levenshtein.distance(left, right)
    return "typo_1_2" if distance <= 2 else "edit_3_5" if distance <= 5 else "heavy_edit"


def estimate_train_noise(reference: pd.DataFrame, pool: pd.DataFrame, positives: np.ndarray,
                         *, seed: int, sample_size: int = 50_000) -> dict:
    """Uniformly sample only eligible TRAIN true links, including retrieval misses."""
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(positives), min(sample_size, len(positives)), replace=False)
    counts: Counter[str] = Counter()
    same_digits = 0
    for left, right in positives[positions]:
        source, target = reference.iloc[int(left)], pool.iloc[int(right)]
        counts[name_noise_class(source.business_name, target.business_name)] += 1
        same_digits += _DIGITS.findall(source.business_address) == _DIGITS.findall(target.business_address)
    keys = _pair_key(positives[positions, 0], positives[positions, 1])
    ordered = ["exact_after_normalization", "script_or_accent", "typo_1_2", "edit_3_5",
               "heavy_edit", "reorder_only"]
    return {"scope": "TRAIN true links only (including retrieval misses)", "seed": seed,
            "sample_count": len(positions), "sample_pair_keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
            "name_op_counts": {name: counts[name] for name in ordered},
            "name_op_probabilities": {name: counts[name] / len(positions) for name in ordered},
            "address_exact_digit_sequence_rate": same_digits / len(positions),
            "accented_latin_subrate": 0.10,
            "accented_latin_note": "Explicit low-weight synthetic stress; no French training labels or French accuracy claim."}


def _typo(text: str, rng: np.random.Generator, edits: int) -> str:
    chars = list(text)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for _ in range(edits):
        choices = [i for i, char in enumerate(chars) if char.isalpha() and char.isascii()]
        if not choices:
            return text
        pos = int(rng.choice(choices))
        char = chars[pos]
        replacement = alphabet[(alphabet.find(char.lower()) + int(rng.integers(1, 26))) % 26]
        chars[pos] = replacement.upper() if char.isupper() else replacement
    return "".join(chars)


def _accent_latin(text: str, rng: np.random.Generator) -> str:
    choices = [i for i, char in enumerate(text) if char.lower() in _LATIN_ACCENTS and char.isascii()]
    if not choices:
        return text
    pos = int(rng.choice(choices))
    char = text[pos]
    replacement = _LATIN_ACCENTS[char.lower()]
    return text[:pos] + (replacement.upper() if char.isupper() else replacement) + text[pos + 1:]


def synthetic_positive(name: str, address: str, profile: dict,
                       rng: np.random.Generator) -> tuple[str, str, str]:
    """Noise a TRAIN S1 into a synthetic target; preserve its address digits."""
    names = list(profile["name_op_probabilities"])
    weights = np.asarray(list(profile["name_op_probabilities"].values()), dtype=np.float64)
    op = str(rng.choice(names, p=weights / weights.sum()))
    if op == "exact_after_normalization":
        noisy_name = name.replace(" ", "  ", 1) if " " in name else name.upper()
    elif op == "reorder_only":
        words = name.split()
        noisy_name = " ".join(words[1:] + words[:1]) if len(words) > 1 else name
    elif op == "script_or_accent":
        if rng.random() < profile["accented_latin_subrate"]:
            noisy_name, op = _accent_latin(name, rng), "accented_latin"
        else:
            from unidecode import unidecode
            noisy_name = unidecode(name) if any(ord(char) > 127 for char in name) else _accent_latin(name, rng)
    else:
        noisy_name = _typo(name, rng, {"typo_1_2": 1, "edit_3_5": 3, "heavy_edit": 6}[op])
    # Spacing noise changes presentation without teaching the model that a
    # genuinely different unit/door number is a positive.
    noisy_address = address.replace(" ", "  ", 1) if rng.random() < 0.4 else address
    if _DIGITS.findall(noisy_address) != _DIGITS.findall(address):
        raise AssertionError("Synthetic positive altered a discriminative address digit")
    return noisy_name, noisy_address, op


def synthetic_different_digit(name: str, address: str, rng: np.random.Generator) -> tuple[str, str] | None:
    """Counterfactual negative with the same name but a different door/unit digit."""
    matches = list(_DIGITS.finditer(address))
    if not matches:
        return None
    match = matches[int(rng.integers(len(matches)))]
    original = match.group()
    pos = int(rng.integers(len(original)))
    replaced = original[:pos] + str((int(original[pos]) + int(rng.integers(1, 10))) % 10) + original[pos + 1:]
    result = address[:match.start()] + replaced + address[match.end():]
    if _DIGITS.findall(result) == _DIGITS.findall(address):
        raise AssertionError("Synthetic negative must change a digit sequence")
    return name, result


@dataclass
class TuneData:
    reference: pd.DataFrame
    pool: pd.DataFrame
    pairs: pd.DataFrame
    truth: dict[str, set[str]]
    labels: np.ndarray
    base: np.ndarray
    identity: dict
    left_text: list[str]
    right_text: list[str]


def load_tune(args, forbidden: set[str], prepared: dict) -> TuneData:
    directory = Path(args.tune_data)
    if not args.smoke and (args.tune_base_dir or not args.tune_tree_checkpoint
                           or Path(args.tune_tree_checkpoint).name != "tree_bundle.joblib"):
        raise ValueError("Full CE TUNE needs the completed P3.1 union-tree scores and selected checkpoint")
    original, meta = _checked_tune_references(directory, forbidden, smoke=args.smoke,
                                              full_pool_rows=prepared["identity"]["full_pool_records"],
                                              pairs_override=getattr(args, "tune_pairs", None))
    if meta != prepared["identity"]["tune"]:
        raise ValueError("TUNE retrieval changed after TRAIN preparation")
    original_pairs = pd.read_parquet(meta["pair_source"])
    if original_pairs.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Duplicate TUNE candidate pairs")
    if (len(original_pairs) != meta["candidate_pairs"]
            or original_pairs.left_idx.min() < 0 or original_pairs.left_idx.max() >= len(original)):
        raise ValueError("Completed TUNE candidate count or reference indices differ")
    if args.tune_base_dir:
        from .cross_encoder_experiment import load_base_bundle

        _, tree_pairs, scores, _, provenance = load_base_bundle(Path(args.tune_base_dir), "tabular")
        tree_checkpoint = Path(args.tune_base_dir) / "best.joblib"
    else:
        if not all((args.tune_scores, args.tune_score_pairs, args.tune_tree_checkpoint)):
            raise ValueError("Provide a completed tabular base or aligned scores, pairs and tree checkpoint")
        tree_pairs = pd.read_parquet(args.tune_score_pairs, columns=["left_idx", "right_idx"])
        scores = np.load(args.tune_scores)
        tree_checkpoint = Path(args.tune_tree_checkpoint)
        provenance = {"format": "aligned_scores", "scores_sha256": file_sha256(Path(args.tune_scores)),
                      "pair_file_sha256": file_sha256(Path(args.tune_score_pairs))}
    if (not tree_checkpoint.is_file() or np.asarray(scores).shape != (len(original_pairs),)
            or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
        raise ValueError("TUNE tree scores must exactly align with the completed full-pool candidate pairs")
    original_keys = _pair_key(original_pairs.left_idx.to_numpy(dtype=np.int64),
                              original_pairs.right_idx.to_numpy(dtype=np.int64))
    tree_keys = _pair_key(tree_pairs.left_idx.to_numpy(dtype=np.int64),
                          tree_pairs.right_idx.to_numpy(dtype=np.int64))
    tree_identity = (_full_tree_identity(tree_checkpoint, tune_pair_sha=meta["pair_file_sha256"],
                                         score_pairs=Path(args.tune_score_pairs), scores=Path(args.tune_scores))
                     if tree_checkpoint.name == "tree_bundle.joblib" else
                     {"path": str(tree_checkpoint.resolve()), "sha256": file_sha256(tree_checkpoint)})
    if not np.array_equal(original_keys, tree_keys):
        if tree_checkpoint.name != "tree_bundle.joblib" or len(tree_keys) != len(original_keys):
            raise ValueError("TUNE tree scores must exactly align with the completed full-pool candidate pairs")
        order = np.argsort(tree_keys, kind="stable")
        sorted_keys = tree_keys[order]
        if len(sorted_keys) > 1 and np.any(sorted_keys[1:] == sorted_keys[:-1]):
            raise ValueError("P3.1 TUNE score pairs contain duplicates")
        positions = np.searchsorted(sorted_keys, original_keys)
        if np.any(positions >= len(sorted_keys)) or not np.array_equal(sorted_keys[positions], original_keys):
            raise ValueError("P3.1 TUNE tree score pairs lost or added a full-pool candidate edge")
        scores = np.asarray(scores)[order[positions]]
        provenance["pair_alignment"] = "verified_exact_key_permutation_from_completed_p3_tree"
    else:
        provenance["pair_alignment"] = "identical_original_pair_order"
    pool = pd.read_parquet(directory / "pool.parquet").fillna("")
    if pool.entity_id.eq("").any() or pool.entity_id.duplicated().any() or (
            original_pairs.right_idx.min() < 0 or original_pairs.right_idx.max() >= len(pool)):
        raise ValueError("TUNE target IDs or pair indices are invalid")
    if (not np.array_equal(original.country.to_numpy()[original_pairs.left_idx.to_numpy(dtype=np.int64)],
                           pool.country.to_numpy()[original_pairs.right_idx.to_numpy(dtype=np.int64)])):
        raise ValueError("TUNE candidates cross country boundaries")
    tuning = original.split.eq("tune").to_numpy()
    kept = tuning[original_pairs.left_idx.to_numpy(dtype=np.int64)]
    remap = np.full(len(original), -1, dtype=np.int32)
    remap[np.flatnonzero(tuning)] = np.arange(int(tuning.sum()), dtype=np.int32)
    pairs = original_pairs.loc[kept, ["left_idx", "right_idx"]].copy().reset_index(drop=True)
    pairs["left_idx"] = remap[pairs.left_idx.to_numpy(dtype=np.int64)]
    reference = original.loc[tuning].reset_index(drop=True)
    # This is a completed TRAIN/TUNE development artifact; never read eval or
    # HOLDOUT-2 truth. The reference/metadata/split guards above run first.
    raw_truth = json.loads((directory / "truth.json").read_text(encoding="utf-8"))
    if set(raw_truth) != set(original.entity_id):
        raise ValueError("TUNE truth must have exactly the committed TRAIN/TUNE S1s, including singletons")
    truth = {entity: set(raw_truth[entity]) for entity in reference.entity_id}
    labels = pair_labels(reference, pool, pairs, truth).astype(bool)
    base = np.asarray(scores[kept], dtype=np.float32)
    identity = {**meta, "tree_checkpoint_sha256": tree_identity["sha256"],
                "tree_checkpoint": tree_identity,
                "tree_provenance": provenance, "tree_probability_sha256": hashlib.sha256(base.tobytes()).hexdigest(),
                "tune_pairs": len(pairs), "tune_pair_key_sha256": hashlib.sha256(
                    _pair_key(pairs.left_idx.to_numpy(dtype=np.uint32),
                              pairs.right_idx.to_numpy(dtype=np.uint32)).tobytes()).hexdigest()}
    return TuneData(reference, pool, pairs, truth, labels, base, identity,
                    record_texts(reference), record_texts(pool))


def _threshold_grid(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max()) if len(values) else 1.0
    return np.unique(np.r_[np.linspace(0.05, 0.9, 35), .925, .95, .975, .99, .995,
                           np.nextafter(max(maximum, .995), float("inf"))])


def _entity_frame(data: TuneData, values: np.ndarray, threshold: float) -> pd.DataFrame:
    left = data.pairs.left_idx.to_numpy(dtype=np.int64)
    keep = values.astype(np.float64) >= threshold
    count = np.bincount(left[keep], minlength=len(data.reference))
    tp = np.bincount(left[keep & data.labels], minlength=len(data.reference))
    actual = np.array([len(data.truth[entity]) for entity in data.reference.entity_id], dtype=np.int64)
    denominator = 4 * count + actual
    f05 = np.divide(5.0 * tp, denominator, out=np.zeros(len(actual)), where=denominator > 0)
    f05[(actual == 0) & (count == 0)] = 1.0
    return pd.DataFrame({"source1_entity_id": data.reference.entity_id.to_numpy(),
                         "country": data.reference.country.to_numpy(), "macro_f05": f05,
                         "script": ["non_ascii" if any(ord(char) > 127 for char in name + address)
                                    else "ascii" for name, address in zip(data.reference.business_name,
                                                                             data.reference.business_address)],
                         "true_links": actual, "predicted_links": count, "true_positive_links": tp,
                         "false_positive_links": count - tp, "missed_true_links": actual - tp})


def _entity_report(frame: pd.DataFrame, threshold: float) -> dict:
    actual, predicted, tp = frame.true_links.sum(), frame.predicted_links.sum(), frame.true_positive_links.sum()
    singletons = frame.true_links.eq(0)
    score = {"macro_f05": float(frame.macro_f05.mean()), "micro_precision": float(tp / predicted) if predicted else 0.,
             "micro_recall": float(tp / actual) if actual else 0., "reference_count": len(frame),
             "true_links": int(actual), "predicted_links": int(predicted), "true_positive_links": int(tp),
             "false_positive_links": int(predicted - tp), "missed_true_links": int(actual - tp),
             "singleton_accuracy": float(frame.loc[singletons, "predicted_links"].eq(0).mean()) if singletons.any() else None,
             "threshold": float(threshold)}
    for field in ("country", "script"):
        groups = {}
        for key, group in frame.groupby(field):
            ntp, ntrue, npred = group.true_positive_links.sum(), group.true_links.sum(), group.predicted_links.sum()
            singletons = group.true_links.eq(0)
            groups[str(key)] = {"references": len(group), "macro_f05": float(group.macro_f05.mean()),
                                "micro_precision": float(ntp / npred) if npred else 0.,
                                "micro_recall": float(ntp / ntrue) if ntrue else 0.,
                                "false_positive_links": int(group.false_positive_links.sum()),
                                "singleton_accuracy": float(group.loc[singletons, "predicted_links"].eq(0).mean())
                                if singletons.any() else None}
            if field == "country":
                score[f"f05_{key}"] = groups[str(key)]["macro_f05"]
        score[f"by_{field}"] = groups
    return score


def _target_script_report(data: TuneData, probabilities: np.ndarray, threshold: float) -> list[dict]:
    """Account for native-script truth targets even when retrieval found no edge."""
    from .retrieval_ablation import _script

    target_ids = sorted({target for links in data.truth.values() for target in links})
    positions = pd.Index(data.pool.entity_id).get_indexer(target_ids)
    if np.any(positions < 0):
        raise ValueError("TUNE truth targets, including retrieval misses, must be retained as records")
    targets = {entity: (data.pool.country.iat[int(pos)], _script(data.pool.business_name.iat[int(pos)]))
               for entity, pos in zip(target_ids, positions)}
    groups: dict[tuple[str, str], dict] = {}

    def group(key: tuple[str, str]) -> dict:
        return groups.setdefault(key, {"country": key[0], "target_name_script": key[1],
                                       "true_links": 0, "retrieved_true_links": 0,
                                       "accepted_true_links": 0, "false_positive_links": 0})

    for links in data.truth.values():
        for target in links:
            group(targets[target])["true_links"] += 1
    accepted = probabilities.astype(np.float64) >= threshold
    right = data.pairs.right_idx.to_numpy(dtype=np.int64)
    for index in np.flatnonzero(data.labels):
        target = data.pool.entity_id.iat[right[index]]
        row = group(targets[target])
        row["retrieved_true_links"] += 1
        row["accepted_true_links"] += int(accepted[index])
    for index in np.flatnonzero(accepted & ~data.labels):
        target = data.pool.iloc[right[index]]
        group((target.country, _script(target.business_name)))["false_positive_links"] += 1
    for row in groups.values():
        row["retrieval_link_recall"] = row["retrieved_true_links"] / row["true_links"] if row["true_links"] else None
        row["accepted_link_recall"] = row["accepted_true_links"] / row["true_links"] if row["true_links"] else None
    return [groups[key] for key in sorted(groups)]


def tune_predictions(data: TuneData, cross: np.ndarray, *, gate: float = .01,
                     near_tie: float = .00025) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Full candidate set, complete S1 list, true singletons and retrieval misses."""
    if (cross.shape != data.base.shape or not np.isfinite(cross).all()
            or np.any((cross < 0) | (cross > 1))):
        raise ValueError("CE scores must align exactly with all TUNE candidate rows")
    mask = data.base >= gate
    comparisons, candidates = [], {}
    for name, weight in (("tree_only", 0.), ("blend_0.5", .5), ("blend_0.75", .75)):
        values = data.base if weight == 0 else np.where(mask, (1 - weight) * data.base + weight * cross, 0.)
        best, trace, best_frame = (-1., -1.), [], None
        for threshold in _threshold_grid(values):
            frame = _entity_frame(data, values, float(threshold))
            score = float(frame.macro_f05.mean())
            trace.append({"threshold": float(threshold), "macro_f05": score})
            if (score, float(threshold)) > best:
                best, best_frame = (score, float(threshold)), frame
        assert best_frame is not None
        comparisons.append({"variant": name, "ce_weight": weight, "tune": _entity_report(best_frame, best[1]),
                            "threshold_trace": trace})
        candidates[name] = (best_frame, values.astype(np.float32))
    highest = max(item["tune"]["macro_f05"] for item in comparisons)
    contenders = [item for item in comparisons if highest - item["tune"]["macro_f05"] <= near_tie]
    chosen = min(contenders, key=lambda item: (item["tune"]["false_positive_links"],
                                               -item["tune"]["macro_f05"], item["variant"]))
    found = int(data.labels.sum())
    actual = chosen["tune"]["true_links"]
    oracle = _entity_frame(data, data.labels.astype(np.float32), .5)
    left = data.pairs.left_idx.to_numpy(dtype=np.int64)
    retrieved = np.bincount(left[data.labels], minlength=len(data.reference))
    true_per_reference = np.array([len(data.truth[entity]) for entity in data.reference.entity_id])
    nonempty = true_per_reference > 0
    report = {"scope": "TUNE full competing target pool (synthetic only when smoke=true)",
              "candidate_pairs": len(data.pairs), "reranked_pairs": int(mask.sum()),
              "retrieved_true_links": found, "retrieval_link_recall": float(found / actual) if actual else 0.,
              "missed_retrieval_links": int(actual - found),
              "retrieval_oracle_macro_f05": float(oracle.macro_f05.mean()),
              "retrieval_complete_set_recall": float(np.mean(retrieved[nonempty] == true_per_reference[nonempty]))
              if nonempty.any() else None, "tree_gate": gate,
              "near_tie_tolerance": near_tie, "highest_tune_macro_f05": highest,
              "comparisons": comparisons, "selected_by_tune": {key: value for key, value in chosen.items()
                                                              if key != "threshold_trace"},
              "limitation": "Selected on these same TUNE S1s; no HOLDOUT-2 or France accuracy claim."}
    frame, values = candidates[chosen["variant"]]
    report["by_target_name_script"] = _target_script_report(data, values, chosen["tune"]["threshold"])
    return report, frame, values


def _model_identity(allowlist: Path) -> tuple[Path, dict]:
    """Pin the licensed offline BGE snapshot, including actual cached weight bytes."""
    import yaml
    from huggingface_hub import snapshot_download

    inventory = yaml.safe_load(Path(allowlist).read_text(encoding="utf-8"))
    models = {row["id"]: row for row in inventory["models"]}
    row = models.get(MODEL_ID)
    encoder = models.get("intfloat/multilingual-e5-small")
    if (row is None or row.get("role") != "reranker" or row.get("licence") not in {"mit", "apache-2.0"}
            or encoder is None or float(row["params_b"]) + float(encoder["params_b"])
            > float(inventory["budget_total_params_b"])):
        raise ValueError("BGE and the retrieval encoder must fit the verified MIT/Apache parameter budget")
    snapshot = Path(snapshot_download(MODEL_ID, local_files_only=True)).resolve()
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights or not (snapshot / "config.json").is_file():
        raise ValueError("Licensed BGE sequence-classification weights are absent from the offline cache")
    identity = {"model_id": MODEL_ID, "license": row["licence"], "params_b": float(row["params_b"]),
                "combined_with_e5_params_b": float(row["params_b"]) + float(encoder["params_b"]),
                "snapshot": str(snapshot), "config_sha256": file_sha256(snapshot / "config.json"),
                "weights_sha256": {file.name: file_sha256(file) for file in weights},
                "allowlist_sha256": file_sha256(Path(allowlist))}
    return snapshot, identity


def _load_model(path: Path, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True,
                                                                trust_remote_code=False)
    if model.config.num_labels != 1:
        raise ValueError("BGE must emit one positive-match logit (not a two-class softmax)")
    return model.to(device), tokenizer


def _amp(torch, device: str, amp: str):
    from contextlib import nullcontext

    if not device.startswith("cuda"):
        return nullcontext()
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if amp == "bf16"
            else torch.autocast(device_type="cuda", enabled=False))


def _tokenize(tokenizer, left: list[str], right: list[str], max_length: int, device: str):
    tokens = tokenizer(left, right, padding=True, truncation=True, max_length=max_length,
                       pad_to_multiple_of=8, return_tensors="pt")
    return {name: value.to(device) for name, value in tokens.items()}


def _score_ce(model, tokenizer, data: TuneData, args) -> tuple[np.ndarray, dict]:
    import torch

    model.eval()
    selected = np.flatnonzero(data.base >= args.tree_gate)
    scores = np.zeros(len(data.pairs), dtype=np.float32)
    left, right = (data.pairs[name].to_numpy(dtype=np.int64) for name in ("left_idx", "right_idx"))
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(selected), 1_000):
            group = selected[start:start + 1_000]
            lengths = np.fromiter((len(data.left_text[left[i]]) + len(data.right_text[right[i]])
                                   for i in group), dtype=np.int32, count=len(group))
            group = group[np.argsort(lengths, kind="stable")]
            for begin in range(0, len(group), args.inference_batch):
                ids = group[begin:begin + args.inference_batch]
                tokens = _tokenize(tokenizer, [data.left_text[left[i]] for i in ids],
                                   [data.right_text[right[i]] for i in ids], args.max_length, args.device)
                with _amp(torch, args.device, args.amp):
                    logits = model(**tokens).logits
                if logits.shape != (len(ids), 1):
                    raise ValueError("CE inference must return exactly one positive-match logit per pair")
                scores[ids] = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    if not np.isfinite(scores).all():
        raise ValueError("CE inference produced non-finite scores")
    elapsed = time.perf_counter() - started
    return scores, {"seconds": elapsed, "scored_pairs": len(selected),
                    "scored_pairs_per_second": len(selected) / elapsed if elapsed else None,
                    "gated_pairs_left_unscored": int(len(scores) - len(selected))}


def _batch_pairs(positive: np.ndarray, negative: np.ndarray, cycle: PositiveCycle,
                 pools: dict[str, np.ndarray], rng: np.random.Generator,
                 augmentation_rng: np.random.Generator, seen: np.ndarray,
                 reference: pd.DataFrame, left_text: list[str], right_text: list[str],
                 profile: dict | None, args, counters: Counter) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    half = args.effective_batch // 2
    pids = cycle.take(half)
    nids = sample_hard_negatives(pools, rng, half)
    p = positive[pids]
    n = negative[nids]
    pnoise = augmentation_rng.random(half) < args.noise_positive_fraction
    nnoise = augmentation_rng.random(half) < args.noise_negative_fraction
    left: list[str] = []
    right: list[str] = []
    weights = np.ones(args.effective_batch, dtype=np.float32)
    for slot, (li, ri) in enumerate(p):
        li, ri = int(li), int(ri)
        left.append(left_text[li])
        if pnoise[slot]:
            if profile is None:
                raise AssertionError("Synthetic positives need a measured TRAIN-only noise profile")
            name, address, op = synthetic_positive(reference.business_name.iat[li],
                                                    reference.business_address.iat[li], profile, augmentation_rng)
            right.append(f"{name} | {address}")
            weights[slot] = args.noise_weight
            counters["synthetic_positive"] += 1
            counters[f"noise_{op}"] += 1
        else:
            right.append(right_text[ri])
            counters["real_positive"] += 1
    real_ids = pids[~pnoise]
    counters["distinct_real_positive_seen"] += int(np.count_nonzero(~seen[np.unique(real_ids)]))
    seen[real_ids] = True
    for slot, (li, ri) in enumerate(zip(n["left"], n["right"])):
        li, ri = int(li), int(ri)
        left.append(left_text[li])
        synthetic = (synthetic_different_digit(reference.business_name.iat[li],
                                               reference.business_address.iat[li], augmentation_rng) if nnoise[slot]
                     else None)
        if synthetic is None:
            right.append(right_text[ri])
            counters["real_hard_negative"] += 1
            if nnoise[slot]:
                counters["synthetic_negative_no_digit_fallback"] += 1
        else:
            right.append(f"{synthetic[0]} | {synthetic[1]}")
            weights[half + slot] = args.noise_weight
            counters["synthetic_same_name_different_digit_negative"] += 1
    order = rng.permutation(args.effective_batch)
    target = np.r_[np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32)]
    return [left[i] for i in order], [right[i] for i in order], target[order], weights[order]


def _cosine_lr(step: int, base: float, warmup: int, total: int) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    return base * .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def _checkpoint_marker(directory: Path) -> dict:
    marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    files = marker.get("files_sha256")
    if (marker.get("state") != "complete" or not isinstance(files, dict) or not files
            or set(files) != {file.name for file in directory.iterdir() if file.name != "complete.json"}
            or any(not (directory / name).is_file() for name in files)
            or any(file_sha256(directory / name) != sha for name, sha in files.items())):
        raise ValueError("Checkpoint is not complete or one of its files changed")
    return marker


def _save_checkpoint(model, tokenizer, optimizer, cycle: PositiveCycle, rng: np.random.Generator,
                     augmentation_rng: np.random.Generator,
                     seen: np.ndarray, counters: Counter, config: dict, config_sha: str, step: int,
                     checkpoint_root: Path, output: Path, train_seconds: float, loss: float, torch) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / f"step_{step:06d}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a completed/partial checkpoint: {destination}")
    stage = checkpoint_root / f".step_{step:06d}.incomplete"
    if stage.exists():
        raise FileExistsError(f"Incomplete checkpoint needs inspection: {stage}")
    stage.mkdir()
    model.save_pretrained(stage, safe_serialization=True)
    tokenizer.save_pretrained(stage)
    torch.save({"optimizer": optimizer.state_dict(), "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               stage / "optimizer.pt")
    np.save(stage / "real_positive_seen.npy", seen)
    json_write(stage / "training_config.json", config)
    state = {"completed_steps": step, "config_sha256": config_sha, "positive_cycle": cycle.state(),
             "sample_rng_state": rng.bit_generator.state,
             "augmentation_rng_state": augmentation_rng.bit_generator.state, "counters": dict(counters),
             "train_seconds": train_seconds, "last_loss": loss}
    json_write(stage / "state.json", state)
    files_sha = {file.name: file_sha256(file) for file in stage.iterdir() if file.is_file()}
    marker = {"state": "complete", "step": step, "config_sha256": config_sha,
              "files_sha256": files_sha}
    json_write(stage / "complete.json", marker)
    stage.replace(destination)
    json_write(output / "latest_checkpoint.json", {"checkpoint": str(destination.resolve()),
                                                   "step": step, "marker_sha256": file_sha256(destination / "complete.json")})
    return destination


def _restore_checkpoint(directory: Path, model, optimizer, config_sha: str,
                        positive_count: int, seed: int, *, fork: bool, torch):
    marker = _checkpoint_marker(directory)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if (not fork and marker["config_sha256"] != config_sha) or state["completed_steps"] != marker["step"]:
        raise ValueError("Resume checkpoint does not belong to this training run")
    saved = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_cpu_rng"])
    if saved["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    cycle_state = state["positive_cycle"]
    if cycle_state["size"] != positive_count or cycle_state["seed"] != seed:
        raise ValueError("Checkpoint positive pool or sampler seed differs")
    cycle = PositiveCycle(**cycle_state)
    seen = np.load(directory / "real_positive_seen.npy")
    if seen.shape != (positive_count,) or seen.dtype != np.bool_:
        raise ValueError("Checkpoint real-positive coverage bitmap does not align with the TRAIN pool")
    rng = np.random.default_rng()
    rng.bit_generator.state = state["sample_rng_state"]
    augmentation_rng = np.random.default_rng()
    augmentation_rng.bit_generator.state = state["augmentation_rng_state"]
    return state, cycle, rng, augmentation_rng, seen, Counter(state["counters"])


def _save_tune_inputs(data: TuneData, output: Path) -> None:
    """Retain complete TUNE candidate rows/order and aligned tree scores once."""
    path = output / "tune_pairs.parquet"
    if path.exists():
        old = pd.read_parquet(path, columns=["left_idx", "right_idx"])
        if not np.array_equal(old.to_numpy(), data.pairs[["left_idx", "right_idx"]].to_numpy()):
            raise ValueError("Saved TUNE candidate pairs changed across checkpoints")
    else:
        data.pairs.to_parquet(path, index=False)
    scores_path = output / "tune_tree_probabilities.npy"
    if scores_path.exists():
        if not np.array_equal(np.load(scores_path), data.base):
            raise ValueError("Saved TUNE base scores changed across checkpoints")
    else:
        np.save(scores_path, data.base)
    identity_path = output / "tune_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != data.identity:
        raise ValueError("TUNE identity changed across checkpoints")
    json_write(identity_path, data.identity)


def evaluate_step(model, tokenizer, data: TuneData, args, *, step: int, checkpoint: Path,
                  train_stats: dict) -> tuple[dict, pd.DataFrame]:
    output = Path(args.output) / "tune" / f"step_{step:06d}"
    marker_sha = file_sha256(checkpoint / "complete.json")
    if (output / "complete.json").is_file():
        marker = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if marker.get("checkpoint_marker_sha256") != marker_sha:
            raise ValueError("TUNE result belongs to a different checkpoint")
        return json.loads((output / "report.json").read_text(encoding="utf-8")), pd.read_csv(
            output / "entities.tsv", sep="\t")
    if output.exists():
        raise ValueError(f"Incomplete TUNE output needs inspection before reuse: {output}")
    output.mkdir(parents=True)
    cross, timing = _score_ce(model, tokenizer, data, args)
    report, frame, selected = tune_predictions(data, cross, gate=args.tree_gate,
                                               near_tie=args.near_tie)
    report.update({"step": step, "smoke": args.smoke, "tune_identity": data.identity,
                   "training": train_stats, "ce_inference": timing,
                   "checkpoint": str(checkpoint.resolve()), "checkpoint_marker_sha256": marker_sha})
    np.save(output / "cross_probabilities.npy", cross)
    np.save(output / "selected_probabilities.npy", selected)
    frame.to_csv(output / "entities.tsv", sep="\t", index=False)
    json_write(output / "report.json", report)
    json_write(output / "complete.json", {"checkpoint_marker_sha256": marker_sha,
                                          "report_sha256": file_sha256(output / "report.json"),
                                          "entities_sha256": file_sha256(output / "entities.tsv")})
    return report, frame


def compare_noise_control(noise_output: Path, control_output: Path, step: int,
                          *, seed: int = 20260925, replicates: int = 10_000) -> dict:
    """Paired TUNE ablation at equal steps/start/data; preserve both scored runs."""
    configs = [json.loads((path / "training_config.json").read_text()) for path in (noise_output, control_output)]
    noise, control = configs
    if noise["variant"] != "noise" or control["variant"] != "baseline":
        raise ValueError("Ablation needs a noise run and an independently continued baseline run")
    for name in ("prepared_identity", "tune_identity", "model_identity", "seed", "schedule_steps",
                 "effective_batch", "positive_fraction", "microbatch", "init_checkpoint_marker_sha256",
                 "max_length", "amp", "learning_rate", "weight_decay", "warmup_fraction", "tree_gate",
                 "dense_top_k", "near_tie", "gradient_checkpointing", "eval_steps", "source_sha256",
                 "sampling_rng_streams"):
        if noise[name] != control[name]:
            raise ValueError(f"Noise and baseline runs must agree on {name}")
    reports = [json.loads((path / "tune" / f"step_{step:06d}" / "report.json").read_text())
               for path in (noise_output, control_output)]
    if any(item["step"] != step or item["smoke"] != reports[0]["smoke"] for item in reports):
        raise ValueError("Ablation reports differ in budget or scope")
    frames = [pd.read_csv(path / "tune" / f"step_{step:06d}" / "entities.tsv", sep="\t")
              for path in (noise_output, control_output)]
    a, b = frames
    if (a.source1_entity_id.duplicated().any() or b.source1_entity_id.duplicated().any()
            or not np.array_equal(a.source1_entity_id.to_numpy(), b.source1_entity_id.to_numpy())
            or not np.array_equal(a.country.to_numpy(), b.country.to_numpy())
            or not np.array_equal(a.true_links.to_numpy(), b.true_links.to_numpy())):
        raise ValueError("Ablation must align every TUNE S1, truth count and country")
    paired = paired_bootstrap(a.macro_f05, b.macro_f05, n_bootstrap=replicates, seed=seed)
    report = {"scope": "same full-pool TUNE; scores/thresholds chosen on TUNE",
              "step": step, "noise_selected": reports[0]["selected_by_tune"],
              "baseline_selected": reports[1]["selected_by_tune"], "paired_f05": paired,
              "false_link_delta": int(a.false_positive_links.sum() - b.false_positive_links.sum()),
              "singleton_correct_delta": int(((a.true_links == 0) & (a.predicted_links == 0)).sum()
                                             - ((b.true_links == 0) & (b.predicted_links == 0)).sum()),
              "control_output": str(control_output.resolve()), "noise_output": str(noise_output.resolve()),
              "limitation": "Conditional on reused tuning selections; HOLDOUT-2 is still sealed."}
    json_write(noise_output / f"noise_ablation_step_{step:06d}.json", report)
    return report


def _validate_training_args(args) -> None:
    if args.smoke:
        if args.smoke_full_pool_rows < 1:
            raise ValueError("Synthetic smoke must specify its toy full target-pool size")
    elif args.smoke_full_pool_rows != 0 or args.schedule_steps != 40_000:
        raise ValueError("Full CE training keeps the 40k cosine schedule and full competing pool")
    elif args.init_from:
        start = int(json.loads((Path(args.init_from) / "complete.json").read_text())["step"])
        if (start < 10_000 or not 0 < args.max_steps - start <= 2_000
                or args.eval_steps != [args.max_steps]):
            raise ValueError("Scoped ablations fork a baseline milestone for <=2k steps and score once on TUNE")
    elif args.eval_steps != [10_000, 20_000, 40_000]:
        raise ValueError("Full baseline training evaluates only at 10k/20k/40k on TUNE")
    if (args.effective_batch != 256 or args.microbatch < 1 or 256 % args.microbatch
            or args.max_length != 256 or not 0 < args.learning_rate <= 2e-5 or args.weight_decay < 0
            or not 0 < args.tree_gate < 1 or args.dense_top_k != 50):
        raise ValueError("P3 needs effective batch256, a divisor microbatch, length256, lr<=2e-5 and dense top50")
    if args.learning_rate != 2e-5 and not args.smoke:
        raise ValueError("Keep the planned 2e-5 learning rate for controlled full comparisons")
    if (args.max_steps < 1 or args.max_steps > args.schedule_steps or args.save_every < 1
            or sorted(set(args.eval_steps)) != args.eval_steps
            or args.eval_steps[0] < 1 or args.eval_steps[-1] > args.schedule_steps
            or args.threads < 1 or args.inference_batch < 1 or not 0 < args.warmup_fraction < 1
            or args.plateau_min_delta < 0 or args.near_tie < 0):
        raise ValueError("Invalid step, evaluation, thread or threshold schedule")
    if (not 0 <= args.noise_positive_fraction <= .10 or not 0 <= args.noise_negative_fraction <= .05
            or not 0 < args.noise_weight <= .20):
        raise ValueError("Noise sample fractions and BCE weight must stay at their capped ablation settings")
    if args.variant == "baseline" and (args.noise_positive_fraction or args.noise_negative_fraction):
        raise ValueError("Baseline may not receive synthetic examples")
    if args.variant == "noise" and not (args.noise_positive_fraction or args.noise_negative_fraction):
        raise ValueError("Noise ablation must change at least one capped sampling fraction")
    if args.control_output and args.variant != "noise":
        raise ValueError("Only the augmented branch compares against a control baseline")
    if args.resume and args.init_from:
        raise ValueError("Resume an existing branch or fork from another checkpoint, not both")
    if (not args.smoke and not args.prepare_only and not getattr(args, "materialize_train_scores", False)
            and not args.launch_full):
        raise ValueError("Full GPU launch requires --launch-full after confirming P2 completion and GPU jobs")
    if not args.smoke and not getattr(args, "materialize_train_scores", False) and (args.device != "cuda" or args.amp != "bf16"):
        raise ValueError("Full P3 CE training requires CUDA BF16")
    if args.amp == "bf16" and not args.device.startswith("cuda"):
        raise ValueError("BF16 requires CUDA; CPU compatibility smoke must use --amp fp32")


def _training_config(args, prepared: dict, tune: TuneData, model_identity: dict) -> dict:
    initial = Path(args.init_from) if args.init_from else None
    initial_sha = file_sha256(initial / "complete.json") if initial else None
    config = {"schema": 1, "variant": args.variant, "scope": prepared["identity"]["scope"],
              "source_sha256": file_sha256(Path(__file__)),
              "prepared_identity": prepared["identity"], "tune_identity": tune.identity,
              "model_identity": model_identity, "seed": args.seed,
              "init_checkpoint_marker_sha256": initial_sha,
              "effective_batch": args.effective_batch, "microbatch": args.microbatch,
              "positive_fraction": .5, "max_length": args.max_length, "amp": args.amp,
              "sampling_rng_streams": ["positive_cycle", "hard_negatives_and_order", "TRAIN_only_noise_ops"],
              "optimizer": "AdamW", "lr_schedule": "linear_warmup_cosine_decay",
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "schedule_steps": args.schedule_steps, "warmup_fraction": args.warmup_fraction,
              "eval_steps": args.eval_steps, "tree_gate": args.tree_gate,
              "dense_top_k": args.dense_top_k, "plateau_min_delta": args.plateau_min_delta,
              "near_tie": args.near_tie, "gradient_checkpointing": args.gradient_checkpointing,
              "noise_positive_fraction": args.noise_positive_fraction,
              "noise_negative_fraction": args.noise_negative_fraction,
              "noise_weight": args.noise_weight}
    if initial:
        old = json.loads((initial / "training_config.json").read_text(encoding="utf-8"))
        if old["variant"] != "baseline":
            raise ValueError("Only a completed baseline checkpoint may seed a bounded ablation")
        for name in ("prepared_identity", "tune_identity", "model_identity", "seed", "effective_batch",
                     "positive_fraction", "microbatch", "max_length", "amp", "learning_rate", "weight_decay",
                     "schedule_steps", "warmup_fraction", "tree_gate", "dense_top_k", "gradient_checkpointing",
                     "source_sha256", "sampling_rng_streams"):
            if old[name] != config[name]:
                raise ValueError(f"Forked CE ablation changed {name}; choose the same completed starting checkpoint")
    return config


def train(args, prepared: dict) -> dict:
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("Selected CUDA device needs BF16 support")
        free, _ = torch.cuda.mem_get_info()
        minimum = (10 if args.smoke else args.min_free_vram_gb) * (1024 ** 3)
        if free < minimum:
            raise ValueError(f"Free VRAM {free / 1024**3:.1f} GiB is below the job's {minimum / 1024**3:.0f} GiB floor")
    forbidden, sealed_sha = sealed_ids(args.sealed_reservation, smoke=args.smoke)
    if sealed_sha != prepared["identity"]["sealed_reservation_sha256"]:
        raise ValueError("The sealed split changed after TRAIN preparation")
    tune = load_tune(args, forbidden, prepared)
    if not args.smoke and (prepared["identity"]["tree_score_source"]["tree_checkpoint"]["sha256"]
                           != tune.identity["tree_checkpoint_sha256"]):
        raise ValueError("TRAIN tree gate and TUNE probabilities need the same selected P3.1 checkpoint")
    source, model_identity = _model_identity(args.allowlist)
    config = _training_config(args, prepared, tune, model_identity)
    output = Path(args.output)
    config_path = output / "training_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config or not args.resume:
            raise ValueError("Training config differs, or a prior run needs --resume")
    else:
        if args.resume:
            raise ValueError("Cannot resume before the training config exists")
        json_write(config_path, config)
    config_sha = file_sha256(config_path)
    _save_tune_inputs(tune, output)
    positive = np.load(output / "positive_pairs.npy", mmap_mode="r")
    negative = np.load(output / "negative_pairs.npy", mmap_mode="r")
    if (len(positive) != prepared["positive_pool"] or len(negative) != prepared["hard_negative_pool"]
            or file_sha256(output / "positive_pairs.npy") != prepared["positive_sha256"]
            or file_sha256(output / "negative_pairs.npy") != prepared["negative_sha256"]):
        raise ValueError("Verified TRAIN sampling pools changed after the P2 truth join")
    pools = hard_negative_indices(negative)
    if any(not len(channel) for channel in pools.values()):
        raise ValueError("The TRAIN negative pool lacks a planned hard-negative channel")
    _, train_files = validate_ready(args.train_ready, full_pool_rows=prepared["identity"]["full_pool_records"])
    reference = _records(train_files["reference"], reference=True)
    train_pool = _records(train_files["pool"], reference=False)
    left_text = record_texts(reference)
    right_text = record_texts(train_pool)
    profile = None
    if args.variant == "noise":
        profile = estimate_train_noise(reference, train_pool, positive, seed=args.seed + 1,
                                       sample_size=args.noise_profile_sample)
        profile_path = output / "train_only_noise_profile.json"
        if profile_path.exists() and json.loads(profile_path.read_text(encoding="utf-8")) != profile:
            raise ValueError("TRAIN noise distribution changed across resumes")
        json_write(profile_path, profile)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    checkpoint_root = Path(args.checkpoint_root) if args.checkpoint_root else output / "checkpoints"
    if args.resume:
        latest = json.loads((output / "latest_checkpoint.json").read_text(encoding="utf-8"))
        checkpoint = Path(latest["checkpoint"])
        if file_sha256(checkpoint / "complete.json") != latest["marker_sha256"]:
            raise ValueError("Latest completed checkpoint pointer changed")
        model, tokenizer = _load_model(checkpoint, args.device)
    elif args.init_from:
        checkpoint = Path(args.init_from)
        _checkpoint_marker(checkpoint)
        model, tokenizer = _load_model(checkpoint, args.device)
    else:
        checkpoint = None
        model, tokenizer = _load_model(source, args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if checkpoint:
        state, cycle, rng, augmentation_rng, seen, counters = _restore_checkpoint(
            checkpoint, model, optimizer, config_sha, len(positive), args.seed,
            fork=bool(args.init_from), torch=torch)
        start_step = state["completed_steps"]
        train_seconds = state["train_seconds"]
        last_loss = state["last_loss"]
    else:
        cycle = PositiveCycle(len(positive), args.seed)
        rng = np.random.default_rng(args.seed + 2)
        augmentation_rng = np.random.default_rng(args.seed + 3)
        seen = np.zeros(len(positive), dtype=bool)
        counters = Counter()
        start_step = 0
        train_seconds = last_loss = 0.
    if start_step > args.max_steps or (args.init_from and start_step == args.max_steps):
        raise ValueError("The requested budget must extend the completed checkpoint")
    summary_path = output / "results.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {
        "scope": config["scope"], "model_id": MODEL_ID, "config_sha256": config_sha,
        "positive_pool": len(positive), "hard_negative_pool": len(negative),
        "budget": {"max_optimizer_steps": args.schedule_steps, "effective_batch": 256,
                   "positive_slots_per_step": 128, "positive_slots_at_40k": 5_120_000,
                   "full_positive_exposure_possible_at_40k": len(positive) <= 5_120_000,
                   "real_positive_coverage_reported_per_checkpoint": True},
        "tune_results": [], "plateau": None}
    if summary["config_sha256"] != config_sha or summary.get("plateau"):
        raise ValueError("This output belongs to another run or already stopped at a TUNE plateau")
    evaluations = summary["tune_results"]
    previous_frame = None
    if evaluations:
        previous_step = evaluations[-1]["step"]
        previous_frame = pd.read_csv(output / "tune" / f"step_{previous_step:06d}" / "entities.tsv", sep="\t")
    if args.init_from and evaluations:
        raise ValueError("A fork must start in a new, separate output directory")
    initial_seconds = train_seconds

    def training_stats(step: int, *, recovered: bool = False) -> dict:
        memory = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3} if args.device.startswith("cuda") else {}
        elapsed_here = train_seconds - initial_seconds
        return {"completed_steps": step, "optimizer_steps_this_invocation": step - start_step,
                "elapsed_train_seconds": train_seconds, "last_bce": last_loss,
                "effective_presentations": step * args.effective_batch,
                "real_positive_presentations": counters["real_positive"],
                "synthetic_positive_presentations": counters["synthetic_positive"],
                "distinct_real_positive_seen": int(seen.sum()),
                "real_positive_pool_coverage": float(seen.mean()),
                "real_hard_negative_presentations": counters["real_hard_negative"],
                "synthetic_negative_presentations": counters["synthetic_same_name_different_digit_negative"],
                "presentations_per_second": step * args.effective_batch / train_seconds if train_seconds else None,
                "invocation_presentations_per_second": ((step - start_step) * args.effective_batch / elapsed_here
                                                        if elapsed_here else None),
                "noise_ops": {key: count for key, count in counters.items() if key.startswith("noise_")},
                "evaluation_recovered_after_interruption": recovered, **memory}

    def record_tune(step: int, checkpoint: Path, stats: dict) -> bool:
        nonlocal previous_frame
        report, frame = evaluate_step(model, tokenizer, tune, args, step=step,
                                      checkpoint=checkpoint, train_stats=stats)
        result = {"step": step, "checkpoint": str(checkpoint.resolve()),
                  "selected_by_tune": report["selected_by_tune"], "training": stats}
        if previous_frame is not None:
            result["paired_vs_previous_step"] = paired_bootstrap(
                frame.macro_f05, previous_frame.macro_f05,
                n_bootstrap=100 if args.smoke else 10_000, seed=args.seed)
        if args.control_output:
            result["noise_ablation"] = compare_noise_control(output, Path(args.control_output), step,
                                                              seed=args.seed,
                                                              replicates=100 if args.smoke else 10_000)
        previous_frame = frame
        previous_best = max((row["selected_by_tune"]["tune"]["macro_f05"] for row in evaluations), default=-1.)
        evaluations.append(result)
        if len(evaluations) >= 2 and report["selected_by_tune"]["tune"]["macro_f05"] <= previous_best + args.plateau_min_delta:
            summary["plateau"] = {"stopped_at_step": step, "minimum_gain_required": args.plateau_min_delta,
                                  "best_previous_tune_f05": previous_best,
                                  "decision": "Stop budget extension; retain all checkpoint reports."}
        summary["latest_step"] = step
        summary["latest_checkpoint"] = str(checkpoint.resolve())
        json_write(summary_path, summary)
        return summary["plateau"] is not None

    # A crash between checkpoint commit and TUNE scoring must not silently
    # advance past an unevaluated milestone on resume.
    if args.resume and start_step in args.eval_steps and not any(row["step"] == start_step for row in evaluations):
        if record_tune(start_step, checkpoint, training_stats(start_step, recovered=True)):
            return summary
    if start_step == args.max_steps:
        return summary
    started = time.perf_counter()
    warmup = max(1, round(args.schedule_steps * args.warmup_fraction))
    for step in range(start_step + 1, args.max_steps + 1):
        model.train()
        left, right, target, weights = _batch_pairs(positive, negative, cycle, pools, rng, augmentation_rng, seen,
                                                     reference, left_text, right_text, profile, args, counters)
        optimizer.zero_grad(set_to_none=True)
        normalizer = float(weights.sum())
        loss_value = 0.
        for begin in range(0, args.effective_batch, args.microbatch):
            stop = begin + args.microbatch
            tokens = _tokenize(tokenizer, left[begin:stop], right[begin:stop], args.max_length, args.device)
            labels = torch.as_tensor(target[begin:stop], dtype=torch.float32, device=args.device)
            importance = torch.as_tensor(weights[begin:stop], dtype=torch.float32, device=args.device)
            with _amp(torch, args.device, args.amp):
                logits = model(**tokens).logits
            if logits.shape != (args.microbatch, 1):
                raise ValueError("CE training needs one logit per presentation")
            loss = (F.binary_cross_entropy_with_logits(logits[:, 0].float(), labels, reduction="none")
                    * importance).sum() / normalizer
            loss.backward()
            loss_value += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        lr = _cosine_lr(step, args.learning_rate, warmup, args.schedule_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        last_loss = loss_value
        if step % args.log_every == 0 or step == 1 or step in args.eval_steps:
            print(json.dumps({"ce_step": step, "bce": loss_value, "lr": lr,
                              "real_positive_seen": counters["distinct_real_positive_seen"],
                              "positive_pool": len(positive)}), flush=True)
        if step not in args.eval_steps and step % args.save_every:
            continue
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - started
        checkpoint = _save_checkpoint(model, tokenizer, optimizer, cycle, rng, augmentation_rng, seen, counters,
                                      config, config_sha, step, checkpoint_root, output,
                                      train_seconds, last_loss, torch)
        if step in args.eval_steps:
            if record_tune(step, checkpoint, training_stats(step)):
                break
        started = time.perf_counter()
    print(json.dumps({"scope": summary["scope"], "latest_step": summary.get("latest_step", start_step),
                      "plateau": summary["plateau"], "checkpoint_root": str(checkpoint_root)}), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ready", type=Path, required=True, help="Atomic completed P2 TRAIN/dense/truth-join marker")
    parser.add_argument("--train-scores", type=Path, help="P3.1 tree probabilities for EVERY P2 TRAIN candidate")
    parser.add_argument("--train-score-pairs", type=Path, help="Exact pair order accompanying P3.1 TRAIN probabilities")
    parser.add_argument("--train-tree-checkpoint", type=Path, help="Completed selected P3.1 TRAIN tree checkpoint")
    parser.add_argument("--sealed-reservation", type=Path, required=True, help="ID-only HOLDOUT-2 reservation directory")
    parser.add_argument("--tune-data", type=Path, help="Completed 12k TUNE full-pool dataset (not read by CPU tree scoring)")
    parser.add_argument("--tune-pairs", type=Path, help="P2 union's completed score-context candidate Parquet")
    parser.add_argument("--tune-base-dir", type=Path, help="Completed aligned tabular tree bundle")
    parser.add_argument("--tune-scores", type=Path, help="Aligned probabilities.npy from a separate tree")
    parser.add_argument("--tune-score-pairs", type=Path, help="Tree probability pair-order file")
    parser.add_argument("--tune-tree-checkpoint", type=Path, help="Selected tree checkpoint for score provenance")
    parser.add_argument("--output", type=Path, default=Path("/tmp/az-ml-v2/ce_screen"))
    parser.add_argument("--checkpoint-root", type=Path, help="Separate durable v2 directory for large checkpoints")
    parser.add_argument("--allowlist", type=Path, default=Path(__file__).resolve().parents[4] / "problem-statement/models.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--materialize-train-scores", action="store_true",
                        help="CPU score all P2 TRAIN candidates with the completed P3.1 tree, in original order")
    parser.add_argument("--score-reference-batch", type=int, default=1_000)
    parser.add_argument("--score-pair-read-batch", type=int, default=100_000)
    parser.add_argument("--score-max-pairs-per-batch", type=int, default=200_000)
    parser.add_argument("--launch-full", action="store_true", help="Acknowledge P2 completion and an inspected GPU queue")
    parser.add_argument("--smoke", action="store_true", help="Synthetic compatibility check only; never a model-selection score")
    parser.add_argument("--smoke-full-pool-rows", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--min-free-vram-gb", type=float, default=30.)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--effective-batch", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--inference-batch", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--tree-gate", type=float, default=.01)
    parser.add_argument("--dense-top-k", type=int, default=50)
    parser.add_argument("--schedule-steps", type=int, default=40_000)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[10_000, 20_000, 40_000])
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--warmup-fraction", type=float, default=.06)
    parser.add_argument("--plateau-min-delta", type=float, default=.0005)
    parser.add_argument("--near-tie", type=float, default=.00025)
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Trade throughput for lower A40 activation memory")
    parser.add_argument("--resume", action="store_true", help="Restore optimizer and all sampling/RNG state")
    parser.add_argument("--init-from", type=Path, help="Fork equal-budget baseline/noise runs from this same checkpoint")
    parser.add_argument("--variant", choices=("baseline", "noise"), default="baseline")
    parser.add_argument("--noise-positive-fraction", type=float, default=0.)
    parser.add_argument("--noise-negative-fraction", type=float, default=0.)
    parser.add_argument("--noise-weight", type=float, default=.2)
    parser.add_argument("--noise-profile-sample", type=int, default=50_000)
    parser.add_argument("--control-output", type=Path, help="Equal-step baseline for the paired TUNE noise ablation")
    args = parser.parse_args(argv)
    if (min(args.log_every, args.noise_profile_sample, args.score_reference_batch,
            args.score_pair_read_batch, args.score_max_pairs_per_batch) < 1 or args.min_free_vram_gb <= 0):
        parser.error("Log interval, data sampling, CPU score batches and free-VRAM floor must be positive")
    if args.materialize_train_scores and (args.prepare_only or args.resume or args.init_from):
        parser.error("TRAIN tree scoring is a separate, one-time CPU materialization")
    if not args.materialize_train_scores and not args.tune_data:
        parser.error("TUNE data is required for CE preparation/training")
    _validate_training_args(args)
    if args.materialize_train_scores:
        report = materialize_train_tree_scores(args)
        print(json.dumps({"state": report["state"], "pairs": report["pairs"],
                          "scores_path": report["scores_path"],
                          "score_pairs_path": report["score_pairs_path"]}), flush=True)
        return
    prepared = prepare(args)
    if args.prepare_only:
        print(json.dumps({"scope": prepared["identity"]["scope"], "positive_pool": prepared["positive_pool"],
                          "hard_negative_pool": prepared["hard_negative_pool"],
                          "missed_train_truth_links": prepared["missed_train_truth_links"],
                          "state": "ready_for_bounded_training"}), flush=True)
    else:
        train(args, prepared)


if __name__ == "__main__":
    main()
