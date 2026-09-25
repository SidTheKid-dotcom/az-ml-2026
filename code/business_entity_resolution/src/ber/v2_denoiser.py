"""P4.2: TRAIN-only ByT5 noisy-target -> clean-S1 canonicalization.

The 200/1k/5k/10k screens fit only the saved 35,999 medium TRAIN businesses.
``prepare-full`` separately joins the 2,072,821 eligible TRAIN S1s to their
7.17M true targets; it does not claim that a checkpoint trained on this pool.
The checkpoint comparison is
the *same* 12,000 medium TUNE S1s and completed, label-free candidate union from
all 10,320,219 targets. TUNE truth is first opened in ``evaluate``, after the
checkpoint and candidate file already exist. Unretrieved true target *records*
are used for canonicalization diagnostics, never inserted as retrieval edges.

``feature-band`` aligns the saved v1 TUNE probabilities to 600,000 original
dense edges, leaving all 837,538 union-only edges unscored; ``feature-generate``
canonicalizes a label-free bounded band, and ``feature-score`` calibrates binary
equality on TRAIN before official F0.5 scoring over all 12,000 TUNE S1s.

On the pod, copy just this script into /tmp/az-ml-v2/denoiser and run it with
PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-v2/code/business_entity_resolution/src,
HF_HOME=/workspace/amlc/cache/hf, HF_HUB_OFFLINE=1, and at most two CPU threads.
The CLI subcommands' ``--help`` lists the inputs.
All new files are confined to --output and --durable; the saved union and frozen
v1 model are read-only. A later *pool-wide* canonical retrieval branch needs its
own complete candidate mining and matcher re-fit; this bounded screen does not
pretend that canonicalizing a sample has increased full-pool recall.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.normalization import normalize_text
from ber.retrieval_ablation import _script
from ber.uncertainty import paired_bootstrap


MODEL_ID = "google/byt5-small"
TRAIN_ID_SHA = "47ec6f4736aaa6069c9dbddde67354637b3046d414165bea0c54ab0dcd5bc0dc"
TUNE_ID_SHA = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
FULL_POOL_ROWS = 10_320_219
RAW_S1_ROWS = 2_206_821
FULL_TRAIN_S1_ROWS = 2_072_821
FULL_TRAIN_ID_SHA = "7720350d18cd1333dbeb0e688bdc67fc4ed47c4ee2c1a54126f7743382944e01"
SOURCE2_ROWS = 5_034_616
TEXT_FIELDS = ("business_name", "business_address")
RECORD_FIELDS = ("entity_id", "business_name", "business_address", "country")
_DIGITS = re.compile(r"\d+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to replace a completed/partial artifact: {path}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verified_copy(source: Path, destination: Path) -> None:
    """Copy only our own output, preserving an existing identical durable file."""
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise ValueError(f"Different durable output already exists: {destination}")
        return
    temporary = Path(str(destination) + ".tmp")
    if temporary.exists():
        raise ValueError(f"Incomplete durable output requires inspection: {temporary}")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != digest:
        raise ValueError(f"Durable copy failed SHA256 verification: {destination}")
    temporary.replace(destination)


def text_for_record(name: str, address: str) -> str:
    """Keep native scripts and original digits; no IDs/country/labels in text."""
    return f"{name} | {address}"


def texts_digest(texts) -> str:
    digest = hashlib.sha256()
    for value in texts:
        raw = value.encode("utf-8")
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return digest.hexdigest()


def checked_reservation(directory: Path, *, strict: bool = True) -> tuple[set[str], set[str], dict]:
    """Open ID-only files; the sealed labels are never an input."""
    path = Path(directory) / "reservation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report["verification"].get("labels_read") is not False
            or report["prior_exclusions"].get("pairwise_disjoint") is not True
            or report["verification"].get("older_and_new_holdouts_excluded_from_future_train_tune") is not True):
        raise ValueError("HOLDOUT-2/older sealed split verification is incomplete")

    def load(entry: dict, key: str, expected: int | None) -> set[str]:
        file = Path(directory) / entry[key]
        raw = file.read_bytes()
        values = raw.decode("utf-8").splitlines()
        file_sha_key = "excluded_file_sha256" if key == "excluded_id_file" else "id_file_sha256"
        if (sha256(file) != entry[file_sha_key]
                or len(values) != entry["excluded_count" if key == "excluded_id_file" else "count"]
                or (expected is not None and len(values) != expected)
                or values != sorted(set(values)) or not raw.endswith(b"\n")
                or raw != ("\n".join(values) + "\n").encode("utf-8")
                or any(not entity.startswith("S1-") for entity in values)
                or ids_sha(values) != entry["excluded_sorted_id_sha256" if key == "excluded_id_file"
                                               else "sorted_id_sha256"]):
            raise ValueError(f"Sealed ID file is incomplete, reordered or changed: {file}")
        return set(values)

    # Reservation uses excluded_file_sha256 and id_file_sha256, respectively.
    forbidden = load(report["future_raw_train_tune"], "excluded_id_file", 134_000 if strict else None)
    held = load(report["holdout2"], "id_file", 50_000 if strict else None)
    if not held <= forbidden:
        raise ValueError("HOLDOUT-2 is not covered by the TRAIN exclusion")
    return forbidden, held, {"path": str(path.resolve()), "sha256": sha256(path),
                              "excluded_ids_sha256": report["future_raw_train_tune"]["excluded_file_sha256"],
                              "holdout2_ids_sha256": report["holdout2"]["id_file_sha256"]}


def checked_medium(medium_dense: Path, medium_splits: Path, reservation: Path,
                   *, strict: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fit on medium TRAIN, excluding TUNE, both evaluated holds and quick IDs."""
    forbidden, held, seal = checked_reservation(reservation, strict=strict)
    metadata = json.loads((medium_dense / "metadata.json").read_text(encoding="utf-8"))
    complete = json.loads((medium_dense / "complete.json").read_text(encoding="utf-8"))
    reserved = json.loads((reservation / "reservation.json").read_text(encoding="utf-8"))
    if (complete.get("state") != "complete" or complete.get("metadata") != metadata
            or complete.get("config", {}).get("labels_used_for_mining") is not False
            or metadata.get("positive_candidate_injection") is not False
            or (strict and metadata.get("mining_full_pool_records") != FULL_POOL_ROWS)
            or (not strict and int(metadata.get("mining_full_pool_records", 0)) <= 0)
            or sha256(medium_splits) != reserved["inputs"]["medium_all_splits"]["file_sha256"]):
        raise ValueError("Saved medium references are not the sealed full-pool TRAIN/TUNE mining result")
    frame = pd.read_parquet(medium_dense / "reference.parquet", columns=[*RECORD_FIELDS, "split"])
    if (frame.entity_id.isna().any() or frame.entity_id.duplicated().any()
            or set(frame.split) != {"train", "tune"} or frame[list(TEXT_FIELDS)].isna().any().any()
            or not frame.country.isin(["India", "US"]).all()):
        raise ValueError("Medium TRAIN/TUNE records are incomplete, duplicated or country-mismatched")
    if ids_sha(frame.entity_id) != metadata["reference_id_sha256"]:
        raise ValueError("Medium references differ from the completed miner's ID set")
    train = frame.loc[frame.split.eq("train")].reset_index(drop=True)
    tune = frame.loc[frame.split.eq("tune")]
    if (strict and (len(train) != 35_999 or len(tune) != 12_000
                    or ids_sha(train.entity_id) != TRAIN_ID_SHA or ids_sha(tune.entity_id) != TUNE_ID_SHA)
            or (set(train.entity_id) & set(tune.entity_id)) or set(train.entity_id) & held
            or not set(train.entity_id) <= forbidden or not set(tune.entity_id) <= forbidden):
        raise ValueError("Denoiser TRAIN overlaps a TUNE/holdout set or its fixed ID digest changed")
    # The reservation verified quick/medium/older-holdout pairwise disjointness.
    return train, {"dense_complete_sha256": sha256(medium_dense / "complete.json"),
                   "dense_reference_sha256": sha256(medium_dense / "reference.parquet"),
                   "dense_pool_sha256": sha256(medium_dense / "pool.parquet"),
                   "medium_all_splits_sha256": sha256(medium_splits),
                   "train_ids_sha256": ids_sha(train.entity_id), "tune_ids_sha256": ids_sha(tune.entity_id),
                   "train_s1_count": len(train), "tune_s1_count": len(tune), "reservation": seal,
                   "full_competing_pool_rows": metadata["mining_full_pool_records"]}


def train_truth_only(raw_truth: Path, train_ids: set[str], *, strict: bool = True) -> tuple[dict, dict]:
    """Do not decode the label field of any non-TRAIN S1 (including held-out IDs)."""
    truth: dict[str, list[str]] = {}
    digest = hashlib.sha256()
    lines = 0
    with raw_truth.open("rb") as stream:
        header = stream.readline()
        digest.update(header)
        if header.rstrip(b"\r\n") != b"source1_entity_id\tmatched_entity_ids":
            raise ValueError("Unexpected supplied ground-truth TSV header")
        for raw in stream:
            digest.update(raw)
            lines += 1
            entity, sep, labels = raw.partition(b"\t")
            if not sep:
                raise ValueError("Malformed source1 ID/label separator in supplied truth")
            key = entity.decode("utf-8")
            if key not in train_ids:
                continue  # In particular, do not decode any TUNE/eval/holdout labels.
            if key in truth:
                raise ValueError("Duplicate TRAIN Source 1 truth row")
            targets = [x for x in labels.rstrip(b"\r\n").decode("utf-8").split(",") if x]
            if len(set(targets)) != len(targets) or any(not x.startswith(("S2-", "S3-")) for x in targets):
                raise ValueError("Duplicate or invalid TRAIN target link")
            truth[key] = sorted(targets)
    if set(truth) != train_ids or (strict and lines != RAW_S1_ROWS):
        raise ValueError("Supplied truth did not contain exactly the selected TRAIN references")
    return truth, {"path": str(raw_truth.resolve()), "sha256": digest.hexdigest(),
                   "source1_rows_scanned": lines, "selected_train_rows_decoded": len(truth),
                   "non_train_labels_decoded": 0}


def prepare(medium_dense: Path, medium_splits: Path, reservation: Path, ground_truth: Path,
            output: Path, durable: Path, *, strict: bool = True) -> dict:
    """Materialize true TRAIN pairs, without reading TUNE truth or touching candidates."""
    output, durable = Path(output), Path(durable)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Denoiser preparation already has output; inspect before rerunning: {output}")
    if durable.exists() and any(durable.iterdir()):
        raise FileExistsError(f"Denoiser durable output already exists: {durable}")
    reference, medium = checked_medium(medium_dense, medium_splits, reservation, strict=strict)
    truth, truth_identity = train_truth_only(ground_truth, set(reference.entity_id), strict=strict)
    pool = pd.read_parquet(medium_dense / "pool.parquet", columns=list(RECORD_FIELDS))
    if pool.entity_id.duplicated().any() or pool[list(RECORD_FIELDS)].isna().any().any():
        raise ValueError("The saved full-pool target record cache contains duplicate/null records")
    left, targets = [], []
    for position, entity in enumerate(reference.entity_id):
        for target in truth[entity]:
            left.append(position)
            targets.append(target)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("TRAIN targets must be unique, nonempty and one-to-one across businesses")
    right = pd.Index(pool.entity_id).get_indexer(targets)
    if np.any(right < 0):
        raise ValueError("Saved target record cache lost a TRAIN match (not a retrieval edge)")
    left = np.asarray(left, dtype=np.int32)
    right = right.astype(np.int32)
    if not np.array_equal(reference.country.to_numpy()[left], pool.country.to_numpy()[right]):
        raise ValueError("TRAIN matched target crossed a country boundary")
    edges = pd.DataFrame({"left_idx": left, "right_idx": right})
    clean = [text_for_record(name, address) for name, address in
             reference[list(TEXT_FIELDS)].itertuples(index=False, name=None)]
    noisy = [text_for_record(name, address) for name, address in
             pool.iloc[right][list(TEXT_FIELDS)].itertuples(index=False, name=None)]
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(output / "train_reference.parquet", index=False)
    edges.to_parquet(output / "train_pairs.parquet", index=False)
    identity = {"scope": "medium_TRAIN_only" if strict else "synthetic_smoke", "medium": medium,
                "raw_truth": truth_identity, "positive_links": len(edges),
                "train_reference_text_sha256": texts_digest(clean),
                "noisy_target_text_in_pair_order_sha256": texts_digest(noisy),
                "field_representation": "original-script business_name | business_address; no ID/country",
                "positive_candidate_injection": False, "medium_eval_labels_read": False,
                "holdout2_labels_read": False, "tune_labels_read": False,
                "train_reference_file_sha256": sha256(output / "train_reference.parquet"),
                "train_pair_file_sha256": sha256(output / "train_pairs.parquet")}
    for name in ("train_reference.parquet", "train_pairs.parquet"):
        verified_copy(output / name, durable / name)
    report = {"state": "complete", "identity": identity, "training_pairs": len(edges),
              "train_businesses": len(reference), "target_sources": dict(Counter(x.split("-", 1)[0] for x in targets)),
              "train_countries": {k: int(v) for k, v in reference.country.value_counts().items()}}
    atomic_json(output / "prepared.json", report)
    verified_copy(output / "prepared.json", durable / "prepared.json")
    return report


def _prepared(output: Path, durable: Path) -> dict:
    report = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    ident = report["identity"]
    if (report.get("state") != "complete" or ident.get("positive_candidate_injection") is not False
            or ident.get("tune_labels_read") is not False or ident.get("holdout2_labels_read") is not False):
        raise ValueError("TRAIN pair preparation did not finish with its split guards")
    for name, key in (("train_reference.parquet", "train_reference_file_sha256"),
                      ("train_pairs.parquet", "train_pair_file_sha256")):
        if sha256(output / name) != ident[key] or sha256(durable / name) != ident[key]:
            raise ValueError(f"Prepared TRAIN bytes changed: {name}")
    if sha256(output / "prepared.json") != sha256(durable / "prepared.json"):
        raise ValueError("Prepared TRAIN identity differs from its durable copy")
    return report


def prepare_full_train(bootstrap_dir: Path, reservation_dir: Path, ground_truth: Path,
                       output: Path, durable: Path, *, strict: bool = True,
                       batch_links: int = 50_000) -> dict:
    """Stream only 2.07M eligible TRAIN labels to positional noisy->clean pairs.

    This is a denoiser training join, independent of P2.5's unfinished candidate
    union. S1 IDs and the full S2/S3 pool come from P2's sealed, label-free
    bootstrap; no retrieved edge is created or modified. The joined Parquet is
    useful for later full-data fitting, but this command does not train on it.
    """
    if not 1 <= batch_links <= 100_000:
        raise ValueError("Full-TRAIN truth mapping batches must stay <=100,000 links")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    bootstrap_dir, reservation_dir = Path(bootstrap_dir), Path(reservation_dir)
    output, durable = Path(output), Path(durable)
    if (output.exists() and any(output.iterdir())) or (durable.exists() and any(durable.iterdir())):
        raise FileExistsError("Full denoiser pair output already exists; inspect before rerunning")
    if shutil.disk_usage(output.parent).free < 4_500_000_000:
        raise ValueError("Full TRAIN pair join would violate the 4 GB scratch reserve")
    forbidden, held, seal = checked_reservation(reservation_dir, strict=strict)
    bootstrap = json.loads((bootstrap_dir / "bootstrap.json").read_text(encoding="utf-8"))
    pool_complete = json.loads((bootstrap_dir / "pool_complete.json").read_text(encoding="utf-8"))
    reference_file = bootstrap_dir / bootstrap["files"]["reference"]["path"]
    pool_file = bootstrap_dir / bootstrap["files"]["pool"]["path"]
    ref_info, pool_info = bootstrap["files"]["reference"], bootstrap["files"]["pool"]
    expected_rows = FULL_TRAIN_S1_ROWS if strict else ref_info["rows"]
    expected_pool = FULL_POOL_ROWS if strict else pool_info["rows"]
    if (bootstrap.get("state") != "complete" or bootstrap.get("truth_read") is not False
            or bootstrap.get("positive_candidate_injection") is not False
            or bootstrap["reservation_sha256"] != seal["sha256"]
            or pool_complete.get("state") != "complete" or pool_complete.get("truth_read") is not False
            or pool_complete["files"]["pool"]["sha256"] != pool_info["sha256"]
            or ref_info["rows"] != expected_rows or pool_info["rows"] != expected_pool
            or (strict and bootstrap["train_reference_id_sha256"] != FULL_TRAIN_ID_SHA)
            or any(path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]
                   or pq.ParquetFile(path).metadata.num_rows != info["rows"]
                   for path, info in ((reference_file, ref_info), (pool_file, pool_info)))):
        raise ValueError("P2 full-eligible reference/target pool or HOLDOUT-2 reservation changed")
    reference = pd.read_parquet(reference_file, columns=["entity_id", "country"])
    if (len(reference) != expected_rows or reference.entity_id.duplicated().any()
            or not reference.entity_id.str.startswith("S1-").all()
            or set(reference.country) - {"US", "India"}
            or set(reference.entity_id) & forbidden or set(reference.entity_id) & held
            or ids_sha(reference.entity_id) != bootstrap["train_reference_id_sha256"]):
        raise ValueError("Full TRAIN reference has duplicate, foreign or reserved S1 IDs")
    # Read only IDs/country from the supplied full pool; the original-script
    # business text remains in the checksum-bound Parquet for future training.
    pool = pd.read_parquet(pool_file, columns=["entity_id", "country"])
    pool_ids = pd.Index(pool.entity_id)
    if (len(pool) != expected_pool or not pool_ids.is_unique
            or set(pool.country) - {"US", "India"}
            or not pool.entity_id.str.startswith(("S2-", "S3-")).all()):
        raise ValueError("Full supplied target pool lost rows/IDs/country")
    ref_codes = pd.Categorical(reference.country, categories=["India", "US"]).codes
    pool_codes = pd.Categorical(pool.country, categories=["India", "US"]).codes
    if np.any(ref_codes < 0) or np.any(pool_codes < 0):
        raise ValueError("Original train country labels changed")
    positions = {entity: i for i, entity in enumerate(reference.entity_id)}
    seen_refs = np.zeros(expected_rows, dtype=bool)
    seen_targets = np.zeros(expected_pool, dtype=bool)
    counts = np.zeros(expected_rows, dtype=np.uint16)
    source_counts: Counter[str] = Counter()
    source2_rows = SOURCE2_ROWS if strict else pool_complete.get("source2_rows", expected_pool // 2)
    pending_left: list[int] = []
    pending_targets: list[str] = []
    raw_rows = links = selected_rows = 0
    truth_hash = hashlib.sha256()
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    path = output / "train_truth_pairs.parquet"
    temporary = output / "train_truth_pairs.parquet.tmp"
    writer = None

    def flush() -> None:
        nonlocal writer, links
        if not pending_targets:
            return
        right = pool_ids.get_indexer(pending_targets)
        left = np.asarray(pending_left, dtype=np.int32)
        if (np.any(right < 0) or np.any(seen_targets[right])
                or len(np.unique(right)) != len(right)
                or not np.array_equal(ref_codes[left], pool_codes[right])
                or np.any(np.asarray([x.startswith("S2-") for x in pending_targets], dtype=bool)
                          != (right < source2_rows))):
            raise ValueError("Full TRAIN link is missing, duplicated or crosses source/country")
        seen_targets[right] = True
        table = pa.table({"left_idx": left, "right_idx": right.astype(np.int32)})
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=len(left))
        links += len(left)
        pending_left.clear()
        pending_targets.clear()

    try:
        with ground_truth.open("rb") as stream:
            header = stream.readline()
            truth_hash.update(header)
            if header != b"source1_entity_id\tmatched_entity_ids\n":
                raise ValueError("Supplied TRAIN ground-truth header changed")
            for line in stream:
                truth_hash.update(line)
                raw_rows += 1
                entity, sep, payload = line.partition(b"\t")
                if not sep:
                    raise ValueError("Malformed supplied ground-truth S1 ID field")
                left_idx = positions.get(entity.decode("ascii"))
                if left_idx is None:
                    continue  # No HOLDOUT-2/older/medium-eval payload is decoded.
                if seen_refs[left_idx]:
                    raise ValueError("Duplicate eligible TRAIN S1 truth row")
                seen_refs[left_idx] = True
                selected_rows += 1
                values = payload.rstrip(b"\r\n")
                if values:
                    targets = values.decode("ascii").split(",")
                    if (len(set(targets)) != len(targets) or
                            any(not target.startswith(("S2-", "S3-")) for target in targets)
                            or counts[left_idx] > np.iinfo(np.uint16).max - len(targets)):
                        raise ValueError("Invalid/duplicate eligible TRAIN target link")
                    counts[left_idx] = len(targets)
                    pending_left.extend([left_idx] * len(targets))
                    pending_targets.extend(targets)
                    source_counts.update(target[:2] for target in targets)
                if len(pending_targets) >= batch_links:
                    flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if (writer is None or selected_rows != expected_rows or not seen_refs.all()
            or (strict and raw_rows != RAW_S1_ROWS)
            or links != int(counts.sum(dtype=np.int64))
            or int(seen_targets.sum()) != links
            or pq.ParquetFile(temporary).metadata.num_rows != links):
        raise ValueError("Full TRAIN truth coverage or streaming pair count is incomplete")
    if shutil.disk_usage(output).free < 4_000_000_000:
        raise ValueError("Full TRAIN pair join violated the 4 GB scratch reserve")
    temporary.replace(path)
    verified_copy(path, durable / path.name)
    code_copy = durable / "v2_denoiser_fullprep.py"
    verified_copy(Path(__file__), code_copy)
    report = {"state": "complete", "scope": "2,072,821 full eligible TRAIN S1s only" if strict else "synthetic_smoke",
              "train_businesses": expected_rows, "true_links": links,
              "true_singletons": int(np.count_nonzero(counts == 0)),
              "matched_target_sources": dict(source_counts), "seed_for_later_training": 20260925,
              "train_reference_id_sha256": ids_sha(reference.entity_id),
              "holdout2_reservation": seal, "bootstrap_sha256": sha256(bootstrap_dir / "bootstrap.json"),
              "reference": {"path": str(reference_file.resolve()), **ref_info},
              "pool": {"path": str(pool_file.resolve()), **pool_info},
              "raw_truth": {"path": str(ground_truth.resolve()), "sha256": truth_hash.hexdigest(),
                            "source1_rows_scanned": raw_rows, "train_rows_decoded": selected_rows,
                            "nontrain_payloads_decoded": 0},
              "train_pairs": {"path": str(path.resolve()), "rows": links,
                              "durable_path": str((durable / path.name).resolve()),
                              "sha256": sha256(path), "bytes": path.stat().st_size},
              "source_code": {"path": str(code_copy.resolve()), "sha256": sha256(code_copy)},
              "field_representation": "original-script S2/S3 business_name | business_address to clean S1 fields",
              "training_has_started_on_full_pairs": False,
              "retrieval_edges_read_or_modified": False, "positive_candidate_injection": False,
              "holdout2_or_prior_sealed_label_values_decoded": False}
    atomic_json(output / "prepared_full.json", report)
    verified_copy(output / "prepared_full.json", durable / "prepared_full.json")
    return report


def licensed_snapshot(snapshot: Path, inventory_file: Path) -> dict:
    """Verify offline Apache snapshot and the combined v1+denoiser 8B budget."""
    import yaml

    inventory = yaml.safe_load(inventory_file.read_text(encoding="utf-8"))
    entries = {row["id"]: row for row in inventory["models"]}
    ids = (MODEL_ID, "intfloat/multilingual-e5-small", "BAAI/bge-reranker-v2-m3")
    if (any(key not in entries or entries[key]["licence"] not in ("mit", "apache-2.0") for key in ids)
            or entries[MODEL_ID]["licence"] != "apache-2.0"
            or sum(float(entries[key]["params_b"]) for key in ids) > 8):
        raise ValueError("ByT5 plus the existing retrieval/reranker models must be MIT/Apache and <=8B")
    snapshot = snapshot.resolve()
    if ("models--google--byt5-small" not in snapshot.parts or snapshot.parent.name != "snapshots"
            or not (snapshot / "pytorch_model.bin").is_file()
            or "license: apache-2.0" not in (snapshot / "README.md").read_text(encoding="utf-8")):
        raise ValueError("Selected offline ByT5 snapshot has no verified Apache-2.0 model card/weights")
    return {"model_id": MODEL_ID, "license": "apache-2.0", "snapshot": str(snapshot),
            "snapshot_commit": snapshot.name, "inventory_sha256": sha256(inventory_file),
            "model_card_sha256": sha256(snapshot / "README.md"),
            "config_sha256": sha256(snapshot / "config.json"),
            "weights_sha256": sha256(snapshot / "pytorch_model.bin"),
            "inventory_params_b": float(entries[MODEL_ID]["params_b"]),
            "combined_with_e5_and_bge_inventory_params_b": sum(float(entries[key]["params_b"]) for key in ids),
            "max_combined_params_b": float(inventory["budget_total_params_b"])}


def _training_texts(output: Path, medium_dense: Path, prepared: dict) -> tuple[list[str], list[str], pd.DataFrame]:
    ident = prepared["identity"]
    if (sha256(medium_dense / "pool.parquet") != ident["medium"]["dense_pool_sha256"]
            or sha256(medium_dense / "reference.parquet") != ident["medium"]["dense_reference_sha256"]):
        raise ValueError("Saved medium record text changed after TRAIN preparation")
    reference = pd.read_parquet(output / "train_reference.parquet", columns=list(RECORD_FIELDS))
    pool = pd.read_parquet(medium_dense / "pool.parquet", columns=list(RECORD_FIELDS))
    edges = pd.read_parquet(output / "train_pairs.parquet")
    if (len(edges) != ident["positive_links"] or edges.duplicated(["left_idx", "right_idx"]).any()
            or edges.left_idx.min() < 0 or edges.left_idx.max() >= len(reference)
            or edges.right_idx.min() < 0 or edges.right_idx.max() >= len(pool)):
        raise ValueError("TRAIN edge order/indices changed after preparing the split")
    clean = [text_for_record(name, address) for name, address in
             reference[list(TEXT_FIELDS)].itertuples(index=False, name=None)]
    noisy = [text_for_record(name, address) for name, address in
             pool.iloc[edges.right_idx.to_numpy(dtype=np.int32)][list(TEXT_FIELDS)].itertuples(index=False, name=None)]
    if (texts_digest(clean) != ident["train_reference_text_sha256"]
            or texts_digest(noisy) != ident["noisy_target_text_in_pair_order_sha256"]):
        raise ValueError("TRAIN field text identity differs from prepared original-script inputs")
    return clean, noisy, edges


def _tokens(tokenizer, source: list[str], target: list[str], max_length: int, device: str) -> dict:
    inputs = tokenizer(source, max_length=max_length, truncation=True, padding=True,
                       pad_to_multiple_of=8, return_tensors="pt")
    labels = tokenizer(text_target=target, max_length=max_length, truncation=True, padding=True,
                       pad_to_multiple_of=8, return_tensors="pt")["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {"input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device), "labels": labels.to(device)}


def _checkpoint_complete(checkpoint: Path) -> dict:
    marker = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
    if marker.get("state") != "complete" or marker.get("model_id") != MODEL_ID:
        raise ValueError("Denoiser checkpoint is not complete")
    for name, digest in marker["files_sha256"].items():
        if sha256(checkpoint / name) != digest:
            raise ValueError(f"Saved denoiser checkpoint file changed: {name}")
    return marker


def train(output: Path, durable: Path, medium_dense: Path, snapshot: Path, inventory: Path,
          *, steps: int = 200, seed: int = 20260925, microbatch: int = 4,
          accumulation: int = 4, max_length: int = 192, learning_rate: float = 1e-4,
          threads: int = 2, log_every: int = 20) -> dict:
    """Single bounded bf16 screen; store full optimizer/RNG for a later extension."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if (not 200 <= steps <= 1_000 or min(seed, microbatch, accumulation, threads, log_every) < 1
            or threads > 6 or max_length not in (192, 256) or not 0 < learning_rate <= 2e-4):
        raise ValueError("Only a bounded 200..1000-step CUDA screen within CPU/text/lr limits is allowed")
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.mem_get_info()[0] < 16 * 1024**3):
        raise ValueError("ByT5 screen requires an available BF16 A40 with at least 16 GiB free")
    prepared = _prepared(output, durable)
    model_identity = licensed_snapshot(snapshot, inventory)
    clean, noisy, edges = _training_texts(output, medium_dense, prepared)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot, local_files_only=True,
                                                   trust_remote_code=False).to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 280_000_000 <= parameter_count <= 320_000_000 or parameter_count > 8_000_000_000:
        raise ValueError("Loaded ByT5 parameter count differs from licensed small architecture")
    model_identity["loaded_parameter_count"] = parameter_count
    model_identity["combined_with_e5_and_bge_actual_upper_bound_b"] = (
        parameter_count / 1e9 + model_identity["combined_with_e5_and_bge_inventory_params_b"]
        - model_identity["inventory_params_b"])
    if model_identity["combined_with_e5_and_bge_actual_upper_bound_b"] > 8:
        raise ValueError("Final model-family parameter budget would be exceeded")
    config = {"scope": prepared["identity"]["scope"], "hypothesis":
              "TRAIN-only noisy-to-clean ByT5 improves normalized name/full-text recovery on fixed TUNE links",
              "prepared_sha256": sha256(output / "prepared.json"), "source_sha256": sha256(Path(__file__)),
              "model": model_identity, "seed": seed, "optimizer": "AdamW", "learning_rate": learning_rate,
              "lr_schedule": "20_step_linear_warmup_then_constant_for_resumability", "weight_decay": .01,
              "steps": steps, "microbatch": microbatch, "accumulation": accumulation,
              "effective_batch": microbatch * accumulation, "max_source_and_target_bytes": max_length,
              "precision": "fp32_weights_bf16_autocast", "threads": threads,
              "train_pairs": len(edges), "train_references": prepared["train_businesses"],
              "validation_labels_used_for_fitting": False, "retrieval_candidate_edges_modified": False,
              "promotion_gate": {"paired_full_normalized_exact_delta_min": .03,
                                 "paired_95pct_CI_lower_min": 0., "extra_false_exact_full_matches_max": 0}}
    config_path = output / "training_config.json"
    atomic_json(config_path, config)
    verified_copy(config_path, durable / config_path.name)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(edges))
    cursor = 0
    epochs = 0
    left = edges.left_idx.to_numpy(dtype=np.int32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    model.train()
    start = time.monotonic()
    losses: list[dict] = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if step <= 20:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * step / 20
        step_loss = 0.
        for _ in range(accumulation):
            if cursor + microbatch > len(order):
                order = rng.permutation(len(edges))
                cursor = 0
                epochs += 1
            selected = order[cursor:cursor + microbatch]
            cursor += microbatch
            batch = _tokens(tokenizer, [noisy[i] for i in selected],
                            [clean[left[i]] for i in selected], max_length, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss.float() / accumulation
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite denoiser loss at optimizer step {step}")
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            event = {"step": step, "loss": step_loss,
                     "seen_pair_presentations": step * microbatch * accumulation,
                     "seconds": time.monotonic() - start,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
            losses.append(event)
            print(json.dumps({"denoiser_training": event}), flush=True)
        if torch.cuda.max_memory_reserved() > 40_000_000_000:
            raise ValueError("Denoiser alone exceeded the 40 GB scheduling cap")
    torch.cuda.synchronize()
    checkpoint = durable / f"checkpoint_step_{steps:06d}"
    stage = durable / f".checkpoint_step_{steps:06d}.incomplete"
    if checkpoint.exists() or stage.exists():
        raise FileExistsError("Denoiser checkpoint already exists; do not overwrite a previous screen")
    stage.mkdir()
    model.save_pretrained(stage, safe_serialization=True, max_shard_size="1GB")
    tokenizer.save_pretrained(stage)
    torch.save({"optimizer": optimizer.state_dict(), "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": rng.bit_generator.state,
                "pair_epoch": epochs, "pair_cursor": cursor, "pair_order": order}, stage / "optimizer.pt")
    atomic_json(stage / "training_config.json", config)
    files = {file.name: sha256(file) for file in stage.iterdir() if file.is_file()}
    atomic_json(stage / "complete.json", {"state": "complete", "model_id": MODEL_ID,
                                           "step": steps, "loaded_parameter_count": parameter_count,
                                           "training_config_sha256": sha256(config_path), "files_sha256": files})
    stage.replace(checkpoint)
    result = {"state": "complete", "steps": steps, "train_pairs": len(edges),
              "pair_presentations": steps * microbatch * accumulation,
              "sampled_pair_fraction": min(1., steps * microbatch * accumulation / len(edges)),
              "train_seconds_excluding_checkpoint": time.monotonic() - start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "loss_trace": losses, "checkpoint": str(checkpoint.resolve()),
              "checkpoint_manifest_sha256": sha256(checkpoint / "complete.json"),
              "model": model_identity, "training_config_sha256": sha256(config_path)}
    atomic_json(output / "train_result.json", result)
    verified_copy(output / "train_result.json", durable / "train_result.json")
    return result


def extend(base_output: Path, base_durable: Path, output: Path, durable: Path,
           medium_dense: Path, *, steps: int = 1_000, threads: int = 2,
           log_every: int = 50, train_data_output: Path | None = None,
           train_data_durable: Path | None = None) -> dict:
    """Continue a gated medium-TRAIN screen, restoring optimizer and pair RNG."""
    import gc
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not 1 <= threads <= 6 or log_every < 1 or steps not in (1_000, 5_000, 10_000):
        raise ValueError("Continuation must end at the planned 1k, 5k or 10k gate")
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.mem_get_info()[0] < 16 * 1024**3):
        raise ValueError("Do not continue while another GPU job occupies the BF16 A40")
    if (output.exists() and any(output.iterdir())) or (durable.exists() and any(durable.iterdir())):
        raise FileExistsError("Continuation needs fresh denoiser-owned scratch/durable directories")
    data_output = train_data_output or base_output
    data_durable = train_data_durable or base_durable
    prepared = _prepared(data_output, data_durable)
    earlier = json.loads((base_output / "train_result.json").read_text(encoding="utf-8"))
    base_config = json.loads((base_output / "training_config.json").read_text(encoding="utf-8"))
    screen = json.loads((base_output / "report.json").read_text(encoding="utf-8"))
    if (sha256(base_output / "report.json") != sha256(base_durable / "report.json")
            or sha256(base_output / "train_result.json") != sha256(base_durable / "train_result.json")
            or earlier.get("state") != "complete" or screen.get("state") != "complete"
            or screen.get("promotion_gate_met") is not True
            or (earlier["steps"], steps) not in ((200, 1_000), (1_000, 5_000), (5_000, 10_000))
            or not earlier["steps"] < steps or earlier["training_config_sha256"] != sha256(
                base_output / "training_config.json")
            or sha256(base_durable / "v2_denoiser.py") != base_config.get(
                "continuation_source_sha256", base_config["source_sha256"])
            or base_config["prepared_sha256"] != sha256(data_output / "prepared.json")
            or base_config["steps"] != earlier["steps"]):
        raise ValueError("A completed promoted checkpoint and identical medium TRAIN preparation are required")
    source_checkpoint = Path(earlier["checkpoint"])
    previous_marker = _checkpoint_complete(source_checkpoint)
    if (previous_marker["step"] != earlier["steps"]
            or sha256(source_checkpoint / "complete.json") != earlier["checkpoint_manifest_sha256"]
            or previous_marker["training_config_sha256"] != sha256(base_output / "training_config.json")):
        raise ValueError("Promising screen checkpoint changed after its TUNE comparison")
    clean, noisy, edges = _training_texts(data_output, medium_dense, prepared)
    if (base_config["train_pairs"] != len(edges)
            or base_config["model"]["loaded_parameter_count"] != previous_marker["loaded_parameter_count"]):
        raise ValueError("Continued TRAIN data or licensed model no longer match the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    config = {**base_config, "steps": steps, "start_step": earlier["steps"],
              "continuation_source_sha256": sha256(Path(__file__)),
              "initial_training_source_sha256": base_config["source_sha256"],
              "base_training_config_sha256": sha256(base_output / "training_config.json"),
              "base_checkpoint_manifest_sha256": sha256(source_checkpoint / "complete.json"),
              "base_tune_report_sha256": sha256(base_output / "report.json"),
              "train_data_output": str(data_output.resolve()),
              "train_data_durable": str(data_durable.resolve()),
              "fixed_evaluation_sample_sha256": screen["sample_sha256"],
              "extension_reason": "Prior medium-TRAIN canonicalization gate met; full-data fitting and matcher gain remain unmeasured"}
    atomic_json(output / "training_config.json", config)
    verified_copy(output / "training_config.json", durable / "training_config.json")
    verified_copy(Path(__file__), durable / "v2_denoiser.py")
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(source_checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(source_checkpoint, local_files_only=True,
                                                   trust_remote_code=False).to("cuda")
    if sum(p.numel() for p in model.parameters()) != previous_marker["loaded_parameter_count"]:
        raise ValueError("Continued checkpoint loaded a different number of parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_config["learning_rate"],
                                   weight_decay=base_config["weight_decay"])
    # This checkpoint was generated exclusively by this project and SHA-verified.
    saved = torch.load(source_checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    if abs(optimizer.param_groups[0]["lr"] - base_config["learning_rate"]) > 1e-12:
        raise ValueError("Continuation optimizer changed the screen's learning-rate schedule")
    rng = np.random.default_rng()
    rng.bit_generator.state = saved["numpy_rng"]
    order = np.asarray(saved["pair_order"], dtype=np.int64)
    if (order.shape != (len(edges),) or not np.array_equal(np.sort(order), np.arange(len(edges)))
            or saved["pair_epoch"] < 0 or not 0 <= saved["pair_cursor"] <= len(edges)):
        raise ValueError("Checkpoint sampler lost its without-replacement TRAIN pair order")
    cursor, epochs = int(saved["pair_cursor"]), int(saved["pair_epoch"])
    torch.set_rng_state(saved["torch_cpu_rng"])
    torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    del saved
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    microbatch = base_config["microbatch"]
    accumulation = base_config["accumulation"]
    max_length = base_config["max_source_and_target_bytes"]
    left = edges.left_idx.to_numpy(dtype=np.int32)
    start = time.monotonic()
    losses = []
    for step in range(earlier["steps"] + 1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.
        for _ in range(accumulation):
            if cursor + microbatch > len(order):
                order = rng.permutation(len(edges))
                cursor = 0
                epochs += 1
            selected = order[cursor:cursor + microbatch]
            cursor += microbatch
            batch = _tokens(tokenizer, [noisy[i] for i in selected],
                            [clean[left[i]] for i in selected], max_length, "cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss.float() / accumulation
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite resumed denoiser loss at step {step}")
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step == earlier["steps"] + 1 or step % log_every == 0 or step == steps):
            event = {"step": step, "loss": step_loss,
                     "total_pair_presentations": step * microbatch * accumulation,
                     "extension_seconds": time.monotonic() - start,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}
            losses.append(event)
            print(json.dumps({"denoiser_extension": event}), flush=True)
        if torch.cuda.max_memory_reserved() > 40_000_000_000:
            raise ValueError("Denoiser continuation exceeded the 40 GB scheduling cap")
    torch.cuda.synchronize()
    checkpoint = durable / f"checkpoint_step_{steps:06d}"
    stage = durable / f".checkpoint_step_{steps:06d}.incomplete"
    if stage.exists() or checkpoint.exists():
        raise FileExistsError("Continuation checkpoint already exists")
    stage.mkdir()
    model.save_pretrained(stage, safe_serialization=True, max_shard_size="1GB")
    tokenizer.save_pretrained(stage)
    torch.save({"optimizer": optimizer.state_dict(), "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": rng.bit_generator.state,
                "pair_epoch": epochs, "pair_cursor": cursor, "pair_order": order}, stage / "optimizer.pt")
    atomic_json(stage / "training_config.json", config)
    atomic_json(stage / "complete.json", {"state": "complete", "model_id": MODEL_ID,
                                           "step": steps, "loaded_parameter_count": previous_marker["loaded_parameter_count"],
                                           "training_config_sha256": sha256(output / "training_config.json"),
                                           "files_sha256": {file.name: sha256(file) for file in stage.iterdir()
                                                            if file.is_file()}})
    stage.replace(checkpoint)
    result = {"state": "complete", "steps": steps, "start_step": earlier["steps"],
              "train_pairs": len(edges), "pair_presentations": steps * microbatch * accumulation,
              "additional_pair_presentations": (steps - earlier["steps"]) * microbatch * accumulation,
              "sampled_pair_fraction": min(1., steps * microbatch * accumulation / len(edges)),
              "train_seconds_excluding_checkpoint": time.monotonic() - start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "loss_trace": losses, "checkpoint": str(checkpoint.resolve()),
              "checkpoint_manifest_sha256": sha256(checkpoint / "complete.json"),
              "base_checkpoint_manifest_sha256": sha256(source_checkpoint / "complete.json"),
              "model": base_config["model"], "training_config_sha256": sha256(output / "training_config.json")}
    atomic_json(output / "train_result.json", result)
    verified_copy(output / "train_result.json", durable / "train_result.json")
    return result


def safe_canonical_text(original: str, generated: str, *, input_truncated: bool = False,
                        output_truncated: bool = False) -> tuple[str, str]:
    """Reject format/digit hallucination; never let denoising change a door number."""
    if input_truncated:
        return original, "input_truncated"
    if output_truncated:
        return original, "output_truncated"
    if generated.count(" | ") != 1:
        return original, "invalid_delimiter"
    name, address = generated.split(" | ", 1)
    if not normalize_text(name):
        return original, "empty_name"
    if _DIGITS.findall(address) != _DIGITS.findall(original.split(" | ", 1)[1]):
        return original, "address_digits_changed"
    return generated, "accepted"


def _fixed_union(union: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Verify *every* candidate before TUNE truth is first opened."""
    report = json.loads((union / "complete.json").read_text(encoding="utf-8"))
    raw_path = union / "candidate_pairs_raw.parquet"
    pair_path = union / "candidate_pairs.parquet"
    if (report.get("state") != "complete" or report.get("references") != 12_000
            or report.get("reference_ids_sha256") != TUNE_ID_SHA
            or report.get("full_target_pool_rows") != FULL_POOL_ROWS
            or report.get("positive_candidate_injection") is not False
            or report.get("truth_used_in_retrieval") is not False
            or sha256(raw_path) != report["raw_pairs"]["sha256"]
            or pq.ParquetFile(raw_path).metadata.num_rows != report["raw_pairs"]["rows"]):
        raise ValueError("TUNE union must be complete, label-free and from the full 10.32M pool")
    reference = pd.read_parquet(union / "reference.parquet", columns=[*RECORD_FIELDS, "split"])
    pool = pd.read_parquet(union / "pool.parquet", columns=list(RECORD_FIELDS))
    if (len(reference) != 12_000 or ids_sha(reference.entity_id) != TUNE_ID_SHA
            or not reference.split.eq("tune").all() or reference.entity_id.duplicated().any()
            or pool.entity_id.duplicated().any()):
        raise ValueError("TUNE reference/compact target records differ from the fixed population")
    pair_cols = ["left_idx", "right_idx", "target_entity_id", "right_raw_idx", "dense_hit", "lexical_hit",
                 "original_dense_score"]
    pairs = pd.read_parquet(pair_path, columns=pair_cols)
    raw = pd.read_parquet(raw_path, columns=["left_idx", "right_raw_idx", "target_entity_id"])
    if (len(pairs) != len(raw) or pairs.duplicated(["left_idx", "right_idx"]).any()
            or not np.array_equal(pairs[["left_idx", "right_raw_idx", "target_entity_id"]].to_numpy(),
                                   raw.to_numpy()) or pairs.left_idx.min() < 0
            or pairs.left_idx.max() >= len(reference) or pairs.right_idx.min() < 0
            or pairs.right_idx.max() >= len(pool)
            or not np.array_equal(pool.entity_id.to_numpy()[pairs.right_idx.to_numpy(dtype=np.int64)],
                                   pairs.target_entity_id.to_numpy())):
        raise ValueError("TUNE candidate pair order/IDs changed or a duplicate edge appeared")
    # Ground truth exists only in this completed 12k-TUNE diagnostic bundle.
    raw_truth = json.loads((union / "truth.json").read_text(encoding="utf-8"))
    if set(raw_truth) != set(reference.entity_id):
        raise ValueError("Saved union TUNE truth does not cover all 12,000 references")
    truth = {entity: set(values) for entity, values in raw_truth.items()}
    if sum(map(len, truth.values())) != len(set().union(*truth.values())):
        raise ValueError("TUNE truth is not one-to-one across businesses")
    identity = {"complete_sha256": sha256(union / "complete.json"),
                "reference_file_sha256": sha256(union / "reference.parquet"),
                "pool_file_sha256": sha256(union / "pool.parquet"),
                "candidate_raw_sha256": report["raw_pairs"]["sha256"],
                "candidate_pairs_sha256": sha256(pair_path),
                "tune_truth_sha256": sha256(union / "truth.json"),
                "reference_text_sha256": texts_digest(text_for_record(a, b) for a, b in
                                                       reference[list(TEXT_FIELDS)].itertuples(index=False, name=None)),
                "pool_text_sha256": texts_digest(text_for_record(a, b) for a, b in
                                                  pool[list(TEXT_FIELDS)].itertuples(index=False, name=None))}
    return reference, pool, pairs, truth, report, identity


def retrieval_ceiling(reference: pd.DataFrame, pairs: pd.DataFrame, truth: dict,
                      completed: dict) -> tuple[dict, np.ndarray]:
    """Recompute the 12k oracle incl. every miss and singleton; no new edges."""
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    ids = reference.entity_id.to_numpy()
    labels = np.fromiter((target in truth[ids[int(i)]] for i, target in
                          zip(left, pairs.target_entity_id)), dtype=bool, count=len(pairs))
    actual = np.asarray([len(truth[entity]) for entity in ids], dtype=np.int32)
    result = {}
    for name, mask in (("dense_k50_saved", pairs.dense_hit.to_numpy(dtype=bool)),
                       ("full_key_union", np.ones(len(pairs), dtype=bool))):
        found = np.bincount(left[mask & labels], minlength=len(reference))
        denominator = 4 * found + actual
        oracle = np.divide(5. * found, denominator, out=np.zeros(len(reference)), where=denominator > 0)
        oracle[actual == 0] = 1.
        count = int(mask.sum())
        result[name] = {"references": len(reference), "pairs": count,
                        "true_links": int(actual.sum()), "retrieved_true_links": int(found.sum()),
                        "missed_true_links": int(actual.sum() - found.sum()),
                        "link_recall": float(found.sum() / actual.sum()),
                        "oracle_macro_f05": float(oracle.mean()), "true_singletons": int((actual == 0).sum()),
                        "by_country": {country: {"references": int((reference.country == country).sum()),
                                                 "oracle_macro_f05": float(oracle[reference.country.eq(country)].mean()),
                                                 "true_links": int(actual[reference.country.eq(country)].sum()),
                                                 "retrieved_true_links": int(found[reference.country.eq(country)].sum())}
                                       for country in sorted(reference.country.unique())}}
        expected = completed["retrieval"][name]
        for key in ("pairs", "true_links", "retrieved_true_links", "missed_true_links", "true_singletons"):
            if result[name][key] != expected[key]:
                raise ValueError(f"Completed full-pool {name} retrieval accounting changed: {key}")
        if abs(result[name]["oracle_macro_f05"] - expected["oracle_macro_f05"]) > 1e-12:
            raise ValueError("The 12k retrieval oracle no longer reproduces the saved full-pool result")
    result["full_key_union"]["target_name_script"] = completed["retrieval"]["full_key_union"]["target_name_script"]
    return result, labels


def _probe_sample(reference: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame,
                  truth: dict, *, seed: int, n_refs: int) -> pd.DataFrame:
    """One positive and one candidate hard negative per sampled non-singleton S1."""
    rng = np.random.default_rng(seed)
    eligible = np.array([i for i, entity in enumerate(reference.entity_id) if truth[entity]], dtype=np.int32)
    chosen = np.sort(rng.choice(eligible, size=min(n_refs, len(eligible)), replace=False))
    pool_index = pd.Index(pool.entity_id)
    lefts = pairs.left_idx.to_numpy(dtype=np.int32)
    # Candidate file is sorted by reference in the saved union; verify before slicing.
    if np.any(lefts[1:] < lefts[:-1]):
        raise ValueError("Saved candidate order is not grouped by S1")
    starts = np.searchsorted(lefts, np.arange(len(reference) + 1), side="left")
    packed_candidates = np.sort((lefts.astype(np.uint64) << np.uint64(32))
                                | pairs.right_idx.to_numpy(dtype=np.uint64))
    rows = []
    for position in chosen:
        entity = reference.entity_id.iat[int(position)]
        target = str(rng.choice(sorted(truth[entity])))
        right = int(pool_index.get_indexer([target])[0])
        if right < 0:
            raise ValueError("An unretrieved true TUNE target record was lost from the compact pool")
        key = (np.uint64(position) << np.uint64(32)) | np.uint64(right)
        where = np.searchsorted(packed_candidates, key)
        retrieved = bool(where < len(packed_candidates) and packed_candidates[where] == key)

        def append(right_idx: int, label: bool, candidate: bool) -> None:
            record = pool.iloc[right_idx]
            if record.country != reference.country.iat[int(position)]:
                raise ValueError("A diagnostic match/candidate crossed a country block")
            rows.append({"left_idx": int(position), "right_idx": right_idx,
                         "source1_entity_id": entity, "target_entity_id": record.entity_id,
                         "country": record.country, "target_name_script": _script(record.business_name),
                         "true_link": bool(label), "candidate_edge": bool(candidate)})

        append(right, True, retrieved)  # A miss is a *diagnostic record*, not a new candidate edge.
        query_name = normalize_text(reference.business_name.iat[int(position)])
        query_digits = _DIGITS.findall(reference.business_address.iat[int(position)])
        hardest = None
        for row in range(starts[position], starts[position + 1]):
            other = pairs.iloc[row]
            if other.target_entity_id in truth[entity]:
                continue
            target_row = pool.iloc[int(other.right_idx)]
            rank = (int(normalize_text(target_row.business_name) == query_name),
                    int(bool(query_digits) and _DIGITS.findall(target_row.business_address) == query_digits),
                    float(other.original_dense_score), -int(other.right_idx))
            if hardest is None or rank > hardest[0]:
                hardest = (rank, int(other.right_idx))
        if hardest is not None:
            append(hardest[1], False, True)
    sample = pd.DataFrame(rows)
    if (sample.empty or sample.duplicated(["left_idx", "right_idx"]).any()
            or int(sample.true_link.sum()) != len(chosen)
            or not sample.loc[~sample.true_link, "candidate_edge"].all()):
        raise ValueError("Diagnostic sample lost a reference, repeated a pair or invented a negative")
    return sample


def _generate(model, tokenizer, inputs: list[str], *, batch_size: int, max_length: int,
              progress_every: int | None = None) -> list[dict]:
    import torch

    records = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            original_lengths = [len(tokenizer.encode(item)) for item in batch]
            tokens = tokenizer(batch, max_length=max_length, truncation=True, padding=True,
                               pad_to_multiple_of=8, return_tensors="pt")
            tokens = {key: value.to("cuda") for key, value in tokens.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated = model.generate(**tokens, do_sample=False, num_beams=1,
                                           max_new_tokens=max_length, return_dict_in_generate=True,
                                           output_scores=True)
            decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
            logprobs = model.compute_transition_scores(generated.sequences, generated.scores,
                                                       normalize_logits=True).float().cpu().numpy()
            seq = generated.sequences[:, 1:].cpu().numpy()
            for i, raw in enumerate(batch):
                count = int(np.count_nonzero(seq[i] != tokenizer.pad_token_id))
                raw_generated = decoded[i]
                ended = tokenizer.eos_token_id in seq[i].tolist()
                canonical, reason = safe_canonical_text(raw, raw_generated,
                                                        input_truncated=original_lengths[i] > max_length,
                                                        output_truncated=not ended)
                records.append({"input": raw, "generated": raw_generated, "canonical": canonical,
                                "fallback_reason": reason, "input_token_length": original_lengths[i],
                                "output_token_length": count,
                                "generation_mean_logprob": float(logprobs[i, :count].sum() / max(1, count))})
            if progress_every and (len(records) // progress_every != start // progress_every
                                   or len(records) == len(inputs)):
                print(json.dumps({"canonical_targets_done": len(records), "total": len(inputs)}), flush=True)
    return records


def _component_metrics(frame: pd.DataFrame, reference: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Fixed deterministic exact-match feature; generator log-prob is not a match probability."""
    computed = frame.copy()
    lookup = {"raw_name": [], "new_name": [], "raw_full": [], "new_full": [],
              "raw_name_digit": [], "new_name_digit": []}
    for row in computed.itertuples(index=False):
        ref, target = reference.iloc[row.left_idx], pool.iloc[row.right_idx]
        qname, qaddr = normalize_text(ref.business_name), normalize_text(ref.business_address)
        raw_name, raw_addr = normalize_text(target.business_name), normalize_text(target.business_address)
        new_name, new_addr = row.canonical.split(" | ", 1)
        new_name, new_addr = normalize_text(new_name), normalize_text(new_addr)
        digit_agreement = bool(_DIGITS.findall(ref.business_address)) and (
            _DIGITS.findall(ref.business_address) == _DIGITS.findall(target.business_address))
        lookup["raw_name"].append(bool(qname and qname == raw_name))
        lookup["new_name"].append(bool(qname and qname == new_name))
        lookup["raw_full"].append(bool(qname and qaddr and qname == raw_name and qaddr == raw_addr))
        lookup["new_full"].append(bool(qname and qaddr and qname == new_name and qaddr == new_addr))
        lookup["raw_name_digit"].append(bool(qname and qname == raw_name and digit_agreement))
        lookup["new_name_digit"].append(bool(qname and qname == new_name and digit_agreement))
    for column, values in lookup.items():
        computed[column] = values
    return computed


def _canonical_report(probe: pd.DataFrame, reference: pd.DataFrame, seed: int) -> dict:
    positives = probe.loc[probe.true_link].sort_values("left_idx")
    negatives = probe.loc[~probe.true_link]
    if positives.left_idx.duplicated().any():
        raise ValueError("Paired screen must use exactly one true link per S1")

    def rates(part: pd.DataFrame) -> dict:
        if part.empty:
            return {"pairs": 0, "references": 0, "name": None, "full": None,
                    "name_digit": None, "fallback_reasons": {}}
        return {"pairs": len(part), "references": part.left_idx.nunique(),
                **{name: {"raw": float(part[f"raw_{name}"].mean()),
                          "denoised": float(part[f"new_{name}"].mean()),
                          "delta": float((part[f"new_{name}"].astype(float)
                                          - part[f"raw_{name}"].astype(float)).mean()),
                          "raw_hits": int(part[f"raw_{name}"].sum()),
                          "denoised_hits": int(part[f"new_{name}"].sum())}
                   for name in ("name", "full", "name_digit")},
                "fallback_reasons": dict(Counter(part.fallback_reason))}

    pos_rate = rates(positives)
    neg_rate = rates(negatives) if len(negatives) else None
    result = {"sampled_true_links_including_retrieval_misses": pos_rate,
              "sampled_retrieved_hard_nonlinks": neg_rate,
              "sampled_retrieval_misses": int((~positives.candidate_edge).sum()),
              "by_country": {country: {"positive": rates(positives.loc[positives.country.eq(country)]),
                                       "negative": rates(negatives.loc[negatives.country.eq(country)])}
                             for country in sorted(probe.country.unique())},
              "by_target_name_script": {script: {"positive": rates(positives.loc[
                  positives.target_name_script.eq(script)]), "negative": rates(negatives.loc[
                      negatives.target_name_script.eq(script)])}
                                        for script in sorted(probe.target_name_script.unique())}}
    for field in ("name", "full", "name_digit"):
        result[f"paired_positive_{field}"] = paired_bootstrap(
            positives[f"new_{field}"].astype(float), positives[f"raw_{field}"].astype(float),
            n_bootstrap=10_000, seed=seed)
    return result


def paired_extension(probe: pd.DataFrame, previous: pd.DataFrame, *, seed: int) -> dict:
    """Compare checkpoints on identical diagnostic S1s/pairs, never pair-bootstrap."""
    aligned = ("left_idx", "right_idx", "source1_entity_id", "target_entity_id",
               "country", "target_name_script", "true_link", "candidate_edge",
               "raw_name", "raw_full", "raw_name_digit")
    if (len(probe) != len(previous)
            or any(not np.array_equal(probe[field].to_numpy(), previous[field].to_numpy()) for field in aligned)
            or probe.loc[probe.true_link, "left_idx"].duplicated().any()):
        raise ValueError("Checkpoint comparison needs the same labeled pair and one positive per S1")
    positive = probe.true_link.to_numpy(dtype=bool)
    npositive = int(positive.sum())
    if npositive < 2:
        raise ValueError("Paired continuation needs at least two different TUNE businesses")
    fields = {}
    for name in ("name", "full", "name_digit"):
        newer = probe.loc[positive, f"new_{name}"].astype(float).to_numpy()
        older = previous.loc[positive, f"new_{name}"].astype(float).to_numpy()
        fields[name] = {"new_hits": int(newer.sum()), "old_hits": int(older.sum()),
                        "paired": paired_bootstrap(newer, older, n_bootstrap=10_000, seed=seed)}
    false_new = int(probe.loc[~positive, "new_full"].sum())
    false_old = int(previous.loc[~positive, "new_full"].sum())
    improved = (fields["full"]["paired"]["confidence_intervals"]["left_minus_right"][0] > 0
                and false_new <= false_old)

    def breakdown(column: str) -> dict:
        return {str(label): {"positive_pairs": int((selected & positive).sum()),
                             "new_full_positive_hits": int(probe.loc[selected & positive, "new_full"].sum()),
                             "old_full_positive_hits": int(previous.loc[selected & positive, "new_full"].sum()),
                             "new_full_false_hits": int(probe.loc[selected & ~positive, "new_full"].sum()),
                             "old_full_false_hits": int(previous.loc[selected & ~positive, "new_full"].sum())}
                for label in sorted(set(probe[column]))
                for selected in [probe[column].eq(label).to_numpy()]}

    return {"scope": "same frozen medium-TUNE full-pool retrieval and identical sampled S1/target pairs",
            "positive_references": npositive, "sample_pairs": len(probe), "positive_exact": fields,
            "negative_full_exact_false_hits": {"new": false_new, "old": false_old},
            "by_country": breakdown("country"), "by_target_name_script": breakdown("target_name_script"),
            "new_checkpoint_wins_screen_gate": bool(improved),
            "limitation": "Repeated selection on the same sampled development TUNE S1s; not a new holdout or a matcher F0.5."}


def align_frozen_dense(v1_reference: pd.DataFrame, v1_pool: pd.DataFrame,
                       v1_pairs: pd.DataFrame, v1_prob: np.ndarray,
                       union_reference: pd.DataFrame, union_pool: pd.DataFrame,
                       union_pairs: pd.DataFrame, *, dense_k: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place all 600k frozen dense TUNE scores on the 1.44M-pair union by original ID.

    Unscored union-only edges remain NaN; no score is manufactured from an ID,
    rank, label, or a model trained on different candidate-dependent features.
    Returns aligned union scores and the v1/union positional indices of TUNE
    dense edges in *original v1 pair order*.
    """
    if (dense_k < 1 or len(v1_pairs) != len(v1_prob) or v1_reference.entity_id.duplicated().any()
            or v1_pool.entity_id.duplicated().any() or union_reference.entity_id.duplicated().any()
            or union_pool.entity_id.duplicated().any() or union_pairs.duplicated(["left_idx", "right_idx"]).any()
            or v1_pairs.duplicated(["left_idx", "right_idx"]).any()
            or not np.isfinite(v1_prob).all()
            or v1_pairs.left_idx.min() < 0 or v1_pairs.left_idx.max() >= len(v1_reference)
            or v1_pairs.right_idx.min() < 0 or v1_pairs.right_idx.max() >= len(v1_pool)):
        raise ValueError("Frozen v1 scores/pairs, references or original IDs are incomplete/duplicated")
    union_left = pd.Index(union_reference.entity_id).get_indexer(v1_reference.entity_id)
    selected = union_left[v1_pairs.left_idx.to_numpy(dtype=np.int64)] >= 0
    positions = np.flatnonzero(selected)
    left = union_left[v1_pairs.left_idx.to_numpy(dtype=np.int64)[positions]]
    target_id = v1_pool.entity_id.to_numpy()[v1_pairs.right_idx.to_numpy(dtype=np.int64)[positions]]
    right = pd.Index(union_pool.entity_id).get_indexer(target_id)
    if (len(positions) != len(union_reference) * dense_k or np.any(right < 0)
            or not np.array_equal(v1_reference.country.to_numpy()[
                v1_pairs.left_idx.to_numpy(dtype=np.int64)[positions]],
                union_reference.country.to_numpy()[left])
            or not np.array_equal(v1_pool.country.to_numpy()[
                v1_pairs.right_idx.to_numpy(dtype=np.int64)[positions]],
                union_pool.country.to_numpy()[right])):
        raise ValueError("Frozen v1 dense TUNE candidates cannot map one-to-one onto the exact union S1/pool")
    for field in TEXT_FIELDS:
        if (not np.array_equal(v1_reference[field].to_numpy()[
                v1_pairs.left_idx.to_numpy(dtype=np.int64)[positions]],
                union_reference[field].to_numpy()[left])
                or not np.array_equal(v1_pool[field].to_numpy()[
                    v1_pairs.right_idx.to_numpy(dtype=np.int64)[positions]],
                    union_pool[field].to_numpy()[right])):
            raise ValueError(f"Frozen dense and union original-script {field} differ")
    keys = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
    dense_rows = np.flatnonzero(union_pairs.dense_hit.to_numpy(dtype=bool))
    if len(dense_rows) != len(positions):
        raise ValueError("Union does not contain exactly the original dense K50 candidates")
    union_keys = ((union_pairs.left_idx.to_numpy(dtype=np.uint64)[dense_rows] << np.uint64(32))
                  | union_pairs.right_idx.to_numpy(dtype=np.uint64)[dense_rows])
    order = np.argsort(union_keys, kind="stable")
    ordered = union_keys[order]
    where = np.searchsorted(ordered, keys)
    if (np.any(where >= len(ordered)) or not np.array_equal(ordered[where], keys)
            or len(np.unique(keys)) != len(keys) or len(np.unique(union_keys)) != len(union_keys)
            or len(np.unique(where)) != len(keys)):
        raise ValueError("Union and frozen v1 dense candidate pair keys differ")
    mapped = dense_rows[order[where]]
    if not np.array_equal(union_pool.entity_id.to_numpy()[right], target_id):
        raise ValueError("Frozen v1 target IDs differ from the original full-pool union")
    aligned = np.full(len(union_pairs), np.nan, dtype=np.float32)
    aligned[mapped] = np.asarray(v1_prob[positions], dtype=np.float32)
    if (not np.isfinite(aligned[dense_rows]).all() or np.isfinite(aligned).sum() != len(positions)
            or not np.isnan(aligned[~union_pairs.dense_hit.to_numpy(dtype=bool)]).all()):
        raise ValueError("An unscored union edge received a fabricated frozen-v1 probability")
    return aligned, positions, mapped


def prepare_feature_band(union: Path, medium_dense: Path, frozen_scores: Path,
                         output: Path, durable: Path, *, seed: int = 20260925,
                         train_sample: int = 8_000, tune_low: float = .50,
                         tune_high: float = .95, tree_low: float = .50,
                         tree_high: float = .99, strict: bool = True) -> dict:
    """Fix label-free TRAIN/TUNE candidate-limited generation before any truth join."""
    if (seed < 1 or train_sample < 1 or not 0 <= tune_low < .775 < tune_high < 1
            or not 0 <= tree_low < tree_high < 1):
        raise ValueError("A fixed score band around the v1 threshold and positive TRAIN cap are required")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    output, durable = Path(output), Path(durable)
    if (output.exists() and any(output.iterdir())) or (durable.exists() and any(durable.iterdir())):
        raise FileExistsError("Candidate-limited feature selection already has output")
    union, medium_dense, frozen_scores = Path(union), Path(medium_dense), Path(frozen_scores)
    union_complete = json.loads((union / "complete.json").read_text(encoding="utf-8"))
    v1_config = json.loads((frozen_scores / "experiment_config.json").read_text(encoding="utf-8"))
    chosen = json.loads((frozen_scores / "selected_model.json").read_text(encoding="utf-8"))
    if (union_complete.get("state") != "complete" or union_complete.get("positive_candidate_injection") is not False
            or union_complete.get("truth_used_in_retrieval") is not False
            or union_complete.get("full_target_pool_rows") != FULL_POOL_ROWS
            or (strict and union_complete["raw_pairs"]["rows"] != 1_437_538)
            or chosen.get("threshold") != .775 or chosen.get("cross_encoder_weight") != .75
            or v1_config.get("full_pool_records") != FULL_POOL_ROWS
            or v1_config.get("tune_only_inference") is not True
            or sha256(union / "candidate_pairs_raw.parquet") != union_complete["raw_pairs"]["sha256"]):
        raise ValueError("Frozen v1/TUNE union provenance, score scope or no-truth retrieval contract changed")
    union_reference = pd.read_parquet(union / "reference.parquet", columns=[*RECORD_FIELDS, "split"])
    union_pool = pd.read_parquet(union / "pool.parquet", columns=list(RECORD_FIELDS))
    union_pairs = pd.read_parquet(union / "candidate_pairs.parquet",
                                  columns=["left_idx", "right_idx", "target_entity_id", "dense_hit"])
    if (len(union_reference) != 12_000 or ids_sha(union_reference.entity_id) != TUNE_ID_SHA
            or not union_reference.split.eq("tune").all()
            or len(union_pairs) != union_complete["raw_pairs"]["rows"]
            or union_pairs.duplicated(["left_idx", "right_idx"]).any()
            or union_pairs.left_idx.min() < 0 or union_pairs.left_idx.max() >= len(union_reference)
            or union_pairs.right_idx.min() < 0 or union_pairs.right_idx.max() >= len(union_pool)
            or not np.array_equal(union_pool.entity_id.to_numpy()[union_pairs.right_idx.to_numpy(dtype=np.int64)],
                                   union_pairs.target_entity_id.to_numpy())):
        raise ValueError("The label-free full union lost original S1/target IDs or pair uniqueness")
    v1_reference = pd.read_parquet(medium_dense / "reference.parquet", columns=[*RECORD_FIELDS, "split"])
    v1_pool = pd.read_parquet(medium_dense / "pool.parquet", columns=list(RECORD_FIELDS))
    v1_pairs = pd.read_parquet(frozen_scores / "pairs.parquet", columns=["left_idx", "right_idx"])
    scores = np.load(frozen_scores / "probabilities.npy", mmap_mode="r")
    tree = np.load(frozen_scores / "base_probabilities.npy", mmap_mode="r")
    if (len(v1_reference) != 47_999 or ids_sha(v1_reference.loc[v1_reference.split.eq("train"), "entity_id"])
            != TRAIN_ID_SHA or ids_sha(v1_reference.loc[v1_reference.split.eq("tune"), "entity_id"]) != TUNE_ID_SHA
            or len(tree) != len(scores) or len(v1_pairs) != len(scores)
            or not np.isfinite(tree).all() or not np.isfinite(scores).all()):
        raise ValueError("Frozen v1 35,999 TRAIN and 12,000 TUNE pair probabilities are misaligned")
    aligned, v1_tune_rows, union_tune_rows = align_frozen_dense(
        v1_reference, v1_pool, v1_pairs, scores, union_reference, union_pool, union_pairs)
    if np.any(np.asarray(scores)[v1_reference.split.to_numpy()[
            v1_pairs.left_idx.to_numpy(dtype=np.int64)] == "train"] != 0):
        raise ValueError("v1 CE TRAIN scores were expected to be absent, never used as labels/scores")
    left = v1_pairs.left_idx.to_numpy(dtype=np.int32)
    right = v1_pairs.right_idx.to_numpy(dtype=np.int32)
    train_pool = np.flatnonzero((v1_reference.split.to_numpy()[left] == "train")
                                & (tree >= tree_low) & (tree < tree_high))
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(train_pool, min(train_sample, len(train_pool)), replace=False))
    tune_selected = union_tune_rows[(aligned[union_tune_rows] >= tune_low)
                                    & (aligned[union_tune_rows] < tune_high)]
    if (len(sampled) < 100 or len(tune_selected) < 100 or not np.array_equal(
            v1_pool.entity_id.to_numpy()[right[v1_tune_rows]],
            union_pool.entity_id.to_numpy()[union_pairs.right_idx.to_numpy(dtype=np.int32)[union_tune_rows]])):
        raise ValueError("Bounded score bands lost candidate coverage or ID alignment")
    tune_to_v1 = np.full(len(union_pairs), -1, dtype=np.int32)
    tune_to_v1[union_tune_rows] = v1_tune_rows.astype(np.int32)
    selected_union_v1 = tune_to_v1[tune_selected]
    train_rows = pd.DataFrame({"split": "train", "v1_pair_position": sampled.astype(np.int32),
                               "v1_left_idx": left[sampled], "v1_right_idx": right[sampled],
                               "union_pair_position": -1,
                               "source1_entity_id": v1_reference.entity_id.to_numpy()[left[sampled]],
                               "target_entity_id": v1_pool.entity_id.to_numpy()[right[sampled]],
                               "tree_probability": np.asarray(tree[sampled], dtype=np.float32),
                               "v1_blend_probability": np.full(len(sampled), np.nan, dtype=np.float32)})
    tune_rows = pd.DataFrame({"split": "tune", "v1_pair_position": selected_union_v1,
                              "v1_left_idx": left[selected_union_v1], "v1_right_idx": right[selected_union_v1],
                              "union_pair_position": tune_selected.astype(np.int32),
                              "source1_entity_id": v1_reference.entity_id.to_numpy()[left[selected_union_v1]],
                              "target_entity_id": v1_pool.entity_id.to_numpy()[right[selected_union_v1]],
                              "tree_probability": np.asarray(tree[selected_union_v1], dtype=np.float32),
                              "v1_blend_probability": aligned[tune_selected]})
    selected_pairs = pd.concat((train_rows, tune_rows), ignore_index=True)
    if (selected_pairs.duplicated(["split", "source1_entity_id", "target_entity_id"]).any()
            or not np.array_equal(selected_pairs.target_entity_id.to_numpy(),
                                   v1_pool.entity_id.to_numpy()[selected_pairs.v1_right_idx.to_numpy(dtype=np.int32)])):
        raise ValueError("Candidate-limited selection duplicated or remapped an original target")
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    np.save(output / "aligned_v1_probabilities.npy", aligned)
    selected_pairs.to_parquet(output / "selected_pairs.parquet", index=False)
    for name in ("aligned_v1_probabilities.npy", "selected_pairs.parquet"):
        verified_copy(output / name, durable / name)
    verified_copy(Path(__file__), durable / "v2_denoiser_band.py")
    report = {"state": "complete", "scope": "same 12k medium TUNE S1s, 1,437,538 union edges from 10,320,219 targets",
              "v1_scored_dense_pairs": len(v1_tune_rows), "unscored_union_only_pairs": int(np.isnan(aligned).sum()),
              "medium_train_tree_band": [tree_low, tree_high], "tune_blend_band": [tune_low, tune_high],
              "medium_train_selected_pairs": len(train_rows), "tune_selected_pairs": len(tune_rows),
              "seed": seed, "frozen_v1_threshold": .775,
              "train_v1_CE_probabilities_available": False,
              "full_pool_rows": FULL_POOL_ROWS, "tune_ids_sha256": TUNE_ID_SHA,
              "train_ids_sha256": TRAIN_ID_SHA,
              "source_code_sha256": sha256(Path(__file__)),
              "inputs": {"union_complete_sha256": sha256(union / "complete.json"),
                         "union_pairs_sha256": sha256(union / "candidate_pairs.parquet"),
                         "union_reference_sha256": sha256(union / "reference.parquet"),
                         "union_pool_sha256": sha256(union / "pool.parquet"),
                         "v1_pairs_sha256": sha256(frozen_scores / "pairs.parquet"),
                         "v1_scores_sha256": sha256(frozen_scores / "probabilities.npy"),
                         "v1_tree_scores_sha256": sha256(frozen_scores / "base_probabilities.npy"),
                         "v1_reference_sha256": sha256(medium_dense / "reference.parquet"),
                         "v1_pool_sha256": sha256(medium_dense / "pool.parquet")},
              "files": {name: sha256(output / name) for name in
                        ("aligned_v1_probabilities.npy", "selected_pairs.parquet")},
              "truth_used_for_selection_or_alignment": False, "positive_candidate_injection": False,
              "matcher_scores_on_unscored_union_edges": None,
              "warning": "TRAIN blend scores were never saved; TRAIN calibration needs frozen-CE rescoring or a tree-band conditional precision rule."}
    atomic_json(output / "feature_band.json", report)
    verified_copy(output / "feature_band.json", durable / "feature_band.json")
    return report


def generate_feature_band(band_dir: Path, medium_dense: Path, checkpoint: Path,
                          output: Path, durable: Path, *, batch_size: int = 16,
                          threads: int = 2) -> dict:
    """Canonicalize only label-free, score-selected original S2/S3 records."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    band_dir, medium_dense, checkpoint = map(Path, (band_dir, medium_dense, checkpoint))
    output, durable = Path(output), Path(durable)
    if (not 1 <= batch_size <= 32 or not 1 <= threads <= 6
            or (output.exists() and any(output.iterdir()))
            or (durable.exists() and any(durable.iterdir()))):
        raise ValueError("Feature generation needs bounded batches and fresh denoiser-owned output")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Feature generation requires available A40 BF16 support")
    band = json.loads((band_dir / "feature_band.json").read_text(encoding="utf-8"))
    selected_file = band_dir / "selected_pairs.parquet"
    if (band.get("state") != "complete" or band.get("truth_used_for_selection_or_alignment") is not False
            or band.get("positive_candidate_injection") is not False
            or sha256(selected_file) != band["files"]["selected_pairs.parquet"]
            or sha256(medium_dense / "pool.parquet") != band["inputs"]["v1_pool_sha256"]):
        raise ValueError("Feature band changed or used labels to choose candidates")
    marker = _checkpoint_complete(checkpoint)
    if marker["step"] not in (1_000, 5_000, 10_000) or marker["loaded_parameter_count"] != 299_637_760:
        raise ValueError("Feature generation requires a completed licensed ByT5-small gate")
    config = json.loads((checkpoint / "training_config.json").read_text(encoding="utf-8"))
    if (config["model"]["license"] != "apache-2.0"
            or config["model"]["combined_with_e5_and_bge_actual_upper_bound_b"] > 8
            or config["scope"] != "medium_TRAIN_only"):
        raise ValueError("Feature checkpoint must stay within the Apache/MIT parameter/split cap")
    selected = pd.read_parquet(selected_file, columns=["v1_right_idx", "target_entity_id"])
    pool = pd.read_parquet(medium_dense / "pool.parquet", columns=list(RECORD_FIELDS))
    rows = np.sort(selected.v1_right_idx.unique())
    if (rows.min() < 0 or rows.max() >= len(pool)
            or not np.array_equal(pool.entity_id.to_numpy()[selected.v1_right_idx.to_numpy(dtype=np.int32)],
                                   selected.target_entity_id.to_numpy())):
        raise ValueError("A fixed TRAIN/TUNE candidate no longer maps to its original S2/S3 text")
    strings = [text_for_record(a, b) for a, b in
               pool.iloc[rows][list(TEXT_FIELDS)].itertuples(index=False, name=None)]
    torch.set_num_threads(threads)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True,
                                                   trust_remote_code=False).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    if sum(p.numel() for p in model.parameters()) != marker["loaded_parameter_count"]:
        raise ValueError("Checkpoint loaded a different ByT5-small parameter count")
    started = time.monotonic()
    generations = _generate(model, tokenizer, strings, batch_size=batch_size,
                            max_length=config["max_source_and_target_bytes"], progress_every=512)
    if len(generations) != len(rows):
        raise ValueError("Not every label-free candidate target was canonicalized")
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(generations)
    frame.insert(0, "target_entity_id", pool.entity_id.to_numpy()[rows])
    frame.insert(0, "v1_right_idx", rows.astype(np.int32))
    file = output / "canonical_targets.parquet"
    frame.to_parquet(file, index=False)
    verified_copy(file, durable / file.name)
    verified_copy(Path(__file__), durable / "v2_denoiser_feature.py")
    report = {"state": "complete", "scope": "label-free canonicalization of selected full-pool TRAIN/TUNE dense candidate targets",
              "checkpoint": str(checkpoint.resolve()),
              "checkpoint_marker_sha256": sha256(checkpoint / "complete.json"),
              "checkpoint_step": marker["step"], "model_parameters": marker["loaded_parameter_count"],
              "license": "apache-2.0", "band_sha256": sha256(band_dir / "feature_band.json"),
              "selected_pair_sha256": band["files"]["selected_pairs.parquet"],
              "pool_file_sha256": band["inputs"]["v1_pool_sha256"],
              "unique_target_id_sha256": ids_sha(frame.target_entity_id),
              "original_field_text_sha256": texts_digest(strings),
              "unique_targets": len(frame), "selected_pairs": len(selected),
              "batch_size": batch_size, "max_length": config["max_source_and_target_bytes"],
              "seconds": time.monotonic() - started,
              "fallback_reasons": dict(Counter(frame.fallback_reason)),
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
              "canonical_target_file": {"path": str(file.resolve()),
                                        "durable_path": str((durable / file.name).resolve()),
                                        "sha256": sha256(file), "rows": len(frame)},
              "generation_mean_logprob_is_not_match_probability": True,
              "truth_read": False, "new_retrieval_edges": 0, "positive_candidate_injection": False,
              "source_sha256": sha256(Path(__file__))}
    atomic_json(output / "feature_generation.json", report)
    verified_copy(output / "feature_generation.json", durable / "feature_generation.json")
    return report


def wilson_lower(positive: int, total: int, *, z: float = 1.96) -> float:
    """Conservative TRAIN-only lower precision bound for a binary equality rule."""
    if total <= 0 or not 0 <= positive <= total:
        return 0.
    p = positive / total
    z2 = z * z
    return (p + z2 / (2 * total) - z * np.sqrt(p * (1 - p) / total + z2 / (4 * total**2))) / (
        1 + z2 / total)


def score_candidate_decisions(reference: pd.DataFrame, left: np.ndarray, labels: np.ndarray,
                              actual: np.ndarray, keep: np.ndarray) -> tuple[dict, np.ndarray]:
    """Official macro F0.5 across *every* S1, including misses and singletons."""
    left = np.asarray(left, dtype=np.int64)
    labels = np.asarray(labels, dtype=bool)
    keep = np.asarray(keep, dtype=bool)
    actual = np.asarray(actual, dtype=np.int64)
    n = len(reference)
    if (n < 1 or actual.shape != (n,) or np.any(actual < 0)
            or keep.shape != left.shape or labels.shape != left.shape
            or np.any(left < 0) or np.any(left >= n)):
        raise ValueError("A complete reference/candidate/truth alignment is required for official F0.5")
    predicted = np.bincount(left[keep], minlength=n)
    tp = np.bincount(left[keep & labels], minlength=n)
    denom = 4 * predicted + actual
    scores = np.divide(5. * tp, denom, out=np.zeros(n, dtype=np.float64), where=denom > 0)
    scores[(actual == 0) & (predicted == 0)] = 1.

    def group(mask: np.ndarray) -> dict:
        ntrue, npred, hit = int(actual[mask].sum()), int(predicted[mask].sum()), int(tp[mask].sum())
        singleton = mask & (actual == 0)
        return {"references": int(mask.sum()), "macro_f05": float(scores[mask].mean()),
                "true_links": ntrue, "predicted_links": npred, "true_positive_links": hit,
                "false_links": npred - hit, "missed_links": ntrue - hit,
                "precision": hit / npred if npred else 0., "recall": hit / ntrue if ntrue else 0.,
                "singleton_accuracy": float((predicted[singleton] == 0).mean()) if singleton.any() else None}

    report = group(np.ones(n, dtype=bool))
    report["by_country"] = {str(country): group(reference.country.eq(country).to_numpy())
                            for country in sorted(reference.country.unique())}
    return report, scores


def score_feature_band(band_dir: Path, feature_dir: Path, union: Path, medium_dense: Path,
                       medium_prepared: Path, output: Path, durable: Path,
                       *, seed: int = 20260925, min_support: int = 20,
                       min_precision: float = .98, min_wilson: float = .95) -> dict:
    """Fit a precision gate on TRAIN only; test the fixed rule on all 12k TUNE S1s."""
    if (seed < 1 or min_support < 1 or not 0 < min_wilson <= min_precision <= 1):
        raise ValueError("TRAIN precision/support gates must be fixed before reading TUNE truth")
    band_dir, feature_dir, union = map(Path, (band_dir, feature_dir, union))
    medium_dense, medium_prepared = Path(medium_dense), Path(medium_prepared)
    output, durable = Path(output), Path(durable)
    if (output.exists() and any(output.iterdir())) or (durable.exists() and any(durable.iterdir())):
        raise FileExistsError("A previously selected feature scorer must not be overwritten")
    band = json.loads((band_dir / "feature_band.json").read_text(encoding="utf-8"))
    generation = json.loads((feature_dir / "feature_generation.json").read_text(encoding="utf-8"))
    if (band.get("state") != "complete" or generation.get("state") != "complete"
            or band.get("truth_used_for_selection_or_alignment") is not False
            or generation.get("truth_read") is not False
            or generation.get("band_sha256") != sha256(band_dir / "feature_band.json")
            or band["files"]["selected_pairs.parquet"] != generation["selected_pair_sha256"]
            or sha256(feature_dir / "canonical_targets.parquet")
            != generation["canonical_target_file"]["sha256"]):
        raise ValueError("Candidate feature generation/sampling was not fixed without labels")
    selected = pd.read_parquet(band_dir / "selected_pairs.parquet")
    targets = pd.read_parquet(feature_dir / "canonical_targets.parquet")
    positions = pd.Index(targets.v1_right_idx).get_indexer(selected.v1_right_idx)
    if (targets.v1_right_idx.duplicated().any() or np.any(positions < 0)
            or not np.array_equal(targets.target_entity_id.to_numpy()[positions],
                                   selected.target_entity_id.to_numpy())):
        raise ValueError("A selected S2/S3 target has no unique saved ByT5 generation")
    reference = pd.read_parquet(medium_dense / "reference.parquet", columns=list(RECORD_FIELDS))
    pool = pd.read_parquet(medium_dense / "pool.parquet", columns=list(RECORD_FIELDS))
    diagnostic = selected.rename(columns={"v1_left_idx": "left_idx", "v1_right_idx": "right_idx"}).copy()
    for field in ("canonical", "fallback_reason", "generation_mean_logprob"):
        diagnostic[field] = targets[field].to_numpy()[positions]
    diagnostic = _component_metrics(diagnostic, reference, pool)
    diagnostic["target_name_script"] = [_script(value) for value in
                                        pool.business_name.to_numpy()[selected.v1_right_idx.to_numpy(np.int32)]]
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    feature_path = output / "selected_candidate_features.parquet"
    diagnostic.to_parquet(feature_path, index=False)
    verified_copy(feature_path, durable / feature_path.name)
    # From this point on, TRAIN labels may calibrate the already-persisted rule.
    train_ref = pd.read_parquet(medium_prepared / "train_reference.parquet", columns=["entity_id"])
    positives = pd.read_parquet(medium_prepared / "train_pairs.parquet")
    original_train_pos = pd.Index(reference.entity_id).get_indexer(train_ref.entity_id)
    if (np.any(original_train_pos < 0) or len(train_ref) != 35_999
            or ids_sha(train_ref.entity_id) != TRAIN_ID_SHA
            or sha256(medium_dense / "pool.parquet") != json.loads(
                (medium_prepared / "prepared.json").read_text())["identity"]["medium"]["dense_pool_sha256"]):
        raise ValueError("TRAIN-only precision calibration does not align with the saved medium split")
    truth_keys = np.sort((original_train_pos[positives.left_idx.to_numpy(np.int32)].astype(np.uint64)
                          << np.uint64(32)) | positives.right_idx.to_numpy(dtype=np.uint64))
    train = diagnostic.loc[diagnostic.split.eq("train")].copy()
    train_keys = ((train.left_idx.to_numpy(dtype=np.uint64) << np.uint64(32))
                  | train.right_idx.to_numpy(dtype=np.uint64))
    found = np.searchsorted(truth_keys, train_keys)
    train["true_link"] = ((found < len(truth_keys))
                          & (truth_keys[np.minimum(found, len(truth_keys) - 1)] == train_keys))
    labels_sha = hashlib.sha256(train.true_link.to_numpy(dtype=np.bool_).tobytes()).hexdigest()
    rules = (("raw_full", None, "raw_full"),
             ("full", "raw_full", "new_full"),
             ("name_digit", "raw_name_digit", "new_name_digit"))
    calibration = {}
    for name, raw_field, new_field in rules:
        incremental = train[new_field].to_numpy(dtype=bool, copy=True)
        if raw_field is not None:
            incremental &= ~train[raw_field].to_numpy(bool)
        tp = int(np.count_nonzero(incremental & train.true_link.to_numpy(bool)))
        fp = int(np.count_nonzero(incremental & ~train.true_link.to_numpy(bool)))
        total = tp + fp
        lower = wilson_lower(tp, total)
        calibration[name] = {"scope": "medium TRAIN candidates; tree score band, no CE TRAIN score",
                             "eligible": total, "true_links": tp, "false_links": fp,
                             "observed_precision": tp / total if total else None,
                             "wilson_95_lower": lower,
                             "approved": bool(total >= min_support and tp / total >= min_precision
                                              and lower >= min_wilson) if total else False}
    # TUNE truth is loaded only after original candidate identities, v1 scores,
    # generated text, and TRAIN-calibrated thresholds have all been persisted.
    atomic_json(output / "train_calibration.json", {"state": "complete", "source": "TRAIN only",
                                                   "calibration": calibration, "seed": seed,
                                                   "min_support": min_support,
                                                   "min_point_precision": min_precision,
                                                   "min_wilson_lower": min_wilson,
                                                   "train_label_sha256": labels_sha,
                                                   "selected_features_sha256": sha256(feature_path)})
    verified_copy(output / "train_calibration.json", durable / "train_calibration.json")
    union_reference, union_pool, pairs, truth, complete, union_identity = _fixed_union(union)
    retrieval, labels = retrieval_ceiling(union_reference, pairs, truth, complete)
    aligned = np.load(band_dir / "aligned_v1_probabilities.npy")
    if (len(aligned) != len(pairs)
            or sha256(band_dir / "aligned_v1_probabilities.npy") != band["files"]["aligned_v1_probabilities.npy"]
            or np.isfinite(aligned).sum() != band["v1_scored_dense_pairs"]
            or not np.isnan(aligned[~pairs.dense_hit.to_numpy(bool)]).all()):
        raise ValueError("Full-pool union-only pairs gained fabricated v1 scores")
    left = pairs.left_idx.to_numpy(np.int32)
    actual = np.asarray([len(truth[entity]) for entity in union_reference.entity_id], dtype=np.int32)
    baseline_keep = np.isfinite(aligned) & (aligned >= band["frozen_v1_threshold"])
    base, base_entities = score_candidate_decisions(union_reference, left, labels, actual, baseline_keep)
    if (abs(base["macro_f05"] - .9689921356332671) > 1e-10 or base["false_links"] != 494
            or base["predicted_links"] != 39_430 or base["references"] != 12_000):
        raise ValueError("Original frozen v1 dense matcher did not replay on the full-union S1s")
    tune = diagnostic.loc[diagnostic.split.eq("tune")].reset_index(drop=True)
    union_positions = tune.union_pair_position.to_numpy(np.int64)
    if (np.any(union_positions < 0) or len(np.unique(union_positions)) != len(union_positions)
            or not np.array_equal(pairs.target_entity_id.to_numpy()[union_positions],
                                   tune.target_entity_id.to_numpy())
            or not np.array_equal(union_reference.entity_id.to_numpy()[left[union_positions]],
                                   tune.source1_entity_id.to_numpy())
            or not np.array_equal(aligned[union_positions], tune.v1_blend_probability.to_numpy(np.float32))):
        raise ValueError("Feature-selected TUNE pairs lost frozen v1 score, original ID, or pair order")
    np.save(output / "baseline_accepted.npy", baseline_keep)
    verified_copy(output / "baseline_accepted.npy", durable / "baseline_accepted.npy")
    variants = {}
    per_variant_entities = {}
    selected_decisions = pd.DataFrame({"source1_entity_id": tune.source1_entity_id,
                                        "target_entity_id": tune.target_entity_id,
                                        "union_pair_position": union_positions,
                                        "v1_probability": tune.v1_blend_probability,
                                        "new_full": tune.new_full, "raw_full": tune.raw_full,
                                        "new_name_digit": tune.new_name_digit,
                                        "raw_name_digit": tune.raw_name_digit,
                                        "target_name_script": tune.target_name_script,
                                        "true_link": labels[union_positions]})
    for name, raw_field, new_field in rules:
        approved = calibration[name]["approved"]
        incremental = tune[new_field].to_numpy(dtype=bool, copy=True)
        incremental &= tune.v1_blend_probability.to_numpy() < band["frozen_v1_threshold"]
        if raw_field is not None:
            incremental &= ~tune[raw_field].to_numpy(bool)
        rescue = incremental & approved
        changed = union_positions[rescue]
        keep = baseline_keep.copy()
        keep[changed] = True
        if np.any(~np.isfinite(aligned[keep])) or np.any(~pairs.dense_hit.to_numpy(bool)[changed]):
            raise ValueError("Feature rule attempted to accept an unscored union-only candidate")
        scored, entity_scores = score_candidate_decisions(union_reference, left, labels, actual, keep)
        per_variant_entities[name] = entity_scores
        paired = paired_bootstrap(entity_scores, base_entities, n_bootstrap=10_000, seed=seed)
        path = output / f"accepted_{name}.npy"
        np.save(path, keep)
        verified_copy(path, durable / path.name)
        selected_decisions[f"promote_{name}"] = rescue
        changed_frame = selected_decisions.loc[rescue]
        variants[name] = {"policy": "TRAIN-approved incremental canonical equality; fixed v1 threshold 0.775",
                          "approved_on_train": bool(approved), "proposed_new_pairs": int(incremental.sum()),
                          "promoted_pairs": len(changed), "promoted_true_links": int(labels[changed].sum()),
                          "promoted_false_links": int((~labels[changed]).sum()),
                          "by_target_name_script": {str(script): {"promoted": len(group),
                                                                    "true": int(group.true_link.sum()),
                                                                    "false": int((~group.true_link).sum())}
                                                    for script, group in changed_frame.groupby("target_name_script")},
                          "tune_matcher": scored, "paired_vs_v1": paired,
                          "accepted_mask_sha256": sha256(path)}
    selected_path = output / "selected_pair_decisions.parquet"
    selected_decisions.to_parquet(selected_path, index=False)
    verified_copy(selected_path, durable / selected_path.name)
    raw_control = paired_bootstrap(per_variant_entities["full"], per_variant_entities["raw_full"],
                                   n_bootstrap=10_000, seed=seed) if (
                                       calibration["full"]["approved"] and calibration["raw_full"]["approved"]) else None
    report = {"state": "complete", "scope": "12k medium TUNE S1s; 1,437,538 full-pool union edges, v1 scores on 600k dense edges only",
              "selected_pairs": len(selected), "generated_unique_targets": generation["unique_targets"],
              "frozen_v1": base, "frozen_v1_threshold": band["frozen_v1_threshold"],
              "full_union_candidate_ceiling": retrieval["full_key_union"],
              "unscored_union_only_pairs": band["unscored_union_only_pairs"],
              "train_calibration": calibration, "variants": variants,
              "denoised_full_minus_raw_exact_control": raw_control,
              "generation_checkpoint_step": generation["checkpoint_step"],
              "generation_checkpoint_sha256": generation["checkpoint_marker_sha256"],
              "truth_read_after_feature_persistence": True, "positive_candidate_injection": False,
              "union_identity": union_identity, "band_sha256": sha256(band_dir / "feature_band.json"),
              "generation_sha256": sha256(feature_dir / "feature_generation.json"),
              "selected_features_sha256": sha256(feature_path),
              "selected_decisions_sha256": sha256(selected_path),
              "training_thresholds_fixed_before_tune_truth": {"min_support": min_support,
                                                        "min_point_precision": min_precision,
                                                        "min_wilson_95_lower": min_wilson},
              "match_probabilities": {"source": "saved frozen v1; NaN for unscored union-only pairs",
                                      "aligned_v1_sha256": band["files"]["aligned_v1_probabilities.npy"],
                                      "generation_mean_logprob_is_not_match_probability": True},
              "decision": ("screen_candidate_limited_feature_against_new_matcher" if any(
                  v["paired_vs_v1"]["confidence_intervals"]["left_minus_right"][0] > 0
                  and v["promoted_false_links"] <= v["promoted_true_links"] // 10
                  for name, v in variants.items() if name != "raw_full") else "no_verified_full_tune_matcher_gain"),
              "limitations": ["TRAIN selection uses tree scores because v1 CE TRAIN probabilities were never saved.",
                              "The TRAIN tree scores and denoiser are in-sample; conditional precision may be optimistic.",
                              "Calibration has no independent entity-disjoint holdout and the 12k TUNE set was reused.",
                              "Unscored union-only pairs are never accepted; this is a dense-v1 matcher screen, not a fully scored union matcher.",
                              "France label-free diagnostics are separate from this US/India score."]}
    atomic_json(output / "feature_matcher_report.json", report)
    verified_copy(output / "feature_matcher_report.json", durable / "feature_matcher_report.json")
    return report


def _french_references(source: Path, seed: int, count: int) -> pd.DataFrame:
    """Uniform reservoir of *unlabeled* supplied French test S1 text only."""
    rng = np.random.default_rng(seed)
    chosen: list[tuple] = []
    seen = 0
    for frame in pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False,
                             usecols=list(RECORD_FIELDS), chunksize=50_000):
        for row in frame.loc[frame.country.eq("France"), list(RECORD_FIELDS)].itertuples(index=False, name=None):
            seen += 1
            if len(chosen) < count:
                chosen.append(row)
            else:
                position = int(rng.integers(seen))
                if position < count:
                    chosen[position] = row
    if not chosen or len({row[0] for row in chosen}) != len(chosen):
        raise ValueError("No unique supplied France Source 1 records for label-free diagnostics")
    return pd.DataFrame(chosen, columns=list(RECORD_FIELDS))


def evaluate(output: Path, durable: Path, union: Path, *, seed: int = 20260925,
             n_refs: int = 1_024, inference_batch: int = 16,
             test_source1: Path | None = None, french_count: int = 128,
             baseline_output: Path | None = None) -> dict:
    """One fixed TUNE canonicalization probe and the independently checked 12k oracle."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not 1 <= n_refs <= 4_096 or not 1 <= inference_batch <= 32 or french_count < 0:
        raise ValueError("TUNE/French probe size or generation batch is outside the bounded screen")
    trained = json.loads((output / "train_result.json").read_text(encoding="utf-8"))
    if trained.get("state") != "complete" or sha256(output / "train_result.json") != sha256(durable / "train_result.json"):
        raise ValueError("Denoiser TRAIN result/checkpoint has not completed durably")
    checkpoint = Path(trained["checkpoint"])
    marker = _checkpoint_complete(checkpoint)
    if marker["loaded_parameter_count"] != trained["model"]["loaded_parameter_count"]:
        raise ValueError("Checkpoint parameter count changed after training")
    evaluator_source = durable / f"v2_denoiser_eval_step_{trained['steps']}.py"
    verified_copy(Path(__file__), evaluator_source)
    reference, pool, pairs, truth, completed, union_identity = _fixed_union(union)
    retrieval, _ = retrieval_ceiling(reference, pairs, truth, completed)
    sample = _probe_sample(reference, pool, pairs, truth, seed=seed, n_refs=n_refs)
    path = output / "eval_sample.parquet"
    if path.exists():
        raise FileExistsError("A previous TUNE sample exists; do not silently reselect it")
    baseline_report = None
    baseline_step = None
    if baseline_output is not None:
        baseline_report = json.loads((baseline_output / "report.json").read_text(encoding="utf-8"))
        baseline_step = json.loads((baseline_output / "train_result.json").read_text(encoding="utf-8"))["steps"]
        old_sample = baseline_output / "eval_sample.parquet"
        if (baseline_report.get("state") != "complete"
                or baseline_report["tune_union"] != union_identity
                or baseline_report["sample_sha256"] != sha256(old_sample)
                or baseline_report["sample_references"] != sample.left_idx.nunique()
                or not pd.read_parquet(old_sample).equals(sample)):
            raise ValueError("Continuation must reuse exactly the completed 200-step TUNE sample and union")
        verified_copy(old_sample, path)
    else:
        sample.to_parquet(path, index=False)
    verified_copy(path, durable / path.name)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Canonicalization screen requires the same BF16 CUDA device")
    config = json.loads((output / "training_config.json").read_text(encoding="utf-8"))
    max_length = config["max_source_and_target_bytes"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True,
                                                   trust_remote_code=False).to("cuda")
    inputs = [text_for_record(pool.business_name.iat[int(i)], pool.business_address.iat[int(i)])
              for i in sample.right_idx]
    started = time.monotonic()
    generations = _generate(model, tokenizer, inputs, batch_size=inference_batch, max_length=max_length)
    for column in ("generated", "canonical", "fallback_reason", "input_token_length",
                   "output_token_length", "generation_mean_logprob"):
        sample[column] = [row[column] for row in generations]
    scored = _component_metrics(sample, reference, pool)
    probe_path = output / "canonicalization_probe.parquet"
    scored.to_parquet(probe_path, index=False)
    verified_copy(probe_path, durable / probe_path.name)
    canonical = _canonical_report(scored, reference, seed)
    comparison = None
    if baseline_output is not None:
        old_probe = baseline_output / "canonicalization_probe.parquet"
        if baseline_report["canonicalization_probe_sha256"] != sha256(old_probe):
            raise ValueError("The earlier checkpoint's aligned probability/decision probe changed")
        comparison = paired_extension(scored, pd.read_parquet(old_probe), seed=seed + 1)
    france = None
    if test_source1 is not None and french_count:
        french = _french_references(test_source1, seed + 1, french_count)
        if set(french.country) != {"France"}:
            raise ValueError("France label-free sample contains another country")
        french_inputs = [text_for_record(a, b) for a, b in
                         french[list(TEXT_FIELDS)].itertuples(index=False, name=None)]
        french_output = _generate(model, tokenizer, french_inputs,
                                  batch_size=inference_batch, max_length=max_length)
        for field in ("generated", "canonical", "fallback_reason", "generation_mean_logprob"):
            french[field] = [row[field] for row in french_output]
        french_path = output / "france_label_free_probe.parquet"
        french.to_parquet(french_path, index=False)
        verified_copy(french_path, durable / french_path.name)
        france = {"scope": "supplied test Source 1 raw text, France; NO labels or accuracy",
                  "source_sha256": sha256(test_source1), "rows": len(french),
                  "id_sha256": ids_sha(french.entity_id), "probe_sha256": sha256(french_path),
                  "fallback_reasons": dict(Counter(french.fallback_reason)),
                  "name_changed_after_normalization": int(sum(
                      normalize_text(a.split(" | ", 1)[0]) != normalize_text(b.split(" | ", 1)[0])
                      for a, b in zip(french_inputs, french.canonical)))}
    delta = canonical["sampled_true_links_including_retrieval_misses"]["full"]["delta"]
    lower = canonical["paired_positive_full"]["confidence_intervals"]["left_minus_right"][0]
    false_new = canonical["sampled_retrieved_hard_nonlinks"]["full"]["denoised_hits"]
    false_old = canonical["sampled_retrieved_hard_nonlinks"]["full"]["raw_hits"]
    gate = config["promotion_gate"]
    raw_gate = (delta >= gate["paired_full_normalized_exact_delta_min"]
                and lower > gate["paired_95pct_CI_lower_min"]
                and false_new - false_old <= gate["extra_false_exact_full_matches_max"])
    promote = raw_gate and (comparison is None or comparison["new_checkpoint_wins_screen_gate"])
    report = {"state": "complete", "scope": "bounded 12k medium-TUNE full-pool retrieval ceiling + sampled canonicalization",
              "decision": (f"retain_{trained['steps']}_step_checkpoint" if promote else
                           f"retain_{baseline_step}_step_checkpoint") if comparison is not None
              else "extend_only_after_full_pool_branch_design" if promote else "stop_weak_screen",
              "promotion_gate_met": bool(promote), "original_raw_normalization_gate_met": bool(raw_gate),
              "seed": seed,
              "training_config_sha256": sha256(output / "training_config.json"),
              "evaluator_source_sha256": sha256(evaluator_source),
              "checkpoint_manifest_sha256": sha256(checkpoint / "complete.json"),
              "sample_references": sample.left_idx.nunique(), "sample_pairs": len(sample),
              "sample_ids_sha256": ids_sha(sample.source1_entity_id.unique()),
              "sample_sha256": sha256(path), "canonicalization_probe_sha256": sha256(probe_path),
              "tune_union": union_identity, "tune_union_directory": str(union.resolve()),
              "complete_candidate_pairs_file": str((union / "candidate_pairs.parquet").resolve()),
              "candidate_retrieval_ceiling": retrieval,
              "canonicalization": canonical,
              "paired_vs_baseline_step": {"baseline_step": baseline_step, **comparison} if comparison else None,
              "baseline_report_sha256": sha256(baseline_output / "report.json") if baseline_output else None,
              "france_label_free": france,
              "generation_seconds": time.monotonic() - started,
              "generation_mean_logprob_is_not_match_probability": True,
              "feature_policy": {"exact_normalized_name_and_address_threshold": 1.0,
                                 "address_digit_sequence_must_equal_input": True,
                                 "classifier_match_probability": None},
              "matcher_macro_f05": None, "matcher_precision_recall_false_links_singleton_accuracy": None,
              "new_retrieval_edges": 0, "positive_candidate_injection": False,
              "limitations": ["Fixed full-pool candidate recall/oracle is unchanged by a target-only sample probe.",
                              "Canonicalization sample is from reused TUNE, not a full-pool matcher score.",
                              "No medium eval/HOLDOUT-2 truth or French match labels were opened.",
                              "A pool-wide denoised retrieval branch needs complete label-free indexing and matcher re-fitting."]}
    if comparison is not None:
        report["limitations"].append("Sequential checkpoint selection reuses the same TUNE sample; no independent matcher gain is established.")
    atomic_json(output / "report.json", report)
    verified_copy(output / "report.json", durable / "report.json")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "prepare-full", "feature-band", "feature-generate", "feature-score",
                 "train", "extend", "evaluate"):
        sub = subs.add_parser(name)
        sub.add_argument("--output", type=Path, required=True, help="Own /tmp/az-ml-v2/denoiser output")
        sub.add_argument("--durable", type=Path, required=True, help="Own /workspace/az-ml-new/v2/denoiser")
    prep = subs.choices["prepare"]
    prep.add_argument("--medium-dense", type=Path, required=True)
    prep.add_argument("--medium-splits", type=Path, required=True)
    prep.add_argument("--reservation", type=Path, required=True)
    prep.add_argument("--ground-truth", type=Path, required=True)
    full = subs.choices["prepare-full"]
    full.add_argument("--bootstrap", type=Path, required=True,
                      help="Completed P2 label-free bootstrap with 2.07M S1s and all 10.32M targets")
    full.add_argument("--reservation", type=Path, required=True)
    full.add_argument("--ground-truth", type=Path, required=True)
    full.add_argument("--batch-links", type=int, default=50_000)
    band = subs.choices["feature-band"]
    band.add_argument("--union", type=Path, required=True)
    band.add_argument("--medium-dense", type=Path, required=True)
    band.add_argument("--frozen-v1-scores", type=Path, required=True)
    band.add_argument("--seed", type=int, default=20260925)
    band.add_argument("--train-sample", type=int, default=8000)
    generated = subs.choices["feature-generate"]
    generated.add_argument("--band", type=Path, required=True)
    generated.add_argument("--medium-dense", type=Path, required=True)
    generated.add_argument("--checkpoint", type=Path, required=True)
    generated.add_argument("--batch-size", type=int, default=16)
    generated.add_argument("--threads", type=int, default=2)
    scored = subs.choices["feature-score"]
    scored.add_argument("--band", type=Path, required=True)
    scored.add_argument("--generated", type=Path, required=True)
    scored.add_argument("--union", type=Path, required=True)
    scored.add_argument("--medium-dense", type=Path, required=True)
    scored.add_argument("--medium-prepared", type=Path, required=True)
    scored.add_argument("--seed", type=int, default=20260925)
    fit = subs.choices["train"]
    fit.add_argument("--medium-dense", type=Path, required=True)
    fit.add_argument("--snapshot", type=Path, required=True)
    fit.add_argument("--inventory", type=Path, required=True)
    fit.add_argument("--steps", type=int, default=200)
    fit.add_argument("--seed", type=int, default=20260925)
    fit.add_argument("--microbatch", type=int, default=4)
    fit.add_argument("--accumulation", type=int, default=4)
    fit.add_argument("--max-length", type=int, default=192)
    fit.add_argument("--learning-rate", type=float, default=1e-4)
    fit.add_argument("--threads", type=int, default=2)
    fit.add_argument("--log-every", type=int, default=20)
    more = subs.choices["extend"]
    more.add_argument("--base-output", type=Path, required=True)
    more.add_argument("--base-durable", type=Path, required=True)
    more.add_argument("--medium-dense", type=Path, required=True)
    more.add_argument("--steps", type=int, default=1000)
    more.add_argument("--threads", type=int, default=2)
    more.add_argument("--log-every", type=int, default=50)
    more.add_argument("--train-data-output", type=Path, help="Original sealed medium TRAIN preparation")
    more.add_argument("--train-data-durable", type=Path, help="Verified durable copy of medium TRAIN preparation")
    score = subs.choices["evaluate"]
    score.add_argument("--union", type=Path, required=True)
    score.add_argument("--seed", type=int, default=20260925)
    score.add_argument("--sample-references", type=int, default=1024)
    score.add_argument("--inference-batch", type=int, default=16)
    score.add_argument("--test-source1", type=Path, help="Optional supplied France Source 1; text-only diagnostics")
    score.add_argument("--french-count", type=int, default=128)
    score.add_argument("--baseline-output", type=Path, help="Freeze/reuse an earlier completed TUNE sample")
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        if args.action == "prepare":
            result = prepare(args.medium_dense, args.medium_splits, args.reservation, args.ground_truth,
                             args.output, args.durable)
        elif args.action == "prepare-full":
            result = prepare_full_train(args.bootstrap, args.reservation, args.ground_truth,
                                        args.output, args.durable, batch_links=args.batch_links)
        elif args.action == "feature-band":
            result = prepare_feature_band(args.union, args.medium_dense, args.frozen_v1_scores,
                                          args.output, args.durable, seed=args.seed,
                                          train_sample=args.train_sample)
        elif args.action == "feature-generate":
            result = generate_feature_band(args.band, args.medium_dense, args.checkpoint,
                                           args.output, args.durable, batch_size=args.batch_size,
                                           threads=args.threads)
        elif args.action == "feature-score":
            result = score_feature_band(args.band, args.generated, args.union, args.medium_dense,
                                        args.medium_prepared, args.output, args.durable, seed=args.seed)
        elif args.action == "train":
            result = train(args.output, args.durable, args.medium_dense, args.snapshot, args.inventory,
                           steps=args.steps, seed=args.seed, microbatch=args.microbatch,
                           accumulation=args.accumulation, max_length=args.max_length,
                           learning_rate=args.learning_rate, threads=args.threads,
                           log_every=args.log_every)
        elif args.action == "extend":
            result = extend(args.base_output, args.base_durable, args.output, args.durable,
                            args.medium_dense, steps=args.steps, threads=args.threads,
                            log_every=args.log_every, train_data_output=args.train_data_output,
                            train_data_durable=args.train_data_durable)
        else:
            result = evaluate(args.output, args.durable, args.union, seed=args.seed,
                              n_refs=args.sample_references, inference_batch=args.inference_batch,
                              test_source1=args.test_source1, french_count=args.french_count,
                              baseline_output=args.baseline_output)
        print(json.dumps({"action": args.action, "state": result["state"], "seconds": time.monotonic() - started,
                          "decision": result.get("decision"), "output": str(args.output),
                          "durable": str(args.durable)}), flush=True)
    except Exception as exc:
        # An interrupted/failed run remains inspectable; never overwrite its evidence.
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"action": args.action, "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started}
        name = f"failure_{args.action}_{time.time_ns()}.json"
        atomic_json(args.output / name, failure)
        if args.durable.is_dir():
            verified_copy(args.output / name, args.durable / name)
        raise


if __name__ == "__main__":
    main()
