"""P2.5: stream the sealed full TRAIN dense/key union, then join TRAIN truth.

The lexical miner is read-only and runs on the workstation. ``stage`` transfers
only its committed, hash-checked parts to the pod. ``screen`` processes the first
10,000 eligible S1s; ``full`` follows committed parts and resumes by shard. Both
fix and hash candidate edges *before* selecting truth from the supplied TSV.
The final ready.json is compatible with the P3 tree and CE consumers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .normalization import normalize_text
from .v2_lexical_mining import _SUFFIXES, text_keys


POOL_ROWS = 10_320_219
SOURCE_ROWS = (5_034_616, 5_285_603)
DENSE_REFERENCES = 2_122_821
TRAIN_REFERENCES = 2_072_821
TRUTH_ROWS = 2_206_821
TRAIN_IDS_SHA256 = "7720350d18cd1333dbeb0e688bdc67fc4ed47c4ee2c1a54126f7743382944e01"
SHARD_REFERENCES = 10_000
LEX_PART_REFERENCES = 1_000
RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
KEY_FLAGS = ("name_exact_hit", "phonetic_key_hit", "digit_sequence_hit",
             "digit_set_hit", "composite_key_hit")
PAIR_DTYPES = {
    "left_idx": "int32", "right_idx": "int32", "right_raw_idx": "int32",
    "name_score": "float32", "address_score": "float32",
    "retrieval_score": "float32", "retrieval_rank": "int32",
    "dense_score": "float32", "dense_rank": "int32", "original_dense_score": "float32",
    "dense_hit": "bool", "lexical_hit": "bool", "legacy_phonetic_hit": "bool",
    "legacy_phonetic_only": "bool", "phonetic_score": "float32", "phonetic_rank": "int32",
    "phonetic_only": "bool", "dense_only": "float32", "lexical_scores_missing": "bool",
    "lexical_retrieval_score": "float32", "lexical_retrieval_rank": "int32",
    "lexical_key_score": "float32", "canonical_raw_idx": "int32",
    "canonical_group_size": "int32", **dict.fromkeys(KEY_FLAGS, "bool"),
}
JOIN_DTYPES = {**PAIR_DTYPES, "label": "bool", "same_name": "bool",
               "same_digits": "bool", "hard_negative_priority": "int8"}
_DIGITS = re.compile(r"\d+")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path: Path, *, parquet: bool = False) -> dict:
    value = {"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": file_sha256(path)}
    if parquet:
        value["rows"] = pq.ParquetFile(path).metadata.num_rows
    return value


def checked_file(path: Path, info: dict, *, parquet: bool = False) -> None:
    if (not path.is_file() or path.stat().st_size != info["bytes"]
            or file_sha256(path) != info["sha256"]
            or (parquet and pq.ParquetFile(path).metadata.num_rows != info["rows"])):
        raise ValueError(f"Committed input/output changed: {path}")


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} P2.5 {message}", flush=True)


def _validate_lexical_progress(progress: dict, *, limit: int | None = None) -> list[dict]:
    config = progress["identity"]
    if (progress.get("state") != "mining" or progress.get("references_total") != TRAIN_REFERENCES
            or config.get("reference_ids_sha256") != TRAIN_IDS_SHA256
            or config.get("splits") != ["train_tune_eligible"]
            or config.get("partition_column") != "population"
            or config.get("partitioned") is not True
            or config.get("batch_size") != LEX_PART_REFERENCES):
        raise ValueError("Staged lexical parts do not belong to the sealed full-eligible population")
    parts = progress["parts"]
    cursor = rows = 0
    for i, part in enumerate(parts):
        if (part.get("name") != f"part_{i:06d}.parquet" or part["start"] != cursor
                or part["end"] != min(cursor + LEX_PART_REFERENCES, TRAIN_REFERENCES)
                or part["rows"] < 0 or part["bytes"] <= 0 or len(part["sha256"]) != 64):
            raise ValueError("Lexical part manifest is not a contiguous immutable S1 prefix")
        cursor, rows = part["end"], rows + part["rows"]
    if (cursor != progress["references_done"] or rows != progress["pairs"]
            or cursor > TRAIN_REFERENCES or (limit is not None and len(parts) < limit)):
        raise ValueError("Lexical progress and committed parts disagree")
    return parts


def _committed_lexical_prefix(source: Path, progress: dict, index_sha: str) -> tuple[list[dict], dict[str, Path]]:
    """Include sealed-prefix and only contiguous, committed disjoint shard parts.

    The lexical publisher later hard-links the very same bytes/names back into
    ``source``. This read-only view lets the TRAIN union proceed while those
    workers run; it does not create or change a lexical retrieval partition.
    """
    prefix = list(_validate_lexical_progress(progress))
    paths = {part["name"]: source / "candidate_pairs.parquet" / part["name"] for part in prefix}
    seal_path = source / "prefix_sealed.json"
    if not seal_path.is_file():
        return prefix, paths
    seal = json.loads(seal_path.read_text())
    if (seal.get("state") != "sealed_prefix" or seal.get("references_total") != TRAIN_REFERENCES
            or seal.get("parent_reference_ids_sha256") != TRAIN_IDS_SHA256
            or seal.get("full_pool_index_manifest_sha256") != index_sha
            or seal.get("full_pool_rows") != POOL_ROWS
            or seal.get("truth_read") is not False
            or seal.get("positive_candidate_injection") is not False
            or seal.get("partition_count") != seal.get("references_done") // LEX_PART_REFERENCES
            or seal["references_done"] > progress["references_done"]
            or seal["full_reference_sha256"] != progress["reference_output_sha256"]):
        raise ValueError("The immutable label-free lexical prefix seal is inconsistent")
    if progress["references_done"] > seal["references_done"]:
        # The lexical owner has published its assembled hard links in the
        # original directory. Stage those original global parts directly.
        if not progress.get("assembly_manifest_sha256") or len(prefix) <= seal["partition_count"]:
            raise ValueError("The lexical publisher extended an unsealed prefix")
        return prefix, paths
    if (progress["references_done"] != seal["references_done"]
            or len(prefix) != seal["partition_count"]
            or progress["pairs"] != seal["candidate_pairs"]
            or file_sha256(source / "mining_progress.json") != seal["progress_sha256"]):
        raise ValueError("The sealed lexical prefix changed while shards were mining")
    directory = source.parent / "full_eligible_shards"
    if not directory.is_dir():
        return prefix, paths
    spans = []
    for shard in directory.iterdir():
        match = re.fullmatch(r"shard_(\d+)_(\d+)", shard.name)
        if not shard.is_dir() or match is None:
            raise ValueError(f"Unexpected lexical shard beside the sealed prefix: {shard}")
        spans.append((int(match.group(1)), int(match.group(2)), shard))
    spans.sort()
    last = seal["references_done"]
    for begin, end, _ in spans:
        if begin != last or not begin < end <= TRAIN_REFERENCES or begin % LEX_PART_REFERENCES:
            raise ValueError("Disjoint lexical shard boundaries overlap or leave eligible S1s uncovered")
        last = end
    if spans and last != TRAIN_REFERENCES:
        raise ValueError("The announced lexical shard ranges omit fit-eligible S1s")
    cursor = seal["references_done"]
    for begin, end, shard in spans:
        state = json.loads((shard / "mining_progress.json").read_text())
        item = state["identity"]
        if (state.get("state") != "mining" or state.get("references_total") != end - begin
                or item.get("left_offset") != begin or item.get("left_stop") != end
                or item.get("parent_references") != TRAIN_REFERENCES
                or item.get("parent_reference_ids_sha256") != TRAIN_IDS_SHA256
                or item.get("reference_file_sha256") != seal["full_reference_sha256"]
                or item.get("index_manifest_sha256") != index_sha
                or item.get("pool_rows", POOL_ROWS) != POOL_ROWS
                or item.get("splits") != ["train_tune_eligible"]
                or item.get("partitioned") is not True
                or item.get("batch_size") != LEX_PART_REFERENCES
                or pq.ParquetFile(shard / "reference.parquet").metadata.num_rows != end - begin
                or file_sha256(shard / "reference.parquet") != state["reference_output_sha256"]):
            raise ValueError(f"A live lexical shard does not belong to the sealed TRAIN-only source: {shard}")
        local_cursor = pair_count = 0
        for number, part in enumerate(state["parts"]):
            if (part["name"] != f"part_{number:06d}.parquet" or part["start"] != local_cursor
                    or part["end"] != min(local_cursor + LEX_PART_REFERENCES, end - begin)
                    or part["rows"] < 0 or part["bytes"] <= 0 or len(part["sha256"]) != 64):
                raise ValueError(f"Invalid committed partition interval in {shard}")
            local_cursor, pair_count = part["end"], pair_count + part["rows"]
        if local_cursor != state["references_done"] or pair_count != state["pairs"]:
            raise ValueError("Lexical shard progress disagrees with committed partition rows")
        complete_path = shard / "complete.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text())
            if (complete.get("state") != "complete" or complete.get("truth_read") is not False
                    or complete.get("positive_candidate_injection") is not False
                    or complete.get("references") != end - begin or complete.get("pairs") != pair_count
                    or complete.get("reference_output_sha256") != state["reference_output_sha256"]
                    or complete.get("full_pool_rows") != POOL_ROWS
                    or complete["pair_file"]["parts"] != state["parts"]):
                raise ValueError(f"Completed lexical shard disagrees with its label-free progress: {shard}")
        if begin != cursor:
            break  # Later shards may be ready but a preceding range is not.
        for part in state["parts"]:
            global_start, global_end = begin + part["start"], begin + part["end"]
            name = f"part_{global_start // LEX_PART_REFERENCES:06d}.parquet"
            descriptor = {"name": name, "start": global_start, "end": global_end,
                          "rows": part["rows"], "bytes": part["bytes"], "sha256": part["sha256"]}
            if name in paths:
                raise ValueError("A disjoint lexical shard overlaps an already published part")
            prefix.append(descriptor)
            paths[name] = shard / "candidate_pairs.parquet" / part["name"]
        cursor = begin + local_cursor
        if cursor != end:
            break
    _validate_lexical_progress({"state": "mining", "identity": progress["identity"],
                                "references_total": TRAIN_REFERENCES, "references_done": cursor,
                                "pairs": sum(part["rows"] for part in prefix), "parts": prefix})
    return prefix, paths


def _ssh_flags(key: Path, port: int) -> list[str]:
    if not key.is_file() or not 0 < port < 65536:
        raise ValueError("An existing authorized SSH key and valid port are required")
    return ["-F", "/dev/null", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=10", "-i", str(key), "-p", str(port)]


def stage_lexical(source: Path, index_manifest: Path, local: Path, remote: str,
                  host: str, key: Path, port: int, *, follow: bool = False,
                  max_parts: int | None = None, poll_seconds: float = 30.) -> dict:
    """Transfer only published lexical parts; never modify the miner's files."""
    source, local = Path(source), Path(local)
    if (not (source / "candidate_pairs.parquet").is_dir() or not remote.startswith("/tmp/az-ml-v2/train_union/")
            or not host.startswith("root@") or poll_seconds <= 0 or (max_parts is not None and max_parts < 1)):
        raise ValueError("Only the separate full-eligible miner and owned pod scratch are valid staging endpoints")
    flags = _ssh_flags(key, port)
    # A source part is first renamed and only then enters mining_progress.json.
    subprocess.run(["ssh", *flags, host, "test", "-d", str(Path(remote).parent)], check=True)
    subprocess.run(["ssh", *flags, host, "mkdir", "-p", remote + "/candidate_pairs.parquet"], check=True)
    local.mkdir(parents=True, exist_ok=True)
    state_path = local / "stage_state.json"
    previous = json.loads(state_path.read_text()) if state_path.is_file() else None
    while True:
        progress_path = source / "mining_progress.json"
        progress = json.loads(progress_path.read_text())
        index_sha = file_sha256(index_manifest)
        all_parts, source_paths = _committed_lexical_prefix(source, progress, index_sha)
        if (progress["identity"]["index_manifest_sha256"] != index_sha
                or file_sha256(source / "reference.parquet") != progress["reference_output_sha256"]
                or pq.ParquetFile(source / "reference.parquet").metadata.num_rows != TRAIN_REFERENCES):
            raise ValueError("Live lexical index or eligible references changed")
        count = min(len(all_parts), max_parts) if max_parts is not None else len(all_parts)
        if previous:
            if (previous["reference_sha256"] != progress["reference_output_sha256"]
                    or previous["index_sha256"] != index_sha
                    or all_parts[:len(previous["parts"])] != previous["parts"]):
                raise ValueError("An already staged lexical partition changed")
        prior = len(previous["parts"]) if previous else 0
        selected = all_parts[:count]
        new_parts = selected[prior:]
        for part in new_parts:
            checked_file(source_paths[part["name"]], part, parquet=True)
        complete_path = source / "complete.json"
        final = json.loads(complete_path.read_text()) if complete_path.is_file() else None
        if final:
            if (final.get("state") != "complete" or final.get("references") != TRAIN_REFERENCES
                    or final.get("pairs") != sum(part["rows"] for part in final["pair_file"]["parts"])
                    or final.get("reference_ids_sha256") != TRAIN_IDS_SHA256
                    or final.get("full_pool_rows") != POOL_ROWS
                    or final.get("positive_candidate_injection") is not False
                    or final.get("truth_read") is not False):
                raise ValueError("Final label-free lexical marker has a different TRAIN-only identity")
            if final["pair_file"].get("parts") != all_parts:
                # A publisher may commit complete.json between the two atomic
                # manifest reads above; try the now-complete source next poll.
                if not follow or progress["references_done"] == TRAIN_REFERENCES:
                    raise ValueError("Final lexical parts disagree with already staged original rows")
                time.sleep(poll_seconds)
                continue
        direct = [f"candidate_pairs.parquet/{part['name']}" for part in new_parts
                  if source_paths[part["name"]] == source / "candidate_pairs.parquet" / part["name"]]
        linked = [part for part in new_parts if source_paths[part["name"]]
                  != source / "candidate_pairs.parquet" / part["name"]]
        if not previous:
            direct.append("reference.parquet")
        if final and count == len(all_parts):
            direct.append("complete.json")
        rsh = "ssh " + " ".join(flags)
        if direct:
            subprocess.run(["rsync", "-a", "--files-from=-", "-e", rsh, str(source) + "/",
                            f"{host}:{remote}/"], input="\n".join(direct) + "\n", text=True, check=True)
        if linked:
            # Canonical names in our own scratch point to immutable committed
            # shard files. -L streams them in one rsync, as regular files in the
            # pod's original-row-indexed staging directory. Source files stay
            # entirely untouched, even while the lexical publisher works.
            links = local / "stage_links" / "candidate_pairs.parquet"
            links.mkdir(parents=True, exist_ok=True)
            for part in linked:
                link = links / part["name"]
                actual = source_paths[part["name"]].resolve()
                if link.is_symlink():
                    if link.resolve() != actual:
                        raise ValueError("Staged shard link points to a different lexical candidate file")
                elif link.exists():
                    raise ValueError("A non-symlink occupies an owned staging link")
                else:
                    link.symlink_to(actual)
            subprocess.run(["rsync", "-aL", "--files-from=-", "-e", rsh,
                            str(local / "stage_links") + "/", f"{host}:{remote}/"],
                           input="\n".join(f"candidate_pairs.parquet/{part['name']}" for part in linked) + "\n",
                           text=True, check=True)
        # Stage the small index manifest for full-pool byte/order checks, not its 5.5-GB search index.
        source_copy = local / "source_index_complete.json"
        if not source_copy.exists():
            shutil.copyfile(index_manifest, source_copy)
        if file_sha256(source_copy) != index_sha:
            raise ValueError("The staged full-pool index identity changed")
        subprocess.run(["rsync", "-a", "-e", rsh, str(source_copy),
                        f"{host}:{remote}/source_index_complete.json"], check=True)
        report = {"schema": 1, "state": "staging", "references_total": TRAIN_REFERENCES,
                  "references_staged": selected[-1]["end"] if selected else 0,
                  "reference_sha256": progress["reference_output_sha256"],
                  "train_reference_id_sha256": TRAIN_IDS_SHA256, "index_sha256": index_sha,
                  "lexical_identity": progress["identity"], "parts": selected,
                  "source_complete": {"sha256": file_sha256(complete_path), "pairs": final["pairs"]}
                  if final and count == len(all_parts) else None,
                  "truth_read": False, "positive_candidate_injection": False}
        atomic_json(state_path, report)
        subprocess.run(["rsync", "-a", "-e", rsh, str(state_path),
                        f"{host}:{remote}/staging_manifest.json"], check=True)
        previous = report
        if direct or linked:
            log(f"staged {len(new_parts)} immutable candidate parts; "
                f"{report['references_staged']:,}/{TRAIN_REFERENCES:,} S1s")
        if not follow or report["source_complete"]:
            return report
        time.sleep(poll_seconds)


def checked_stage(directory: Path) -> dict:
    report = json.loads((directory / "staging_manifest.json").read_text())
    if (report.get("state") != "staging" or report.get("truth_read") is not False
            or report.get("positive_candidate_injection") is not False
            or report.get("train_reference_id_sha256") != TRAIN_IDS_SHA256
            or file_sha256(directory / "reference.parquet") != report["reference_sha256"]
            or file_sha256(directory / "source_index_complete.json") != report["index_sha256"]):
        raise ValueError("Staged lexical identity, full-pool index or reference bytes changed")
    progress = {"state": "mining", "identity": report["lexical_identity"],
                "references_total": TRAIN_REFERENCES, "references_done": report["references_staged"],
                "parts": report["parts"], "pairs": sum(p["rows"] for p in report["parts"])}
    _validate_lexical_progress(progress)
    if report["source_complete"]:
        path = directory / "complete.json"
        complete = json.loads(path.read_text())
        if (file_sha256(path) != report["source_complete"]["sha256"]
                or complete.get("references") != TRAIN_REFERENCES
                or complete.get("pairs") != report["source_complete"]["pairs"]
                or complete["pair_file"]["parts"] != report["parts"]
                or report["references_staged"] != TRAIN_REFERENCES
                or complete.get("truth_read") is not False):
            raise ValueError("The staged lexical completion marker is incomplete or changed")
    return report


def _text_rank_keys(name: str, address: str) -> tuple[str, str]:
    """The key miner's core/digit-set semantics, without an unused phonetic pass."""
    words = normalize_text(name).split()
    while words and words[-1] in _SUFFIXES:
        words.pop()
    digits = _DIGITS.findall(normalize_text(address))
    return " ".join(words), "|".join(sorted({value.lstrip("0") or "0" for value in digits}))


def build_full_pool(raw_dir: Path, index_manifest: dict, index_manifest_sha256: str,
                    dense_identity: dict, durable: Path, *, expected: tuple[int, int] = SOURCE_ROWS,
                    chunk_size: int = 50_000) -> dict:
    """One small full-pool Parquet, raw S2 followed by S3, plus rank keys.

    The Arrow key cache is an O(1) positional text lookup, never a new retrieval
    index. The entire original pool remains available for missed truth records.
    """
    durable.mkdir(parents=True, exist_ok=True)
    marker = durable / "pool_complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        for name in ("pool", "keys"):
            checked_file(durable / saved["files"][name]["path"], saved["files"][name],
                         parquet=name == "pool")
        if saved["source_index_sha256"] != index_manifest_sha256:
            raise ValueError("Pool records were built from another full original target pool")
        return saved
    if any((durable / name).exists() for name in ("pool.parquet", "pool_keys.arrow",
                                                   "pool.parquet.tmp", "pool_keys.arrow.tmp")):
        raise ValueError("Uncommitted full-pool records need inspection before resuming")
    if index_manifest.get("state") != "complete" or index_manifest.get("rows") != sum(expected):
        raise ValueError("Only the original complete 10,320,219-target index is an input")
    if (dense_identity["pool_embeddings"]["rows"] != sum(expected)
            or index_manifest["source_order"] != [f"train_source{i}.tsv" for i in (2, 3)]):
        raise ValueError("Dense vectors and original S2/S3 raw source order differ")
    country_codes = (np.load(dense_identity["country_cache_path"], mmap_mode="r")
                     if "country_cache_path" in dense_identity else None)
    country_labels = ({name: i for i, name in enumerate(dense_identity["country_labels"])}
                      if country_codes is not None else {})
    if country_codes is not None and country_codes.shape != (sum(expected),):
        raise ValueError("Saved full-pool country blocking is not aligned with original target rows")
    sources = index_manifest["sources"]
    for i, (source, rows) in enumerate(zip(sources, expected)):
        raw = raw_dir / source["name"]
        old = dense_identity.get("raw_sources", [])
        if (source["expected_rows"] != rows or raw.stat().st_size != source["bytes"]
                or file_sha256(raw) != source["sha256"]
                or (old and (old[i]["size_bytes"] != source["bytes"]
                             or old[i]["mtime_ns"] != raw.stat().st_mtime_ns))):
            raise ValueError(f"Original {raw.name} bytes/rows changed before pool materialization")
    pool_tmp, keys_tmp = durable / "pool.parquet.tmp", durable / "pool_keys.arrow.tmp"
    pool_writer = None
    seen: set[str] = set()
    total = 0
    keys_schema = pa.schema([("core", pa.string()), ("digit_set", pa.string())])
    try:
        with pa.OSFile(str(keys_tmp), "wb") as sink, ipc.new_file(sink, keys_schema) as key_writer:
            for source_number, (source, expected_rows) in enumerate(zip(sources, expected), 2):
                count = 0
                for frame in pd.read_csv(raw_dir / source["name"], sep="\t", dtype=str,
                                         keep_default_na=False, usecols=list(RECORD_COLUMNS),
                                         chunksize=chunk_size):
                    ids = frame.entity_id.tolist()
                    if (len(ids) != len(set(ids)) or seen.intersection(ids)
                            or not all(item.startswith(f"S{source_number}-") for item in ids)
                            or not frame.country.isin(["India", "US"]).all()):
                        raise ValueError("Original full-pool target IDs/countries are invalid or repeated")
                    if country_codes is not None:
                        observed = frame.country.map(lambda value: country_labels.get(normalize_text(value), -1)).to_numpy(np.int32)
                        if not np.array_equal(observed, country_codes[total:total + len(frame)]):
                            raise ValueError("Saved dense country cache differs from original S2/S3 raw text")
                    seen.update(ids)
                    frame["raw_row_idx"] = np.arange(total, total + len(frame), dtype=np.int32)
                    record = pa.Table.from_pandas(frame[[*RECORD_COLUMNS, "raw_row_idx"]], preserve_index=False)
                    if pool_writer is None:
                        pool_writer = pq.ParquetWriter(pool_tmp, record.schema, compression="zstd")
                    pool_writer.write_table(record, row_group_size=len(frame))
                    rank_keys = [_text_rank_keys(name, address) for name, address in
                                 frame[["business_name", "business_address"]].itertuples(index=False, name=None)]
                    key_writer.write_batch(pa.RecordBatch.from_arrays(
                        [pa.array([row[i] for row in rank_keys], type=pa.string()) for i in (0, 1)],
                        schema=keys_schema))
                    count += len(frame)
                    total += len(frame)
                    if total % 1_000_000 < chunk_size:
                        log(f"full target records indexed by raw row: {total:,}/{sum(expected):,}")
                if count != expected_rows:
                    raise ValueError(f"Source {source_number} contains {count:,} rows, expected {expected_rows:,}")
    finally:
        if pool_writer is not None:
            pool_writer.close()
    del seen
    if total != sum(expected) or pq.ParquetFile(pool_tmp).metadata.num_rows != total:
        raise ValueError("Original full pool did not preserve every S2/S3 row")
    for temporary, final in ((pool_tmp, durable / "pool.parquet"), (keys_tmp, durable / "pool_keys.arrow")):
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(final)
    with pa.memory_map(str(durable / "pool_keys.arrow"), "r") as stream:
        if ipc.open_file(stream).read_all().num_rows != total:
            raise ValueError("Full target negative-ranking keys are not positional")
    files = {name: {**identity(durable / path, parquet=name == "pool"), "path": path}
             for name, path in (("pool", "pool.parquet"), ("keys", "pool_keys.arrow"))}
    report = {"state": "complete", "schema": 1, "source_order": [s["name"] for s in sources],
              "source_sha256": [s["sha256"] for s in sources],
              "source_index_sha256": index_manifest_sha256,
              "full_pool_records": total, "files": files, "truth_read": False,
              "raw_row_rule": "right_idx == 0-based S2 row, followed by S3 row + 5,034,616",
              "keys_rule": "same core name/digit set as ber.v2_lexical_mining.text_keys"}
    atomic_json(marker, report)
    return report


def bootstrap(dense: Path, staged: Path, raw_dir: Path, reservation: Path,
              scratch: Path, durable: Path) -> dict:
    """Seal disjoint fit-only references and complete original-row target records."""
    staged_manifest = checked_stage(staged)
    durable.mkdir(parents=True, exist_ok=True)
    (durable / "full").mkdir(exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    saved_path = durable / "bootstrap.json"
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text())
        for name in ("reference", "fit_to_dense", "pool"):
            item = saved["files"][name]
            checked_file(durable / item["path"], item, parquet=name in ("reference", "pool"))
        if (saved["lexical_reference_sha256"] != staged_manifest["reference_sha256"]
                or saved["dense_complete_sha256"] != file_sha256(dense / "dense_complete.json")):
            raise ValueError("Bootstrap is associated with different complete retrieval inputs")
        pool_marker = json.loads((durable / "pool_complete.json").read_text())
        checked_file(durable / "pool_keys.arrow", pool_marker["files"]["keys"])
        return saved
    dense_complete = json.loads((dense / "dense_complete.json").read_text())
    descriptor = dense_complete["identity"]["identity"]
    dense_config = dense_complete["identity"]
    reference_manifest = json.loads((dense / "reference_manifest.json").read_text())
    index_manifest = json.loads((staged / "source_index_complete.json").read_text())
    excluded_manifest = json.loads((reservation / "reservation.json").read_text())
    if (dense_complete.get("state") != "complete"
            or dense_complete.get("references") != DENSE_REFERENCES
            or dense_complete.get("candidate_pairs") != DENSE_REFERENCES * 50
            or dense_config.get("labels_read") is not False
            or dense_config.get("positive_candidate_injection") is not False
            or dense_config.get("right_index") != "global raw S2-then-S3 row number"
            or descriptor["pool_embeddings"]["rows"] != POOL_ROWS
            or descriptor["reference_sha256"] != reference_manifest["parquet_sha256"]["reference.parquet"]
            or reference_manifest.get("labels_read") is not False
            or reference_manifest["population"]["groups"]["train_tune_eligible"]["sorted_id_sha256"] != TRAIN_IDS_SHA256
            or reference_manifest["population"]["groups"]["holdout2"]["count"] != 50_000
            or excluded_manifest["verification"]["labels_read"] is not False
            or excluded_manifest["future_raw_train_tune"]["eligible_count"] != TRAIN_REFERENCES
            or index_manifest["rows"] != POOL_ROWS):
        raise ValueError("Dense, lexical, reservation, or full target identities are incomplete")
    if (file_sha256(dense / "candidate_pairs.parquet") != dense_complete["candidate_pairs_sha256"]
            or file_sha256(dense / "reference.parquet") != descriptor["reference_sha256"]
            or file_sha256(dense / "query_embeddings.npy") != json.loads(
                (dense / "query_complete.json").read_text())["embedding_sha256"]):
        raise ValueError("Completed dense candidates, query text or vectors changed")
    snapshot = dense_complete["identity"]["topk_snapshot"]["files"]
    for filename, info in snapshot.items():
        if file_sha256(dense / filename) != info["sha256"]:
            raise ValueError("Dense top-50 checkpoint is no longer complete")
    if excluded_manifest["future_raw_train_tune"]["excluded_file_sha256"] != file_sha256(
            reservation / excluded_manifest["future_raw_train_tune"]["excluded_id_file"]):
        raise ValueError("TRAIN exclusion IDs differ from the sealed reservation")
    if excluded_manifest["holdout2"]["id_file_sha256"] != file_sha256(
            reservation / excluded_manifest["holdout2"]["id_file"]):
        raise ValueError("HOLDOUT-2 ID file differs from the sealed reservation")
    source = pd.read_parquet(dense / "reference.parquet")
    lexical = pd.read_parquet(staged / "reference.parquet")
    mask = source.population.eq("train_tune_eligible").to_numpy()
    selected = source.loc[mask].reset_index(drop=True)
    if (len(source) != DENSE_REFERENCES or len(selected) != TRAIN_REFERENCES
            or len(lexical) != TRAIN_REFERENCES or ids_sha256(selected.entity_id) != TRAIN_IDS_SHA256
            or not selected[[*RECORD_COLUMNS, "raw_source1_row", "population"]].equals(
                lexical[[*RECORD_COLUMNS, "raw_source1_row", "population"]])
            or selected.entity_id.duplicated().any() or not source.population[~mask].eq("holdout2").all()
            or ids_sha256(source.loc[~mask, "entity_id"]) != excluded_manifest["holdout2"]["sorted_id_sha256"]):
        raise ValueError("Eligible lexical and dense references are not the same original S1 rows")
    excluded = set((reservation / excluded_manifest["future_raw_train_tune"]["excluded_id_file"]).read_text().splitlines())
    if (len(excluded) != excluded_manifest["future_raw_train_tune"]["excluded_count"]
            or any(item in excluded for item in selected.entity_id)):
        raise ValueError("An earlier split or HOLDOUT-2 reference entered TRAIN")
    del source, lexical, excluded
    mapping = np.flatnonzero(mask).astype(np.int32)
    path = durable / "fit_to_dense.npy"
    with path.with_suffix(".npy.tmp").open("wb") as stream:
        np.save(stream, mapping)
        stream.flush()
        os.fsync(stream.fileno())
    path.with_suffix(".npy.tmp").replace(path)
    reference = durable / "full" / "reference.parquet"
    selected[[*RECORD_COLUMNS, "raw_source1_row"]].assign(split="train").to_parquet(
        str(reference) + ".tmp", index=False, compression="zstd")
    Path(str(reference) + ".tmp").replace(reference)
    if pq.ParquetFile(reference).metadata.num_rows != TRAIN_REFERENCES:
        raise ValueError("TRAIN-only reference parquet omitted an original S1")
    pool_descriptor = {**descriptor, "raw_sources": json.loads(
        (dense / "reference_manifest.json").read_text())["identity"].get("raw_sources", [])}
    # Dense full-pool identity records original S2/S3 sizes and mtimes separately.
    preserved = json.loads((Path(descriptor["pool_embeddings"]["path"]).parent / "complete.json").read_text())
    pool_descriptor["raw_sources"] = preserved["identity"]["inputs_in_order"]
    pool = build_full_pool(raw_dir, index_manifest, staged_manifest["index_sha256"],
                           pool_descriptor, durable)
    files = {"reference": {**identity(reference, parquet=True), "path": "full/reference.parquet"},
             "fit_to_dense": {**identity(path), "path": "fit_to_dense.npy"},
             "pool": {**pool["files"]["pool"], "path": "pool.parquet"}}
    result = {"state": "complete", "schema": 1, "full_pool_records": POOL_ROWS,
              "dense_complete_sha256": file_sha256(dense / "dense_complete.json"),
              "dense_candidate_pairs_sha256": dense_complete["candidate_pairs_sha256"],
              "dense_query_sha256": json.loads((dense / "query_complete.json").read_text())["embedding_sha256"],
              "pool_vector_sha256": descriptor["pool_embeddings"]["sha256"],
              "index_manifest_sha256": staged_manifest["index_sha256"],
              "lexical_reference_sha256": staged_manifest["reference_sha256"],
              "reservation_sha256": file_sha256(reservation / "reservation.json"),
              "train_reference_id_sha256": TRAIN_IDS_SHA256,
              "holdout2_reference_id_sha256": excluded_manifest["holdout2"]["sorted_id_sha256"],
              "files": files, "truth_read": False, "positive_candidate_injection": False}
    atomic_json(saved_path, result)
    log("sealed 2,072,821 fit-only S1s and the full 10,320,219-row target pool")
    return result


def _effective_cgroup_memory() -> int | None:
    root = Path("/sys/fs/cgroup")
    try:
        if (root / "memory.max").is_file():
            limit = int((root / "memory.max").read_text())
            used = int((root / "memory.current").read_text())
            key = "inactive_file"
            stat = root / "memory.stat"
        else:
            base = root / "memory"
            limit = int((base / "memory.limit_in_bytes").read_text())
            used = int((base / "memory.usage_in_bytes").read_text())
            key = "total_inactive_file"
            stat = base / "memory.stat"
        inactive = next((int(line.split()[1]) for line in stat.read_text().splitlines()
                         if line.split()[0] == key), 0)
        return max(0, limit - used + inactive) if limit < 1 << 60 else None
    except (OSError, ValueError, StopIteration):
        return None


class UnionInputs:
    """Read-only complete dense checkpoints and a separate staged lexical prefix."""

    def __init__(self, dense: Path, staged: Path, durable: Path, *, pool_vector_residency: str = "mmap",
                 preload_vectors: bool = True):
        self.dense, self.staged, self.durable = dense, staged, durable
        self.bootstrap = json.loads((durable / "bootstrap.json").read_text())
        self.reference = pd.read_parquet(durable / "full" / "reference.parquet")
        self.mapping = np.load(durable / "fit_to_dense.npy", mmap_mode="r")
        self.raw = np.load(dense / "topk_raw_rows.npy", mmap_mode="r")
        self.scores = np.load(dense / "topk_scores.npy", mmap_mode="r")
        self.queries = np.load(dense / "query_embeddings.npy", mmap_mode="r")
        complete = json.loads((dense / "dense_complete.json").read_text())
        descriptor = complete["identity"]["identity"]
        self.pool_vectors_path = Path(descriptor["pool_embeddings"]["path"])
        if pool_vector_residency not in ("ram", "mmap"):
            raise ValueError("Pool vectors must be either a read-only mmap or a bounded in-RAM copy")
        self.pool_vector_residency = pool_vector_residency
        self.vectors = None
        mmap_vectors = np.load(self.pool_vectors_path, mmap_mode="r")
        vector_shape, vector_dtype = mmap_vectors.shape, mmap_vectors.dtype
        del mmap_vectors
        if preload_vectors:
            self.ensure_pool_vectors()
        self.countries = np.load(descriptor["country_cache_path"], mmap_mode="r")
        labels = {country: i for i, country in enumerate(descriptor["country_labels"])}
        self.query_country = self.reference.country.map(lambda value: labels.get(normalize_text(value), -1)).to_numpy(np.int8)
        if (len(self.mapping) != TRAIN_REFERENCES or len(self.reference) != TRAIN_REFERENCES
                or self.raw.shape != (DENSE_REFERENCES, 50)
                or self.scores.shape != self.raw.shape or self.queries.shape != (DENSE_REFERENCES, 384)
                or vector_shape != (POOL_ROWS, 384) or vector_dtype != np.float16
                or self.countries.shape != (POOL_ROWS,) or np.any(self.query_country < 0)
                or file_sha256(dense / "dense_complete.json") != self.bootstrap["dense_complete_sha256"]):
            raise ValueError("The sealed full dense/query/pool positional identities have changed")

    def ensure_pool_vectors(self) -> None:
        if self.vectors is not None:
            return
        if self.pool_vector_residency == "mmap":
            self.vectors = np.load(self.pool_vectors_path, mmap_mode="r")
            return
        effective = _effective_cgroup_memory()
        required = self.pool_vectors_path.stat().st_size + 4 * 1024**3
        if effective is None or effective < required:
            raise MemoryError(f"Cannot hold the saved E5 pool in RAM: effective cgroup headroom "
                              f"{effective}, required {required:,} bytes")
        log("loading saved pool vectors sequentially into pod RAM to avoid object-backed random I/O")
        self.vectors = np.load(self.pool_vectors_path)
        self.vectors.flags.writeable = False

    def release_pool_vectors(self) -> None:
        if self.pool_vector_residency == "ram" and self.vectors is not None:
            self.vectors = None
            gc.collect()
            log("released 8-GB E5 pool RAM while waiting for immutable lexical partitions")


def cosine_on_new_only(inputs: UnionInputs, global_left: np.ndarray, raw_right: np.ndarray,
                       *, batch_size: int = 20_000) -> np.ndarray:
    """Gather only the missing key branch's saved E5 vectors, not another index."""
    output = np.empty(len(raw_right), dtype=np.float32)
    # The preserved 7.93-GB pool file lives on the object-backed workspace.
    # Increasing raw-row order turns random small page faults into forward I/O;
    # restore the exact original pair order for the candidate Parquet afterwards.
    order = np.argsort(raw_right, kind="stable")
    for first in range(0, len(raw_right), batch_size):
        stop = min(first + batch_size, len(raw_right))
        positions = order[first:stop]
        output[positions] = np.einsum(
            "ij,ij->i", np.asarray(inputs.queries[global_left[positions]], dtype=np.float32),
            np.asarray(inputs.vectors[raw_right[positions]], dtype=np.float32))
    if not np.isfinite(output).all():
        raise ValueError("The saved E5 vectors produced a non-finite dense cosine")
    return np.clip(output, -1, 1)


def union_shard(inputs: UnionInputs, start: int, end: int, parts: list[dict]) -> pd.DataFrame:
    if (start % SHARD_REFERENCES or end <= start or end > TRAIN_REFERENCES
            or len(parts) != (end + LEX_PART_REFERENCES - 1) // LEX_PART_REFERENCES - start // LEX_PART_REFERENCES
            or parts[0]["start"] != start or parts[-1]["end"] != end):
        raise ValueError("Union needs one complete contiguous, committed S1/lexical shard")
    global_rows = np.asarray(inputs.mapping[start:end], dtype=np.int32)
    rights = np.asarray(inputs.raw[global_rows], dtype=np.int32)
    scores = np.asarray(inputs.scores[global_rows], dtype=np.float32)
    dense_left = np.repeat(np.arange(start, end, dtype=np.int32), rights.shape[1])
    dense_right = rights.ravel()
    if (np.any((rights < 0) | (rights >= POOL_ROWS)) or not np.isfinite(scores).all()
            or np.any(np.diff(scores, axis=1) > 1e-5)
            or np.any(np.sort(rights, axis=1)[:, 1:] == np.sort(rights, axis=1)[:, :-1])
            or not np.array_equal(inputs.query_country[dense_left], inputs.countries[dense_right])):
        raise ValueError("Dense top-50 rows are duplicated, cross-country or misaligned")
    dense = pd.DataFrame({"left_idx": dense_left, "right_idx": dense_right,
                          "original_dense_score": np.clip(scores.ravel(), -1, 1),
                          "dense_rank": np.tile(np.arange(1, 51, dtype=np.int32), end - start)})
    frames = []
    for part in parts:
        path = inputs.staged / "candidate_pairs.parquet" / part["name"]
        checked_file(path, part, parquet=True)
        frame = pd.read_parquet(path)
        if len(frame):
            if frame.duplicated(["left_idx", "right_raw_idx"]).any():
                raise ValueError("A lexical partition has duplicate original target pairs")
            if (frame.left_idx.min() < part["start"] or frame.left_idx.max() >= part["end"]
                    or frame.right_raw_idx.min() < 0 or frame.right_raw_idx.max() >= POOL_ROWS
                    or not np.array_equal(inputs.query_country[frame.left_idx.to_numpy(np.int32)],
                                           inputs.countries[frame.right_raw_idx.to_numpy(np.int32)])):
                raise ValueError(f"A key candidate escaped its sealed S1/country/raw-row partition: {path}")
            if (frame.canonical_raw_idx.lt(0).any() or frame.canonical_group_size.lt(1).any()
                    or frame.canonical_raw_idx.gt(frame.right_raw_idx).any()):
                raise ValueError("Key retrieval lost original-row canonical expansion")
        frames.append(frame)
    key = pd.concat(frames, ignore_index=True).rename(columns={
        "right_raw_idx": "right_idx", "phonetic_hit": "phonetic_key_hit"})
    if key.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Two lexical partitions claim the same original target for a TRAIN S1")
    if len(key) != sum(part["rows"] for part in parts):
        raise ValueError("Lexical partition row counts changed after stage/validation")
    merged = dense.merge(key, how="outer", on=["left_idx", "right_idx"],
                         validate="one_to_one", indicator="_branch", sort=False)
    if len(merged) < len(dense) or merged.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("The label-free union dropped or duplicated a dense top-50 edge")
    merged["dense_hit"] = merged._branch.ne("right_only").to_numpy()
    merged["original_dense_score"] = merged.original_dense_score.fillna(0).astype(np.float32)
    merged["dense_rank"] = merged.dense_rank.fillna(0).astype(np.int32)
    for name in KEY_FLAGS:
        merged[name] = merged[name].fillna(False).astype(bool)
    for name in ("canonical_raw_idx", "canonical_group_size"):
        merged[name] = merged[name].fillna(0).astype(np.int32)
    merged["lexical_key_score"] = merged.lexical_key_score.fillna(0).astype(np.float32)
    if (merged.loc[~merged.dense_hit, list(KEY_FLAGS)].any(axis=1) == False).any():
        raise ValueError("A non-dense candidate has no recorded label-free key branch")
    values = merged.original_dense_score.to_numpy(dtype=np.float32, copy=True)
    new = ~merged.dense_hit.to_numpy(dtype=bool)
    left = merged.left_idx.to_numpy(dtype=np.int32)
    right = merged.right_idx.to_numpy(dtype=np.int32)
    if new.any():
        values[new] = cosine_on_new_only(inputs, np.asarray(inputs.mapping[left[new]], dtype=np.int32), right[new])
    # Score a bounded deterministic sample of old dense hits against the same
    # vectors; the original top-50 scores are otherwise reused verbatim.
    overlap = np.flatnonzero(~new)
    sample = overlap[np.linspace(0, len(overlap) - 1, min(len(overlap), 2_048), dtype=np.int64)]
    replay = cosine_on_new_only(inputs, np.asarray(inputs.mapping[left[sample]], dtype=np.int32), right[sample])
    if len(sample) and np.max(np.abs(replay - values[sample])) > 2e-4:
        raise ValueError("Dense top-50 score is not compatible with the saved full-pool vectors")
    merged["dense_score"] = values
    merged["retrieval_score"] = values
    merged["retrieval_rank"] = merged.groupby("left_idx", sort=False).dense_score.rank(
        method="min", ascending=False).astype(np.int32)
    # All TRAIN lexical records come from the new full-pool key miner. Legacy
    # char-TF-IDF/phonetic scores exist on medium TUNE only; do not fabricate them.
    merged["right_raw_idx"] = merged.right_idx.astype(np.int32)
    merged["lexical_hit"] = False
    merged["legacy_phonetic_hit"] = False
    merged["legacy_phonetic_only"] = False
    merged["name_score"] = np.float32(0)
    merged["address_score"] = np.float32(0)
    merged["phonetic_score"] = np.float32(0)
    merged["phonetic_rank"] = np.int32(0)
    merged["phonetic_only"] = merged.phonetic_key_hit.to_numpy(dtype=bool)
    merged["dense_only"] = merged.dense_hit.to_numpy(dtype=np.float32)
    merged["lexical_scores_missing"] = True
    merged["lexical_retrieval_score"] = np.float32(0)
    merged["lexical_retrieval_rank"] = np.int32(0)
    merged.drop(columns="_branch", inplace=True)
    merged.sort_values(["left_idx", "right_idx"], kind="stable", inplace=True, ignore_index=True)
    result = merged[list(PAIR_DTYPES)].astype(PAIR_DTYPES)
    if (not np.array_equal(result.right_idx.to_numpy(), result.right_raw_idx.to_numpy())
            or result["dense_hit"].sum() != (end - start) * 50
            or not np.isfinite(result.dense_score.to_numpy()).all()
            or not result.lexical_scores_missing.all()
            or result.duplicated(["left_idx", "right_idx"]).any()):
        raise ValueError("Scored union lost exact dense/key provenance or original target rows")
    return result


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or Path(str(path) + ".tmp").exists():
        raise ValueError(f"An uncommitted candidate shard already exists: {path}")
    temporary = Path(str(path) + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary,
                   compression="zstd", row_group_size=100_000)
    if pq.ParquetFile(temporary).metadata.num_rows != len(frame):
        raise ValueError("Candidate shard writer lost original pair rows")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    return identity(path, parquet=True)


def _raw_identity(inputs: UnionInputs) -> dict:
    return {"schema": 1, "references": TRAIN_REFERENCES, "full_pool_records": POOL_ROWS,
            "dense_complete_sha256": inputs.bootstrap["dense_complete_sha256"],
            "dense_candidate_pairs_sha256": inputs.bootstrap["dense_candidate_pairs_sha256"],
            "lexical_reference_sha256": inputs.bootstrap["lexical_reference_sha256"],
            "lexical_index_sha256": inputs.bootstrap["index_manifest_sha256"],
            "fit_mapping_sha256": inputs.bootstrap["files"]["fit_to_dense"]["sha256"],
            "labels_read": False, "positive_candidate_injection": False,
             "right_idx": "original raw S2 row, then S3 row; no compact remapping"}


def _sealed_raw(inputs: UnionInputs, directory: Path, references: int, *,
                screen: bool, create: bool = False) -> tuple[dict, dict]:
    """An immutable completion seal, independent of geesefs's mutable JSON cache.

    Geesefs can return an older ``raw_progress.json`` after a rapid overwrite in
    the same second. Its immutable *part list* must agree with this create-once
    marker, and all raw shard hashes/countries are checked before labels open.
    """
    progress_path = directory / "raw_progress.json"
    raw = json.loads(progress_path.read_text())
    stage = checked_stage(inputs.staged)
    if (raw.get("identity") != _raw_identity(inputs)
            or stage["index_sha256"] != inputs.bootstrap["index_manifest_sha256"]
            or stage["reference_sha256"] != inputs.bootstrap["lexical_reference_sha256"]
            or raw.get("truth_read") is not False
            or raw.get("positive_candidate_injection") is not False
            or raw.get("state") not in ("merging", "complete")
            or raw.get("references_done") != references
            or len(raw.get("parts", [])) != (references + SHARD_REFERENCES - 1) // SHARD_REFERENCES
            or (not screen and (stage["source_complete"] is None
                                or stage["references_staged"] != TRAIN_REFERENCES))):
        raise ValueError("Every label-free raw S1 shard and lexical completion must precede TRAIN truth")
    if not screen and file_sha256(inputs.staged / "complete.json") != stage["source_complete"]["sha256"]:
        raise ValueError("The staged full lexical retrieval completion checksum changed")
    canonical = json.dumps({"identity": raw["identity"], "references_done": references,
                            "parts": raw["parts"]}, sort_keys=True, separators=(",", ":")).encode()
    expected = {"state": "complete", "schema": 1, "references": references,
                "raw_candidate_pairs": sum(item["file"]["rows"] for item in raw["parts"]),
                "raw_part_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
                "lexical_complete_sha256": stage["source_complete"]["sha256"] if not screen else None,
                "truth_read": False, "positive_candidate_injection": False}
    marker = directory / "raw_complete.json"
    if not marker.is_file():
        if not create:
            raise ValueError("TRAIN truth requires the immutable raw_complete.json seal")
        _validate_fixed_raw(inputs, directory, raw, references)
        atomic_json(marker, expected)  # New name: no overwrite on object-backed geesefs.
    if json.loads(marker.read_text()) != expected:
        raise ValueError("The immutable raw completion seal differs from original candidate shards")
    return raw, expected


def produce_raw(inputs: UnionInputs, directory: Path, *, screen: bool = False,
                follow: bool = False, poll_seconds: float = 30.) -> dict:
    """Checkpoint one label-free, complete-S1 union shard as parts arrive."""
    limit = SHARD_REFERENCES if screen else TRAIN_REFERENCES
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "raw_progress.json"
    stable = _raw_identity(inputs)
    report = json.loads(marker.read_text()) if marker.is_file() else {
        "state": "merging", "identity": stable, "references_done": 0, "parts": [],
        "truth_read": False, "positive_candidate_injection": False}
    if report.get("identity") != stable or report.get("truth_read") is not False:
        raise ValueError("A prior raw union belongs to a different dense/lexical input")
    cursor = 0
    for part in report["parts"]:
        if part["start"] != cursor or part["end"] != min(cursor + SHARD_REFERENCES, limit):
            raise ValueError("Raw union resume has a noncontiguous S1 shard")
        checked_file(directory / "raw" / part["file"]["path"], part["file"], parquet=True)
        cursor = part["end"]
    if (cursor != report["references_done"] or report["state"] not in ("merging", "complete")
            or (report["state"] == "complete" and cursor != limit)):
        raise ValueError("Raw union marker disagrees with the committed S1 prefix")
    while cursor < limit:
        end = min(cursor + SHARD_REFERENCES, limit)
        stage = checked_stage(inputs.staged)
        if stage["references_staged"] < end:
            if stage["source_complete"]:
                raise ValueError("Full lexical completion omitted an expected eligible-S1 partition")
            if not follow:
                break
            inputs.release_pool_vectors()
            log(f"waiting for committed lexical parts through S1 {end:,} (staged {stage['references_staged']:,})")
            time.sleep(poll_seconds)
            continue
        inputs.ensure_pool_vectors()
        parts = stage["parts"][cursor // LEX_PART_REFERENCES:(end + LEX_PART_REFERENCES - 1) // LEX_PART_REFERENCES]
        path = directory / "raw" / f"part_{cursor // SHARD_REFERENCES:06d}.parquet"
        # The 10k compatibility screen's raw shard is exactly the first full
        # shard, and may be hardlinked rather than rescored/copying saved vectors.
        screen_path = inputs.durable / "screen" / "raw" / path.name
        screen_report = inputs.durable / "screen" / "raw_progress.json"
        if (not screen and cursor == 0 and screen_report.is_file()
                and json.loads(screen_report.read_text()).get("identity") == stable):
            saved = json.loads(screen_report.read_text())["parts"][0]
            if saved["lexical_parts"] != parts:
                raise ValueError("The screen's raw candidate edges used different lexical parts")
            checked_file(screen_path, saved["file"], parquet=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise ValueError("Unclaimed full raw shard exists")
            try:
                os.link(screen_path, path)
            except OSError:
                shutil.copyfile(screen_path, path)
            result = identity(path, parquet=True)
        else:
            pairs = union_shard(inputs, cursor, end, parts)
            result = _write_parquet(pairs, path)
        report["parts"].append({"start": cursor, "end": end,
                                "file": {**result, "path": path.name}, "lexical_parts": parts})
        report["references_done"] = end
        cursor = end
        atomic_json(marker, report)
        log(f"RAW union {cursor:,}/{limit:,} TRAIN S1s; {result['rows']:,} label-free edges in shard")
    if cursor == limit and report["state"] != "complete":
        if not screen and checked_stage(inputs.staged)["source_complete"] is None:
            if follow:
                while checked_stage(inputs.staged)["source_complete"] is None:
                    time.sleep(poll_seconds)
            else:
                return report
        report["state"] = "complete"
        report["raw_candidate_pairs"] = sum(part["file"]["rows"] for part in report["parts"])
        report["raw_retrieval_manifest_sha256"] = file_sha256(inputs.staged / "complete.json") if not screen else None
        atomic_json(marker, report)
    if cursor == limit and not screen:
        # A previous fast overwrite of raw_progress.json may still read as
        # `merging` on geesefs. The separate create-once seal is authoritative.
        _sealed_raw(inputs, directory, limit, screen=screen, create=True)
        report["state"] = "complete"
        report["raw_candidate_pairs"] = sum(item["file"]["rows"] for item in report["parts"])
    return report


def _positional_keys(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if (len(left) != len(right) or np.any(left < 0) or np.any(right < 0)
            or np.any(left >= 2**32) or np.any(right >= 2**32)):
        raise ValueError("Original S1/S2/S3 positions must fit uint32 pair keys")
    return (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)


def selected_truth(inputs: UnionInputs, raw_dir: Path, directory: Path,
                   *, references: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read ONLY eligible selected S1 label values after their raw edges are fixed.

    The supplied TSV includes holdout rows. We inspect their ID field to skip
    them, but never decode, split or retain their matched-ID field.
    """
    progress_path = directory / "raw_progress.json"
    progress = json.loads(progress_path.read_text())
    seal = directory / "raw_complete.json"
    if seal.is_file():
        fixed = json.loads(seal.read_text())
        canonical = json.dumps({"identity": progress["identity"], "references_done": references,
                                "parts": progress["parts"]}, sort_keys=True,
                               separators=(",", ":")).encode()
        if (fixed.get("state") != "complete" or fixed.get("references") != references
                or fixed.get("truth_read") is not False
                or fixed.get("positive_candidate_injection") is not False
                or fixed.get("raw_candidate_pairs") != sum(p["file"]["rows"] for p in progress["parts"])
                or fixed.get("raw_part_manifest_sha256") != hashlib.sha256(canonical).hexdigest()):
            raise ValueError("Immutable raw completion disagrees with selected TRAIN candidate shards")
    elif references != SHARD_REFERENCES or progress.get("state") != "complete":
        raise ValueError("Truth join cannot precede the complete saved raw candidate set")
    if progress.get("references_done") != references or progress.get("truth_read") is not False:
        raise ValueError("Truth join cannot precede the complete saved raw candidate set")
    truth_path, marker = directory / "train_truth_pairs.parquet", directory / "truth_complete.json"
    if marker.is_file():
        old = json.loads(marker.read_text())
        raw_changed = (old.get("raw_complete_sha256") != file_sha256(seal) if seal.is_file()
                       else old.get("raw_progress_sha256") != file_sha256(progress_path))
        if (raw_changed or old["reference_id_sha256"] != ids_sha256(inputs.reference.entity_id.iloc[:references])
                or old["ground_truth_sha256"] != file_sha256(raw_dir / "train_ground_truth.tsv")):
            raise ValueError("TRAIN truth belongs to different completed retrieval/split/raw bytes")
        checked_file(truth_path, old["truth_file"], parquet=True)
        frame = pd.read_parquet(truth_path)
        left, right = frame.left_idx.to_numpy(np.int32), frame.right_idx.to_numpy(np.int32)
        counts = np.bincount(left, minlength=references).astype(np.int32)
        if (len(left) != old["true_links"] or int(np.count_nonzero(counts == 0)) != old["singletons"]):
            raise ValueError("Resumed TRAIN truth lost one or more entities/links")
        return np.sort(_positional_keys(left, right)), counts, old
    if truth_path.exists() or Path(str(truth_path) + ".tmp").exists():
        raise ValueError("Uncommitted TRAIN truth exists; inspect before joining")
    raw_file = raw_dir / "train_ground_truth.tsv"
    query = inputs.reference.iloc[:references]
    positions = {entity: i for i, entity in enumerate(query.entity_id)}
    if len(positions) != references or (references == TRAIN_REFERENCES and
                                      ids_sha256(positions) != TRAIN_IDS_SHA256):
        raise ValueError("Selected truth population must be unique sealed TRAIN S1 IDs")
    # No target lookup is needed for retrieval. Build it only NOW, after the raw
    # edge files are fixed, so truth can never affect candidate generation.
    target_ids = pd.read_parquet(inputs.durable / "pool.parquet", columns=["entity_id"]).entity_id
    pool_index = pd.Index(target_ids)
    if len(pool_index) != POOL_ROWS or not pool_index.is_unique:
        raise ValueError("Target pool IDs must be exactly the supplied unique original rows")
    seen = np.zeros(references, dtype=bool)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    pending_left: list[int] = []
    pending_targets: list[str] = []
    selected_rows = raw_rows = 0

    def flush() -> None:
        if not pending_targets:
            return
        mapped = pool_index.get_indexer(pending_targets)
        if np.any(mapped < 0):
            raise ValueError("A selected TRAIN true target ID is missing from the original full target pool")
        left = np.asarray(pending_left, dtype=np.int32)
        right = mapped.astype(np.int32)
        if (not np.array_equal(inputs.query_country[left], inputs.countries[right])
                or np.any(np.asarray([value.startswith("S2-") for value in pending_targets])
                          != (right < SOURCE_ROWS[0]))):
            raise ValueError("TRAIN truth crosses country or maps a target ID to a wrong source row")
        left_parts.append(left)
        right_parts.append(right)
        pending_left.clear()
        pending_targets.clear()

    with raw_file.open("rb") as stream:
        if stream.readline() != b"source1_entity_id\tmatched_entity_ids\n":
            raise ValueError("Original TRAIN ground truth has an unexpected header")
        for line in stream:
            raw_rows += 1
            source_id, separator, payload = line.partition(b"\t")
            if not separator:
                raise ValueError("Original TRAIN truth has a row without an S1/target separator")
            entity = source_id.decode("ascii")
            left_idx = positions.get(entity)
            if left_idx is None:
                continue  # No held-out or old development matched IDs are examined.
            if seen[left_idx]:
                raise ValueError("A selected TRAIN S1 appears twice in original ground truth")
            seen[left_idx] = True
            selected_rows += 1
            values = payload.strip(b"\r\n")
            if values:
                targets = values.decode("ascii").split(",")
                if (len(targets) != len(set(targets))
                        or any(not target.startswith(("S2-", "S3-")) for target in targets)):
                    raise ValueError("Selected TRAIN truth contains duplicate/invalid target IDs")
                pending_targets.extend(targets)
                pending_left.extend([left_idx] * len(targets))
            if len(pending_targets) >= 100_000:
                flush()
    flush()
    if (selected_rows != references or raw_rows != TRUTH_ROWS or not seen.all()
            or len(left_parts) == 0):
        raise ValueError("Truth coverage omitted a selected S1 or the original raw TSV changed")
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    keys = np.sort(_positional_keys(left, right))
    if np.any(keys[1:] == keys[:-1]) or len(np.unique(right)) != len(right):
        raise ValueError("Selected TRAIN truth has duplicate pairs or two S1 claims on one target")
    counts = np.bincount(left, minlength=references).astype(np.int32)
    truth = pd.DataFrame({"left_idx": (keys >> np.uint64(32)).astype(np.int32),
                          "right_idx": (keys & np.uint64(0xffffffff)).astype(np.int32)})
    file_info = _write_parquet(truth, truth_path)
    report = {"state": "complete", "schema": 1, "scope": "eligible TRAIN S1 only",
              "raw_progress_sha256": file_sha256(progress_path),
              "raw_complete_sha256": file_sha256(seal) if seal.is_file() else None,
              "ground_truth_sha256": file_sha256(raw_file),
              "reference_id_sha256": ids_sha256(positions),
              "selected_references": references, "original_truth_id_rows_scanned": raw_rows,
              "true_links": len(keys), "singletons": int(np.count_nonzero(counts == 0)),
              "truth_file": {**file_info, "path": truth_path.name},
              "holdout2_label_values_decoded": False, "older_split_label_values_decoded": False,
              "labels_read_after_raw_candidates_fixed": True, "positive_candidate_injection": False}
    atomic_json(marker, report)
    log(f"sealed complete TRAIN truth {references:,} S1s / {len(keys):,} links; "
        f"{report['singletons']:,} true singletons")
    return keys, counts, report


def _found_truth(truth_keys: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(truth_keys):
        return np.zeros(len(left), dtype=bool)
    query = _positional_keys(left, right)
    positions = np.searchsorted(truth_keys, query)
    found = positions < len(truth_keys)
    found[found] = truth_keys[positions[found]] == query[found]
    return found


def compact_shard(raw: pd.DataFrame, ref: pd.DataFrame, pool_keys: pa.Table,
                  truth_keys: np.ndarray, *, start: int, end: int,
                  max_negatives: int = 40) -> tuple[pd.DataFrame, dict]:
    """Select 40 hardest false edges/S1; never select a missed truth edge."""
    if (not 0 <= max_negatives <= 40 or list(raw.columns) != list(PAIR_DTYPES)
            or raw.empty or raw.left_idx.min() < start or raw.left_idx.max() >= end
            or raw.duplicated(["left_idx", "right_idx"]).any()
            or np.any(np.diff(raw.left_idx.to_numpy()) < 0)
            or pool_keys.num_rows != POOL_ROWS):
        raise ValueError("Compaction requires a complete sorted raw union and original full-pool keys")
    left = raw.left_idx.to_numpy(np.int32)
    right = raw.right_idx.to_numpy(np.int32)
    labels = _found_truth(truth_keys, left, right)
    original = ref.iloc[start:end]
    query = [_text_rank_keys(name, address) for name, address in
             original[["business_name", "business_address"]].itertuples(index=False, name=None)]
    query_core = np.asarray([entry[0] for entry in query], dtype=object)
    query_digits = np.asarray([entry[1] for entry in query], dtype=object)
    unique_right, inverse = np.unique(right, return_inverse=True)
    records = pool_keys.take(pa.array(unique_right))
    target_core = np.asarray(records.column("core").to_pylist(), dtype=object)[inverse]
    target_digits = np.asarray(records.column("digit_set").to_pylist(), dtype=object)[inverse]
    qcore, qdigits = query_core[left - start], query_digits[left - start]
    same_name = (qcore != "") & (qcore == target_core)
    same_digits = (qdigits != "") & (qdigits == target_digits)
    overlap = np.zeros(len(raw), dtype=bool)
    some = np.flatnonzero(same_name & ~same_digits & (qdigits != "") & (target_digits != ""))
    for pos in some:
        overlap[pos] = bool(set(qdigits[pos].split("|")) & set(target_digits[pos].split("|")))
    priority = np.select((same_name & same_digits, same_name & overlap, same_name,
                          same_digits), (0, 1, 2, 3), default=4).astype(np.int8)
    priority[labels] = -1
    negative = pd.DataFrame({"position": np.flatnonzero(~labels), "left_idx": left[~labels],
                             "priority": priority[~labels], "dense_score": raw.dense_score.to_numpy()[~labels],
                             "lexical_key_score": raw.lexical_key_score.to_numpy()[~labels],
                             "right_idx": right[~labels]})
    negative.sort_values(["left_idx", "priority", "dense_score", "lexical_key_score", "right_idx"],
                         ascending=[True, True, False, False, True], kind="stable", inplace=True)
    keep = labels.copy()
    if len(negative) and max_negatives:
        chosen = negative.groupby("left_idx", sort=False).cumcount().to_numpy() < max_negatives
        keep[negative.position.to_numpy(np.int64)[chosen]] = True
    compact = raw.loc[keep].copy().reset_index(drop=True)
    compact["label"] = labels[keep]
    compact["same_name"] = same_name[keep]
    compact["same_digits"] = same_digits[keep]
    compact["hard_negative_priority"] = priority[keep]
    compact = compact[list(JOIN_DTYPES)].astype(JOIN_DTYPES)
    found = int(labels.sum())
    dense_found = int(np.count_nonzero(labels & raw.dense_hit.to_numpy(dtype=bool)))
    negative_counts = np.bincount(compact.loc[~compact.label, "left_idx"].to_numpy(np.int32) - start,
                                  minlength=end - start)
    if (compact.label.sum() != found or np.any(negative_counts > max_negatives)
            or compact.duplicated(["left_idx", "right_idx"]).any()
            or not np.array_equal(compact.label.to_numpy(dtype=bool),
                                   _found_truth(truth_keys, compact.left_idx.to_numpy(np.int32),
                                                compact.right_idx.to_numpy(np.int32)))):
        raise ValueError("TRAIN compaction removed a retrieved positive, injected an edge, or exceeded 40 negatives")
    return compact, {"raw_candidate_pairs": len(raw), "retained_candidate_pairs": len(compact),
                     "retrieved_true_links": found, "dense_retrieved_true_links": dense_found,
                     "retained_negative_pairs": int((~compact.label).sum()),
                     "same_name_same_digit_negatives": int((~compact.label & compact.same_name & compact.same_digits).sum()),
                     "max_negatives_per_s1": max_negatives, "truth_edge_injection": False}


def _oracle_stats(retrieved: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> dict:
    found, truth = retrieved[mask], actual[mask]
    values = np.divide(5. * found, 4 * found + truth, out=np.zeros(len(found), dtype=np.float64),
                       where=(4 * found + truth) > 0)
    values[truth == 0] = 1.
    return {"references": int(mask.sum()), "true_links": int(truth.sum()),
            "retrieved_true_links": int(found.sum()),
            "missed_true_links": int(truth.sum() - found.sum()),
            "true_singletons": int(np.count_nonzero(truth == 0)),
            "complete_match_sets": int(np.count_nonzero((truth > 0) & (truth == found))),
            "oracle_macro_f05_sum": float(values.sum())}


def _shard_retrieval(compact: pd.DataFrame, actual: np.ndarray,
                     country: pd.Series, *, start: int) -> dict:
    labels = compact.label.to_numpy(bool)
    left = compact.left_idx.to_numpy(np.int32) - start
    found = np.bincount(left[labels], minlength=len(actual))
    dense = np.bincount(left[labels & compact.dense_hit.to_numpy(bool)], minlength=len(actual))
    if np.any(found > actual) or np.any(dense > found):
        raise ValueError("A duplicate/foreign retrieved positive inflated a TRAIN truth count")
    countries = country.to_numpy()
    return {"union": {"all": _oracle_stats(found, actual, np.ones(len(actual), dtype=bool)),
                      **{value: _oracle_stats(found, actual, countries == value)
                         for value in sorted(set(countries))}},
            "dense_only": {"all": _oracle_stats(dense, actual, np.ones(len(actual), dtype=bool)),
                           **{value: _oracle_stats(dense, actual, countries == value)
                              for value in sorted(set(countries))}}}


def _summarize_retrieval(parts: list[dict], *, references: int,
                         max_negatives: int) -> dict:
    groups: dict[str, dict[str, Counter]] = {name: {} for name in ("union", "dense_only")}
    counts = Counter()
    for part in parts:
        counts.update({key: value for key, value in part["compaction"].items()
                       if isinstance(value, int) and key != "max_negatives_per_s1"})
        for branch, categories in part["retrieval"].items():
            for category, value in categories.items():
                groups[branch].setdefault(category, Counter()).update(value)
    if groups["union"]["all"]["references"] != references:
        raise ValueError("All S1s, including true singletons, must appear in oracle accounting")
    result = {}
    for branch, by_country in groups.items():
        result[branch] = {}
        for country, totals in by_country.items():
            n = totals["references"]
            truth = totals["true_links"]
            result[branch][country] = dict(totals)
            result[branch][country].update(
                link_recall=totals["retrieved_true_links"] / truth if truth else None,
                oracle_macro_f05=totals["oracle_macro_f05_sum"] / n if n else None,
                oracle_false_links=0, oracle_singleton_accuracy=1.0 if totals["true_singletons"] else None,
                oracle_precision=1.0 if totals["retrieved_true_links"] else None)
    total = counts["retained_candidate_pairs"]
    if (result["union"]["all"]["retrieved_true_links"] != counts["retrieved_true_links"]
            or result["dense_only"]["all"]["retrieved_true_links"] != counts["dense_retrieved_true_links"]
            or result["union"]["all"]["retrieved_true_links"] > result["union"]["all"]["true_links"]
            or total != counts["retained_negative_pairs"] + counts["retrieved_true_links"]):
        raise ValueError("Full-group truth and hard-negative accounting disagree")
    return {"counts": dict(counts), "retrieval": result,
            "raw_pairs_per_s1": counts["raw_candidate_pairs"] / references,
            "retained_pairs_per_s1": total / references,
            "max_negative_pairs_per_s1": max_negatives,
            "candidate_cap_applied_after_raw_union": True,
            "missed_truth_in_denominator": True, "truth_edge_injection": False,
            "scope": "TRAIN retrieval oracle, perfect decisions only; NOT a learned matcher or TUNE/test score"}


def _assembled_pairs(directory: Path, joined: list[dict]) -> dict:
    path = directory / "candidate_pairs.parquet"
    expected = sum(item["file"]["rows"] for item in joined)
    if path.exists():
        existing = json.loads((directory / "assembly.json").read_text()) if (directory / "assembly.json").exists() else None
        if (existing is None or existing["input_join_progress_sha256"] != file_sha256(directory / "join_progress.json")
                or existing["candidate_pairs"]["rows"] != expected):
            raise ValueError("Unclaimed full candidate Parquet exists without its assembly identity")
        checked_file(path, existing["candidate_pairs"], parquet=True)
        return existing["candidate_pairs"]
    temp = Path(str(path) + ".tmp")
    if temp.exists():
        raise ValueError("Interrupted final pair assembly needs inspection before retry")
    writer = None
    count = 0
    try:
        for item in joined:
            shard = directory / "joined" / item["file"]["path"]
            checked_file(shard, item["file"], parquet=True)
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=100_000, use_threads=False):
                if writer is None:
                    writer = pq.ParquetWriter(temp, batch.schema, compression="zstd")
                elif not writer.schema.equals(batch.schema, check_metadata=False):
                    raise ValueError("A joined shard changed the P3 candidate feature schema/order")
                writer.write_batch(batch)
                count += len(batch)
    finally:
        if writer is not None:
            writer.close()
    if count != expected or pq.ParquetFile(temp).metadata.num_rows != count:
        raise ValueError("Assembly omitted/repeated a selected TRAIN candidate pair")
    with temp.open("rb") as stream:
        os.fsync(stream.fileno())
    temp.replace(path)
    record = identity(path, parquet=True)
    atomic_json(directory / "assembly.json", {"state": "complete", "input_join_progress_sha256":
                file_sha256(directory / "join_progress.json"), "candidate_pairs": record})
    return record


def _paired_screen_interval(directory: Path, counts: np.ndarray, references: int) -> dict:
    from .uncertainty import paired_bootstrap

    union = np.zeros(references, dtype=np.int32)
    dense = np.zeros(references, dtype=np.int32)
    for item in json.loads((directory / "join_progress.json").read_text())["parts"]:
        frame = pd.read_parquet(directory / "joined" / item["file"]["path"],
                                columns=["left_idx", "label", "dense_hit"])
        left = frame.left_idx.to_numpy(np.int32)
        found = frame.label.to_numpy(bool)
        union += np.bincount(left[found], minlength=references)
        dense += np.bincount(left[found & frame.dense_hit.to_numpy(bool)], minlength=references)

    def per_entity(found: np.ndarray) -> np.ndarray:
        score = np.divide(5. * found, 4 * found + counts, out=np.zeros(references, dtype=np.float64),
                          where=(4 * found + counts) > 0)
        score[counts == 0] = 1.
        return score

    return paired_bootstrap(per_entity(union), per_entity(dense), n_bootstrap=2_000,
                            seed=20260925)


def _validate_fixed_raw(inputs: UnionInputs, directory: Path, raw: dict, references: int) -> None:
    """Verify candidate subset/country/duplicates BEFORE opening any truth value."""
    cursor = 0
    for part in raw["parts"]:
        start, end = part["start"], part["end"]
        if start != cursor or end != min(start + SHARD_REFERENCES, references):
            raise ValueError("Raw candidate S1 shards do not cover each selected reference")
        path = directory / "raw" / part["file"]["path"]
        checked_file(path, part["file"], parquet=True)
        parquet = pq.ParquetFile(path)
        if list(parquet.schema_arrow.names) != list(PAIR_DTYPES):
            raise ValueError("Raw candidate features changed or contain label columns")
        prior_left, prior_right = -1, -1
        dense_count = 0
        for batch in parquet.iter_batches(batch_size=100_000,
                                          columns=["left_idx", "right_idx", "dense_hit"], use_threads=False):
            left = batch.column(0).to_numpy()
            right = batch.column(1).to_numpy()
            dense_hit = batch.column(2).to_numpy(zero_copy_only=False)
            if len(left) == 0:
                continue
            if (left.min() < start or left.max() >= end or right.min() < 0 or right.max() >= POOL_ROWS
                    or np.any(left[1:] < left[:-1])
                    or np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1]))
                    or (left[0] == prior_left and right[0] <= prior_right)
                    or not np.array_equal(inputs.query_country[left], inputs.countries[right])):
                raise ValueError("Raw candidates contain a duplicate, country error or foreign S1/target row")
            prior_left, prior_right = int(left[-1]), int(right[-1])
            dense_count += int(dense_hit.sum())
        if dense_count != (end - start) * 50:
            raise ValueError("The fixed raw union omitted a dense top-50 retrieval edge")
        cursor = end
    if cursor != references:
        raise ValueError("The raw candidate union omitted a selected S1 interval")


def finalize(inputs: UnionInputs, directory: Path, progress: dict, truth: dict,
             summary: dict, *, references: int, screen: bool, max_negatives: int) -> dict:
    """Atomic P3 TRAIN handoff after all independently hashed shards are complete."""
    if progress["state"] != "complete" or len(progress["parts"]) != (references + SHARD_REFERENCES - 1) // SHARD_REFERENCES:
        raise ValueError("P2.5 cannot publish a partial TRAIN truth join")
    candidate = _assembled_pairs(directory, progress["parts"])
    ref_path = directory / "reference.parquet"
    if screen:
        if not ref_path.exists():
            inputs.reference.iloc[:references].to_parquet(str(ref_path) + ".tmp", index=False,
                                                           compression="zstd")
            Path(str(ref_path) + ".tmp").replace(ref_path)
    elif ref_path.resolve() != (inputs.durable / "full" / "reference.parquet").resolve():
        raise ValueError("The complete TRAIN reference must be the sealed fit-only original-row artifact")
    ref_info = identity(ref_path, parquet=True)
    pool_path = inputs.durable / "pool.parquet"
    pool_info = identity(pool_path, parquet=True)
    truth_path = directory / "train_truth_pairs.parquet"
    truth_info = identity(truth_path, parquet=True)
    if (ref_info["rows"] != references or pool_info["rows"] != POOL_ROWS
            or truth_info["rows"] != truth["true_links"]
            or candidate["rows"] != summary["counts"]["retained_candidate_pairs"]
            or summary["retrieval"]["union"]["all"]["true_links"] != truth_info["rows"]):
        raise ValueError("Final P3 record/pair/truth files do not match the full population")
    actual_columns = pq.ParquetFile(directory / "candidate_pairs.parquet").schema_arrow.names
    tune_columns = ["phonetic_score", "phonetic_only", "dense_score", "dense_only",
                    "lexical_scores_missing", "lexical_retrieval_score"]
    if (not {"retrieval_score", "retrieval_rank", "dense_score", "label", "dense_rank",
             "same_name", "same_digits", "hard_negative_priority"} <= set(actual_columns)
            or [name for name in tune_columns if name in actual_columns] != tune_columns):
        raise ValueError("P3 tree/CE optional retrieval features or TRAIN hard-negative fields are missing")
    ready = {"schema": 1, "state": "complete", "mode": "screen_10k" if screen else "full",
             "screen_only": bool(screen), "split": "train", "dense_mining_complete": True,
             "train_truth_join_complete": True, "positive_candidate_injection": False,
             "truth_used_in_retrieval": False, "full_pool_records": POOL_ROWS,
             "train_reference_id_sha256": ids_sha256(inputs.reference.entity_id.iloc[:references]),
             "reservation_sha256": inputs.bootstrap["reservation_sha256"],
             "dense_complete_sha256": inputs.bootstrap["dense_complete_sha256"],
             "lexical_index_manifest_sha256": inputs.bootstrap["index_manifest_sha256"],
              "raw_progress_sha256": file_sha256(directory / "raw_progress.json"),
              "raw_complete_sha256": file_sha256(directory / "raw_complete.json")
              if (directory / "raw_complete.json").is_file() else None,
              "truth_complete_sha256": file_sha256(directory / "truth_complete.json"),
             "join_progress_sha256": file_sha256(directory / "join_progress.json"),
             "max_hard_negatives_per_train_s1": max_negatives,
             "files": {"reference": {**ref_info, "path": "reference.parquet"},
                       "pool": {**pool_info, "path": "../pool.parquet"},
                       "candidate_pairs": {**candidate, "path": "candidate_pairs.parquet"},
                       "train_truth_pairs": {**truth_info, "path": "train_truth_pairs.parquet"}},
             "retrieval": summary, "truth_links_not_in_candidate_pairs":
             truth["true_links"] - summary["counts"]["retrieved_true_links"],
             "pool_includes_unretrieved_truth_records": True,
             "features": {"rank_and_cosine": "full, pre-compaction raw union; new-only cosines from saved E5 vectors",
                          "relative_features": "recompute in P3 from complete *retained* per-S1 groups",
                          "train_vs_tune": "TRAIN includes dense+new-key branches; medium TUNE also has legacy lexical candidate edges. P3 masks legacy scores on BOTH using its dense_key_shared feature policy"}}
    if screen:
        ready["screen_retrieval_paired"] = _paired_screen_interval(directory, np.bincount(
            pd.read_parquet(truth_path, columns=["left_idx"]).left_idx.to_numpy(np.int32),
            minlength=references), references)
    atomic_json(directory / "ready.json", ready)
    log(f"READY {'screen' if screen else 'FULL TRAIN'}: {references:,} S1s; "
        f"{candidate['rows']:,} candidate edges; {truth['true_links']:,} true links; "
        f"retrieval oracle F0.5={summary['retrieval']['union']['all']['oracle_macro_f05']:.6f}")
    return ready


def join_and_publish(inputs: UnionInputs, raw_dir: Path, directory: Path,
                     *, screen: bool = False, max_negatives: int = 40) -> dict:
    references = SHARD_REFERENCES if screen else TRAIN_REFERENCES
    ready_path = directory / "ready.json"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        for name, file in ready["files"].items():
            checked_file((directory / file["path"]).resolve(), file, parquet=True)
        return ready
    raw_path = directory / "raw_progress.json"
    if screen:
        raw = json.loads(raw_path.read_text())
        if (raw["state"] != "complete" or raw["references_done"] != references
                or raw["identity"] != _raw_identity(inputs)):
            raise ValueError("All TRAIN candidate edges must be fixed before truth join")
    else:
        raw, _ = _sealed_raw(inputs, directory, references, screen=False)
    _validate_fixed_raw(inputs, directory, raw, references)
    keys, counts, truth = selected_truth(inputs, raw_dir, directory, references=references)
    with pa.memory_map(str(inputs.durable / "pool_keys.arrow"), "r") as stream:
        pool_keys = ipc.open_file(stream).read_all()
    marker = directory / "join_progress.json"
    existing = json.loads(marker.read_text()) if marker.is_file() else None
    source = {"raw_complete_sha256": file_sha256(directory / "raw_complete.json") if not screen else None,
              "raw_progress_sha256": file_sha256(raw_path) if screen else None,
              "truth_complete_sha256": file_sha256(directory / "truth_complete.json"),
              "max_negatives_per_s1": max_negatives, "references": references}
    progress = existing or {"state": "joining", "identity": source, "references_done": 0, "parts": []}
    if progress["identity"] != source:
        raise ValueError("A prior joined shard used other raw candidates, truth or hard-negative rules")
    for index, raw_part in enumerate(raw["parts"]):
        if index < len(progress["parts"]):
            saved = progress["parts"][index]
            if (saved["start"] != raw_part["start"] or saved["end"] != raw_part["end"]
                    or saved["raw_sha256"] != raw_part["file"]["sha256"]):
                raise ValueError("Joined TRAIN shard does not belong to the saved raw S1 interval")
            checked_file(directory / "joined" / saved["file"]["path"], saved["file"], parquet=True)
            continue
        start, end = raw_part["start"], raw_part["end"]
        path = directory / "raw" / raw_part["file"]["path"]
        checked_file(path, raw_part["file"], parquet=True)
        frame = pd.read_parquet(path)
        left, right = frame.left_idx.to_numpy(np.int32), frame.right_idx.to_numpy(np.int32)
        if (not np.array_equal(inputs.query_country[left], inputs.countries[right])
                or np.any((right < 0) | (right >= POOL_ROWS))):
            raise ValueError("Raw TRAIN candidates crossed countries or original target bounds")
        start_key = np.uint64(start) << np.uint64(32)
        end_key = np.uint64(end) << np.uint64(32)
        local_truth = keys[np.searchsorted(keys, start_key):np.searchsorted(keys, end_key)]
        compact, stats = compact_shard(frame, inputs.reference, pool_keys, local_truth,
                                       start=start, end=end, max_negatives=max_negatives)
        country = inputs.reference.country.iloc[start:end]
        audit = _shard_retrieval(compact, counts[start:end], country, start=start)
        target = directory / "joined" / f"part_{index:06d}.parquet"
        file = _write_parquet(compact, target)
        progress["parts"].append({"start": start, "end": end,
                                  "raw_sha256": raw_part["file"]["sha256"],
                                  "file": {**file, "path": target.name},
                                  "compaction": stats, "retrieval": audit})
        progress["references_done"] = end
        atomic_json(marker, progress)
        log(f"JOINED {end:,}/{references:,} eligible TRAIN S1s, "
            f"{stats['retrieved_true_links']:,} retrieved positives / {stats['retained_negative_pairs']:,} hard negatives in shard")
    if progress["references_done"] != references:
        raise ValueError("TRAIN truth join ended before every eligible S1, including singletons")
    progress["state"] = "complete"
    atomic_json(marker, progress)
    summary = _summarize_retrieval(progress["parts"], references=references,
                                   max_negatives=max_negatives)
    atomic_json(directory / "retrieval_report.json", summary)
    return finalize(inputs, directory, progress, truth, summary, references=references,
                    screen=screen, max_negatives=max_negatives)


def run(args) -> dict:
    if not 0 <= args.max_negatives <= 40 or args.poll_seconds <= 0:
        raise ValueError("TRAIN hard negatives must be capped at 40 per S1 and polling must be positive")
    bootstrap(args.dense, args.staged, args.raw_dir, args.reservation, args.scratch, args.durable)
    screen = args.action == "screen"
    directory = args.durable / ("screen" if screen else "full")
    raw_marker = directory / "raw_progress.json"
    prior = json.loads(raw_marker.read_text()) if raw_marker.is_file() else {}
    raw_completed = prior.get("state") == "complete"
    next_start = prior.get("references_done", 0)
    limit = SHARD_REFERENCES if screen else TRAIN_REFERENCES
    staged_prefix = checked_stage(args.staged)["references_staged"]
    preload = (not raw_completed and next_start < limit
               and staged_prefix >= min(next_start + SHARD_REFERENCES, limit))
    inputs = UnionInputs(args.dense, args.staged, args.durable,
                         pool_vector_residency=args.pool_vector_residency,
                         preload_vectors=preload)
    progress = produce_raw(inputs, directory, screen=screen, follow=args.follow,
                           poll_seconds=args.poll_seconds)
    if progress["state"] != "complete":
        log(f"raw candidate prefix ready {progress['references_done']:,}/{TRAIN_REFERENCES:,}; "
            "lexical mining/staging still in progress")
        return progress
    # Full truth indexing and 40-negative compaction do not use the 8-GB E5
    # tensor. Give that RAM back before opening selected labels on a 50-GB pod.
    inputs.release_pool_vectors()
    inputs.vectors = None
    gc.collect()
    return join_and_publish(inputs, args.raw_dir, directory, screen=screen,
                            max_negatives=args.max_negatives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    stage = sub.add_parser("stage", help="Read-only transfer of committed local lexical parts to owned pod scratch")
    stage.add_argument("--lexical", type=Path, required=True)
    stage.add_argument("--index-manifest", type=Path, required=True)
    stage.add_argument("--local-cache", type=Path, default=Path("/tmp/az-ml-v2/train_union"))
    stage.add_argument("--remote", default="/tmp/az-ml-v2/train_union/lexical")
    stage.add_argument("--host", required=True)
    stage.add_argument("--ssh-key", type=Path, required=True)
    stage.add_argument("--ssh-port", type=int, default=22141)
    stage.add_argument("--follow", action="store_true")
    stage.add_argument("--max-parts", type=int, help="Bound a first compatibility screen to committed parts")
    stage.add_argument("--poll-seconds", type=float, default=30.)
    for action in ("screen", "full"):
        command = sub.add_parser(action, help=f"{action}: label-free raw merge then selected TRAIN truth join")
        command.add_argument("--dense", type=Path, default=Path("/tmp/az-ml-v2/dense"))
        command.add_argument("--staged", type=Path, default=Path("/tmp/az-ml-v2/train_union/lexical"))
        command.add_argument("--raw-dir", type=Path, default=Path("/tmp/az-ml-new/data/raw/dataset/train"))
        command.add_argument("--reservation", type=Path, default=Path("/workspace/az-ml-new/v2/holdout2"))
        command.add_argument("--scratch", type=Path, default=Path("/tmp/az-ml-v2/train_union"))
        command.add_argument("--durable", type=Path, default=Path("/workspace/az-ml-new/v2/train_union"))
        command.add_argument("--max-negatives", type=int, default=40)
        command.add_argument("--poll-seconds", type=float, default=30.)
        command.add_argument("--follow", action="store_true", help="Wait for committed lexical partitions")
        command.add_argument("--pool-vector-residency", choices=("mmap", "ram"), default="ram",
                             help="RAM requires saved E5 pool + 4 GiB effective cgroup headroom; avoids slow object-backed random reads")
    args = parser.parse_args()
    if args.action == "stage":
        result = stage_lexical(args.lexical, args.index_manifest, args.local_cache,
                               args.remote, args.host, args.ssh_key, args.ssh_port,
                               follow=args.follow, max_parts=args.max_parts,
                               poll_seconds=args.poll_seconds)
    else:
        result = run(args)
    print(json.dumps({"state": result["state"], "mode": args.action,
                      "references": result.get("references_done", result.get("files", {}).get("reference", {}).get("rows")),
                      "candidate_pairs": result.get("files", {}).get("candidate_pairs", {}).get("rows")},
                     allow_nan=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
