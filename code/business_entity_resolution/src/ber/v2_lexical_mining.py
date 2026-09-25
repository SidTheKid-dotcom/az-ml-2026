"""Label-free, full-pool exact-text, phonetic and address-number retrieval.

The SQLite index covers *every* S2/S3 row in original file order. Its separate
canonical maps group only byte-identical (country, name, address) records. Search
selects groups, then expands every selected group back to its original raw rows;
no representative replaces an entity ID in the resulting candidate set. The
existing full-pool character-TF-IDF lexical/phonetic medium_hard candidates can
be unioned with these additional branches by :mod:`ber.v2_candidate_union`.

No ground-truth path is accepted by the builder or miner. Large postings are
screened around the query's name in an indexed *full* target pool, never by
sampling a competing pool. The screen is approximate and its capped-query count
is reported; exact composite-key postings are always read without a cap.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from .normalization import normalize_text
from .phonetic_ablation import _skeleton, _LEGAL


POOL_ROWS = 10_320_219
TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
_SUFFIXES = frozenset("llc llp ltd limited inc incorporated corp corporation company co pvt private plc".split())
_DIGITS = re.compile(r"\d+")
_COLUMNS = ("raw_row", "entity_id", "raw_country", "name", "address", "name_norm", "core", "phonetic", "digit_seq", "digit_set")
_SELECT = ", ".join(_COLUMNS)
_BRANCHES = {
    "exact_name": ("name_norm", "name_exact_hit"),
    "phonetic": ("phonetic", "phonetic_hit"),
    "digit_sequence": ("digit_seq", "digit_sequence_hit"),
    "digit_set": ("digit_set", "digit_set_hit"),
}
_PAIR_DTYPES = {
    "left_idx": "int32", "right_raw_idx": "int32", "canonical_raw_idx": "int32",
    "canonical_group_size": "int32", "lexical_key_score": "float32",
    "name_exact_hit": "bool", "phonetic_hit": "bool", "digit_sequence_hit": "bool",
    "digit_set_hit": "bool", "composite_key_hit": "bool",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _phonetic_from_normalized(name: str) -> str:
    # Identical to phonetic_text(name, "harmonized_core") without transliterating
    # the same raw name twice during full-pool indexing.
    return " ".join(part for token in name.split()
                    if (part := _skeleton(token, True)) and part not in _LEGAL)


@dataclass(frozen=True)
class TextKeys:
    country: str
    name: str
    core: str
    phonetic: str
    address: str
    digit_seq: str
    digit_set: str


def text_keys(name: str, address: str, country: str) -> TextKeys:
    normalized = normalize_text(name)
    words = normalized.split()
    while words and words[-1] in _SUFFIXES:
        words.pop()
    core = " ".join(words)
    address_norm = normalize_text(address)
    digits = _DIGITS.findall(address_norm)
    return TextKeys(normalize_text(country), normalized, core,
                    _phonetic_from_normalized(normalized), address_norm,
                    "|".join(digits), "|".join(sorted({x.lstrip("0") or "0" for x in digits})))


def _sources(raw_dir: Path, expected: tuple[int, int] | None) -> list[dict]:
    paths = [raw_dir / f"train_source{source}.tsv" for source in (2, 3)]
    result = []
    for i, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({"name": path.name, "path": str(path.resolve()),
                       "bytes": path.stat().st_size, "sha256": file_sha256(path),
                       "expected_rows": expected[i] if expected is not None else None})
    return result


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
    else:
        connection = sqlite3.connect(path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")  # 64 MiB; leave RAM to other work
    return connection


def _finish_group(group: list[int], canonical: np.ndarray, members: np.ndarray,
                  starts: np.ndarray, sizes: np.ndarray, offset: int) -> int:
    rep = min(group)
    positions = np.asarray(group, dtype=np.int32)
    canonical[positions] = rep
    members[offset:offset + len(group)] = positions
    starts[rep] = offset
    sizes[rep] = len(group)
    return offset + len(group)


def build_pool_index(raw_dir: Path, output: Path, *, chunksize: int = 50_000,
                     expected_rows: tuple[int, int] | None = (5_034_616, 5_285_603)) -> dict:
    """Build/resume a durable exact original-row index without reading labels."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    raw_dir, output = Path(raw_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = _sources(raw_dir, expected_rows)
    complete_path, index_path = output / "complete.json", output / "pool.sqlite"
    if complete_path.exists():
        saved = json.loads(complete_path.read_text())
        if saved.get("state") != "complete" or saved["sources"] != sources:
            raise ValueError("Completed index has a different full-pool identity")
        if any((output / name).stat().st_size != info["bytes"]
               for name, info in saved["files"].items()):
            raise ValueError("Completed index files changed")
        return saved
    building, progress = output / "pool.sqlite.building", output / "build_progress.json"
    if index_path.exists():
        raise ValueError("Index exists without a completion marker; investigate before overwriting it")
    if building.exists() != progress.exists():
        raise ValueError("Partial index/progress disagreement; investigate before resuming")
    if progress.exists():
        previous = json.loads(progress.read_text())
        if previous["sources"] != sources or previous["schema"] != 1:
            raise ValueError("Partial index belongs to different source bytes or schema")
    else:
        atomic_json(progress, {"schema": 1, "sources": sources, "stage": "ingesting", "rows": 0})
    started = time.monotonic()
    db = _connect(building)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS records ("
                   "raw_row INTEGER PRIMARY KEY, entity_id TEXT NOT NULL, country TEXT NOT NULL, "
                   "raw_country TEXT NOT NULL, name TEXT NOT NULL, address TEXT NOT NULL, "
                   "name_norm TEXT NOT NULL, core TEXT NOT NULL, phonetic TEXT NOT NULL, "
                   "digit_seq TEXT NOT NULL, digit_set TEXT NOT NULL)")
        already = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        if already > sum(item["expected_rows"] or 0 for item in sources) and expected_rows:
            raise ValueError("Partial index contains too many rows")
        offset = 0
        for i, source in enumerate(sources):
            count = 0
            for frame in pd.read_csv(source["path"], sep="\t", dtype=str, keep_default_na=False,
                                     usecols=["entity_id", "business_name", "business_address", "country"],
                                     chunksize=chunksize):
                if expected_rows is not None and count + len(frame) > source["expected_rows"]:
                    raise ValueError("A source has more records than expected")
                count += len(frame)
                stop = offset + len(frame)
                if already >= stop:
                    offset = stop
                    continue
                if already > offset:
                    frame = frame.iloc[already - offset:]
                    offset = already
                batch = []
                for raw_row, (entity, name, address, country) in enumerate(
                    frame.itertuples(index=False, name=None), offset):
                    if not entity.startswith(f"S{i+2}-"):
                        raise ValueError(f"Invalid source-{i+2} target ID prefix")
                    key = text_keys(name, address, country)
                    batch.append((raw_row, entity, key.country, country, name, address,
                                  key.name, key.core, key.phonetic, key.digit_seq, key.digit_set))
                db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
                db.commit()
                offset = stop
                if offset % (chunksize * 20) < chunksize:
                    atomic_json(progress, {"schema": 1, "sources": sources,
                                           "stage": "ingesting", "rows": offset,
                                           "elapsed_seconds": time.monotonic() - started})
                    print(f"pool index: {offset:,} raw rows committed", flush=True)
            if source["expected_rows"] is not None and count != source["expected_rows"]:
                raise ValueError(f"{source['name']} row count changed: {count}")
        total = offset
        if total == 0 or total >= np.iinfo(np.int32).max or db.execute("SELECT COUNT(*) FROM records").fetchone()[0] != total:
            raise ValueError("Incomplete/oversized pool index")
        atomic_json(progress, {"schema": 1, "sources": sources, "stage": "indexing", "rows": total})
        for name, columns in (
            ("ix_entity_id", "entity_id"),
            ("ix_exact_raw", "raw_country, name, address"),
            ("ix_name", "country, name_norm, core"),
            ("ix_phonetic", "country, phonetic, core"),
            ("ix_digit", "country, digit_seq, core"),
            ("ix_set", "country, digit_set, core"),
            ("ix_composite", "country, core, digit_seq"),
        ):
            constraint = "UNIQUE " if name == "ix_entity_id" else ""
            db.execute(f"CREATE {constraint}INDEX IF NOT EXISTS {name} ON records ({columns})")
            db.commit()
            print(f"pool index: built {name}", flush=True)
        atomic_json(progress, {"schema": 1, "sources": sources, "stage": "grouping", "rows": total})
        paths = {name: output / f"{name}.npy.tmp" for name in ("canonical_of", "group_members", "group_starts", "group_sizes")}
        arrays = {name: np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(total,))
                  for name, path in paths.items()}
        canonical, members, starts, sizes = (arrays[name] for name in paths)
        canonical[:] = -1
        starts[:] = sizes[:] = 0
        cursor, group, old_key, groups, next_log = 0, [], None, 0, 1_000_000
        for raw, country, name, address in db.execute(
                "SELECT raw_row, raw_country, name, address FROM records "
                "ORDER BY raw_country, name, address, raw_row"):
            key = (country, name, address)
            if group and key != old_key:
                cursor = _finish_group(group, canonical, members, starts, sizes, cursor)
                groups += 1
                group = []
                if cursor >= next_log:
                    print(f"pool canonicalization: {cursor:,}/{total:,} exact original rows", flush=True)
                    next_log += 1_000_000
            group.append(raw)
            old_key = key
        if group:
            cursor = _finish_group(group, canonical, members, starts, sizes, cursor)
            groups += 1
        if cursor != total or np.any(canonical < 0) or int(sizes.sum(dtype=np.int64)) != total:
            raise ValueError("Exact group expansion did not cover every original target")
        for array in arrays.values():
            array.flush()
        del arrays, canonical, members, starts, sizes
        db.close()
        for name, path in paths.items():
            path.replace(output / f"{name}.npy")
        building.replace(index_path)
        file_names = ("pool.sqlite", "canonical_of.npy", "group_members.npy", "group_starts.npy", "group_sizes.npy")
        files = {name: {"bytes": (output / name).stat().st_size,
                        "sha256": file_sha256(output / name)} for name in file_names}
        report = {"state": "complete", "schema": 1, "rows": total,
                  "source_order": [item["name"] for item in sources], "sources": sources,
                  "canonical_groups": groups, "duplicate_rows": total - groups,
                  "files": files, "elapsed_seconds": time.monotonic() - started,
                  "country_normalization": "ber.normalization.normalize_text",
                  "canonical_group_key": "exact raw country, business name and business address; sources may mix within a group",
                  "raw_row_mapping": "0-based: every train_source2 row, then every train_source3 row",
                  "truth_used": False, "target_pool_sampled": False}
        atomic_json(complete_path, report)
        print(f"pool index COMPLETE: {total:,} rows, {groups:,} exact groups, {total-groups:,} duplicates", flush=True)
        return report
    finally:
        db.close()


class FullPoolIndex:
    """Read-only original-row lookup with exact, lossless group expansion."""

    def __init__(self, directory: Path, *, verify_hashes: bool = False):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "complete.json").read_text())
        if self.manifest.get("state") != "complete" or self.manifest.get("schema") != 1:
            raise ValueError("Full-pool index is incomplete")
        for name, info in self.manifest["files"].items():
            path = self.directory / name
            if path.stat().st_size != info["bytes"] or (verify_hashes and file_sha256(path) != info["sha256"]):
                raise ValueError(f"Full-pool index artifact changed: {path}")
        self.db = _connect(self.directory / "pool.sqlite", readonly=True)
        self.canonical = np.load(self.directory / "canonical_of.npy", mmap_mode="r")
        self.members = np.load(self.directory / "group_members.npy", mmap_mode="r")
        self.starts = np.load(self.directory / "group_starts.npy", mmap_mode="r")
        self.sizes = np.load(self.directory / "group_sizes.npy", mmap_mode="r")
        if any(array.shape != (self.manifest["rows"],) or array.dtype != np.int32
               for array in (self.canonical, self.members, self.starts, self.sizes)):
            raise ValueError("Full-pool canonical mappings have invalid shape/dtype")

    def close(self) -> None:
        self.db.close()

    def expand(self, raw_row: int) -> np.ndarray:
        if not 0 <= raw_row < self.manifest["rows"]:
            raise IndexError("Target row is outside the original complete pool")
        rep = int(self.canonical[raw_row])
        start, count = int(self.starts[rep]), int(self.sizes[rep])
        if count < 1 or not (start <= self.manifest["rows"] - count):
            raise ValueError("Canonical group mapping lost a target")
        return self.members[start:start + count]

    def lookup_ids(self, values: list[str]) -> dict[str, tuple]:
        """Resolve original IDs, never their numeric portions, to raw rows/text."""
        result = {}
        for start in range(0, len(values), 500):
            chunk = values[start:start + 500]
            if not chunk:
                continue
            query = f"SELECT {_SELECT}, country FROM records WHERE entity_id IN ({','.join('?' * len(chunk))})"
            for row in self.db.execute(query, chunk):
                result[row[1]] = row
        if len(result) != len(set(values)):
            raise ValueError("A target ID is absent from the full S2/S3 pool")
        return result

    def lookup_rows(self, values: list[int]) -> dict[int, tuple]:
        result = {}
        for start in range(0, len(values), 500):
            chunk = [int(value) for value in values[start:start + 500]]
            if not chunk:
                continue
            query = f"SELECT {_SELECT}, country FROM records WHERE raw_row IN ({','.join('?' * len(chunk))})"
            for row in self.db.execute(query, chunk):
                result[row[0]] = row
        if len(result) != len(set(values)):
            raise ValueError("A raw row is absent from the full S2/S3 pool")
        return result

    def _posting(self, country: str, column: str, value: str, core: str,
                 max_scan: int, counts: Counter) -> list[tuple]:
        if not value:
            return []
        count = self.db.execute(f"SELECT COUNT(*) FROM records WHERE country=? AND {column}=?",
                                (country, value)).fetchone()[0]
        counts[f"{column}_queries"] += 1
        counts[f"{column}_matching_rows"] += count
        if count <= max_scan:
            found = list(self.db.execute(f"SELECT {_SELECT} FROM records WHERE country=? AND {column}=?",
                                         (country, value)))
        else:
            counts[f"{column}_capped_queries"] += 1
            half = max(1, (max_scan - min(32, max_scan // 8)) // 2)
            found = list(self.db.execute(
                f"SELECT {_SELECT} FROM records WHERE country=? AND {column}=? AND core>=? "
                f"ORDER BY core, raw_row LIMIT ?", (country, value, core, half)))
            found += list(self.db.execute(
                f"SELECT {_SELECT} FROM records WHERE country=? AND {column}=? AND core<? "
                f"ORDER BY core DESC, raw_row DESC LIMIT ?", (country, value, core, half)))
            remaining = max_scan - 2 * half
            if remaining:
                found += list(self.db.execute(
                    f"SELECT {_SELECT} FROM records WHERE country=? AND {column}=? "
                    "ORDER BY core, raw_row LIMIT ?", (country, value, remaining)))
        counts["posting_rows_read"] += len(found)
        return found

    def search(self, name: str, address: str, country: str, *, max_scan: int = 512,
               name_k: int = 30, phonetic_k: int = 20, digit_k: int = 20,
               digit_set_k: int = 10) -> tuple[list[dict], Counter]:
        """Select text/number groups, then expand all members in original row order."""
        if min(max_scan, name_k, phonetic_k, digit_k, digit_set_k) < 1:
            raise ValueError("Posting scan and branch group limits must be positive")
        keys = text_keys(name, address, country)
        counts = Counter()
        selected: dict[int, dict] = {}
        for branch, (column, flag) in _BRANCHES.items():
            value = {"exact_name": keys.name, "phonetic": keys.phonetic,
                     "digit_sequence": keys.digit_seq, "digit_set": keys.digit_set}[branch]
            if not value:
                continue
            found = self._posting(keys.country, column, value, keys.core or keys.name, max_scan, counts)
            representatives = {}
            for row in found:
                rep = int(self.canonical[row[0]])
                representatives.setdefault(rep, row)
            ranked = sorted(representatives.items(), key=lambda item: (
                -_text_score(keys, item[1]), item[0]))
            limit = {"exact_name": name_k, "phonetic": phonetic_k,
                     "digit_sequence": digit_k, "digit_set": digit_set_k}[branch]
            for rep, row in ranked[:limit]:
                selected.setdefault(rep, {"score": 0., "flags": set()})["flags"].add(flag)
                selected[rep]["score"] = max(selected[rep]["score"], _text_score(keys, row))
            counts[f"{branch}_selected_groups"] += min(len(ranked), limit)
        if keys.core and keys.digit_seq:
            found = self.db.execute(
                f"SELECT {_SELECT} FROM records WHERE country=? AND core=? AND digit_seq=?",
                (keys.country, keys.core, keys.digit_seq))
            for row in found:
                rep = int(self.canonical[row[0]])
                selected.setdefault(rep, {"score": 0., "flags": set()})["flags"].add("composite_key_hit")
                selected[rep]["score"] = max(selected[rep]["score"], _text_score(keys, row))
                counts["composite_posting_rows"] += 1
        output = []
        for rep, value in sorted(selected.items(), key=lambda item: (-item[1]["score"], item[0])):
            for raw in self.expand(rep):
                item = {"right_raw_idx": int(raw), "canonical_raw_idx": rep,
                        "canonical_group_size": int(self.sizes[rep]),
                        "lexical_key_score": np.float32(value["score"])}
                item.update({column: column in value["flags"] for column in
                             ("name_exact_hit", "phonetic_hit", "digit_sequence_hit",
                              "digit_set_hit", "composite_key_hit")})
                output.append(item)
        counts["selected_representatives"] = len(selected)
        counts["expanded_candidates"] = len(output)
        counts["duplicate_expansion_pairs"] = len(output) - len(selected)
        if len({row["right_raw_idx"] for row in output}) != len(output):
            raise ValueError("Candidate branches produced duplicate original targets")
        return output, counts


def _text_score(query: TextKeys, row: tuple) -> float:
    name = fuzz.WRatio(query.core or query.name, row[6] or row[5]) / 100 if query.name and row[5] else 0.
    address = fuzz.token_set_ratio(query.address, normalize_text(row[4])) / 100 if query.address and row[4] else 0.
    return .7 * name + .3 * address


def _coerce_pairs(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in _PAIR_DTYPES.items()})
    frame = pd.DataFrame(rows)
    return frame.astype(_PAIR_DTYPES)[list(_PAIR_DTYPES)]


def mine_reference(reference_file: Path, index_dir: Path, output: Path, *,
                   splits: tuple[str, ...] = ("tune",), expected_id_sha256: str | None = TUNE_ID_SHA256,
                   batch_size: int = 500, max_scan: int = 512,
                   partitioned: bool | None = None, start_row: int = 0,
                   stop_row: int | None = None, allow_holdout2_unlabeled: bool = False) -> dict:
    """Mine all selected S1s with no labels and preserve even zero-hit S1 rows."""
    started = time.monotonic()
    reference_file, output = Path(reference_file), Path(output)
    if (not splits or set(splits) - {"train", "tune", "medium_tune", "train_tune_eligible", "holdout2"}
            or ("train_tune_eligible" in splits and len(splits) != 1)
            or ("holdout2" in splits and (splits != ("holdout2",) or not allow_holdout2_unlabeled))):
        raise ValueError("Only explicitly selected TRAIN/TUNE text is accepted; holdout/eval are excluded")
    if (start_row < 0 or (start_row or stop_row is not None) and splits != ("train_tune_eligible",)):
        raise ValueError("Row shards require exactly the shared fit-eligible population")
    if not expected_id_sha256 or len(expected_id_sha256) != 64:
        raise ValueError("A known sorted S1-ID SHA256 is required before mining")
    if min(batch_size, max_scan) < 1:
        raise ValueError("Batch size and posting scan cap must be positive")
    if (output / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite finished candidates: {output}")
    if (output / "prefix_sealed.json").exists():
        raise FileExistsError(f"A committed prefix was sealed for disjoint shards: {output}")
    source = pd.read_parquet(reference_file)
    partition = "split" if "split" in source else "population" if "population" in source else None
    if partition is None or set(("entity_id", "business_name", "business_address", "country")) - set(source):
        raise ValueError("References need original ID/text/country and saved partition")
    mask = source[partition].isin(splits)
    # Shards read the already-filtered parent Parquet. Avoid a second ~2M-row
    # string-frame allocation per worker when every source row is eligible.
    parent = source if mask.all() else source.loc[mask].reset_index(drop=True)
    if not len(parent) or any(not parent[partition].eq(split).any() for split in splits):
        raise ValueError("Each requested TRAIN/TUNE population must have references")
    if parent.entity_id.duplicated().any() or not parent.entity_id.str.startswith("S1-").all():
        raise ValueError("Repeated/invalid Source 1 IDs")
    parent_digest = ids_digest(parent.entity_id)
    if parent_digest != expected_id_sha256:
        raise ValueError(f"Reference ID digest mismatch: {parent_digest}")
    end = len(parent) if stop_row is None else stop_row
    if start_row >= end or end > len(parent):
        raise ValueError("Reference shard has an empty or out-of-bounds original-row range")
    reference = parent.iloc[start_row:end].reset_index(drop=True)
    observed = ids_digest(reference.entity_id)
    parent_count = len(parent)
    del source, parent
    if partitioned is None:
        partitioned = len(reference) > 50_000 or splits == ("holdout2",)
    output.mkdir(parents=True, exist_ok=True)
    index = FullPoolIndex(index_dir)
    reference_sha = file_sha256(reference_file)
    index_sha = file_sha256(Path(index_dir) / "complete.json")
    identity = {"reference_file_sha256": reference_sha, "reference_ids_sha256": observed,
                "parent_reference_ids_sha256": parent_digest, "parent_references": parent_count,
                "left_offset": start_row, "left_stop": end,
                "index_manifest_sha256": index_sha, "pool_rows": index.manifest["rows"],
                "splits": list(splits),
                "partition_column": partition, "batch_size": batch_size,
                "posting_scan_rows": max_scan, "partitioned": partitioned}
    ref_path = output / "reference.parquet"
    if ref_path.exists():
        if not partitioned or not (output / "mining_progress.json").exists():
            raise ValueError("Reference output exists without a resumable mining state")
    else:
        reference.to_parquet(output / "reference.parquet.tmp", index=False)
        (output / "reference.parquet.tmp").replace(ref_path)
    reference_output_sha = file_sha256(ref_path)
    writer, total, stats = None, 0, Counter()
    temp = output / "candidate_pairs.parquet.tmp"
    try:
        parts, resume_at = [], 0
        if partitioned:
            directory = output / "candidate_pairs.parquet"
            directory.mkdir(exist_ok=True)
            progress_file = output / "mining_progress.json"
            if progress_file.exists():
                previous = json.loads(progress_file.read_text())
                if (previous.get("identity") != identity
                        or previous.get("reference_output_sha256") != reference_output_sha):
                    raise ValueError("A partial mining run belongs to different references/index/settings")
                parts = previous["parts"]
                resume_at, total, stats = previous["references_done"], previous["pairs"], Counter(previous["statistics"])
                cursor = 0
                for part in parts:
                    path = directory / part["name"]
                    if (part["start"] != cursor or part["end"] <= cursor or path.stat().st_size != part["bytes"]
                            or file_sha256(path) != part["sha256"]):
                        raise ValueError("Saved candidate partition is incomplete/changed")
                    cursor = part["end"]
                if cursor != resume_at or sum(part["rows"] for part in parts) != total:
                    raise ValueError("Mining progress is not a contiguous complete-reference prefix")
            else:
                if any(directory.iterdir()):
                    raise ValueError("Unclaimed candidate parts exist; inspect before starting")
                atomic_json(progress_file, {"state": "mining", "identity": identity,
                            "reference_output_sha256": reference_output_sha, "parts": [],
                            "references_done": 0, "references_total": len(reference),
                            "pairs": 0, "statistics": {}})
        else:
            if temp.exists():
                raise ValueError("An unfinished single-file candidate output needs inspection")
        for start in range(resume_at, len(reference), batch_size):
            rows = []
            for position, (entity, name, address, country) in enumerate(
                reference.iloc[start:start + batch_size][
                    ["entity_id", "business_name", "business_address", "country"]].itertuples(index=False, name=None), start):
                candidates, counts = index.search(str(name), str(address), str(country), max_scan=max_scan)
                for item in candidates:
                    item["left_idx"] = start_row + position
                rows.extend(candidates)
                stats.update(counts)
                stats["zero_candidate_references"] += not candidates
            frame = _coerce_pairs(rows)
            if partitioned:
                part_path = directory / f"part_{len(parts):06d}.parquet"
                if part_path.exists():
                    raise ValueError("An unclaimed candidate partition exists; inspect before resuming")
                temporary_part = Path(str(part_path) + ".tmp")
                if temporary_part.exists():
                    temporary_part.unlink()  # only our uncommitted partition, never a published part
                frame.to_parquet(temporary_part, index=False)
                temporary_part.replace(part_path)
                parts.append({"name": part_path.name, "start": start, "end": min(start + batch_size, len(reference)),
                              "rows": len(frame), "bytes": part_path.stat().st_size,
                              "sha256": file_sha256(part_path)})
            else:
                if writer is None:
                    writer = pq.ParquetWriter(temp, pa.Table.from_pandas(frame, preserve_index=False).schema,
                                              compression="snappy")
                writer.write_table(pa.Table.from_pandas(frame, preserve_index=False))
            total += len(frame)
            progress = {"state": "mining", "references_done": min(start + batch_size, len(reference)),
                        "references_total": len(reference), "pairs": total, "elapsed_seconds": time.monotonic() - started}
            if partitioned:
                progress.update({"identity": identity, "reference_output_sha256": reference_output_sha,
                                 "parts": parts, "statistics": dict(stats)})
            atomic_json(output / "mining_progress.json", progress)
            print(f"lexical retrieval: {min(start + batch_size, len(reference)):,}/{len(reference):,} S1; {total:,} pairs", flush=True)
        if partitioned:
            pair_info = {"name": "candidate_pairs.parquet", "type": "parquet_partition_directory",
                         "parts": parts, "ordered_part_manifest_sha256": hashlib.sha256(
                             json.dumps(parts, sort_keys=True).encode()).hexdigest()}
        else:
            writer.close()
            writer = None
            temp.replace(output / "candidate_pairs.parquet")
            pair_info = {"name": "candidate_pairs.parquet", "type": "single_parquet_file",
                         "bytes": (output / "candidate_pairs.parquet").stat().st_size,
                         "sha256": file_sha256(output / "candidate_pairs.parquet")}
        report = {"state": "complete", "schema": 1, "references": len(reference),
                  "pairs": total, "reference_ids_sha256": observed,
                  "parent_reference_ids_sha256": parent_digest, "parent_references": parent_count,
                  "left_offset": start_row, "left_stop": end,
                  "reference_file_sha256": reference_sha,
                  "reference_output_sha256": reference_output_sha,
                  "selected_splits": list(splits), "partition_column": partition,
                  "full_pool_rows": index.manifest["rows"],
                  "full_pool_index_manifest_sha256": index_sha,
                  "source_pool": index.manifest["sources"], "positive_candidate_injection": False,
                  "truth_read": False, "holdout2_text_only": splits == ("holdout2",),
                  "reference_coverage_file": "reference.parquet",
                  "pair_file": pair_info,
                  "branch_limits": {"name_exact_groups": 30, "phonetic_groups": 20,
                                    "digit_sequence_groups": 20, "digit_set_groups": 10,
                                    "composite_groups": "uncapped", "posting_scan_rows": max_scan},
                  "statistics": dict(stats), "elapsed_seconds": time.monotonic() - started,
                  "feature_contract": "Candidate-dependent ranks, scores, gaps and counts must be recomputed on the complete union before fitting or predicting; raw-row IDs map through the exact full-pool index."}
        atomic_json(output / "complete.json", report)
        return report
    finally:
        if writer is not None:
            writer.close()
        index.close()


def _check_part(path: Path, part: dict, offset: int, pool_rows: int) -> None:
    """Check hash and exact positional bounds without opening any labels."""
    if path.stat().st_size != part["bytes"] or file_sha256(path) != part["sha256"]:
        raise ValueError(f"Mined candidate partition changed: {path}")
    table = pq.ParquetFile(path).read(columns=["left_idx", "right_raw_idx"])
    left = table.column("left_idx").to_numpy()
    right = table.column("right_raw_idx").to_numpy()
    if (len(left) != part["rows"] or (len(left) and
            (left.min() < offset + part["start"] or left.max() >= offset + part["end"]
             or np.any(np.diff(left) < 0) or right.min() < 0 or right.max() >= pool_rows
             or pd.DataFrame({"left": left, "right": right}).duplicated().any()))):
        raise ValueError(f"Mined partition has invalid/duplicate original-row pairs: {path}")


def _validate_partition_list(directory: Path, parts: list[dict], offset: int,
                             expected_end: int, pool_rows: int) -> int:
    listed = {part["name"] for part in parts}
    actual = {entry.name for entry in directory.iterdir()}
    if listed != actual or len(listed) != len(parts):
        raise ValueError("Missing or uncommitted candidate partitions; inspect before sealing")
    cursor, pair_count = 0, 0
    for number, part in enumerate(parts):
        if part["name"] != f"part_{number:06d}.parquet" or part["start"] != cursor or part["end"] <= cursor:
            raise ValueError("Candidate partitions are not contiguous and deterministically named")
        _check_part(directory / part["name"], part, offset, pool_rows)
        cursor = part["end"]
        pair_count += part["rows"]
    if cursor != expected_end:
        raise ValueError("Candidate partitions omit a reference range")
    return pair_count


def seal_completed_prefix(directory: Path) -> dict:
    """Pin an interrupted one-core miner at its last committed S1 boundary."""
    directory = Path(directory)
    marker = directory / "prefix_sealed.json"
    if marker.exists() or (directory / "complete.json").exists():
        raise FileExistsError("Prefix is already sealed or fully mined")
    progress_path = directory / "mining_progress.json"
    progress_sha = file_sha256(progress_path)
    progress = json.loads(progress_path.read_text())
    identity = progress["identity"]
    if (progress.get("state") != "mining" or identity["splits"] != ["train_tune_eligible"]
            or progress["references_done"] <= 0
            or progress["references_done"] >= progress["references_total"]
            or progress["references_done"] % identity["batch_size"]
            or identity["reference_ids_sha256"] != ids_digest(pd.read_parquet(
                directory / "reference.parquet", columns=["entity_id"]).entity_id)
            or progress["reference_output_sha256"] != file_sha256(directory / "reference.parquet")):
        raise ValueError("Prefix progress does not pin a complete, fixed fit-eligible S1 span")
    total = _validate_partition_list(directory / "candidate_pairs.parquet", progress["parts"], 0,
                                     progress["references_done"], identity.get("pool_rows", POOL_ROWS))
    if total != progress["pairs"] or file_sha256(progress_path) != progress_sha:
        raise ValueError("A miner changed its checkpoint while the prefix was being sealed")
    record = {"schema": 1, "state": "sealed_prefix", "references_done": progress["references_done"],
              "references_total": progress["references_total"], "candidate_pairs": total,
              "partition_count": len(progress["parts"]), "progress_sha256": progress_sha,
              "full_reference_sha256": progress["reference_output_sha256"],
              "parent_reference_ids_sha256": identity["reference_ids_sha256"],
              "full_pool_index_manifest_sha256": identity["index_manifest_sha256"],
              "full_pool_rows": identity.get("pool_rows", POOL_ROWS), "truth_read": False,
              "positive_candidate_injection": False}
    atomic_json(marker, record)
    return record


def assemble_shards(prefix_dir: Path, shard_dirs: list[Path], output: Path) -> dict:
    """Hard-link verified, disjoint partitions into one original-indexed dataset.

    This runs *after* all miners complete. Partitions/IDs are never renumbered,
    copied over each other, or joined using truth; ``complete.json`` appears last.
    """
    prefix_dir, output = Path(prefix_dir), Path(output)
    if not shard_dirs or (output.exists() and any(output.iterdir())):
        raise ValueError("Need finished shards and an empty new assembly directory")
    seal = json.loads((prefix_dir / "prefix_sealed.json").read_text())
    progress_path = prefix_dir / "mining_progress.json"
    prefix = json.loads(progress_path.read_text())
    if (seal.get("state") != "sealed_prefix" or file_sha256(progress_path) != seal["progress_sha256"]
            or prefix["references_done"] != seal["references_done"]
            or file_sha256(prefix_dir / "reference.parquet") != seal["full_reference_sha256"]):
        raise ValueError("The sealed prefix or source S1 population changed")
    source_ref = pd.read_parquet(prefix_dir / "reference.parquet", columns=["entity_id", "population"])
    if (len(source_ref) != seal["references_total"]
            or not source_ref.population.eq("train_tune_eligible").all()
            or ids_digest(source_ref.entity_id) != seal["parent_reference_ids_sha256"]):
        raise ValueError("Assembled references must exclude every sealed holdout S1")
    segments = [(0, seal["references_done"], prefix_dir, prefix["parts"])]
    for path in map(Path, shard_dirs):
        complete = json.loads((path / "complete.json").read_text())
        begin, end = complete["left_offset"], complete["left_stop"]
        if (complete.get("state") != "complete" or complete.get("selected_splits") != ["train_tune_eligible"]
                or complete.get("positive_candidate_injection") is not False
                or complete.get("truth_read") is not False
                or complete["reference_file_sha256"] != seal["full_reference_sha256"]
                or complete["parent_reference_ids_sha256"] != seal["parent_reference_ids_sha256"]
                or complete["full_pool_index_manifest_sha256"] != seal["full_pool_index_manifest_sha256"]
                or complete["full_pool_rows"] != seal["full_pool_rows"] or complete["references"] != end - begin
                or complete["reference_ids_sha256"] != ids_digest(source_ref.entity_id.iloc[begin:end])
                or file_sha256(path / "reference.parquet") != complete["reference_output_sha256"]):
            raise ValueError(f"Shard does not belong to this exact label-free S1/pool population: {path}")
        segments.append((begin, end, path, complete["pair_file"]["parts"]))
    segments.sort(key=lambda item: item[0])
    if (segments[0][0] != 0 or segments[-1][1] != len(source_ref)
            or any(current[1] != next_[0] for current, next_ in zip(segments, segments[1:]))):
        raise ValueError("Lexical shards overlap or leave uncovered original S1 rows")
    # Verify *before* publishing any links or a completion marker.
    pairs = 0
    for begin, end, path, parts in segments:
        count = _validate_partition_list(path / "candidate_pairs.parquet", parts, begin,
                                         end - begin, seal["full_pool_rows"])
        if path == prefix_dir and count != seal["candidate_pairs"]:
            raise ValueError("Sealed prefix pair count differs from its verified checkpoint")
        pairs += count
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "candidate_pairs.parquet"
    destination.mkdir()
    (output / "reference.parquet").hardlink_to(prefix_dir / "reference.parquet")
    linked = []
    for begin, end, path, parts in segments:
        for part in parts:
            original = path / "candidate_pairs.parquet" / part["name"]
            target = destination / f"part_{len(linked):06d}.parquet"
            target.hardlink_to(original)
            linked.append({"name": target.name, "start": begin + part["start"],
                           "end": begin + part["end"], "rows": part["rows"],
                           "bytes": part["bytes"], "sha256": part["sha256"],
                           "source": str(original.resolve())})
    manifest = {"schema": 1, "state": "complete", "references": len(source_ref),
                "reference_ids_sha256": seal["parent_reference_ids_sha256"],
                "reference_file_sha256": seal["full_reference_sha256"],
                "full_pool_index_manifest_sha256": seal["full_pool_index_manifest_sha256"],
                "full_pool_rows": seal["full_pool_rows"], "candidate_pairs": pairs,
                "candidate_pairs_location": "candidate_pairs.parquet/ (ordered hard-linked Parquet parts)",
                "pair_partitions": linked, "prefix_seal_sha256": file_sha256(prefix_dir / "prefix_sealed.json"),
                "shard_manifest_sha256": {str(path.resolve()): file_sha256(path / "complete.json")
                                           for _, _, path, _ in segments[1:]},
                "holdout2_excluded_from_training_references": True,
                "truth_read": False, "positive_candidate_injection": False,
                "context_features_pending": True}
    atomic_json(output / "complete.json", manifest)
    return manifest


def publish_assembled_prefix(prefix_dir: Path, assembled_dir: Path) -> dict:
    """Append only new hard links; commit the original P2.5-compatible path last.

    The original source's first N parts and their manifest rows remain byte-for-
    byte identical, so an existing ``v2_train_union stage --follow`` resumes at
    the next part without another queue or any edits to P2.5's implementation.
    """
    prefix_dir, assembled_dir = Path(prefix_dir), Path(assembled_dir)
    merged = json.loads((assembled_dir / "complete.json").read_text())
    seal = json.loads((prefix_dir / "prefix_sealed.json").read_text())
    original_progress = json.loads((prefix_dir / "mining_progress.json").read_text())
    if (merged.get("state") != "complete" or merged["prefix_seal_sha256"] != file_sha256(
            prefix_dir / "prefix_sealed.json") or merged["full_pool_rows"] != seal["full_pool_rows"]
            or merged["reference_file_sha256"] != seal["full_reference_sha256"]
            or merged["reference_ids_sha256"] != seal["parent_reference_ids_sha256"]
            or file_sha256(prefix_dir / "reference.parquet") != seal["full_reference_sha256"]):
        raise ValueError("Assembled candidate parts differ from the sealed original prefix")
    full_parts = [{key: item[key] for key in ("name", "start", "end", "rows", "bytes", "sha256")}
                  for item in merged["pair_partitions"]]
    prefix_count = seal["partition_count"]
    if (full_parts[:prefix_count] != original_progress["parts"][:prefix_count]
            or len(full_parts) <= prefix_count or full_parts[-1]["end"] != merged["references"]
            or original_progress["identity"].get("batch_size", 0) <= 0
            or original_progress["identity"].get("index_manifest_sha256")
            != seal["full_pool_index_manifest_sha256"]):
        raise ValueError("Published parts must extend the original deterministic 1,000-S1 prefix")
    if (len(original_progress["parts"]) == prefix_count
            and file_sha256(prefix_dir / "mining_progress.json") != seal["progress_sha256"]):
        raise ValueError("Sealed progress changed before publishing new shard links")
    if original_progress["parts"] not in (full_parts[:prefix_count], full_parts):
        raise ValueError("Original progress has an unexpected partial or overlapping part list")
    original_dir = prefix_dir / "candidate_pairs.parquet"
    assembled_parts = assembled_dir / "candidate_pairs.parquet"
    for item in full_parts[prefix_count:]:
        source = assembled_parts / item["name"]
        target = original_dir / item["name"]
        _check_part(source, item, 0, seal["full_pool_rows"])
        if target.exists():
            if not target.samefile(source):
                raise ValueError(f"Unexpected original-prefix part cannot be replaced: {target}")
        else:
            target.hardlink_to(source)
    if len(original_progress["parts"]) == prefix_count:
        original_progress["parts"] = full_parts
        original_progress["references_done"] = merged["references"]
        original_progress["pairs"] = merged["candidate_pairs"]
        original_progress["assembly_manifest_sha256"] = file_sha256(assembled_dir / "complete.json")
        original_progress["statistics_scope"] = "prefix only; shard counters in each completed shard manifest"
        atomic_json(prefix_dir / "mining_progress.json", original_progress)
    complete_path = prefix_dir / "complete.json"
    document = {"state": "complete", "schema": 1, "references": merged["references"],
                "pairs": merged["candidate_pairs"],
                "reference_ids_sha256": seal["parent_reference_ids_sha256"],
                "reference_file_sha256": seal["full_reference_sha256"],
                "full_pool_rows": seal["full_pool_rows"],
                "full_pool_index_manifest_sha256": seal["full_pool_index_manifest_sha256"],
                "selected_splits": ["train_tune_eligible"],
                "pair_file": {"name": "candidate_pairs.parquet", "type": "parquet_partition_directory",
                              "parts": full_parts, "ordered_part_manifest_sha256": hashlib.sha256(
                                  json.dumps(full_parts, sort_keys=True).encode()).hexdigest()},
                "assembly_manifest_sha256": file_sha256(assembled_dir / "complete.json"),
                "sealed_prefix_sha256": file_sha256(prefix_dir / "prefix_sealed.json"),
                "holdout2_excluded_from_training_references": True,
                "truth_read": False, "positive_candidate_injection": False}
    if complete_path.exists():
        if json.loads(complete_path.read_text()) != document:
            raise ValueError("An existing full completion marker differs from the assembled shards")
    else:
        atomic_json(complete_path, document)
    return document


def wait_and_publish(prefix_dir: Path, shard_dirs: list[Path], output: Path, *,
                     worker_pids: list[int] | None = None, poll_seconds: float = 30.,
                     max_wait_seconds: float = 6 * 3600) -> dict:
    """Tiny bounded watcher: publish only after every disjoint shard commits."""
    if (not shard_dirs or poll_seconds <= 0 or max_wait_seconds <= 0
            or (worker_pids is not None and len(worker_pids) != len(shard_dirs))):
        raise ValueError("Need positive wait bounds and one optional PID per owned shard")
    shards = list(map(Path, shard_dirs))
    started = time.monotonic()
    next_log = started
    while True:
        finished = [(path / "complete.json").is_file() for path in shards]
        if all(finished):
            break
        if worker_pids is not None:
            for path, pid, done in zip(shards, worker_pids, finished):
                if done:
                    continue
                try:
                    os.kill(pid, 0)
                except ProcessLookupError as error:
                    raise RuntimeError(f"Lexical worker exited without a completion marker: {path}") from error
        now = time.monotonic()
        if now - started > max_wait_seconds:
            raise TimeoutError("Disjoint full-pool lexical mining exceeded its bounded wait")
        if now >= next_log:
            progress = [json.loads((path / "mining_progress.json").read_text()) for path in shards]
            print(f"lexical shard queue: {sum(item['references_done'] for item in progress):,}/"
                  f"{sum(item['references_total'] for item in progress):,} remaining-range S1s committed; "
                  f"{sum(item['pairs'] for item in progress):,} pairs", flush=True)
            next_log = now + 300
        time.sleep(poll_seconds)
    output = Path(output)
    if (output / "complete.json").exists():
        assembled = json.loads((output / "complete.json").read_text())
        if assembled.get("state") != "complete":
            raise ValueError("An assembled candidate dataset has an invalid completion marker")
    else:
        assembled = assemble_shards(prefix_dir, shards, output)
    published = publish_assembled_prefix(prefix_dir, output)
    return {"state": "complete", "references": published["references"],
            "candidate_pairs": published["pairs"], "assembly_manifest": str((output / "complete.json").resolve()),
            "p2_5_source_manifest": str((Path(prefix_dir) / "complete.json").resolve()),
            "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    build = subs.add_parser("index", help="Build full 10.32M-target index from unlabeled S2/S3 TSVs")
    build.add_argument("--raw-dir", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--chunksize", type=int, default=50_000)
    mine = subs.add_parser("mine", help="Search specified saved TRAIN/TUNE references, no truth input")
    mine.add_argument("--references", required=True, type=Path)
    mine.add_argument("--index", required=True, type=Path)
    mine.add_argument("--output", required=True, type=Path)
    mine.add_argument("--splits", default="tune", help="tune, train_tune_eligible, or explicitly label-free holdout2")
    mine.add_argument("--expected-id-sha256", default=TUNE_ID_SHA256,
                      help="Saved sorted S1 ID digest; override for the shared future TRAIN/TUNE population")
    mine.add_argument("--batch-size", type=int, default=500)
    mine.add_argument("--max-scan", type=int, default=512)
    mine.add_argument("--start-row", type=int, default=0,
                      help="0-based first fit-eligible S1 in the saved parent order; pair indexes remain global")
    mine.add_argument("--stop-row", type=int,
                      help="Exclusive fit-eligible S1 position; omit to mine through the parent end")
    mine.add_argument("--allow-holdout2-unlabeled", action="store_true",
                      help="Permit only sealed HOLDOUT-2 source text, with no truth input or scoring")
    seal = subs.add_parser("seal-prefix", help="Verify/lock the last committed prefix before disjoint shard assembly")
    seal.add_argument("--output", required=True, type=Path)
    assemble = subs.add_parser("assemble", help="Verify disjoint completed shards and link one ordered label-free dataset")
    assemble.add_argument("--prefix", required=True, type=Path)
    assemble.add_argument("--shards", nargs="+", required=True, type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    publish = subs.add_parser("publish-prefix", help="Append verified assembled hard links to the original P2.5 source")
    publish.add_argument("--prefix", required=True, type=Path)
    publish.add_argument("--assembled", required=True, type=Path)
    await_ = subs.add_parser("await-and-publish", help="Wait for disjoint workers, then publish the original P2.5 source")
    await_.add_argument("--prefix", required=True, type=Path)
    await_.add_argument("--shards", nargs="+", required=True, type=Path)
    await_.add_argument("--output", required=True, type=Path)
    await_.add_argument("--worker-pids", nargs="+", type=int)
    await_.add_argument("--poll-seconds", type=float, default=30.)
    await_.add_argument("--max-wait-seconds", type=float, default=6 * 3600)
    args = parser.parse_args()
    if args.action == "index":
        result = build_pool_index(args.raw_dir, args.output, chunksize=args.chunksize)
    elif args.action == "mine":
        result = mine_reference(args.references, args.index, args.output,
                                 splits=tuple(args.splits.split(",")),
                                 expected_id_sha256=args.expected_id_sha256 or None,
                                 batch_size=args.batch_size, max_scan=args.max_scan,
                                 start_row=args.start_row, stop_row=args.stop_row,
                                 allow_holdout2_unlabeled=args.allow_holdout2_unlabeled)
    elif args.action == "seal-prefix":
        result = seal_completed_prefix(args.output)
    elif args.action == "assemble":
        result = assemble_shards(args.prefix, args.shards, args.output)
    elif args.action == "publish-prefix":
        result = publish_assembled_prefix(args.prefix, args.assembled)
    else:
        result = wait_and_publish(args.prefix, args.shards, args.output,
                                  worker_pids=args.worker_pids, poll_seconds=args.poll_seconds,
                                  max_wait_seconds=args.max_wait_seconds)
    print(json.dumps({"state": result["state"], "rows": result.get("rows"),
                      "references": result.get("references"), "pairs": result.get("pairs"),
                      "candidate_pairs": result.get("candidate_pairs"),
                      "elapsed_seconds": result.get("elapsed_seconds")}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
