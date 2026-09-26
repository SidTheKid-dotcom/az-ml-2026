"""Build the raw Source-2/3 parquet cache for one split (train or test); text only, no labels."""
import sys

from D_common import log, raw_sources

if __name__ == "__main__":
    frame = raw_sources(sys.argv[1])
    log(f"{sys.argv[1]}: {len(frame):,} rows; by source {frame.source.value_counts().to_dict()}")
