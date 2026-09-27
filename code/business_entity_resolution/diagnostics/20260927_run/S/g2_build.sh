#!/bin/bash
set -u
R=/home/ubuntu/run27/S
export PYTHONPATH=$R/repo/src OMP_NUM_THREADS=2
exec /home/ubuntu/s-venv/bin/python -u -m ber.frenchify_arena build --output $R/arena_fr_v3 \
  --arena $R/in/arena/v1 --labels $R/in/arena/labels --arena-partition $R/in/arena/finalize_v1/s1_partition.npy \
  --sealed-reservation $R/v2_holdout2 --monitor-from $R/no_france_mix --vocab-counts $R/vocab_probe.json
