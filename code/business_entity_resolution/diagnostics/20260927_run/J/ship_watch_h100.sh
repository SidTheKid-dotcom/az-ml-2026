#!/bin/bash
# Agent J (H100): ship run2 step_001000 as tag s1000 when it is written (early read on gpu2).
set -u
cd /root/J
RUN=/dev/shm/J/run2
until [ -e $RUN/checkpoints/step_001000/complete.json ]; do sleep 15; done
bash ship_ck.sh put_ck.txt s1000 $RUN/checkpoints/step_001000 $RUN
echo "SHIPPED s1000 $(date)"
