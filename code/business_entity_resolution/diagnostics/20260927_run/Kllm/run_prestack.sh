#!/bin/bash
cd /home/ubuntu/kllm
OMP_NUM_THREADS=8 /home/ubuntu/run27/U/venv/bin/python kllm_prestack.py > logs/prestack.log 2>&1
echo "EXIT $?" >> logs/prestack.log
