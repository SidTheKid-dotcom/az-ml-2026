#!/bin/bash
cd /home/ubuntu/kllm
OMP_NUM_THREADS=8 /home/ubuntu/run27/U/venv/bin/python kllm_test.py no_us_dense/llm > logs/test.log 2>&1
echo "EXIT $?" >> logs/test.log
