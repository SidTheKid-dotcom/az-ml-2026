#!/bin/bash
cd /home/ubuntu/kllm
OMP_NUM_THREADS=8 /home/ubuntu/run27/U/venv/bin/python kllm_brest.py brest_report_v2.json > logs/brest2.log 2>&1
echo "EXIT $?" >> logs/brest2.log
if grep -q "BREST DONE" logs/brest2.log; then
  OMP_NUM_THREADS=8 /home/ubuntu/run27/U/venv/bin/python kllm_test.py no_us_dense_no_route/llm brest_report_v2.json > logs/test2.log 2>&1
  echo "EXIT $?" >> logs/test2.log
fi
