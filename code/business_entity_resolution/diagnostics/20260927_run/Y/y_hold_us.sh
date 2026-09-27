#!/bin/bash
# Agent Y: RAM guard (30 GB box). Holds (SIGSTOP) Agent Y's own U-pair scoring processes until the French chain
# (export, rebuild 0.10/0.70, agreement, French upload) has finished, then resumes them (SIGCONT).
Y=/home/ubuntu/run27/Y
held=""
while ! grep -q "french upload ok\|after2 done" $Y/logs/after2.log 2>/dev/null; do
  for p in $(pgrep -u ubuntu -f "ber.laya_c2 score .*run27/Y/us/(brest|test)/left.npy"); do
    case " $held " in *" $p "*) ;; *) kill -STOP $p && held="$held $p" && echo "held $p $(date -u +%T)";; esac
  done
  sleep 1
done
for p in $held; do kill -CONT $p && echo "resumed $p $(date -u +%T)"; done
echo "hold done $(date -u +%T)"
