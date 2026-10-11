#!/usr/bin/env bash
# Overnight driver: train all subjects (resumable), then perturbation passes.
# Run from Git Bash:  bash run_all.sh
cd "$(dirname "$0")/rv" || exit 1
PY=../../交付/.venv_q2/Scripts/python.exe
for S in A01 A02 A03 A04 A05 A06 A07 A08 A09; do
  "$PY" train_subject.py "$S" >> ../logs/run_all.out 2>&1 || echo "FAILED train $S" >> ../logs/run_all.out
done
for S in A01 A02 A03 A04 A05 A06 A07 A08 A09; do
  "$PY" perturb.py "$S" >> ../logs/run_all.out 2>&1 || echo "FAILED perturb $S" >> ../logs/run_all.out
done
echo "ALL DONE $(date)" >> ../logs/run_all.out
