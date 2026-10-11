#!/usr/bin/env bash
# 2b validation driver (protocol_2b.json): prep -> train all subjects -> one-shot evaluation.
# Run from Git Bash:  bash run_2b.sh
cd "$(dirname "$0")" || exit 1
PY=../../交付/.venv_q2/Scripts/python.exe
mkdir -p logs
cd /e/tmp2b || exit 1          # neutral cwd (python -I)
B=/e/20260725桌面文件/BCI/revision/bci2b
"$PY" -I "$B/rv2b/prep2b.py" >> "$B/logs/run_2b.out" 2>&1 || { echo "FAILED prep" >> "$B/logs/run_2b.out"; exit 1; }
for S in B01 B02 B03 B04 B05 B06 B07 B08 B09; do
  "$PY" -I "$B/rv2b/train2b.py" "$S" >> "$B/logs/run_2b.out" 2>&1 || { echo "FAILED train $S" >> "$B/logs/run_2b.out"; exit 1; }
done
"$PY" -I "$B/rv2b/eval2b.py" >> "$B/logs/run_2b.out" 2>&1 || echo "FAILED eval" >> "$B/logs/run_2b.out"
echo "ALL DONE $(date)" >> "$B/logs/run_2b.out"
