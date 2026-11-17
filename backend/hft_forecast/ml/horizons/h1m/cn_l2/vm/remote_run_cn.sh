#!/usr/bin/env bash
# Runs on the GPU VM: pull the China L2 5 s bar cache from the bucket, execute the notebook
# headless (June locked test on), upload the executed notebook, log and results, then power
# the VM off so GPU billing stops.
set -uo pipefail
BUCKET="gs://hftf-capstone-1109-taq"
STAMP="$(date -u +%Y%m%dT%H%MZ)"
OUT="$BUCKET/runs/cn_l2_$STAMP"
cd "$HOME/run"
trap 'gcloud storage cp run.log executed.ipynb "$OUT/" 2>/dev/null; gcloud storage cp -r cn_l2_60s_research/results_60s_multihorizon "$OUT/" 2>/dev/null; echo "$OUT" > last_run.txt; sudo poweroff' EXIT

{
  echo "== $(date -u) start, GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  # Use the interpreter that ships with PyTorch on the Deep Learning VM image.
  PY=""
  for p in python3 /opt/conda/bin/python /opt/python/bin/python3 /usr/bin/python3; do
    if command -v "$p" >/dev/null && "$p" -c "import torch" 2>/dev/null; then PY="$(command -v "$p")"; break; fi
  done
  [ -n "$PY" ] || { echo "No Python with torch found"; exit 1; }
  echo "Python: $PY"; "$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  mkdir -p cn_l2_60s_research/trade_bars_5s
  gcloud storage rsync -r "$BUCKET/cn_l2/trade_bars_5s" cn_l2_60s_research/trade_bars_5s
  "$PY" -m pip install -q --no-warn-script-location polars pyarrow pandas scipy matplotlib papermill ipykernel
  "$PY" -m ipykernel install --user --name python3 >/dev/null
  "$PY" -c "import numpy, pandas, scipy, polars, pyarrow, matplotlib, torch, IPython, papermill; print('imports OK')" || exit 1
  echo "== $(date -u) executing notebook"
  RUN_FINAL_TEST=1 "$PY" -m papermill notebook.ipynb executed.ipynb --kernel python3 --log-output --request-save-on-cell-execute
  rc=$?
  echo "== $(date -u) papermill exit $rc"
} > run.log 2>&1
