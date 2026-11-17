#!/usr/bin/env bash
# Creates the GPU VM (retrying while GPUs are out of stock), copies the China L2 notebook and
# runner, and starts the run. The VM uploads results to the bucket and powers itself off.
set -uo pipefail
cd "$(dirname "$0")/.."
PR="hftf-capstone-1109"
for attempt in $(seq 1 "${MAX_ATTEMPTS:-30}"); do
  OUT=$(bash vm/create_vm.sh 2>&1 | grep -E "CREATED|NO_CAPACITY")
  Z=$(echo "$OUT" | awk '/CREATED/{print $2}')
  [ -n "$Z" ] && break
  echo "attempt $attempt: no GPU capacity, retrying in 2 minutes"; sleep 120
done
[ -n "${Z:-}" ] || { echo "gave up"; exit 1; }
echo "$OUT"
ssh_vm() { gcloud compute ssh hftf-run --zone="$Z" --project="$PR" --tunnel-through-iap --command="$1"; }
for i in $(seq 1 15); do ssh_vm 'mkdir -p ~/run' >/dev/null 2>&1 && break; sleep 20; done
gcloud compute scp --zone="$Z" --project="$PR" --tunnel-through-iap L2_China_60s_Trade_Forecasting.ipynb hftf-run:~/run/notebook.ipynb >/dev/null 2>&1
gcloud compute scp --zone="$Z" --project="$PR" --tunnel-through-iap vm/remote_run_cn.sh hftf-run:~/run/remote_run_cn.sh >/dev/null 2>&1
ssh_vm 'cd ~/run && ls && chmod +x remote_run_cn.sh && (setsid nohup ./remote_run_cn.sh >/dev/null 2>&1 < /dev/null &); echo launched' 2>&1 | grep -E "notebook|remote|launched"
