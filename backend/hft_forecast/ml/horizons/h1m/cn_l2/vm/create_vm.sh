#!/usr/bin/env bash
# Creates one GPU VM (no public IP, per Columbia org policy) for the 60 s notebook run.
# Tries L4 then T4 across zones because GPUs are often out of stock.
set -uo pipefail
PROJECT="hftf-capstone-1109"
NAME="hftf-run"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
L4_ZONES="us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-c us-east1-d us-west1-a us-west1-b us-west4-a"
T4_ZONES="us-east4-a us-central1-a us-central1-b us-central1-c us-central1-f us-east1-c us-east1-d us-east4-a us-east4-b us-west1-a us-west1-b us-west4-a us-west4-b"

try() {  # zone machine accelerator
  gcloud compute instances create "$NAME" --project="$PROJECT" --zone="$1" \
    --machine-type="$2" --accelerator="type=$3,count=1" --maintenance-policy=TERMINATE \
    --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release \
    --boot-disk-size=150GB --boot-disk-type=pd-balanced --no-address \
    --scopes=cloud-platform --metadata=install-nvidia-driver=True 2>&1 | grep -v "environment' tag"
  return "${PIPESTATUS[0]}"
}

for z in $L4_ZONES; do
  echo "Trying L4 in $z"; try "$z" "${L4_MACHINE:-g2-standard-8}" nvidia-l4 && { echo "CREATED $z L4"; exit 0; }
done
for z in $T4_ZONES; do
  echo "Trying T4 in $z"; try "$z" "${T4_MACHINE:-n1-standard-8}" nvidia-tesla-t4 && { echo "CREATED $z T4"; exit 0; }
done
echo "NO_CAPACITY"; exit 1
