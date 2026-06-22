#!/usr/bin/env bash
# Run the RLHF pipeline on Kaggle's GPU headless, via the Kaggle API — no browser/UI.
#
# One-time setup:
#   pip install kaggle
#   Kaggle ▸ Account ▸ "Create New API Token" → save to ~/.kaggle/kaggle.json (chmod 600)
#   Edit kernel-metadata.json: set "id" to "<your-kaggle-username>/rlhf-pipeline-run"
#
# Then from the repo root:  bash scripts/run_on_kaggle.sh
#
# It pushes the notebook (which clones this repo and trains), polls until done, and
# downloads RESULTS.md + checkpoints to ./kaggle_out.  Still uses your GPU quota and
# can be preempted (the API only removes the manual clicking).
set -euo pipefail
cd "$(dirname "$0")/.."

ID=$(python3 -c "import json;print(json.load(open('kernel-metadata.json'))['id'])")
if [[ "$ID" == YOUR_KAGGLE_USERNAME/* ]]; then
  echo "Edit kernel-metadata.json: set \"id\" to <your-kaggle-username>/rlhf-pipeline-run"; exit 1
fi

echo ">> pushing + starting headless run: $ID"
kaggle kernels push -p .

echo ">> polling status (Ctrl-C is safe; the run keeps going on Kaggle)"
while true; do
  status=$(kaggle kernels status "$ID" 2>/dev/null | tr -d '"' | tr '[:upper:]' '[:lower:]')
  echo "   $(date +%H:%M:%S)  $status"
  case "$status" in
    *complete*|*error*|*cancel*) break ;;
  esac
  sleep 60
done

echo ">> downloading outputs to ./kaggle_out"
mkdir -p kaggle_out
kaggle kernels output "$ID" -p ./kaggle_out
echo ">> done. Look for ./kaggle_out/RESULTS.md"
