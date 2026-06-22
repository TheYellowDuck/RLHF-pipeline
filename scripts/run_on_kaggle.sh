#!/usr/bin/env bash
# Run the RLHF pipeline on Kaggle's GPU headless, via the Kaggle API — no browser/UI.
#
# One-time setup:
#   pip install kaggle
#   Kaggle ▸ Settings ▸ "Create New API Token" → save to ~/.kaggle/kaggle.json (chmod 600)
#
# Then from the repo root:  bash scripts/run_on_kaggle.sh
#
# It reads your username from ~/.kaggle/kaggle.json (or $KAGGLE_USERNAME), pushes the
# notebook (which clones this repo and trains), polls until done, and downloads
# RESULTS.md + checkpoints to ./kaggle_out.  Still uses your GPU quota and can be
# preempted (the API only removes the manual clicking).
set -euo pipefail
cd "$(dirname "$0")/.."

SLUG="rlhf-pipeline-run"
USER_NAME="${KAGGLE_USERNAME:-}"
if [ -z "$USER_NAME" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
  USER_NAME=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
fi
if [ -z "$USER_NAME" ]; then
  echo "No Kaggle username found. Put your token at ~/.kaggle/kaggle.json (Kaggle ▸ Settings ▸ Create New API Token)."
  exit 1
fi
ID="$USER_NAME/$SLUG"

# Stamp your username into the kernel id so the push targets your account.
python3 - "$ID" <<'PY'
import json, sys
m = json.load(open("kernel-metadata.json")); m["id"] = sys.argv[1]
json.dump(m, open("kernel-metadata.json", "w"), indent=2)
print("kernel id ->", sys.argv[1])
PY

echo ">> pushing + starting headless GPU run: $ID"
kaggle kernels push -p .

echo ">> polling status (Ctrl-C is safe; the run keeps going on Kaggle)"
while true; do
  status=$(kaggle kernels status "$ID" 2>/dev/null | tr -d '"' | tr '[:upper:]' '[:lower:]')
  echo "   $(date +%H:%M:%S)  $status"
  case "$status" in *complete*|*error*|*cancel*) break ;; esac
  sleep 60
done

echo ">> downloading outputs to ./kaggle_out"
mkdir -p kaggle_out && kaggle kernels output "$ID" -p ./kaggle_out
echo ">> done. See ./kaggle_out/RESULTS.md"
