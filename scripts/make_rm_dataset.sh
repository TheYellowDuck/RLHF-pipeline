#!/usr/bin/env bash
# Create a Kaggle Dataset from a downloaded reward-model checkpoint, so a PPO kernel can
# mount it as input (avoids the "buried old kernel version" problem — a kernel's output is
# only API-reachable while it is the LATEST version, but a Dataset is permanent).
#
# Usage:  scripts/make_rm_dataset.sh <folder>
#   <folder> = the extracted v14 output, OR the reward_model dir itself. The script drills
#   down to whatever directory actually contains reward_config.json.
#
# After it prints the slug, launch the 1.5B PPO (see the tail of this script / HANDOFF #3).
set -euo pipefail

RM_DIR="${1:?usage: scripts/make_rm_dataset.sh <path-to-downloaded-reward_model-folder>}"
KAGGLE="${KAGGLE:-.venv/bin/kaggle}"
KuSER="georgezhang06"
SLUG="rlhf-rm-1p5b-08025"            # -> dataset_sources: ["georgezhang06/rlhf-rm-1p5b-08025"]

# Resolve to the dir that directly holds reward_config.json (RewardModel.save_pretrained marker).
if [ ! -f "$RM_DIR/reward_config.json" ]; then
  found="$(find "$RM_DIR" -name reward_config.json -print 2>/dev/null | head -1 || true)"
  [ -n "$found" ] && RM_DIR="$(dirname "$found")"
fi
[ -f "$RM_DIR/reward_config.json" ] \
  || { echo "ERROR: no reward_config.json found under '$1' — is this the right folder?"; exit 1; }

# Guard against partial UI/CLI downloads: the merged weights must be non-trivial in size.
biggest=0; bigfile=""
for f in "$RM_DIR"/*.safetensors "$RM_DIR"/*.bin; do
  [ -f "$f" ] || continue
  s=$(stat -f%z "$f")
  [ "$s" -gt "$biggest" ] && { biggest=$s; bigfile="$f"; }
done
[ "$biggest" -gt 10000000 ] \
  || { echo "ERROR: model weights look empty/partial (largest='$bigfile' = ${biggest} bytes)."; \
       echo "       Re-download the COMPLETE v14 output from the Kaggle UI and retry."; exit 1; }
echo "RM dir : $RM_DIR"
echo "weights: $bigfile = $((biggest/1000000)) MB"

cat > "$RM_DIR/dataset-metadata.json" <<JSON
{
  "title": "RLHF 1.5B Reward Model (0.8025)",
  "id": "$KuSER/$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

echo "Creating Kaggle dataset $KuSER/$SLUG (uploads the checkpoint — a few minutes for ~3 GB)..."
"$KAGGLE" datasets create -p "$RM_DIR" --dir-mode skip

cat <<NEXT

Dataset created: $KuSER/$SLUG
Next — launch the 1.5B PPO (#3) on a forced T4:
  git push                                   # kernel git-clones from GitHub; push GRM + the PPO notebook first
  python3 - <<'PY'
import json
m = json.load(open('kernel-metadata.json'))
m['id'] = '$KuSER/rlhf-ppo-1p5b'            # separate kernel id so it never buries another output
m['code_file'] = 'notebooks/kaggle_ppo_1.5b.ipynb'
m['dataset_sources'] = ['$KuSER/$SLUG']
json.dump(m, open('kernel-metadata.json','w'), indent=2)
print('kernel-metadata.json ->', m['id'], m['code_file'], m['dataset_sources'])
PY
  $KAGGLE kernels push -p . --accelerator NvidiaTeslaT4
NEXT
