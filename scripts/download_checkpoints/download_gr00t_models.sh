#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PCD_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
UPSTREAM_DIR="${PCD_ROOT}/third_party/Isaac-GR00T"
CHECKPOINT_ROOT="${PCD_CHECKPOINT_ROOT:-${PCD_ROOT}/checkpoint}"
MODEL_ROOT="${CHECKPOINT_ROOT}/gr00t"
COSMOS_LINK="${CHECKPOINT_ROOT}/Cosmos-Reason2-2B"
TRAINING_ONLY_PATTERNS=(
  "global_step*/**"
  "rng_state_*.pth"
  "scheduler.pt"
  "trainer_state.json"
  "training_args.bin"
  "zero_to_fp32.py"
  "latest"
  "wandb_config.json"
)

if [[ ! -x "${UPSTREAM_DIR}/.venv/bin/hf" ]]; then
  echo "Run bash scripts/install_dependencies/install_gr00t.sh first." >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_ROOT}" "${MODEL_ROOT}"
echo "Make sure you accepted access to nvidia/Cosmos-Reason2-2B and ran: hf auth login"

cosmos_snapshot="$("${UPSTREAM_DIR}/.venv/bin/hf" download nvidia/Cosmos-Reason2-2B)"
if [[ -L "${COSMOS_LINK}" ]]; then
  ln -sfn -- "${cosmos_snapshot}" "${COSMOS_LINK}"
elif [[ -e "${COSMOS_LINK}" ]]; then
  echo "Cannot create Cosmos link because the path already exists: ${COSMOS_LINK}" >&2
  exit 1
else
  ln -s -- "${cosmos_snapshot}" "${COSMOS_LINK}"
fi

"${UPSTREAM_DIR}/.venv/bin/hf" download \
  nvidia/GR00T-N1.7-SimplerEnv-Fractal \
  --exclude "${TRAINING_ONLY_PATTERNS[@]}" \
  --local-dir "${MODEL_ROOT}/GR00T-N1.7-SimplerEnv-Fractal"
"${UPSTREAM_DIR}/.venv/bin/hf" download \
  nvidia/GR00T-N1.7-SimplerEnv-Bridge \
  --exclude "${TRAINING_ONLY_PATTERNS[@]}" \
  --local-dir "${MODEL_ROOT}/GR00T-N1.7-SimplerEnv-Bridge"

echo "Cosmos checkpoint: ${COSMOS_LINK} -> ${cosmos_snapshot}"
echo "Downloaded SimplerEnv checkpoints under ${MODEL_ROOT}"
