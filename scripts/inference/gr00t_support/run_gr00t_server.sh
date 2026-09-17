#!/usr/bin/env bash
set -euo pipefail

PCD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
UPSTREAM_DIR="${PCD_ROOT}/third_party/Isaac-GR00T"
ROBOT="${1:-google}"

case "${ROBOT}" in
  google)
    DEFAULT_MODEL="${PCD_ROOT}/checkpoint/gr00t/GR00T-N1.7-SimplerEnv-Fractal"
    EMBODIMENT="SIMPLER_ENV_GOOGLE"
    POLICY_SETUP="google_robot"
    ;;
  widowx)
    DEFAULT_MODEL="${PCD_ROOT}/checkpoint/gr00t/GR00T-N1.7-SimplerEnv-Bridge"
    EMBODIMENT="SIMPLER_ENV_WIDOWX"
    POLICY_SETUP="widowx_bridge"
    ;;
  *)
    echo "Usage: $0 {google|widowx}" >&2
    exit 2
    ;;
esac

MODEL_PATH="${GR00T_MODEL_PATH:-${DEFAULT_MODEL}}"
DEVICE="${GR00T_DEVICE:-cuda:0}"
PORT="${GR00T_PORT:-5555}"

if [[ "${MODEL_PATH}" = /* && ! -d "${MODEL_PATH}" ]]; then
  echo "Checkpoint not found: ${MODEL_PATH}" >&2
  echo "Run bash scripts/download_checkpoints/download_gr00t_models.sh or set GR00T_MODEL_PATH to a model ID/path." >&2
  exit 1
fi

cd "${UPSTREAM_DIR}"
export PYTHONPATH="${PCD_ROOT}:${PYTHONPATH:-}"
exec uv run --with pyzmq python -m contrast_policies.gr00t.server \
    --model-path "${MODEL_PATH}" \
    --embodiment-tag "${EMBODIMENT}" \
    --policy-setup "${POLICY_SETUP}" \
    --device "${DEVICE}" \
    --host 127.0.0.1 \
    --port "${PORT}"
