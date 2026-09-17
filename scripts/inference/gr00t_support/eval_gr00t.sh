#!/usr/bin/env bash
set -euo pipefail

PCD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK="${1:-google_robot_pick_coke_can}"
MODE="${2:-pcd-fast}"
GPU_DEVICES="${GPU_DEVICES:-${GPU_ID:-0,1,2,3}}"
IFS=',' read -r -a GPU_DEVICE_ARRAY <<< "${GPU_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#GPU_DEVICE_ARRAY[@]}}"
N_TRAJS="${N_TRAJS:-100}"
SEED="${SEED:-0}"
SIMPLER_PYTHON="$(command -v "${SIMPLER_PYTHON:-python}" || true)"
RESULT_ROOT="${RESULT_ROOT:-${PCD_ROOT}/Results}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PCD_ROOT}/runtime/gr00t/client}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_ROOT}/matplotlib}"
export TMPDIR="${TMPDIR:-${RUNTIME_ROOT}/tmp}"
export DISPLAY=""
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"

if [[ -z "${SIMPLER_PYTHON}" || ! -x "${SIMPLER_PYTHON}" ]]; then
  echo "SimplerEnv Python not found." >&2
  echo "Set SIMPLER_PYTHON to the Python executable of your working SimplerEnv environment." >&2
  exit 1
fi
if ! "${SIMPLER_PYTHON}" -c 'import zmq' >/dev/null 2>&1; then
  echo "GR00T client is not ready. Run bash scripts/install_dependencies/install_gr00t.sh first." >&2
  exit 1
fi

case "${TASK}" in
  google_robot*) CHECKPOINT="${PCD_ROOT}/checkpoint/gr00t/GR00T-N1.7-SimplerEnv-Fractal" ;;
  widowx*) CHECKPOINT="${PCD_ROOT}/checkpoint/gr00t/GR00T-N1.7-SimplerEnv-Bridge" ;;
  *) echo "Unsupported task prefix: ${TASK}" >&2; exit 2 ;;
esac

if (( NUM_GPUS < 1 || NUM_GPUS > ${#GPU_DEVICE_ARRAY[@]} )); then
  echo "NUM_GPUS=${NUM_GPUS} is incompatible with GPU_DEVICES=${GPU_DEVICES}" >&2
  exit 2
fi

METHOD_ARGS=()
if [[ "${MODE}" == "pcd-fast" ]]; then
  METHOD_ARGS=(
    --alpha "${ALPHA:-0.2}"
    --mask-by "${MASK_BY:-gt}"
  )
elif [[ "${MODE}" == "pcd-mask" ]]; then
  METHOD_ARGS=(
    --alpha "${ALPHA:-0.2}"
    --mask-by "${MASK_BY:-gt}"
    --inpaint-mode "${INPAINT_MODE:-lama}"
  )
elif [[ "${MODE}" != "baseline" ]]; then
  echo "Usage: $0 TASK {baseline|pcd-mask|pcd-fast}" >&2
  exit 2
fi

RUNNER_ARGS=(
  --num-gpus "${NUM_GPUS}"
  --gpu-ids "${GPU_DEVICES}"
  --policy gr00t
  --method "${MODE}"
  --checkpoint "${CHECKPOINT}"
  --task "${TASK}"
  --result-root "${RESULT_ROOT}"
  --n-trajs "${N_TRAJS}"
  --seed "${SEED}"
)
RUNNER_ARGS+=("${METHOD_ARGS[@]}")

cd "${PCD_ROOT}"
export PCD_CHECKPOINT_ROOT="${PCD_ROOT}/checkpoint"
PYTHONPATH="${PCD_ROOT}:${PYTHONPATH:-}" \
  "${SIMPLER_PYTHON}" parallel_inference.py "${RUNNER_ARGS[@]}"
