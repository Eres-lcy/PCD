#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${project_root}/scripts/inference/gr00t_support/eval_gr00t_multigpu_common.sh"

gpu_devices="${GPU_DEVICES:-0,1,2,3}"
IFS=',' read -r -a gpu_device_array <<< "${gpu_devices}"
num_gpus="${NUM_GPUS:-${#gpu_device_array[@]}}"
n_trajs="${N_TRAJS:-100}"
seeds=("${SEED:-0}")
result_root="${RESULT_ROOT:-${project_root}/Results}"
server_start_timeout="${SERVER_START_TIMEOUT:-900}"
runtime_root="${RUNTIME_ROOT:-${project_root}/runtime/gr00t/server_logs}"

tasks=(
    google_robot_pick_coke_can
    google_robot_move_near
    google_robot_close_drawer
    google_robot_open_drawer
    widowx_carrot_on_plate
    widowx_spoon_on_towel
    widowx_put_eggplant_in_basket
    widowx_stack_cube
    google_robot_place_apple_in_closed_top_drawer
)

run_task() {
    GPU_DEVICES="${gpu_devices}" \
    NUM_GPUS="${num_gpus}" \
    N_TRAJS="${n_trajs}" \
    SEED="$2" \
    RESULT_ROOT="${result_root}" \
        bash "${project_root}/scripts/inference/gr00t_support/eval_gr00t.sh" "$1" baseline
}

run_gr00t_multigpu baseline run_task
