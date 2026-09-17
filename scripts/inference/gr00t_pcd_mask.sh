#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${project_root}/scripts/inference/gr00t_support/eval_gr00t_multigpu_common.sh"

gpu_devices="${GPU_DEVICES:-0,1,2,3}"
IFS=',' read -r -a gpu_device_array <<< "${gpu_devices}"
num_gpus="${NUM_GPUS:-${#gpu_device_array[@]}}"
n_trajs="${N_TRAJS:-100}"
seeds=("${SEED:-0}")
alphas=(0.2)
tracking_modes=(grounded_sam_tracking) #point_tracking box_tracking 
inpaint_modes=(lama)
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

export PCD_CHECKPOINT_ROOT="${project_root}/checkpoint"

run_task() {
    local alpha tracking_mode inpaint_mode
    for alpha in "${alphas[@]}"; do
        for tracking_mode in "${tracking_modes[@]}"; do
            for inpaint_mode in "${inpaint_modes[@]}"; do
                GPU_DEVICES="${gpu_devices}" \
                NUM_GPUS="${num_gpus}" \
                N_TRAJS="${n_trajs}" \
                SEED="$2" \
                RESULT_ROOT="${result_root}" \
                ALPHA="${alpha}" \
                MASK_BY="${tracking_mode}" \
                INPAINT_MODE="${inpaint_mode}" \
                    bash "${project_root}/scripts/inference/gr00t_support/eval_gr00t.sh" "$1" pcd-mask
            done
        done
    done
}

run_gr00t_multigpu pcd-mask run_task
