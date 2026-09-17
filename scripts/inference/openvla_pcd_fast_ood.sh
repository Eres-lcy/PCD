#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
runner="${project_root}/parallel_inference.py"
checkpoint="${project_root}/checkpoint/openvla-7b"
result_root="${RESULT_ROOT:-${project_root}/Results}"

gpu_devices="${GPU_DEVICES:-0,1,2,3}"
IFS=',' read -r -a gpu_device_array <<< "${gpu_devices}"
num_gpus="${NUM_GPUS:-${#gpu_device_array[@]}}"
n_trajs="${N_TRAJS:-100}"
seed="${SEED:-0}"
alphas=(0.8)
tracking_modes=(grounded_sam_tracking)

tasks=(
    google_robot_pick_coke_can
    google_robot_pick_coke_can_drawer_variant
    google_robot_pick_coke_can_light_variant
    google_robot_pick_coke_can_dark
    google_robot_pick_coke_can_table_paper_variant
    google_robot_pick_coke_can_table_stone_variant
)

export PCD_CHECKPOINT_ROOT="${project_root}/checkpoint"
export PYTHONPATH="${project_root}:${project_root}/third_party/ManiSkill2_real2sim${PYTHONPATH:+:${PYTHONPATH}}"

for alpha in "${alphas[@]}"; do
    for tracking_mode in "${tracking_modes[@]}"; do
        for task in "${tasks[@]}"; do
            "${python_bin}" "${runner}" \
                --policy openvla \
                --method pcd-fast \
                --checkpoint "${checkpoint}" \
                --task "${task}" \
                --result-root "${result_root}" \
                --n-trajs "${n_trajs}" \
                --seed "${seed}" \
                --num-gpus "${num_gpus}" \
                --gpu-ids "${gpu_devices}" \
                --alpha "${alpha}" \
                --mask-by "${tracking_mode}"
        done
    done
done
