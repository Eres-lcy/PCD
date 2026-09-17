#!/usr/bin/env bash

gr00t_server_pids=()

gr00t_stop_servers() {
    local pid
    for pid in "${gr00t_server_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    for pid in "${gr00t_server_pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
    done
    gr00t_server_pids=()
}

gr00t_validate_common_config() {
    local task robot
    IFS=',' read -r -a gr00t_gpu_array <<< "${gpu_devices}"
    if (( num_gpus < 1 || num_gpus > ${#gr00t_gpu_array[@]} )); then
        echo "num_gpus=${num_gpus} is incompatible with gpu_devices=${gpu_devices}." >&2
        return 2
    fi
    if (( ${#tasks[@]} == 0 )); then
        echo "tasks must contain at least one task." >&2
        return 2
    fi
    if (( ${#seeds[@]} == 0 )); then
        echo "seeds must contain at least one seed." >&2
        return 2
    fi
    for task in "${tasks[@]}"; do
        task="${task//[[:space:]]/}"
        case "${task}" in
            google_robot*) robot=google ;;
            widowx*) robot=widowx ;;
            *)
                echo "Unsupported task name: ${task}" >&2
                return 2
                ;;
        esac
    done
}

gr00t_start_servers() {
    local robot="$1"
    local mode="$2"
    local model_path embodiment gpu port log_path index deadline pid

    case "${robot}" in
        google) model_path="${project_root}/checkpoint/gr00t/GR00T-N1.7-SimplerEnv-Fractal" ;;
        widowx) model_path="${project_root}/checkpoint/gr00t/GR00T-N1.7-SimplerEnv-Bridge" ;;
        *) echo "Unsupported robot: ${robot}" >&2; return 2 ;;
    esac
    if [[ "${model_path}" = /* && ! -d "${model_path}" ]]; then
        echo "Checkpoint directory is missing: ${model_path}" >&2
        return 1
    fi

    current_server_logs=()
    gr00t_server_pids=()
    local run_dir="${runtime_root}/${mode}/${robot}"
    mkdir -p "${run_dir}"

    for ((index = 0; index < num_gpus; index++)); do
        gpu="${gr00t_gpu_array[$index]//[[:space:]]/}"
        port=$((5555 + index))
        log_path="${run_dir}/server_gpu${gpu}_port${port}.log"
        current_server_logs+=("${log_path}")
        echo "Starting GR00T ${robot} server: gpu=${gpu}, port=${port}, log=${log_path}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            GR00T_DEVICE="cuda:0" \
            GR00T_PORT="${port}" \
            GR00T_MODEL_PATH="${model_path}" \
            bash "${project_root}/scripts/inference/gr00t_support/run_gr00t_server.sh" "${robot}" \
            >"${log_path}" 2>&1 &
        gr00t_server_pids+=("$!")
    done

    deadline=$((SECONDS + server_start_timeout))
    for ((index = 0; index < num_gpus; index++)); do
        pid="${gr00t_server_pids[$index]}"
        log_path="${current_server_logs[$index]}"
        while ! grep -q "GR00T-PCD-Fast ready" "${log_path}" 2>/dev/null; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "GR00T server exited during startup: ${log_path}" >&2
                tail -n 80 "${log_path}" >&2 || true
                return 1
            fi
            if (( SECONDS >= deadline )); then
                echo "Timed out after ${server_start_timeout}s: ${log_path}" >&2
                tail -n 40 "${log_path}" >&2 || true
                return 1
            fi
            sleep 2
        done
    done
    echo "All ${num_gpus} GR00T ${robot} servers are ready."
}

run_gr00t_multigpu() {
    local mode="$1"
    local task_callback="$2"
    local seed robot task has_tasks

    gr00t_validate_common_config
    trap gr00t_stop_servers EXIT INT TERM

    echo "GR00T ${mode}: GPUs=${gpu_devices}, num_gpus=${num_gpus}, trajectories=${n_trajs}"
    for seed in "${seeds[@]}"; do
        seed="${seed//[[:space:]]/}"
        for robot in google widowx; do
            has_tasks=0
            for task in "${tasks[@]}"; do
                task="${task//[[:space:]]/}"
                if [[ ( "${robot}" == google && "${task}" == google_robot* ) || \
                      ( "${robot}" == widowx && "${task}" == widowx* ) ]]; then
                    has_tasks=1
                    break
                fi
            done
            (( has_tasks == 1 )) || continue

            gr00t_start_servers "${robot}" "${mode}"
            for task in "${tasks[@]}"; do
                task="${task//[[:space:]]/}"
                if [[ ( "${robot}" == google && "${task}" == google_robot* ) || \
                      ( "${robot}" == widowx && "${task}" == widowx* ) ]]; then
                    "${task_callback}" "${task}" "${seed}"
                fi
            done
            gr00t_stop_servers
        done
    done
}
