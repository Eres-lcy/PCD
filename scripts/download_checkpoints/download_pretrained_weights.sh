#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pcd_root="$(cd -- "${script_dir}/../.." && pwd)"
checkpoint_root="${PCD_CHECKPOINT_ROOT:-${pcd_root}/checkpoint}"

grounding_dino_revision="12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
openvla_revision="31f090d05236101ebfc381b61c674dd4746d4ce0"
pi0_revision="8518347d4ae0c6cfc69fbdda970b3f38c6ff76ca"
paligemma_revision="35e4f46485b4d07967e7e9935bc3786aad50687c"

usage() {
    cat <<'EOF'
Usage: download_pretrained_weights.sh [common|openvla|pi0|all]...

With no arguments, downloads all non-GR00T checkpoints. GR00T checkpoints are
handled by download_gr00t_models.sh in this directory.
EOF
}

download_common=false
download_openvla=false
download_pi0=false

if (( $# == 0 )); then
    set -- all
fi

for component in "$@"; do
    case "${component}" in
        common) download_common=true ;;
        openvla) download_openvla=true ;;
        pi0) download_pi0=true ;;
        all)
            download_common=true
            download_openvla=true
            download_pi0=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown checkpoint group: ${component}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${HF_BIN:-}" ]]; then
    hf=("${HF_BIN}")
elif command -v hf >/dev/null 2>&1; then
    hf=("$(command -v hf)")
elif command -v huggingface-cli >/dev/null 2>&1; then
    hf=("$(command -v huggingface-cli)")
else
    echo "Hugging Face CLI was not found. Install huggingface-hub first." >&2
    exit 1
fi

download_url() {
    local url="$1"
    local output="$2"
    local partial="${output}.part"

    if [[ -s "${output}" ]]; then
        echo "Using existing file: ${output}"
        return
    fi

    mkdir -p "$(dirname -- "${output}")"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --output "${partial}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="${partial}" "${url}"
    else
        echo "Neither curl nor wget is available." >&2
        exit 1
    fi
    mv -- "${partial}" "${output}"
}

mkdir -p "${checkpoint_root}"

if [[ "${download_common}" == true ]]; then
    download_url \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
        "${checkpoint_root}/sam2.1_hiera_large.pt"
    "${hf[@]}" download \
        IDEA-Research/grounding-dino-base \
        --revision "${grounding_dino_revision}" \
        --local-dir "${checkpoint_root}/grounding-dino-base"

    if [[ ! -f "${checkpoint_root}/big-lama/config.yaml" ]]; then
        cat >&2 <<EOF
Warning: LaMa was not downloaded automatically.
Download the big-lama directory from:
  https://drive.google.com/drive/folders/1ST0aRbDRZGli0r7OVVOQvXwtadMCuWXg?usp=sharing
and place it at:
  ${checkpoint_root}/big-lama
EOF
    fi
fi

if [[ "${download_openvla}" == true ]]; then
    "${hf[@]}" download \
        openvla/openvla-7b \
        --revision "${openvla_revision}" \
        --local-dir "${checkpoint_root}/openvla-7b"
fi

if [[ "${download_pi0}" == true ]]; then
    echo "Pi0 requires access to google/paligemma-3b-pt-224; authenticate with 'hf auth login' if needed."
    "${hf[@]}" download \
        allenzren/open-pi-zero \
        bridge_beta_step19296_2024-12-26_22-30_42.pt \
        fractal_beta_step29576_2024-12-29_13-10_42.pt \
        --revision "${pi0_revision}" \
        --local-dir "${checkpoint_root}/pi0"
    "${hf[@]}" download \
        google/paligemma-3b-pt-224 \
        --revision "${paligemma_revision}" \
        --local-dir "${checkpoint_root}/pi0/paligemma-3b"
fi

echo "Downloaded requested checkpoints under ${checkpoint_root}"
