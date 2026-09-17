#!/usr/bin/env bash
set -euo pipefail

environment_name="pcd-pi0"
pytorch_index_url="https://download.pytorch.org/whl/cu124"
pi0_repository="https://github.com/allenzren/open-pi-zero.git"
pi0_commit="c3df7fb062175c16f69d7ca4ce042958ea238fb7"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pcd_root="$(cd -- "${script_dir}/../.." && pwd)"
third_party_root="${pcd_root}/third_party"
pi0_root="${third_party_root}/pi0"
maniskill_root="${third_party_root}/ManiSkill2_real2sim"
sam2_root="${third_party_root}/grounded_sam_2"
lama_requirements="${third_party_root}/inpaint_anything/lama/requirements.txt"

mkdir -p "${third_party_root}"
if [[ ! -d "${pi0_root}" ]]; then
    if [[ -e "${pi0_root}" ]]; then
        echo "${pi0_root} exists but is not a directory" >&2
        exit 1
    fi
    git clone "${pi0_repository}" "${pi0_root}"
    git -C "${pi0_root}" checkout --detach "${pi0_commit}"
else
    echo "Using existing Pi0 directory: ${pi0_root}"
fi

for required_path in "${maniskill_root}" "${sam2_root}" "${lama_requirements}" "${pcd_root}/setup.py"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required repository path is missing: ${required_path}" >&2
        exit 1
    fi
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found" >&2
    exit 1
fi

if ! conda env list | awk -v name="${environment_name}" '$1 == name { found = 1 } END { exit !found }'; then
    conda create -y -n "${environment_name}" python=3.10
fi

run=(conda run --no-capture-output -n "${environment_name}")
pip=("${run[@]}" python -m pip)

"${pip[@]}" install --upgrade pip setuptools==80.9.0 wheel==0.45.1
"${pip[@]}" install \
    --index-url="${pytorch_index_url}" \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0

"${pip[@]}" install \
    numpy==1.26.4 \
    opencv-python==4.6.0.66 \
    opencv-python-headless==4.6.0.66
"${pip[@]}" install -r "${lama_requirements}"

"${pip[@]}" install \
    accelerate==1.14.0 \
    bitsandbytes==0.45.0 \
    einops==0.8.2 \
    gsutil==5.35 \
    huggingface-hub==0.36.1 \
    hydra-core==1.3.2 \
    imageio==2.37.2 \
    matplotlib==3.10.8 \
    mediapy==1.2.6 \
    numpy==1.26.4 \
    omegaconf==2.3.1 \
    opencv-python==4.6.0.66 \
    opencv-python-headless==4.6.0.66 \
    pillow==12.1.0 \
    pre-commit==4.5.1 \
    pretty-errors==1.2.25 \
    protobuf==3.20.3 \
    scipy==1.15.3 \
    tensorflow==2.15.0 \
    tensorflow-datasets==4.9.2 \
    tensorflow-metadata==1.16.1 \
    timm==1.0.28 \
    tokenizers==0.21.4 \
    transformers==4.47.1 \
    tqdm==4.67.2 \
    wandb==0.24.1

"${pip[@]}" install --no-deps -e "${pi0_root}"
"${pip[@]}" install -e "${maniskill_root}"
"${pip[@]}" install --no-deps -e "${pcd_root}"
"${pip[@]}" install --no-build-isolation -e "${sam2_root}"

"${pip[@]}" check
"${run[@]}" env PYTHONPATH="${pcd_root}:${pi0_root}" python -c \
    'import simpler_env, timm, tokenizers, torch, transformers; import contrast_policies.pizero.pcd_mask_policy_output; import contrast_policies.pizero.pcd_fast_policy_output; assert torch.__version__.startswith("2.5.0"); assert timm.__version__ == "1.0.28"; assert tokenizers.__version__ == "0.21.4"; assert transformers.__version__ == "4.47.1"; print("pcd-pi0 is ready")'
