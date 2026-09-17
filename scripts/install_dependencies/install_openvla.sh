#!/usr/bin/env bash
set -euo pipefail

environment_name="pcd-openvla"
pytorch_index_url="https://download.pytorch.org/whl/cu121"
openvla_repository="https://github.com/openvla/openvla.git"
openvla_commit="c8f03f48af692657d3060c19588038c7220e9af9"
dlimp_commit="040105d256bd28866cc6620621a3d5f7b6b91b46"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pcd_root="$(cd -- "${script_dir}/../.." && pwd)"
third_party_root="${pcd_root}/third_party"
openvla_root="${third_party_root}/openvla"
maniskill_root="${third_party_root}/ManiSkill2_real2sim"
sam2_root="${third_party_root}/grounded_sam_2"
lama_requirements="${third_party_root}/inpaint_anything/lama/requirements.txt"
grounding_dino_patch="${script_dir}/transformers_grounding_dino_patch/src/transformers/kernels/grounding_dino"

mkdir -p "${third_party_root}"
if [[ ! -d "${openvla_root}" ]]; then
    if [[ -e "${openvla_root}" ]]; then
        echo "${openvla_root} exists but is not a directory" >&2
        exit 1
    fi
    git clone "${openvla_repository}" "${openvla_root}"
    git -C "${openvla_root}" checkout --detach "${openvla_commit}"
else
    echo "Using existing OpenVLA directory: ${openvla_root}"
fi

for required_path in "${maniskill_root}" "${sam2_root}" "${lama_requirements}" "${grounding_dino_patch}" "${pcd_root}/setup.py"; do
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
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0

"${pip[@]}" install \
    numpy==1.26.4 \
    opencv-python==4.6.0.66 \
    opencv-python-headless==4.6.0.66
"${pip[@]}" install -r "${lama_requirements}"

"${pip[@]}" install \
    accelerate==1.12.0 \
    draccus==0.8.0 \
    einops==0.8.1 \
    huggingface-hub==0.36.0 \
    json-numpy==2.1.1 \
    jsonlines==4.0.0 \
    matplotlib==3.10.8 \
    mediapy==1.2.5 \
    peft==0.11.1 \
    protobuf==4.25.8 \
    rich==14.2.0 \
    sentencepiece==0.1.99 \
    tensorflow==2.15.0 \
    tensorflow-datasets==4.9.3 \
    tensorflow-metadata==1.17.3 \
    tensorflow-graphics==2021.12.3 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1 \
    wandb==0.24.0 \
    numpy==1.26.4 \
    opencv-python==4.6.0.66 \
    opencv-python-headless==4.6.0.66

"${pip[@]}" install \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla@${dlimp_commit}"
"${pip[@]}" install --no-deps -e "${openvla_root}"
"${pip[@]}" install -e "${maniskill_root}"
"${pip[@]}" install --no-deps -e "${pcd_root}"
"${pip[@]}" install --no-build-isolation -e "${sam2_root}"

"${pip[@]}" install ninja==1.13.0 packaging==25.0
"${pip[@]}" install --no-build-isolation flash-attn==2.5.5

transformers_root="$("${run[@]}" python -c 'from pathlib import Path; import transformers; print(Path(transformers.__file__).resolve().parent)')"
mkdir -p "${transformers_root}/kernels/grounding_dino"
cp -a \
    "${grounding_dino_patch}/." \
    "${transformers_root}/kernels/grounding_dino/"

"${pip[@]}" check
"${run[@]}" env PYTHONPATH="${pcd_root}:${openvla_root}" python -c \
    'import simpler_env, timm, tokenizers, torch, transformers; import contrast_policies.openvla.pcd_fast_action_generation; assert torch.__version__.startswith("2.2.0"); assert timm.__version__ == "0.9.10"; assert tokenizers.__version__ == "0.19.1"; assert transformers.__version__ == "4.40.1"; print("pcd-openvla is ready")'
