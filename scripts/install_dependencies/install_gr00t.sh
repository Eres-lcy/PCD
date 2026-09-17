#!/usr/bin/env bash
set -euo pipefail

environment_name="pcd-gr00t"
pytorch_index_url="https://download.pytorch.org/whl/cu124"
gr00t_repository="https://github.com/NVIDIA/Isaac-GR00T.git"
gr00t_commit="51d4c89f72fda44cbf77285c6a8114b52676b8a1"
uv_version="0.8.18"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pcd_root="$(cd -- "${script_dir}/../.." && pwd)"
third_party_root="${pcd_root}/third_party"
gr00t_root="${third_party_root}/Isaac-GR00T"
maniskill_root="${third_party_root}/ManiSkill2_real2sim"
sam2_root="${third_party_root}/grounded_sam_2"
lama_requirements="${third_party_root}/inpaint_anything/lama/requirements.txt"
lfs_wheel="scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl"

mkdir -p "${third_party_root}"
if [[ ! -d "${gr00t_root}" ]]; then
    if [[ -e "${gr00t_root}" ]]; then
        echo "${gr00t_root} exists but is not a directory" >&2
        exit 1
    fi
    GIT_LFS_SKIP_SMUDGE=1 git clone --no-recurse-submodules \
        "${gr00t_repository}" "${gr00t_root}"
    GIT_LFS_SKIP_SMUDGE=1 git -C "${gr00t_root}" checkout --detach "${gr00t_commit}"
else
    echo "Using existing GR00T directory: ${gr00t_root}"
fi

for required_path in \
    "${gr00t_root}/pyproject.toml" \
    "${gr00t_root}/uv.lock" \
    "${gr00t_root}/${lfs_wheel}" \
    "${maniskill_root}" \
    "${sam2_root}" \
    "${lama_requirements}" \
    "${pcd_root}/setup.py"; do
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
uv=("${run[@]}" uv)

"${pip[@]}" install --upgrade pip setuptools==80.9.0 wheel==0.45.1
"${pip[@]}" install "uv==${uv_version}"
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
    protobuf==3.20.3 \
    pyzmq==27.0.1 \
    scipy==1.15.3 \
    tensorflow==2.15.0 \
    tensorflow-datasets==4.9.2 \
    tensorflow-metadata==1.16.1 \
    timm==1.0.28 \
    tokenizers==0.21.4 \
    transformers==4.47.1 \
    tqdm==4.67.2

"${pip[@]}" install -e "${maniskill_root}"
"${pip[@]}" install --no-deps -e "${pcd_root}"
"${pip[@]}" install --no-build-isolation -e "${sam2_root}"

if ! grep -q "return_all_hidden_states" "${gr00t_root}/gr00t/model/modules/dit.py"; then
    echo "The GR00T source does not expose the hidden states required by PCD-Fast" >&2
    exit 1
fi

if grep -aq '^version https://git-lfs.github.com/spec/v1' "${gr00t_root}/${lfs_wheel}"; then
    if [[ ! -d "${gr00t_root}/.git" ]] || ! git lfs version >/dev/null 2>&1; then
        echo "Git LFS is required to materialize ${lfs_wheel}" >&2
        exit 1
    fi
    git -C "${gr00t_root}" lfs pull --include="${lfs_wheel}" --exclude=""
fi

uv_http_timeout="${UV_HTTP_TIMEOUT:-600}"
(
    cd "${gr00t_root}"
    UV_HTTP_TIMEOUT="${uv_http_timeout}" "${uv[@]}" sync --locked --python 3.12
)

"${pip[@]}" check
"${run[@]}" env PYTHONPATH="${pcd_root}" python -c \
    'import mediapy, simpler_env, tensorflow, timm, transformers, zmq; import contrast_policies.gr00t.remote_policy; assert timm.__version__ == "1.0.28"; assert transformers.__version__ == "4.47.1"; print("pcd-gr00t client is ready")'
env PYTHONPATH="${pcd_root}" "${gr00t_root}/.venv/bin/python" -c \
    'import sys, torch, transformers; import gr00t; import contrast_policies.gr00t.server; assert sys.version_info[:2] == (3, 12); assert torch.__version__.startswith("2.9.0"); assert transformers.__version__ == "4.57.3"; print("GR00T server is ready")'
