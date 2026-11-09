#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export OPENCV_IO_ENABLE_OPENEXR=1

# Keep the caller's activated environment and selected devices.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a selected_devices <<< "$CUDA_VISIBLE_DEVICES"
    default_processes=${#selected_devices[@]}
else
    default_processes=1
fi
processes=${NPROC_PER_NODE:-$default_processes}
if [[ ! "$processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$processes" -ne "$default_processes" ]]; then
    echo "NPROC_PER_NODE must match the number of CUDA_VISIBLE_DEVICES entries" >&2
    exit 2
fi
gpu_ids=$(seq -s, 0 "$((processes - 1))")

launch_diffusiongs() {
    local mode=$1 config=$2
    shift 2
    TORCHELASTIC_TIMEOUT=${TORCHELASTIC_TIMEOUT:-18000} \
        torchrun --standalone --nnodes=1 --nproc-per-node="$processes" \
        launch.py "$mode" --use_ema --gpu "$gpu_ids" --config "$config" "$@"
}
