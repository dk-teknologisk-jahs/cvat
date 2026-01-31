#!/bin/bash
# Sample commands to deploy nuclio functions on GPU
# Usage: ./deploy_gpu.sh [functions_dir] [gpu_id]
# Example: ./deploy_gpu.sh serverless/pytorch/facebookresearch/sam2 1

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
FUNCTIONS_DIR=${1:-$SCRIPT_DIR}
GPU_ID=${2:-}

nuctl create project cvat --platform local

shopt -s globstar

for func_config in "$FUNCTIONS_DIR"/**/function-gpu.yaml
do
    func_root="$(dirname "$func_config")"
    func_rel_path="$(realpath --relative-to="$SCRIPT_DIR" "$(dirname "$func_root")")"

    echo "Deploying $func_rel_path function..."

    if [ -n "$GPU_ID" ]; then
        # Deploy with specific GPU device
        # Nuclio's local platform uses --gpus all when nvidia.com/gpu is set,
        # which ignores NVIDIA_VISIBLE_DEVICES. However, CUDA_VISIBLE_DEVICES
        # is read by PyTorch/CUDA at runtime and works regardless of Docker GPU mount.
        nuctl deploy --project-name cvat --path "$func_root" \
            --file "$func_config" --platform local \
            --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
            --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
            --env CUDA_VISIBLE_DEVICES="$GPU_ID" \
            --platform-config '{"attributes": {"network": "cvat_cvat"}}'
    else
        nuctl deploy --project-name cvat --path "$func_root" \
            --file "$func_config" --platform local \
            --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
            --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
            --platform-config '{"attributes": {"network": "cvat_cvat"}}'
    fi
done

nuctl get function --platform local
