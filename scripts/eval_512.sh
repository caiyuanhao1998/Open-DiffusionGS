#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/launch_common.sh"
launch_diffusiongs --validate diffusionGS/configs/diffusionGS_scene_eval_512.yaml "$@"
