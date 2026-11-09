#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/launch_common.sh"
launch_diffusiongs --train diffusionGS/configs/diffusionGS_scene.yaml "$@"
