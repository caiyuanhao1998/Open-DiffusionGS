export OPENCV_IO_ENABLE_OPENEXR=1
source activate diffusiongs
TORCHELASTIC_TIMEOUT=18000 torchrun  --standalone --nnodes=1 --nproc-per-node=1 \
    launch.py --validate --use_ema --gpu 0 \
    --config diffusionGS/configs/diffusionGS_scene_eval.yaml 