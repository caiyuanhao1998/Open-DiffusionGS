export OPENCV_IO_ENABLE_OPENEXR=1
source activate diffusiongs

#wandb
TORCHELASTIC_TIMEOUT=18000 torchrun  --standalone --nnodes=1 --nproc-per-node=8 \
    launch.py --train --use_ema --gpu 0,1,2,3,4,5,6,7 \
    --config diffusionGS/configs/diffusionGS_rel.yaml 