export OPENCV_IO_ENABLE_OPENEXR=1
source activate diffusiongs
python eval_scene_result.py --path outputs/diffusion_gs_scene_re10k_256_stage1_eval/diffusion-gs-model-scene+lr0.0001/save/it0 --chunk 64