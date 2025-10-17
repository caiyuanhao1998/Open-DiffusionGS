&nbsp;

<div align="center">

<h3>Baking Gaussian Splatting into Diffusion Denoiser for Fast <br> and Scalable Single-stage Image-to-3D Generation and Reconstruction</h3> 

[![arXiv](https://img.shields.io/badge/paper-arxiv-179bd3)](https://arxiv.org/abs/2411.14384)
[![project](https://img.shields.io/badge/project-page-green)](https://caiyuanhao1998.github.io/project/DiffusionGS/)
[![hf](https://img.shields.io/badge/hugging-face-green)](https://huggingface.co/datasets/CaiYuanhao/DiffusionGS)
[![MrNeRF](https://img.shields.io/badge/media-MrNeRF-yellow)](https://x.com/janusch_patas/status/1859867424859856997?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Etweet)

<p align="center">
  <img src="img/abo.gif" width="24%" alt="abo">
  <img src="img/gso.gif" width="24%" alt="gso">
  <img src="img/real_img.gif" width="24%" alt="real_img">
  <img src="img/wild.gif" width="24%" alt="wild">
</p>
<p align="center">
  <img src="img/sd_2.gif" width="24%" alt="sd_2">
  <img src="img/sd_1.gif" width="24%" alt="sd_1">
  <img src="img/flux_1.gif" width="24%" alt="flux_1">
  <img src="img/green_man.gif" width="24%" alt="green_man">
</p>
<p align="center">
  <img src="img/plaza.gif" width="50%" alt="plaza">
  <img src="img/town.gif" width="48%" alt="town">
</p>
<p align="center">
  <img src="img/cliff.gif" width="49.5%" alt="cliff">
  <img src="img/art_gallery.gif" width="48.5%" alt="art_gallery">
</p>


&nbsp;

</div>



### Introduction
This is an implementation of our work "Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction
". The code and checkpoints here is a **re-implementation** and **re-training** and **differs** from the original version developed at Adobe. Our DiffusionGS is single-stage and does not rely on 2D multi-view diffusion model. DiffusionGS can be applied to single-view 3D object generation and scene reconstruction without using depth estimator in ~6 seconds. If you find our repo useful, please give it a star ⭐ and consider citing our paper. Thank you :)

![pipeline](img/pipeline.png)


### News
- **2025.10.10 :** Code and models have been released. Feel free to check and use them.  💫
- **2024.11.22 :** Our [project page](https://caiyuanhao1998.github.io/project/DiffusionGS/) has been built up. Feel free to check the video and interactive generation results on the project page.
- **2024.11.21 :** We upload the prompt image and our generation results to our [hugging face dataset](https://huggingface.co/datasets/CaiYuanhao/DiffusionGS). Feel free to download and make a comparison with your method. 🤗
- **2024.11.20 :** Our paper is on [arxiv](https://arxiv.org/abs/2411.14384) now. 🚀

### Comparison with State-of-the-Art Methods


<details close>
<summary><b>Quantitative Comparison in the Paper</b></summary>

![results1](img/compare_table.png)

</details>

<details close>
<summary><b>Qualitative Comparison in the paper</b></summary>

![visual_results](img/compare_figure.png)

</details>

<details open>
<summary><b>Qualitative Comparison between Hunyuan-v2.5 and Our Open-source Version Model</b></summary>


`Note:` The first row is the prompt image. The second row is Hunyuan-v2.5. The third row is our open-source model. Our model only takes 24s for inference, while Hunyuan-v2.5 takes about 180s. Our model is **7.5x** faster.

<p align="center">
  <img src="img/1.jpg" width="32%" alt="1">
  <img src="img/2.jpg" width="32%" alt="2">
  <img src="img/3.jpg" width="32%" alt="3">
</p>
<p align="center">
  <img src="img/hunyuan_1.gif" width="32%" alt="hunyuan_1">
  <img src="img/hunyuan_2.gif" width="32%" alt="hunyuan_2">
  <img src="img/hunyuan_3.gif" width="32%" alt="hunyuan_3">
</p>
<p align="center">
  <img src="img/ours_1.gif" width="32%" alt="ours_1">
  <img src="img/ours_2.gif" width="32%" alt="ours_2">
  <img src="img/ours_3.gif" width="32%" alt="ours_3">
</p>

</details>

&nbsp;

&nbsp;

## 1. Create Environment
```sh
conda create -n diffusiongs python=3.11 -y
conda activate diffusiongs
# conda install -c "nvidia/label/cuda-12.1.1" cudatoolkit
# conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch==2.5.1 torchvision==0.20.1
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

&nbsp;

## 2. Quick Demo
For object-centric image-to-3D generation model, we provide a single-line script to use the code:
```shell
python run.py
```
This code will automatically download the model checkpoints and config files from [HuggingFace](). Or you can manually download it from [this link](https://huggingface.co/coast01/LVSM/tree/main) and set it to local dir.


&nbsp;

## 3. Data Preparation

### 3.1 Scene-level Dataset

Download the RealEstate10K dataset from [this link](http://schadenfreude.csail.mit.edu:8000/), which is provided by [pixelSplat](https://github.com/dcharatan/pixelsplat), and `unzip` the zip file and put the data in `YOUR_RAW_DATAPATH`.
Run the following command to preprocess the data into our format.
```bash
python process_data.py --base_path YOUR_RAW_DATAPATH --output_dir YOUR_PROCESSED_DATAPATH --mode ['train' or 'test']
```

### 3.2 Object-level Dataset

We retrained our model using only the Objaverse dataset, which differs from the approaches adopted by Adobe. Additionally, we provide a dataloader that allows you to leverage the open-source G-Objaverse to train object models from scratch.

For prepare the G-objaverse dataset, please follow the instructions in [G-objaverse](https://github.com/modelscope/richdreamer/tree/main/dataset/gobjaverse).

After you download and unzip the dataset. You can see the following structure:
```
gobjaverse
├──0
    ├── 10010
    ├── 10013
    └── ...          
```

After that, you need to prepare a folder that contains 3 json files call `json` like:
```
json
├── test.json ## set a subset for eval
├── train.json ##Use the download script as the training jsons
└── val.json  ## set a subset for eval
```

Then, specified the `local_dir` to this json file and the `image_dir` to the `gobjaverse` file in the config file (`diffusionGS/configs/diffusionGS_rel.yaml`) so that you can train our model using gobjaverse.


&nbsp;

## 4. Evaluation for Single-view Scene Reconstruction

The scene-level evaluation is conducted on the [RealEstate10K](http://schadenfreude.csail.mit.edu:8000/) dataset prepocessed by [pixelSplat](https://github.com/dcharatan/pixelsplat). The model checkpoints are host on [HuggingFace](https://huggingface.co/CaiYuanhao/DiffusionGS/tree/main). 

| Model | PSNR  | SSIM  | LPIPS |
| ----- | ----- | ----- | ----- |
| [Open-DiffusionGS(res256)](https://huggingface.co/CaiYuanhao/DiffusionGS/blob/main/scene_ckpt_256.ckpt) | 21.26 | 0.672 | 0.257 |
| [Open-DiffusionGS(res512)](https://huggingface.co/CaiYuanhao/DiffusionGS/blob/main/scene_ckpt_512.ckpt) | - | - | - |

We use `./extra_files/evaluation_index_re10k.json` to specify the input and target view indice. This json file is originally from [pixelSplat](https://github.com/dcharatan/pixelsplat). 

We only provide evaluation codes for scene as
```bash
bash script/eval.sh
```

This code will evaluate all testsets, and generate the `.pt` for you to caculate metrics, if you want to store the scene videos and gaussians, plese turn `system.save_intermediate_video` to `True` in the config file (`diffusiongs/configs/diffusionGS_scene_eval.yaml`).

After run this codes, the result will specified in `{exp_root_dir}/{name}/{tags}` in the config file.

For provided config, the result will be form like
```
outputs/diffusion_gs_scene_re10k_256_stage1_eval/diffusion-gs-model-scene+lr0.0001/save/it0
├── 0a3b5fb184936a83.pt
├── 0a4cf8d9b81b4c6e.pt
├── ...
```
each `.pt` store the rendered images and gt images for you to calculate metrics. if you turn `system.save_intermediate_video = True` you will see rendered videos of the scene.

If you want to calculate metrics, please run:
```bash
bash cal_metrics.sh
```
after you replace the `exp_root_dir` in `cal_metrics.sh`, you can run this script to calculate metrics.

&nbsp;


## 5. Training
We provide 4 stages training scripts for you to train your own models:
```bash
bash scripts/train_scene_stage1.py # train object model (res256)
bash scripts/train_scene_stage2.py # train object model (res512)
bash scripts/train_obj_stage1.py  # train scene model (res256)
bash scripts/train_obj_stage2.py  # train scene model (res512)
```
Before training, you need to specified your data path in the config files by replace `local_dir` to your processed `RealEstate10K` (For scene). Or `image_dir` and `local_dir` to the `gobjaverse` file and the prepared json folder (For object).

Note: when you train the second stage model, remember to replace   `shape_model.pretrained_model_name_or_path: ` to the trained first stage checkpoint.

&nbsp;

## 6. Citation
```sh
@inproceedings{diffusiongs,
  title={Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation and Reconstruction},
  author={Yuanhao Cai and He Zhang and Kai Zhang and Yixun Liang and Mengwei Ren and Fujun Luan and Qing Liu and Soo Ye Kim and Jianming Zhang and Zhifei Zhang and Yuqian Zhou and Yulun Zhang and Xiaokang Yang and Zhe Lin and Alan Yuille},
  booktitle={ICCV},
  year={2025}
}
```

&nbsp;

## Acknowledgments
We would like to thank the following projects: [DiffSplat](https://github.com/chenguolin/DiffSplat), [CraftsMan3D](https://github.com/HKUST-SAIL/CraftsMan3D), [LVSM](https://github.com/Haian-Jin/LVSM).
