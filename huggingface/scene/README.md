---
title: DiffusionGS Scene Generation
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
short_description: Image-to-3D scenes, videos and point-cloud downloads
models:
  - CaiYuanhao/DiffusionGS
datasets:
  - CaiYuanhao/DiffusionGS
tags:
  - image-to-3d
  - image-to-video
  - gaussian-splatting
  - diffusiongs
---

# DiffusionGS · Scene Generation

Generate a 3D scene from one reference image. Upload an image or choose an example, click **Generate scene & video**, then preview the generated views and video and download the Gaussian point cloud.

[Object demo](https://huggingface.co/spaces/CaiYuanhao/DiffusionGS-Object) · [GitHub](https://github.com/caiyuanhao1998/Open-DiffusionGS) · [Models](https://huggingface.co/CaiYuanhao/DiffusionGS) · [Paper](https://arxiv.org/abs/2411.14384)

![Scene generation interface](img/demos/scene_demo.png)

This Space embeds the Scene Gradio interface hosted at the shared [ZeroGPU backend](https://caiyuanhao-diffusiongs-object.hf.space/scene). It runs the same `scene_demo.py` as the local demo, including its examples, controls, video preview and downloads.
