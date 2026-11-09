---
title: DiffusionGS Object Generation
emoji: 🧸
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12.12
app_file: space_app.py
pinned: false
short_description: Image-to-3D objects with point-cloud and mesh downloads
models:
  - CaiYuanhao/DiffusionGS
datasets:
  - CaiYuanhao/DiffusionGS
tags:
  - image-to-3d
  - gaussian-splatting
  - diffusiongs
---

# DiffusionGS · Object Generation

Generate a 3D object from one reference image. Upload an image or choose an example, click **Generate object**, then preview and download the rendered views and Gaussian point cloud. **Extract mesh** creates a colored mesh preview with GLB and OBJ downloads.

[Scene demo](https://huggingface.co/spaces/CaiYuanhao/DiffusionGS-Scene) · [GitHub](https://github.com/caiyuanhao1998/Open-DiffusionGS) · [Models](https://huggingface.co/CaiYuanhao/DiffusionGS) · [Paper](https://arxiv.org/abs/2411.14384)

![Object generation interface](img/demos/object_demo.png)

The Object and Scene demos share this Space's ZeroGPU backend. The Scene interface is also available at `/scene`. Models are loaded from the official model repository at startup; GPU time is allocated only during inference.
