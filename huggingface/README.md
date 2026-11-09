# Hugging Face Spaces deployment

The [Object Space](https://huggingface.co/spaces/CaiYuanhao/DiffusionGS-Object) hosts both Gradio pages on one ZeroGPU allocation. The [Scene Space](https://huggingface.co/spaces/CaiYuanhao/DiffusionGS-Scene) embeds the `/scene` page. Both pages reuse the repository's demo scripts. This keeps Retinexformer running within the account's two free ZeroGPU Space slots.

The backend uses Python 3.12, PyTorch 2.9.1 with CUDA 12.8, Gradio 5.50.0 and Spaces 0.51.3. Its rasterizer wheel is compiled from this repository's `submodules/diff-gaussian-rasterization` for SM 80, 90 and 120, with forward-compatible PTX. PyTorch SDPA replaces the local environment's xFormers kernels. The object loader omits training losses; all three released denoisers are preloaded for ZeroGPU.

Build the wheel in a Python 3.12 environment containing PyTorch 2.9.1, setuptools, wheel and ninja:

```sh
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST='8.0;9.0;12.0+PTX' \
  MAX_JOBS=4 python -m pip wheel --no-build-isolation --no-deps \
  ./submodules/diff-gaussian-rasterization --wheel-dir /path/to/wheels
```

With the two Spaces already created and Hugging Face authentication configured, publish their source and assets:

```sh
python huggingface/deploy.py \
  --wheel /path/to/wheels/diff_gaussian_rasterization-0.0.0-cp312-cp312-linux_x86_64.whl
```
