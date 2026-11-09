# README validation — 2026-11-09

The README was checked in an independent clone, starting at `c61e412`, with a newly created Conda environment. The original instructions were run first to capture failures. The corrected installation was then run in a second empty Conda environment. Neither environment inherited the existing demo environment or its compiled extensions. Released model files and download caches were reused where available.

## Environment

Linux, NVIDIA RTX A5000 (24 GB), Python 3.11.18, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, xFormers 0.0.29.post1, CUDA toolkit 12.1, GCC/G++ 12, Gradio 5.5.0, Pydantic 2.10.6, setuptools 80.9.0, PyArrow 16.1.0 and OpenCV headless 4.9.0. Both CUDA extensions were compiled from the clean checkout. `pip check` passed, and module paths were checked to ensure imports came from the new environment and checkout.

## Command coverage

| README command or workflow | Result and scope |
| --- | --- |
| Clone, Conda creation/activation, PyTorch installation, `pip install -r requirement.txt` | Passed in the new environment using the corrected instructions. |
| Both editable CUDA extension installations | Passed with `--no-build-isolation`. The KNN CUDA result also matched a brute-force three-neighbor reference. |
| `python object_demo.py` | Default URL `http://127.0.0.1:7863` returned HTTP 200. Browser generation was checked on the separate audit port 17863: seven examples, rendered views, an interactive point cloud and a real PLY download; no browser exceptions. |
| `python object_infer.py` | Passed with the default image, released model and mesh extraction enabled. Saved PNG, PLY and OBJ into a newly created output directory. |
| `python scene_demo.py` | Browser test at the default `http://127.0.0.1:7860`: three examples, successful generation, ten-second video playback and a real PLY download; no browser exceptions. |
| `python scene_infer.py` | Passed with the default released 256px model and image. Saved views, point cloud, video and camera JSON. |
| `process_data.py --mode train` and `--mode test` | Both passed on one genuine 16-frame RealEstate10K reference clip packaged in pixelSplat's `.torch` structure. Decoded images, camera metadata and `full_list.txt` were checked. |
| `bash scripts/eval.sh` | Passed with the released 256px checkpoint, one scene and five requested target views. |
| `bash scripts/eval_512.sh` | Passed with the released 512px checkpoint, the same scene and five requested target views. |
| `bash scripts/cal_metrics.sh --path ... --chunk 1` | Passed on both evaluation outputs, including single-image LPIPS batches. PSNR, SSIM and LPIPS were finite and `num_images` was five. |
| All four training scripts | Each completed eight steps with a small model and fixture dataset. Checkpoints had finite weights and nonzero AdamW optimizer moments. Each second stage loaded its first-stage Lightning checkpoint. |
| Two-GPU object training | Completed eight steps with socket transport overrides; see the host-specific note below. |

The scene fixture used clip `0bf152ef84195293`, with target indices `[2, 5, 8, 11, 15]`. Evaluation retained all five targets and excluded the conditioning image. The object training fixture reproduced G-Objaverse's PNG/camera JSON/EXR directory structure using generated data; it was not a real Objaverse benchmark sample.

Training smoke tests used 32×32 inputs, width 64, one transformer layer, batch size one, mixed precision and eight steps per stage. These checks exercised loading, forward/backward passes, optimizer updates, EMA and checkpoint writing. They do not establish full-scale training quality or memory requirements.

## Output checks

- Object CLI and browser PLYs both contained **100,051** finite Gaussians and had the same SHA-256: `619de6a4f611ff9ff1b7640a7cc857eb2d1af024b53fff8b971a6e0da5489514`.
- The extracted object mesh contained **49,529 vertices and 100,000 faces**, with finite coordinates.
- Scene CLI and browser PLYs both contained **189,679** finite Gaussians and matched byte for byte.
- The default scene video decoded as **256×256, 240 frames, 24 FPS, ten seconds**. Only the forward 120 frames are rendered; the return trip reuses them in reverse.
- **25 regression tests passed**, covering trajectories, attention fallback, checkpoint formats, dataset failures, metrics, script dispatch and distributed rank handling.

## Fixes made during validation

- Made the PyTorch CUDA build, toolkit and compiler selection explicit; disabled isolated builds for extensions that import Torch at build time.
- Pinned setuptools and PyArrow for the existing Lightning/datasets versions. Used a single headless OpenCV package to avoid two packages overwriting `cv2`.
- Fixed simple-knn's missing Python package initialization and editable-install metadata.
- Corrected README script paths, `.sh` suffixes, object/scene labels and stage-specific config selection. Launchers now retain the active environment, forward config overrides and match process counts to visible GPUs.
- Made object CLI inference create its output directory, accept input/model/output arguments and guard mesh export when disabled.
- Fixed stage-two loading for Lightning `state_dict`, raw denoiser weights and legacy checkpoint formats.
- Removed the unconditional call to an unimplemented test stage after training.
- Fixed object image paths without a trailing slash, static-camera normalization and recursive dataset retries that hid bad samples.
- Made preprocessing fail on corrupt chunks instead of reporting success after dropping bad images.
- Enabled 512px evaluation result saving and reduced evaluation batches to one scene, allowing varying target-view counts.
- Corrected evaluation output to cover every requested target and exclude the input image. The target render pass explicitly disables autocast for the native float32 rasterizer.
- Removed placeholder zero validation metrics. Actual metrics come from `eval_scene_result.py`.
- Fixed single-image LPIPS output shape and streamed metric inputs instead of stacking the entire evaluation dataset in memory.
- Separated local CUDA device selection from global rank, and preserved global world/rank values in distributed callbacks.
- Removed interactive debugger stops from the scene checkpoint downloader and invalid-batch handling.

## Multi-GPU check

Default NCCL initialization stalled on this shared test host. An independent two-GPU all-reduce passed using socket transport, and the actual two-GPU training script then completed eight steps with:

```sh
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
```

These are host-specific troubleshooting settings, not repository defaults. Apply them only if the normal transport stalls; they can reduce communication performance. Multi-node training was not exercised.

## Limits

The complete RealEstate10K/G-Objaverse datasets, full benchmark and full-scale training were not run. The small-sample metrics are functional checks and do not reproduce the published table. The README now distinguishes the previously reported release scores from the corrected target-only evaluation protocol. Shared GPU load makes the measured runtimes unsuitable for speed comparisons.

Machine-readable results, command logs, frozen package versions, browser screenshots and outputs were retained in the local audit directory alongside the verified Conda environment.

## Second independent installation and expanded checks

A second checkout of GitHub commit `09240b3` was tested in another newly created Python 3.11 Conda environment. The README's PyTorch installation, requirements installation, both native extension builds and `pip check` all passed without changing the installation instructions. Imports were checked against the new environment and checkout, and the newly compiled KNN kernel matched a brute-force reference. Model/download caches were reused; installed packages and native extensions were installed again.

The expanded checks covered:

| Workflow | Second-pass result |
| --- | --- |
| Both default CLI commands | Passed with released checkpoints and 30 diffusion steps. Object inference included the default 256-grid mesh extraction. Scene inference produced the default 256px, 240-frame, ten-second video. |
| Both default Gradio URLs | Browser generation, previews and PLY downloads passed, with no browser exceptions. Object has seven examples and Scene has three. |
| Gradio mesh extraction | A real browser generated an object, extracted its mesh and downloaded the OBJ. The downloaded vertices and faces matched the CLI mesh; no browser exceptions occurred. |
| Every bundled example | All seven Object images passed, including opaque images requiring background removal. All three Scene images passed at both 256px and 512px with 30 steps and ten-second videos. |
| Every trajectory preset | All 17 presets generated finite point clouds and two-second videos at 256px, ten diffusion steps and 8 FPS. Exported camera sequences exactly retraced the forward path. |
| RealEstate10K preprocessing | Both README commands passed with ten genuine reference clips, 16 frames each, packaged into two `.torch` chunks per split. |
| 256px and 512px evaluation | Both passed on ten scenes with different target-view counts, covering 55 targets at each resolution. Saved target indices, image shapes and finite pixels were checked. |
| Metrics | Actual PSNR, SSIM and LPIPS computation passed with chunks of seven images and with single-image batches. These functional checks are not a reproduction of the published benchmark. |
| Evaluation videos | The released 512px checkpoint also passed with `system.save_intermediate_video=true`. |
| All four training scripts | Each completed eight steps with width 128, two layers, 64px inputs, batch size one and two data workers. Default loss calculations, sanity validation, subsequent validation, video output, stage-two checkpoint loading, EMA and nonzero optimizer states were exercised. |
| Distributed training and resume | Two GPUs completed four steps with the host-specific socket transport settings above. A separate run resumed an eight-step checkpoint and reached step ten. |
| Object pipeline inputs | Real checkpoint inference passed with the default omitted seed and an RGBA CHW tensor after the fixes below. |
| Regression tests | All 33 tests passed. |

This pass found and fixed additional issues:

- The object pipeline called `torch.manual_seed(None)` when the optional seed was omitted. Tensor input also referenced an unimported `TF` module. Single-image CHW and BCHW inputs now work, and multi-image tensors are rejected explicitly.
- Empty foreground masks caused ambiguous preprocessing errors, and very thin foregrounds could resize to zero pixels. Foreground ratios are now validated, missing foregrounds produce a clear error, and dimensions remain at least one pixel. Palette transparency and EXIF orientation are preserved.
- Empty object foreground masks produced `0 / 0` in the XYZ training loss. Its denominator is now clamped, with a regression check for finite losses.
- Mesh extraction centered and scaled the Gaussian cloud for density queries but did not undo that transform when exporting vertices. The exported OBJ now uses the cloud's world coordinates. The default result contains 49,529 vertices and 100,000 faces; its vertices matched the previous mesh after the correct world transform to within `1e-8`.
- Default mesh extraction repeatedly scanned the complete cloud for every tile, including empty tiles. It exceeded the second pass's 20-minute check limit on the busy shared GPU host. A CPU spatial index now selects tile candidates while preserving strict bounds and Gaussian accumulation order. A separate CUDA comparison produced identical density values, and the full default object CLI subsequently completed in 304 seconds. These host timings are observations, not an inference-speed benchmark.

The default Object and Scene PLY files still matched their browser downloads byte for byte, including the SHA-256 values recorded above. The expanded object training fixture contains two generated objects with 40 G-Objaverse-format views each; it is not a real Objaverse training sample. Full-scale training, the complete datasets, the complete benchmark and multi-node training remain untested.
