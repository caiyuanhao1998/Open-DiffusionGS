# RealEstate10K reference images

The demo retains the original bundled room image unchanged and adds one frame from each of two different RealEstate10K scenes. All three images are **256x256 RGB PNGs**.

The two new frames are extracted from the RealEstate10K reference clips provided by [CameraCtrl](https://github.com/hehao13/CameraCtrl/tree/af45a7a0dbeedb7f6ed66aac4a9e5411a279f0d2/assets/reference_videos). The matching pose files identify the original YouTube videos. Each reference clip contributes its zero-based frame 8, center cropped to square. Exact source URLs, repository revision, dimensions, frame indices, available original timestamps, and SHA256 hashes are recorded in [credits.json](credits.json).

| Image | Scene ID | Source video | Reference clip |
| --- | --- | --- | --- |
| `re10k_0bf152ef84195293.png` (Open kitchen) | `0bf152ef84195293` | [YouTube](https://www.youtube.com/watch?v=QShWPZxTDoE) | [CameraCtrl reference](https://raw.githubusercontent.com/hehao13/CameraCtrl/af45a7a0dbeedb7f6ed66aac4a9e5411a279f0d2/assets/reference_videos/0bf152ef84195293.mp4) |
| `re10k_0f47577ab3441480.png` (Home office) | `0f47577ab3441480` | [YouTube](https://www.youtube.com/watch?v=in69BD2eZqg) | [CameraCtrl reference](https://raw.githubusercontent.com/hehao13/CameraCtrl/af45a7a0dbeedb7f6ed66aac4a9e5411a279f0d2/assets/reference_videos/0f47577ab3441480.mp4) |

RealEstate10K dataset metadata: Google LLC, [CC BY 4.0](https://google.github.io/realestate10k/download.html). Original video sources are credited above.
