# Browser detector

`yolo11n.onnx` is the pretrained Ultralytics YOLO11 nano detector exported at 640×640, FP32, ONNX opset 17, without embedded NMS. It is not a model trained on DriveBC data.

Rebuild with `python -m pipeline.export_browser_model` after installing `onnx>=1.14,<1.18` and the project's Python dependencies. The worker applies confidence 0.30, class-specific IoU 0.45 NMS, and only publishes people, bicycles, cars, motorcycles, buses and trucks.

Model source and license: https://docs.ultralytics.com/models/yolo11/ — Ultralytics AGPL-3.0 / Enterprise terms apply. This repository uses AGPL-3.0. The runtime in `../vendor/ort/` is ONNX Runtime Web 1.20.1 (Microsoft MIT license included there).

## Single-crop vehicle facing

`drivebc-facing-v3.onnx` (MobileNetV3-Large, FP32, batch 2 = crop + mirror, outputs 8 CFV view logits + an angle vector) and `vehicle-facing-v3.json` (checksum and confidence-tier thresholds). It starts from the CFV-trained student (`pipeline/train_facing.py`, [CFV dataset](https://huggingface.co/datasets/fort-cyber/CFV-Dataset), Apache-2.0, see `CFV-LICENSE`; [timm MobileNetV3](https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k), see `TIMM-LICENSE`) and is adapted to DriveBC with hand labels and DINOv2-distilled pseudo-labels (`pipeline/train_drivebc_facing.py`). Cross-validation, tier accuracy and the curve are in `facing-training-v3.json`; the method and numbers are in the main README under *Vehicle facing on every car*.

The browser averages the crop and mirrored-crop probabilities, picks front / rear / side, and maps CFV views to a travel direction relative to the camera's look direction. CFV's left-side view shows the vehicle's left flank, so its nose points image-right (checked on DriveBC crops). Earlier versions used a CFV ConvNeXt full-resolution checker. It was removed because it was less accurate on DriveBC views (53%) and cost ~520 MB of model files.

## GPU execution and fast road context

Workers first request a real WebGPU adapter with a high-performance preference, then retry the default adapter and use the vendored ONNX Runtime Web **1.20.1 WebGPU/JSEP** build. YOLO11n and SegFormer-B0 use the same FP32 graph on GPU and CPU. The facing model uses FP32 on every adapter. Facing and road variants are checksum-verified; all models are cached locally. Adapter absence, unsupported graph creation, or GPU device loss falls back to WASM; widgets report the actual selected provider. Some control operators may still execute on CPU. See the [official WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) and the [1.20.1 operator matrix](https://github.com/microsoft/onnxruntime/blob/v1.20.1/js/web/docs/webgpu-operators.md).


Earlier full-resolution-checker baseline, warm real Chrome on the available Apple GPU (one source image, after initialization): YOLO **21 ms**, road model **15 ms**, facing (old CFV checker, both passes) **51 ms**. A forced-WASM run on the same machine measured **256 / 95 / 868 ms**, respectively. These are single-machine model/job timings, exclude first downloads and initialization, and are not a device-independent speed guarantee. The app also reuses exact crop-pixel hashes across changed source-image hashes, updates lamp coordinates for the new box, prioritizes road-supported cars, and removes inference idle gaps from cheap abstentions. YOLO GPU jobs use 15% of the configured idle delay; facing/road GPU jobs use 3%, so short jobs are not dominated by idle time. CPU jobs retain the full profile delay.

`segformer-b0-road-256.onnx` is a pinned **NVIDIA SegFormer-B0 Cityscapes** checkpoint exported at 256×256 with ImageNet channel normalization. It returns road probabilities, pooled to a 32×24 display grid. Every fresh analyzed photo gets its own road result; a matching published road mask is reused immediately. The model requires ~14.3 MiB and uses the NVIDIA research/evaluation-only terms in `SegFormer-LICENSE` rather than the repository's AGPL license. [Model source](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/tree/21b3847fae21ddee674abd31129307b6a1235bd9); [paper](https://arxiv.org/abs/2105.15203).

A footprint patch below each box supplies road support. A thinned road mask supplies local axes; when at least three of the nearest 24 road-supported vehicle anchors form a consistent corridor, that corridor can refine the axis around broad roads/intersections. Mask continuity, local span, anisotropy and facing alignment gates reject unsupported directions. Results are approximate **image-space road-aligned facing estimates**. The mask does not establish lane markings, legal travel direction, geographic bearings, identity, speed, or actual movement. No movement state is changed by these estimates.

Rebuild the road graph with `python -m pipeline.export_road_model` (same optional ONNX/PyTorch dependencies as the facing exporter, plus the existing transformers dependency). Metadata, source revision and checksum are in `road-v1.json`. Road masks, skeletons and facing results stay attached to their exact camera/frame in IndexedDB and the shared timeline; cached geometry retains new temporal motion evidence.

## WebGL fallback (browsers without WebGPU)

Workers now try **WebGPU → WebGL → WASM**. The WebGL tier uses ONNX Runtime Web 1.20.1's separate WebGL build (`../vendor/ort/ort.webgl.min.js`, MIT), loaded lazily under its own global so the WASM runtime stays available. It needs WebGL2 via `OffscreenCanvas` in workers; graph creation or run errors (including lost contexts) fall back to the CPU model, and the pane shows the backend used.

ORT's WebGL backend supports fewer operators and has several kernel bugs, so `pipeline/export_webgl_models.py` builds dedicated graphs (`webgl-v1.json`, checksummed) at opset 12 with static shapes folded by onnx-simplifier, plus exact rewrites: 2× nearest `Resize` → depthwise `ConvTranspose` of ones; identity `Resize` removed; integer `Pow` → `Mul` (GLSL `pow` is NaN for negative bases); million-element `Relu` → `0.5·(x+|x|)`; SegFormer's wide 1×1 conv over a channel concat → per-branch convs summed; trailing `Slice` → one-hot conv. Exact GELU becomes the tanh approximation written as `x·σ(2z)` (WebGL `Tanh` saturates).

| Model | WebGL graph vs original (ORT CPU) | Real Chrome, no WebGPU: WASM → WebGL (warm, per frame) |
| --- | --- | --- |
| YOLO11n (packed textures) | identical outputs; 48/48 boxes on 4 DriveBC frames | ~148 ms → ~93 ms |
| SegFormer-B0 road (unpacked; packed leaks state between runs) | max prob. diff 0.008; 100% agreement on the 0.6 road mask, 4 frames | ~85 ms → ~35 ms |

First WebGL use compiles shaders (≈0.4–0.6 s once per worker). The facing model stays on WASM when WebGPU is missing; at ~16 MiB it costs a few tens of milliseconds per crop on CPU. Timings are single-machine measurements.

Rebuild: `python -m pip install 'onnxsim>=0.4,<0.5'` then `python -m pipeline.export_webgl_models`.

### Multi-car batch graphs

`cfv-{fast,gpu,gpu32}-cars{2,4,8}.onnx` specialize the shipped facing graphs for
4, 8, or 16 input views (an original/mirror pair for each car). They preserve all
weights, operators, preprocessing and output semantics; only batch shapes change.
`vehicle-facing-v2.json` records their checksums and shared external data.
`fast` and `gpu` graphs address weight byte ranges inside the original compact
and FP16 ONNX files, respectively. `gpu32` reuses the existing FP32 weight shards.
Keep those original files byte-identical when publishing these derived graphs;
regenerate them with `python -m pipeline.export_gpu_batches` if weights change.
All nine derived graphs add about 1.7 MiB without duplicating the trained weights.
