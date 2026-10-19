# Browser detector

`yolo11n.onnx` is the pretrained Ultralytics YOLO11 nano detector exported at 640×640, FP32, ONNX opset 17, without embedded NMS. It is not a model trained on DriveBC data.

Rebuild with `python -m pipeline.export_browser_model` after installing `onnx>=1.14,<1.18` and the project's Python dependencies. The worker applies confidence 0.30, class-specific IoU 0.45 NMS, and only publishes people, bicycles, cars, motorcycles, buses and trucks.

Model source and license: https://docs.ultralytics.com/models/yolo11/ — Ultralytics AGPL-3.0 / Enterprise terms apply. This repository uses AGPL-3.0. The runtime in `../vendor/ort/` is ONNX Runtime Web 1.20.1 (Microsoft MIT license included there).

## Single-crop vehicle facing

The current browser path uses a **newly trained MobileNetV3-Large** first, then the original CFV ConvNeXt-Small for difficult crops. The compact model was fine-tuned on labeled, isolated CFV car crops with brightness/contrast, correctly mirrored angles and 24–112px degradation. It predicts eight views and a circular sine/cosine angle head. It checks the original crop and its mirror in one static batch of two; no verification pass was removed.

The fast path requires an independently selected validation threshold, confident matching mirrored classes, class/angle agreement within 45°, mirror error at most 25°, and the existing lamp-conflict safeguards. Other crops get the original full-resolution model, loaded on demand. Its new FP16 and FP32 GPU graphs batch the two views and replays a captured command graph with persistent GPU input/output buffers. If capture fails, ordinary GPU execution is tried before CPU fallback. CPU and last-resort mixed-precision checker paths retain conditional single-view then mirror execution; GPU loss maps the batch back to the CPU model’s compatible input shape. The threshold and temperature are fitted only on held-out training identities. Official test identities are excluded from training and checkpoint/policy selection. Exact training curves, split identities, checkpoint hash and independent test comparisons are in [facing-training-v2.json](facing-training-v2.json). These are CFV measurements; **DriveBC deployment accuracy remains unmeasured**.

The compact FP32 graph is about 16 MiB and works on WebGPU without `shader-f16`. Both GPU and CPU use identical compact weights. The larger checker uses ~96 MiB FP16 weights on suitable GPUs, or ~192 MiB FP32 weights in five checksum-verified external shards on GPUs without FP16. Shards are at most 48 MiB so they can be served directly by GitHub Pages. CPU fallback retains the ~55 MiB checker. Only the chosen variant is downloaded; each is cached for reuse. Backend, fast/full-resolution path and CPU-fallback reason are visible in the pane and crop inspector and stored with each result. The current manifest is `vehicle-facing-v2.json`; v1 assets remain available for older tabs. Current results use `cfv-facing-cascade-v2`; historical v1 results remain readable at their original timeline cursor. Both model tiers are cached separately from the photo budget.

Sources: [CFV labeled dataset](https://huggingface.co/datasets/fort-cyber/CFV-Dataset/tree/e17aaeb4860b33757c582a57e5a05e70ec40b10e) (Apache-2.0, `CFV-LICENSE`); [ImageNet-pretrained MobileNetV3 backbone](https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k/tree/96f46a1c52932f27492dff66c72378eb99b443a7) and [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0, `TIMM-LICENSE`). The new model uses supervised CFV labels, not guessed DriveBC labels. Other vehicle types remain unsupported.

Reproduce the optional training job locally (not part of routine camera collection):

```sh
mkdir -p .cache/cfv-data
curl -L --fail -o .cache/cfv-data/train.zip https://huggingface.co/datasets/fort-cyber/CFV-Dataset/resolve/e17aaeb4860b33757c582a57e5a05e70ec40b10e/data/train.zip
curl -L --fail -o .cache/cfv-data/test.zip https://huggingface.co/datasets/fort-cyber/CFV-Dataset/resolve/e17aaeb4860b33757c582a57e5a05e70ec40b10e/data/test.zip
python -m pipeline.train_facing --epochs 12 --batch 96 --lr .0008 --seed 731
python -m pipeline.evaluate_facing --checkpoint .cache/cfv-convnext-multi-weak.pth
python -m pipeline.build_orientation_model --checkpoint .cache/cfv-convnext-multi-weak.pth --batched-gpu-only
python -m pipeline.export_fast_facing
```

Use the optional PyTorch/timm/ONNX dependencies listed below, including torchvision. Export refuses a fast policy that fails the independent eight-view accuracy gate. Training uses CUDA first, then Apple MPS, then CPU. Validation temperatures are not a promise of calibrated confidence under webcam domain shift.

### Accuracy and latency measurements

The shipped cascade was evaluated on **4,960 crops of 14 official test car identities**, excluded from training and policy/checkpoint selection. Each crop was tested at 160px and after downsampling to 40px; both models receive the same crop through a 224px interface. Training used 13,719 valid crops; validation used 1,715 crops from ten separate training identities, at both resolutions. Training ran for 12 epochs on Apple MPS, selecting epoch 10 on validation accuracy. The fixed fast threshold is 0.95 after validation-only temperature scaling (0.92338), with 25° mirror tolerance.

| Test input | Original checker accuracy | Shipped cascade accuracy | Original / cascade mean angle error | Fast route |
| --- | ---: | ---: | ---: | ---: |
| 160px crops | 90.24% | 91.19% | 5.70° / 5.45° | 2,482 / 4,960 |
| 40px crops | 85.26% | 88.57% | 15.52° / 11.05° | 2,366 / 4,960 |

Accuracy is eight-view top-1, before the final display abstentions/lamp checks. The standalone compact model scored 92.76% / 92.78%; the shipped cascade deliberately requires stronger confidence before using it. Test fast-route accuracy was 98.19% / 97.46%. These are correlated views of fourteen cars, not 4,960 independent vehicles or a DriveBC benchmark.

Real Chrome on the available Apple GPU: the compact model takes about **4–5 ms** per original+mirror pair after warming. On a difficult DriveBC crop, the full shipped route (including the fast screening call) had **45.1 ms median**, versus **47.5 ms** previously, across 20 warm jobs each. GPU idle pacing was reduced from 15% to 3% of the slider delay (Balanced: 52.5 → 10.5 ms), improving queue throughput independently of model latency. CPU compact pairs took about 27 ms. These exclude first downloads/initialization and are device-specific. **None of the 16 sampled DriveBC crops qualified for the strict fast route**; do not interpret the compact-model timing as all-camera latency. No confidence gate was lowered to force fast-path coverage. On the same Apple GPU with the application’s FP16 choice disabled, the new compatible FP32 GPU route measured about **70 ms**, versus **840 ms** for the previous mixed-precision GPU route, including fast screening. This exercises the non-FP16 software path on real GPU hardware, not a benchmark of a different physical GPU.

The compact FP32 export matched the PyTorch reference on **32/32 original/mirrored view classes** across those 16 crops on both real WebGPU and WASM (maximum logit error below 0.00002). The batched FP16 checker retained 16/16 original view classes, with 0.25° mean / 0.77° maximum circular difference from FP32, matching the prior export’s fidelity. Chrome also exercises real command capture, GPU device loss recovery, model cache reload, road alignment, history and exact-frame boxes.

### Full-resolution checker

`cfv-convnext-facing-int8.onnx` is the **pretrained CFV ConvNeXt-Small multitask model**, exported from [fort-cyber/Car-orientation-image](https://huggingface.co/fort-cyber/Car-orientation-image/tree/a8fddadf6af25a6f53ed6a449f74e4106c9c34a2). It predicts eight car views and a camera-relative angle. The authors trained the model on their Car Full View dataset; this application has not trained or calibrated it on DriveBC. Truck, bus, motorcycle and bicycle orientation is unsupported by this checkpoint and abstains.

Citation: Andy Catruna et al., *Car Full View Dataset: Fine-Grained Predictions of Car Orientation from Images*, Electronics 12(24), 4947 (2023), [DOI](https://doi.org/10.3390/electronics12244947). Original code/model terms: Apache-2.0; see `CFV-LICENSE`. The exported ONNX is modified by quantizing dense weights per channel to INT8; convolutions retain FP32. Model source, pinned revision, checkpoint/model SHA-256, preprocessing and angle convention are recorded in `vehicle-facing-v1.json`.

Input: one isolated **car** crop, bilinear-resized to 224×224, RGB CHW /255, no mean/std normalization, matching the authors' [prepare_data.py](https://github.com/fort-cyber/car-orientation/blob/master/scripts/prepare_data.py). Outputs: `view_logits` [1,8] in front, front-left, left-side, rear-left, rear, rear-right, right-side, front-right order; `angle_fraction` [1,1] ×360. These are relative viewpoints, not compass headings or velocity. Crops smaller than 20×16 or 600 pixels, and frame-edge clipped crops, abstain before loading the model.

The browser requires front/rear/side group score ≥0.68 and group margin ≥0.30, angle/class agreement within 67.5°, mirrored group agreement with score ≥0.60 and mirrored angle error ≤45°. Strong class scores (≥0.65) plus mirror agreement allow an approximate image-facing glyph. Scores are uncalibrated. White/yellow and red connected-component pairs provide heuristic lamp evidence; contradictory front/rear lamp evidence causes abstention. A paired color cue alone never establishes a view or movement. A single crop cannot establish whether a car is moving or stationary.

The 57,976,957-byte CPU model (GPU variant described below) is SHA-256 verified and stored in Cache Storage `locaintel-models-v1`, separate from the adjustable photo archive. WASM inference uses one thread, serial crop jobs, power-profile idle delays, visible-feed priority, and pauses queued work when disabled, hidden or reviewing history. Results are stored with the exact camera/frame/crop fingerprint and replayed on the shared timeline.

Export validation: on 16 eligible car crops from locally cached DriveBC camera images, the quantized model matched the FP32 model's top view class on **16/16** crops. Mean circular angle difference: **1.66°**, maximum **3.83°**. This checks numerical export fidelity, **not classification accuracy**; there are no manually labeled DriveBC test metrics. The browser model and cache were exercised in real Chrome/WASM, including a front-quarter car with a mirror-consistent facing estimate. Paired-light handling is separately tested with controlled pixel fixtures; bright mirror reflections are excluded by restricting cues to the lower body.

Rebuild (optional dependencies, outside the scheduled collection job):

```sh
python -m pip install 'timm>=0.9,<1.1' 'onnx>=1.14,<1.18' 'onnxruntime>=1.19,<1.20' 'onnxconverter-common>=1.14,<1.17'
curl -L --fail -o .cache/cfv-convnext-multi-weak.pth 'https://huggingface.co/fort-cyber/Car-orientation-image/resolve/a8fddadf6af25a6f53ed6a449f74e4106c9c34a2/convnext-multi-weak.pth?download=true'
python -m pipeline.build_orientation_model --checkpoint .cache/cfv-convnext-multi-weak.pth
```

Validated export environment: PyTorch 2.8.0, timm 1.0.30, ONNX 1.17.0, ONNX Runtime 1.19.2. Original FP32 intermediate files and checkpoints remain under ignored `.cache/`; only the browser model and manifest are published.


## GPU execution and fast road context

Workers first request a real WebGPU adapter with a high-performance preference, then retry the default adapter and use the vendored ONNX Runtime Web **1.20.1 WebGPU/JSEP** build. YOLO11n and SegFormer-B0 use the same FP32 graph on GPU and CPU. The new compact model uses FP32 on every adapter. The full-resolution checker tries `cfv-convnext-facing-batch2-fp16.onnx` on adapters supporting `shader-f16`, then a full FP32 graph with external weight shards if needed. Its mixed INT8-dense/FP32-convolution graph is a final GPU compatibility attempt before WASM fallback. Lack of FP16 support alone no longer forces CPU execution. Facing and road variants are checksum-verified; all models are cached locally. Adapter absence, unsupported graph creation, or GPU device loss falls back to WASM; widgets report the actual selected provider. Some control operators may still execute on CPU. See the [official WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) and the [1.20.1 operator matrix](https://github.com/microsoft/onnxruntime/blob/v1.20.1/js/web/docs/webgpu-operators.md).

The full-resolution checker inputs stay at **224×224** and both verification passes remain enabled. The new FP16 export matched the FP32 view class on 16/16 eligible cached car crops; mean circular angle difference on actual WebGPU inference was **0.25°**, maximum **0.77°** (export fidelity, not labeled traffic-camera accuracy). The FP16 graph is 100,738,451 bytes (~96 MiB) and is only requested for suitable GPU adapters; CPU-only browsers keep using the ~55 MiB graph. The export script fixes the FP16 view-logit reuse between the class and angle heads while retaining float32 public input/output types.

Earlier full-resolution-checker baseline, warm real Chrome on the available Apple GPU (one source image, after initialization): YOLO **21 ms**, road model **15 ms**, CFV facing including both passes **51 ms**. A forced-WASM run on the same machine measured **256 / 95 / 868 ms**, respectively. These are single-machine model/job timings, exclude first downloads and initialization, and are not a device-independent speed guarantee. The app also reuses exact crop-pixel hashes across changed source-image hashes, updates lamp coordinates for the new box, prioritizes road-supported cars, and removes inference idle gaps from cheap abstentions. YOLO GPU jobs use 15% of the configured idle delay; facing/road GPU jobs use 3%, so short jobs are not dominated by idle time. CPU jobs retain the full profile delay.

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

First WebGL use compiles shaders (≈0.4–0.6 s once per worker). The car-facing models stay on WASM when WebGPU is missing: the compact model was slower on WebGL (two batch-1 calls), and the ~200 MB FP32 ConvNeXt checker crashed the tab under WebGL. With detection and road on the GPU, the facing worker no longer waits for them, but a difficult crop still costs roughly 0.85 s of CPU time on the tested Mac. Timings are single-machine measurements.

Rebuild: `python -m pip install 'onnxsim>=0.4,<0.5'` then `python -m pipeline.export_webgl_models`.
