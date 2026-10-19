# Browser detector

`yolo11n.onnx` is the pretrained Ultralytics YOLO11 nano detector exported at 640×640, FP32, ONNX opset 17, without embedded NMS. It is not a model trained on DriveBC data.

Rebuild with `python -m pipeline.export_browser_model` after installing `onnx>=1.14,<1.18` and the project's Python dependencies. The worker applies confidence 0.30, class-specific IoU 0.45 NMS, and only publishes people, bicycles, cars, motorcycles, buses and trucks.

Model source and license: https://docs.ultralytics.com/models/yolo11/ — Ultralytics AGPL-3.0 / Enterprise terms apply. This repository uses AGPL-3.0. The runtime in `../vendor/ort/` is ONNX Runtime Web 1.20.1 (Microsoft MIT license included there).

## Single-crop vehicle facing

`cfv-convnext-facing-int8.onnx` is the **pretrained CFV ConvNeXt-Small multitask model**, exported from [fort-cyber/Car-orientation-image](https://huggingface.co/fort-cyber/Car-orientation-image/tree/a8fddadf6af25a6f53ed6a449f74e4106c9c34a2). It predicts eight car views and a camera-relative angle. The authors trained the model on their Car Full View dataset; this application has not trained or calibrated it on DriveBC. Truck, bus, motorcycle and bicycle orientation is unsupported by this checkpoint and abstains.

Citation: Andy Catruna et al., *Car Full View Dataset: Fine-Grained Predictions of Car Orientation from Images*, Electronics 12(24), 4947 (2023), [DOI](https://doi.org/10.3390/electronics12244947). Original code/model terms: Apache-2.0; see `CFV-LICENSE`. The exported ONNX is modified by quantizing dense weights per channel to INT8; convolutions retain FP32. Model source, pinned revision, checkpoint/model SHA-256, preprocessing and angle convention are recorded in `vehicle-facing-v1.json`.

Input: one isolated **car** crop, bilinear-resized to 224×224, RGB CHW /255, no mean/std normalization, matching the authors' [prepare_data.py](https://github.com/fort-cyber/car-orientation/blob/master/scripts/prepare_data.py). Outputs: `view_logits` [1,8] in front, front-left, left-side, rear-left, rear, rear-right, right-side, front-right order; `angle_fraction` [1,1] ×360. These are relative viewpoints, not compass headings or velocity. Crops smaller than 20×16 or 600 pixels, and frame-edge clipped crops, abstain before loading the model.

The browser requires front/rear/side group score ≥0.68 and group margin ≥0.30, angle/class agreement within 67.5°, mirrored group agreement with score ≥0.60 and mirrored angle error ≤45°. Strong class scores (≥0.65) plus mirror agreement allow an approximate image-facing glyph. Scores are uncalibrated. White/yellow and red connected-component pairs provide heuristic lamp evidence; contradictory front/rear lamp evidence causes abstention. A paired color cue alone never establishes a view or movement. A single crop cannot establish whether a car is moving or stationary.

The 57,976,957-byte model is SHA-256 verified and stored in Cache Storage `locaintel-models-v1`, separate from the adjustable photo archive. WASM inference uses one thread, serial crop jobs, power-profile idle delays, visible-feed priority, and pauses queued work when disabled, hidden or reviewing history. Results are stored with the exact camera/frame/crop fingerprint and replayed on the shared timeline.

Export validation: on 16 eligible car crops from six locally cached DriveBC camera images, the quantized model matched the FP32 model's top view class on **16/16** crops. Mean circular angle difference: **1.66°**, maximum **3.83°**. This checks numerical export fidelity, **not classification accuracy**; there are no manually labeled DriveBC test metrics. The browser model and cache were exercised in real Chrome/WASM, including a front-quarter car with a mirror-consistent facing estimate. Paired-light handling is separately tested with controlled pixel fixtures; bright mirror reflections are excluded by restricting cues to the lower body.

Rebuild (optional dependencies, outside the scheduled collection job):

```sh
python -m pip install 'timm>=0.9,<1.1' 'onnx>=1.14,<1.18' 'onnxruntime>=1.19,<1.20'
curl -L --fail -o .cache/cfv-convnext-multi-weak.pth 'https://huggingface.co/fort-cyber/Car-orientation-image/resolve/a8fddadf6af25a6f53ed6a449f74e4106c9c34a2/convnext-multi-weak.pth?download=true'
python -m pipeline.build_orientation_model --checkpoint .cache/cfv-convnext-multi-weak.pth
```

Validated export environment: PyTorch 2.8.0, timm 1.0.30, ONNX 1.17.0, ONNX Runtime 1.19.2. Original FP32 intermediate files and checkpoints remain under ignored `.cache/`; only the browser model and manifest are published.
