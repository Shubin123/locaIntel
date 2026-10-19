# Browser detector

`yolo11n.onnx` is the pretrained Ultralytics YOLO11 nano detector exported at 640×640, FP32, ONNX opset 17, without embedded NMS. It is not a model trained on DriveBC data.

Rebuild with `python -m pipeline.export_browser_model` after installing `onnx>=1.14,<1.18` and the project's Python dependencies. The worker applies confidence 0.30, class-specific IoU 0.45 NMS, and only publishes people, bicycles, cars, motorcycles, buses and trucks.

Model source and license: https://docs.ultralytics.com/models/yolo11/ — Ultralytics AGPL-3.0 / Enterprise terms apply. This repository uses AGPL-3.0. The runtime in `../vendor/ort/` is ONNX Runtime Web 1.20.1 (Microsoft MIT license included there).
