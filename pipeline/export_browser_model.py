"""Optional build step for the checked-in browser YOLO model.

pip install 'onnx>=1.14,<1.18'
python -m pipeline.export_browser_model
"""
from pathlib import Path
from ultralytics import YOLO

if __name__ == '__main__':
    target=Path(__file__).resolve().parents[1]/'web/models/yolo11n.onnx'
    exported=YOLO('yolo11n.pt').export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,nms=False)
    target.parent.mkdir(parents=True,exist_ok=True)
    Path(exported).replace(target)
    print(target)
