"""Export the pretrained CFV ConvNeXt-Small multitask vehicle-view model.

Optional dependencies: timm>=0.9,<1.1, onnx>=1.14,<1.18,
onnxruntime>=1.19,<1.20. Checkpoint: Apache-2.0, fort-cyber authors.
Architecture follows https://github.com/fort-cyber/car-orientation
This exports existing trained weights; it does not train on DriveBC images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import torch
import timm
from onnxruntime.quantization import quantize_dynamic, QuantType

REVISION = 'a8fddadf6af25a6f53ed6a449f74e4106c9c34a2'
SOURCE = f'https://huggingface.co/fort-cyber/Car-orientation-image/tree/{REVISION}'
VIEWS = ['front', 'front_left', 'left_side', 'rear_left', 'rear', 'rear_right', 'right_side', 'front_right']


class VehicleViewModel(torch.nn.Module):
    """CFV multitask model, ported from the authors' Apache-2.0 implementation."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=False)
        self.backbone.head.fc = torch.nn.Linear(768, 1024)
        self.class_prediction = torch.nn.Linear(1024, 8)
        self.angle_prediction = torch.nn.Linear(1032, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, image):
        x = self.backbone(image)
        logits = self.class_prediction(x)
        angle = self.sigmoid(self.angle_prediction(torch.cat((x, logits), dim=1)))
        return logits, angle


def build(checkpoint, output=Path('web/models')):
    torch.set_num_threads(2)
    model = VehicleViewModel()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    full = Path('.cache/cfv-facing-fp32.onnx')
    full.parent.mkdir(exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), full, input_names=['image'], output_names=['view_logits', 'angle_fraction'], opset_version=17, do_constant_folding=True, dynamo=False)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'cfv-convnext-facing-int8.onnx'
    # Keep convolutions in FP32; dynamic per-channel INT8 dense weights reduce
    # download size. Validate predictions against FP32 before publishing.
    quantize_dynamic(str(full), str(target), op_types_to_quantize=['MatMul', 'Gemm'], per_channel=True, weight_type=QuantType.QInt8)
    manifest = {'version': 'cfv-convnext-facing-v1', 'model': 'CFV ConvNeXt-Small (INT8 dense / FP32 convolution)', 'source': SOURCE, 'revision': REVISION,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size,
                'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                'input': {'size': 224, 'rescale': 1 / 255, 'normalize': False, 'crop': 'isolated vehicle resized to square, as in CFV prepare_data.py'},
                'views': VIEWS, 'angle_convention': 'CFV relative view: front=0, left-side=90, rear=180, right-side=270. Not a compass heading.',
                'citation': 'Catruna et al. Car Full View Dataset: Fine-Grained Predictions of Car Orientation from Images. Electronics 12(24), 4947 (2023). https://doi.org/10.3390/electronics12244947',
                'limitations': 'Trained on car photographs, not DriveBC. Car crops only. Scores are uncalibrated. Facing is not measured motion.'}
    (output / 'vehicle-facing-v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
    license_url = 'https://raw.githubusercontent.com/fort-cyber/car-orientation/master/LICENSE'
    (output / 'CFV-LICENSE').write_bytes(urllib.request.urlopen(license_url, timeout=30).read())
    print(target, manifest['bytes'], manifest['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    build(parser.parse_args().checkpoint)
