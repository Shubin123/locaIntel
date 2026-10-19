"""Export the pretrained CFV ConvNeXt-Small multitask vehicle-view model.

Optional dependencies: timm>=0.9,<1.1, onnx>=1.14,<1.18,
onnxruntime>=1.19,<1.20, onnxconverter-common>=1.14,<1.17. Checkpoint: Apache-2.0, fort-cyber authors.
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
    # Use the full-resolution model on GPUs. Preserve float32 IO while keeping
    # the internal view logits in FP16 when reused by the angle head.
    import onnx
    from onnxconverter_common import float16
    half = float16.convert_float_to_float16(onnx.load(full), keep_io_types=True)
    cast = next(node for node in half.graph.node if node.op_type == 'Cast' and 'view_logits' in node.output)
    for node in half.graph.node:
        if node is cast:
            continue
        for index, name in enumerate(node.input):
            if name == 'view_logits':
                node.input[index] = cast.input[0]
    onnx.checker.check_model(half, full_check=True)
    gpu_target = output / 'cfv-convnext-facing-fp16.onnx'
    onnx.save(half, gpu_target)
    gpu_bytes = gpu_target.read_bytes()
    manifest['gpu'] = {'file': gpu_target.name, 'bytes': len(gpu_bytes), 'sha256': hashlib.sha256(gpu_bytes).hexdigest(),
                       'precision': 'FP16; float32 input/output', 'requires': 'WebGPU shader-f16'}
    (output / 'vehicle-facing-v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
    license_url = 'https://raw.githubusercontent.com/fort-cyber/car-orientation/master/LICENSE'
    (output / 'CFV-LICENSE').write_bytes(urllib.request.urlopen(license_url, timeout=30).read())
    print(target, manifest['bytes'], manifest['sha256'])


def build_batched_gpu(checkpoint, output=Path('web/models')):
    """Keep the v1 checker intact for older tabs; export a two-view GPU graph."""
    import onnx
    from onnxconverter_common import float16
    torch.set_num_threads(2)
    model=VehicleViewModel();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    full=Path('.cache/cfv-facing-fp32-batch2.onnx')
    torch.onnx.export(model,torch.zeros(2,3,224,224),full,input_names=['image'],output_names=['view_logits','angle_fraction'],opset_version=17,dynamo=False)
    half=float16.convert_float_to_float16(onnx.load(full),keep_io_types=True)
    cast=next(node for node in half.graph.node if node.op_type=='Cast' and 'view_logits' in node.output)
    for node in half.graph.node:
        if node is not cast:
            for index,name in enumerate(node.input):
                if name=='view_logits':node.input[index]=cast.input[0]
    onnx.checker.check_model(half,full_check=True)
    target=output/'cfv-convnext-facing-batch2-fp16.onnx';onnx.save(half,target);data=target.read_bytes()
    manifest_path=output/'vehicle-facing-v2.json'
    manifest=json.loads((manifest_path if manifest_path.exists() else output/'vehicle-facing-v1.json').read_text())
    manifest['version']='cfv-facing-cascade-v2'
    manifest['gpu']={'file':target.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                     'precision':'FP16; float32 input/output','requires':'WebGPU shader-f16','batch':2}
    # Full precision avoids CPU-only dynamic INT8 operators on older GPUs.
    # Keep each external weight shard below GitHub's per-file size limit.
    full_model=onnx.load(full);parts=[];part=None;part_size=0;limit=48*1024*1024
    for tensor in full_model.graph.initializer:
        if not tensor.raw_data:continue
        raw=tensor.raw_data
        if part is None or part_size+len(raw)>limit:
            if part:part.close()
            name=f'cfv-facing-fp32-weights-{len(parts):02d}.bin';parts.append(name);part=(output/name).open('wb');part_size=0
        onnx.external_data_helper.set_external_data(tensor,location=parts[-1],offset=part_size,length=len(raw))
        part.write(raw);part_size+=len(raw);tensor.ClearField('raw_data');tensor.data_location=onnx.TensorProto.EXTERNAL
    if part:part.close()
    fp32_target=output/'cfv-convnext-facing-batch2-fp32.onnx';onnx.save(full_model,fp32_target)
    onnx.checker.check_model(str(fp32_target),full_check=True)
    graph=fp32_target.read_bytes()
    external=[]
    for name in parts:
        chunk=(output/name).read_bytes();external.append({'file':name,'path':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    manifest['gpu32']={'file':fp32_target.name,'bytes':len(graph),'sha256':hashlib.sha256(graph).hexdigest(),
                       'precision':'FP32; WebGPU without shader-f16','batch':2,'externalData':external}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(target,len(data),manifest['gpu']['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batched-gpu-only',action='store_true')
    args=parser.parse_args()
    (build_batched_gpu if args.batched_gpu_only else build)(args.checkpoint)
