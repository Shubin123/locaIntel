"""Export WebGL-compatible graphs for browsers without WebGPU.

ONNX Runtime Web's WebGL backend supports a narrower operator set than its
WebGPU/WASM backends: no opset-13+ Split, no fused LayerNormalization, no Erf
and no int64 tensors at runtime; its Tanh kernel saturates and GLSL pow() is NaN
for negative bases. These exports use opset 12, fold constant
shape arithmetic, and swap exact GELU for the tanh approximation.

Optional dependencies: onnxsim>=0.4,<0.5 plus the existing export dependencies.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import onnx
import torch
import torch.nn.functional as F

MODELS = Path('web/models')
_exact_gelu = F.gelu


def tanh_gelu(x, approximate='none'):
    # 0.5x(1 + tanh z) == x * sigmoid(2z); ORT Web's WebGL Tanh kernel saturates.
    return x * torch.sigmoid(2 * math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3)))


class TanhGELU(torch.nn.Module):
    def forward(self, x):
        return tanh_gelu(x)


def swap_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, TanhGELU())
        else:
            swap_gelu(child)
    # transformers' GELUActivation caches F.gelu as an attribute at init.
    if getattr(module, 'act', None) is _exact_gelu:
        module.act = tanh_gelu


def webgl_compatible(target):
    """Rewrite nodes the WebGL backend lacks into exactly equivalent ones."""
    from onnx import helper, numpy_helper, shape_inference
    import numpy as np
    from onnxsim import simplify
    # Inputs are static, so all Shape/Gather/Concat arithmetic folds to constants.
    model, ok = simplify(onnx.load(target))
    if not ok:
        raise RuntimeError('onnxsim could not validate ' + str(target))
    model = shape_inference.infer_shapes(model)
    graph_outputs = {o.name for o in model.graph.output}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*model.graph.value_info, *model.graph.input]}
    nodes, extra, alias = [], [], {}
    producers = {o: n for n in model.graph.node for o in n.output}
    weights = {i.name: i for i in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        node.input[:] = [alias.get(name, name) for name in node.input]
        tag = f'webgl_{index}'
        if node.op_type == 'Constant' and node.attribute[0].name == 'value':
            # WebGL kernels read Slice/Reshape parameters only from initializers.
            tensor = helper.get_attribute_value(node.attribute[0])
            tensor.name = node.output[0]
            extra.append(tensor)
        elif node.op_type == 'Relu' and int(np.prod(shapes.get(node.input[0]) or [0])) >= 1 << 20:
            # ORT Web's WebGL Relu returns zeros for million-element tensors.
            extra.append(numpy_helper.from_array(np.array(.5, np.float32), tag + '_half'))
            source = alias.get(node.input[0], node.input[0])
            nodes += [helper.make_node('Abs', [source], [tag + '_abs']),
                      helper.make_node('Add', [source, tag + '_abs'], [tag + '_sum']),
                      helper.make_node('Mul', [tag + '_sum', tag + '_half'], list(node.output))]
        elif (node.op_type == 'Pow' and node.input[1] in weights
              and float(numpy_helper.to_array(weights[node.input[1]])) in (2.0, 3.0)):
            # GLSL pow() is undefined for negative bases, e.g. LayerNorm's (x - mean)^2.
            source = alias.get(node.input[0], node.input[0])
            if float(numpy_helper.to_array(weights[node.input[1]])) == 2.0:
                nodes.append(helper.make_node('Mul', [source, source], list(node.output)))
            else:
                nodes += [helper.make_node('Mul', [source, source], [tag + '_sq']),
                          helper.make_node('Mul', [tag + '_sq', source], list(node.output))]
        elif (node.op_type == 'Conv' and producers.get(node.input[0]) is not None
              and producers[node.input[0]].op_type == 'Concat' and node.input[1] in weights
              and numpy_helper.to_array(weights[node.input[1]]).shape[1] > 512
              and all(helper.get_attribute_value(a) == 1 for a in node.attribute if a.name == 'group')):
            # A wide 1x1 conv over a channel concat returns zeros on WebGL; it
            # equals the sum of per-branch convs with sliced weights.
            concat = producers[node.input[0]]
            kernel = numpy_helper.to_array(weights[node.input[1]])
            if any(a.name == 'axis' and a.i != 1 for a in concat.attribute) or kernel.shape[2:] != (1, 1):
                raise ValueError('Unsupported wide conv: ' + node.name)
            offset, parts = 0, []
            for part, branch in enumerate(concat.input):
                width = shapes[branch][1]
                extra.append(numpy_helper.from_array(kernel[:, offset:offset + width].copy(), f'{tag}_w{part}'))
                inputs = [alias.get(branch, branch), f'{tag}_w{part}'] + (list(node.input[2:3]) if part == 0 else [])
                nodes.append(helper.make_node('Conv', inputs, [f'{tag}_c{part}'], kernel_shape=[1, 1]))
                parts.append(f'{tag}_c{part}')
                offset += width
            total = parts[0]
            for part, name in enumerate(parts[1:], 1):
                out = node.output[0] if part == len(parts) - 1 else f'{tag}_s{part}'
                nodes.append(helper.make_node('Add', [total, name], [out]))
                total = out
        elif node.op_type == 'Resize' and shapes.get(node.input[0]) == shapes.get(node.output[0]) and node.output[0] not in graph_outputs:
            # ORT Web's WebGL identity-resize shader samples an undeclared texture.
            alias[node.output[0]] = node.input[0]
        elif node.op_type == 'Resize' and any(a.name == 'mode' and a.s == b'nearest' for a in node.attribute):
            source, target_shape = shapes[node.input[0]], shapes[node.output[0]]
            if source[2] * 2 != target_shape[2] or source[3] * 2 != target_shape[3]:
                raise ValueError('Only 2x nearest upsampling is rewritten: ' + node.name)
            channels = source[1]
            extra.append(numpy_helper.from_array(np.ones((channels, 1, 2, 2), np.float32), tag + '_w'))
            # A depthwise 2x2 stride-2 transposed conv of ones copies each pixel into a 2x2 block.
            nodes.append(helper.make_node('ConvTranspose', [node.input[0], tag + '_w'], list(node.output),
                                          group=channels, kernel_shape=[2, 2], strides=[2, 2]))
        else:
            nodes.append(node)
    # Drop nodes (e.g. a split conv's concat) whose outputs nobody reads.
    while True:
        used = {i for n in nodes for i in n.input} | graph_outputs
        kept = [n for n in nodes if any(o in used for o in n.output)]
        if len(kept) == len(nodes):
            break
        nodes = kept
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(extra)
    del model.graph.value_info[:]
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, target)


def describe(target):
    data = target.read_bytes()
    return {'file': target.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'opset': 12}


def export(model, shape, target, inputs, outputs):
    swap_gelu(model)
    F.gelu = tanh_gelu
    try:
        torch.onnx.export(model.eval(), torch.zeros(*shape), target, input_names=inputs, output_names=outputs,
                          opset_version=12, do_constant_folding=True, dynamo=False)
    finally:
        F.gelu = _exact_gelu
    webgl_compatible(target)
    return describe(target)


def yolo():
    from ultralytics import YOLO
    exported = YOLO('yolo11n.pt').export(format='onnx', imgsz=640, opset=12, simplify=False, dynamic=False, nms=False)
    target = MODELS / 'yolo11n-webgl.onnx'
    shutil.move(exported, target)
    webgl_compatible(target)
    return describe(target)


class WebGLRoad(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        select = torch.zeros(1, model.config.num_labels, 1, 1)
        select[0, 0] = 1
        self.register_buffer('select', select)

    def forward(self, pixels):
        # Same road-class probability; a one-hot 1x1 conv avoids a trailing Slice view.
        probabilities = self.model(pixel_values=pixels).logits.softmax(dim=1)
        return F.conv2d(probabilities, self.select)


def road(local):
    from transformers import SegformerForSemanticSegmentation
    from .export_road_model import SOURCE, REVISION
    model = SegformerForSemanticSegmentation.from_pretrained(local or SOURCE, **({} if local else {'revision': REVISION}))
    return export(WebGLRoad(model), (1, 3, 256, 256), MODELS / 'segformer-b0-road-256-webgl.onnx', ['pixels'], ['road_probability'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--road-local', default='.cache/segformer-browser')
    parser.add_argument('--only', choices=['yolo', 'road'], action='append')
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest_path = MODELS / 'webgl-v1.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = {'yolo': yolo, 'road': lambda: road(args.road_local)}
    # Verified in Chrome: YOLO is exact only with packed textures; SegFormer's
    # packed kernels leak state between runs.
    settings = {'yolo': {'pack': True}, 'road': {'pack': False}}
    for name in args.only or jobs:
        manifest[name] = {**jobs[name](), **settings[name]}
        print(name, manifest[name])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
