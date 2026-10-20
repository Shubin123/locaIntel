"""Specialize shipped facing graphs for real multi-car WebGPU batches.

Run: python -m pipeline.export_gpu_batches (requires onnx).
Weights are shared, including embedded weights addressed as external-data slices
of the original ONNX file. No checkpoint, retraining or duplicate weights needed.
"""
import copy
import hashlib
import json
from pathlib import Path

import onnx


def fields(data, start=0, end=None):
    """Yield protobuf field number, wire type, payload start and payload end."""
    end = len(data) if end is None else end
    def varint(pos):
        value = shift = 0
        while True:
            byte = data[pos]; pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
            shift += 7
    while start < end:
        tag, start = varint(start)
        wire = tag & 7
        if wire == 0:
            _, stop = varint(start)
        elif wire in (1, 5):
            stop = start + (8 if wire == 1 else 4)
        elif wire == 2:
            length, start = varint(start)
            stop = start + length
        else:
            raise ValueError(f'Unsupported protobuf wire type {wire}')
        yield tag >> 3, wire, start, stop
        start = stop


def shared_model(path):
    model = onnx.load(path, load_external_data=False)
    if any(t.external_data for t in model.graph.initializer):
        return model, []  # Existing FP32 shards already share their weights.
    data = path.read_bytes()
    offsets = {}
    graph = next((a, b) for n, w, a, b in fields(data) if n == 7 and w == 2)
    for number, wire, start, end in fields(data, *graph):
        if number != 5 or wire != 2:
            continue
        tensor = onnx.TensorProto.FromString(data[start:end])
        for n, w, a, b in fields(data, start, end):
            if n == 9 and w == 2 and b - a > 64:
                offsets[tensor.name] = (a, b - a)
    for tensor in model.graph.initializer:
        if tensor.name in offsets:
            offset, length = offsets[tensor.name]
            onnx.external_data_helper.set_external_data(tensor, location=path.name, offset=offset, length=length)
            tensor.ClearField('raw_data')
            tensor.data_location = onnx.TensorProto.EXTERNAL
    return model, [{'file': path.name, 'path': path.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]


def build(output=Path('web/models')):
    manifest_path = output / 'vehicle-facing-v2.json'
    manifest = json.loads(manifest_path.read_text())
    batches = {}
    for key in ('fast', 'gpu', 'gpu32'):
        spec = manifest[key]
        model, external = shared_model(output / spec['file'])
        for cars in (2, 4, 8):
            graph = copy.deepcopy(model)
            # These graphs contain no batch-dependent reshape constants. The
            # compact graph obtains Resize's batch dimension from Shape(image).
            for value in [*graph.graph.input, *graph.graph.output]:
                value.type.tensor_type.shape.dim[0].dim_value = cars * 2
            del graph.graph.value_info[:]
            graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
            name = f'cfv-{key}-cars{cars}.onnx'
            target = output / name
            onnx.save(graph, target)
            onnx.checker.check_model(str(target), full_check=True)
            data = target.read_bytes()
            entry = {**spec, 'file': name, 'batch': cars * 2, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            if external:
                entry['externalData'] = external
            batches.setdefault(str(cars), {})[key] = entry
            print(name, len(data))
    manifest['gpuBatches'] = batches
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    build()
