"""Export the DriveBC-adapted facing model and its confidence policy for the browser."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from .train_facing import FastFacing


def export(output, models):
    crossval = json.loads((output / 'crossval.json').read_text())
    model = FastFacing(); model.load_state_dict(torch.load(output / 'student.pt', map_location='cpu', weights_only=True)); model.eval()
    target = models / 'drivebc-facing-v3.onnx'
    # Rows come in crop/mirror pairs; a dynamic batch lets the worker send 1–8 cars in one dispatch.
    torch.onnx.export(model, torch.zeros(2, 3, 224, 224), target, input_names=['image'], output_names=['view_logits', 'angle_vector'], opset_version=17, dynamo=False,
                      dynamic_axes={'image': {0: 'rows'}, 'view_logits': {0: 'rows'}, 'angle_vector': {0: 'rows'}})
    onnx.checker.check_model(onnx.load(target), full_check=True)
    data = target.read_bytes()
    curve = {round(row['min_confidence'], 2): row for row in crossval['curve']}
    manifest = {
        'version': 'drivebc-facing-v3', 'file': target.name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
        'model': 'MobileNetV3-Large, CFV + DriveBC adapted', 'input': {'size': 224, 'internal_size': 160, 'rescale': 1 / 255},
        'views': ['front', 'front_left', 'left_side', 'rear_left', 'rear', 'rear_right', 'right_side', 'front_right'],
        'policy': {'high': crossval['thresholds']['high'], 'medium': crossval['thresholds']['medium']},
        'crossval': {k: crossval[k] for k in ['labels', 'cameras', 'folds', 'baseline_accuracy', 'adapted_accuracy', 'tiers']},
        'labels': 'pipeline/data/drivebc-facing-labels.json',
        'limitations': 'Visible side of the vehicle from one photo. Accuracy measured on hand-labeled DriveBC crops with camera-grouped cross-validation; facing is not measured motion.',
        'checkpoint_sha256': hashlib.sha256((output / 'student.pt').read_bytes()).hexdigest(),
    }
    (models / 'vehicle-facing-v3.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = {**manifest, 'curve': crossval['curve'], 'training_script': 'https://github.com/Shubin123/locaIntel/blob/main/pipeline/train_drivebc_facing.py',
              'dataset_source': 'https://huggingface.co/datasets/fort-cyber/CFV-Dataset plus hand-labeled DriveBC crops (Open Government Licence – BC imagery)'}
    (models / 'facing-training-v3.json').write_text(json.dumps(report, indent=2) + '\n')
    print(target, len(data), manifest['sha256'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('.cache/facing-drivebc'))
    p.add_argument('--models', type=Path, default=Path('web/models'))
    args = p.parse_args(); export(args.output, args.models)
