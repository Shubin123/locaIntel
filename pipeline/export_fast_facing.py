"""Export a validated two-view MobileNet facing checkpoint for browser use."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch

from .train_facing import FastFacing, BACKBONE_REVISION, ARCHIVE_SHA256


def export(output, models):
    evaluation=json.loads((output/'evaluation.json').read_text())
    policy=evaluation['validation_policy']
    if not policy:
        raise RuntimeError('No validation policy met the accuracy gate')
    for name,result in evaluation['official_test'].items():
        if result['cascade']['eight_view_accuracy'] < result['teacher']['eight_view_accuracy']:
            raise RuntimeError('Cascade failed the independent accuracy gate: '+name)
    model=FastFacing();model.load_state_dict(torch.load(output/'student.pt',map_location='cpu',weights_only=True));model.eval()
    target=models/'cfv-mobilenet-facing-v2.onnx'
    # Batch2 is the original crop and its mirror. One dispatch, same independent
    # verification, with static shapes and no FP16 feature requirement.
    torch.onnx.export(model,torch.zeros(2,3,224,224),target,input_names=['image'],output_names=['view_logits','angle_vector'],opset_version=17,dynamo=False)
    onnx.checker.check_model(onnx.load(target),full_check=True)
    data=target.read_bytes();training=json.loads((output/'training.json').read_text())
    training.setdefault('backbone_revision',BACKBONE_REVISION)
    training.setdefault('optimizer',{'name':'AdamW','lr':.0008,'weight_decay':.02,'batch_size':96,
                                     'schedule':'cosine','epochs':12,'minimum_lr':.00004})
    manifest_path=models/'vehicle-facing-v2.json'
    manifest=json.loads((manifest_path if manifest_path.exists() else models/'vehicle-facing-v1.json').read_text())
    manifest['version']='cfv-facing-cascade-v2'
    manifest['fast']={'file':target.name,'model':'CFV MobileNetV3-Large, low-resolution fine-tuned',
                     'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'batch':2,'input_size':224,'internal_size':160,
                     'output':'8 view logits + sine/cosine angle vector','precision':'FP32, GPU without shader-f16; WASM fallback',
                     'policy':policy,'training_report':'facing-training-v2.json',
                     'checkpoint_sha256':hashlib.sha256((output/'student.pt').read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    report={'training':training,'evaluation':evaluation,'checkpoint_sha256':manifest['fast']['checkpoint_sha256'],
            'archive_sha256':ARCHIVE_SHA256,
            'training_script':'https://github.com/Shubin123/locaIntel/blob/main/pipeline/train_facing.py',
            'dataset_source':'https://huggingface.co/datasets/fort-cyber/CFV-Dataset/tree/'+training['revision'],
            'pretrained_backbone':'https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k',
            'limitations':'CFV labeled isolated crops, not DriveBC labels. Confidence is validation-scaled; deployment accuracy under camera/weather/domain shift remains unmeasured.'}
    (models/'facing-training-v2.json').write_text(json.dumps(report,indent=2)+'\n')
    print(target,len(data),manifest['fast']['sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('.cache/facing-training'));p.add_argument('--models',type=Path,default=Path('web/models'));args=p.parse_args();export(args.output,args.models)
