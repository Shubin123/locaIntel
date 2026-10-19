"""Identity-held-out validation and untouched official-test comparison for facing.

Writes all predictions locally so thresholds can be audited. Threshold selection
uses validation identities only; official test data never selects weights/policy.
"""
import argparse
import json
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn.functional as F

from .build_orientation_model import VehicleViewModel
from .train_facing import FastFacing, load_crops, batch


@torch.inference_mode()
def predict(model, images, angles, indices, device, student, degraded):
    results = []
    model.eval()
    for start in range(0, len(indices), 48):
        x, y, _ = batch(images, angles, indices[start:start+48], device, degraded=degraded)
        x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
        logits, angle = model(x)
        flip_logits, flip_angle = model(x.flip(-1))
        norms=angle.norm(dim=1) if student else torch.ones(len(x),device=device)
        flip_norms=flip_angle.norm(dim=1) if student else torch.ones(len(x),device=device)
        if student:
            angle = torch.remainder(torch.atan2(angle[:,0], angle[:,1]) * 180/math.pi + 360, 360)
            flip_angle = torch.remainder(torch.atan2(flip_angle[:,0], flip_angle[:,1]) * 180/math.pi + 360, 360)
        else:
            angle = angle[:,0] * 360; flip_angle = flip_angle[:,0] * 360
        for i in range(len(x)):
            results.append({'label': y[i].item(), 'angle': float(angles[indices[start+i]]),
                            'logits': logits[i].cpu().tolist(), 'predicted_angle': angle[i].item(),
                            'vector_norm':norms[i].item(),'flip_vector_norm':flip_norms[i].item(),
                            'flip_logits': flip_logits[i].cpu().tolist(), 'flip_angle': flip_angle[i].item()})
    return results


def probabilities(logits, temperature=1):
    x = np.asarray(logits)/temperature; x = np.exp(x-x.max()); return x/x.sum()


def error(a, b):
    return abs((a-b+180)%360-180)


def accepted(row, threshold, temperature=1):
    p, flip = probabilities(row['logits'], temperature), probabilities(row['flip_logits'], temperature)
    k, fk = int(p.argmax()), int(flip.argmax())
    return bool(row.get('vector_norm',1)>=.1 and row.get('flip_vector_norm',1)>=.1
                and p[k]>=threshold and flip[fk]>=threshold and fk==(8-k)%8
                and error(row['predicted_angle'], k*45)<=45
                and error(row['flip_angle'], (360-row['predicted_angle'])%360)<=25)


def metrics(rows):
    def group(k):
        return 0 if k in [0,1,7] else 1 if k in [3,4,5] else 2
    def predicted_group(row):
        p=probabilities(row['logits']);return int(np.argmax([p[0]+p[1]+p[7],p[3]+p[4]+p[5],p[2]+p[6]]))
    return {'count': len(rows), 'eight_view_accuracy': float(np.mean([int(np.argmax(r['logits']))==r['label'] for r in rows])),
            'front_rear_side_accuracy':float(np.mean([predicted_group(r)==group(r['label']) for r in rows])),
            'mean_angle_error': float(np.mean([error(r['predicted_angle'],r['angle']) for r in rows]))}


def evaluate(args):
    torch.set_num_threads(4); device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    student=FastFacing(); student.load_state_dict(torch.load(args.output/'student.pt', map_location='cpu', weights_only=True));student.to(device).eval()
    report=json.loads((args.output/'training.json').read_text())
    train_images, train_angles, identities=load_crops(args.data/'train.zip')
    valid=np.flatnonzero(np.isin(identities,report['validation_identities']))[::2]
    validation=[]
    for degraded in [False,True]:
        validation+=predict(student,train_images,train_angles,valid,device,True,degraded)
    # Fit one scalar on validation labels, without modifying the held-out test.
    logits=torch.tensor([r['logits'] for r in validation]); labels=torch.tensor([r['label'] for r in validation])
    log_temperature=torch.zeros((),requires_grad=True);optimizer=torch.optim.LBFGS([log_temperature],max_iter=60)
    def closure():
        optimizer.zero_grad();loss=F.cross_entropy(logits/log_temperature.exp(),labels);loss.backward();return loss
    optimizer.step(closure);temperature=float(log_temperature.exp().detach().clamp(.3,5))
    policy=None
    for threshold in [.8,.85,.9,.93,.95,.97,.98,.99,.995]:
        selected=[r for r in validation if accepted(r,threshold,temperature)]
        if len(selected)>=100 and metrics(selected)['eight_view_accuracy']>=.98:
            policy={'threshold':threshold,'temperature':temperature,'mirror_max_error':25,
                    'validation_accepted':len(selected),'validation_total':len(validation),**metrics(selected)};break
    print('VALIDATION',json.dumps(policy),flush=True)
    (args.output/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    del train_images
    images,angles,test_identities=load_crops(args.data/'test.zip'); indices=np.arange(len(images))
    if set(test_identities)&set(identities):
        raise RuntimeError('Official test identities overlap the training archive')
    student_predictions={}
    for degraded in [False,True]:
        name='degraded_40px' if degraded else 'native_160px'
        student_predictions[name]=predict(student,images,angles,indices,device,True,degraded)
        print('STUDENT',name,json.dumps(metrics(student_predictions[name])),flush=True)
    student.cpu();del student;torch.mps.empty_cache() if device=='mps' else None
    teacher=VehicleViewModel();teacher.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True));teacher.to(device).eval()
    evaluation={'validation_policy':policy,'official_test':{},'official_test_identities':sorted(int(v) for v in set(test_identities)),
                'scope':'CFV identity-held-out test, not measured DriveBC accuracy. Same square crops at 160px / 40px, both resized to224 at the model interface.'}
    for degraded in [False,True]:
        name='degraded_40px' if degraded else 'native_160px'
        original=predict(teacher,images,angles,indices,device,False,degraded)
        quick=student_predictions[name];mask=[accepted(r,policy['threshold'],temperature) if policy else False for r in quick]
        combined=[q if use else t for q,t,use in zip(quick,original,mask)]
        result={'teacher':metrics(original),'student':metrics(quick),'cascade':metrics(combined),'fast_path_count':sum(mask),
                'fast_path_accuracy':metrics([r for r,use in zip(quick,mask) if use]) if any(mask) else None}
        evaluation['official_test'][name]=result
        (args.output/(name+'-predictions.json')).write_text(json.dumps({'teacher':original,'student':quick,'mask':mask}))
        print('TEST',name,json.dumps(result),flush=True)
    (args.output/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('.cache/cfv-data'));p.add_argument('--output',type=Path,default=Path('.cache/facing-training'));p.add_argument('--checkpoint',type=Path,default=Path('.cache/cfv-convnext-multi-weak.pth'));evaluate(p.parse_args())
