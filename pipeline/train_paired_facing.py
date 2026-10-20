"""Train a small supervised two-view residual facing head from frozen evidence.

No guessed labels from cached DriveBC photos are used. CFV pairs are distinct
same-identity stills, not verified camera sequences. Native NumPy only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'paired-facing-v1'


def vector(current, previous):
    a, b = current['angle_degrees'], previous['angle_degrees']
    return np.array([*np.log(np.clip(current['view_scores'], 1e-6, 1)),
                     *np.log(np.clip(previous['view_scores'], 1e-6, 1)),
                     np.sin(np.deg2rad(a)), np.cos(np.deg2rad(a)),
                     np.sin(np.deg2rad(b)), np.cos(np.deg2rad(b)),
                     current['score'], previous['score'], abs((a-b+180)%360-180)/180])


def softmax(logits):
    values = np.exp(logits-logits.max(axis=1, keepdims=True))
    return values/values.sum(axis=1, keepdims=True)


def metrics(probabilities, labels, accepted=None):
    predicted = probabilities.argmax(1)
    groups = np.array([0, 0, 1, 2, 2, 2, 1, 0])
    if accepted is None:
        accepted = np.ones(len(labels), dtype=bool)
    n = int(accepted.sum())
    return {'total': len(labels), 'accepted': n, 'coverage': n/len(labels),
            'eight_view_accuracy': float((predicted[accepted] == labels[accepted]).mean()) if n else None,
            'grouped_eight_view_accuracy': float((groups[predicted[accepted]] == groups[labels[accepted]]).mean()) if n else None,
            'wrong_accepted': int((predicted[accepted] != labels[accepted]).sum())}


def train(directory, models):
    data = json.loads((directory/'pairs.json').read_text())
    evidence = json.loads((directory/'predictions.json').read_text())
    records, pairs = evidence['records'], data['pairs']
    assert len(records) == len(data['crops'])
    x = np.stack([vector(records[p['current']], records[p['previous']]) for p in pairs])
    latest = np.array([records[p['current']]['view_scores'] for p in pairs])
    previous = np.array([records[p['previous']]['view_scores'] for p in pairs])
    y = np.array([data['crops'][p['current']]['label'] for p in pairs])
    splits = np.array([p['split'] for p in pairs])
    train, valid, test = (splits == split for split in ('train', 'validation', 'test'))
    identity_sets = {s: set(p['identity'] for p in pairs if p['split'] == s) for s in ('train', 'validation', 'test')}
    assert not identity_sets['train'] & identity_sets['validation'] and not identity_sets['train'] & identity_sets['test'] and not identity_sets['validation'] & identity_sets['test']
    mean, std = x[train].mean(0), np.maximum(x[train].std(0), .1)
    z = (x-mean)/std
    baseline = np.log(np.clip(latest, 1e-6, 1))
    # Hyperparameters and threshold selection never read final test labels.
    best, history = None, []
    target = np.eye(8)[y[train]]
    for regularization in (.003, .03, .3):
        w, bias = np.zeros((23, 8)), np.zeros(8)
        mw, vw, mb, vb = np.zeros_like(w), np.zeros_like(w), np.zeros_like(bias), np.zeros_like(bias)
        for epoch in range(1, 401):
            predicted = softmax(baseline[train]+z[train]@w+bias)
            error = (predicted-target)/train.sum()
            gw, gb = z[train].T@error+regularization*w, error.sum(0)
            mw, vw = .9*mw+.1*gw, .999*vw+.001*gw*gw
            mb, vb = .9*mb+.1*gb, .999*vb+.001*gb*gb
            w -= .01*(mw/(1-.9**epoch))/(np.sqrt(vw/(1-.999**epoch))+1e-8)
            bias -= .01*(mb/(1-.9**epoch))/(np.sqrt(vb/(1-.999**epoch))+1e-8)
            if epoch % 10:
                continue
            p = softmax(baseline[valid]+z[valid]@w+bias)
            accuracy = (p.argmax(1) == y[valid]).mean()
            loss = float(-np.log(np.clip(p[np.arange(valid.sum()),y[valid]],1e-8,1)).mean())
            row = {'regularization': regularization, 'epoch': epoch, 'validation_accuracy': float(accuracy), 'validation_nll': loss}
            history.append(row)
            score = (accuracy, -loss)
            if best is None or score > best['score']:
                best = {'score': score, 'w': w.copy(), 'bias': bias.copy(), **row}
    w, bias = best['w'], best['bias']
    probabilities = softmax(baseline+z@w+bias)
    baseline_ready = np.array([records[p['current']]['state']=='estimated' and records[p['previous']]['state']=='estimated'
                               and records[p['current']].get('mirror_consistent') is True and records[p['previous']].get('mirror_consistent') is True
                               and np.argmax(latest[i]) == np.argmax(previous[i])
                               and abs((records[p['current']]['angle_degrees']-records[p['previous']]['angle_degrees']+180)%360-180) <= 22.5
                               for i,p in enumerate(pairs)])
    base_precision = metrics(latest[valid], y[valid], baseline_ready[valid])['eight_view_accuracy']
    threshold = None
    for candidate in (0., .5, .7, .8, .9, .95, .975, .99, .995):
        accepted = baseline_ready & (probabilities.argmax(1)==latest.argmax(1)) & (probabilities.max(1)>=candidate)
        result = metrics(probabilities[valid],y[valid],accepted[valid])
        if result['accepted'] >= 20 and result['eight_view_accuracy'] >= max(.99,base_precision or 0):
            threshold = candidate
            break
    if threshold is None:
        raise RuntimeError('No two-view head threshold passed validation precision/coverage requirements')
    accepted = baseline_ready & (probabilities.argmax(1)==latest.argmax(1)) & (probabilities.max(1)>=threshold)
    results = {}
    for split in ('train','validation','test'):
        results[split] = {}
        for resolution in ('clean','40px'):
            select = (splits == split) & np.array([p['resolution']==resolution for p in pairs])
            results[split][resolution] = {
                'latest_frame': metrics(latest[select],y[select]),
                'mean_two_frames': metrics((latest[select]+previous[select])/2,y[select]),
                'learned_two_frames': metrics(probabilities[select],y[select]),
                'two_verified_frames': metrics(latest[select],y[select],baseline_ready[select]),
                'learned_gated': metrics(probabilities[select],y[select],accepted[select])}
    # Final test cannot choose a new threshold; if it fails, nothing is shipped.
    for condition, result in results['test'].items():
        gated, comparison = result['learned_gated'], result['two_verified_frames']
        if gated['accepted'] < 10 or gated['eight_view_accuracy'] < .99 or gated['eight_view_accuracy'] < (comparison['eight_view_accuracy'] or 0):
            raise RuntimeError('Locked paired head failed final held-out precision gate: '+condition+' '+json.dumps(result))
    model = {'version': VERSION, 'architecture': '23-feature residual softmax head; frozen single-frame GPU cascade',
             'mean': mean.tolist(), 'std': std.tolist(), 'weights': w.tolist(), 'bias': bias.tolist(), 'threshold': threshold,
             'feature_contract': 'log(latest[8]), log(previous[8]), sin/cos latest angle, sin/cos previous angle, latest/previous group score, circular angle difference/180',
             'minimum_frames': 2, 'max_frames': 5, 'max_gap_ms': 15000, 'base_version': 'cfv-facing-cascade-v4',
             'limitations': data['pair_provenance']+' Uncalibrated scores. Shipping gate can reject but cannot override latest single-frame class.'}
    serialized = json.dumps(model,indent=2)+'\n'
    report = {'dataset':data['dataset'],'revision':data['revision'],'archive_sha256':data['archive_sha256'],
              'pair_provenance':data['pair_provenance'],'sampling':'At most16 pairs/identity near each class center ±10°. Boundaries are underrepresented; clean and40px examples are correlated.',
              'identities':{k:sorted(v) for k,v in identity_sets.items()},'pair_counts':{s:int((splits==s).sum()) for s in ('train','validation','test')},
              'base_model_provenance':evidence['model_provenance'],'extraction':{'adapter':evidence['adapter'],'browser':evidence['browser'],'crops':len(records)},
              'optimizer':{'name':'NumPy Adam','learning_rate':.01,'epochs':400,'regularization_search':[.003,.03,.3]},
              'selected':{k:v for k,v in best.items() if k not in ('w','bias','score')},'threshold':threshold,
              'results':results,'model_sha256':hashlib.sha256(serialized.encode()).hexdigest(),
              'previous_evaluation_disclosure':'The upstream compact model was trained on train identities and tuned on the same validation identities. Official test identities were evaluated previously for the original cascade; they were not used to fit this head or select its threshold.',
              'supervision':'Independent sub-agent reviewed association/persistence assumptions and required identity-disjoint head evaluation, baseline/coverage reporting, own-image labels and no cached pseudo-label training.',
              'limitations':'No labeled real same-camera sequence accuracy measurement. Cache consensus may improve consistency/selective precision but is not online supervised learning. Final output remains latest-frame class and measured motion remains independent.'}
    models.mkdir(parents=True, exist_ok=True)
    (models/'paired-facing-v1.json').write_text(serialized)
    (models/'paired-facing-training-v1.json').write_text(json.dumps(report,indent=2)+'\n')
    (directory/'optimization.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps({'selected':report['selected'],'threshold':threshold,'results':results},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('.cache/paired-facing'));p.add_argument('--models',type=Path,default=Path('web/models'));args=p.parse_args();train(args.data,args.models)
