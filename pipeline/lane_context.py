"""Lane context for vehicle facing: does knowing nearby cars' facing improve each car's?

Cars in the same or an adjacent lane usually show the same side to the camera;
cars several lanes across (often past the median, or on a cross street) usually
show another. This learns, from hand-labeled pairs on the same camera, how a
neighbor's front / rear / side predicts a car's own, by lateral offset in car
widths, and applies it as a prior on the crop model's scores:

    p'(c) ∝ p(c) · (m(c) / π(c))^α,   m(c) = mean over neighbors j of Σ_k T_bin(ij)(c | k) p_j(k)

Evaluation is camera-grouped: every score, labeled or not, comes from a
cross-validation model that never saw the camera (train_drivebc_facing --folds 5
--distill writes them), and each held-out fold's matrices and α are fitted on
the other folds only. Neighbors are the other cars in the frame, plus, with
--collect / --score-history, cars in later frames of the same camera.

Result (README, *Lane context*): the pattern is real in the labels, but the crop
model's mistakes follow the lane too, so neighbors' scores add no independent
evidence and α is fitted to ≈0. It is not used in the browser.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ['front', 'rear', 'side']
# Lateral offset between bottom-centers, in car widths. Lanes are ~2 car widths wide.
BINS = [0, 1, 2, 4, 8]
ALPHAS = [0, .25, .5, .75, 1, 1.5, 2]
HISTORY = ROOT / '.cache' / 'lane-history'


def lateral(a, b):
    """Horizontal offset between two boxes' bottom-centers in mean car widths."""
    width = max(1e-6, ((a[2] - a[0]) + (b[2] - b[0])) / 2)
    return abs((a[0] + a[2]) - (b[0] + b[2])) / 2 / width


def bin_of(a, b):
    u = lateral(a, b)
    return max(i for i, edge in enumerate(BINS) if u >= edge)


def fit_compatibility(pairs, smoothing=4.0):
    """T[b][c][k] = P(car is c | neighbor is k, offset bin b), smoothed toward the label marginal."""
    counts = np.zeros((len(BINS), 3, 3)); marginal = np.zeros(3)
    for b, c, k in pairs:
        counts[b, c, k] += 1; counts[b, k, c] += 1; marginal[c] += 1; marginal[k] += 1
    marginal = (marginal + 1) / (marginal + 1).sum()
    table = (counts + smoothing * marginal[None, :, None]) / (counts.sum(1, keepdims=True) + smoothing)
    return table, marginal


def adjust(probs, neighbors, table, marginal, alpha):
    """Combine a car's scores with its neighbors' (box, scores) on the same camera."""
    p = np.asarray(probs, dtype=float)
    if alpha <= 0 or not neighbors:
        return p / p.sum()
    message = np.zeros(3)
    for b, q in neighbors:
        message += table[b] @ np.asarray(q, dtype=float)
    message /= len(neighbors)
    out = p * (message / marginal) ** alpha
    return out / out.sum()


def load(crossval, labels, history=None):
    """Every vehicle with out-of-fold scores, grouped by camera; labeled ones carry their truth."""
    rows = json.loads(Path(labels).read_text())['labels']
    report = json.loads(Path(crossval).read_text())
    cars = [{'camera': r['camera'], 'file': r['file'], 'box': r['box'], 'probs': p, 'truth': GROUPS.index(r['group'])}
            for r, p in zip(rows, report['predictions'])]
    labeled = {(r['file'], str(r['box'])) for r in rows}
    cars += [{'camera': c['camera'], 'file': c['file'], 'box': c['box'], 'probs': c['probs'], 'truth': None}
             for c in report.get('context', []) if (c['file'], str(c['box'])) not in labeled]
    if history and Path(history).exists():
        cars += [{**c, 'truth': None, 'history': True} for c in json.loads(Path(history).read_text())]
    by_camera = defaultdict(list)
    for car in cars:
        by_camera[car['camera']].append(car)
    return by_camera


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0, w) * max(0, h)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def neighbors_of(car, cars, history=True):
    # A near-identical box in another frame is most likely the same parked vehicle, not a lane neighbor.
    return [(bin_of(car['box'], other['box']), other['probs']) for other in cars
            if other is not car and (history or not other.get('history'))
            and not (other['file'] != car['file'] and iou(car['box'], other['box']) > .85)]


class HistoryFrames(dict):
    def __missing__(self, name):
        from PIL import Image
        self[name] = Image.open(HISTORY / 'frames' / name).convert('RGB'); return self[name]


def score_history(crossval):
    """Score history vehicles with the cross-validation model that never saw their camera."""
    import torch
    from .train_drivebc_facing import FastFacing, predict_groups
    report = json.loads(Path(crossval).read_text()); fold_of = report['fold_of']
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    items = []
    for line in (HISTORY / 'index.jsonl').read_text().splitlines():
        row = json.loads(line)
        for d in row['detections']:
            x, y, xx, yy = d['box']
            if row['camera'] in fold_of and d['label'] in ('car', 'truck', 'bus') and (xx - x) * row['width'] >= 8 and (yy - y) * row['height'] >= 6:
                items.append({'camera': row['camera'], 'file': row['file'], 'box': d['box'], 'observed': row['observed']})
    frames, scored = HistoryFrames(), []
    for k in sorted(set(fold_of.values())):
        model = FastFacing(); model.load_state_dict(torch.load(Path(crossval).parent / f'fold-{k}.pt', map_location='cpu', weights_only=True)); model.to(device)
        part = [r for r in items if fold_of[r['camera']] == k]
        scored += [{**r, 'probs': p.tolist()} for r, p in zip(part, predict_groups(model, part, frames, device))]
    (HISTORY / 'scores.json').write_text(json.dumps(scored))
    return scored


def labeled_pairs(by_camera, cameras):
    return [(bin_of(a['box'], b['box']), a['truth'], b['truth'])
            for camera in cameras for i, a in enumerate(by_camera[camera]) for b in by_camera[camera][i + 1:]
            if a['truth'] is not None and b['truth'] is not None]


def targets(by_camera, cameras, history=True):
    return [(car, neighbors_of(car, by_camera[camera], history)) for camera in cameras for car in by_camera[camera] if car['truth'] is not None]


def score(items, table, marginal, alpha):
    pred = np.array([adjust(car['probs'], near, table, marginal, alpha) for car, near in items])
    truth = np.array([car['truth'] for car, _ in items])
    return pred, truth


def select_alpha(by_camera, cameras, history=True):
    table, marginal = fit_compatibility(labeled_pairs(by_camera, cameras))
    items = targets(by_camera, cameras, history)
    loss = {a: -np.log(np.clip(score(items, table, marginal, a)[0][np.arange(len(items)), [c['truth'] for c, _ in items]], 1e-6, 1)).mean() for a in ALPHAS}
    return min(loss, key=loss.get), table, marginal


def tiers(pred, truth, thresholds):
    conf, ok = pred.max(1), pred.argmax(1) == truth
    out = {}
    for name, keep in [('high', conf >= thresholds['high']), ('medium', (conf >= thresholds['medium']) & (conf < thresholds['high'])), ('low', conf < thresholds['medium'])]:
        out[name] = {'share': float(keep.mean()), 'accuracy': float(ok[keep].mean()) if keep.any() else None}
    return out


def evaluate(by_camera, folds=5, seed=731, thresholds=None, history=True):
    cameras = sorted(c for c in by_camera if any(car['truth'] is not None for car in by_camera[c]))
    random.Random(seed).shuffle(cameras)
    fold_of = {c: i % folds for i, c in enumerate(cameras)}
    base, adjusted, truth, alphas, with_context = [], [], [], [], []
    for k in range(folds):
        train = [c for c in cameras if fold_of[c] != k]; test = [c for c in cameras if fold_of[c] == k]
        alpha, table, marginal = select_alpha(by_camera, train, history)
        items = targets(by_camera, test, history)
        p, t = score(items, table, marginal, alpha)
        base += [np.asarray(car['probs']) / sum(car['probs']) for car, _ in items]; adjusted += list(p); truth += list(t)
        with_context += [bool(near) for _, near in items]; alphas.append(alpha)
    base, adjusted, truth, with_context = np.array(base), np.array(adjusted), np.array(truth), np.array(with_context)
    acc = lambda p, keep=slice(None): float((p[keep].argmax(1) == truth[keep]).mean())
    nll = lambda p: float(-np.log(np.clip(p[np.arange(len(p)), truth], 1e-6, 1)).mean())
    report = {'labels': int(len(truth)), 'cameras': len(cameras), 'folds': folds, 'alphas': alphas,
              'with_neighbors': float(with_context.mean()),
              'accuracy': {'crop_model': acc(base), 'lane_context': acc(adjusted)},
              'accuracy_with_neighbors': {'crop_model': acc(base, with_context), 'lane_context': acc(adjusted, with_context)},
              'log_loss': {'crop_model': nll(base), 'lane_context': nll(adjusted)},
              'changed': int((base.argmax(1) != adjusted.argmax(1)).sum()),
              'fixed': int(((base.argmax(1) != truth) & (adjusted.argmax(1) == truth)).sum()),
              'broken': int(((base.argmax(1) == truth) & (adjusted.argmax(1) != truth)).sum())}
    if thresholds:
        report['tiers'] = {'crop_model': tiers(base, truth, thresholds), 'lane_context': tiers(adjusted, truth, thresholds)}
    return report


def collect_history(rounds, minutes, workers=8):
    """Fresh frames + YOLO boxes from the labeled cameras, kept apart from the training pool."""
    import csv
    import hashlib
    import io
    import time
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    from PIL import Image
    from ultralytics import YOLO
    cameras = {r['camera'] for r in json.loads((ROOT / 'pipeline' / 'data' / 'drivebc-facing-labels.json').read_text())['labels']}
    rows = [r for r in csv.DictReader(io.StringIO((ROOT / '.cache' / 'catalog.csv').read_text(encoding='utf-8-sig'))) if r['id'] in cameras]
    (HISTORY / 'frames').mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / 'yolo11n.pt'))
    with httpx.Client(timeout=20, follow_redirects=True, headers={'User-Agent': 'LocaIntel/1.0 (lane history)'}) as client:
        def fetch(row):
            try:
                r = client.get(row['links_imageURL']); r.raise_for_status(); Image.open(io.BytesIO(r.content)).verify(); return row, r.content
            except Exception:
                return row, None
        for n in range(rounds):
            began, added = time.time(), 0
            with (HISTORY / 'index.jsonl').open('a') as index:
                for row, content in ThreadPoolExecutor(workers).map(fetch, rows):
                    if not content:
                        continue
                    path = HISTORY / 'frames' / f"{row['id']}-{hashlib.sha256(content).hexdigest()[:12]}.jpg"
                    if path.exists():
                        continue
                    image = Image.open(io.BytesIO(content)).convert('RGB')
                    result = model.predict(image, conf=.3, classes=[2, 3, 5, 7], verbose=False)[0]
                    detections = [dict(label=result.names[int(b.cls.item())], confidence=round(float(b.conf.item()), 3),
                                       box=[round(float(v), 4) for v in b.xyxyn[0].tolist()]) for b in result.boxes]
                    path.write_bytes(content); added += 1
                    index.write(json.dumps(dict(camera=row['id'], file=path.name, observed=time.time(), width=image.width, height=image.height, detections=detections)) + '\n')
            print(json.dumps({'round': n + 1, 'cameras': len(rows), 'new_frames': added}), flush=True)
            if n + 1 < rounds:
                time.sleep(max(0, minutes * 60 - (time.time() - began)))


def shared_errors(by_camera):
    """When the crop model is wrong, do same-lane cars in the frame vote for the truth or for its mistake?"""
    out = {'model_wrong': 0, 'neighbors_vote_truth': 0, 'neighbors_vote_model': 0}
    for cars in by_camera.values():
        for car in cars:
            near = [o for o in cars if o is not car and o['file'] == car['file'] and bin_of(car['box'], o['box']) == 0]
            own = int(np.argmax(car['probs']))
            if car['truth'] is None or not near or own == car['truth']:
                continue
            votes = defaultdict(int)
            for o in near:
                votes[int(np.argmax(o['probs']))] += 1
            top = max(votes, key=votes.get)
            out['model_wrong'] += 1; out['neighbors_vote_truth'] += top == car['truth']; out['neighbors_vote_model'] += top == own
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--collect', type=int, metavar='ROUNDS', help='collect this many rounds of history frames for the labeled cameras, then exit')
    p.add_argument('--minutes', type=float, default=10, help='minutes between collection rounds')
    p.add_argument('--crossval', type=Path, default=ROOT / '.cache' / 'facing-lanes' / 'crossval.json')
    p.add_argument('--labels', type=Path, default=ROOT / 'pipeline' / 'data' / 'drivebc-facing-labels.json')
    p.add_argument('--policy', type=Path, default=ROOT / 'web' / 'models' / 'vehicle-facing-v3.json')
    p.add_argument('--score-history', action='store_true', help='score collected history vehicles with the fold models first')
    p.add_argument('--report', type=Path, help='also write the report here')
    args = p.parse_args()
    if args.collect:
        return collect_history(args.collect, args.minutes)
    if args.score_history:
        score_history(args.crossval)
    by_camera = load(args.crossval, args.labels, HISTORY / 'scores.json')
    policy = json.loads(args.policy.read_text()).get('policy', {}) if args.policy.exists() else {}
    thresholds = {'high': policy['high'], 'medium': policy['medium']} if 'high' in policy else None
    history = sum(car.get('history', False) for cars in by_camera.values() for car in cars)
    report = {'history_vehicles': history, 'same_frames': evaluate(by_camera, thresholds=thresholds, history=False),
              'with_history': evaluate(by_camera, thresholds=thresholds), 'shared_errors': shared_errors(by_camera)}
    text = json.dumps(report, indent=1)
    print(text)
    if args.report:
        args.report.write_text(text + '\n')


if __name__ == '__main__':
    main()
