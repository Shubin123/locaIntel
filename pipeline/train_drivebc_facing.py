"""Adapt the fast CFV facing model to elevated DriveBC camera views.

CFV photos are street-level; DriveBC cameras look down on traffic, where the
original model mistakes hoods for trunks. Training mixes CFV (8 views + angle)
with hand-labeled DriveBC crops (front / rear / side group loss). With
`--distill`, a frozen DINOv2-base + linear teacher (trained only on the fold's
labels) soft-labels every other DriveBC crop for the student. `--folds`
runs camera-grouped cross-validation so no camera appears in both train and
test; `--final` trains on every label and writes the checkpoint for export.
`--backend mlx` runs the training steps in MLX on the Apple-silicon GPU
(pipeline/mlx_facing.py) instead of PyTorch.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .train_facing import FastFacing, load_crops, batch as cfv_batch

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / '.cache' / 'drivebc-facing' / 'frames'
GROUPS = {'front': 0, 'rear': 1, 'side': 2}
GROUP_VIEWS = [[0, 1, 7], [3, 4, 5], [2, 6]]


def group_log_probs(logits):
    logp = F.log_softmax(logits, 1)
    return torch.stack([torch.logsumexp(logp[:, g], 1) for g in GROUP_VIEWS], 1)


def drivebc_crop(image, box, jitter=0.0, size=160):
    x, y, xx, yy = box
    w, h = xx - x, yy - y
    if jitter:
        x, xx = x + random.uniform(-jitter, jitter) * w, xx + random.uniform(-jitter, jitter) * w
        y, yy = y + random.uniform(-jitter, jitter) * h, yy + random.uniform(-jitter, jitter) * h
    region = tuple(round(v) for v in (x * image.width, y * image.height, xx * image.width, yy * image.height))
    return np.asarray(image.crop(region).resize((size, size), Image.Resampling.BILINEAR))


class Frames(dict):
    def __missing__(self, name):
        self[name] = Image.open(FRAMES / name).convert('RGB'); return self[name]

    def __call__(self, name):
        return self[name]


def load_labels(path):
    return json.loads(Path(path).read_text())['labels'], Frames()


def drivebc_batch(rows, images, device, augment):
    crops, labels = [], []
    for r in rows:
        c = drivebc_crop(images[r['file']], r['box'], jitter=.08 if augment else 0)
        if augment and random.random() < .5:
            c = c[:, ::-1]  # mirroring keeps front / rear / side
        crops.append(c); labels.append(GROUPS.get(r.get('group'), 0))
    x = torch.from_numpy(np.stack(crops).copy()).to(device).permute(0, 3, 1, 2).contiguous().float() / 255
    if augment:
        x = (x * torch.empty(len(x), 1, 1, 1, device=device).uniform_(.7, 1.3)).clamp(0, 1)
    return x, torch.tensor(labels, device=device)


@torch.inference_mode()
def predict_groups(model, rows, images, device):
    model.eval(); out = []
    for i in range(0, len(rows), 64):
        x, _ = drivebc_batch(rows[i:i+64], images, device, False)
        p = group_log_probs(model(x)[0]).exp() + group_log_probs(model(x.flip(-1))[0]).exp()
        out.append(p / 2)
    return torch.cat(out).cpu().numpy()


def fit(train_rows, images, cfv, device, epochs, steps, seed, init, pseudo=None):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = FastFacing(); model.load_state_dict(torch.load(init, map_location='cpu', weights_only=True)); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps, eta_min=1e-5)
    cfv_images, cfv_angles, cfv_train = cfv
    # Class-balanced DriveBC sampling: rear views are rarer in the labeled set.
    counts = {g: sum(r['group'] == g for r in train_rows) for g in GROUPS}
    weights = [1 / counts[r['group']] for r in train_rows]
    for _ in range(epochs * steps):
        model.train()
        x, y, circle = cfv_batch(cfv_images, cfv_angles, np.random.choice(cfv_train, 48), device, augment=True)
        logits, direction = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=.05) + .6 * (1 - (F.normalize(direction, dim=1) * circle).sum(1)).mean()
        dx, dy = drivebc_batch(random.choices(train_rows, weights=weights, k=48), images, device, True)
        loss = loss + 2 * F.nll_loss(group_log_probs(model(dx)[0]), dy)
        if pseudo:
            rows, targets = pseudo
            pick = np.random.choice(len(rows), 64)
            px, _ = drivebc_batch([rows[i] for i in pick], images, device, True)
            soft = torch.tensor(targets[pick], device=device, dtype=torch.float32)
            loss = loss - 1.5 * (soft * group_log_probs(model(px)[0])).sum(1).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step(); scheduler.step()
    return model


TEACHER = 'facebook/dinov2-base'


def pool_items():
    """Every DriveBC car/truck/bus box large enough for the browser model (≥8×6 px)."""
    items = []
    for line in (ROOT / '.cache' / 'drivebc-facing' / 'index.jsonl').read_text().splitlines():
        row = json.loads(line)
        for d in row['detections']:
            x, y, xx, yy = d['box']
            if d['label'] in ('car', 'truck', 'bus') and (xx - x) * row['width'] >= 8 and (yy - y) * row['height'] >= 6:
                items.append({'file': row['file'], 'camera': row['camera'], 'label': d['label'], 'box': d['box']})
    return items


@torch.inference_mode()
def teacher_features(items, images, device, cache):
    """Frozen DINOv2 [CLS ‖ mean patch] features, averaged over the crop and its mirror."""
    from transformers import AutoImageProcessor, AutoModel
    key = lambda r: r['file'] + str(r['box'])
    stored = dict(np.load(cache, allow_pickle=True)['features'].item()) if cache.exists() else {}
    missing = [r for r in items if key(r) not in stored]
    if missing:
        processor = AutoImageProcessor.from_pretrained(TEACHER); model = AutoModel.from_pretrained(TEACHER).to(device).eval()
        for i in range(0, len(missing), 32):
            chunk = missing[i:i+32]; crops = []
            for r in chunk:
                image = images(r['file']); x, y, xx, yy = r['box']; w, h = (xx - x) * image.width, (yy - y) * image.height
                crop = image.crop((round(x * image.width - w * .1), round(y * image.height - h * .1), round(xx * image.width + w * .1), round(yy * image.height + h * .1)))
                crops += [crop, crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
            out = model(**processor(images=crops, return_tensors='pt').to(device)).last_hidden_state
            f = torch.cat([out[:, 0], out[:, 1:].mean(1)], 1).view(len(chunk), 2, -1).mean(1).cpu().numpy()
            for r, v in zip(chunk, f):
                stored[key(r)] = v.astype(np.float32)
        np.savez(cache, features=np.array(stored, dtype=object))
    return np.stack([stored[key(r)] for r in items])


def fit_probe(x, y, epochs=400):
    """Class-balanced softmax regression on standardized teacher features."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.tensor((x - mean) / std), torch.tensor(y)
    weight = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3).clip(1)), dtype=torch.float32)
    head = torch.nn.Linear(x.shape[1], 3); opt = torch.optim.LBFGS(head.parameters(), max_iter=epochs)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(head(xt), yt, weight=weight) + 1e-3 * head.weight.pow(2).sum(); loss.backward(); return loss
    opt.step(closure)
    return lambda z: F.softmax(head(torch.tensor((z - mean) / std)), 1).detach().numpy()


def select_policy(pred, truth, high=.9, medium=.8):
    """Lowest group-probability thresholds whose held-out accuracy reaches each tier target."""
    score, right = pred.max(1), pred.argmax(1) == truth
    def threshold(target):
        for t in np.arange(.34, 1, .01):
            keep = score >= t
            if keep.sum() >= 20 and right[keep].mean() >= target:
                return round(float(t), 2)
        return .99
    hi, med = threshold(high), threshold(medium)
    med = min(med, hi)
    tiers = {}
    for name, mask in {'high': score >= hi, 'medium': (score >= med) & (score < hi), 'low': score < med}.items():
        tiers[name] = {'share': float(mask.mean()), 'accuracy': float(right[mask].mean()) if mask.any() else None}
    return {'thresholds': {'high': hi, 'medium': med}, 'tiers': tiers}


def select_backend(name):
    """The student's training function: PyTorch fit, or the MLX port with the same signature."""
    if name == 'torch':
        return fit
    try:
        from . import mlx_facing
    except ImportError:
        raise SystemExit('--backend mlx needs MLX on Apple silicon: pip install mlx')
    if not mlx_facing.available():
        raise SystemExit('--backend mlx needs a Metal GPU')
    print('training with MLX on the Apple GPU', flush=True)
    return mlx_facing.fit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', default=ROOT / 'pipeline' / 'data' / 'drivebc-facing-labels.json')
    p.add_argument('--init', type=Path, default=ROOT / '.cache' / 'facing-training' / 'student.pt')
    p.add_argument('--output', type=Path, default=ROOT / '.cache' / 'facing-drivebc')
    p.add_argument('--folds', type=int, default=0)
    p.add_argument('--final', action='store_true')
    p.add_argument('--distill', action='store_true', help='pseudo-label unlabeled DriveBC crops with a DINOv2 teacher')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--seed', type=int, default=731)
    p.add_argument('--backend', choices=['torch', 'mlx'], default='torch',
                   help='train the student with PyTorch, or with MLX on the Apple-silicon GPU (pipeline.mlx_facing)')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    train = select_backend(args.backend)
    rows, images = load_labels(args.labels)
    cfv_images, cfv_angles, identities = load_crops(ROOT / '.cache' / 'cfv-data' / 'train.zip')
    cfv = (cfv_images, cfv_angles, np.arange(len(cfv_images)))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.folds:
        cameras = sorted({r['camera'] for r in rows}); random.Random(args.seed).shuffle(cameras)
        fold_of = {c: i % args.folds for i, c in enumerate(cameras)}
        base = FastFacing(); base.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=True)); base.to(device)
        predictions = [None] * len(rows); baseline = predict_groups(base, rows, images, device)
        teacher_pred = np.zeros((len(rows), 3)); context = []
        if args.distill:
            labelled = {r['file'] + str(r['box']) for r in rows}
            pool = [r for r in pool_items() if r['file'] + str(r['box']) not in labelled]
            feats = teacher_features(rows, images, device, args.output / 'teacher-features.npz')
            pool_feats = teacher_features(pool, images, device, args.output / 'teacher-features.npz')
        for k in range(args.folds):
            test = [i for i, r in enumerate(rows) if fold_of[r['camera']] == k]
            train_rows = [r for r in rows if fold_of[r['camera']] != k]
            pseudo = None
            if args.distill:
                probe = fit_probe(feats[[i for i, r in enumerate(rows) if fold_of[r['camera']] != k]], np.array([GROUPS[r['group']] for r in train_rows]))
                keep = [i for i, r in enumerate(pool) if fold_of.get(r['camera']) != k]
                pseudo = ([pool[i] for i in keep], probe(pool_feats[keep]))
                teacher_pred[[i for i, r in enumerate(rows) if fold_of[r['camera']] == k]] = probe(feats[[i for i, r in enumerate(rows) if fold_of[r['camera']] == k]])
            model = train(train_rows, images, cfv, device, args.epochs, args.steps, args.seed + k, args.init, pseudo)
            for i, pr in zip(test, predict_groups(model, [rows[i] for i in test], images, device)):
                predictions[i] = pr.tolist()
            if args.distill:
                # Unlabeled vehicles on this fold's cameras, predicted by a model that never saw
                # those cameras: the neighbor context for pipeline.lane_context.
                held = [r for r in pool if fold_of.get(r['camera']) == k]
                context += [{**r, 'probs': pr.tolist()} for r, pr in zip(held, predict_groups(model, held, images, device))]
            torch.save(model.state_dict(), args.output / f'fold-{k}.pt')
            print('fold', k, 'done', flush=True)
        truth = np.array([GROUPS[r['group']] for r in rows]); pred = np.array(predictions)
        report = {'labels': len(rows), 'cameras': len(cameras), 'folds': args.folds,
                  'baseline_accuracy': float((baseline.argmax(1) == truth).mean()),
                  'adapted_accuracy': float((pred.argmax(1) == truth).mean()), 'curve': []}
        if args.distill:
            report['teacher_accuracy'] = float((teacher_pred.argmax(1) == truth).mean()); report['pseudo_labelled_pool'] = len(pool)
        for t in [0, .5, .6, .7, .8, .9]:
            keep = pred.max(1) >= t
            report['curve'].append({'min_confidence': t, 'coverage': float(keep.mean()),
                                    'accuracy': float((pred[keep].argmax(1) == truth[keep]).mean()) if keep.any() else None})
        report.update(select_policy(pred, truth))
        (args.output / 'crossval.json').write_text(json.dumps({**report, 'predictions': predictions, 'context': context, 'fold_of': fold_of}, indent=1))
        print(json.dumps(report, indent=1))
    if args.final:
        pseudo = None
        if args.distill:
            labelled = {r['file'] + str(r['box']) for r in rows}
            pool = [r for r in pool_items() if r['file'] + str(r['box']) not in labelled]
            probe = fit_probe(teacher_features(rows, images, device, args.output / 'teacher-features.npz'), np.array([GROUPS[r['group']] for r in rows]))
            pseudo = (pool, probe(teacher_features(pool, images, device, args.output / 'teacher-features.npz')))
        model = train(rows, images, cfv, device, args.epochs, args.steps, args.seed, args.init, pseudo)
        torch.save(model.cpu().state_dict(), args.output / 'student.pt')
        print('saved', args.output / 'student.pt')


if __name__ == '__main__':
    main()
