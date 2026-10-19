"""Train a compact, low-resolution-augmented car-facing model on labeled CFV.

Optional dependencies: torch, torchvision, timm, numpy, Pillow, onnx.
Dataset Apache-2.0: https://huggingface.co/datasets/fort-cyber/CFV-Dataset
Official test identities are never used for training or checkpoint selection.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import timm

DATA_REVISION = 'e17aaeb4860b33757c582a57e5a05e70ec40b10e'
BACKBONE_REVISION = '96f46a1c52932f27492dff66c72378eb99b443a7'
ARCHIVE_SHA256 = {'train.zip':'3831521f520895e8ce016286f5dca2d0a84a5c5c42bc28f3a21ba3f601383812',
                  'test.zip':'c5b26548dbcf9417c8d852b50813d50bedc7c03350daaf66972f1d80ed3fbf62'}


class FastFacing(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=10,
                                         pretrained_cfg_overlay={'hf_hub_id':'timm/mobilenetv3_large_100.ra_in1k@'+BACKBONE_REVISION})
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, image):
        image = F.interpolate(image, size=(160, 160), mode='bilinear', align_corners=False)
        output = self.backbone((image - self.mean) / self.std)
        return output[:, :8], output[:, 8:]


def label(angle):
    # Match CFV's inclusive upper class boundaries, including 22.5 degrees.
    return int(math.ceil((angle - 22.5) / 45)) % 8


def load_crops(path, size=160):
    digest=hashlib.sha256()
    with path.open('rb') as archive_file:
        for chunk in iter(lambda:archive_file.read(8*1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=ARCHIVE_SHA256.get(path.name):
        raise ValueError('Dataset checksum differs from the pinned CFV archive: '+str(path))
    target = path.with_suffix('.crops.npz' if size==160 else '.crops-'+str(size)+'.npz')
    if target.exists():
        data = np.load(target)
        return data['images'], data['angles'], data['identities']
    with zipfile.ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read('metadata.jsonl').splitlines()]
        images, angles, identities = [], [], []
        for row in rows:
            image = Image.open(io.BytesIO(archive.read(row['file_name']))).convert('RGB')
            box = [row[k] for k in ['x1', 'y1', 'x2', 'y2']]
            if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            image = image.crop(box).resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(image)); angles.append(row['angle']); identities.append(row['identity'])
    images, angles, identities = np.stack(images), np.array(angles), np.array(identities)
    np.savez(target, images=images, angles=angles, identities=identities)
    return images, angles, identities


def batch(images, angles, indices, device, augment=False, degraded=False):
    x = torch.from_numpy(images[indices].copy()).to(device).permute(0, 3, 1, 2).contiguous().float() / 255
    a = torch.from_numpy(angles[indices].astype(np.float32)).to(device)
    if augment:
        flips = torch.rand(len(x), device=device) < .5
        x[flips] = x[flips].flip(-1); a[flips] = (360 - a[flips]) % 360
        x = (x * torch.empty(len(x), 1, 1, 1, device=device).uniform_(.6, 1.4)).clamp(0, 1)
        contrast = torch.empty(len(x), 1, 1, 1, device=device).uniform_(.65, 1.35)
        x = ((x - .5) * contrast + .5).clamp(0, 1)
    if degraded or (augment and random.random() < .7):
        low = 40 if degraded else random.choice([24, 32, 40, 56, 80, 112])
        x = F.interpolate(F.interpolate(x, (low, low), mode='bilinear', align_corners=False), (160, 160), mode='bilinear', align_corners=False)
    y = torch.remainder(torch.ceil((a - 22.5) / 45).long(), 8)
    radians = a * math.pi / 180
    circle = torch.stack([radians.sin(), radians.cos()], dim=1)
    return x, y, circle


@torch.inference_mode()
def evaluate(model, images, angles, indices, device, degraded=False):
    model.eval(); correct, error, count = 0, 0., 0
    logits_all, circles_all = [], []
    for start in range(0, len(indices), 96):
        x, y, circle = batch(images, angles, indices[start:start+96], device, degraded=degraded)
        logits, predicted = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        predicted = F.normalize(predicted, dim=1)
        error += torch.acos((predicted * circle).sum(1).clamp(-1, 1)).sum().item() * 180 / math.pi
        count += len(x); logits_all.append(logits.cpu()); circles_all.append(predicted.cpu())
    return {'accuracy': correct / count, 'mean_angle_error': error / count, 'count': count}, torch.cat(logits_all), torch.cat(circles_all)


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    images, angles, identities = load_crops(args.data / 'train.zip')
    unique = sorted(set(identities)); random.Random(args.seed).shuffle(unique)
    validation_ids = sorted(int(v) for v in unique[:max(4, len(unique)//5)])
    valid = np.flatnonzero(np.isin(identities, validation_ids))[::2]
    training = np.flatnonzero(~np.isin(identities, validation_ids))
    model = FastFacing(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/20)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'dataset': 'fort-cyber/CFV-Dataset', 'revision': DATA_REVISION, 'seed': args.seed, 'device': device,
              'architecture': 'mobilenetv3_large_100', 'backbone_revision':BACKBONE_REVISION, 'input': 160, 'train_count': len(training),
              'optimizer': {'name':'AdamW','lr':args.lr,'weight_decay':.02,'batch_size':args.batch,
                            'schedule':'cosine','epochs':args.epochs,'minimum_lr':args.lr/20},
              'validation_count': len(valid), 'validation_identities': validation_ids,
              'augmentation': 'horizontal mirror + correct angle; brightness/contrast; 24–112px degradation', 'epochs': []}
    best = -1
    print(json.dumps({k:v for k,v in report.items() if k != 'epochs'}), flush=True)
    for epoch in range(args.epochs):
        model.train(); order = np.random.permutation(training); total = 0.; start = time.time()
        for offset in range(0, len(order), args.batch):
            x, y, circle = batch(images, angles, order[offset:offset+args.batch], device, augment=True)
            optimizer.zero_grad(set_to_none=True)
            logits, direction = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=.05) + .6 * (1 - (F.normalize(direction, dim=1) * circle).sum(1)).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            total += loss.item() * len(x)
        scheduler.step()
        clean, _, _ = evaluate(model, images, angles, valid, device)
        degraded, _, _ = evaluate(model, images, angles, valid, device, degraded=True)
        row = {'epoch': epoch+1, 'loss': total/len(order), 'seconds': round(time.time()-start, 1), 'clean': clean, 'degraded_40px': degraded}
        report['epochs'].append(row); print(json.dumps(row), flush=True)
        score = (clean['accuracy'] + degraded['accuracy']) / 2
        if score > best:
            best = score; report['selected_epoch'] = epoch + 1
            torch.save(model.cpu().state_dict(), args.output / 'student.pt'); model.to(device)
        (args.output / 'training.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Training complete; official test has not been used.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.cache/cfv-data'))
    parser.add_argument('--output', type=Path, default=Path('.cache/facing-training'))
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=96)
    parser.add_argument('--lr', type=float, default=.0008)
    parser.add_argument('--seed', type=int, default=731)
    train(parser.parse_args())
