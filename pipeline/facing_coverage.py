"""Grow a DriveBC vehicle-crop dataset and measure browser facing coverage on it.

`collect` appends fresh DriveBC photos plus YOLO boxes to .cache/drivebc-facing.
`coverage` replays the browser facing policy (web/orientation.js) with ONNX
Runtime on every vehicle box and reports how many receive an estimate and at
which confidence tier. Accuracy comes from the hand labels and camera-grouped
cross-validation in pipeline.train_drivebc_facing.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / '.cache' / 'drivebc-facing'
MODELS = ROOT / 'web' / 'models'
VEHICLES = {'car', 'truck', 'bus', 'motorcycle'}


def collect(count, workers=8):
    import httpx
    from ultralytics import YOLO
    (DATASET / 'frames').mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(io.StringIO((ROOT / '.cache' / 'catalog.csv').read_text(encoding='utf-8-sig'))))
    index_path = DATASET / 'index.jsonl'
    seen = collections.Counter(json.loads(l)['camera'] for l in index_path.read_text().splitlines()) if index_path.exists() else {}
    # Prefer cameras with the fewest saved frames so the dataset spreads across the province.
    rows.sort(key=lambda r: (seen.get(r['id'], 0), hashlib.md5((r['id'] + str(time.time() // 3600)).encode()).hexdigest()))
    rows = rows[:count]
    with httpx.Client(timeout=20, follow_redirects=True, headers={'User-Agent': 'LocaIntel/1.0 (facing coverage)'}) as client:
        def fetch(row):
            try:
                r = client.get(row['links_imageURL']); r.raise_for_status(); Image.open(io.BytesIO(r.content)).verify(); return row, r.content
            except Exception:
                return row, None
        fetched = list(ThreadPoolExecutor(workers).map(fetch, rows))
    model = YOLO(str(ROOT / 'yolo11n.pt'))
    added = 0
    with index_path.open('a') as index:
        for row, content in fetched:
            if not content:
                continue
            digest = hashlib.sha256(content).hexdigest()
            path = DATASET / 'frames' / f"{row['id']}-{digest[:12]}.jpg"
            if path.exists():
                continue
            image = Image.open(io.BytesIO(content)).convert('RGB')
            result = model.predict(image, conf=.3, classes=[2, 3, 5, 7], verbose=False)[0]
            detections = [dict(label=result.names[int(b.cls.item())], confidence=round(float(b.conf.item()), 3),
                               box=[round(float(v), 4) for v in b.xyxyn[0].tolist()]) for b in result.boxes]
            path.write_bytes(content)
            index.write(json.dumps(dict(camera=row['id'], file=path.name, observed=time.time(), orientation=row['orientation'],
                                        width=image.width, height=image.height, detections=detections)) + '\n')
            added += 1
    return added


def softmax(x):
    x = np.exp(x - x.max()); return x / x.sum()


def crop_input(image, box, size=224):
    """Square-resized box crop, as in web/orientation.js prepare()."""
    x, y, xx, yy = box
    crop = image.crop(tuple(round(v) for v in (x * image.width, y * image.height, xx * image.width, yy * image.height))).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255


def coverage(model_path=MODELS / 'vehicle-facing-v3.json'):
    """Replay the browser v3 policy (web/orientation.js classify) on every detected vehicle."""
    import onnxruntime as ort
    manifest = json.loads(Path(model_path).read_text())
    options = ort.SessionOptions(); options.log_severity_level = 3
    session = ort.InferenceSession(str(MODELS / manifest['file']), options)
    policy = manifest['policy']
    totals = collections.Counter(); groups = collections.Counter()
    for line in (DATASET / 'index.jsonl').read_text().splitlines():
        row = json.loads(line); image = None
        for d in row['detections']:
            totals['vehicles'] += 1
            x, y, xx, yy = d['box']; w, h = (xx - x) * row['width'], (yy - y) * row['height']
            if d['label'] not in {'car', 'truck', 'bus'}:
                totals['skipped_unsupported'] += 1; continue
            if w < 8 or h < 6:
                totals['skipped_tiny'] += 1; continue
            image = image or Image.open(DATASET / 'frames' / row['file']).convert('RGB')
            crop = crop_input(image, d['box'])
            logits = session.run(None, {'image': np.stack([crop, crop[:, :, ::-1]]).astype(np.float32)})[0]
            p0, p1 = softmax(logits[0]), softmax(logits[1]); p = (p0 + p1[[(8 - k) % 8 for k in range(8)]]) / 2
            g = {'front': p[0] + p[1] + p[7], 'rear': p[3] + p[4] + p[5], 'side': p[2] + p[6]}
            best = max(g, key=g.get); score = g[best]
            tier = 'high' if score >= policy['high'] else 'medium' if score >= policy['medium'] else 'low'
            if x <= .001 or y <= .001 or xx >= .999 or yy >= .999:
                tier = {'high': 'medium', 'medium': 'low', 'low': 'low'}[tier]
            totals['estimated'] += 1; totals[tier] += 1; groups[best] += 1
    report = {**totals, 'coverage': totals['estimated'] / max(1, totals['vehicles']),
              'car_truck_bus_coverage': totals['estimated'] / max(1, totals['vehicles'] - totals['skipped_unsupported']),
              'groups': dict(groups), 'frames': len((DATASET / 'index.jsonl').read_text().splitlines())}
    print(json.dumps(report, indent=1))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect'); c.add_argument('--count', type=int, default=120)
    sub.add_parser('coverage')
    args = p.parse_args()
    if args.command == 'collect':
        print(json.dumps({'added': collect(args.count)}))
    else:
        coverage()


if __name__ == '__main__':
    main()
