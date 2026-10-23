"""Sample one highway corridor repeatedly to test cross-camera routes on real frames.

Each round fetches every corridor camera, runs YOLO and the v3 facing model, and
appends raw outputs (facing logits + a 12×12 colour patch per vehicle) to
.cache/route-survey/<highway>.jsonl. `node scripts/route-survey.cjs` then runs
the browser's own web/orientation.js and web/routes.js over the file.
"""
import argparse
import csv
import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .facing_coverage import crop_input, MODELS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.cache' / 'route-survey'


def corridor(highway, region):
    rows = csv.DictReader(io.StringIO((ROOT / '.cache' / 'catalog.csv').read_text(encoding='utf-8-sig')))
    return [r for r in rows if r['highway_number'] == highway and (not region or r['region_name'] == region)]


def colour_patch(image, box):
    """Interior patch matching VehicleRoutes.describe (18%/15% inset, 64%×60%)."""
    x, y, xx, yy = box; w, h = (xx - x) * image.width, (yy - y) * image.height
    left, top = (x + (xx - x) * .18) * image.width, (y + (yy - y) * .15) * image.height
    patch = image.crop((round(left), round(top), round(left + w * .64), round(top + h * .6))).resize((12, 12), Image.Resampling.BILINEAR)
    return np.asarray(patch).reshape(-1, 3).tolist()


def survey(highway, region, rounds, interval):
    import httpx
    import onnxruntime as ort
    from ultralytics import YOLO
    cams = corridor(highway, region); OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f'{highway}.jsonl'
    manifest = json.loads((MODELS / 'vehicle-facing-v3.json').read_text())
    options = ort.SessionOptions(); options.log_severity_level = 3
    facing = ort.InferenceSession(str(MODELS / manifest['file']), options)
    yolo = YOLO(str(ROOT / 'yolo11n.pt')); last = {}
    with httpx.Client(timeout=20, follow_redirects=True, headers={'User-Agent': 'LocaIntel/1.0 (route survey)'}) as client, out_path.open('a') as out:
        for round_index in range(rounds):
            started = time.time()
            def fetch(row):
                try:
                    r = client.get(row['links_imageURL']); r.raise_for_status(); return row, r.content, r.headers.get('last-modified') or r.headers.get('expires')
                except Exception:
                    return row, None, None
            new = 0
            for row, content, modified in ThreadPoolExecutor(8).map(fetch, cams):
                if not content:
                    continue
                digest = hashlib.sha256(content).hexdigest()
                if modified and not modified.startswith(('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')):
                    modified = None
                if last.get(row['id']) == digest:
                    continue
                last[row['id']] = digest; new += 1
                image = Image.open(io.BytesIO(content)).convert('RGB')
                result = yolo.predict(image, conf=.3, classes=[2, 3, 5, 7], verbose=False)[0]
                detections = []
                for b in result.boxes:
                    label = result.names[int(b.cls.item())]; box = [round(float(v), 4) for v in b.xyxyn[0].tolist()]
                    w, h = (box[2] - box[0]) * image.width, (box[3] - box[1]) * image.height
                    d = dict(label=label, box=box)
                    if label in ('car', 'truck', 'bus') and w >= 8 and h >= 6:
                        crop = crop_input(image, box)
                        d['logits'] = facing.run(None, {'image': np.stack([crop, crop[:, :, ::-1]]).astype(np.float32)})[0].reshape(-1).tolist()
                        d['patch'] = colour_patch(image, box)
                        d['truncated'] = box[0] <= .001 or box[1] <= .001 or box[2] >= .999 or box[3] >= .999
                    detections.append(d)
                out.write(json.dumps(dict(camera=dict(id=row['id'], name=row['camName'], caption=row['caption'], orientation=row['orientation'],
                                                      highway=row['highway_number'], lat=float(row['latitude']), lon=float(row['longitude'])),
                                          hash=digest, observedAt=int(time.time() * 1000), sourceModified=modified, detections=detections)) + '\n')
            out.flush()
            print(json.dumps({'round': round_index + 1, 'cameras': len(cams), 'new_frames': new}), flush=True)
            if round_index + 1 < rounds:
                time.sleep(max(0, interval - (time.time() - started)))
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--highway', default='1'); p.add_argument('--region', default='Lower Mainland')
    p.add_argument('--rounds', type=int, default=20); p.add_argument('--interval', type=int, default=60)
    args = p.parse_args(); print(survey(args.highway, args.region, args.rounds, args.interval))
