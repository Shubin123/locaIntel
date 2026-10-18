"""Bounded DriveBC snapshot ingestion, content cache and YOLO inference."""
import argparse
import csv
import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image
from .analysis import RoadModel, analyze_scene, VERSION
from .patterns import record, summarize

CATALOG = 'https://catalogue.data.gov.bc.ca/dataset/6b39a910-6c77-476f-ac96-7b4f18849b1c/resource/a9d52d85-8402-4ce7-b2ac-a2779837c48a/download/webcams.csv'
ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
CACHE = ROOT / '.cache'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, separators=(',', ':')))
    temp.replace(path)


def catalog(client):
    path = CACHE / 'catalog.csv'
    if not path.exists() or time.time() - path.stat().st_mtime > 86400:
        response = client.get(CATALOG)
        response.raise_for_status()
        path.write_text(response.content.decode('utf-8-sig'))
    return [dict(id=r['id'], name=r['camName'], region=r['region_name'], highway=r['highway_number'],
                 caption=r['caption'], image=r['links_imageURL'], source=r['links_camDetailsPage'],
                 lat=float(r['latitude']), lon=float(r['longitude']))
            for r in csv.DictReader(io.StringIO(path.read_text()))]


def collect(limit=48, workers=6, ttl=120, model_name='yolo11n.pt', region='', road_model='segformer'):
    CACHE.mkdir(exist_ok=True)
    (WEB / 'frames').mkdir(parents=True, exist_ok=True)
    state_path = CACHE / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    history_path = CACHE / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    calibrations = json.loads((ROOT / 'calibrations.json').read_text()) if (ROOT / 'calibrations.json').exists() else {}
    model_errors = []
    road = None
    road_attempted = False
    with httpx.Client(timeout=25, follow_redirects=True, headers={'User-Agent': 'LocaIntel/1.0 (DriveBC snapshot analysis)'}) as client:
        cameras = catalog(client)
        atomic_json(WEB / 'data/cameras.json', cameras)
        candidates = [c for c in cameras if not region or c['region'] == region]
        # Rotate through the entire catalogue between runs, prioritizing oldest observations.
        candidates.sort(key=lambda c: state.get(c['id'], {}).get('checked', 0))
        selected = candidates[:limit]
        def download(camera):
            previous = state.get(camera['id'], {})
            if time.time() - previous.get('checked', 0) < ttl:
                return camera, None, previous, None
            try:
                response = client.get(camera['image'])
                response.raise_for_status()
                content = response.content
                Image.open(io.BytesIO(content)).verify()
                return camera, content, previous, response.headers.get('last-modified')
            except Exception as exc:
                return camera, None, {**previous, 'error': str(exc), 'checked': time.time()}, None
        fetched = list(ThreadPoolExecutor(max_workers=workers).map(download, selected))
    model = None
    for camera, content, previous, modified in fetched:
        if content is None:
            state[camera['id']] = previous
            continue
        digest = hashlib.sha256(content).hexdigest()
        if digest == previous.get('hash') and previous.get('model') == model_name and previous.get('analysis', {}).get('version') == VERSION and (road_model == 'none' or previous.get('analysis', {}).get('road_model')):
            state[camera['id']] = {**previous, 'checked': time.time(), 'error': None}
            continue
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_name)
        if road_model != 'none' and not road_attempted:
            road_attempted = True
            try:
                road = RoadModel()
            except Exception as exc:
                model_errors.append({'model': 'SegFormer', 'error': str(exc)})
        result = model.predict(Image.open(io.BytesIO(content)).convert('RGB'), conf=.3,
                               classes=[0, 1, 2, 3, 5, 7], verbose=False)[0]
        detections = []
        counts = {}
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            counts[label] = counts.get(label, 0) + 1
            detections.append(dict(label=label, confidence=round(float(box.conf.item()), 3),
                                   box=[round(float(v), 4) for v in box.xyxyn[0].tolist()]))
        frame = WEB / 'frames' / (camera['id'] + '.jpg')
        image = Image.open(io.BytesIO(content)).convert('RGB')
        old_image = Image.open(frame).convert('RGB') if frame.exists() else None
        # Last-Modified is upstream time; download wall-clock spacing alone is not capture cadence.
        gap = None
        if modified and previous.get('source_modified'):
            from email.utils import parsedate_to_datetime
            try:
                gap = (parsedate_to_datetime(modified)-parsedate_to_datetime(previous['source_modified'])).total_seconds()
            except (ValueError, TypeError):
                pass
        analysis = analyze_scene(image, detections, old_image, previous.get('detections', []), gap, road=road, calibration=calibrations.get(camera['id']))
        tmp = frame.with_suffix('.tmp')
        tmp.write_bytes(content)
        tmp.replace(frame)
        state[camera['id']] = dict(hash=digest, checked=time.time(), observed=datetime.now(timezone.utc).isoformat(),
            source_modified=modified, model=model_name, counts=counts, detections=detections, error=None, analysis=analysis)
        record(history, camera['id'], state[camera['id']], time.time())
    atomic_json(state_path, state)
    atomic_json(history_path, history)
    atomic_json(WEB / 'data/patterns.json', summarize(history, time.time()))
    rows = [{**c, **state[c['id']], 'frame': 'frames/' + c['id'] + '.jpg'}
            for c in cameras if c['id'] in state and state[c['id']].get('hash')]
    # Keep errors visible, including cameras which have never yielded a valid frame.
    errors = [dict(id=c['id'], name=c['name'], error=state[c['id']]['error'])
              for c in cameras if state.get(c['id'], {}).get('error')]
    payload = dict(generated=datetime.now(timezone.utc).isoformat(), cameras=rows, errors=errors,
                   catalog_size=len(cameras), model=model_name, model_errors=model_errors, schema_version=VERSION)
    atomic_json(WEB / 'data/latest.json', payload)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=48)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--ttl', type=int, default=120)
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--region', default='')
    parser.add_argument('--road-model', choices=['segformer', 'none'], default='segformer')
    args = parser.parse_args()
    if args.limit < 1 or not 1 <= args.workers <= 16 or args.ttl < 0:
        parser.error('limit must be positive, workers 1–16, TTL nonnegative')
    print(json.dumps({'analyzed': len(collect(args.limit, args.workers, args.ttl, args.model, args.region, args.road_model)['cameras'])}))
