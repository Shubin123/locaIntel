"""Analyze an authorized video/RTSP source with real short-gap frames and Meta DINOv2.

python -m pipeline.video --source traffic.mp4 --camera-id 864 --frames 120
Writes the same dashboard schema; run collection and video writers separately.
"""
import argparse
import json
import time
from datetime import datetime, timezone
import cv2
from PIL import Image
from .collect import CACHE, WEB, ROOT, atomic_json
from .analysis import AppearanceModel, RoadModel, analyze_scene
from .patterns import record, summarize


def run(source, camera_id, frames=120, appearance='dinov2', road_model='segformer'):
    from ultralytics import YOLO
    catalog_path=WEB/'data/cameras.json'
    if not catalog_path.exists():
        raise ValueError('Run pipeline.collect once to load the camera catalog first')
    cameras=json.loads(catalog_path.read_text())
    camera=next((c for c in cameras if c['id']==camera_id),None)
    if not camera:
        raise ValueError('camera-id must identify the supplied view in the DriveBC catalog')
    payload_path=WEB/'data/latest.json'
    payload=json.loads(payload_path.read_text()) if payload_path.exists() else dict(cameras=[],errors=[],catalog_size=len(cameras),model='yolo11n.pt')
    model=YOLO('yolo11n.pt')
    embedding=AppearanceModel() if appearance=='dinov2' else None
    road=RoadModel() if road_model=='segformer' else None
    calibration=json.loads((ROOT/'calibrations.json').read_text()).get(camera_id)
    history_path=CACHE/'history.json';history=json.loads(history_path.read_text()) if history_path.exists() else {}
    capture=cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError('Cannot open video source')
    fps=capture.get(cv2.CAP_PROP_FPS)
    is_file=__import__('pathlib').Path(source).is_file()
    if is_file and not 0<fps<=240:
        raise ValueError('Video must provide a valid frame rate for temporal measurements')
    previous_image=None;previous_detections=[];previous_stamp=None
    import hashlib
    try:
        for index in range(frames):
            ok,frame=capture.read()
            if not ok:
                break
            stamp=index/fps if is_file else time.monotonic()
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            result=model.predict(image,conf=.3,classes=[0,1,2,3,5,7],verbose=False)[0]
            detections=[];counts={}
            for box in result.boxes:
                label=result.names[int(box.cls.item())];counts[label]=counts.get(label,0)+1
                detections.append(dict(label=label,confidence=round(float(box.conf.item()),3),box=[round(float(v),4) for v in box.xyxyn[0].tolist()]))
            scene=analyze_scene(image,detections,previous_image,previous_detections,
                                stamp-previous_stamp if previous_stamp is not None else None,
                                road=road,appearance=embedding,calibration=calibration)
            path=WEB/'frames'/f'{camera_id}.jpg';temp=path.with_suffix('.tmp');image.save(temp,format='JPEG');content=temp.read_bytes();temp.replace(path)
            observed=datetime.now(timezone.utc).isoformat()
            state=dict(hash=hashlib.sha256(content).hexdigest(),checked=time.time(),observed=observed,model='yolo11n.pt',counts=counts,
                       detections=detections,analysis=scene,error=None,source_modified=None,input_kind='video',
                       media_time_seconds=round(stamp,3) if is_file else None,
                       media_note='Recorded video; observation time is analysis time' if is_file else 'Video source; timestamps use local arrival time')
            CACHE.mkdir(exist_ok=True)
            record(history,camera_id,state,time.time())
            row={**camera,**state,'frame':f'frames/{camera_id}.jpg'}
            payload['cameras']=[c for c in payload['cameras'] if c['id']!=camera_id]+[row]
            payload['generated']=observed
            atomic_json(payload_path,payload)
            atomic_json(WEB/'data/patterns.json',summarize(history,time.time()))
            previous_image=image;previous_detections=detections;previous_stamp=stamp
    finally:
        capture.release()
    atomic_json(history_path,history)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--camera-id',required=True)
    p.add_argument('--frames',type=int,default=120)
    p.add_argument('--appearance',choices=['dinov2','histogram'],default='dinov2')
    p.add_argument('--road-model',choices=['segformer','none'],default='segformer')
    a=p.parse_args()
    if a.frames<2:
        p.error('--frames must be at least 2')
    run(a.source,a.camera_id,a.frames,a.appearance,a.road_model)
