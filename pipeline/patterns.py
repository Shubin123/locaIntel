"""Bounded 24-hour sampled occupancy, never traffic throughput."""
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from .analysis import VEHICLES


def record(history, camera_id, state, now):
    rows=history.setdefault(camera_id,[])
    if not rows or rows[-1]['hash']!=state['hash']:
        detections=[d for d in state['detections'] if d['label'] in VEHICLES]
        rows.append(dict(hash=state['hash'],time=now,count=len(detections),
                         moving=sum(d.get('motion',{}).get('state')=='moving' for d in detections),
                         points=[[(d['box'][0]+d['box'][2])/2,d['box'][3],d.get('on_road') is True] for d in detections]))
    for key in list(history):
        history[key]=[r for r in history[key] if r['time']>=now-86400][-48:]
        if not history[key]:
            del history[key]


def summarize(history, now):
    cameras={};hours=defaultdict(lambda:defaultdict(list))
    for camera_id,rows in history.items():
        rows=[r for r in rows if now-86400<=r['time']<=now]
        if not rows:
            continue
        grid=np.zeros((24,24));road=np.zeros((24,24))
        for row in rows:
            hour=datetime.fromtimestamp(row['time'],timezone.utc).strftime('%Y-%m-%dT%H:00:00Z')
            hours[hour][camera_id].append(row['count'])
            for x,y,on_road in row['points']:
                i=min(23,max(0,int(y*24)));j=min(23,max(0,int(x*24)))
                grid[i,j]+=1
                if on_road:
                    road[i,j]+=1
        cameras[camera_id]=dict(samples=len(rows),first=rows[0]['time'],last=rows[-1]['time'],
            mean_vehicles=round(sum(r['count'] for r in rows)/len(rows),2),
            occupancy_grid=np.round(grid/len(rows),4).tolist(),road_grid=np.round(road/len(rows),4).tolist())
    hourly=[dict(hour=h,cameras=len(per_camera),samples=sum(len(v) for v in per_camera.values()),
                 mean_vehicles=round(sum(sum(v)/len(v) for v in per_camera.values())/len(per_camera),2))
            for h,per_camera in sorted(hours.items())]
    return dict(window_hours=24,cameras=cameras,hourly=hourly,
                metric='visible vehicles per distinct sampled frame; hourly means weight each camera equally',
                generated=datetime.fromtimestamp(now,timezone.utc).isoformat())
