"""Conservative model fusion: YOLO objects, SegFormer roads, LK flow, DINOv2 appearance.

The meta-classifier is an explicit rule ensemble, not a trained/calibrated probability model.
Motion requires <=2 s frames; sparse webcam snapshots always abstain.
"""
import math
import cv2
import numpy as np
from PIL import Image

VEHICLES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
VERSION = 2
ROAD_MODEL = 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
DINO_MODEL = 'facebook/dinov2-small'


class RoadModel:
    def __init__(self):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        self.processor = AutoImageProcessor.from_pretrained(ROAD_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(ROAD_MODEL).eval()

    def predict(self, image):
        import torch
        inputs = self.processor(images=image, return_tensors='pt', size={'height': 512, 'width': 512})
        with torch.inference_mode():
            probs = self.model(**inputs).logits.softmax(dim=1)
        road_id = next(int(k) for k, v in self.model.config.id2label.items() if v.lower() == 'road')
        return probs[0, road_id].cpu().numpy()


class AppearanceModel:
    """Meta DINOv2 crop embeddings, used only for short-gap association."""
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.model = AutoModel.from_pretrained(DINO_MODEL).eval()

    def embed(self, crops):
        import torch
        with torch.inference_mode():
            features = self.model(**self.processor(images=crops, return_tensors='pt')).last_hidden_state[:, 0]
        features = features.cpu().numpy()
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def crop(image, box):
    w, h = image.size
    x, y, xx, yy = box
    return image.crop((max(0, int(x*w)), max(0, int(y*h)), min(w, max(int(xx*w), int(x*w)+1)),
                       min(h, max(int(yy*h), int(y*h)+1))))


def histogram(image):
    arr = cv2.cvtColor(np.asarray(image.resize((32, 32)).convert('RGB')), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([arr], [0, 1], None, [12, 8], [0, 180, 0, 256]).flatten()
    return hist / max(np.linalg.norm(hist), 1e-8)


def classify_motion(vector, spread, points, similarity, detector_confidence):
    """Evidence fusion with abstention; thresholds are heuristics, not trained probabilities."""
    magnitude = float(np.linalg.norm(vector))
    if points < 4 or similarity < .85 or detector_confidence < .3 or spread > max(1.5, magnitude*.5):
        return 'unknown'
    if magnitude >= 2:
        return 'moving'
    if magnitude <= .65:
        return 'stationary'
    return 'unknown'


def direction(vector):
    # Image coordinates: positive y points down; these are not compass headings.
    angle = math.degrees(math.atan2(vector[1], vector[0])) % 360
    return ['right', 'down-right', 'down', 'down-left', 'left', 'up-left', 'up', 'up-right'][int((angle+22.5)//45)%8]


def geographic_heading(vector, center, calibration):
    # A user-surveyed image road axis is required; camera-facing metadata is insufficient.
    if not calibration or 'road_axis' not in calibration or 'away_bearing' not in calibration:
        return None
    axis = np.asarray(calibration['road_axis'], dtype=float)
    if axis.shape != (2, 2) or np.linalg.norm(axis[1]-axis[0]) < .01:
        return None
    projection = float(np.dot(vector, axis[1]-axis[0]))
    if abs(projection) < .0001:
        return None
    return round((float(calibration['away_bearing']) + (0 if projection > 0 else 180)) % 360, 1)


def motion_analysis(previous_image, image, previous_detections, detections, gap, appearance=None, calibration=None):
    for d in detections:
        d['motion'] = {'state': 'unknown', 'reason': 'no_short_gap_pair'}
    if previous_image is None or gap is None or not 0 < gap <= 2:
        return {'status': 'insufficient_cadence', 'gap_seconds': gap, 'max_gap_seconds': 2}
    if previous_image.size != image.size:
        return {'status': 'camera_geometry_changed', 'gap_seconds': gap}
    old = [d for d in previous_detections if d['label'] in VEHICLES]
    new = [d for d in detections if d['label'] in VEHICLES]
    if not old or not new:
        return {'status': 'no_vehicle_pair', 'gap_seconds': gap}
    # Bound optical-flow work and keep all thresholds in this 640px analysis space.
    width = min(640, image.width)
    height = round(image.height*width/image.width)
    gray0 = cv2.cvtColor(np.asarray(previous_image.resize((width, height)).convert('RGB')), cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(np.asarray(image.resize((width, height)).convert('RGB')), cv2.COLOR_RGB2GRAY)
    if np.array_equal(gray0, gray1):
        return {'status': 'duplicate_frame', 'gap_seconds': gap}
    mask = np.full(gray0.shape, 255, dtype=np.uint8)
    for d in old:
        x, y, xx, yy = d['box']
        cv2.rectangle(mask, (int(x*width), int(y*height)), (int(xx*width), int(yy*height)), 0, -1)
    mask[:max(1, int(height*.08))] = 0  # Exclude timestamp overlays.
    background = cv2.goodFeaturesToTrack(gray0, 240, .02, 7, mask=mask)

    def track(points):
        if points is None or len(points) < 4:
            return np.empty((0, 2)), np.empty((0, 2))
        forward, status, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None, winSize=(21, 21), maxLevel=3)
        if forward is None:
            return np.empty((0, 2)), np.empty((0, 2))
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, forward, None, winSize=(21, 21), maxLevel=3)
        if back is None:
            return np.empty((0, 2)), np.empty((0, 2))
        valid = (status.ravel()==1) & (back_status.ravel()==1) & (np.linalg.norm(back-points, axis=2).ravel()<1)
        return points.reshape(-1, 2)[valid], forward.reshape(-1, 2)[valid]
    bg0, bg1 = track(background)
    if len(bg0) < 12:
        return {'status': 'insufficient_background_features', 'gap_seconds': gap}
    transform, inliers = cv2.estimateAffinePartial2D(bg0, bg1, method=cv2.RANSAC, ransacReprojThreshold=1)
    if transform is None or inliers.mean() < .7:
        return {'status': 'unstable_background', 'gap_seconds': gap}
    background_shift = float(np.median(np.linalg.norm(bg1-bg0, axis=1)))
    if background_shift > 2 or abs(float(np.linalg.det(transform[:, :2]))-1) > .015:
        return {'status': 'camera_moved', 'gap_seconds': gap}
    crops = [crop(previous_image, d['box']) for d in old]+[crop(image, d['box']) for d in new]
    features = appearance.embed(crops) if appearance else np.asarray([histogram(c) for c in crops])
    candidates = []
    for i, d in enumerate(old):
        x, y, xx, yy = d['box']
        roi = np.zeros_like(gray0)
        cv2.rectangle(roi, (int(x*width), int(y*height)), (int(xx*width), int(yy*height)), 255, -1)
        p, q = track(cv2.goodFeaturesToTrack(gray0, 60, .02, 3, mask=roi))
        if len(p) < 4:
            continue
        residual = q-(p @ transform[:, :2].T + transform[:, 2])
        vector = np.median(residual, axis=0)
        spread = float(np.median(np.linalg.norm(residual-vector, axis=1)))
        raw_vector = np.median(q-p, axis=0)
        projected = np.array([(x+xx)*width/2, (y+yy)*height/2])+raw_vector
        for j, current in enumerate(new):
            if d['label'] != current['label']:
                continue
            a,b,c,e=current['box']
            center=np.array([(a+c)*width/2,(b+e)*height/2])
            distance=float(np.linalg.norm(center-projected))
            size=max(4, min((c-a)*width,(e-b)*height))
            similarity=float(np.dot(features[i],features[len(old)+j]))
            if distance > max(5,size*.3) or similarity < .85:
                continue
            # Competing matches are rejected below instead of assigning identities by proximity alone.
            candidates.append((i,j,distance/max(size,1),similarity,vector,spread,len(p)))
    matched = 0
    for i,j,cost,similarity,vector,spread,points in sorted(candidates,key=lambda v:v[2]):
        if sum(c[0]==i for c in candidates)!=1 or sum(c[1]==j for c in candidates)!=1:
            continue
        current=new[j]
        state=classify_motion(vector,spread,points,similarity,current['confidence'])
        normalized = [round(float(vector[0])/width, 5),round(float(vector[1])/height, 5)]
        current['motion']=dict(state=state,reason='evidence_fusion',vector=normalized,
            direction=direction(vector) if state=='moving' else None,
            heading=geographic_heading(normalized,None,calibration) if state=='moving' else None,
            evidence=dict(points=points,appearance_similarity=round(similarity,3),flow_spread_px=round(spread,3),
                          displacement_px=round(float(np.linalg.norm(vector)),3)),
            appearance_model=DINO_MODEL if appearance else 'HSV histogram',gap_seconds=gap,
            track_id=previous_detections.index(old[i]))
        matched+=1
    return dict(status='analyzed',gap_seconds=gap,matched=matched,background_shift_px=round(background_shift,3))


def spatial_grid(detections, size=24, road_only=False):
    grid=np.zeros((size,size),dtype=float)
    for d in detections:
        if d['label'] not in VEHICLES or (road_only and d.get('on_road') is not True):
            continue
        x,y,xx,yy=d['box']; col=min(size-1,max(0,int((x+xx)*.5*size))); row=min(size-1,max(0,int(yy*size)))
        grid[row,col]+=1
    return grid.tolist()


def analyze_scene(image, detections, previous_image=None, previous_detections=(), gap=None,
                  road=None, appearance=None, calibration=None):
    road_grid=None
    if road is not None:
        probabilities=road.predict(image)
        road_grid=cv2.resize(probabilities,(32,24),interpolation=cv2.INTER_AREA)
    for d in detections:
        d['on_road']=None
        if road_grid is not None and d['label'] in VEHICLES:
            x,y,xx,yy=d['box']; px=min(31,max(0,int((x+xx)*16))); py=min(23,max(0,int(min(.999,yy+.015)*24)))
            score=float(road_grid[max(0,py-1):min(24,py+2),max(0,px-1):min(32,px+2)].mean())
            d['road_score']=round(score,3)
            d['on_road']=True if score>=.6 else False if score<=.15 else None
    motion=motion_analysis(previous_image,image,previous_detections,detections,gap,appearance,calibration)
    states={s:sum(d.get('motion',{}).get('state')==s for d in detections if d['label'] in VEHICLES)
            for s in ('moving','stationary','unknown')}
    return dict(version=VERSION,motion=motion,states=states,
                on_road=sum(d.get('on_road') is True for d in detections),
                road_model=ROAD_MODEL if road is not None else None,
                road_mask=np.round(road_grid,2).tolist() if road_grid is not None else None,
                occupancy_grid=spatial_grid(detections),road_grid=spatial_grid(detections,road_only=True),
                classifier='rule-based evidence fusion (not a calibrated probability)')
