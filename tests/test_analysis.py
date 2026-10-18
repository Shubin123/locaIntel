import unittest
import numpy as np
import cv2
from PIL import Image
from pipeline.analysis import analyze_scene, classify_motion, geographic_heading, motion_analysis
from pipeline.patterns import record, summarize


def pair(shift=6):
    rng=np.random.default_rng(42)
    background=rng.integers(0,170,(180,240,3),dtype=np.uint8)
    vehicle=rng.integers(80,255,(40,40,3),dtype=np.uint8)
    a=background.copy();b=background.copy()
    a[80:120,80:120]=vehicle;b[80:120,80+shift:120+shift]=vehicle
    b[170,230]=[255,255,255] # Independent scene change avoids identical-frame abstention.
    old=[dict(label='car',confidence=.95,box=[80/240,80/180,120/240,120/180])]
    new=[dict(label='car',confidence=.95,box=[(80+shift)/240,80/180,(120+shift)/240,120/180])]
    return Image.fromarray(a),Image.fromarray(b),old,new


class MotionTests(unittest.TestCase):
    def test_sparse_snapshots_abstain(self):
        a,b,old,new=pair()
        result=motion_analysis(a,b,old,new,120)
        self.assertEqual(result['status'],'insufficient_cadence')
        self.assertEqual(new[0]['motion']['state'],'unknown')

    def test_translated_vehicle_direction(self):
        a,b,old,new=pair()
        result=motion_analysis(a,b,old,new,.1)
        self.assertEqual(result['status'],'analyzed')
        self.assertEqual(new[0]['motion']['state'],'moving')
        self.assertEqual(new[0]['motion']['direction'],'right')
        self.assertIsNone(new[0]['motion']['heading'])

    def test_stationary_vehicle(self):
        a,b,old,new=pair(0)
        motion_analysis(a,b,old,new,.1)
        self.assertEqual(new[0]['motion']['state'],'stationary')

    def test_duplicate_frame_is_not_stationary_evidence(self):
        a,b,old,new=pair(0)
        result=motion_analysis(a,a,old,new,.1)
        self.assertEqual(result['status'],'duplicate_frame')
        self.assertEqual(new[0]['motion']['state'],'unknown')

    def test_camera_shift_abstains(self):
        a,b,old,new=pair()
        shifted=Image.fromarray(cv2.warpAffine(np.array(a),np.float32([[1,0,6],[0,1,0]]),a.size))
        result=motion_analysis(a,shifted,old,new,.1)
        self.assertEqual(result['status'],'camera_moved')
        self.assertEqual(new[0]['motion']['state'],'unknown')

    def test_weak_and_ambiguous_evidence_abstains(self):
        self.assertEqual(classify_motion([8,0],0,2,.99,.9),'unknown')
        self.assertEqual(classify_motion([8,0],0,12,.5,.9),'unknown')
        self.assertEqual(classify_motion([8,0],8,12,.99,.9),'unknown')
        a,b,old,new=pair()
        old.append(dict(old[0]))
        motion_analysis(a,b,old,new,.1)
        self.assertEqual(new[0]['motion']['state'],'unknown')

    def test_compass_requires_calibration(self):
        self.assertIsNone(geographic_heading([0,-.1],None,None))
        calibration={'road_axis':[[.5,.9],[.5,.1]],'away_bearing':90}
        self.assertEqual(geographic_heading([0,-.1],None,calibration),90)
        self.assertEqual(geographic_heading([0,.1],None,calibration),270)

    def test_road_classification_and_density(self):
        a,b,old,new=pair()
        class Road:
            def predict(self,image):
                return np.ones((24,32))*.9
        result=analyze_scene(b,new,road=Road())
        self.assertTrue(new[0]['on_road'])
        self.assertEqual(result['on_road'],1)
        self.assertEqual(sum(map(sum,result['occupancy_grid'])),1)
        self.assertEqual(sum(map(sum,result['road_grid'])),1)

    def test_road_model_absence_is_unknown(self):
        a,b,old,new=pair()
        analyze_scene(b,new)
        self.assertIsNone(new[0]['on_road'])


class PatternTests(unittest.TestCase):
    def test_dedup_prune_and_exposure_normalization(self):
        now=100000
        history={'expired':[dict(hash='x',time=now-90000,count=4,points=[])]}
        state=dict(hash='a',detections=[dict(label='car',box=[.1,.1,.2,.2],on_road=True)])
        record(history,'1',state,now-10)
        record(history,'1',state,now)
        self.assertNotIn('expired',history)
        self.assertEqual(len(history['1']),1)
        state=dict(hash='b',detections=[])
        record(history,'1',state,now)
        result=summarize(history,now)
        self.assertEqual(result['cameras']['1']['samples'],2)
        self.assertEqual(result['cameras']['1']['mean_vehicles'],.5)
        self.assertAlmostEqual(sum(map(sum,result['cameras']['1']['occupancy_grid'])),.5)
