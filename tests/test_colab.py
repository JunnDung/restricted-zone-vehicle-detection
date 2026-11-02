import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from restore_compact import archived_writer
from train_large import metrics_dict, detector_admission


class ColabRecoveryTest(unittest.TestCase):
    def test_average_gain_cannot_hide_forgetting_one_domain(self):
        baseline={s:{'mAP50_95':.3} for s in ('mio','visdrone','coco')}
        candidate={'mio':{'mAP50_95':.5},'visdrone':{'mAP50_95':.25},'coco':{'mAP50_95':.5}}
        self.assertFalse(detector_admission(baseline,candidate)[0])
        candidate['visdrone']['mAP50_95']=.29
        self.assertTrue(detector_admission(baseline,candidate)[0])
    def test_metrics_keep_accuracy_and_latency_separate(self):
        box=SimpleNamespace(mp=.8,mr=.7,map50=.75,map=.5,ap50=[.75],ap=[.5],ap_class_index=[1])
        metrics=SimpleNamespace(box=box,names={1:'car'},speed={'preprocess':1.,'inference':5.,'postprocess':2.})
        result=metrics_dict(metrics)
        self.assertEqual(result['mAP50_95'],.5)
        self.assertEqual(result['speed_ms_per_image']['inference'],5.)
        self.assertEqual(result['per_class'][0]['class'],'car')
    def test_rebuild_preserves_windows_text_hashes_on_linux(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json'
            archived_writer(path,json.dumps({'files':['a','b']},indent=2),encoding='utf-8')
            self.assertIn(b'\r\n',path.read_bytes())
            self.assertNotIn(b'\n',path.read_bytes().replace(b'\r\n',b''))
            self.assertEqual(json.loads(path.read_text()),{'files':['a','b']})
