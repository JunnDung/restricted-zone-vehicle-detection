import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_portfolio import case_study


class PortfolioTest(unittest.TestCase):
    def test_rejected_candidate_cannot_be_described_as_deployed(self):
        metrics={s:{'mAP50_95':.2} for s in ('mio','visdrone','coco')}
        video={key:{'class_aware':{'f1':.5},'alert_only':{'f1':1},'processing_fps':10} for key in ('baseline','candidate')}
        report={'baseline_test':metrics,'test':metrics,'accepted':False,'parameters':100,'weights_bytes':1000,
                'device':'test GPU','settings':{'imgsz':640,'batch':16},'admission_video':video,'limitations':'Synthetic fixture'}
        text=case_study(report)
        self.assertIn('failed; the previous serving model is retained',text)
        self.assertIn('not field accuracy',text)
        self.assertNotIn('candidate available in the dashboard',text)
        self.assertNotIn('improved equal-source',text)
