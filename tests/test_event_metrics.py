import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_events
from zones import ZoneMap


class EventMetricsTest(unittest.TestCase):
    def setUp(self):
        self.zone_map=ZoneMap.model_validate({'zones':[{'id':'z','name':'Gate','points':[[0,0],[1,0],[1,1],[0,1]]}]})
        self.reference={'source_sha256':'same','events':[{'class':'car','zone_id':'z','min_seconds':1,'max_seconds':2}]}
        self.result={'source_sha256':'same','phase':'completed','metadata':{'fps':10,'frames':18000},
                     'processed_frames':18000,'elapsed_seconds':2,'settings':{'zone_map':self.zone_map.model_dump(mode='json')},
                     'events':[{'class':'car','zone_id':'z','confirmed_seconds':1.5}]}

    def test_overlapping_windows_do_not_invent_false_alerts(self):
        expected=[{'class':'car','zone_id':'z','min_seconds':0,'max_seconds':10},
                  {'class':'car','zone_id':'z','min_seconds':0,'max_seconds':6}]
        actual=[{'class':'car','zone_id':'z','confirmed_seconds':seconds} for seconds in (7,4.5)]
        metrics=evaluate_events.score(expected,actual,duration_seconds=12)
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(2,0,0))
        self.assertEqual(metrics['false_positives_per_hour'],0)

    def test_duplicate_alert_uses_video_hours_not_processing_time(self):
        self.result['events']*=2
        metrics=evaluate_events.evaluate(self.reference,self.result,self.zone_map)['alert_only']
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(1,1,0))
        self.assertEqual((metrics['precision'],metrics['recall']),(.5,1))
        self.assertEqual(metrics['observed_seconds'],1800)
        self.assertEqual(metrics['false_positives_per_hour'],2)

    def test_wrong_class_is_separate_from_a_false_zone_alert(self):
        self.result['events'][0]['class']='truck'
        metrics=evaluate_events.evaluate(self.reference,self.result,self.zone_map)
        self.assertEqual(metrics['alert_only']['false_positives_per_hour'],0)
        self.assertEqual(metrics['alert_only']['precision'],1)
        self.assertEqual(metrics['class_aware']['false_positives_per_hour'],2)
        self.assertEqual(metrics['class_aware']['recall'],0)

    def test_empty_and_negative_video_still_has_an_hourly_rate(self):
        self.result['events']=[]
        metrics=evaluate_events.evaluate(self.reference,self.result,self.zone_map)['alert_only']
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']),(0,0,1))
        self.assertEqual((metrics['precision'],metrics['recall'],metrics['false_positives_per_hour']),(0,0,0))
        self.reference['events']=[]
        self.result['events']=[{'class':'car','zone_id':'z','confirmed_seconds':1.5}]
        self.assertEqual(evaluate_events.evaluate(self.reference,self.result,self.zone_map)['alert_only']['false_positives_per_hour'],2)

    def test_refuses_partial_video_source_or_zone_mismatch(self):
        variants=[{'phase':'stopped'},{'processed_frames':100},{'source_sha256':'different'},
                  {'settings':{'zone_map':{'zones':[{'id':'z','name':'Changed','points':[[0,0],[.5,0],[.5,.5]]}]}}}]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                evaluate_events.evaluate(self.reference,self.result|variant,self.zone_map)

    def test_refuses_invalid_exposure_and_events_outside_observation(self):
        for fps in (0,-1,float('inf'),float('nan')):
            result=copy.deepcopy(self.result);result['metadata']['fps']=fps
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                evaluate_events.evaluate(self.reference,result,self.zone_map)
        for seconds in (-1,1801,float('nan')):
            result=copy.deepcopy(self.result);result['events'][0]['confirmed_seconds']=seconds
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                evaluate_events.evaluate(self.reference,result,self.zone_map)

    def test_score_without_exposure_does_not_invent_hourly_rate(self):
        metrics=evaluate_events.score([],[])
        self.assertIsNone(metrics['observed_seconds'])
        self.assertIsNone(metrics['false_positives_per_hour'])


if __name__=='__main__':
    unittest.main()
