import unittest

from classification import TrackLabel


class ClassificationTest(unittest.TestCase):
    def test_conflicting_white_car_is_not_reported_as_a_definite_truck(self):
        label=TrackLabel()
        observations=[('car',.4455),('car',.6431),('truck',.5768),('truck',.3768),('truck',.6446),('truck',.5538),('truck',.5967),('car',.5233)]
        for i,(category,confidence) in enumerate(observations):
            result=label.update(category,confidence,i*.08)
        self.assertEqual(result['class'],'unknown')
        self.assertEqual(result['detection_confidence'],.5233)
        self.assertLess(result['class_support'],.75)

    def test_consistent_real_truck_retains_its_class_and_own_score(self):
        label=TrackLabel()
        for i,confidence in enumerate((.8,.9,.7)):
            result=label.update('truck',confidence,i*.1)
        self.assertEqual(result['class'],'truck')
        self.assertAlmostEqual(result['confidence'],.8)
        self.assertEqual(result['class_support'],1)

    def test_strong_consensus_tolerates_one_disagreement_without_borrowing_its_score(self):
        label=TrackLabel()
        for i in range(5):label.update('car',.9,i*.1)
        result=label.update('truck',.4,.6)
        self.assertEqual(result['class'],'car')
        self.assertAlmostEqual(result['confidence'],.9)
        self.assertEqual(result['detection_confidence'],.4)

    def test_old_or_weak_votes_do_not_lock_a_new_label(self):
        label=TrackLabel()
        for i in range(15):label.update('truck',.9,i*.1)
        for i in range(3):result=label.update('car',.8,3+i*.1)
        self.assertEqual(result['class'],'car')
        result=label.update('truck',.1,3.3)
        self.assertEqual(result['class'],'unknown')
        result=label.update('car',.8,3.4)
        self.assertEqual(result['class'],'car')
