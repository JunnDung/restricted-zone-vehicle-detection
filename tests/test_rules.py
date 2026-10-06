import unittest
from zones import IntrusionEngine, ZoneMap


def config(**overrides):
    zone = dict(id="road", name="Đường cấm", points=[(.2,.2),(.8,.2),(.8,.8),(.2,.8)], dwell_seconds=.3, exit_seconds=.2)
    zone.update(overrides)
    return ZoneMap.model_validate({"zones": [zone]})


def det(x=.5, cls="car", track=1):
    return {"track_id": track, "class": cls, "point": (x,.5), "confidence": .9}


class RulesTest(unittest.TestCase):
    def test_visit_exit_and_reentry(self):
        e = IntrusionEngine(config())
        for t in (0, .1, .2):
            self.assertEqual(e.update([det(.1)], t), [])
        self.assertEqual(e.update([det()], .3), [])
        self.assertEqual(e.update([det()], .4), [])
        events = e.update([det()], .6)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "entry")
        self.assertEqual(e.update([det()], .7), [])
        for t in (.8, 1.0):
            e.update([det(.1)], t)
        for t in (1.1, 1.2):
            self.assertEqual(e.update([det()], t), [])
        self.assertEqual(len(e.update([det()], 1.4)), 1)

    def test_initial_presence_and_entry_only(self):
        for alert_initial, expected in ((True, 1), (False, 0)):
            e = IntrusionEngine(config(alert_initial=alert_initial))
            events = sum((e.update([det()], t) for t in (0, .1, .3)), [])
            self.assertEqual(len(events), expected)

    def test_gap_does_not_count_as_dwell(self):
        e = IntrusionEngine(config())
        e.update([det()], 0)
        self.assertEqual(e.update([det()], 2), [])
        self.assertEqual(e.update([det()], 2.1), [])
        self.assertEqual(len(e.update([det()], 2.3)), 1)

    def test_margin_class_and_independent_zones(self):
        e = IntrusionEngine(config(classes=["truck"]))
        for t in (0, .1, .3, 1):
            self.assertEqual(e.update([det(cls="car"), det(.201, "truck", 2)], t), [])
        m = config()
        m.zones.append(m.zones[0].model_copy(update={"id": "second"}))
        e = IntrusionEngine(m)
        e.update([det()], 0)
        e.update([det()], .1)
        self.assertEqual(len(e.update([det()], .3)), 2)

    def test_exit_gap_does_not_rearm(self):
        e = IntrusionEngine(config())
        for t in (0,.1,.3):
            e.update([det()], t)
        e.update([det(.1)], .4)
        e.update([det(.1)], 2)
        self.assertEqual(e.update([det()], 2.1), [])

    def test_invalid_polygons_and_time(self):
        for points in ([(0,0),(1,1),(0,1),(1,0)], [(0,0),(0,0),(1,1)], [(0,0),(.5,.5),(1,1)], [(0,0),(2,0),(1,1)]):
            with self.assertRaises(ValueError):
                config(points=points)
        with self.assertRaises(ValueError):
            config(dwell_seconds=float("nan"))
        e = IntrusionEngine(config())
        e.update([], 1)
        with self.assertRaises(ValueError):
            e.update([], .5)


if __name__ == "__main__":
    unittest.main()
