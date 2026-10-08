import sys
import struct
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_compact import balanced, coco_labels
from data_utils import memory_headroom_mb


class CompactTest(unittest.TestCase):
    def test_windows_memory_check_catches_commit_pressure_with_free_ram(self):
        def status(buffer):
            mb=1024**2
            struct.pack_into("<II7Q",buffer,0,64,0,8192*mb,4096*mb,16384*mb,512*mb,0,0,0)
            return 1
        kernel=SimpleNamespace(GlobalMemoryStatusEx=status)
        with patch("data_utils.sys.platform","win32"),patch("data_utils.ctypes.windll",SimpleNamespace(kernel32=kernel),create=True):
            self.assertEqual(memory_headroom_mb(),512)
        with patch("data_utils.sys.platform","linux"):
            self.assertIsNone(memory_headroom_mb())

    def test_selection_keeps_rare_classes_unique_and_deterministic(self):
        records=[{"image":f"car_{i}.jpg","classes":[1]} for i in range(30)]
        records += [{"image":f"rare_{c}.jpg","classes":[c]} for c in (0,2,3,4)]
        selected=balanced(records,5)
        self.assertEqual({c for r in selected for c in r["classes"]},set(range(5)))
        self.assertEqual(len({r["image"] for r in selected}),5)
        self.assertEqual(selected,balanced(list(reversed(records)),5))
        backgrounds=[{"image":f"empty_{i}.jpg","classes":[]} for i in range(10)]
        self.assertEqual(sum(not r["classes"] for r in balanced(records+backgrounds,20)),1)

    def test_coco_mapping_clips_and_deduplicates_without_fake_crowd_boxes(self):
        car={"category_id":3,"bbox":[-10,20,30,20]}
        labels=coco_labels([car,car,{"category_id":1,"bbox":[0,0,10,10]}],100,100)
        self.assertEqual(labels,[(1,.1,.3,.2,.2)])
        for category,expected in ((4,0),(3,1),(8,2),(6,3),(2,4)):
            self.assertEqual(coco_labels([{"category_id":category,"bbox":[0,0,10,10]}],100,100)[0][0],expected)
        with self.assertRaises(ValueError):
            coco_labels([{**car,"iscrowd":1}],100,100)


if __name__=="__main__":
    unittest.main()
