import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_large import convert_mio, split_for
from evaluate_events import score
from data_utils import sha256, verify_files


class DataTest(unittest.TestCase):
    def test_parallel_verification_checks_images_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);records=[]
            for i in range(2):
                image=base/f"{i}.jpg";label=base/f"{i}.txt"
                image.write_bytes(b"sample image");label.write_text("1 .5 .5 .2 .2")
                records.append({"image":image.name,"label":label.name,"image_sha256":sha256(image),"label_sha256":sha256(label)})
            progress=[]
            verify_files(base,records,lambda done,total:progress.append((done,total)))
            self.assertEqual(progress,[(2,2)])
            label.write_text("tampered label")
            with self.assertRaisesRegex(ValueError,"Stale dataset content"):
                verify_files(base,records)
    def test_inclusive_coordinates_clipping_and_unknown_type(self):
        labels,ignored=convert_mio([["0","car","10","20","29","39"],["0","motorized_vehicle","0","0","9","9"],["0","pedestrian","0","0","10","10"]],100,100)
        self.assertEqual(labels,[(1,.2,.3,.2,.2)])
        self.assertEqual(ignored,[(0,0,9,9)])
        with self.assertRaises(ValueError):
            convert_mio([["0","invalid","0","0","10","10"]],100,100)

    def test_adjacent_ids_stay_in_the_same_partition(self):
        self.assertEqual(split_for("00001000"),split_for("00001999"))
        self.assertEqual({split_for(str(i*1000)) for i in range(100)},{"train","val","test"})

    def test_event_matching_penalizes_duplicates_and_wrong_classes(self):
        expected=[{"class":"car","zone_id":"z","min_seconds":1,"max_seconds":2}]
        actual=[{"class":"truck","zone_id":"z","confirmed_seconds":1.5}]
        self.assertEqual(score(expected,actual)["f1"],0)
        self.assertEqual(score(expected,actual,False)["f1"],1)
        actual[0]["class"]="car"
        result=score(expected,actual+actual)
        self.assertEqual((result["tp"],result["fp"],result["fn"]),(1,1,0))


if __name__=="__main__":
    unittest.main()
