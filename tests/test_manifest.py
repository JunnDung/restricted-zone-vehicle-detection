import gzip
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]


class ManifestTest(unittest.TestCase):
    def test_archived_manifest_matches_the_run_and_has_no_exact_split_duplicates(self):
        raw=gzip.decompress((ROOT/'data/compact-manifest.json.gz').read_bytes())
        report=json.loads((ROOT/'reports/compact-dataset.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),report['manifest_sha256'])
        manifest=json.loads(raw);files=manifest['files']
        self.assertEqual(len(files),15000)
        self.assertEqual(len({r['original_sha256'] for r in files}),len(files))
        self.assertEqual(dict(Counter(r['source']+'/'+r['split'] for r in files)),manifest['usable_images'])
        for record in files:
            for key in ('image','label'):
                path=PurePosixPath(record[key])
                self.assertFalse(path.is_absolute())
                self.assertNotIn('..',path.parts)
