import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from restore_compact import archived_writer


class ColabRecoveryTest(unittest.TestCase):
    def test_rebuild_preserves_windows_text_hashes_on_linux(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.json'
            archived_writer(path,json.dumps({'files':['a','b']},indent=2),encoding='utf-8')
            self.assertIn(b'\r\n',path.read_bytes())
            self.assertNotIn(b'\n',path.read_bytes().replace(b'\r\n',b''))
            self.assertEqual(json.loads(path.read_text()),{'files':['a','b']})
