"""Run with unittest; uses temporary storage and no AI model inference."""
import importlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from zones import ZoneMap


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        os.environ["ZONEGUARD_RUNTIME"] = cls.temp.name
        import app
        cls.module = importlib.reload(app)
        cls.client = TestClient(cls.module.app)
        cls.client.__enter__()
        cls.map = json.loads((Path(__file__).resolve().parents[1] / "config/parking.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None)
        cls.temp.cleanup()
        os.environ.pop("ZONEGUARD_RUNTIME", None)

    def test_boundary_validation(self):
        self.assertEqual(self.client.get("/api/health").status_code,200)
        self.assertEqual(self.client.post("/api/jobs",json={"source_id":"../../private","zone_map":self.map}).status_code,422)
        self.assertEqual(self.client.put("/api/map",json={"zones":[]}).status_code,422)
        self.assertEqual(self.client.get("/api/jobs/not-a-job/files/config.json").status_code,404)
        self.assertEqual(self.client.post("/api/jobs",json={},headers={"Origin":"https://attacker.example"}).status_code,403)
        self.assertEqual(self.client.get("/api/health",headers={"Host":"attacker.example"}).status_code,400)
        self.assertEqual(self.client.post("/api/sources",files={"file":("fake.mp4",b"not video","video/mp4")}).status_code,422)
        self.assertEqual(list((Path(self.temp.name)/"sources").glob("*/source.json")),[])

    def test_maps_are_per_source(self):
        modified = ZoneMap.model_validate(self.map).model_dump()
        modified["zones"][0]["name"] = "Riêng nguồn đường phố"
        self.assertEqual(self.client.put("/api/map?source_id=street",json=modified).status_code,200)
        self.assertEqual(self.client.get("/api/map?source_id=street").json()["zones"][0]["name"],modified["zones"][0]["name"])
        self.assertNotEqual(self.client.get("/api/map?source_id=parking").json()["zones"][0]["name"],modified["zones"][0]["name"])

    def test_worker_busy_stop_immutable_snapshot_and_delete(self):
        entered = threading.Event()
        def fake_analyze(source, output, zone_map, **kwargs):
            output.mkdir(parents=True)
            (output/"config.json").write_text(json.dumps(zone_map.model_dump()))
            entered.set()
            kwargs["stop"].__self__.wait(3)
            return {"phase":"stopped","event_count":0,"progress":1}
        with patch.object(self.module,"analyze",side_effect=fake_analyze):
            first=self.client.post("/api/jobs",json={"source_id":"parking","zone_map":self.map})
            self.assertEqual(first.status_code,201)
            job_id=first.json()["id"]
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.client.post("/api/jobs",json={"source_id":"parking","zone_map":self.map}).status_code,409)
            self.assertEqual(self.client.delete(f"/api/jobs/{job_id}").status_code,409)
            modified=ZoneMap.model_validate(self.map).model_dump();modified["zones"][0]["name"]="Updated after start"
            self.client.put("/api/map?source_id=parking",json=modified)
            original=self.client.get(f"/api/jobs/{job_id}").json()["request"]["zone_map"]["zones"][0]["name"]
            self.assertEqual(original,self.map["zones"][0]["name"])
            self.assertEqual(self.client.post(f"/api/jobs/{job_id}/stop").status_code,200)
            self.module.manager.thread.join(3)
            self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json()["status"],"stopped")
            self.assertEqual(self.client.get(f"/api/jobs/{job_id}/files/config.json").status_code,200)
            self.assertEqual(self.client.get(f"/api/jobs/{job_id}/files/zoneguard.sqlite3").status_code,404)
            self.assertEqual(self.client.delete(f"/api/jobs/{job_id}").status_code,200)
            self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code,404)

    def test_restart_retains_events_and_recovers_pending(self):
        from store import Store
        folder=Path(self.temp.name)/"restart-test"
        store=Store(folder)
        store.save_job({"id":"pending","created":"2026-10-06","status":"running"})
        event={"id":"e1","confirmed_seconds":2.5,"zone_id":"gate","class":"car"}
        store.add_event("pending",event)
        restarted=Store(folder);restarted.recover()
        self.assertEqual(restarted.job("pending")["status"],"error")
        self.assertEqual(restarted.events("pending"),[event])

    def test_training_reserves_gpu_and_auto_uses_cpu(self):
        root=Path(self.temp.name)/"training-test"
        (root/"reports").mkdir(parents=True)
        (root/"reports/training-live.json").write_text('{"status":"training"}')
        def fake_analyze(source, output, zone_map, **kwargs):
            self.assertEqual(kwargs["device"],"cpu")
            return {"phase":"completed","event_count":0,"progress":1}
        with patch.object(self.module,"ROOT",root), patch.object(self.module,"analyze",side_effect=fake_analyze):
            request={"source_id":"parking","zone_map":self.map,"device":"0"}
            self.assertEqual(self.client.post("/api/jobs",json=request).status_code,409)
            request["device"]="auto"
            response=self.client.post("/api/jobs",json=request)
            self.assertEqual(response.status_code,201)
            self.module.manager.thread.join(3)
            job=self.client.get("/api/jobs/"+response.json()["id"]).json()
            self.assertEqual(job["request"]["device"],"cpu")
            self.assertEqual(job["status"],"completed")


if __name__ == "__main__":
    unittest.main()
