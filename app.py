"""Local dashboard/API: one process, one analysis worker, persistent SQLite history."""
import json
import os
import re
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from pipeline import ROOT, analyze, atomic_json, inspect_video
from store import Store
from zones import LABELS, ZoneMap

RUNTIME = Path(os.environ.get("ZONEGUARD_RUNTIME", ROOT / "runtime")).resolve()
SAMPLES = {"parking": ("Bãi xe · Intel", ROOT / "data/samples/car-detection.mp4"),
           "street": ("Đường phố · Intel", ROOT / "data/samples/person-bicycle-car-detection.mp4")}


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source_id: str = Field(pattern=r"^(parking|street|camera:[0-5]|[a-f0-9]{32})$")
    zone_map: ZoneMap
    model_key: Literal["vehicle", "coco"] = "vehicle"
    confidence: float = Field(default=.3, ge=.05, le=.9)
    imgsz: Literal[640, 960] = 640
    device: Literal["auto", "cpu", "0"] = "auto"
    max_seconds: int = Field(default=900, ge=1, le=900)


class Manager:
    def __init__(self):
        self.store = Store(RUNTIME)
        self.lock = threading.Lock()
        self.active = None
        self.stop = threading.Event()
        self.thread = None
        (RUNTIME / "sources").mkdir(exist_ok=True)
        (RUNTIME / "jobs").mkdir(exist_ok=True)

    def source(self, source_id):
        if source_id in SAMPLES:
            name, path = SAMPLES[source_id]
            return name, path
        if re.fullmatch(r"camera:[0-5]", source_id):
            return "Camera USB " + source_id[-1], int(source_id[-1])
        if not re.fullmatch(r"[a-f0-9]{32}", source_id):
            raise HTTPException(404, "Không có nguồn này.")
        meta = RUNTIME / "sources" / source_id / "source.json"
        if not meta.exists():
            raise HTTPException(404, "Không có video này.")
        info = json.loads(meta.read_text(encoding="utf-8"))
        return info["name"], meta.parent / info["file"]

    def start(self, settings):
        name, source = self.source(settings.source_id)
        with self.lock:
            if self.active:
                raise HTTPException(409, "Đang xử lý một phiên. Hãy dừng hoặc chờ hoàn tất.")
            job_id = uuid.uuid4().hex
            self.stop.clear()
            job = {"id": job_id, "created": datetime.now(timezone.utc).isoformat(), "status": "queued", "name": name,
                   "request": settings.model_dump(), "progress": 0, "event_count": 0}
            self.store.save_job(job)
            self.active = job_id
            self.thread = threading.Thread(target=self.run, args=(job,source,settings), daemon=True)
            self.thread.start()
            return job.copy()

    def run(self, job, source, settings):
        def progress(status):
            job.update(status)
            job["status"] = "running"
            self.store.save_job(job)
        try:
            job["status"] = "running"
            self.store.save_job(job)
            result = analyze(source, RUNTIME / "jobs" / job["id"], settings.zone_map, model_key=settings.model_key, confidence=settings.confidence,
                             imgsz=settings.imgsz, device=settings.device, max_seconds=settings.max_seconds, progress=progress,
                             on_event=lambda event: self.store.add_event(job["id"],event), stop=self.stop.is_set)
            job.update({k:v for k,v in result.items() if k != "events"}, status=result["phase"])
        except Exception as exc:
            job.update(status="error", error=str(exc), event_count=len(self.store.events(job["id"])))
        finally:
            job["finished"] = datetime.now(timezone.utc).isoformat()
            self.store.save_job(job)
            with self.lock:
                self.active = None


manager = Manager()


@asynccontextmanager
async def lifespan(app):
    manager.store.recover()
    yield
    manager.stop.set()
    if manager.thread:
        manager.thread.join(timeout=10)


app = FastAPI(title="ZoneGuard · Phương tiện vùng cấm", version="1.0.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1","localhost","testserver"])


@app.middleware("http")
async def local_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method not in ("GET","HEAD","OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Yêu cầu phải đến từ cùng địa chỉ dashboard."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
    return response


@app.get("/api/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "active_job": manager.active, "labels": LABELS, "version": "1.0.0"}


@app.get("/api/sources")
def sources():
    entries = [{"id": sid, "name": value[0], "kind": "sample"} for sid,value in SAMPLES.items()]
    for meta in (RUNTIME / "sources").glob("*/source.json"):
        info = json.loads(meta.read_text(encoding="utf-8"))
        entries.append({"id": meta.parent.name, "name": info["name"], "kind": "upload"})
    return entries


@app.post("/api/sources", status_code=201)
async def upload(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in (".mp4",".avi",".mov",".mkv",".webm"):
        raise HTTPException(422, "Chọn video MP4, AVI, MOV, MKV hoặc WebM.")
    sid = uuid.uuid4().hex
    folder = RUNTIME / "sources" / sid
    folder.mkdir()
    target = folder / ("video" + extension)
    try:
        size = 0
        with target.open("wb") as dest:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > 500*1024*1024:
                    raise ValueError("Video tối đa 500 MB.")
                dest.write(chunk)
        metadata, frame = inspect_video(target)
        info = {"id": sid, "name": (file.filename or "video")[:160], "file": target.name, "metadata": metadata}
        atomic_json(folder / "source.json", info)
        if not cv2.imwrite(str(folder / "preview.jpg"), frame):
            raise ValueError("Không tạo được ảnh xem trước.")
        return info
    except (ValueError, OSError) as exc:
        if folder.resolve().parent == (RUNTIME / "sources").resolve():
            shutil.rmtree(folder)
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@app.get("/api/sources/{source_id}/preview")
def source_preview(source_id: str):
    _, source = manager.source(source_id)
    if isinstance(source,int):
        with manager.lock:
            if manager.active:
                raise HTTPException(409,"Dừng phiên hiện tại trước khi xem camera.")
            cap = cv2.VideoCapture(source)
            try:
                ok, frame = cap.read()
                if not ok:
                    raise HTTPException(422,"Không đọc được camera. Kiểm tra chỉ số và quyền camera.")
            finally:
                cap.release()
        path = RUNTIME / "sources" / f"camera-{source}.jpg"
    else:
        path = RUNTIME / "sources" / f"{source_id}.jpg" if source_id in SAMPLES else source.parent / "preview.jpg"
        if path.exists():
            return FileResponse(path)
        _, frame = inspect_video(source)
    cv2.imwrite(str(path), frame)
    return FileResponse(path)


def map_path(source_id):
    manager.source(source_id)
    folder = RUNTIME / "maps"
    folder.mkdir(exist_ok=True)
    return folder / (source_id.replace(":", "-") + ".json")


@app.get("/api/map")
def get_map(source_id: str = "parking"):
    path = map_path(source_id)
    return json.loads((path if path.exists() else ROOT / "config/parking.json").read_text(encoding="utf-8"))


@app.put("/api/map")
def save_map(zone_map: ZoneMap, source_id: str = "parking"):
    atomic_json(map_path(source_id),zone_map.model_dump())
    return zone_map


@app.get("/api/jobs")
def jobs():
    return manager.store.jobs()


@app.post("/api/jobs", status_code=201)
def start_job(settings: JobRequest):
    return manager.start(settings)


def require_job(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}",job_id) or not (job := manager.store.job(job_id)):
        raise HTTPException(404,"Không có phiên này.")
    return job


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    return require_job(job_id)


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    require_job(job_id)
    with manager.lock:
        if manager.active != job_id:
            raise HTTPException(409,"Phiên đã dừng.")
        manager.stop.set()
    return {"stopping": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    require_job(job_id)
    with manager.lock:
        if manager.active == job_id:
            raise HTTPException(409,"Dừng phiên trước khi xóa.")
        folder = (RUNTIME / "jobs" / job_id).resolve()
        if folder.parent != (RUNTIME / "jobs").resolve():
            raise HTTPException(422,"Đường dẫn không hợp lệ.")
        if folder.exists():
            shutil.rmtree(folder)
        manager.store.delete_job(job_id)
    return {"deleted": True}


@app.get("/api/jobs/{job_id}/events")
def events(job_id: str, zone_id: str | None = Query(default=None), vehicle_class: str | None = Query(default=None)):
    require_job(job_id)
    return [e for e in manager.store.events(job_id) if (zone_id is None or e["zone_id"] == zone_id) and (vehicle_class is None or e["class"] == vehicle_class)]


@app.get("/api/jobs/{job_id}/files/{filename}")
def artifact(job_id: str, filename: str):
    require_job(job_id)
    if filename not in ("preview.jpg","annotated.mp4","events.csv","result.json","config.json","tracker.yaml"):
        raise HTTPException(404,"Không có tệp này.")
    path = RUNTIME / "jobs" / job_id / filename
    if not path.is_file():
        raise HTTPException(404,"Tệp chưa sẵn sàng.")
    return FileResponse(path, filename=filename if filename.endswith(("csv","json","yaml")) else None)


@app.get("/api/jobs/{job_id}/evidence/{event_id}")
def evidence(job_id: str, event_id: str):
    require_job(job_id)
    if not re.fullmatch(r"[a-f0-9]{32}",event_id):
        raise HTTPException(404)
    path = RUNTIME / "jobs" / job_id / "evidence" / (event_id+".jpg")
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="dashboard")
