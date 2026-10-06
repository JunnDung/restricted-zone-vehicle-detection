"""YOLO + ByteTrack + polygon rules, shared by web and CLI."""
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from zones import CLASSES, IntrusionEngine, ZoneMap

ROOT = Path(__file__).resolve().parent
COCO_MAP = {3: "motorcycle", 2: "car", 7: "truck", 5: "bus", 1: "other"}
MODELS = {"vehicle": ROOT / "models/vehicle-yolo11n.pt", "coco": ROOT / "models/yolo11n.pt"}


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not cap.isOpened() or not all(math.isfinite(v) for v in (fps, frames, w, h)):
            raise ValueError("Không mở được video hợp lệ.")
        if not 1 <= fps <= 120 or not 1 <= frames <= 54000 or frames / fps > 900:
            raise ValueError("Video tối đa 15 phút, 54.000 khung; FPS phải từ 1–120.")
        if min(w, h) < 32 or max(w, h) > 4096 or w*h > 9000000:
            raise ValueError("Video tối đa 4096 px mỗi chiều và 9 megapixel.")
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Không giải mã được khung đầu.")
        return {"fps": fps, "frames": int(frames), "width": int(w), "height": int(h), "duration_seconds": round(frames/fps, 3)}, frame
    finally:
        cap.release()


def draw(frame, zone_map, detections, active, seconds, count):
    h, w = frame.shape[:2]
    canvas = frame.copy()
    overlay = frame.copy()
    for z in zone_map.zones:
        points = np.array([(int(x*w), int(y*h)) for x,y in z.points], dtype=np.int32)
        color = tuple(int(z.color[i:i+2],16) for i in (5,3,1))
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(canvas, [points], True, color, 2)
        cv2.putText(canvas, z.id, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    canvas = cv2.addWeighted(overlay, .14, canvas, .86, 0)
    for d in detections:
        x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
        alert = any((d["track_id"], z.id) in active for z in zone_map.zones)
        color = (80,80,255) if alert else (150,225,50)
        cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 2)
        px,py = d["point"]
        cv2.circle(canvas, (int(px*w),int(py*h)), 5, color, -1)
        cv2.putText(canvas, f"#{d['track_id']} {d['class']} {d['confidence']:.2f}", (x1,max(20,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"ZoneGuard | {seconds:.1f}s | alerts: {count}", (15,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2, cv2.LINE_AA)
    return canvas


def analyze(source, output, zone_map, *, model_key="vehicle", confidence=.3, imgsz=640, device="auto", max_seconds=900, progress=None, on_event=None, stop=None):
    if model_key not in MODELS or not .05 <= confidence <= .9 or imgsz not in (640,960) or not 1 <= max_seconds <= 900:
        raise ValueError("Cấu hình phân tích không hợp lệ.")
    if device not in ("auto", "cpu", "0"):
        raise ValueError("Thiết bị phải là auto, cpu hoặc 0.")
    zone_map = ZoneMap.model_validate(zone_map) if isinstance(zone_map, dict) else ZoneMap.model_validate(zone_map.model_dump())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise ValueError("Phiên đã tồn tại; chọn thư mục khác để tránh ghi đè.")
    live = isinstance(source, int)
    metadata = {"live": True} if live else inspect_video(source)[0]
    weights = MODELS[model_key]
    if not weights.is_file():
        raise ValueError("Thiếu trọng số trong models/.")
    settings = {"zone_map": zone_map.model_dump(), "model_key": model_key, "confidence": confidence, "imgsz": imgsz, "device": device, "max_seconds": max_seconds}
    atomic_json(output / "config.json", settings)
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    model = YOLO(str(weights))  # Fresh model => no tracker state across sources.
    coco = len(model.names) == 80
    if not coco and list(model.names.values()) != CLASSES:
        raise ValueError("Checkpoint cần 5 lớp motorcycle/car/truck/bus/other hoặc COCO 80 lớp.")
    tracker = yaml.safe_load((ROOT / "config/bytetrack.yaml").read_text())
    tracker.update(track_high_thresh=confidence, new_track_thresh=confidence, track_low_thresh=min(.1,confidence))
    tracker_path = output / "tracker.yaml"
    tracker_path.write_text(yaml.safe_dump(tracker), encoding="utf-8")
    engine = IntrusionEngine(zone_map)
    events, votes, last_seen = [], defaultdict(lambda: defaultdict(float)), {}
    cap = cv2.VideoCapture(source if live else str(source))
    writer = None
    raw = output / "raw.mp4"
    start, last_progress = time.monotonic(), 0.0
    frames, inference = 0, 0.0
    status = {"metadata": metadata, "device": device, "model": weights.name, "model_sha256": sha256(weights), "progress": 0, "event_count": 0, "phase": "detecting"}
    stopped = False
    try:
        if not cap.isOpened():
            raise ValueError("Không mở được camera/video. Kiểm tra thiết bị và quyền truy cập.")
        if live:
            fps = cap.get(cv2.CAP_PROP_FPS)
            metadata["fps"] = fps if math.isfinite(fps) and 1 <= fps <= 120 else 25.0
        while True:
            if stop and stop():
                stopped = True
                break
            seconds = time.monotonic() - start if live else frames / metadata["fps"]
            if seconds >= max_seconds:
                stopped = live or frames < metadata.get("frames", 0)
                break
            ok, frame = cap.read()
            if not ok:
                if live:
                    raise ValueError("Mất tín hiệu camera; các bằng chứng đã ghi được giữ lại.")
                if frames < metadata["frames"] - 2:
                    raise ValueError("Video dừng giải mã sớm; không thể coi phiên đã hoàn tất.")
                break
            if frames >= 54000:
                raise ValueError("Vượt giới hạn 54.000 khung.")
            h,w = frame.shape[:2]
            if max(h,w) > 4096 or h*w > 9000000:
                raise ValueError("Độ phân giải vượt giới hạn.")
            if writer is None:
                metadata.update(width=w,height=h)
                size = (w//2*2, h//2*2)
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], size)
                if not writer.isOpened():
                    raise RuntimeError("Không tạo được video đầu ra.")
            tick = time.monotonic()
            result = model.track(frame, persist=True, tracker=str(tracker_path), conf=min(.1,confidence), iou=.5, agnostic_nms=True,
                                 imgsz=imgsz, device=device, classes=list(COCO_MAP) if coco else None, max_det=300, verbose=False)[0]
            inference += time.monotonic() - tick
            detections = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for box, cls, conf, track in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.id.cpu().tolist()):
                    track = int(track)
                    category = COCO_MAP[int(cls)] if coco else CLASSES[int(cls)]
                    votes[track][category] += float(conf)
                    last_seen[track] = seconds
                    category = max(votes[track], key=votes[track].get)
                    detections.append({"track_id": track, "class": category, "confidence": float(conf), "bbox": box,
                                       "point": (min(1,max(0,(box[0]+box[2])/(2*w))), min(1,max(0,box[3]/h)))})
            # Bound metadata for long camera sessions; ByteTrack IDs are monotonic.
            for track in list(last_seen):
                if seconds-last_seen[track] > 30:
                    del last_seen[track], votes[track]
            new_events = engine.update(detections, seconds)
            canvas = draw(frame, zone_map, detections, engine.active(seconds), seconds, len(events)+len(new_events))
            for event in new_events:
                event.update(id=uuid.uuid4().hex, analyzed_utc=datetime.now(timezone.utc).isoformat(), frame_index=frames)
                event["evidence"] = f"evidence/{event['id']}.jpg"
                folder = output / "evidence"
                folder.mkdir(exist_ok=True)
                if not cv2.imwrite(str(output / event["evidence"]), canvas):
                    raise OSError("Không ghi được ảnh bằng chứng.")
                events.append(event)
                if on_event:
                    on_event(event)
            writer.write(cv2.resize(canvas, size))
            frames += 1
            if time.monotonic()-last_progress >= .5:
                last_progress = time.monotonic()
                ok_jpg, jpeg = cv2.imencode(".jpg", canvas)
                if ok_jpg:
                    temporary = output / "preview.tmp"
                    temporary.write_bytes(jpeg.tobytes())
                    os.replace(temporary, output / "preview.jpg")
                status.update(processed_frames=frames, video_seconds=round(seconds,3), event_count=len(events), active_count=len(engine.active(seconds)),
                              elapsed_seconds=round(time.monotonic()-start,2), progress=0 if live else min(.95,frames/metadata["frames"]*.95))
                if progress:
                    progress(status.copy())
    finally:
        cap.release()
        if writer:
            writer.release()
    if frames:
        status.update(phase="encoding", progress=.96)
        if progress:
            progress(status.copy())
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output / "annotated.mp4")]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
        raw.unlink()
    fields = ["id","track_id","class","zone_id","zone_name","severity","kind","entered_seconds","confirmed_seconds","confidence","analyzed_utc","frame_index","evidence"]
    with (output / "events.csv").open("w", newline="", encoding="utf-8-sig") as f:
        export = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        export.writeheader()
        # Escape spreadsheet formulas in user-entered names.
        export.writerows({k: "'"+v if isinstance(v,str) and v.startswith(("=","+","-","@")) else v for k,v in event.items()} for event in events)
    elapsed = time.monotonic()-start
    status.update(phase="stopped" if stopped else "completed", progress=1, processed_frames=frames, event_count=len(events),
                  elapsed_seconds=round(elapsed,2), processing_fps=round(frames/elapsed,2), inference_ms=round(1000*inference/max(1,frames),2),
                  source_sha256=None if live else sha256(source), events=events, settings=settings,
                  video_timing="constant FPS source" if not live else "camera output is encoded at nominal FPS; event times use monotonic elapsed time")
    atomic_json(output / "result.json", status)
    if progress:
        progress({k:v for k,v in status.items() if k != "events"})
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Video path; camera:0 for USB camera")
    parser.add_argument("--zones", type=Path, default=ROOT / "config/parking.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODELS), default="vehicle")
    parser.add_argument("--device", choices=["auto","cpu","0"], default="auto")
    parser.add_argument("--confidence", type=float, default=.3)
    parser.add_argument("--imgsz", type=int, choices=[640,960], default=640)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    source = int(args.source.split(":")[1]) if args.source.startswith("camera:") else Path(args.source)
    result = analyze(source, args.output, ZoneMap.model_validate_json(args.zones.read_text(encoding="utf-8")), model_key=args.model, device=args.device,
                     confidence=args.confidence, imgsz=args.imgsz, max_seconds=args.max_seconds,
                     progress=lambda s: print(f"{s['phase']} {s['progress']:.0%} alerts={s['event_count']}", flush=True))
    print(json.dumps({k:result[k] for k in ("phase","processed_frames","event_count","processing_fps")}, indent=2))
