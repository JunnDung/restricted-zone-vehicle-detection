"""Validated normalized polygons and temporal intrusion rules, independent of AI."""
import math
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

CLASSES = ["motorcycle", "car", "truck", "bus", "other"]
LABELS = dict(zip(CLASSES, ["Xe máy", "Ô tô", "Xe tải", "Xe buýt", "Khác / xe đạp"]))


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def intersects(a, b, c, d):
    def on(a, b, p):
        return abs(cross(a, b, p)) < 1e-9 and all(min(a[i], b[i]) - 1e-9 <= p[i] <= max(a[i], b[i]) + 1e-9 for i in (0, 1))
    return (cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0) or any((on(a, b, c), on(a, b, d), on(c, d, a), on(c, d, b)))


class Zone(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,40}$")
    name: str = Field(min_length=1, max_length=80)
    color: str = Field(default="#ff665e", pattern=r"^#[0-9a-fA-F]{6}$")
    points: list[tuple[float, float]] = Field(min_length=3, max_length=32)
    classes: list[Literal["motorcycle", "car", "truck", "bus", "other"]] = Field(default_factory=lambda: CLASSES.copy(), min_length=1, max_length=5)
    dwell_seconds: float = Field(default=.4, ge=0, le=60)
    min_frames: int = Field(default=3, ge=2, le=60)
    exit_seconds: float = Field(default=.6, ge=.1, le=10)
    margin: float = Field(default=.006, ge=0, le=.05)
    alert_initial: bool = True
    severity: Literal["warning", "high"] = "high"

    @model_validator(mode="after")
    def geometry(self):
        pts = self.points
        if any(not math.isfinite(v) or not 0 <= v <= 1 for point in pts for v in point):
            raise ValueError("Tọa độ vùng phải hữu hạn và nằm trong [0,1].")
        if len(set(pts)) != len(pts):
            raise ValueError("Các đỉnh phải khác nhau; không lặp lại điểm đầu ở cuối.")
        area = abs(sum(pts[i][0] * pts[(i+1) % len(pts)][1] - pts[(i+1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2
        if area < .0001:
            raise ValueError("Vùng quá nhỏ hoặc các đỉnh thẳng hàng.")
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                if j == i+1 or (i == 0 and j == len(pts)-1):
                    continue
                if intersects(pts[i], pts[(i+1) % len(pts)], pts[j], pts[(j+1) % len(pts)]):
                    raise ValueError("Đa giác tự giao; hãy vẽ các đỉnh theo thứ tự quanh vùng.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Loại xe bị lặp.")
        return self


class ZoneMap(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zones: list[Zone] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({z.id for z in self.zones}) != len(self.zones):
            raise ValueError("Mỗi vùng cần ID riêng.")
        return self


@dataclass
class State:
    last: float
    position: bool = False
    since: float | None = None
    outside_since: float | None = None
    hits: int = 0
    alerted: bool = False
    saw_outside: bool = False


class IntrusionEngine:
    """One event per confirmed visit per track/zone; IDs are not physical identities."""
    def __init__(self, zone_map: ZoneMap):
        self.zones = zone_map.zones
        self.polygons = {z.id: np.array(z.points, dtype=np.float32) for z in self.zones}
        self.states = {}
        self.last_time = -1.0

    def update(self, detections, seconds):
        if not math.isfinite(seconds) or seconds < self.last_time:
            raise ValueError("Timestamp phải hữu hạn và không giảm.")
        self.last_time = seconds
        self.states = {k: v for k, v in self.states.items() if seconds - v.last <= 30}
        events = []
        for det in detections:
            for z in self.zones:
                key = (det["track_id"], z.id)
                state = self.states.setdefault(key, State(last=seconds))
                gap = seconds - state.last
                state.last = seconds
                if gap > 1:
                    # No dwell/exit confirmation may bridge a detection gap.
                    state.since, state.outside_since, state.hits = None, None, 0
                distance = cv2.pointPolygonTest(self.polygons[z.id], tuple(det["point"]), True)
                eligible = det["class"] in z.classes
                if distance > z.margin:
                    state.position = True
                elif distance < -z.margin:
                    state.position = False
                if state.position:
                    state.outside_since = None
                    # Low-score boxes maintain identity/position, not violation evidence.
                    if not eligible or not det.get("confirmable",True):
                        state.since, state.hits = None, 0
                        continue
                    if state.since is None:
                        state.since = seconds
                    state.hits += 1
                    if not state.alerted and state.hits >= z.min_frames and seconds - state.since + 1e-9 >= z.dwell_seconds and (z.alert_initial or state.saw_outside):
                        state.alerted = True
                        events.append({"track_id": det["track_id"], "class": det["class"], "confidence": round(det["confidence"], 4),
                                       "zone_id": z.id, "zone_name": z.name, "severity": z.severity,
                                       "entered_seconds": round(state.since, 3), "confirmed_seconds": round(seconds, 3),
                                       "kind": "entry" if state.saw_outside else "initial_presence", "point": list(det["point"])})
                else:
                    if distance < -z.margin:
                        state.saw_outside = True
                    state.since, state.hits = None, 0
                    if state.outside_since is None:
                        state.outside_since = seconds
                    if seconds - state.outside_since + 1e-9 >= z.exit_seconds:
                        state.alerted = False
        return events

    def active(self, seconds):
        return {key for key, state in self.states.items() if state.alerted and state.position and seconds - state.last <= 1}
