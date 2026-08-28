from __future__ import annotations

import json
import math
import os
import copy
import queue
import shutil
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
from platform_utils import open_folder
from error_dialog import show_copyable_error, show_runtime_log


VIDEO_PROJECT_VERSION = 3
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
INPUT_MODES = {
    "仅视频（前后帧估算背景）": "single",
    "流星视频＋干净参考视频": "clean_video",
}
EFFECT_MODES = ("延长并淡出", "仅流星慢放", "慢放并淡出")
CURVE_MODES = ("匀速", "渐慢", "渐快", "S曲线", "自定义")
ENCODING_QUALITIES = ("匹配源视频（推荐）", "极高质量 CRF 12", "平衡文件 CRF 18")


@dataclass
class VideoStroke:
    points: list[list[float]]
    width: int
    feather: int
    erase: bool = False
    locked: bool = False
    auto_score: int | None = None
    frame_offset: int = 0


@dataclass
class VideoEvent:
    frame: int
    score: int
    lines: list[list[float]] = field(default_factory=list)
    strokes: list[VideoStroke] = field(default_factory=list)
    accepted: bool = False
    locked: bool = False
    decision: str = "auto"
    use_custom: bool = False
    effect_mode: str | None = None
    local_speed: float | None = None
    hold_seconds: float | None = None
    fade_seconds: float | None = None
    brightness: float | None = None
    mask_width: int | None = None
    mask_feather: int | None = None
    curve: str | None = None
    curve_start: float | None = None
    curve_mid: float | None = None
    curve_end: float | None = None


@dataclass
class EventSettings:
    effect_mode: str
    local_speed: float
    hold_seconds: float
    fade_seconds: float
    brightness: float
    mask_width: int
    mask_feather: int
    curve: str
    curve_start: float
    curve_mid: float
    curve_end: float


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    bitrate: int = 0


@dataclass
class ResidualLayer:
    frame: int
    x0: int
    y0: int
    x1: int
    y1: int
    residual: np.ndarray
    settings: EventSettings
    source_offset: int = 0


@dataclass
class EventClip:
    start_frame: int
    layers: list[ResidualLayer]
    settings: EventSettings


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_v{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法创建不重名的输出文件：{path}")


def probe_video(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bitrate = max(0, round(float(capture.get(cv2.CAP_PROP_BITRATE)) * 1000.0))
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise ValueError("无法读取视频尺寸、帧率或帧数")
    return VideoInfo(width, height, fps, frames, frames / fps, bitrate)


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"无法读取第 {frame_index} 帧")
    return frame


def line_metrics(
    temporal: np.ndarray,
    spatial: np.ndarray,
    allowed: np.ndarray,
    raw: np.ndarray,
) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in raw)
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 8.0:
        return -1.0, length
    core = np.zeros_like(temporal, dtype=np.uint8)
    wide = np.zeros_like(temporal, dtype=np.uint8)
    p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
    cv2.line(core, p1, p2, 255, 3, cv2.LINE_AA)
    cv2.line(wide, p1, p2, 255, 11, cv2.LINE_AA)
    core_pixels = core > 32
    if not np.any(core_pixels) or float(np.mean(allowed[core_pixels])) < 0.92:
        return -1.0, length
    side_pixels = (wide > 32) & ~core_pixels & (allowed > 0)
    temporal_core = float(np.mean(temporal[core_pixels]))
    temporal_side = float(np.mean(temporal[side_pixels])) if np.any(side_pixels) else 0.0
    spatial_core = float(np.mean(spatial[core_pixels]))
    spatial_side = float(np.mean(spatial[side_pixels])) if np.any(side_pixels) else 0.0
    temporal_contrast = max(0.0, temporal_core - 0.40 * temporal_side)
    spatial_contrast = max(0.0, spatial_core - 0.55 * spatial_side)
    score = (temporal_contrast + spatial_contrast * 0.30) * math.sqrt(length)
    return score, length


def frame_candidates(
    target: np.ndarray,
    background: np.ndarray,
    preview_scale: float,
    ignore_bottom_percent: float,
) -> list[tuple[float, list[float]]]:
    width = target.shape[1]
    height = target.shape[0]
    preview_width = max(320, round(width * preview_scale))
    preview_height = max(240, round(height * preview_scale))
    target_small = cv2.resize(target, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    background_small = cv2.resize(background, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    delta = target_small.astype(np.float32) - background_small.astype(np.float32)
    temporal = np.maximum(
        0.0,
        delta[..., 0] * 0.114 + delta[..., 1] * 0.587 + delta[..., 2] * 0.299,
    )
    temporal = cv2.GaussianBlur(temporal, (0, 0), 0.7)
    gray = cv2.cvtColor(target_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spatial = np.maximum(0.0, gray - cv2.GaussianBlur(gray, (0, 0), 3.0))
    allowed = np.ones_like(gray, dtype=np.uint8)
    ignored_rows = round(preview_height * np.clip(ignore_bottom_percent, 0.0, 80.0) / 100.0)
    if ignored_rows:
        allowed[preview_height - ignored_rows :, :] = 0
        temporal[allowed == 0] = 0.0
        spatial[allowed == 0] = 0.0

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(np.uint8(np.clip(spatial * 5.0, 0, 255)))[0]
    if detected is None:
        return []
    ranked: list[tuple[float, np.ndarray, float]] = []
    for raw in detected[:, 0, :]:
        score, length = line_metrics(temporal, spatial, allowed, raw)
        if score > 0.8:
            ranked.append((score, raw.astype(np.float32), length))
    ranked.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[float, np.ndarray, float]] = []
    for score, line, length in ranked:
        x1, y1, x2, y2 = line
        midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        angle = math.atan2(y2 - y1, x2 - x1)
        duplicate = False
        for _, existing, _ in selected:
            ex1, ey1, ex2, ey2 = existing
            existing_midpoint = np.array([(ex1 + ex2) * 0.5, (ey1 + ey2) * 0.5])
            existing_angle = math.atan2(ey2 - ey1, ex2 - ex1)
            angle_delta = abs(math.atan2(math.sin(angle - existing_angle), math.cos(angle - existing_angle)))
            angle_delta = min(angle_delta, abs(math.pi - angle_delta))
            if np.linalg.norm(midpoint - existing_midpoint) < 28 and angle_delta < math.radians(20):
                duplicate = True
                break
        if duplicate:
            continue
        if selected and score < selected[0][0] * 0.12:
            continue
        selected.append((score, line, length))
        if len(selected) == 8:
            break

    result: list[tuple[float, list[float]]] = []
    for score, line, length in selected:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        norm = max(1.0, math.hypot(dx, dy))
        extension = max(5.0, length * 0.25)
        ux, uy = dx / norm, dy / norm
        x1 -= ux * extension
        y1 -= uy * extension
        x2 += ux * extension
        y2 += uy * extension
        normalized = [
            float(np.clip(x1 / max(1, preview_width - 1), 0.0, 1.0)),
            float(np.clip(y1 / max(1, preview_height - 1), 0.0, 1.0)),
            float(np.clip(x2 / max(1, preview_width - 1), 0.0, 1.0)),
            float(np.clip(y2 / max(1, preview_height - 1), 0.0, 1.0)),
        ]
        result.append((score, normalized))
    return result


def analyze_video(
    video_path: Path,
    clean_video_path: Path | None,
    ignore_bottom_percent: float,
    progress: Callable[[float, str], None],
) -> tuple[VideoInfo, list[VideoEvent]]:
    info = probe_video(video_path)
    clean_info = probe_video(clean_video_path) if clean_video_path else None
    if clean_info and (
        clean_info.width != info.width
        or clean_info.height != info.height
        or clean_info.frames != info.frames
        or abs(clean_info.fps - info.fps) > 0.01
    ):
        raise ValueError("干净参考视频必须与流星视频具有相同尺寸、帧率和帧数")

    scale = min(1.0, 960.0 / max(info.width, info.height))
    capture = cv2.VideoCapture(str(video_path))
    clean_capture = cv2.VideoCapture(str(clean_video_path)) if clean_video_path else None
    previous: np.ndarray | None = None
    current: np.ndarray | None = None
    clean_current: np.ndarray | None = None
    raw_candidates: list[tuple[int, float, list[float]]] = []
    index = 0
    while True:
        ok, following = capture.read()
        if not ok:
            break
        clean_following = None
        if clean_capture is not None:
            clean_ok, clean_following = clean_capture.read()
            if not clean_ok:
                raise ValueError("干净参考视频提前结束")
        if current is not None:
            if clean_current is not None:
                background = clean_current
            elif previous is not None:
                background = ((previous.astype(np.uint16) + following.astype(np.uint16)) // 2).astype(np.uint8)
            else:
                background = following
            candidates = frame_candidates(current, background, scale, ignore_bottom_percent)
            for score, line in candidates:
                raw_candidates.append((index - 1, score, line))
        previous, current = current, following
        clean_current = clean_following
        index += 1
        if index % 20 == 0:
            progress(index / info.frames * 92.0, f"分析视频 {index}/{info.frames}")
    capture.release()
    if clean_capture is not None:
        clean_capture.release()

    if not raw_candidates:
        return info, []

    # Static foreground edges recur at nearly the same position and angle in many
    # frames. Penalize that recurrence; a meteor is normally a one-off transient.
    recurrence: dict[tuple[int, int, int], int] = {}
    for _frame, _score, line in raw_candidates:
        x1, y1, x2, y2 = line
        midpoint_x = (x1 + x2) * 0.5
        midpoint_y = (y1 + y2) * 0.5
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        signature = (round(midpoint_x * 24), round(midpoint_y * 18), round(angle / math.radians(15)))
        recurrence[signature] = recurrence.get(signature, 0) + 1

    by_frame: dict[int, list[tuple[float, list[float]]]] = {}
    for frame, score, line in raw_candidates:
        x1, y1, x2, y2 = line
        midpoint_x = (x1 + x2) * 0.5
        midpoint_y = (y1 + y2) * 0.5
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        signature = (round(midpoint_x * 24), round(midpoint_y * 18), round(angle / math.radians(15)))
        count = recurrence[signature]
        adjusted = score / (1.0 + 0.55 * math.sqrt(max(0, count - 1)))
        by_frame.setdefault(frame, []).append((adjusted, line))

    raw_events: list[tuple[int, float, list[list[float]]]] = []
    for frame, candidates in by_frame.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][0]
        lines = [line for score, line in candidates[:2] if score >= best * 0.48]
        raw_events.append((frame, best, lines))

    raw_scores = np.array([item[1] for item in raw_events], dtype=np.float32)
    low = float(np.percentile(raw_scores, 35.0))
    high = float(np.percentile(raw_scores, 99.0))
    span = max(1e-6, high - low)
    normalized = [
        VideoEvent(
            frame=frame,
            score=int(np.clip(round(1 + 99 * (score - low) / span), 1, 100)),
            lines=lines,
        )
        for frame, score, lines in raw_events
    ]
    # Keep the review list bounded while preserving high-recall candidates.
    normalized.sort(key=lambda event: event.score, reverse=True)
    normalized = normalized[: min(400, len(normalized))]
    normalized.sort(key=lambda event: event.frame)
    progress(100.0, f"分析完成：{len(normalized)} 个候选")
    return info, normalized


def resolve_settings(event: VideoEvent, defaults: EventSettings) -> EventSettings:
    if not event.use_custom:
        return defaults
    return EventSettings(
        effect_mode=event.effect_mode or defaults.effect_mode,
        local_speed=float(event.local_speed if event.local_speed is not None else defaults.local_speed),
        hold_seconds=float(event.hold_seconds if event.hold_seconds is not None else defaults.hold_seconds),
        fade_seconds=float(event.fade_seconds if event.fade_seconds is not None else defaults.fade_seconds),
        brightness=float(event.brightness if event.brightness is not None else defaults.brightness),
        mask_width=int(event.mask_width if event.mask_width is not None else defaults.mask_width),
        mask_feather=int(event.mask_feather if event.mask_feather is not None else defaults.mask_feather),
        curve=event.curve or defaults.curve,
        curve_start=float(event.curve_start if event.curve_start is not None else defaults.curve_start),
        curve_mid=float(event.curve_mid if event.curve_mid is not None else defaults.curve_mid),
        curve_end=float(event.curve_end if event.curve_end is not None else defaults.curve_end),
    )


def decode_stroke(item: VideoStroke | dict) -> VideoStroke:
    if isinstance(item, VideoStroke):
        return item
    return VideoStroke(
        points=[[float(x), float(y)] for x, y in item.get("points", [])],
        width=int(item.get("width", 24)),
        feather=int(item.get("feather", 14)),
        erase=bool(item.get("erase", False)),
        locked=bool(item.get("locked", False)),
        auto_score=item.get("auto_score"),
        frame_offset=int(item.get("frame_offset", 0)),
    )


def ensure_event_strokes(event: VideoEvent, defaults: EventSettings) -> list[VideoStroke]:
    event.strokes = [decode_stroke(stroke) for stroke in event.strokes]
    if not event.strokes and event.lines:
        event.strokes = [
            VideoStroke(
                points=[[line[0], line[1]], [line[2], line[3]]],
                width=defaults.mask_width,
                feather=defaults.mask_feather,
                auto_score=event.score,
            )
            for line in event.lines
        ]
    return event.strokes


def event_offsets(event: VideoEvent, defaults: EventSettings) -> list[int]:
    strokes = ensure_event_strokes(event, defaults)
    values = sorted({stroke.frame_offset for stroke in strokes if not stroke.erase})
    return values or [0]


def build_stroke_mask(
    strokes: list[VideoStroke],
    width: int,
    height: int,
    settings: EventSettings,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    for stroke in strokes:
        if not stroke.points:
            continue
        hard_width = settings.mask_width if stroke.auto_score is not None else stroke.width
        feather = settings.mask_feather if stroke.auto_score is not None else stroke.feather
        layer = np.zeros_like(mask)
        points = np.array(
            [[round(x * (width - 1)), round(y * (height - 1))] for x, y in stroke.points],
            dtype=np.int32,
        )
        if len(points) == 1:
            cv2.circle(layer, tuple(points[0]), max(1, hard_width // 2), 1.0, -1, cv2.LINE_AA)
        else:
            cv2.polylines(layer, [points], False, 1.0, max(1, hard_width), cv2.LINE_AA)
        if feather > 0:
            layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(0.5, feather / 2.5))
        layer = np.clip(layer, 0.0, 1.0)
        if stroke.erase:
            mask *= 1.0 - layer
        else:
            mask = np.maximum(mask, layer)
    return np.clip(mask, 0.0, 1.0)


def build_residual_layer(
    event: VideoEvent,
    target: np.ndarray,
    background: np.ndarray,
    settings: EventSettings,
    source_offset: int = 0,
) -> ResidualLayer | None:
    height, width = target.shape[:2]
    strokes = [
        stroke for stroke in ensure_event_strokes(event, settings)
        if stroke.frame_offset == source_offset
    ]
    mask = build_stroke_mask(strokes, width, height, settings)
    delta = target.astype(np.float32) - background.astype(np.float32)
    positive_luma = np.maximum(
        0.0,
        delta[..., 0] * 0.114 + delta[..., 1] * 0.587 + delta[..., 2] * 0.299,
    )
    gate = np.clip(positive_luma / 2.5, 0.0, 1.0)
    effective = mask * gate
    ys, xs = np.where(effective > 0.002)
    if len(xs) == 0:
        return None
    pad = settings.mask_feather + 8
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(width, int(xs.max()) + pad + 1)
    y1 = min(height, int(ys.max()) + pad + 1)
    residual = delta[y0:y1, x0:x1] * effective[y0:y1, x0:x1, None] * settings.brightness
    return ResidualLayer(event.frame, x0, y0, x1, y1, residual, settings, source_offset)


def speed_curve_progress(position: float, settings: EventSettings) -> float:
    position = float(np.clip(position, 0.0, 1.0))
    speeds = {
        "匀速": (1.0, 1.0, 1.0),
        "渐慢": (1.8, 0.8, 0.25),
        "渐快": (0.25, 0.8, 1.8),
        "S曲线": (0.35, 1.8, 0.35),
        "自定义": (settings.curve_start, settings.curve_mid, settings.curve_end),
    }.get(settings.curve, (1.0, 1.0, 1.0))
    start, middle, end = (max(0.05, float(value)) for value in speeds)
    # Integral of a quadratic Bezier speed curve, normalized to 0..1.
    a = start - 2.0 * middle + end
    b = -2.0 * start + 2.0 * middle
    c = start
    integral = a * position**3 / 3.0 + b * position**2 / 2.0 + c * position
    total = a / 3.0 + b / 2.0 + c
    return float(np.clip(integral / max(1e-6, total), 0.0, 1.0))


def effect_strength(age_frames: int, fps: float, settings: EventSettings) -> float:
    source_frame_seconds = 1.0 / fps
    slowed_frame_seconds = source_frame_seconds / max(0.05, settings.local_speed / 100.0)
    hold = settings.hold_seconds
    fade = settings.fade_seconds
    if settings.effect_mode == "仅流星慢放":
        hold = max(hold, slowed_frame_seconds)
        fade = 0.0
    elif settings.effect_mode == "慢放并淡出":
        hold = max(hold, slowed_frame_seconds)
    elapsed = age_frames / fps
    if elapsed <= hold:
        return 1.0
    if fade <= 0 or elapsed > hold + fade:
        return 0.0
    progress = speed_curve_progress((elapsed - hold) / fade, settings)
    return 1.0 - progress


def prepare_layers(
    video_path: Path,
    clean_video_path: Path | None,
    events: list[VideoEvent],
    defaults: EventSettings,
    progress: Callable[[float, str], None],
) -> tuple[VideoInfo, list[EventClip]]:
    info = probe_video(video_path)
    accepted = [
        event for event in events
        if event.accepted and any(not stroke.erase for stroke in ensure_event_strokes(event, defaults))
    ]
    if not accepted:
        raise ValueError("没有已确认并带有流星蒙版的候选")

    meteor_frames = {
        event.frame + offset
        for event in accepted
        for offset in event_offsets(event, defaults)
        if 0 <= event.frame + offset < info.frames
    }
    jobs = [
        (event, offset, event.frame + offset)
        for event in accepted
        for offset in event_offsets(event, defaults)
        if 0 <= event.frame + offset < info.frames
    ]
    layers_by_event: dict[int, list[ResidualLayer]] = {id(event): [] for event in accepted}
    for position, (event, offset, frame_index) in enumerate(jobs, start=1):
        target = read_video_frame(video_path, frame_index)
        if clean_video_path is not None:
            background = read_video_frame(clean_video_path, frame_index)
        else:
            before_index = next(
                (value for value in range(frame_index - 1, max(-1, frame_index - 5), -1) if value not in meteor_frames),
                max(0, frame_index - 1),
            )
            after_index = next(
                (value for value in range(frame_index + 1, min(info.frames, frame_index + 5)) if value not in meteor_frames),
                min(info.frames - 1, frame_index + 1),
            )
            before = read_video_frame(video_path, before_index)
            after = read_video_frame(video_path, after_index)
            background = ((before.astype(np.uint16) + after.astype(np.uint16)) // 2).astype(np.uint8)
        settings = resolve_settings(event, defaults)
        layer = build_residual_layer(event, target, background, settings, offset)
        if layer is not None:
            layer.frame = frame_index
            layers_by_event[id(event)].append(layer)
        progress(position / max(1, len(jobs)) * 25.0, f"准备流星层 {position}/{len(jobs)}")

    clips = [
        EventClip(event.frame, sorted(layers_by_event[id(event)], key=lambda layer: layer.source_offset), resolve_settings(event, defaults))
        for event in accepted
        if layers_by_event[id(event)]
    ]
    return info, clips


def clip_duration_frames(clip: EventClip, fps: float) -> int:
    source_span = max(1, max(layer.source_offset for layer in clip.layers) + 1)
    if clip.settings.effect_mode == "延长并淡出":
        playback = source_span
    else:
        playback = max(source_span, math.ceil(source_span / max(0.05, clip.settings.local_speed / 100.0)))
    hold = round(max(0.0, clip.settings.hold_seconds) * fps)
    fade = round(max(0.0, clip.settings.fade_seconds) * fps)
    if clip.settings.effect_mode == "仅流星慢放":
        hold = fade = 0
    return max(1, playback + hold + fade)


def sample_clip(clip: EventClip, age: int, fps: float) -> list[tuple[ResidualLayer, float]]:
    """Return residual layers and blend weights for one retimed event frame."""
    if age < 0:
        return []
    source_span = max(1, max(layer.source_offset for layer in clip.layers) + 1)
    if clip.settings.effect_mode == "延长并淡出":
        playback = source_span
    else:
        playback = max(source_span, math.ceil(source_span / max(0.05, clip.settings.local_speed / 100.0)))
    if age < playback:
        position = 1.0 if playback <= 1 else age / (playback - 1)
        source_position = speed_curve_progress(position, clip.settings) * max(0, source_span - 1)
        ordered = sorted(clip.layers, key=lambda layer: layer.source_offset)
        lower = max((layer for layer in ordered if layer.source_offset <= source_position), key=lambda layer: layer.source_offset, default=ordered[0])
        upper = min((layer for layer in ordered if layer.source_offset >= source_position), key=lambda layer: layer.source_offset, default=ordered[-1])
        if lower is upper or upper.source_offset == lower.source_offset:
            return [(lower, 1.0)]
        weight = (source_position - lower.source_offset) / (upper.source_offset - lower.source_offset)
        return [(lower, 1.0 - weight), (upper, weight)]
    if clip.settings.effect_mode == "仅流星慢放":
        return []
    hold = round(max(0.0, clip.settings.hold_seconds) * fps)
    fade = round(max(0.0, clip.settings.fade_seconds) * fps)
    tail_age = age - playback
    if tail_age < hold:
        return [(max(clip.layers, key=lambda layer: layer.source_offset), 1.0)]
    if fade <= 0 or tail_age >= hold + fade:
        return []
    strength = 1.0 - (tail_age - hold) / fade
    return [(max(clip.layers, key=lambda layer: layer.source_offset), strength)]


def ffmpeg_executable() -> str | None:
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    candidates: list[Path] = []
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / name)
    candidates.append(Path(sys.executable).resolve().parent / name)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffmpeg")


def atempo_filter(speed: float) -> str:
    values: list[float] = []
    remaining = max(0.05, float(speed))
    while remaining < 0.5:
        values.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        values.append(2.0)
        remaining /= 2.0
    values.append(remaining)
    return ",".join(f"atempo={value:.6f}" for value in values)


def render_video(
    video_path: Path,
    output_path: Path,
    info: VideoInfo,
    clips: list[EventClip],
    background_speed: float,
    keep_audio: bool,
    progress: Callable[[float, str], None],
    encoding_quality: str = "匹配源视频（推荐）",
) -> Path:
    ffmpeg = ffmpeg_executable()
    if ffmpeg is None:
        raise RuntimeError("没有找到 FFmpeg。请安装 FFmpeg 并加入 PATH，或在打包时附带 FFmpeg。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = unique_path(output_path)
    background_speed = float(np.clip(background_speed, 0.05, 4.0))
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
        "-r", f"{info.fps:g}", "-i", "-",
    ]
    if keep_audio:
        command += ["-i", str(video_path), "-map", "0:v:0", "-map", "1:a?"]
    else:
        command += ["-map", "0:v:0", "-an"]
    command += [
        "-c:v", "libx264", "-profile:v", "main", "-bf", "0", "-refs", "1",
        "-g", str(max(1, round(info.fps))), "-keyint_min", str(max(1, round(info.fps))),
        "-sc_threshold", "0", "-preset", "medium",
    ]
    if encoding_quality == "平衡文件 CRF 18":
        command += ["-crf", "18"]
    elif encoding_quality == "极高质量 CRF 12" or info.bitrate <= 0:
        command += ["-crf", "12"]
    else:
        target = max(2_000_000, info.bitrate)
        command += [
            "-b:v", str(target), "-maxrate", str(round(target * 1.18)),
            "-bufsize", str(round(target * 2.0)),
        ]
    command += ["-pix_fmt", "yuv420p"]
    if keep_audio:
        if abs(background_speed - 1.0) < 1e-6:
            command += ["-c:a", "copy"]
        else:
            command += ["-filter:a", atempo_filter(background_speed), "-c:a", "aac", "-b:a", "320k"]
        command += ["-shortest"]
    command += ["-movflags", "+faststart", str(output_path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("无法启动视频编码器")
    capture = cv2.VideoCapture(str(video_path))
    starts: dict[int, list[EventClip]] = {}
    natural_layers: dict[int, list[ResidualLayer]] = {}
    for clip in clips:
        starts.setdefault(round(clip.start_frame / background_speed), []).append(clip)
        for layer in clip.layers:
            natural_layers.setdefault(layer.frame, []).append(layer)
    active: list[tuple[int, EventClip]] = []
    cache: dict[int, np.ndarray] = {}
    last_read = -1
    output_total = math.floor((info.frames - 1) / background_speed) + 1

    def get_clean_frame(source_index: int) -> np.ndarray:
        nonlocal last_read
        source_index = int(np.clip(source_index, 0, info.frames - 1))
        while last_read < source_index:
            ok, source_frame = capture.read()
            if not ok:
                raise RuntimeError(f"无法读取源视频第 {last_read + 1} 帧")
            last_read += 1
            source_float = source_frame.astype(np.float32)
            for layer in natural_layers.get(last_read, []):
                source_float[layer.y0:layer.y1, layer.x0:layer.x1] -= layer.residual / max(0.05, layer.settings.brightness)
            cache[last_read] = np.clip(source_float, 0, 255).astype(np.uint8)
            for old in [value for value in cache if value < last_read - 2]:
                del cache[old]
        return cache[source_index]

    try:
        for output_index in range(output_total):
            source_position = min(info.frames - 1, output_index * background_speed)
            first_index = math.floor(source_position)
            second_index = min(info.frames - 1, first_index + 1)
            blend = source_position - first_index
            first = get_clean_frame(first_index)
            second = get_clean_frame(second_index)
            frame = first.copy() if blend <= 1e-6 else cv2.addWeighted(first, 1.0 - blend, second, blend, 0.0)
            frame_float = frame.astype(np.float32)
            active.extend((output_index, clip) for clip in starts.get(output_index, []))
            remaining: list[tuple[int, EventClip]] = []
            for trigger, clip in active:
                age = output_index - trigger
                samples = sample_clip(clip, age, info.fps)
                for layer, strength in samples:
                    frame_float[layer.y0:layer.y1, layer.x0:layer.x1] += layer.residual * strength
                if age + 1 < clip_duration_frames(clip, info.fps):
                    remaining.append((trigger, clip))
            active = remaining
            frame = np.clip(frame_float, 0, 255).astype(np.uint8)
            encoder.stdin.write(frame.tobytes())
            if output_index % 20 == 0:
                progress(25.0 + (output_index + 1) / output_total * 74.0, f"渲染视频 {output_index + 1}/{output_total}")
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("FFmpeg 编码失败")

    progress(100.0, f"导出完成：{output_path.name}")
    return output_path


def video_autosave_path() -> Path:
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / "MeteorStudio"
    elif os.name == "nt":
        root = Path(os.environ.get("APPDATA", str(Path.home()))) / "MeteorStudio"
    else:
        root = Path.home() / ".config" / "MeteorStudio"
    root.mkdir(parents=True, exist_ok=True)
    return root / "video_autosave.json"


class VideoMeteorWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("流星影像工坊 — 视频动态")
        self.geometry("1420x900")
        self.minsize(1120, 720)

        self.video_path = tk.StringVar()
        self.clean_video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.input_mode_label = tk.StringVar(value=next(iter(INPUT_MODES)))
        self.threshold = tk.IntVar(value=72)
        self.ignore_bottom = tk.DoubleVar(value=15.0)
        self.preview_mode = tk.StringVar(value="original")
        self.edit_mode = tk.StringVar(value="brush")
        self.brush_width = tk.IntVar(value=24)
        self.eraser_width = tk.IntVar(value=42)
        self.brush_feather = tk.IntVar(value=14)
        self.source_offset = tk.IntVar(value=0)
        self.status = tk.StringVar(value="选择视频后执行自动分析。源视频始终只读。")

        self.effect_mode = tk.StringVar(value="慢放并淡出")
        self.local_speed = tk.DoubleVar(value=20.0)
        self.hold_seconds = tk.DoubleVar(value=0.30)
        self.fade_seconds = tk.DoubleVar(value=0.70)
        self.brightness = tk.DoubleVar(value=1.0)
        self.mask_width = tk.IntVar(value=24)
        self.mask_feather = tk.IntVar(value=14)
        self.curve = tk.StringVar(value="渐慢")
        self.curve_start = tk.DoubleVar(value=1.8)
        self.curve_mid = tk.DoubleVar(value=0.8)
        self.curve_end = tk.DoubleVar(value=0.25)
        self.overall_speed = tk.DoubleVar(value=100.0)
        self.encoding_quality = tk.StringVar(value=ENCODING_QUALITIES[0])
        self.keep_audio = tk.BooleanVar(value=False)

        self.info: VideoInfo | None = None
        self.events: list[VideoEvent] = []
        self.current_event: VideoEvent | None = None
        self.preview_frame: np.ndarray | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.display_box = (0, 0, 1, 1)
        self.active_points: list[list[float]] = []
        self.active_canvas_points: list[tuple[int, int]] = []
        self.active_item: int | None = None
        self.shift_anchors: dict[tuple[int, int], list[float]] = {}
        self.history: dict[int, list[list[VideoStroke]]] = {}
        self.redo_history: dict[int, list[list[VideoStroke]]] = {}
        self.alt_previous_mode: str | None = None
        self.mask_visible = True
        self.cursor_item: int | None = None
        self.context_stroke_index: int | None = None
        self.work_queue: queue.Queue = queue.Queue()
        self.busy = False
        self.autosave_file = video_autosave_path()
        self.autosave_after_id: str | None = None

        self._build_ui()
        self._bind_shortcuts()
        self.after(150, self._poll_queue)
        self.after(300, self._restore_autosave)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="视频流星动态", font=("TkDefaultFont", 15, "bold")).pack(side="left")
        ttk.Label(
            header,
            text="只改变流星时间层；背景保持正常播放。所有输出写入新文件。",
        ).pack(side="left", padx=14)
        ttk.Button(header, text="返回图片合成工作区", command=self.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="运行日志", command=lambda: show_runtime_log(self)).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="保存视频项目", command=self.save_project).pack(side="right")
        ttk.Button(header, text="载入视频项目", command=self.load_project).pack(side="right", padx=6)

        paths = ttk.LabelFrame(root, text="输入方案与输出", padding=8)
        paths.pack(fill="x")
        ttk.Label(paths, text="素材方案").grid(row=0, column=0, sticky="w")
        mode = ttk.Combobox(paths, textvariable=self.input_mode_label, values=tuple(INPUT_MODES), state="readonly", width=30)
        mode.grid(row=0, column=1, sticky="w", padx=6)
        mode.bind("<<ComboboxSelected>>", lambda _e: self._mode_changed())
        ttk.Label(paths, text="流星视频").grid(row=1, column=0, sticky="w")
        ttk.Entry(paths, textvariable=self.video_path).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(paths, text="选择…", command=self._browse_video).grid(row=1, column=4)
        self.clean_label = ttk.Label(paths, text="干净参考视频")
        self.clean_label.grid(row=2, column=0, sticky="w")
        self.clean_entry = ttk.Entry(paths, textvariable=self.clean_video_path)
        self.clean_entry.grid(row=2, column=1, columnspan=3, sticky="ew", padx=6)
        self.clean_button = ttk.Button(paths, text="选择…", command=self._browse_clean_video)
        self.clean_button.grid(row=2, column=4)
        ttk.Label(paths, text="输出文件夹（可留空）").grid(row=3, column=0, sticky="w")
        ttk.Entry(paths, textvariable=self.output_dir).grid(row=3, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(paths, text="选择…", command=self._browse_output).grid(row=3, column=4)
        ttk.Button(paths, text="打开文件夹", command=self._open_output_folder).grid(row=3, column=5, padx=(6, 0))
        ttk.Button(paths, text="自动分析视频", command=self.analyze).grid(row=0, column=4, rowspan=1, padx=(8, 0))
        paths.columnconfigure(3, weight=1)
        self._mode_changed()

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 6))

        left = ttk.Frame(body, width=350)
        body.add(left, weight=0)
        review = ttk.LabelFrame(left, text="候选审核", padding=7)
        review.pack(fill="x")
        ttk.Label(review, text="自动接受分数").grid(row=0, column=0, sticky="w")
        ttk.Scale(review, from_=1, to=100, variable=self.threshold, orient="horizontal", command=self._threshold_changed).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(review, textvariable=self.threshold, width=4).grid(row=0, column=2)
        ttk.Label(review, text="忽略底部(%)").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Scale(review, from_=0, to=45, variable=self.ignore_bottom, orient="horizontal", command=lambda _v: self._schedule_autosave()).grid(row=1, column=1, sticky="ew", padx=5, pady=(5, 0))
        ttk.Label(review, textvariable=self.ignore_bottom, width=4).grid(row=1, column=2, pady=(5, 0))
        review.columnconfigure(1, weight=1)

        self.tree = ttk.Treeview(left, columns=("time", "score", "state", "lock"), show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="帧")
        self.tree.heading("time", text="时间")
        self.tree.heading("score", text="分数")
        self.tree.heading("state", text="判断")
        self.tree.heading("lock", text="锁定")
        self.tree.column("#0", width=55, anchor="e")
        self.tree.column("time", width=70, anchor="e")
        self.tree.column("score", width=48, anchor="center")
        self.tree.column("state", width=58, anchor="center")
        self.tree.column("lock", width=42, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=6)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        decision = ttk.Frame(left)
        decision.pack(fill="x")
        ttk.Button(decision, text="✓ 保留", command=self.accept_current).pack(side="left", fill="x", expand=True)
        ttk.Button(decision, text="✕ 排除", command=self.reject_current).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(decision, text="锁定/解锁", command=self.toggle_lock).pack(side="left", fill="x", expand=True)
        ttk.Button(left, text="＋ 手工添加当前帧", command=self.add_manual_event).pack(fill="x", pady=(5, 0))
        groups = ttk.Frame(left)
        groups.pack(fill="x", pady=(5, 0))
        ttk.Button(groups, text="合并所选连续帧", command=self.merge_selected_events).pack(side="left", fill="x", expand=True)
        ttk.Button(groups, text="拆分多帧事件", command=self.split_current_event).pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Button(left, text="当前流星单独参数…", command=self.edit_current_settings).pack(fill="x", pady=(5, 0))

        center = ttk.Frame(body)
        body.add(center, weight=1)
        preview_bar = ttk.Frame(center)
        preview_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(preview_bar, text="查看：").pack(side="left")
        for text, value in (("原视频帧", "original"), ("提取的流星层", "residual"), ("融合后效果", "effect")):
            ttk.Radiobutton(preview_bar, text=text, variable=self.preview_mode, value=value, command=self._render_current).pack(side="left", padx=(0, 7))
        ttk.Label(preview_bar, text="B画笔 / E橡皮擦 / Alt临时切换 / Shift连线 / H临时隐藏蒙版").pack(side="right")
        self.canvas = tk.Canvas(center, background="#171717", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render_current())
        self.canvas.bind("<ButtonPress-1>", self._stroke_start)
        self.canvas.bind("<B1-Motion>", self._stroke_move)
        self.canvas.bind("<ButtonRelease-1>", self._stroke_end)
        self.canvas.bind("<Motion>", self._update_brush_cursor)
        self.canvas.bind("<Leave>", self._hide_brush_cursor)
        self.canvas.bind("<Button-3>", self._show_stroke_menu)
        self.canvas.bind("<Control-Button-1>", self._delete_stroke_at)
        self.canvas.bind("<Command-Button-1>", self._delete_stroke_at)
        self.canvas.bind("<Shift-Button-3>", self._delete_stroke_at)

        tools_bar = ttk.Frame(center)
        tools_bar.pack(fill="x", pady=(5, 0))
        ttk.Radiobutton(tools_bar, text="画笔 (B)", variable=self.edit_mode, value="brush", command=self._tool_changed).pack(side="left")
        ttk.Radiobutton(tools_bar, text="橡皮擦 (E)", variable=self.edit_mode, value="eraser", command=self._tool_changed).pack(side="left", padx=(5, 12))
        ttk.Label(tools_bar, text="画笔宽").pack(side="left")
        ttk.Scale(tools_bar, from_=2, to=180, variable=self.brush_width, orient="horizontal", length=105, command=lambda _v: self._tool_changed()).pack(side="left")
        ttk.Label(tools_bar, textvariable=self.brush_width, width=4).pack(side="left")
        ttk.Label(tools_bar, text="橡皮宽").pack(side="left", padx=(8, 0))
        ttk.Scale(tools_bar, from_=2, to=240, variable=self.eraser_width, orient="horizontal", length=105, command=lambda _v: self._tool_changed()).pack(side="left")
        ttk.Label(tools_bar, textvariable=self.eraser_width, width=4).pack(side="left")
        ttk.Label(tools_bar, text="羽化").pack(side="left", padx=(8, 0))
        ttk.Scale(tools_bar, from_=0, to=100, variable=self.brush_feather, orient="horizontal", length=90, command=lambda _v: self._tool_changed()).pack(side="left")
        ttk.Label(tools_bar, textvariable=self.brush_feather, width=4).pack(side="left")
        ttk.Button(tools_bar, text="撤销", command=self.undo_mask).pack(side="right")
        ttk.Button(tools_bar, text="重做", command=self.redo_mask).pack(side="right", padx=4)

        frame_bar = ttk.Frame(center)
        frame_bar.pack(fill="x", pady=(4, 0))
        ttk.Label(frame_bar, text="事件内流星帧：").pack(side="left")
        ttk.Button(frame_bar, text="◀", width=3, command=lambda: self._change_source_offset(-1)).pack(side="left")
        ttk.Label(frame_bar, textvariable=self.source_offset, width=5, anchor="center").pack(side="left")
        ttk.Button(frame_bar, text="▶", width=3, command=lambda: self._change_source_offset(1)).pack(side="left")
        ttk.Label(frame_bar, text="（合并连续候选后，可逐帧编辑并用曲线慢放）").pack(side="left", padx=8)

        right = ttk.LabelFrame(body, text="全局流星动态参数", padding=8, width=330)
        body.add(right, weight=0)
        row = 0
        ttk.Label(right, text="效果模式").grid(row=row, column=0, sticky="w")
        ttk.Combobox(right, textvariable=self.effect_mode, values=EFFECT_MODES, state="readonly", width=18).grid(row=row, column=1, sticky="ew")
        row += 1
        self._scale_row(right, row, "流星局部速度(%)", self.local_speed, 5, 100)
        row += 1
        self._scale_row(right, row, "清晰停留(秒)", self.hold_seconds, 0, 2.0)
        row += 1
        self._scale_row(right, row, "淡出时间(秒)", self.fade_seconds, 0, 3.0)
        row += 1
        self._scale_row(right, row, "流星亮度", self.brightness, 0.25, 3.0)
        row += 1
        self._scale_row(right, row, "蒙版宽度(px)", self.mask_width, 2, 140)
        row += 1
        self._scale_row(right, row, "蒙版羽化(px)", self.mask_feather, 0, 100)
        row += 1
        ttk.Separator(right).grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1
        ttk.Label(right, text="流星时间曲线").grid(row=row, column=0, sticky="w")
        ttk.Combobox(right, textvariable=self.curve, values=CURVE_MODES, state="readonly", width=18).grid(row=row, column=1, sticky="ew")
        row += 1
        self._scale_row(right, row, "起始速度", self.curve_start, 0.1, 3.0)
        row += 1
        self._scale_row(right, row, "中段速度", self.curve_mid, 0.1, 3.0)
        row += 1
        self._scale_row(right, row, "结束速度", self.curve_end, 0.1, 3.0)
        row += 1
        ttk.Separator(right).grid(row=row, column=0, columnspan=3, sticky="ew", pady=8)
        row += 1
        self._scale_row(right, row, "背景播放速度(%)", self.overall_speed, 25, 100)
        row += 1
        ttk.Label(right, text="导出质量").grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(right, textvariable=self.encoding_quality, values=ENCODING_QUALITIES, state="readonly", width=20).grid(row=row, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        row += 1
        ttk.Checkbutton(right, text="保留音频（背景变速时同步变速）", variable=self.keep_audio, command=self._schedule_autosave).grid(row=row, column=0, columnspan=3, sticky="w", pady=(6, 0))
        row += 1
        ttk.Label(
            right,
            text="流星速度和背景速度互相独立。\n背景减速不会拉长流星动态。\n匹配源视频默认使用兼容编码。",
            foreground="#555555",
            justify="left",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
        right.columnconfigure(1, weight=1)

        bottom = ttk.Frame(root)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=260)
        self.progress.pack(side="left", padx=8)
        ttk.Button(bottom, text="导出动态流星视频", command=self.export).pack(side="right")

        watched = (
            self.video_path, self.clean_video_path, self.output_dir, self.input_mode_label,
            self.effect_mode, self.local_speed, self.hold_seconds, self.fade_seconds,
            self.brightness, self.mask_width, self.mask_feather, self.curve,
            self.curve_start, self.curve_mid, self.curve_end, self.overall_speed,
            self.edit_mode, self.brush_width, self.eraser_width, self.brush_feather,
            self.encoding_quality,
        )
        for variable in watched:
            variable.trace_add("write", lambda *_args: self._schedule_autosave())

        self.mask_menu = tk.Menu(self, tearoff=False)
        self.mask_menu.add_command(label="删除整条蒙版", command=self._delete_context_stroke)
        self.mask_menu.add_command(label="锁定/解锁整条蒙版", command=self._toggle_context_stroke_lock)

    def _scale_row(self, parent: ttk.Frame, row: int, text: str, variable: tk.Variable, low: float, high: float) -> None:
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(parent, from_=low, to=high, variable=variable, orient="horizontal", command=lambda _v: self._global_parameter_changed()).grid(row=row, column=1, sticky="ew", padx=5, pady=(6, 0))
        ttk.Label(parent, textvariable=variable, width=6).grid(row=row, column=2, pady=(6, 0))

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-s>", lambda _e: self.save_project())
        self.bind("<Command-s>", lambda _e: self.save_project())
        self.bind("<Control-o>", lambda _e: self.load_project())
        self.bind("<Command-o>", lambda _e: self.load_project())
        self.bind("<Return>", lambda _e: self.accept_current())
        self.bind("<Delete>", lambda _e: self.reject_current())
        self.bind("<space>", lambda _e: self.toggle_lock())
        self.bind("<Left>", lambda _e: self._select_relative(-1))
        self.bind("<Right>", lambda _e: self._select_relative(1))
        self.bind("b", lambda _e: self._set_tool("brush"))
        self.bind("B", lambda _e: self._set_tool("brush"))
        self.bind("e", lambda _e: self._set_tool("eraser"))
        self.bind("E", lambda _e: self._set_tool("eraser"))
        self.bind("<Control-z>", lambda _e: self.undo_mask())
        self.bind("<Command-z>", lambda _e: self.undo_mask())
        self.bind("<Control-y>", lambda _e: self.redo_mask())
        self.bind("<Command-Shift-z>", lambda _e: self.redo_mask())
        self.bind("<bracketleft>", lambda _e: self._change_tool_width(-2))
        self.bind("<bracketright>", lambda _e: self._change_tool_width(2))
        self.bind("<Shift-bracketleft>", lambda _e: self._change_feather(-2))
        self.bind("<Shift-bracketright>", lambda _e: self._change_feather(2))
        self.bind("<KeyPress-Alt_L>", self._temporary_alt_press)
        self.bind("<KeyRelease-Alt_L>", self._temporary_alt_release)
        self.bind("<KeyPress-Alt_R>", self._temporary_alt_press)
        self.bind("<KeyRelease-Alt_R>", self._temporary_alt_release)
        self.bind("<KeyPress-h>", self._mask_hold_press)
        self.bind("<KeyRelease-h>", self._mask_hold_release)
        self.bind("<KeyPress-H>", self._mask_hold_press)
        self.bind("<KeyRelease-H>", self._mask_hold_release)
        self.bind("1", lambda _e: self._set_preview_mode("original"))
        self.bind("2", lambda _e: self._set_preview_mode("residual"))
        self.bind("3", lambda _e: self._set_preview_mode("effect"))
        self.bind("<Escape>", lambda _e: self._cancel_active_stroke())
        self.bind("<Control-Return>", lambda _e: self.export())
        self.bind("<Command-Return>", lambda _e: self.export())

    def _mode_changed(self) -> None:
        enabled = INPUT_MODES.get(self.input_mode_label.get()) == "clean_video"
        state = "normal" if enabled else "disabled"
        self.clean_entry.configure(state=state)
        self.clean_button.configure(state=state)
        self._schedule_autosave()

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("视频", "*.mp4 *.mov *.m4v *.avi *.mkv"), ("所有文件", "*.*")])
        if path:
            self.video_path.set(path)
            if not self.output_dir.get():
                self.output_dir.set(str(Path(path).parent / "MeteorStudio_Output"))

    def _browse_clean_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("视频", "*.mp4 *.mov *.m4v *.avi *.mkv"), ("所有文件", "*.*")])
        if path:
            self.clean_video_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _validate_paths(self) -> tuple[Path, Path | None, Path]:
        video = Path(self.video_path.get()).expanduser()
        if not video.is_file() or video.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("请选择有效的流星视频")
        clean = None
        if INPUT_MODES.get(self.input_mode_label.get()) == "clean_video":
            clean = Path(self.clean_video_path.get()).expanduser()
            if not clean.is_file():
                raise ValueError("当前模式需要选择干净参考视频")
            if clean.resolve() == video.resolve():
                raise ValueError("流星视频和干净参考视频不能是同一个文件")
        output_text = self.output_dir.get().strip()
        if not output_text:
            output = video.parent / "MeteorStudio_Output"
            self.output_dir.set(str(output))
        else:
            output = Path(output_text).expanduser()
        output.mkdir(parents=True, exist_ok=True)
        return video, clean, output

    def _start_worker(self, worker: Callable[[], tuple], label: str) -> None:
        if self.busy:
            messagebox.showwarning(self.title(), "当前任务尚未完成")
            return
        self.busy = True
        self.progress["value"] = 0
        self.status.set(label)

        def run() -> None:
            try:
                self.work_queue.put(worker())
            except Exception as exc:
                self.work_queue.put(("error", str(exc), traceback.format_exc()))

        threading.Thread(target=run, daemon=True).start()

    def analyze(self) -> None:
        try:
            video, clean, _output = self._validate_paths()
            ignore_bottom = float(self.ignore_bottom.get())
        except Exception as exc:
            show_copyable_error(self.title(), str(exc), parent=self)
            return

        def worker() -> tuple:
            info, events = analyze_video(
                video,
                clean,
                ignore_bottom,
                lambda value, text: self.work_queue.put(("progress", value, text)),
            )
            return "analysis_done", info, events

        self._start_worker(worker, "开始分析视频…")

    def _apply_threshold(self) -> None:
        value = int(self.threshold.get())
        for event in self.events:
            if event.locked or event.decision != "auto":
                continue
            event.accepted = event.score >= value
        self._populate_tree()
        self._schedule_autosave()

    def _threshold_changed(self, _value=None) -> None:
        self.after_idle(self._apply_threshold)

    def _populate_tree(self) -> None:
        selected_frame = self.current_event.frame if self.current_event else None
        self.tree.delete(*self.tree.get_children())
        fps = self.info.fps if self.info else 25.0
        for event in self.events:
            state = "保留" if event.accepted else "排除"
            if event.decision == "auto":
                state = "自动" + state
            self.tree.insert(
                "", "end", iid=str(event.frame), text=str(event.frame),
                values=(f"{event.frame / fps:.2f}s", event.score, state, "🔒" if event.locked else ""),
            )
        if selected_frame is not None and self.tree.exists(str(selected_frame)):
            self.tree.selection_set(str(selected_frame))
            self.tree.see(str(selected_frame))

    def _tree_selected(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        frame = int(selection[0])
        self.current_event = next((event for event in self.events if event.frame == frame), None)
        if self.current_event:
            offsets = event_offsets(self.current_event, self._current_settings())
            self.source_offset.set(offsets[0] if offsets else 0)
        self._load_current_frame()

    def _load_current_frame(self) -> None:
        if self.current_event is None or not self.video_path.get():
            return
        try:
            actual_frame = self.current_event.frame + int(self.source_offset.get())
            self.preview_frame = read_video_frame(Path(self.video_path.get()), actual_frame)
            self._render_current()
            strokes = ensure_event_strokes(self.current_event, self._current_settings())
            count = sum(stroke.frame_offset == self.source_offset.get() for stroke in strokes)
            self.status.set(
                f"事件第 {self.current_event.frame} 帧，当前素材帧 {actual_frame}，分数 {self.current_event.score}；"
                f"蒙版 {count} 条。"
            )
        except Exception as exc:
            self.status.set(str(exc))

    def _current_settings(self) -> EventSettings:
        return EventSettings(
            effect_mode=self.effect_mode.get(),
            local_speed=float(self.local_speed.get()),
            hold_seconds=float(self.hold_seconds.get()),
            fade_seconds=float(self.fade_seconds.get()),
            brightness=float(self.brightness.get()),
            mask_width=int(self.mask_width.get()),
            mask_feather=int(self.mask_feather.get()),
            curve=self.curve.get(),
            curve_start=float(self.curve_start.get()),
            curve_mid=float(self.curve_mid.get()),
            curve_end=float(self.curve_end.get()),
        )

    def _preview_array(self) -> np.ndarray | None:
        if self.preview_frame is None or self.current_event is None:
            return self.preview_frame
        mode = self.preview_mode.get()
        if mode == "original":
            return self.preview_frame.copy()
        try:
            frame = self.current_event.frame + int(self.source_offset.get())
            video = Path(self.video_path.get())
            if INPUT_MODES.get(self.input_mode_label.get()) == "clean_video" and self.clean_video_path.get():
                background = read_video_frame(Path(self.clean_video_path.get()), frame)
            else:
                previous = read_video_frame(video, max(0, frame - 1))
                following = read_video_frame(video, min((self.info.frames - 1) if self.info else frame + 1, frame + 1))
                background = ((previous.astype(np.uint16) + following.astype(np.uint16)) // 2).astype(np.uint8)
            settings = resolve_settings(self.current_event, self._current_settings())
            layer = build_residual_layer(self.current_event, self.preview_frame, background, settings, int(self.source_offset.get()))
            if layer is None:
                return self.preview_frame.copy()
            if mode == "residual":
                result = np.zeros_like(self.preview_frame)
                patch = np.clip(layer.residual * 4.0 + 18.0, 0, 255).astype(np.uint8)
                result[layer.y0:layer.y1, layer.x0:layer.x1] = patch
                return result
            result = background.astype(np.float32)
            result[layer.y0:layer.y1, layer.x0:layer.x1] += layer.residual
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception:
            return self.preview_frame.copy()

    def _render_current(self) -> None:
        array = self._preview_array()
        if array is None or not hasattr(self, "canvas"):
            return
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self.current_event and self.mask_visible:
            strokes = [
                stroke for stroke in ensure_event_strokes(self.current_event, self._current_settings())
                if stroke.frame_offset == int(self.source_offset.get())
            ]
            mask = build_stroke_mask(strokes, rgb.shape[1], rgb.shape[0], self._current_settings())
            alpha = (mask * 0.42)[..., None]
            red = np.zeros_like(rgb, dtype=np.float32)
            red[..., 0] = 255
            rgb = np.uint8(np.clip(rgb.astype(np.float32) * (1.0 - alpha) + red * alpha, 0, 255))
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        height, width = rgb.shape[:2]
        scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, round(width * scale))
        display_height = max(1, round(height * scale))
        x0 = (canvas_width - display_width) // 2
        y0 = (canvas_height - display_height) // 2
        image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(x0, y0, anchor="nw", image=self.preview_photo)
        self.display_box = (x0, y0, display_width, display_height)
        if self.current_event and self.mask_visible:
            for stroke in ensure_event_strokes(self.current_event, self._current_settings()):
                if stroke.frame_offset != int(self.source_offset.get()) or len(stroke.points) < 1:
                    continue
                coords = [value for point in stroke.points for value in (x0 + point[0] * display_width, y0 + point[1] * display_height)]
                if len(coords) >= 4:
                    self.canvas.create_line(*coords, fill="#75ff8a" if stroke.locked else ("#ff9f0a" if stroke.erase else "#ff453a"), width=2, smooth=True)
        self.cursor_item = None

    def _canvas_normalized(self, x: int, y: int) -> tuple[float, float] | None:
        x0, y0, width, height = self.display_box
        if not (x0 <= x <= x0 + width and y0 <= y <= y0 + height):
            return None
        return ((x - x0) / max(1, width), (y - y0) / max(1, height))

    def _push_history(self) -> None:
        if self.current_event is None:
            return
        frame = self.current_event.frame
        self.history.setdefault(frame, []).append(copy.deepcopy(ensure_event_strokes(self.current_event, self._current_settings())))
        self.history[frame] = self.history[frame][-50:]
        self.redo_history[frame] = []

    def _stroke_start(self, event) -> None:
        point = self._canvas_normalized(event.x, event.y)
        if self.current_event is None or point is None:
            return
        self._push_history()
        key = (self.current_event.frame, int(self.source_offset.get()))
        normalized = [point[0], point[1]]
        if event.state & 0x0001 and key in self.shift_anchors:
            self.active_points = [self.shift_anchors[key], normalized]
        else:
            self.active_points = [normalized]
        self.active_canvas_points = [(event.x, event.y)]
        self.active_item = self.canvas.create_line(event.x, event.y, event.x, event.y, fill="#ff9f0a" if self.edit_mode.get() == "eraser" else "#ff453a", width=3, smooth=True)

    def _stroke_move(self, event) -> None:
        if not self.active_points:
            return
        point = self._canvas_normalized(event.x, event.y)
        if point is None:
            return
        if math.hypot(point[0] - self.active_points[-1][0], point[1] - self.active_points[-1][1]) > 0.001:
            self.active_points.append([point[0], point[1]])
            self.active_canvas_points.append((event.x, event.y))
            if self.active_item and len(self.active_canvas_points) >= 2:
                self.canvas.coords(self.active_item, *[v for p in self.active_canvas_points for v in p])

    def _stroke_end(self, event) -> None:
        if self.current_event is None or not self.active_points:
            return
        point = self._canvas_normalized(event.x, event.y)
        if point is not None and (len(self.active_points) == 1 or point != tuple(self.active_points[-1])):
            self.active_points.append([point[0], point[1]])
        points = self.active_points
        self.active_points = []
        self.active_canvas_points = []
        self.active_item = None
        if not points:
            self._render_current()
            return
        mode = self.edit_mode.get()
        stroke = VideoStroke(
            points=points,
            width=int(self.eraser_width.get() if mode == "eraser" else self.brush_width.get()),
            feather=int(self.brush_feather.get()),
            erase=mode == "eraser",
            frame_offset=int(self.source_offset.get()),
        )
        ensure_event_strokes(self.current_event, self._current_settings()).append(stroke)
        self.shift_anchors[(self.current_event.frame, int(self.source_offset.get()))] = points[-1]
        self.current_event.accepted = True
        self.current_event.decision = "accepted"
        self._populate_tree()
        self._render_current()
        self._schedule_autosave()

    @staticmethod
    def _point_stroke_distance(point: tuple[float, float], stroke: VideoStroke) -> float:
        if not stroke.points:
            return 999.0
        if len(stroke.points) == 1:
            return math.hypot(point[0] - stroke.points[0][0], point[1] - stroke.points[0][1])
        best = 999.0
        for first, second in zip(stroke.points, stroke.points[1:]):
            dx, dy = second[0] - first[0], second[1] - first[1]
            denom = dx * dx + dy * dy
            t = 0.0 if denom == 0 else float(np.clip(((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denom, 0, 1))
            best = min(best, math.hypot(point[0] - first[0] - t * dx, point[1] - first[1] - t * dy))
        return best

    def _show_stroke_menu(self, event) -> None:
        point = self._canvas_normalized(event.x, event.y)
        if self.current_event is None or point is None:
            return
        strokes = ensure_event_strokes(self.current_event, self._current_settings())
        candidates = [(self._point_stroke_distance(point, stroke), index) for index, stroke in enumerate(strokes) if stroke.frame_offset == int(self.source_offset.get())]
        if not candidates or min(candidates)[0] > 0.045:
            return
        self.context_stroke_index = min(candidates)[1]
        self.mask_menu.tk_popup(event.x_root, event.y_root)

    def _delete_stroke_at(self, event) -> str:
        point = self._canvas_normalized(event.x, event.y)
        if self.current_event is None or point is None:
            return "break"
        strokes = ensure_event_strokes(self.current_event, self._current_settings())
        candidates = [(self._point_stroke_distance(point, stroke), index) for index, stroke in enumerate(strokes) if stroke.frame_offset == int(self.source_offset.get())]
        if candidates and min(candidates)[0] <= 0.045:
            self.context_stroke_index = min(candidates)[1]
            self._delete_context_stroke()
        return "break"

    def _delete_context_stroke(self) -> None:
        if self.current_event is None or self.context_stroke_index is None:
            return
        strokes = ensure_event_strokes(self.current_event, self._current_settings())
        if 0 <= self.context_stroke_index < len(strokes) and not strokes[self.context_stroke_index].locked:
            self._push_history()
            del strokes[self.context_stroke_index]
            self._render_current()
            self._schedule_autosave()

    def _toggle_context_stroke_lock(self) -> None:
        if self.current_event is None or self.context_stroke_index is None:
            return
        strokes = ensure_event_strokes(self.current_event, self._current_settings())
        if 0 <= self.context_stroke_index < len(strokes):
            strokes[self.context_stroke_index].locked = not strokes[self.context_stroke_index].locked
            self._render_current()
            self._schedule_autosave()

    def undo_mask(self) -> None:
        if self.current_event is None:
            return
        frame = self.current_event.frame
        stack = self.history.get(frame, [])
        if not stack:
            return
        self.redo_history.setdefault(frame, []).append(copy.deepcopy(ensure_event_strokes(self.current_event, self._current_settings())))
        self.current_event.strokes = stack.pop()
        self.current_event.lines = []
        self._render_current()
        self._schedule_autosave()

    def redo_mask(self) -> None:
        if self.current_event is None:
            return
        frame = self.current_event.frame
        stack = self.redo_history.get(frame, [])
        if not stack:
            return
        self.history.setdefault(frame, []).append(copy.deepcopy(ensure_event_strokes(self.current_event, self._current_settings())))
        self.current_event.strokes = stack.pop()
        self.current_event.lines = []
        self._render_current()
        self._schedule_autosave()

    def _set_tool(self, mode: str) -> None:
        self.edit_mode.set(mode)
        self._tool_changed()

    def _set_preview_mode(self, mode: str) -> None:
        self.preview_mode.set(mode)
        self._render_current()

    def _cancel_active_stroke(self) -> None:
        self.active_points = []
        self.active_canvas_points = []
        self.active_item = None
        self._render_current()

    def _tool_changed(self) -> None:
        self._schedule_autosave()

    def _change_tool_width(self, delta: int) -> None:
        variable = self.eraser_width if self.edit_mode.get() == "eraser" else self.brush_width
        variable.set(int(np.clip(variable.get() + delta, 2, 240)))

    def _change_feather(self, delta: int) -> None:
        self.brush_feather.set(int(np.clip(self.brush_feather.get() + delta, 0, 100)))

    def _temporary_alt_press(self, _event=None) -> None:
        if self.alt_previous_mode is None:
            self.alt_previous_mode = self.edit_mode.get()
            self.edit_mode.set("eraser" if self.alt_previous_mode == "brush" else "brush")

    def _temporary_alt_release(self, _event=None) -> None:
        if self.alt_previous_mode is not None:
            self.edit_mode.set(self.alt_previous_mode)
            self.alt_previous_mode = None

    def _mask_hold_press(self, _event=None) -> None:
        if self.mask_visible:
            self.mask_visible = False
            self._render_current()

    def _mask_hold_release(self, _event=None) -> None:
        if not self.mask_visible:
            self.mask_visible = True
            self._render_current()

    def _update_brush_cursor(self, event) -> None:
        if self._canvas_normalized(event.x, event.y) is None:
            self._hide_brush_cursor()
            return
        if self.cursor_item is not None:
            self.canvas.delete(self.cursor_item)
        x0, _y0, display_width, _display_height = self.display_box
        source_width = self.preview_frame.shape[1] if self.preview_frame is not None else max(1, display_width)
        width = self.eraser_width.get() if self.edit_mode.get() == "eraser" else self.brush_width.get()
        radius = max(2.0, width * display_width / source_width / 2.0)
        color = "#ffb340" if self.edit_mode.get() == "eraser" else "#ffffff"
        self.cursor_item = self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, outline=color, width=2)

    def _hide_brush_cursor(self, _event=None) -> None:
        if self.cursor_item is not None:
            self.canvas.delete(self.cursor_item)
            self.cursor_item = None

    def _change_source_offset(self, delta: int) -> None:
        if self.current_event is None:
            return
        offsets = event_offsets(self.current_event, self._current_settings())
        current = int(self.source_offset.get())
        if current not in offsets:
            target = offsets[0]
        else:
            target = offsets[int(np.clip(offsets.index(current) + delta, 0, len(offsets) - 1))]
        self.source_offset.set(target)
        self._load_current_frame()

    def merge_selected_events(self) -> None:
        frames = sorted(int(item) for item in self.tree.selection())
        selected = [event for event in self.events if event.frame in frames]
        if len(selected) < 2:
            messagebox.showinfo(self.title(), "请在左侧用 Ctrl/Shift 选择至少两个连续流星帧")
            return
        if any(second.frame - first.frame > 2 for first, second in zip(selected, selected[1:])):
            messagebox.showwarning(self.title(), "只能合并相邻或间隔一帧的候选")
            return
        start = selected[0].frame
        merged_strokes: list[VideoStroke] = []
        for item in selected:
            for stroke in ensure_event_strokes(item, self._current_settings()):
                copied = copy.deepcopy(stroke)
                copied.frame_offset += item.frame - start
                merged_strokes.append(copied)
        merged = copy.deepcopy(selected[0])
        merged.frame = start
        merged.score = max(event.score for event in selected)
        merged.lines = []
        merged.strokes = merged_strokes
        merged.accepted = True
        merged.decision = "accepted"
        self.events = [event for event in self.events if event not in selected] + [merged]
        self.events.sort(key=lambda event: event.frame)
        self.current_event = merged
        self.source_offset.set(event_offsets(merged, self._current_settings())[0])
        self._populate_tree()
        self.tree.selection_set(str(merged.frame))
        self._load_current_frame()
        self._schedule_autosave()

    def split_current_event(self) -> None:
        if self.current_event is None:
            return
        strokes = ensure_event_strokes(self.current_event, self._current_settings())
        offsets = sorted({stroke.frame_offset for stroke in strokes})
        if len(offsets) < 2:
            messagebox.showinfo(self.title(), "当前事件只有一个素材帧，无需拆分")
            return
        original = self.current_event
        split: list[VideoEvent] = []
        for offset in offsets:
            item = copy.deepcopy(original)
            item.frame = original.frame + offset
            item.lines = []
            item.strokes = []
            for stroke in strokes:
                if stroke.frame_offset == offset:
                    copied = copy.deepcopy(stroke)
                    copied.frame_offset = 0
                    item.strokes.append(copied)
            split.append(item)
        self.events = [event for event in self.events if event is not original] + split
        self.events.sort(key=lambda event: event.frame)
        self.current_event = split[0]
        self.source_offset.set(0)
        self._populate_tree()
        self.tree.selection_set(str(split[0].frame))
        self._load_current_frame()
        self._schedule_autosave()

    def accept_current(self) -> None:
        if self.current_event is None:
            return
        self.current_event.accepted = True
        self.current_event.decision = "accepted"
        self._populate_tree()
        self._render_current()
        self._schedule_autosave()

    def reject_current(self) -> None:
        if self.current_event is None or self.current_event.locked:
            return
        self.current_event.accepted = False
        self.current_event.decision = "rejected"
        self._populate_tree()
        self._render_current()
        self._schedule_autosave()

    def toggle_lock(self) -> None:
        if self.current_event is None:
            return
        self.current_event.locked = not self.current_event.locked
        if self.current_event.locked:
            self.current_event.accepted = True
            self.current_event.decision = "accepted"
        self._populate_tree()
        self._render_current()
        self._schedule_autosave()

    def _select_relative(self, delta: int) -> None:
        children = self.tree.get_children()
        if not children:
            return
        current = self.tree.selection()
        index = children.index(current[0]) if current and current[0] in children else 0
        target = children[int(np.clip(index + delta, 0, len(children) - 1))]
        self.tree.selection_set(target)
        self.tree.see(target)
        self._tree_selected()

    def add_manual_event(self) -> None:
        if self.info is None:
            messagebox.showwarning(self.title(), "请先分析或载入一个视频项目")
            return
        frame = self.current_event.frame if self.current_event else 0
        value = simpledialog.askinteger("手工添加流星", "帧编号", initialvalue=frame, minvalue=0, maxvalue=self.info.frames - 1, parent=self)
        if value is None:
            return
        existing = next((event for event in self.events if event.frame == value), None)
        if existing is None:
            existing = VideoEvent(value, 100, accepted=True, locked=True, decision="accepted")
            self.events.append(existing)
            self.events.sort(key=lambda event: event.frame)
        self.current_event = existing
        self._populate_tree()
        self.tree.selection_set(str(value))
        self.tree.see(str(value))
        self._load_current_frame()
        self._schedule_autosave()

    def edit_current_settings(self) -> None:
        if self.current_event is None:
            messagebox.showwarning(self.title(), "请先选择一个候选")
            return
        event = self.current_event
        defaults = self._current_settings()
        resolved = resolve_settings(event, defaults)
        dialog = tk.Toplevel(self)
        dialog.title(f"第 {event.frame} 帧 — 单独参数")
        dialog.transient(self)
        dialog.grab_set()
        use_custom = tk.BooleanVar(value=event.use_custom)
        variables = {
            "effect_mode": tk.StringVar(value=resolved.effect_mode),
            "local_speed": tk.DoubleVar(value=resolved.local_speed),
            "hold_seconds": tk.DoubleVar(value=resolved.hold_seconds),
            "fade_seconds": tk.DoubleVar(value=resolved.fade_seconds),
            "brightness": tk.DoubleVar(value=resolved.brightness),
            "mask_width": tk.IntVar(value=resolved.mask_width),
            "mask_feather": tk.IntVar(value=resolved.mask_feather),
            "curve": tk.StringVar(value=resolved.curve),
            "curve_start": tk.DoubleVar(value=resolved.curve_start),
            "curve_mid": tk.DoubleVar(value=resolved.curve_mid),
            "curve_end": tk.DoubleVar(value=resolved.curve_end),
        }
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Checkbutton(frame, text="当前流星使用单独参数", variable=use_custom).grid(row=0, column=0, columnspan=2, sticky="w")
        fields = (
            ("效果模式", "effect_mode", EFFECT_MODES),
            ("局部速度(%)", "local_speed", None),
            ("清晰停留(秒)", "hold_seconds", None),
            ("淡出时间(秒)", "fade_seconds", None),
            ("亮度", "brightness", None),
            ("蒙版宽度", "mask_width", None),
            ("蒙版羽化", "mask_feather", None),
            ("时间曲线", "curve", CURVE_MODES),
            ("起始速度", "curve_start", None),
            ("中段速度", "curve_mid", None),
            ("结束速度", "curve_end", None),
        )
        for row, (label, key, choices) in enumerate(fields, start=1):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if choices:
                ttk.Combobox(frame, textvariable=variables[key], values=choices, state="readonly").grid(row=row, column=1, sticky="ew")
            else:
                ttk.Entry(frame, textvariable=variables[key]).grid(row=row, column=1, sticky="ew")
        frame.columnconfigure(1, weight=1)

        def save() -> None:
            event.use_custom = bool(use_custom.get())
            if event.use_custom:
                for key, variable in variables.items():
                    setattr(event, key, variable.get())
            dialog.destroy()
            self._render_current()
            self._schedule_autosave()

        buttons = ttk.Frame(frame)
        buttons.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="保存", command=save).pack(side="right", padx=6)

    def _global_parameter_changed(self) -> None:
        self._render_current()
        self._schedule_autosave()

    def _project_data(self) -> dict:
        return {
            "version": VIDEO_PROJECT_VERSION,
            "app": "Meteor Studio",
            "video_path": self.video_path.get(),
            "clean_video_path": self.clean_video_path.get(),
            "output_dir": self.output_dir.get(),
            "input_mode": INPUT_MODES.get(self.input_mode_label.get(), "single"),
            "threshold": int(self.threshold.get()),
            "ignore_bottom": float(self.ignore_bottom.get()),
            "info": asdict(self.info) if self.info else None,
            "defaults": asdict(self._current_settings()),
            "overall_speed": float(self.overall_speed.get()),
            "keep_audio": bool(self.keep_audio.get()),
            "encoding_quality": self.encoding_quality.get(),
            "editor": {
                "mode": self.edit_mode.get(),
                "brush_width": int(self.brush_width.get()),
                "eraser_width": int(self.eraser_width.get()),
                "feather": int(self.brush_feather.get()),
            },
            "events": [asdict(event) for event in self.events],
        }

    def _apply_project_data(self, data: dict) -> None:
        self.video_path.set(data.get("video_path", ""))
        self.clean_video_path.set(data.get("clean_video_path", ""))
        self.output_dir.set(data.get("output_dir", ""))
        mode_code = data.get("input_mode", "single")
        self.input_mode_label.set(next((label for label, code in INPUT_MODES.items() if code == mode_code), next(iter(INPUT_MODES))))
        self.threshold.set(int(data.get("threshold", 72)))
        self.ignore_bottom.set(float(data.get("ignore_bottom", 15.0)))
        info = data.get("info")
        self.info = VideoInfo(**info) if info else None
        defaults = data.get("defaults", {})
        for key, variable in (
            ("effect_mode", self.effect_mode), ("local_speed", self.local_speed),
            ("hold_seconds", self.hold_seconds), ("fade_seconds", self.fade_seconds),
            ("brightness", self.brightness), ("mask_width", self.mask_width),
            ("mask_feather", self.mask_feather), ("curve", self.curve),
            ("curve_start", self.curve_start), ("curve_mid", self.curve_mid),
            ("curve_end", self.curve_end),
        ):
            if key in defaults:
                variable.set(defaults[key])
        self.overall_speed.set(float(data.get("overall_speed", 100.0)))
        self.keep_audio.set(bool(data.get("keep_audio", False)))
        self.encoding_quality.set(data.get("encoding_quality", ENCODING_QUALITIES[0]))
        editor = data.get("editor", {})
        self.edit_mode.set(editor.get("mode", "brush"))
        self.brush_width.set(int(editor.get("brush_width", 24)))
        self.eraser_width.set(int(editor.get("eraser_width", 42)))
        self.brush_feather.set(int(editor.get("feather", 14)))
        self.events = []
        for raw in data.get("events", []):
            item = dict(raw)
            item["strokes"] = [decode_stroke(stroke) for stroke in item.get("strokes", [])]
            self.events.append(VideoEvent(**item))
        self.current_event = None
        self.preview_frame = None
        self._mode_changed()
        self._populate_tree()
        self.status.set(f"已载入视频项目：{len(self.events)} 个候选")

    def _schedule_autosave(self) -> None:
        if not hasattr(self, "autosave_file"):
            return
        if self.autosave_after_id:
            self.after_cancel(self.autosave_after_id)
        self.autosave_after_id = self.after(900, self._write_autosave)

    def _write_autosave(self) -> None:
        self.autosave_after_id = None
        try:
            data = json.dumps(self._project_data(), ensure_ascii=False, indent=2)
            temporary = self.autosave_file.with_suffix(".writing.json")
            temporary.write_text(data, encoding="utf-8")
            temporary.replace(self.autosave_file)
        except Exception:
            pass

    def _restore_autosave(self) -> None:
        if not self.autosave_file.is_file():
            return
        try:
            self._apply_project_data(json.loads(self.autosave_file.read_text(encoding="utf-8")))
            self.status.set("已自动恢复上次视频项目")
        except Exception:
            pass

    def save_project(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("流星影像项目", "*.json")], initialfile="流星视频项目.json")
        if path:
            Path(path).write_text(json.dumps(self._project_data(), ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.set(f"项目已保存：{path}")

    def load_project(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("流星影像项目", "*.json"), ("所有文件", "*.*")])
        if not path:
            return
        try:
            self._apply_project_data(json.loads(Path(path).read_text(encoding="utf-8")))
        except Exception as exc:
            show_copyable_error(self.title(), f"载入失败：{exc}", parent=self)

    def export(self) -> None:
        try:
            video, clean, output_dir = self._validate_paths()
            accepted = [
                event for event in self.events
                if event.accepted and any(not stroke.erase for stroke in ensure_event_strokes(event, self._current_settings()))
            ]
            if not accepted:
                raise ValueError("没有已确认的流星候选")
            defaults = self._current_settings()
            speed = float(self.overall_speed.get()) / 100.0
            if speed <= 0:
                raise ValueError("整体速度必须大于0")
            keep_audio = bool(self.keep_audio.get())
            encoding_quality = self.encoding_quality.get()
            events_snapshot = [VideoEvent(**json.loads(json.dumps(asdict(event)))) for event in self.events]
        except Exception as exc:
            show_copyable_error(self.title(), str(exc), parent=self)
            return
        output_path = output_dir / f"{video.stem}_meteor_dynamic_{datetime.now():%Y%m%d_%H%M%S}.mp4"

        def worker() -> tuple:
            info, clips = prepare_layers(
                video, clean, events_snapshot, defaults,
                lambda value, text: self.work_queue.put(("progress", value, text)),
            )
            if len(clips) != len(accepted):
                self.work_queue.put(("progress", 24.0, f"警告：{len(accepted) - len(clips)} 个候选没有提取到有效亮度层"))
            result = render_video(
                video, output_path, info, clips, speed, keep_audio,
                lambda value, text: self.work_queue.put(("progress", value, text)),
                encoding_quality,
            )
            return "export_done", result, len(clips)

        self._start_worker(worker, "开始准备流星亮度层…")

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.work_queue.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, value, text = item
                    self.progress["value"] = value
                    self.status.set(text)
                elif kind == "analysis_done":
                    _, self.info, self.events = item
                    self.busy = False
                    self.current_event = None
                    self.preview_frame = None
                    self._apply_threshold()
                    accepted = sum(event.accepted for event in self.events)
                    self.status.set(f"分析完成：{len(self.events)} 个候选，当前自动保留 {accepted} 个。请逐个审核。")
                    self.progress["value"] = 100
                    self._schedule_autosave()
                elif kind == "export_done":
                    _, result, count = item
                    self.busy = False
                    self.progress["value"] = 100
                    self.status.set(f"导出完成：{result}")
                    if messagebox.askyesno(
                        self.title(),
                        f"已导出 {count} 个流星事件：\n{result}\n\n是否打开所在文件夹？",
                        parent=self,
                    ):
                        self._open_output_folder(result)
                elif kind == "error":
                    _, text, details = item
                    self.busy = False
                    self.status.set("处理失败")
                    show_copyable_error(
                        self.title(), text, parent=self, details=details
                    )
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(150, self._poll_queue)

    def _open_output_folder(self, path: str | Path | None = None) -> None:
        target = path or self.output_dir.get().strip()
        try:
            open_folder(target)
        except Exception as exc:
            show_copyable_error(self.title(), str(exc), parent=self)


def open_video_workspace(master: tk.Misc) -> VideoMeteorWindow:
    return VideoMeteorWindow(master)
