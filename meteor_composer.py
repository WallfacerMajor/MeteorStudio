from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk
from alignment_workspace import open_alignment_workspace
from video_meteor import open_video_workspace


APP_NAME = "流星影像工坊"
PROJECT_VERSION = 18
TIFF_SUFFIXES = {".tif", ".tiff"}


@dataclass
class Stroke:
    points: list[tuple[float, float]]
    width: int
    feather: int
    erase: bool = False
    locked: bool = False
    auto_score: int | None = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0
    length_scale: float = 1.0
    width_scale: float = 1.0
    opacity: float = 1.0


def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"不支持的图像数组形状：{array.shape}")
    return array[..., :3]


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in TIFF_SUFFIXES:
        return normalize_array(tifffile.imread(path))
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def to_uint16(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint16:
        return array.copy()
    if array.dtype == np.uint8:
        return array.astype(np.uint16) * 257
    result = array.astype(np.float32)
    finite = result[np.isfinite(result)]
    if finite.size == 0:
        raise ValueError("图像没有有效像素")
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(finite)) <= 1.5:
            result *= 65535.0
    return np.nan_to_num(result, nan=0.0, posinf=65535.0, neginf=0.0).clip(0, 65535).astype(np.uint16)


def image_info(path: Path) -> tuple[int, int, str, int]:
    if path.suffix.lower() in TIFF_SUFFIXES:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            shape = page.shape
            dtype = np.dtype(page.dtype)
        if len(shape) == 2:
            height, width = shape
            channels = 1
        elif shape[-1] in (3, 4):
            height, width, channels = shape
        elif shape[0] in (3, 4):
            channels, height, width = shape
        else:
            raise ValueError(f"无法识别 TIFF 尺寸：{shape}")
        return width, height, str(dtype), channels
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        return width, height, image.mode, len(image.getbands())


def atomic_write_tiff(path: Path, array: np.ndarray) -> None:
    temp = path.with_name(path.stem + ".writing" + path.suffix)
    tifffile.imwrite(temp, array, photometric="rgb", compression="deflate")
    with tifffile.TiffFile(temp) as tif:
        if tif.pages[0].shape[:2] != array.shape[:2]:
            raise IOError("TIFF 写入校验失败")
    os.replace(temp, path)


def atomic_write_jpeg(path: Path, array16: np.ndarray, quality: int = 95) -> None:
    temp = path.with_name(path.stem + ".writing" + path.suffix)
    array8 = np.right_shift(array16, 8).astype(np.uint8)
    Image.fromarray(array8, "RGB").save(temp, quality=quality, subsampling=0)
    with Image.open(temp) as check:
        check.verify()
    os.replace(temp, path)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def autosave_file_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return root / "MeteorComposer" / "autosave.json"


def user_model_file_path() -> Path:
    return autosave_file_path().parent / "models" / "meteor_ranker_user.json"


def bundled_resource_path(name: str) -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidate = Path(bundle_root) / name
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parent / name


def build_mask_crop(
    strokes: Iterable[Stroke], width: int, height: int
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    strokes = list(strokes)
    if not strokes:
        return None
    all_points = [(x * (width - 1), y * (height - 1)) for s in strokes for x, y in s.points]
    if not all_points:
        return None
    pad = max(s.width + s.feather * 4 for s in strokes)
    xs, ys = zip(*all_points)
    x0 = max(0, int(min(xs) - pad))
    y0 = max(0, int(min(ys) - pad))
    x1 = min(width, int(max(xs) + pad + 1))
    y1 = min(height, int(max(ys) + pad + 1))
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for stroke in strokes:
        layer = np.zeros_like(mask)
        pts = np.array(
            [[int(x * (width - 1)) - x0, int(y * (height - 1)) - y0] for x, y in stroke.points],
            dtype=np.int32,
        )
        if len(pts) == 1:
            cv2.circle(layer, tuple(pts[0]), max(1, stroke.width // 2), 1.0, -1, cv2.LINE_AA)
        else:
            cv2.polylines(layer, [pts], False, 1.0, max(1, stroke.width), cv2.LINE_AA)
        if stroke.feather > 0:
            sigma = max(0.5, stroke.feather / 2.5)
            layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma, sigmaY=sigma)
        layer = np.clip(layer, 0.0, 1.0)
        if stroke.erase:
            mask *= 1.0 - layer
        else:
            mask = np.maximum(mask, layer)
    return np.clip(mask, 0.0, 1.0), (x0, y0, x1, y1)


def detection_preview(array: np.ndarray, max_dimension: int = 1600) -> tuple[np.ndarray, float]:
    rgb16 = to_uint16(array)
    height, width = rgb16.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    if scale < 1.0:
        rgb16 = cv2.resize(rgb16, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return np.right_shift(rgb16, 8).astype(np.uint8), scale


def expand_trail_segment(
    start: tuple[int, int], end: tuple[int, int], width: int, height: int,
    extension_ratio: float = 0.38
) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Expand an automatically detected core line to include its head and faint tail."""
    vector = np.array((end[0] - start[0], end[1] - start[1]), dtype=np.float32)
    length = float(np.hypot(vector[0], vector[1]))
    if length < 1.0:
        return start, end, length
    direction = vector / length
    extra = max(6.0, length * extension_ratio)
    first = np.array(start, dtype=np.float32) - direction * extra
    second = np.array(end, dtype=np.float32) + direction * extra
    first[0], first[1] = np.clip(first[0], 0, width - 1), np.clip(first[1], 0, height - 1)
    second[0], second[1] = np.clip(second[0], 0, width - 1), np.clip(second[1], 0, height - 1)
    first_point = tuple(int(v) for v in np.rint(first))
    second_point = tuple(int(v) for v in np.rint(second))
    return first_point, second_point, length


def count_true_runs(values: np.ndarray) -> int:
    padded = np.pad(values.astype(np.int8), (1, 1))
    return int(np.count_nonzero(np.diff(padded) == 1))


ML_FEATURE_NAMES = [
    "legacy_score", "length_ratio", "mid_x", "mid_y", "abs_horizontal", "abs_vertical",
    "center_mean", "center_median", "center_q90", "center_q99", "center_std",
    "background_mean", "background_q90", "contrast_mean", "contrast_q90",
    "source_contrast", "base_contrast", "inner_outer_ratio", "peak_mean_ratio",
    "smoothness", "gradient_std", "first_second_ratio", "middle_edge_ratio",
    "active95", "active99", "active997", "runs95", "runs99", "runs997",
    "positive_fraction", "negative_fraction", "signed_mean", "signed_q90", "signed_q10",
]


def prepare_ml_maps(source: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, ...]:
    src = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dst = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    s2, s98 = np.percentile(src, (2, 98))
    b2, b98 = np.percentile(dst, (2, 98))
    mapped = (src - s2) * float((b98 - b2) / max(5.0, s98 - s2)) + b2
    difference = mapped - dst
    sigma = max(12.0, min(src.shape) / 45.0)
    residual = difference - cv2.GaussianBlur(difference, (0, 0), sigmaX=sigma, sigmaY=sigma)
    magnitude = np.abs(residual)
    limits = np.percentile(magnitude, (95.0, 99.0, 99.7)).astype(np.float32)
    return src, dst, residual, magnitude, limits


def candidate_feature_vector(
    maps: tuple[np.ndarray, ...], start: tuple[int, int], end: tuple[int, int], legacy_score: float
) -> np.ndarray:
    src, dst, residual, magnitude, limits = maps
    height, width = magnitude.shape
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = max(1.0, float(np.hypot(dx, dy)))
    direction = np.array((dx / length, dy / length), dtype=np.float32)
    normal = np.array((-direction[1], direction[0]), dtype=np.float32)
    samples = int(np.clip(round(length * 1.25), 48, 256))
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    center_points = np.array(start, np.float32)[None, :] + t[:, None] * np.array((dx, dy), np.float32)[None, :]

    def profile(image: np.ndarray, offset: float) -> np.ndarray:
        points = center_points + normal[None, :] * offset
        xs = points[:, 0].clip(0, width - 1).astype(np.int32)
        ys = points[:, 1].clip(0, height - 1).astype(np.int32)
        return image[ys, xs]

    inner = np.max(np.stack([profile(magnitude, offset) for offset in (-2, 0, 2)]), axis=0)
    outer = np.mean(np.stack([profile(magnitude, offset) for offset in (-20, -16, 16, 20)]), axis=0)
    source_inner = profile(src, 0)
    source_outer = np.mean(np.stack([profile(src, -16), profile(src, 16)]), axis=0)
    base_inner = profile(dst, 0)
    base_outer = np.mean(np.stack([profile(dst, -16), profile(dst, 16)]), axis=0)
    signed = profile(residual, 0)
    centered = inner - float(np.median(inner))
    smoothness = float(np.mean(np.abs(np.diff(inner))) / max(1e-3, np.std(inner)))
    gradient_std = float(np.std(np.diff(inner)) / max(1e-3, np.mean(inner)))
    half = max(1, len(inner) // 2)
    first_second = float((np.mean(inner[:half]) + 1e-3) / (np.mean(inner[half:]) + 1e-3))
    edge_count = max(1, len(inner) // 5)
    edge_mean = float((np.mean(inner[:edge_count]) + np.mean(inner[-edge_count:])) * 0.5)
    middle_mean = float(np.mean(inner[edge_count:-edge_count])) if len(inner) > edge_count * 2 else float(np.mean(inner))
    active = [inner > float(limit) for limit in limits]
    features = [
        float(legacy_score) / 100.0,
        length / max(width, height),
        (start[0] + end[0]) * 0.5 / max(1, width - 1),
        (start[1] + end[1]) * 0.5 / max(1, height - 1),
        abs(direction[0]), abs(direction[1]),
        float(np.mean(inner)), float(np.median(inner)), float(np.percentile(inner, 90)),
        float(np.percentile(inner, 99)), float(np.std(inner)),
        float(np.mean(outer)), float(np.percentile(outer, 90)),
        float(np.mean(inner) - np.mean(outer)),
        float(np.percentile(inner, 90) - np.percentile(outer, 90)),
        float(np.mean(source_inner - source_outer)), float(np.mean(base_inner - base_outer)),
        float((np.mean(inner) + 1e-3) / (np.mean(outer) + 1e-3)),
        float((np.max(inner) + 1e-3) / (np.mean(inner) + 1e-3)),
        smoothness, gradient_std, first_second,
        float((middle_mean + 1e-3) / (edge_mean + 1e-3)),
        *(float(np.mean(values)) for values in active),
        *(float(count_true_runs(values)) / max(1, len(values)) for values in active),
        float(np.mean(signed > 0)), float(np.mean(signed < 0)), float(np.mean(signed)),
        float(np.percentile(signed, 90)), float(np.percentile(signed, 10)),
    ]
    return np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def predict_gradient_boosting(features: np.ndarray, model: dict) -> float:
    raw = float(model["base_raw"])
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            if float(features[tree["feature"][node]]) <= tree["threshold"][node]:
                node = tree["left"][node]
            else:
                node = tree["right"][node]
        raw += float(model["learning_rate"]) * tree["value"][node]
    return float(1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0))))


def load_meteor_ranker() -> dict | None:
    paths = [user_model_file_path(), bundled_resource_path("meteor_ranker.json")]
    for path in paths:
        if path.is_file():
            try:
                model = json.loads(path.read_text(encoding="utf-8"))
                if (model.get("format") == "meteor-gradient-boosting-v1"
                        and model.get("feature_names") == ML_FEATURE_NAMES):
                    model["_personalized"] = path == user_model_file_path()
                    return model
            except (OSError, ValueError, TypeError):
                pass
    return None


def detect_trails(source: np.ndarray, base: np.ndarray, ranked: bool = False):
    src = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dst = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    s2, s98 = np.percentile(src, (2, 98))
    b2, b98 = np.percentile(dst, (2, 98))
    gain = float((b98 - b2) / max(5.0, s98 - s2))
    mapped = (src - s2) * gain + b2
    difference = mapped - dst
    sigma = max(12.0, min(src.shape) / 45.0)
    residual = difference - cv2.GaussianBlur(difference, (0, 0), sigmaX=sigma, sigmaY=sigma)
    magnitude = np.abs(residual)
    height, width = magnitude.shape
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    detected_parts = []
    for low_percentile, high_percentile in ((97.0, 99.90), (93.0, 99.65), (88.0, 99.30)):
        low, high = np.percentile(magnitude, (low_percentile, high_percentile))
        enhanced = np.clip((magnitude - low) * 255.0 / max(1.0, high - low), 0, 255).astype(np.uint8)
        enhanced[int(height * 0.82):] = 0
        lines = detector.detect(enhanced)[0]
        if lines is not None:
            detected_parts.append(lines)
    if not detected_parts:
        return [], 0
    detected = np.concatenate(detected_parts, axis=0)

    min_length = max(18.0, max(height, width) * 0.018)
    base_u8 = np.clip(dst, 0, 255).astype(np.uint8)
    structural_edges = cv2.Canny(base_u8, 35, 90)
    structural_edges = cv2.dilate(structural_edges, np.ones((5, 5), np.uint8)) > 0
    bright_limit = float(np.percentile(dst, 97))
    raw_candidates = []
    for raw in detected[:, 0]:
        x1, y1, x2, y2 = (float(v) for v in raw)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        start, end = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        samples = max(30, int(length))
        xs = np.linspace(start[0], end[0], samples).clip(0, width - 1).astype(int)
        ys = np.linspace(start[1], end[1], samples).clip(0, height - 1).astype(int)
        if float(np.median(ys)) > height * 0.79:
            continue
        edge_fraction = float(np.mean(structural_edges[ys, xs]))
        bright_fraction = float(np.mean(dst[ys, xs] > bright_limit))
        if edge_fraction > 0.92 or bright_fraction > 0.24:
            continue
        dx, dy = end[0] - start[0], end[1] - start[1]
        angle = float(np.arctan2(dy, dx))
        horizontal = abs(np.sin(angle)) < 0.20
        vertical = abs(np.cos(angle)) < 0.20
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        if length < max(height, width) * 0.025 and midpoint[1] > height * 0.68:
            continue
        # JPEG borders and broad tone edits often leave short horizontal residuals.
        if horizontal and (midpoint[0] < width * 0.09 or midpoint[1] < height * 0.055):
            continue
        if vertical and (midpoint[0] < width * 0.015 or midpoint[0] > width * 0.985):
            continue
        line_strength = float(np.percentile(magnitude[ys, xs], 60))
        altitude_weight = 1.0 - 0.35 * (midpoint[1] / max(1, height)) ** 2
        score = line_strength * (length ** 2.20) * altitude_weight
        raw_candidates.append((score, length, angle, midpoint, start, end))

    if not raw_candidates:
        return [], 0
    raw_candidates.sort(reverse=True, key=lambda item: item[0])

    # Merge duplicate LSD edges belonging to the same physical streak.
    candidates = []
    for item in raw_candidates:
        _score, _length, angle, midpoint, start, end = item
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        duplicate = False
        for kept in candidates:
            delta = angle - kept[2]
            angle_delta = abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta))) / 2
            perpendicular = abs(float(np.dot(np.array(midpoint) - np.array(kept[3]), normal)))
            center_distance = float(np.hypot(midpoint[0] - kept[3][0], midpoint[1] - kept[3][1]))
            if angle_delta < np.deg2rad(8) and perpendicular < 10 and center_distance < max(30, _length):
                duplicate = True
                break
        if not duplicate:
            candidates.append(item)

    meteors = []
    ranked_meteors = []
    planes = 0
    best_score = candidates[0][0]
    if os.environ.get("METEOR_DEBUG"):
        print("candidates", [(round(c[0], 1), round(c[1], 1), c[4], c[5]) for c in candidates[:10]])
    for score, length, angle, midpoint, start, end in candidates[:60 if ranked else 20]:
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        center = np.array(midpoint)
        span = int(np.hypot(width, height))
        ts = np.arange(-span, span + 1, dtype=np.float32)
        points = center[None, :] + ts[:, None] * direction[None, :]
        inside = ((points[:, 0] >= 0) & (points[:, 0] < width) &
                  (points[:, 1] >= 0) & (points[:, 1] < height))
        points = points[inside]
        profile_parts = []
        for offset in (-2, -1, 0, 1, 2):
            shifted = points + normal[None, :] * offset
            px = shifted[:, 0].clip(0, width - 1).astype(int)
            py = shifted[:, 1].clip(0, height - 1).astype(int)
            profile_parts.append(magnitude[py, px])
        profile = np.max(np.stack(profile_parts), axis=0)
        profile_threshold = max(float(np.percentile(magnitude, 99.65)), float(np.median(profile) + 6 * np.median(np.abs(profile - np.median(profile)))))
        active = profile > profile_threshold
        active = cv2.morphologyEx(active.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))[0] > 0
        padded = np.pad(active.astype(np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        runs = [(a, b) for a, b in zip(starts, ends) if b - a >= max(9, int(min_length * 0.40))]
        if len(runs) >= 3:
            gaps = np.diff([(a + b) * 0.5 for a, b in runs])
            regular = len(gaps) >= 2 and float(np.std(gaps) / max(1.0, np.mean(gaps))) < 0.55
            coverage = sum(b - a for a, b in runs) / max(1, runs[-1][1] - runs[0][0])
            if regular or coverage < 0.58:
                if os.environ.get("METEOR_DEBUG"):
                    print("plane", start, end, "runs", runs, "coverage", round(coverage, 3), "regular", regular)
                planes += 1
                continue
        if ranked:
            ranked_meteors.append((start, end, score))
            continue
        # One strong trail is guaranteed by the selected TIFF folder. A second is
        # accepted only when it is independently strong, preventing mass false positives.
        if meteors and score < best_score * 0.58:
            continue
        if len(meteors) >= 2:
            break
        meteors.append((start, end))
    if ranked:
        if not ranked_meteors:
            return [], planes
        strongest = max(item[2] for item in ranked_meteors)
        scored = [
            (start, end, int(np.clip(round(100.0 * (score / strongest) ** 0.35), 1, 100)))
            for start, end, score in ranked_meteors
        ]
        return scored, planes
    return meteors, planes


def locally_match(source: np.ndarray, base: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    hard = (alpha > 0.04).astype(np.uint8)
    radius = max(9, int(max(source.shape[:2]) * 0.025))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius | 1, radius | 1))
    ring = (cv2.dilate(hard, kernel) > 0) & (alpha < 0.01)
    if int(ring.sum()) < 300:
        ring = alpha < 0.01
    corrected = source.astype(np.float32)
    src = source.astype(np.float32)
    dst = base.astype(np.float32)
    for channel in range(3):
        s = src[..., channel][ring]
        b = dst[..., channel][ring]
        if s.size < 50:
            continue
        sm, bm = float(np.median(s)), float(np.median(b))
        s10, s90 = np.percentile(s, (10, 90))
        b10, b90 = np.percentile(b, (10, 90))
        gain = float((b90 - b10) / max(128.0, s90 - s10))
        gain = float(np.clip(gain, 0.65, 1.55))
        corrected[..., channel] = (src[..., channel] - sm) * gain + bm
    # Exposure matching is estimated from the surrounding sky. If it darkens the
    # patch, restore that lost brightness only where a positive residual against
    # the clean base identifies the meteor core or tail. The matched sky remains
    # unchanged while the meteor keeps most of its original intensity and color.
    src_luminance = src[..., 0] * 0.2126 + src[..., 1] * 0.7152 + src[..., 2] * 0.0722
    corrected_luminance = (
        corrected[..., 0] * 0.2126 + corrected[..., 1] * 0.7152 + corrected[..., 2] * 0.0722
    )
    base_luminance = dst[..., 0] * 0.2126 + dst[..., 1] * 0.7152 + dst[..., 2] * 0.0722
    loss = src_luminance - corrected_luminance
    active = alpha > 0.025
    signal = corrected_luminance - base_luminance
    positive = signal[active & (signal > 0)]
    if positive.size >= 5 and np.any(loss[active] > 0):
        low, high = (float(value) for value in np.percentile(positive, (12.0, 97.0)))
        if high - low < max(1e-3, high * 0.015):
            highlight_weight = (signal > 0).astype(np.float32)
        else:
            x = np.clip((signal - low) / (high - low), 0.0, 1.0)
            highlight_weight = x * x * (3.0 - 2.0 * x)
            # Square root retains more of a faint, gradually fading meteor tail.
            highlight_weight = np.sqrt(highlight_weight)
        darkened = (loss > 0).astype(np.float32)
        blend = np.clip(highlight_weight * np.sqrt(np.clip(alpha, 0.0, 1.0)) * darkened * 0.90, 0.0, 0.90)
        corrected = corrected * (1.0 - blend[..., None]) + src * blend[..., None]
    return np.clip(corrected, 0, 65535)


def apply_meteor_curve(
    source: np.ndarray, alpha: np.ndarray, shadows: float, highlights: float, clip_max: float
) -> np.ndarray:
    """Apply a smooth, endpoint-preserving S-curve to the pasted meteor patch."""
    image = source.astype(np.float32)
    active = alpha > 0.04
    if int(active.sum()) < 20:
        return image
    luminance = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    values = luminance[active]
    low, high = (float(v) for v in np.percentile(values, (2.0, 99.8)))
    if high - low < max(2.0, clip_max / 4096.0):
        return image
    x = np.clip((luminance - low) / (high - low), 0.0, 1.0)
    shadow_strength = np.clip(float(shadows) / 100.0, 0.0, 1.0)
    highlight_strength = np.clip(float(highlights) / 100.0, 0.0, 1.0)
    curved = x.copy()
    curved -= shadow_strength * 0.8 * x * (1.0 - x) ** 2
    curved += highlight_strength * 0.8 * x ** 2 * (1.0 - x)
    curved = np.clip(curved, 0.0, 1.0)
    target = low + curved * (high - low)
    within = (luminance >= low) & (luminance <= high)
    ratio = np.ones_like(luminance, dtype=np.float32)
    ratio[within] = target[within] / np.maximum(1.0, luminance[within])
    ratio = np.clip(ratio, 0.35, 2.5)
    return np.clip(image * ratio[..., None], 0, clip_max)


def transformed_stroke_points(stroke: Stroke, width: int, height: int) -> list[tuple[float, float]]:
    points = np.asarray([(x * (width - 1), y * (height - 1)) for x, y in stroke.points], dtype=np.float32)
    if not len(points):
        return []
    center = points.mean(axis=0)
    if len(points) > 1:
        direction = points[-1] - points[0]
        theta = float(np.arctan2(direction[1], direction[0]))
    else:
        theta = 0.0
    def rotation(angle: float) -> np.ndarray:
        return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    axis = rotation(theta)
    user = rotation(np.deg2rad(stroke.rotation))
    matrix = user @ axis @ np.diag([max(0.05, stroke.length_scale), max(0.05, stroke.width_scale)]) @ axis.T
    moved = (points - center) @ matrix.T + center + np.asarray([stroke.offset_x, stroke.offset_y], np.float32)
    return [(float(x / max(1, width - 1)), float(y / max(1, height - 1))) for x, y in moved]


def transformed_object_crop(
    source: np.ndarray, stroke: Stroke
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    height, width = source.shape[:2]
    original = Stroke(stroke.points, stroke.width, stroke.feather)
    built = build_mask_crop([original], width, height)
    if built is None:
        return None
    alpha, (x0, y0, x1, y1) = built
    patch = source[y0:y1, x0:x1]
    points = np.asarray([(x * (width - 1), y * (height - 1)) for x, y in stroke.points], dtype=np.float32)
    center = points.mean(axis=0)
    direction = points[-1] - points[0] if len(points) > 1 else np.asarray([1.0, 0.0], np.float32)
    theta = float(np.arctan2(direction[1], direction[0]))
    def rot(angle: float) -> np.ndarray:
        return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    axis = rot(theta)
    linear = rot(np.deg2rad(stroke.rotation)) @ axis @ np.diag(
        [max(0.05, stroke.length_scale), max(0.05, stroke.width_scale)]
    ) @ axis.T
    destination_center = center + np.asarray([stroke.offset_x, stroke.offset_y], np.float32)
    translation = destination_center - linear @ center
    full_matrix = np.column_stack((linear, translation)).astype(np.float32)
    corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    warped_corners = corners @ linear.T + translation
    dx0 = max(0, int(np.floor(warped_corners[:, 0].min())) - 2)
    dy0 = max(0, int(np.floor(warped_corners[:, 1].min())) - 2)
    dx1 = min(width, int(np.ceil(warped_corners[:, 0].max())) + 2)
    dy1 = min(height, int(np.ceil(warped_corners[:, 1].max())) + 2)
    if dx1 <= dx0 or dy1 <= dy0:
        return None
    local = full_matrix.copy()
    local[:, 2] += linear @ np.asarray([x0, y0], np.float32)
    local[:, 2] -= np.asarray([dx0, dy0], np.float32)
    warped_source = cv2.warpAffine(
        patch, local, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_alpha = cv2.warpAffine(
        alpha, local, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha * float(np.clip(stroke.opacity, 0.0, 1.0)), 0.0, 1.0)
    return warped_source, warped_alpha, (dx0, dy0, dx1, dy1)


def compose_meteor_objects(
    source: np.ndarray,
    base: np.ndarray,
    strokes: Iterable[Stroke],
    match_exposure: bool,
    curve_enabled: bool,
    curve_shadows: float,
    curve_highlights: float,
    blend_mode: str = "natural",
) -> tuple[np.ndarray, np.ndarray]:
    clip_max = 65535.0 if base.dtype == np.uint16 or source.dtype == np.uint16 else 255.0
    result = base.astype(np.float32).copy()
    height, width = base.shape[:2]
    union = np.zeros((height, width), np.float32)
    erasers = [s for s in strokes if s.erase]
    erase_mask = np.zeros((height, width), np.float32)
    if erasers:
        erased = build_mask_crop(erasers, width, height)
        if erased is not None:
            crop, (x0, y0, x1, y1) = erased
            erase_mask[y0:y1, x0:x1] = crop
    for stroke in (s for s in strokes if not s.erase):
        transformed = transformed_object_crop(source, stroke)
        if transformed is None:
            continue
        source_patch, alpha, (x0, y0, x1, y1) = transformed
        alpha *= 1.0 - erase_mask[y0:y1, x0:x1]
        if not np.any(alpha > 0.001):
            continue
        base_patch = result[y0:y1, x0:x1]
        source_float = source_patch.astype(np.float32)
        if match_exposure:
            source_float = locally_match(source_float, base_patch, alpha)
        if curve_enabled:
            source_float = apply_meteor_curve(source_float, alpha, curve_shadows, curve_highlights, clip_max)
        if blend_mode in {"normal", "普通粘贴"}:
            candidate = source_float
        elif blend_mode in {"residual", "亮度残差"}:
            positive = np.maximum(source_float - base_patch, 0.0)
            candidate = np.clip(base_patch + positive, 0.0, clip_max)
        else:
            candidate = np.maximum(base_patch, source_float)
        a = alpha[..., None]
        result[y0:y1, x0:x1] = base_patch * (1.0 - a) + candidate * a
        union[y0:y1, x0:x1] = np.maximum(union[y0:y1, x0:x1], alpha)
    return np.clip(result, 0, clip_max).astype(base.dtype), union


class MeteorComposer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1280x820")
        self.minsize(1000, 680)

        self.source_dir = tk.StringVar()
        self.base_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.brush_width = tk.IntVar(value=18)
        self.eraser_width = tk.IntVar(value=40)
        self.feather = tk.IntVar(value=10)
        self.match_exposure = tk.BooleanVar(value=False)
        self.curve_enabled = tk.BooleanVar(value=False)
        self.curve_shadows = tk.IntVar(value=15)
        self.curve_highlights = tk.IntVar(value=25)
        self.candidate_threshold = tk.IntVar(value=55)
        self.candidate_summary = tk.StringVar(value="当前图尚未分析候选")
        self.ai_model_status = tk.StringVar()
        self.autosave_status = tk.StringVar(value="自动保存：等待更改")
        self.export_tiff = tk.BooleanVar(value=False)
        self.blend_mode = tk.StringVar(value="自然融合")
        self.edit_mode = tk.StringVar(value="paint")
        self.view_mode = tk.StringVar(value="source")
        self.show_mask = tk.BooleanVar(value=True)
        self.h_mask_held = False
        self.status = tk.StringVar(value="请选择输入和输出位置。")

        self.files: list[Path] = []
        self.pairs: dict[str, Path] = {}
        self.original_sources: dict[str, Path] = {}
        self.use_original_sources: set[str] = set()
        self.alignment_statuses: dict[str, str] = {}
        self.strokes: dict[str, list[Stroke]] = {}
        self.candidates: dict[str, list[Stroke]] = {}
        self.candidate_thresholds: dict[str, int] = {}
        self.adjustment_defaults = {
            "match_exposure": False, "curve_enabled": False,
            "curve_shadows": 15, "curve_highlights": 25,
        }
        self.image_adjustments: dict[str, dict] = {}
        self.loading_adjustments = False
        self.setting_candidate_threshold = False
        self.edit_history: dict[str, list[tuple[str, int, object]]] = {}
        self.edit_redo: dict[str, list[tuple[str, int, object]]] = {}
        self.current_path: Path | None = None
        self.preview_rgb: np.ndarray | None = None
        self.preview_source: np.ndarray | None = None
        self.preview_base: np.ndarray | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.current_dims = (1, 1)
        self.display_box = (0, 0, 1, 1)
        self.active_points: list[tuple[float, float]] = []
        self.active_canvas_line: int | None = None
        self.active_shift_line = False
        self.active_tool_mode: str | None = None
        self.active_tool_width = 0
        self.active_tool_feather = 0
        self.live_erase_stroke: Stroke | None = None
        self.last_live_render = 0.0
        self.shift_anchors: dict[str, tuple[float, float]] = {}
        self.cursor_position: tuple[float, float] | None = None
        self.cursor_items: list[int] = []
        self.alt_previous_mode: str | None = None
        self.active_action_index = -1
        self.context_stroke_index: int | None = None
        self.context_highlight: int | None = None
        self.hover_candidate_index: int | None = None
        self.hover_candidate_items: list[int] = []
        self.work_queue: queue.Queue = queue.Queue()
        self.ranker_model = load_meteor_ranker()
        if self.ranker_model:
            self.ai_model_status.set(
                "AI模型：个性化" if self.ranker_model.get("_personalized") else "AI模型：内置基础版"
            )
        else:
            self.ai_model_status.set("AI模型：未找到，使用旧评分")
        self.autosave_path = autosave_file_path()
        self.autosave_after_id: str | None = None
        self.autosave_suspended = True
        self.video_window = None
        self.alignment_window = None

        self._build_ui()
        self._bind_shortcuts()
        self._setup_autosave()
        self.after(150, self._poll_queue)
        self.after(350, self._restore_autosave)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text=APP_NAME, font=("TkDefaultFont", 15, "bold")).pack(side="left")
        ttk.Label(header, text="图片合成工作区").pack(side="left", padx=12)
        ttk.Button(header, text="打开视频动态工作区…", command=self.open_video_workspace).pack(side="right")
        ttk.Button(header, text="Siril＋PTGui星空对齐…", command=self.open_alignment_workspace).pack(side="right", padx=(0, 6))

        paths = ttk.LabelFrame(root, text="输入与输出（源素材只读）", padding=8)
        paths.pack(fill="x")
        self._path_row(paths, 0, "原图 TIFF 文件夹", self.source_dir, self._browse_source)
        self._path_row(paths, 1, "干净 JPG 文件夹", self.base_dir, self._browse_base)
        self._path_row(paths, 2, "独立输出文件夹", self.output_dir, self._browse_output)
        ttk.Button(paths, text="只读扫描", command=self.scan_inputs).grid(row=0, column=3, rowspan=3, padx=8, sticky="ns")
        paths.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 6))

        left = ttk.Frame(body, width=310)
        body.add(left, weight=0)
        ttk.Label(left, text="TIFF 素材（双击加载）").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("status",), show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="文件")
        self.tree.heading("status", text="蒙版")
        self.tree.column("#0", width=210)
        self.tree.column("status", width=70, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=4)
        self.tree.bind("<Double-1>", lambda _e: self.load_selected())
        ttk.Button(left, text="加载所选", command=self.load_selected).pack(fill="x")
        ttk.Button(left, text="切换：自动对齐 / 原始状态", command=self.toggle_original_state).pack(fill="x", pady=(4, 0))

        center = ttk.Frame(body)
        body.add(center, weight=1)
        view_bar = ttk.Frame(center)
        view_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(view_bar, text="查看：").pack(side="left")
        ttk.Radiobutton(view_bar, text="1 原图 TIFF", variable=self.view_mode, value="source", command=self._render_preview).pack(side="left")
        ttk.Radiobutton(view_bar, text="2 干净 JPG", variable=self.view_mode, value="base", command=self._render_preview).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(view_bar, text="3 融合后预览", variable=self.view_mode, value="blend", command=self._render_preview).pack(side="left", padx=(8, 0))
        ttk.Label(view_bar, text="红色蒙版默认显示；按住 H 临时隐藏").pack(side="left", padx=(16, 0))
        ttk.Label(view_bar, text="普通拖动绘制；单击后 Shift+单击可画直线").pack(side="right")
        self.canvas = tk.Canvas(center, background="#181818", cursor="none", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render_preview())
        self.canvas.bind("<Motion>", self._cursor_motion)
        self.canvas.bind("<Leave>", self._cursor_leave)
        self.canvas.bind("<ButtonPress-1>", self._stroke_start)
        self.canvas.bind("<B1-Motion>", self._stroke_move)
        self.canvas.bind("<ButtonRelease-1>", self._stroke_end)
        self.canvas.bind("<Button-3>", self._show_mask_menu)
        self.canvas.bind("<Shift-Button-3>", self._delete_mask_at_event)
        self.canvas.bind("<Control-Button-1>", self._delete_mask_at_event)
        try:
            self.canvas.bind("<Command-Button-1>", self._delete_mask_at_event)
        except tk.TclError:
            pass
        self.mask_menu = tk.Menu(self, tearoff=0)
        self.mask_menu.add_command(label="删除整条蒙版", command=self._delete_context_stroke)
        self.mask_menu.add_command(label="锁定这条蒙版", command=self._toggle_context_lock)
        self.mask_menu.add_command(label="移动／旋转／拉伸这颗流星…", command=self._transform_context_stroke)

        tools = ttk.LabelFrame(root, text="流星蒙版", padding=8)
        tools.pack(fill="x")
        ttk.Radiobutton(tools, text="B ✎ 画笔", variable=self.edit_mode, value="paint", command=self._tool_settings_changed).grid(row=0, column=0, padx=(0, 4))
        ttk.Radiobutton(tools, text="E ▱ 橡皮擦", variable=self.edit_mode, value="erase", command=self._tool_settings_changed).grid(row=0, column=1, padx=(0, 12))
        ttk.Label(tools, text="画笔宽度(px)").grid(row=0, column=2)
        ttk.Scale(tools, from_=2, to=100, variable=self.brush_width, orient="horizontal", command=lambda _v: self._tool_settings_changed()).grid(row=0, column=3, sticky="ew", padx=5)
        ttk.Label(tools, textvariable=self.brush_width, width=4).grid(row=0, column=4)
        ttk.Label(tools, text="羽化(px)").grid(row=0, column=5, padx=(15, 0))
        ttk.Scale(tools, from_=0, to=80, variable=self.feather, orient="horizontal", command=lambda _v: self._tool_settings_changed()).grid(row=0, column=6, sticky="ew", padx=5)
        ttk.Label(tools, textvariable=self.feather, width=4).grid(row=0, column=7)
        ttk.Button(tools, text="撤销 Ctrl+Z", command=self.undo_stroke).grid(row=0, column=8, padx=3)
        ttk.Button(tools, text="清除此图蒙版", command=self.clear_strokes).grid(row=0, column=9, padx=3)
        ttk.Button(tools, text="自动检测全部", command=self.auto_detect_all).grid(row=0, column=10, padx=3)
        ttk.Button(tools, text="保存项目", command=self.save_project).grid(row=0, column=11, padx=3)
        ttk.Button(tools, text="载入项目", command=self.load_project).grid(row=0, column=12, padx=3)
        ttk.Label(tools, text="橡皮擦宽度(px)").grid(row=1, column=2, pady=(6, 0))
        ttk.Scale(tools, from_=2, to=200, variable=self.eraser_width, orient="horizontal", command=lambda _v: self._tool_settings_changed()).grid(row=1, column=3, sticky="ew", padx=5, pady=(6, 0))
        ttk.Label(tools, textvariable=self.eraser_width, width=4).grid(row=1, column=4, pady=(6, 0))
        ttk.Checkbutton(tools, text="当前图：局部匹配曝光/颜色＋保护流星亮部", variable=self.match_exposure, command=self._render_preview).grid(row=1, column=5, columnspan=3, sticky="w", pady=(6, 0), padx=(15, 0))
        ttk.Checkbutton(tools, text="同时导出16位TIFF（占用空间较大）", variable=self.export_tiff).grid(row=1, column=8, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(tools, text="合成方式").grid(row=2, column=8, sticky="e", pady=(6, 0))
        blend_combo = ttk.Combobox(
            tools, textvariable=self.blend_mode, state="readonly", width=12,
            values=("自然融合", "亮度残差", "普通粘贴"),
        )
        blend_combo.grid(row=2, column=9, columnspan=2, sticky="w", pady=(6, 0), padx=5)
        blend_combo.bind("<<ComboboxSelected>>", lambda _event: self._render_preview())
        ttk.Checkbutton(tools, text="当前图：启用流星曲线", variable=self.curve_enabled, command=self._render_preview).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Label(tools, text="暗部压低").grid(row=2, column=2, pady=(6, 0))
        ttk.Scale(tools, from_=0, to=100, variable=self.curve_shadows, orient="horizontal", command=lambda _v: self._render_preview()).grid(row=2, column=3, sticky="ew", padx=5, pady=(6, 0))
        ttk.Label(tools, textvariable=self.curve_shadows, width=4).grid(row=2, column=4, pady=(6, 0))
        ttk.Label(tools, text="亮部提升").grid(row=2, column=5, padx=(15, 0), pady=(6, 0))
        ttk.Scale(tools, from_=0, to=100, variable=self.curve_highlights, orient="horizontal", command=lambda _v: self._render_preview()).grid(row=2, column=6, sticky="ew", padx=5, pady=(6, 0))
        ttk.Label(tools, textvariable=self.curve_highlights, width=4).grid(row=2, column=7, pady=(6, 0))
        ttk.Button(tools, text="AI分析当前单张候选", command=self.detect_current_candidates).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Label(tools, text="AI分数阈值").grid(row=3, column=2, pady=(7, 0))
        ttk.Scale(tools, from_=1, to=100, variable=self.candidate_threshold, orient="horizontal",
                  command=self._candidate_threshold_changed).grid(row=3, column=3, sticky="ew", padx=5, pady=(7, 0))
        ttk.Label(tools, textvariable=self.candidate_threshold, width=4).grid(row=3, column=4, pady=(7, 0))
        ttk.Label(tools, textvariable=self.candidate_summary).grid(row=3, column=5, columnspan=4, sticky="w", padx=(15, 0), pady=(7, 0))
        ttk.Label(tools, textvariable=self.ai_model_status).grid(row=3, column=9, sticky="e", pady=(7, 0))
        ttk.Label(tools, text="悬停候选点＋可选中并锁定").grid(row=3, column=10, columnspan=3, sticky="e", pady=(7, 0))
        tools.columnconfigure(3, weight=1)
        tools.columnconfigure(6, weight=1)

        bottom = ttk.Frame(root)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
        ttk.Label(bottom, textvariable=self.autosave_status).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=220)
        self.progress.pack(side="left", padx=8)
        ttk.Button(bottom, text="导出合成结果", command=self.export).pack(side="right")
        ttk.Button(bottom, text="快捷键 F1", command=self.show_shortcuts).pack(side="right", padx=(0, 6))

    def open_video_workspace(self) -> None:
        if self.video_window is not None:
            try:
                if self.video_window.winfo_exists():
                    self.video_window.deiconify()
                    self.video_window.lift()
                    self.video_window.focus_force()
                    return
            except tk.TclError:
                pass
        self.video_window = open_video_workspace(self)

    def open_alignment_workspace(self) -> None:
        if self.alignment_window is not None:
            try:
                if self.alignment_window.winfo_exists():
                    self.alignment_window.deiconify()
                    self.alignment_window.lift()
                    return
            except tk.TclError:
                pass
        self.alignment_window = open_alignment_workspace(self, self._load_alignment_result)

    def _load_alignment_result(self, result) -> None:
        loaded_items = [(Path(item.output_layer), item.status) for item in result.items if item.output_layer]
        exported = [path for path, _status in loaded_items]
        if not exported or not result.base_layer:
            messagebox.showwarning(APP_NAME, "对齐任务没有可回载的图层")
            return
        base = Path(result.base_layer)
        self.source_dir.set(str(exported[0].parent))
        self.base_dir.set(str(base))
        final_output = Path(result.project_dir) / "final_composite"
        final_output.mkdir(parents=True, exist_ok=True)
        self.output_dir.set(str(final_output))
        self.files = exported
        self.pairs = {str(path): base for path in exported}
        self.original_sources = {
            str(Path(item.output_layer)): Path(item.source)
            for item in result.items
            if item.output_layer and Path(item.output_layer) != Path(item.source)
        }
        self.use_original_sources.clear()
        self.alignment_statuses = {
            str(Path(item.output_layer)): item.status for item in result.items if item.output_layer
        }
        self.tree.delete(*self.tree.get_children())
        status_by_path = {str(path): status for path, status in loaded_items}
        for index, path in enumerate(exported):
            count = len(self.strokes.get(str(path), []))
            status = status_by_path.get(str(path), "")
            prefix = "[原始状态] " if "原始状态" in status else ("[创意] " if status == "创意放置" else "")
            self.tree.insert("", "end", iid=str(index), text=prefix + path.name, values=(count or "—",))
        unresolved = sum(item.status == "需处理" for item in result.items)
        review = sum("需复查" in item.status for item in result.items)
        creative = sum(item.status == "创意放置" for item in result.items)
        self.status.set(
            f"已载入 {len(exported)} 张（需复查 {review} 张，创意放置 {creative} 张）；"
            f"仍未处理 {unresolved} 张。现在可检测并抠流星。"
        )
        if exported:
            self.tree.selection_set("0")
            self.load_selected()
        self._schedule_autosave()

    def _bind_shortcuts(self) -> None:
        plain = {
            "<KeyPress-b>": lambda: self._set_edit_mode("paint"),
            "<KeyPress-e>": lambda: self._set_edit_mode("erase"),
            "<KeyPress-bracketleft>": lambda: self._change_tool_width(-2),
            "<KeyPress-bracketright>": lambda: self._change_tool_width(2),
            "<Shift-KeyPress-bracketleft>": lambda: self._change_feather(-2),
            "<Shift-KeyPress-bracketright>": lambda: self._change_feather(2),
            "<KeyPress-1>": lambda: self._set_view_mode("source"),
            "<KeyPress-2>": lambda: self._set_view_mode("base"),
            "<KeyPress-3>": lambda: self._set_view_mode("blend"),
            "<KeyPress-Left>": lambda: self._select_relative(-1),
            "<KeyPress-Right>": lambda: self._select_relative(1),
        }
        for sequence, callback in plain.items():
            self.bind_all(sequence, lambda event, fn=callback: self._run_plain_shortcut(event, fn))
        self.bind_all("<Escape>", lambda _e: self._cancel_active_stroke())
        self.bind_all("<F1>", lambda _e: self.show_shortcuts())
        self.bind_all("<KeyPress-h>", self._mask_hold_press)
        self.bind_all("<KeyRelease-h>", self._mask_hold_release)
        for key in ("Alt_L", "Alt_R", "Option_L", "Option_R"):
            try:
                self.bind_all(f"<KeyPress-{key}>", self._temporary_alt_press)
                self.bind_all(f"<KeyRelease-{key}>", self._temporary_alt_release)
            except tk.TclError:
                pass
        for modifier in ("Control", "Command"):
            bindings = {
                f"<{modifier}-z>": self.undo_stroke,
                f"<{modifier}-Shift-z>": self.redo_stroke,
                f"<{modifier}-y>": self.redo_stroke,
                f"<{modifier}-s>": self.save_project,
                f"<{modifier}-o>": self.load_project,
                f"<{modifier}-Return>": self.export,
            }
            for sequence, callback in bindings.items():
                try:
                    self.bind_all(sequence, lambda _event, fn=callback: (fn(), "break")[1])
                except tk.TclError:
                    pass

    def _run_plain_shortcut(self, event, callback):
        widget_class = event.widget.winfo_class() if event.widget else ""
        if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox", "Scale", "TScale"}:
            return None
        callback()
        return "break"

    def _set_edit_mode(self, mode: str) -> None:
        self.edit_mode.set(mode)
        self._tool_settings_changed()
        self.status.set("已切换到画笔（B）" if mode == "paint" else "已切换到橡皮擦（E）")

    def _temporary_alt_press(self, event):
        widget_class = event.widget.winfo_class() if event.widget else ""
        if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox", "Scale", "TScale"}:
            return None
        if self.alt_previous_mode is None and not self.active_points:
            self.alt_previous_mode = self.edit_mode.get()
            self.edit_mode.set("erase" if self.alt_previous_mode == "paint" else "paint")
            self._tool_settings_changed()
        return "break"

    def _temporary_alt_release(self, _event=None):
        self._restore_alt_tool()
        return "break"

    def _restore_alt_tool(self) -> None:
        if self.alt_previous_mode is not None:
            self.edit_mode.set(self.alt_previous_mode)
            self.alt_previous_mode = None
            self._tool_settings_changed()

    def _change_tool_width(self, delta: int) -> None:
        variable = self.eraser_width if self.edit_mode.get() == "erase" else self.brush_width
        maximum = 200 if self.edit_mode.get() == "erase" else 100
        variable.set(int(np.clip(variable.get() + delta, 2, maximum)))
        self._tool_settings_changed()

    def _change_feather(self, delta: int) -> None:
        self.feather.set(int(np.clip(self.feather.get() + delta, 0, 80)))
        self._tool_settings_changed()

    def _set_view_mode(self, mode: str) -> None:
        self.view_mode.set(mode)
        self._render_preview()

    def _mask_hold_press(self, event):
        widget_class = event.widget.winfo_class() if event.widget else ""
        if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"}:
            return None
        if not self.h_mask_held:
            self.h_mask_held = True
            self.show_mask.set(False)
            self._render_preview()
        return "break"

    def _mask_hold_release(self, _event=None):
        if self.h_mask_held:
            self.h_mask_held = False
            self.show_mask.set(True)
            self._render_preview()
        return "break"

    def _select_relative(self, delta: int) -> None:
        if not self.files:
            return
        if self.current_path in self.files:
            index = self.files.index(self.current_path)
        else:
            selection = self.tree.selection()
            index = int(selection[0]) if selection else 0
        index = int(np.clip(index + delta, 0, len(self.files) - 1))
        self.tree.selection_set(str(index))
        self.tree.focus(str(index))
        self.tree.see(str(index))
        self.load_selected()

    def show_shortcuts(self) -> None:
        messagebox.showinfo(APP_NAME + " — 快捷键", """工具
B：画笔    E：橡皮擦
按住 Alt/Option：临时在画笔与橡皮擦之间切换，松开恢复
[ / ]：减小 / 增大当前工具宽度
Shift+[ / Shift+]：减小 / 增大羽化
单击后 Shift+单击：画直线或直线擦除

编辑
Ctrl/Command+Z：撤销
Ctrl/Command+Shift+Z 或 Ctrl/Command+Y：重做
右键单击蒙版：打开“删除整条蒙版”菜单
右键菜单“锁定这条蒙版”：锁定项不受候选阈值、重新检测或清除影响
Ctrl/Command+单击蒙版：直接删除整条蒙版
Shift+右键单击蒙版：直接删除整条蒙版
Esc：取消当前尚未完成的一笔

单张候选
点击“AI分析当前单张候选”，再拖动 AI 分数阈值；阈值越低，加入的候选越多
鼠标靠近候选轨迹：弹出“＋选中”按钮，点击后直接加入并锁定
红色蒙版及候选分数默认显示；按住 H 可临时隐藏

查看与文件
1：原图 TIFF    2：干净 JPG    3：融合后预览
红色蒙版默认显示；按住 H：临时隐藏，松开恢复
← / →：上一张 / 下一张
Ctrl/Command+S：保存项目
Ctrl/Command+O：载入项目
Ctrl/Command+Enter：导出合成结果
项目更改约 0.8 秒后自动保存；下次启动自动恢复
F1：显示本快捷键表""")

    def _path_row(self, parent, row, label, variable, command) -> None:
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=5)
        ttk.Button(parent, text="选择…", command=command).grid(row=row, column=2)

    def _browse_source(self) -> None:
        value = filedialog.askdirectory(title="选择原图 TIFF 文件夹")
        if value:
            self.source_dir.set(value)

    def _browse_base(self) -> None:
        value = filedialog.askdirectory(title="选择干净底图 JPG 文件夹")
        if value:
            self.base_dir.set(value)

    def _browse_output(self) -> None:
        value = filedialog.askdirectory(title="选择独立输出文件夹")
        if value:
            self.output_dir.set(value)

    def scan_inputs(self) -> None:
        try:
            source = Path(self.source_dir.get()).expanduser()
            base_dir = Path(self.base_dir.get()).expanduser()
            output_text = self.output_dir.get().strip()
            if not output_text:
                raise ValueError("请选择独立输出文件夹")
            output = Path(output_text).expanduser()
            if not source.is_dir():
                raise ValueError("请选择有效的 TIFF 文件夹")
            if not (base_dir.is_dir() or base_dir.is_file()):
                raise ValueError("请选择有效的干净底图文件夹，或从PTGui工作区回载单张底图")
            if (path_is_within(output, source) or path_is_within(source, output)
                    or (base_dir.is_dir() and (path_is_within(output, base_dir) or path_is_within(base_dir, output)))):
                raise ValueError("输出目录必须与输入素材位置完全分开")
            source_files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)
            if not source_files:
                raise ValueError("原图文件夹中没有 TIFF 文件")
            if base_dir.is_file():
                base_files = [base_dir]
            else:
                base_files = sorted(
                    p for p in base_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
                )
            base_by_stem: dict[str, Path] = {}
            duplicates = set()
            for path in base_files:
                key = path.stem.casefold()
                if key in base_by_stem:
                    duplicates.add(path.stem)
                else:
                    base_by_stem[key] = path
            if duplicates:
                raise ValueError("干净底图文件夹存在同主文件名的重复 JPG/JPEG：" + "、".join(sorted(duplicates)[:8]))
            valid, mismatched, missing = [], [], []
            pairs: dict[str, Path] = {}
            for path in source_files:
                base = base_files[0] if base_dir.is_file() else base_by_stem.get(path.stem.casefold())
                if base is None:
                    missing.append(path.name)
                    continue
                sw, sh, _sdepth, _schannels = image_info(path)
                bw, bh, _bdepth, _bchannels = image_info(base)
                if (sw, sh) == (bw, bh):
                    valid.append(path)
                    pairs[str(path)] = base
                else:
                    mismatched.append(f"{path.name}: TIFF {sw}×{sh} / JPG {bw}×{bh}")
            self.files = valid
            self.pairs = pairs
            self.tree.delete(*self.tree.get_children())
            for index, path in enumerate(valid):
                count = len(self.strokes.get(str(path), []))
                prefix = "[原始状态] " if str(path) in self.use_original_sources else ""
                self.tree.insert("", "end", iid=str(index), text=prefix + path.name, values=(count or "—",))
            base_label = "单张PTGui底图" if base_dir.is_file() else f"底图 {len(base_files)} 张"
            message = f"TIFF {len(source_files)} 张，{base_label}；成功配对 {len(valid)} 对"
            if missing:
                message += f"；缺少同名 JPG {len(missing)} 张"
            if mismatched:
                message += f"；尺寸不符 {len(mismatched)} 张，已跳过"
            self.status.set(message)
        except Exception as exc:
            messagebox.showerror(APP_NAME, str(exc))

    def load_selected(self) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        path = self.files[int(selection[0])]
        image_path = self._effective_source_path(path)
        mode = "原始状态" if str(path) in self.use_original_sources else "自动对齐"
        self.status.set(f"正在加载 {path.name}（{mode}）…")
        self._run_worker(self._load_preview_worker, path, image_path, self.pairs[str(path)])

    def _effective_source_path(self, path: Path) -> Path:
        key = str(path)
        if key in self.use_original_sources:
            return self.original_sources.get(key, path)
        return path

    def toggle_original_state(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning(APP_NAME, "请先选择一张图片")
            return
        index = int(selection[0])
        path = self.files[index]
        key = str(path)
        if key not in self.original_sources:
            self.status.set("这张图目前就是原始状态，没有可切换的对齐版")
            return
        if key in self.use_original_sources:
            self.use_original_sources.remove(key)
            mode = "自动对齐"
        else:
            self.use_original_sources.add(key)
            mode = "原始状态"
        status = self.alignment_statuses.get(key, "")
        review = "[需复查] " if "需复查" in status else ""
        self.tree.item(str(index), text=f"[{mode}] {review}{path.name}")
        self._schedule_autosave()
        self.load_selected()

    def _current_adjustment_values(self) -> dict:
        return {
            "match_exposure": bool(self.match_exposure.get()),
            "curve_enabled": bool(self.curve_enabled.get()),
            "curve_shadows": int(self.curve_shadows.get()),
            "curve_highlights": int(self.curve_highlights.get()),
        }

    def _current_adjustment_changed(self, *_args) -> None:
        if self.loading_adjustments or not self.current_path:
            return
        self.image_adjustments[str(self.current_path)] = self._current_adjustment_values()

    def _load_image_adjustments(self, key: str) -> None:
        values = {**self.adjustment_defaults, **self.image_adjustments.get(key, {})}
        self.loading_adjustments = True
        try:
            self.match_exposure.set(bool(values["match_exposure"]))
            self.curve_enabled.set(bool(values["curve_enabled"]))
            self.curve_shadows.set(int(values["curve_shadows"]))
            self.curve_highlights.set(int(values["curve_highlights"]))
        finally:
            self.loading_adjustments = False

    def auto_detect_all(self) -> None:
        if not self.files:
            messagebox.showwarning(APP_NAME, "请先执行只读扫描")
            return
        if any(self.strokes.values()):
            if not messagebox.askyesno(APP_NAME, "AI 自动检测会替换未锁定蒙版；锁定蒙版继续保留。继续吗？"):
                return
        self.progress["value"] = 0
        self.status.set("正在用内置 AI 自动检测并排序流星候选…")
        read_paths = {str(path): self._effective_source_path(path) for path in self.files}
        self._run_worker(self._auto_detect_worker, self.files.copy(), self.pairs.copy(), read_paths)

    def detect_current_candidates(self) -> None:
        if not self.current_path or str(self.current_path) not in self.pairs:
            messagebox.showwarning(APP_NAME, "请先扫描并加载一张图片")
            return
        self.status.set(f"正在用内置 AI 分析当前单张候选：{self.current_path.name}…")
        self.progress["value"] = 10
        self._run_worker(
            self._candidate_worker, self.current_path, self._effective_source_path(self.current_path),
            self.pairs[str(self.current_path)],
        )

    def _candidate_worker(self, path: Path, image_path: Path, base_path: Path):
        base_preview, _ = detection_preview(read_image(base_path))
        source_preview, source_scale = detection_preview(read_image(image_path))
        if source_preview.shape != base_preview.shape:
            source_preview = cv2.resize(
                source_preview, (base_preview.shape[1], base_preview.shape[0]), interpolation=cv2.INTER_AREA
            )
        trails, planes = detect_trails(source_preview, base_preview, ranked=True)
        height, width = source_preview.shape[:2]
        candidates = []
        maps = prepare_ml_maps(source_preview, base_preview) if self.ranker_model else None
        for start, end, legacy_score in trails:
            score = int(round(100 * predict_gradient_boosting(
                candidate_feature_vector(maps, start, end, legacy_score), self.ranker_model
            ))) if self.ranker_model else legacy_score
            start, end, core_length = expand_trail_segment(start, end, width, height)
            points = [
                (start[0] / max(1, width - 1), start[1] / max(1, height - 1)),
                (end[0] / max(1, width - 1), end[1] / max(1, height - 1)),
            ]
            preview_width = float(np.clip(core_length * 0.12, 10.0, 28.0))
            full_width = int(np.clip(round(preview_width / max(0.001, source_scale)), 36, 140))
            full_feather = int(np.clip(round(preview_width * 0.70 / max(0.001, source_scale)), 20, 100))
            candidates.append(Stroke(points, full_width, full_feather, False, False, int(score)))
        candidates.sort(key=lambda item: item.auto_score or 0, reverse=True)
        return "candidates", path, candidates, planes

    def _candidate_threshold_changed(self, _value=None) -> None:
        if self.setting_candidate_threshold or not self.current_path:
            return
        key = str(self.current_path)
        self.candidate_thresholds[key] = int(round(self.candidate_threshold.get()))
        self.edit_history.pop(key, None)
        self.edit_redo.pop(key, None)
        self._apply_candidate_threshold(key)
        self._update_tree_status()
        self._render_preview()
        self._schedule_autosave()

    def _apply_candidate_threshold(self, key: str) -> None:
        threshold = self.candidate_thresholds.get(key, int(self.candidate_threshold.get()))
        existing = self.strokes.get(key, [])
        # Locked candidates remain even if a new analysis no longer finds them.
        retained = [stroke for stroke in existing if stroke.auto_score is None or stroke.locked]
        selected = [
            stroke for stroke in self.candidates.get(key, [])
            if (stroke.locked or (stroke.auto_score or 0) >= threshold)
            and not any(
                kept.auto_score == stroke.auto_score and kept.points == stroke.points for kept in retained
            )
        ]
        # Eraser operations stay last so they can correct both manual and automatic masks.
        positive = [stroke for stroke in retained if not stroke.erase]
        erasers = [stroke for stroke in retained if stroke.erase]
        self.strokes[key] = positive + selected + erasers
        self._update_candidate_summary(key)

    def _update_candidate_summary(self, key: str | None = None) -> None:
        if key is None:
            key = str(self.current_path) if self.current_path else ""
        if self.current_path and key != str(self.current_path):
            return
        pool = self.candidates.get(key, [])
        if not pool:
            self.candidate_summary.set("当前图尚未分析候选")
            return
        threshold = self.candidate_thresholds.get(key, int(self.candidate_threshold.get()))
        visible = sum((item.auto_score or 0) >= threshold or item.locked for item in pool)
        locked = sum(item.locked for item in self.strokes.get(key, []))
        self.candidate_summary.set(f"候选 {len(pool)} 条，当前加入 {visible} 条，锁定 {locked} 条")

    def _auto_detect_worker(self, files: list[Path], pairs: dict[str, Path], read_paths: dict[str, Path]):
        found: dict[str, list[Stroke]] = {}
        plane_count = 0
        for index, path in enumerate(files, start=1):
            base_path = pairs[str(path)]
            base_preview, _ = detection_preview(read_image(base_path))
            source_preview, source_scale = detection_preview(read_image(read_paths.get(str(path), path)))
            if source_preview.shape != base_preview.shape:
                source_preview = cv2.resize(source_preview, (base_preview.shape[1], base_preview.shape[0]), interpolation=cv2.INTER_AREA)
            trails, planes = detect_trails(source_preview, base_preview, ranked=True)
            plane_count += planes
            height, width = source_preview.shape[:2]
            maps = prepare_ml_maps(source_preview, base_preview) if self.ranker_model else None
            ranked_trails = []
            for start, end, legacy_score in trails:
                score = int(round(100 * predict_gradient_boosting(
                    candidate_feature_vector(maps, start, end, legacy_score), self.ranker_model
                ))) if self.ranker_model else legacy_score
                ranked_trails.append((score, start, end))
            ranked_trails.sort(reverse=True, key=lambda item: item[0])
            selected = [item for item in ranked_trails if item[0] >= 55][:4]
            strokes = []
            for score, start, end in selected:
                start, end, core_length = expand_trail_segment(start, end, width, height)
                points = [(start[0] / max(1, width - 1), start[1] / max(1, height - 1)),
                          (end[0] / max(1, width - 1), end[1] / max(1, height - 1))]
                preview_width = float(np.clip(core_length * 0.12, 10.0, 28.0))
                full_width = int(np.clip(round(preview_width / max(0.001, source_scale)), 36, 140))
                full_feather = int(np.clip(round(preview_width * 0.70 / max(0.001, source_scale)), 20, 100))
                strokes.append(Stroke(
                    points, width=full_width, feather=full_feather, auto_score=int(score)
                ))
            if strokes:
                found[str(path)] = strokes
            self.work_queue.put(("progress", index / len(files) * 100, f"自动检测 {index}/{len(files)}：{path.name}"))
        return "autodetected", found, plane_count

    def _load_preview_worker(self, path: Path, image_path: Path, base_path: Path):
        rgb16 = to_uint16(read_image(image_path))
        height, width = rgb16.shape[:2]
        scale = min(1.0, 1800 / max(width, height))
        preview = cv2.resize(rgb16, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        source_preview = np.right_shift(preview, 8).astype(np.uint8)
        base16 = to_uint16(read_image(base_path))
        if base16.shape[:2] != (height, width):
            raise ValueError(f"尺寸不一致：{path.name} / {base_path.name}")
        base_preview = cv2.resize(base16, (source_preview.shape[1], source_preview.shape[0]), interpolation=cv2.INTER_AREA)
        base_preview = np.right_shift(base_preview, 8).astype(np.uint8)
        return "preview", path, source_preview, base_preview, (width, height)

    def _preview_mask(self) -> np.ndarray:
        if self.preview_source is None or not self.current_path:
            return np.zeros((1, 1), dtype=np.float32)
        height, width = self.preview_source.shape[:2]
        scale = width / max(1, self.current_dims[0])
        scaled = []
        for item in self.strokes.get(str(self.current_path), []):
            scaled.append(Stroke(
                item.points, max(1, int(round(item.width * scale))),
                max(0, int(round(item.feather * scale))), item.erase, item.locked,
                item.auto_score, item.offset_x * scale, item.offset_y * scale,
                item.rotation, item.length_scale, item.width_scale, item.opacity,
            ))
        _composed, mask = compose_meteor_objects(
            self.preview_source, self.preview_base, scaled,
            bool(self.match_exposure.get()), bool(self.curve_enabled.get()),
            float(self.curve_shadows.get()), float(self.curve_highlights.get()),
            self.blend_mode.get(),
        )
        return mask

    def _render_preview(self) -> None:
        if self.preview_source is None or self.preview_base is None:
            return
        height, width = self.preview_source.shape[:2]
        scale_to_preview = width / max(1, self.current_dims[0])
        scaled_strokes = [Stroke(
            item.points, max(1, int(round(item.width * scale_to_preview))),
            max(0, int(round(item.feather * scale_to_preview))), item.erase, item.locked,
            item.auto_score, item.offset_x * scale_to_preview, item.offset_y * scale_to_preview,
            item.rotation, item.length_scale, item.width_scale, item.opacity,
        ) for item in self.strokes.get(str(self.current_path), [])]
        composed, mask = compose_meteor_objects(
            self.preview_source, self.preview_base, scaled_strokes,
            bool(self.match_exposure.get()), bool(self.curve_enabled.get()),
            float(self.curve_shadows.get()), float(self.curve_highlights.get()),
            self.blend_mode.get(),
        )
        mode = self.view_mode.get()
        if mode == "base":
            shown = self.preview_base.copy()
        elif mode == "blend":
            shown = composed
        else:
            shown = self.preview_source.copy()
        if self.show_mask.get() and mode != "blend" and np.any(mask > 0.001):
            opacity = (mask * 0.55)[..., None]
            red = np.empty_like(shown)
            red[:] = (255, 35, 25)
            shown = np.clip(shown.astype(np.float32) * (1.0 - opacity) + red.astype(np.float32) * opacity, 0, 255).astype(np.uint8)
        self.preview_rgb = shown
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())
        h, w = self.preview_rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
        image = Image.fromarray(self.preview_rgb).resize((dw, dh), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        x0, y0 = (canvas_w - dw) // 2, (canvas_h - dh) // 2
        self.display_box = (x0, y0, x0 + dw, y0 + dh)
        self.canvas.delete("all")
        self.context_highlight = None
        self.hover_candidate_items = []
        self.hover_candidate_index = None
        self.canvas.create_image(x0, y0, anchor="nw", image=self.preview_photo)
        if self.show_mask.get() and mode != "blend":
            self._draw_mask_annotations()
        self.cursor_items = []
        self._update_brush_cursor()
        if self.cursor_position is not None:
            self._update_candidate_hover(*self.cursor_position)

    def _draw_mask_annotations(self) -> None:
        if not self.current_path:
            return
        x0, y0, x1, y1 = self.display_box
        for stroke in self.strokes.get(str(self.current_path), []):
            if stroke.erase or not stroke.points or (stroke.auto_score is None and not stroke.locked):
                continue
            transformed = transformed_stroke_points(stroke, *self.current_dims)
            px, py = transformed[len(transformed) // 2]
            label_parts = []
            if stroke.locked:
                label_parts.append("[锁]")
            if stroke.auto_score is not None:
                label_parts.append(f"{stroke.auto_score}分")
            self.canvas.create_text(
                x0 + px * (x1 - x0), y0 + py * (y1 - y0) - 12,
                text=" ".join(label_parts), fill="#ffe66d", font=("TkDefaultFont", 10, "bold")
            )

    def _tool_settings_changed(self) -> None:
        self._update_brush_cursor()

    def _cursor_motion(self, event) -> None:
        self.cursor_position = (float(event.x), float(event.y))
        self._update_brush_cursor()
        self._update_candidate_hover(event.x, event.y)

    def _cursor_leave(self, _event=None) -> None:
        self.cursor_position = None
        for item in self.cursor_items:
            self.canvas.delete(item)
        self.cursor_items = []
        self._clear_candidate_hover()
        self.canvas.configure(cursor="arrow")

    def _clear_candidate_hover(self) -> None:
        for item in self.hover_candidate_items:
            self.canvas.delete(item)
        self.hover_candidate_items = []
        self.hover_candidate_index = None

    def _find_candidate_near(self, canvas_x: float, canvas_y: float) -> int | None:
        if not self.current_path or self.preview_source is None:
            return None
        x0, y0, x1, y1 = self.display_box
        if not (x0 <= canvas_x <= x1 and y0 <= canvas_y <= y1):
            return None
        best_index = None
        best_distance = float("inf")
        for index, stroke in enumerate(self.candidates.get(str(self.current_path), [])):
            if stroke.locked or not stroke.points:
                continue
            transformed = transformed_stroke_points(stroke, *self.current_dims)
            points = [(x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in transformed]
            if len(points) == 1:
                distance = float(np.hypot(canvas_x - points[0][0], canvas_y - points[0][1]))
            else:
                distance = min(
                    self._point_segment_distance(canvas_x, canvas_y, *start, *end)
                    for start, end in zip(points, points[1:])
                )
            if distance <= 14.0 and distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _update_candidate_hover(self, canvas_x: float, canvas_y: float) -> None:
        index = self._find_candidate_near(canvas_x, canvas_y)
        if index is None:
            self._clear_candidate_hover()
            return
        if index == self.hover_candidate_index and self.hover_candidate_items:
            return
        self._clear_candidate_hover()
        self.hover_candidate_index = index
        candidate = self.candidates[str(self.current_path)][index]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        label = f"＋ 选中 {candidate.auto_score or 0}分"
        icon_x = min(canvas_w - 56, max(56, canvas_x + 58))
        icon_y = min(canvas_h - 17, max(17, canvas_y - 20))
        tag = "candidate_pick"
        rectangle = self.canvas.create_rectangle(
            icon_x - 52, icon_y - 14, icon_x + 52, icon_y + 14,
            fill="#176b3a", outline="#8dffb8", width=2, tags=(tag,)
        )
        text_item = self.canvas.create_text(
            icon_x, icon_y, text=label, fill="#ffffff", font=("TkDefaultFont", 9, "bold"), tags=(tag,)
        )
        self.hover_candidate_items = [rectangle, text_item]
        self.canvas.tag_bind(tag, "<Button-1>", self._pick_hover_candidate)

    def _pick_hover_candidate(self, _event=None):
        if not self.current_path or self.hover_candidate_index is None:
            return "break"
        key = str(self.current_path)
        pool = self.candidates.get(key, [])
        index = self.hover_candidate_index
        if not (0 <= index < len(pool)):
            return "break"
        candidate = pool[index]
        candidate.locked = True
        matching = next(
            (stroke for stroke in self.strokes.get(key, [])
             if stroke.auto_score == candidate.auto_score and stroke.points == candidate.points),
            None,
        )
        if matching is not None:
            matching.locked = True
        else:
            values = self.strokes.setdefault(key, [])
            insert_at = next((i for i, stroke in enumerate(values) if stroke.erase), len(values))
            values.insert(insert_at, candidate)
        score = candidate.auto_score or 0
        self.edit_history.pop(key, None)
        self.edit_redo.pop(key, None)
        self._clear_candidate_hover()
        self._update_candidate_summary(key)
        self._update_tree_status()
        self._render_preview()
        self.status.set(f"已选中并锁定 {score} 分候选；调阈值或清除蒙版时都会保留")
        self._schedule_autosave()
        return "break"

    def _update_brush_cursor(self) -> None:
        for item in self.cursor_items:
            self.canvas.delete(item)
        self.cursor_items = []
        if self.cursor_position is None or self.preview_source is None:
            return
        x, y = self.cursor_position
        x0, y0, x1, y1 = self.display_box
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            self.canvas.configure(cursor="arrow")
            return
        self.canvas.configure(cursor="none")
        pixel_scale = (x1 - x0) / max(1, self.current_dims[0])
        hard_radius = max(2.0, self._tool_width() * pixel_scale * 0.5)
        feather_radius = hard_radius + max(0.0, float(self.feather.get()) * pixel_scale)
        color = "#64d2ff" if self.edit_mode.get() == "erase" else "#ffffff"
        # A dark under-ring keeps the cursor visible over bright stars and terrain.
        self.cursor_items.append(self.canvas.create_oval(
            x - hard_radius - 1, y - hard_radius - 1, x + hard_radius + 1, y + hard_radius + 1,
            outline="#000000", width=3
        ))
        self.cursor_items.append(self.canvas.create_oval(
            x - hard_radius, y - hard_radius, x + hard_radius, y + hard_radius,
            outline=color, width=1
        ))
        if feather_radius > hard_radius + 1.0:
            self.cursor_items.append(self.canvas.create_oval(
                x - feather_radius, y - feather_radius, x + feather_radius, y + feather_radius,
                outline=color, width=1, dash=(3, 3)
            ))
        if self.edit_mode.get() == "erase":
            # Small tilted eraser glyph inside the size ring.
            self.cursor_items.append(self.canvas.create_polygon(
                x - 7, y + 2, x - 2, y - 6, x + 7, y - 1, x + 2, y + 7,
                fill="#64d2ff", outline="#000000", width=1
            ))
            self.cursor_items.append(self.canvas.create_line(
                x - 3, y + 4, x + 3, y - 4, fill="#ffffff", width=1
            ))
        else:
            center = 2.0
            self.cursor_items.append(self.canvas.create_oval(
                x - center, y - center, x + center, y + center, fill=color, outline="#000000"
            ))

    def _event_normalized(self, event) -> tuple[float, float] | None:
        x0, y0, x1, y1 = self.display_box
        if not (x0 <= event.x <= x1 and y0 <= event.y <= y1):
            return None
        return ((event.x - x0) / max(1, x1 - x0), (event.y - y0) / max(1, y1 - y0))

    @staticmethod
    def _point_segment_distance(
        px: float, py: float, ax: float, ay: float, bx: float, by: float
    ) -> float:
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 <= 1e-12:
            return float(np.hypot(px - ax, py - ay))
        amount = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
        return float(np.hypot(px - (ax + amount * dx), py - (ay + amount * dy)))

    def _find_stroke_near(self, canvas_x: float, canvas_y: float) -> int | None:
        """Return the closest positive stroke under/near the pointer."""
        if not self.current_path or self.preview_source is None:
            return None
        x0, y0, x1, y1 = self.display_box
        if not (x0 <= canvas_x <= x1 and y0 <= canvas_y <= y1):
            return None
        display_scale = (x1 - x0) / max(1, self.current_dims[0])
        best_index: int | None = None
        best_edge_distance = float("inf")
        for index, stroke in enumerate(self.strokes.get(str(self.current_path), [])):
            if stroke.erase or not stroke.points:
                continue
            points = [(x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in stroke.points]
            if len(points) == 1:
                distance = float(np.hypot(canvas_x - points[0][0], canvas_y - points[0][1]))
            else:
                distance = min(
                    self._point_segment_distance(canvas_x, canvas_y, *start, *end)
                    for start, end in zip(points, points[1:])
                )
            edge_distance = distance - max(2.0, stroke.width * stroke.width_scale * display_scale * 0.5)
            if edge_distance <= 12.0 and edge_distance < best_edge_distance:
                best_edge_distance = edge_distance
                best_index = index
        return best_index

    def _highlight_stroke(self, index: int) -> None:
        if self.context_highlight is not None:
            self.canvas.delete(self.context_highlight)
            self.context_highlight = None
        if not self.current_path:
            return
        values = self.strokes.get(str(self.current_path), [])
        if not (0 <= index < len(values)) or not values[index].points:
            return
        x0, y0, x1, y1 = self.display_box
        transformed = transformed_stroke_points(values[index], *self.current_dims)
        coords = [value for point in transformed
                  for value in (x0 + point[0] * (x1 - x0), y0 + point[1] * (y1 - y0))]
        if len(coords) == 2:
            x, y = coords
            self.context_highlight = self.canvas.create_oval(
                x - 6, y - 6, x + 6, y + 6, outline="#ffd60a", width=3
            )
        else:
            self.context_highlight = self.canvas.create_line(
                *coords, fill="#ffd60a", width=3, capstyle="round", joinstyle="round"
            )

    def _show_mask_menu(self, event):
        index = self._find_stroke_near(event.x, event.y)
        if index is None:
            self.status.set("此处附近没有可整条删除的蒙版")
            return "break"
        self.context_stroke_index = index
        self._highlight_stroke(index)
        stroke = self.strokes[str(self.current_path)][index]
        self.mask_menu.entryconfigure(0, state="disabled" if stroke.locked else "normal")
        self.mask_menu.entryconfigure(1, label="解除锁定" if stroke.locked else "锁定这条蒙版")
        self.status.set("已选中锁定蒙版；可解除锁定" if stroke.locked else "已选中整条蒙版；可删除或锁定")
        try:
            self.mask_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.mask_menu.grab_release()
        return "break"

    def _delete_mask_at_event(self, event):
        index = self._find_stroke_near(event.x, event.y)
        if index is None:
            self.status.set("此处附近没有可整条删除的蒙版")
            return "break"
        self.context_stroke_index = index
        self._delete_context_stroke()
        return "break"

    def _record_edit(self, key: str, action: tuple[str, int, object]) -> None:
        self.edit_history.setdefault(key, []).append(action)
        self.edit_redo.pop(key, None)
        self._schedule_autosave()

    def _delete_context_stroke(self) -> None:
        if not self.current_path or self.context_stroke_index is None:
            return
        key = str(self.current_path)
        values = self.strokes.get(key, [])
        index = self.context_stroke_index
        if not (0 <= index < len(values)) or values[index].erase:
            self.context_stroke_index = None
            return
        if values[index].locked:
            self.status.set("这条蒙版已锁定；请先右键解除锁定")
            return
        stroke = values.pop(index)
        self._record_edit(key, ("delete", index, stroke))
        self.context_stroke_index = None
        if self.context_highlight is not None:
            self.canvas.delete(self.context_highlight)
            self.context_highlight = None
        self._restore_shift_anchor()
        self._update_tree_status()
        self._render_preview()
        self.status.set("已删除整条蒙版；可用 Ctrl/Command+Z 恢复")

    def _toggle_context_lock(self) -> None:
        if not self.current_path or self.context_stroke_index is None:
            return
        key = str(self.current_path)
        values = self.strokes.get(key, [])
        index = self.context_stroke_index
        if not (0 <= index < len(values)) or values[index].erase:
            return
        values[index].locked = not values[index].locked
        locked = values[index].locked
        for candidate in self.candidates.get(key, []):
            if candidate.auto_score == values[index].auto_score and candidate.points == values[index].points:
                candidate.locked = locked
        self.context_stroke_index = None
        self.edit_history.pop(key, None)
        self.edit_redo.pop(key, None)
        self._update_candidate_summary(key)
        self._render_preview()
        self.status.set("已锁定这条蒙版，调阈值或清除此图时都会保留" if locked else "已解除这条蒙版的锁定")
        self._schedule_autosave()

    def _transform_context_stroke(self) -> None:
        if not self.current_path or self.context_stroke_index is None:
            return
        values = self.strokes.get(str(self.current_path), [])
        index = self.context_stroke_index
        if not (0 <= index < len(values)) or values[index].erase:
            return
        stroke = values[index]
        dialog = tk.Toplevel(self)
        dialog.title("变换这颗流星")
        dialog.transient(self)
        dialog.grab_set()
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both", expand=True)
        variables = {
            "offset_x": tk.DoubleVar(value=stroke.offset_x),
            "offset_y": tk.DoubleVar(value=stroke.offset_y),
            "rotation": tk.DoubleVar(value=stroke.rotation),
            "length_scale": tk.DoubleVar(value=stroke.length_scale),
            "width_scale": tk.DoubleVar(value=stroke.width_scale),
            "opacity": tk.DoubleVar(value=stroke.opacity * 100.0),
        }
        rows = [
            ("水平移动(px)", "offset_x", -10000, 10000, 1),
            ("垂直移动(px)", "offset_y", -10000, 10000, 1),
            ("旋转(度)", "rotation", -180, 180, 0.1),
            ("沿流星方向拉伸", "length_scale", 0.05, 10, 0.01),
            ("横向宽度缩放", "width_scale", 0.05, 10, 0.01),
            ("不透明度(%)", "opacity", 0, 100, 1),
        ]
        for row, (label, key, minimum, maximum, increment) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Spinbox(
                frame, from_=minimum, to=maximum, increment=increment,
                textvariable=variables[key], width=14,
            ).grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
        ttk.Label(
            frame,
            text="所有变换都是非破坏性的；“恢复真实位置”会清除移动、旋转和拉伸。",
        ).grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(8, 4))
        buttons = ttk.Frame(frame)
        buttons.grid(row=len(rows) + 1, column=0, columnspan=2, sticky="e", pady=(8, 0))

        def apply_values(reset: bool = False) -> None:
            try:
                if reset:
                    stroke.offset_x = stroke.offset_y = stroke.rotation = 0.0
                    stroke.length_scale = stroke.width_scale = stroke.opacity = 1.0
                else:
                    stroke.offset_x = float(variables["offset_x"].get())
                    stroke.offset_y = float(variables["offset_y"].get())
                    stroke.rotation = float(variables["rotation"].get())
                    stroke.length_scale = max(0.05, float(variables["length_scale"].get()))
                    stroke.width_scale = max(0.05, float(variables["width_scale"].get()))
                    stroke.opacity = float(np.clip(float(variables["opacity"].get()) / 100.0, 0.0, 1.0))
            except (ValueError, tk.TclError):
                messagebox.showerror(APP_NAME, "变换参数无效", parent=dialog)
                return
            self.context_stroke_index = None
            self.edit_history.pop(str(self.current_path), None)
            self.edit_redo.pop(str(self.current_path), None)
            self._render_preview()
            self._schedule_autosave()
            dialog.destroy()

        ttk.Button(buttons, text="恢复真实位置", command=lambda: apply_values(True)).pack(side="left")
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="应用", command=apply_values).pack(side="left")

    def _stroke_start(self, event) -> None:
        if not self.current_path:
            return
        point = self._event_normalized(event)
        if point is None:
            return
        anchor = self.shift_anchors.get(str(self.current_path))
        self.active_shift_line = bool(event.state & 0x0001) and anchor is not None
        self.active_points = [anchor, point] if self.active_shift_line else [point]
        self.active_tool_mode = self.edit_mode.get()
        self.active_tool_width = self._tool_width()
        self.active_tool_feather = int(self.feather.get())
        self.active_canvas_line = None
        self.cursor_position = (float(event.x), float(event.y))
        if self.active_tool_mode == "erase":
            self.live_erase_stroke = Stroke(
                self.active_points.copy(), self.active_tool_width, self.active_tool_feather, True
            )
            values = self.strokes.setdefault(str(self.current_path), [])
            self.active_action_index = len(values)
            values.append(self.live_erase_stroke)
            self._refresh_live_erase(force=True)
        elif self.active_shift_line:
            self._draw_active_stroke()

    def _draw_active_stroke(self) -> None:
        if not self.active_points:
            return
        x0, y0, x1, y1 = self.display_box
        coords = []
        for x, y in self.active_points:
            coords.extend((x0 + x * (x1 - x0), y0 + y * (y1 - y0)))
        if self.active_canvas_line:
            self.canvas.delete(self.active_canvas_line)
        color = "#64d2ff" if self.active_tool_mode == "erase" else "#ffd60a"
        active_width = self.active_tool_width or self._tool_width()
        shown_width = max(3, int(active_width * (x1 - x0) / max(1, self.current_dims[0])))
        self.active_canvas_line = self.canvas.create_line(*coords, fill=color, width=shown_width, capstyle="round", joinstyle="round")

    def _refresh_live_erase(self, force: bool = False) -> None:
        if self.live_erase_stroke is None:
            return
        now = time.monotonic()
        if force or now - self.last_live_render >= 0.04:
            self.last_live_render = now
            self._render_preview()

    def _tool_width(self) -> int:
        return int(self.eraser_width.get() if self.edit_mode.get() == "erase" else self.brush_width.get())

    def _stroke_move(self, event) -> None:
        if not self.active_points:
            return
        point = self._event_normalized(event)
        if point is None:
            return
        self.cursor_position = (float(event.x), float(event.y))
        if self.active_shift_line:
            self.active_points[-1] = point
        else:
            self.active_points.append(point)
        if self.live_erase_stroke is not None:
            self.live_erase_stroke.points = self.active_points.copy()
            self._refresh_live_erase()
        else:
            self._draw_active_stroke()

    def _stroke_end(self, event) -> None:
        if not self.active_points or not self.current_path:
            return
        point = self._event_normalized(event)
        if point is not None:
            if self.active_shift_line:
                self.active_points[-1] = point
            elif not self.active_points or np.hypot(point[0] - self.active_points[-1][0], point[1] - self.active_points[-1][1]) > 1e-6:
                self.active_points.append(point)
        if self.live_erase_stroke is not None:
            self.live_erase_stroke.points = self.active_points.copy()
            stroke = self.live_erase_stroke
        else:
            stroke = Stroke(
                self.active_points.copy(), self.active_tool_width or self._tool_width(),
                self.active_tool_feather, False
            )
            values = self.strokes.setdefault(str(self.current_path), [])
            self.active_action_index = len(values)
            values.append(stroke)
        self._record_edit(str(self.current_path), ("add", self.active_action_index, stroke))
        self.shift_anchors[str(self.current_path)] = self.active_points[-1]
        self.active_points = []
        self.active_canvas_line = None
        self.active_shift_line = False
        self.active_tool_mode = None
        self.active_tool_width = 0
        self.active_tool_feather = 0
        self.live_erase_stroke = None
        self.active_action_index = -1
        self._update_tree_status()
        self._render_preview()
        self._schedule_autosave()

    def undo_stroke(self) -> None:
        if not self.current_path:
            return
        key = str(self.current_path)
        values = self.strokes.setdefault(key, [])
        history = self.edit_history.setdefault(key, [])
        if history:
            action = history.pop()
            kind, index, payload = action
            if kind == "add":
                if 0 <= index < len(values) and values[index] is payload:
                    values.pop(index)
                elif payload in values:
                    values.remove(payload)
            elif kind == "delete":
                values.insert(min(index, len(values)), payload)
            elif kind == "clear":
                before, _after = payload
                values[:] = list(before)
        elif values:
            index = len(values) - 1
            payload = values.pop()
            action = ("add", index, payload)
        else:
            return
        self.edit_redo.setdefault(key, []).append(action)
        self._restore_shift_anchor()
        self._update_tree_status()
        self._render_preview()
        self._schedule_autosave()

    def redo_stroke(self) -> None:
        if not self.current_path:
            return
        key = str(self.current_path)
        if not self.edit_redo.get(key):
            return
        action = self.edit_redo[key].pop()
        kind, index, payload = action
        values = self.strokes.setdefault(key, [])
        if kind == "add":
            values.insert(min(index, len(values)), payload)
        elif kind == "delete":
            if 0 <= index < len(values) and values[index] is payload:
                values.pop(index)
            elif payload in values:
                values.remove(payload)
        elif kind == "clear":
            _before, after = payload
            values[:] = list(after)
        self.edit_history.setdefault(key, []).append(action)
        self._restore_shift_anchor()
        self._update_tree_status()
        self._render_preview()
        self._schedule_autosave()

    def _restore_shift_anchor(self) -> None:
        if not self.current_path:
            return
        key = str(self.current_path)
        values = self.strokes.get(key, [])
        if values and values[-1].points:
            self.shift_anchors[key] = values[-1].points[-1]
        else:
            self.shift_anchors.pop(key, None)

    def _cancel_active_stroke(self):
        if self.current_path and self.live_erase_stroke is not None:
            key = str(self.current_path)
            values = self.strokes.get(key, [])
            if values and values[-1] is self.live_erase_stroke:
                values.pop()
        if self.active_canvas_line:
            self.canvas.delete(self.active_canvas_line)
        had_active = bool(self.active_points)
        self.active_points = []
        self.active_canvas_line = None
        self.active_shift_line = False
        self.active_tool_mode = None
        self.active_tool_width = 0
        self.active_tool_feather = 0
        self.live_erase_stroke = None
        self.active_action_index = -1
        if had_active:
            self._restore_shift_anchor()
            self._update_tree_status()
            self._render_preview()
            self.status.set("已取消当前操作")
        return "break"

    def clear_strokes(self) -> None:
        if self.current_path:
            key = str(self.current_path)
            existing = self.strokes.get(key, []).copy()
            remaining = [stroke for stroke in existing if stroke.locked]
            if existing == remaining:
                if remaining:
                    self.status.set("当前蒙版都已锁定；解除锁定后才能清除")
                return
            self._record_edit(key, ("clear", 0, (existing, remaining)))
            self.strokes[key] = remaining
            self.shift_anchors.pop(key, None)
            self._update_tree_status()
            self._update_candidate_summary(key)
            self._render_preview()
            self.status.set(f"已清除未锁定蒙版；保留 {len(remaining)} 条锁定蒙版")

    def _update_tree_status(self) -> None:
        if not self.current_path:
            return
        for index, path in enumerate(self.files):
            if path == self.current_path and self.tree.exists(str(index)):
                count = len(self.strokes.get(str(path), []))
                self.tree.set(str(index), "status", count or "—")

    def _setup_autosave(self) -> None:
        variables = (
            self.source_dir, self.base_dir, self.output_dir, self.brush_width,
            self.eraser_width, self.feather, self.match_exposure, self.curve_enabled,
            self.curve_shadows, self.curve_highlights, self.export_tiff, self.blend_mode,
        )
        for variable in variables:
            variable.trace_add("write", lambda *_args: self._schedule_autosave())
        for variable in (
            self.match_exposure, self.curve_enabled, self.curve_shadows, self.curve_highlights
        ):
            variable.trace_add("write", self._current_adjustment_changed)
        self.autosave_suspended = False

    def _project_data(self) -> dict:
        return {
            "version": PROJECT_VERSION,
            "source_dir": self.source_dir.get(),
            "base_dir": self.base_dir.get(),
            "output_dir": self.output_dir.get(),
            "match_exposure": self.match_exposure.get(),
            "curve_enabled": self.curve_enabled.get(),
            "curve_shadows": self.curve_shadows.get(),
            "curve_highlights": self.curve_highlights.get(),
            "adjustment_defaults": self.adjustment_defaults,
            "image_adjustments": self.image_adjustments,
            "export_tiff": self.export_tiff.get(),
            "blend_mode": self.blend_mode.get(),
            "brush_width": self.brush_width.get(),
            "eraser_width": self.eraser_width.get(),
            "feather": self.feather.get(),
            "candidate_thresholds": self.candidate_thresholds,
            "original_sources": {key: str(value) for key, value in self.original_sources.items()},
            "use_original_sources": sorted(self.use_original_sources),
            "alignment_statuses": self.alignment_statuses,
            "candidates": {key: [asdict(item) for item in value] for key, value in self.candidates.items()},
            "strokes": {key: [asdict(item) for item in value] for key, value in self.strokes.items()},
        }

    def _apply_project_data(self, data: dict) -> None:
        self.autosave_suspended = True
        self.loading_adjustments = True
        try:
            self.current_path = None
            self.preview_source = None
            self.preview_base = None
            self.source_dir.set(data.get("source_dir", ""))
            self.base_dir.set(data.get("base_dir", ""))
            self.output_dir.set(data.get("output_dir", ""))
            self.match_exposure.set(bool(data.get("match_exposure", False)))
            self.curve_enabled.set(bool(data.get("curve_enabled", False)))
            self.curve_shadows.set(int(data.get("curve_shadows", 15)))
            self.curve_highlights.set(int(data.get("curve_highlights", 25)))
            legacy_defaults = {
                "match_exposure": bool(data.get("match_exposure", False)),
                "curve_enabled": bool(data.get("curve_enabled", False)),
                "curve_shadows": int(data.get("curve_shadows", 15)),
                "curve_highlights": int(data.get("curve_highlights", 25)),
            }
            self.adjustment_defaults = {
                **legacy_defaults, **data.get("adjustment_defaults", {})
            }
            self.image_adjustments = {
                key: {
                    "match_exposure": bool(value.get("match_exposure", self.adjustment_defaults["match_exposure"])),
                    "curve_enabled": bool(value.get("curve_enabled", self.adjustment_defaults["curve_enabled"])),
                    "curve_shadows": int(value.get("curve_shadows", self.adjustment_defaults["curve_shadows"])),
                    "curve_highlights": int(value.get("curve_highlights", self.adjustment_defaults["curve_highlights"])),
                }
                for key, value in data.get("image_adjustments", {}).items()
            }
            self.export_tiff.set(bool(data.get("export_tiff", False)))
            self.blend_mode.set(str(data.get("blend_mode", "自然融合")))
            self.brush_width.set(int(data.get("brush_width", 18)))
            self.eraser_width.set(int(data.get("eraser_width", 40)))
            self.feather.set(int(data.get("feather", 10)))
            self.candidate_thresholds = {
                key: int(value) for key, value in data.get("candidate_thresholds", {}).items()
            }
            self.original_sources = {
                key: Path(value) for key, value in data.get("original_sources", {}).items()
            }
            self.use_original_sources = set(data.get("use_original_sources", []))
            self.alignment_statuses = {
                str(key): str(value) for key, value in data.get("alignment_statuses", {}).items()
            }

            def decode_strokes(values):
                return [Stroke(
                    points=[tuple(p) for p in item["points"]], width=item["width"],
                    feather=item["feather"], erase=bool(item.get("erase", False)),
                    locked=bool(item.get("locked", False)), auto_score=item.get("auto_score"),
                    offset_x=float(item.get("offset_x", 0.0)), offset_y=float(item.get("offset_y", 0.0)),
                    rotation=float(item.get("rotation", 0.0)),
                    length_scale=float(item.get("length_scale", 1.0)),
                    width_scale=float(item.get("width_scale", 1.0)),
                    opacity=float(item.get("opacity", 1.0)),
                ) for item in values]

            self.candidates = {
                key: decode_strokes(values) for key, values in data.get("candidates", {}).items()
            }
            self.strokes = {
                key: decode_strokes(values) for key, values in data.get("strokes", {}).items()
            }
            self.edit_history.clear()
            self.edit_redo.clear()
            self.shift_anchors.clear()
        finally:
            self.loading_adjustments = False
            self.autosave_suspended = False

    def _schedule_autosave(self) -> None:
        if self.autosave_suspended:
            return
        if self.autosave_after_id is not None:
            self.after_cancel(self.autosave_after_id)
        self.autosave_status.set("自动保存：等待写入…")
        self.autosave_after_id = self.after(800, self._write_autosave)

    def _write_autosave(self) -> None:
        self.autosave_after_id = None
        try:
            self.autosave_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.autosave_path.with_suffix(".writing.json")
            temporary.write_text(
                json.dumps(self._project_data(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            os.replace(temporary, self.autosave_path)
            self.autosave_status.set("自动保存：已保存 " + datetime.now().strftime("%H:%M:%S"))
        except Exception as exc:
            self.autosave_status.set(f"自动保存失败：{exc}")

    def _restore_autosave(self) -> None:
        if not self.autosave_path.is_file():
            self.autosave_status.set("自动保存：已开启")
            return
        try:
            data = json.loads(self.autosave_path.read_text(encoding="utf-8"))
            self._apply_project_data(data)
            source = Path(self.source_dir.get()).expanduser()
            base = Path(self.base_dir.get()).expanduser()
            if source.is_dir() and (base.is_dir() or base.is_file()) and self.output_dir.get().strip():
                self.scan_inputs()
            self.autosave_status.set("自动保存：已恢复上次会话")
        except Exception as exc:
            self.autosave_status.set(f"自动恢复失败：{exc}")

    def save_project(self) -> None:
        path = filedialog.asksaveasfilename(title="保存项目", defaultextension=".json", filetypes=[("流星项目", "*.json")])
        if not path:
            return
        Path(path).write_text(json.dumps(self._project_data(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.status.set(f"项目已保存：{path}")

    def load_project(self) -> None:
        path = filedialog.askopenfilename(title="载入项目", filetypes=[("流星项目", "*.json")])
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._apply_project_data(data)
        self.scan_inputs()
        self._schedule_autosave()
        self.status.set(f"项目已载入：{path}")

    def export(self) -> None:
        marked = {
            Path(key): [Stroke(
                item.points.copy(), item.width, item.feather, item.erase, item.locked, item.auto_score,
                item.offset_x, item.offset_y, item.rotation, item.length_scale,
                item.width_scale, item.opacity,
            ) for item in value]
            for key, value in self.strokes.items() if value
        }
        if not marked:
            messagebox.showwarning(APP_NAME, "还没有检测或标记任何流星蒙版")
            return
        source = Path(self.source_dir.get())
        base_dir = Path(self.base_dir.get())
        output_text = self.output_dir.get().strip()
        if not output_text:
            messagebox.showerror(APP_NAME, "请选择独立输出文件夹")
            return
        output = Path(output_text)
        if (path_is_within(output, source) or path_is_within(source, output)
                or path_is_within(output, base_dir) or path_is_within(base_dir, output)):
            messagebox.showerror(APP_NAME, "输出目录必须与 TIFF/JPG 两个输入文件夹完全分开")
            return
        learning_choice = messagebox.askyesnocancel(
            APP_NAME + " — 导出与 AI 学习",
            f"将逐对处理 {len(marked)} 组同名 TIFF/JPG。\n"
            f"源素材只读，结果写入新的运行目录：\n{output}\n\n"
            "是否在导出完成后，用本次最终蒙版继续训练本地 AI？\n\n"
            "是：导出并自动学习\n否：仅导出\n取消：停止",
        )
        if learning_choice is None:
            return
        self.progress["value"] = 0
        self.status.set("开始导出…")
        self._run_worker(
            self._export_worker, output, marked, self.pairs.copy(),
            {key: value.copy() for key, value in self.image_adjustments.items()},
            self.adjustment_defaults.copy(),
            bool(self.export_tiff.get()), bool(learning_choice), self.blend_mode.get(),
            {str(path): self._effective_source_path(path) for path in marked},
        )

    def _export_worker(
        self, output_dir: Path, marked: dict[Path, list[Stroke]], pairs: dict[str, Path],
        adjustments: dict[str, dict], adjustment_defaults: dict,
        export_tiff: bool, learn_after_export: bool, blend_mode: str,
        read_paths: dict[str, Path],
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = unique_path(output_dir / ("meteor_restore_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
        jpg_dir = run_dir / "final_jpg"
        masks_dir = run_dir / "masks"
        jpg_dir.mkdir(parents=True)
        masks_dir.mkdir()
        tif_dir = run_dir / "final_tiff"
        if export_tiff:
            tif_dir.mkdir()
        report = []
        total = len(marked)
        for index, (source_path, strokes) in enumerate(marked.items(), start=1):
            adjustment = {**adjustment_defaults, **adjustments.get(str(source_path), {})}
            match = bool(adjustment["match_exposure"])
            curve_enabled = bool(adjustment["curve_enabled"])
            curve_shadows = float(adjustment["curve_shadows"])
            curve_highlights = float(adjustment["curve_highlights"])
            if str(source_path) not in pairs:
                raise ValueError(f"找不到同名底图配对：{source_path.name}")
            base_path = pairs[str(source_path)]
            base = to_uint16(read_image(base_path))
            height, width = base.shape[:2]
            result = base.copy()
            actual_source_path = read_paths.get(str(source_path), source_path)
            source = read_image(actual_source_path)
            if source.shape[:2] != (height, width):
                raise ValueError(f"尺寸不一致：{source_path.name}")
            source16 = to_uint16(source)
            result, mask_float = compose_meteor_objects(
                source16, base, strokes, match, curve_enabled,
                curve_shadows, curve_highlights, blend_mode,
            )
            if not np.any(mask_float > 0.001):
                continue
            mask_full = np.clip(mask_float * 65535, 0, 65535).astype(np.uint16)
            mask_path = masks_dir / f"{source_path.stem}_mask.png"
            ok, encoded = cv2.imencode(".png", mask_full)
            if not ok:
                raise IOError(f"蒙版保存失败：{mask_path.name}")
            encoded.tofile(mask_path)
            jpg_path = jpg_dir / base_path.name
            atomic_write_jpeg(jpg_path, result)
            outputs = [str(jpg_path)]
            if export_tiff:
                tif_path = tif_dir / f"{source_path.stem}.tif"
                atomic_write_tiff(tif_path, result)
                outputs.append(str(tif_path))
            report.append({
                "source_tiff": str(actual_source_path), "workspace_layer": str(source_path),
                "clean_jpg": str(base_path),
                "strokes": len(strokes), "mask": str(mask_path),
                "local_match": match, "meteor_curve": curve_enabled,
                "meteor_highlight_protection": match,
                "curve_shadows": curve_shadows, "curve_highlights": curve_highlights,
                "blend_mode": blend_mode,
                "outputs": outputs,
            })
            del source, source16, mask_full, base, result
            self.work_queue.put(("progress", index / total * 90, f"正在合成 {index}/{total}：{source_path.name}"))
        manifest = {
            "version": PROJECT_VERSION,
            "per_image_adjustments": True,
            "meteor_highlight_protection": True,
            "adjustment_defaults": adjustment_defaults,
            "export_tiff": export_tiff,
            "items": report,
        }
        (run_dir / "processing_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return "exported", run_dir, len(report), learn_after_export, marked, pairs, read_paths

    def _learn_worker(self, marked: dict[Path, list[Stroke]], pairs: dict[str, Path]):
        try:
            if self.ranker_model is None:
                raise ValueError("找不到可继续训练的基础 AI 模型")
            dataset_path = bundled_resource_path("candidate_dataset.npz")
            if not dataset_path.is_file():
                raise ValueError("安装包缺少基础训练数据 candidate_dataset.npz")
            from meteor_learning import learn_from_feedback
            toolkit = {
                "Stroke": Stroke,
                "read_image": read_image,
                "detection_preview": detection_preview,
                "build_mask_crop": build_mask_crop,
                "detect_trails": detect_trails,
                "prepare_ml_maps": prepare_ml_maps,
                "candidate_feature_vector": candidate_feature_vector,
                "ML_FEATURE_NAMES": ML_FEATURE_NAMES,
            }
            report = learn_from_feedback(
                marked, pairs, toolkit, dataset_path, self.ranker_model,
                user_model_file_path(),
                lambda value, text: self.work_queue.put(("progress", value, text)),
            )
            return "learned", report
        except Exception as exc:
            return "learning_failed", str(exc), traceback.format_exc()

    def _run_worker(self, function, *args) -> None:
        def runner():
            try:
                self.work_queue.put(function(*args))
            except Exception as exc:
                self.work_queue.put(("error", str(exc), traceback.format_exc()))
        threading.Thread(target=runner, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.work_queue.get_nowait()
                kind = item[0]
                if kind == "preview":
                    _, path, source_preview, base_preview, dims = item
                    self.current_path = path
                    self.preview_source = source_preview
                    self.preview_base = base_preview
                    self.preview_rgb = source_preview
                    self.current_dims = dims
                    key = str(path)
                    self._load_image_adjustments(key)
                    self.setting_candidate_threshold = True
                    self.candidate_threshold.set(self.candidate_thresholds.get(key, 55))
                    self.setting_candidate_threshold = False
                    self._update_candidate_summary(key)
                    self.status.set(f"已加载 {path.name}；可拖动画笔，或单击起点后 Shift+单击终点画直线。")
                    self._render_preview()
                elif kind == "progress":
                    _, value, text = item
                    self.progress["value"] = value
                    self.status.set(text)
                elif kind == "autodetected":
                    _, found, plane_count = item
                    locked = {
                        key: [stroke for stroke in values if stroke.locked]
                        for key, values in self.strokes.items() if any(stroke.locked for stroke in values)
                    }
                    self.strokes = found
                    for key, values in locked.items():
                        destination = self.strokes.setdefault(key, [])
                        destination.extend(
                            stroke for stroke in values
                            if not any(existing.points == stroke.points for existing in destination)
                        )
                    self.edit_history.clear()
                    self.edit_redo.clear()
                    self._schedule_autosave()
                    self.shift_anchors.clear()
                    for index, path in enumerate(self.files):
                        if self.tree.exists(str(index)):
                            count = len(self.strokes.get(str(path), []))
                            self.tree.set(str(index), "status", count or "—")
                    meteor_count = sum(len(value) for value in found.values())
                    self.progress["value"] = 100
                    self.status.set(f"自动检测完成：候选流星 {meteor_count} 条，疑似飞机线 {plane_count} 条已排除。请抽查红色蒙版。")
                    if found:
                        first_path = Path(next(iter(found)))
                        first_index = self.files.index(first_path)
                        self.tree.selection_set(str(first_index))
                        self.tree.focus(str(first_index))
                        self.load_selected()
                    else:
                        messagebox.showinfo(APP_NAME, "没有检测到可靠的流星候选。可调整素材或用画笔补充。")
                elif kind == "candidates":
                    _, path, candidates, plane_count = item
                    key = str(path)
                    self.candidates[key] = candidates
                    self.candidate_thresholds.setdefault(key, int(self.candidate_threshold.get()))
                    self.edit_history.pop(key, None)
                    self.edit_redo.pop(key, None)
                    self._apply_candidate_threshold(key)
                    self._schedule_autosave()
                    self.progress["value"] = 100
                    if self.current_path == path:
                        self.setting_candidate_threshold = True
                        self.candidate_threshold.set(self.candidate_thresholds[key])
                        self.setting_candidate_threshold = False
                        self._update_tree_status()
                        self._render_preview()
                    self.status.set(
                        f"单张候选分析完成：{len(candidates)} 条候选，排除疑似飞机线 {plane_count} 条；"
                        "拖动分数阈值筛选，确认后右键锁定。"
                    )
                elif kind == "exported":
                    _, run_dir, count, learn_after_export, marked, pairs, read_paths = item
                    self.progress["value"] = 100
                    self.status.set(f"导出完成：{run_dir}")
                    if learn_after_export:
                        messagebox.showinfo(
                            APP_NAME,
                            f"导出完成，共处理 {count} 对文件。\n\n结果目录：\n{run_dir}\n\n"
                            "现在开始在后台验证并训练个性化 AI。导出结果不受训练成败影响。",
                        )
                        self.progress["value"] = 0
                        self.status.set("正在从本次最终蒙版学习…")
                        learning_marked = {
                            read_paths.get(str(path), path): strokes for path, strokes in marked.items()
                        }
                        learning_pairs = {
                            str(read_paths.get(str(path), path)): pairs[str(path)] for path in marked
                        }
                        self._run_worker(self._learn_worker, learning_marked, learning_pairs)
                    else:
                        messagebox.showinfo(APP_NAME, f"导出完成，共处理 {count} 对文件。\n\n结果目录：\n{run_dir}")
                elif kind == "learned":
                    _, report = item
                    self.progress["value"] = 100
                    validation = report["validation"]
                    if report["accepted"]:
                        self.ranker_model = load_meteor_ranker()
                        self.ai_model_status.set("AI模型：个性化")
                        self.status.set("个性化 AI 学习完成并已启用")
                        messagebox.showinfo(
                            APP_NAME + " — AI 学习完成",
                            f"新模型已通过留图验证并启用。\n"
                            f"反馈样本：{report['feedback_samples']}（正样本 {report['feedback_positive']}）\n"
                            f"平均精确率：{validation['average_precision']:.3f}\n"
                            f"逐图最高分正确率：{validation['top1_hit']}/{validation['top1_total']} "
                            f"({validation['top1_accuracy']:.1%})\n\n旧模型已自动备份。",
                        )
                    else:
                        self.status.set("新模型验证未通过，继续使用原模型")
                        messagebox.showwarning(
                            APP_NAME + " — 未替换 AI",
                            f"训练已完成，但新模型验证没有达到安全标准，因此未替换原模型。\n"
                            f"平均精确率：{validation['average_precision']:.3f}\n"
                            f"逐图最高分正确率：{validation['top1_accuracy']:.1%}",
                        )
                elif kind == "learning_failed":
                    _, text, details = item
                    self.progress["value"] = 100
                    self.status.set("AI 学习失败；导出结果不受影响")
                    messagebox.showerror(
                        APP_NAME + " — AI 学习失败",
                        f"照片已经正常导出，但 AI 学习没有完成：\n{text}\n\n原模型保持不变。",
                    )
                    print(details)
                elif kind == "error":
                    _, text, details = item
                    self.status.set("处理失败")
                    messagebox.showerror(APP_NAME, f"{text}\n\n详细信息已打印到终端。")
                    print(details)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)


if __name__ == "__main__":
    MeteorComposer().mainloop()
