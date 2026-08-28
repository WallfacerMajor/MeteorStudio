from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
import gc
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import psutil
import tifffile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk
from alignment_workspace import open_alignment_workspace
from meteor_screening import open_screening_workspace
from video_meteor import open_video_workspace
from platform_utils import open_folder
from error_dialog import show_copyable_error, show_runtime_log


APP_NAME = "流星影像工坊"
APP_VERSION = "0.2.0-full-resolution-preview"
PROJECT_VERSION = 28
TIFF_SUFFIXES = {".tif", ".tiff"}


def preview_memory_budgets() -> tuple[int, int, int]:
    """Choose bounded caches from currently available RAM, not image count."""
    available = int(psutil.virtual_memory().available)
    display = int(np.clip(available * 0.12, 512 << 20, 1536 << 20))
    precision = int(np.clip(available * 0.07, 384 << 20, 1024 << 20))
    viewport = int(np.clip(available * 0.02, 96 << 20, 256 << 20))
    return display, precision, viewport


def ordered_prefetch(items: Iterable, loader, workers: int = 2):
    """Yield in source order while decoding a bounded number of future items."""
    sequence = iter(items)
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="meteor-pipeline") as pool:
        pending = deque()
        for _ in range(max(1, workers)):
            try:
                item = next(sequence)
            except StopIteration:
                break
            pending.append((item, pool.submit(loader, item)))
        while pending:
            item, future = pending.popleft()
            try:
                following = next(sequence)
            except StopIteration:
                following = None
            if following is not None:
                pending.append((following, pool.submit(loader, following)))
            yield item, future.result()


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
    brightness_override: float | None = None
    background_cleanup_override: float | None = None
    saturation_override: float | None = None
    preserve_brightness_override: bool | None = None
    match_exposure_override: bool | None = None
    blend_mode_override: str | None = None
    auto_blend_enabled: bool = True
    auto_strength: str = "标准"
    auto_black_point: float | None = None
    auto_cleanup: float | None = None
    auto_brightness: float | None = None
    auto_feather: int | None = None
    # Pixel source used by this meteor.  Geometry remains independent, so a
    # meteor extracted from the untouched frame can still be moved onto the
    # aligned clean-base canvas without forcing its neighbours back to raw.
    source_mode: str = "aligned"


def normalized_source_mode(stroke: Stroke) -> str:
    return "original" if stroke.source_mode == "original" else "aligned"


def stroke_is_transformed(stroke: Stroke, tolerance: float = 1e-4) -> bool:
    """Return whether a meteor has a non-default geometric transform."""
    return (
        abs(float(stroke.offset_x)) > tolerance
        or abs(float(stroke.offset_y)) > tolerance
        or abs(float(stroke.rotation)) > tolerance
        or abs(float(stroke.length_scale) - 1.0) > tolerance
        or abs(float(stroke.width_scale) - 1.0) > tolerance
    )


def reset_stroke_geometry(stroke: Stroke) -> None:
    """Restore only geometry; keep mask, opacity, and per-meteor adjustments."""
    stroke.offset_x = 0.0
    stroke.offset_y = 0.0
    stroke.rotation = 0.0
    stroke.length_scale = 1.0
    stroke.width_scale = 1.0


def active_stroke_keys(
    strokes: dict[str, list[Stroke]], pairs: dict[str, Path],
) -> list[str]:
    """Return only marked sources that belong to the current input pairing.

    Autosave intentionally remembers masks from earlier batches.  Those masks
    must remain recoverable, but they must never leak into preview or export
    after the user selects a different source set and clean base.
    """
    return [key for key, values in strokes.items() if values and key in pairs]


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
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.nanmax(array))
        if not np.isfinite(maximum):
            raise ValueError("图像没有有效像素")
        normalized_linear = maximum <= 1.5
        output = np.empty(array.shape, dtype=np.uint16)
        pixels_per_row = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
        rows_per_chunk = max(1, 4_000_000 // max(1, pixels_per_row))
        # Work in bounded row chunks: a 7952×5304 float TIFF is ~500 MB, and a
        # whole-frame gamma expression would temporarily allocate several copies.
        for start in range(0, array.shape[0], rows_per_chunk):
            stop = min(array.shape[0], start + rows_per_chunk)
            chunk = np.asarray(array[start:stop], dtype=np.float32).copy()
            np.nan_to_num(chunk, copy=False, nan=0.0, posinf=1.0 if normalized_linear else 65535.0, neginf=0.0)
            if normalized_linear:
                np.clip(chunk, 0.0, 1.0, out=chunk)
                low = chunk <= 0.0031308
                chunk[low] *= 12.92
                high = ~low
                chunk[high] = 1.055 * np.power(chunk[high], 1.0 / 2.4) - 0.055
                chunk *= 65535.0
            else:
                np.clip(chunk, 0.0, 65535.0, out=chunk)
            output[start:stop] = chunk.astype(np.uint16)
        return output
    result = array.astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=65535.0, neginf=0.0).clip(0, 65535).astype(np.uint16)


def read_uint16_pair(first: Path, second: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode two independent layers concurrently without decoding duplicates."""
    if first == second:
        image = to_uint16(read_image(first))
        return image, image
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="meteor-decode") as pool:
        future_first = pool.submit(lambda: to_uint16(read_image(first)))
        future_second = pool.submit(lambda: to_uint16(read_image(second)))
        return future_first.result(), future_second.result()


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


def meteor_mask_boxes(mask: np.ndarray, preview_limit: int = 2000) -> list[tuple[int, int, int, int]]:
    """Return compact full-resolution boxes for disconnected meteor regions."""
    height, width = mask.shape[:2]
    scale = min(1.0, preview_limit / max(1, width, height))
    if scale < 1.0:
        # OpenCV resize has no float16 implementation. Export masks are kept in
        # float16 to limit 8K memory use, so promote only this small analysis
        # input rather than failing on long/transformed meteor footprints.
        resize_input = mask.astype(np.float32) if mask.dtype == np.float16 else mask
        preview = cv2.resize(
            resize_input, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = mask
    binary = np.where(preview > 0.06, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return []
    kernel_size = max(3, int(round(min(binary.shape[:2]) * 0.003)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inverse = 1.0 / scale
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < 4:
            continue
        x0 = max(0, int(np.floor(x * inverse)))
        y0 = max(0, int(np.floor(y * inverse)))
        x1 = min(width, int(np.ceil((x + w) * inverse)))
        y1 = min(height, int(np.ceil((y + h) * inverse)))
        boxes.append((x0, y0, x1, y1))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def annotate_meteor_sources(
    image16: np.ndarray, annotations: list[dict]
) -> tuple[np.ndarray, list[dict]]:
    """Create an 8-bit review copy with source names beside each meteor region."""
    shown = np.right_shift(to_uint16(image16), 8).astype(np.uint8)
    height, width = shown.shape[:2]
    longest = max(width, height)
    font_scale = float(np.clip(longest / 3600.0, 0.65, 2.4))
    thickness = max(1, round(font_scale * 1.6))
    pad = max(4, round(font_scale * 5))
    palette = (
        (255, 190, 45), (80, 220, 255), (255, 105, 190), (120, 255, 120),
        (195, 135, 255), (255, 145, 80),
    )
    label_records = []
    original_count = sum(
        len(annotation.get("boxes", []))
        for annotation in annotations if annotation.get("original_state")
    )
    transformed_count = sum(
        len(annotation.get("boxes", []))
        for annotation in annotations if annotation.get("transformed")
    )
    banner_bottom = 0
    if original_count:
        warning = f"WARNING: RED = ORIGINAL / UNALIGNED SOURCE - CHECK POSITION ({original_count})"
        warning_scale = max(0.55, font_scale * 0.82)
        warning_thickness = max(1, round(thickness * 0.9))
        (warning_width, warning_height), warning_baseline = cv2.getTextSize(
            warning, cv2.FONT_HERSHEY_SIMPLEX, warning_scale, warning_thickness
        )
        banner_height = min(height, warning_height + warning_baseline + pad * 3)
        banner_bottom = banner_height
        cv2.rectangle(shown, (0, 0), (min(width - 1, warning_width + pad * 3), banner_height), (145, 0, 0), -1)
        cv2.putText(
            shown, warning, (pad, warning_height + pad), cv2.FONT_HERSHEY_SIMPLEX,
            warning_scale, (255, 235, 80), warning_thickness, cv2.LINE_AA,
        )
    if transformed_count:
        notice = f"ORANGE = MANUALLY TRANSFORMED; CYAN DASH = ORIGINAL POSITION ({transformed_count})"
        notice_scale = max(0.52, font_scale * 0.75)
        notice_thickness = max(1, round(thickness * 0.85))
        (notice_width, notice_height), notice_baseline = cv2.getTextSize(
            notice, cv2.FONT_HERSHEY_SIMPLEX, notice_scale, notice_thickness
        )
        notice_top = min(height - 1, banner_bottom)
        notice_bottom = min(height, notice_top + notice_height + notice_baseline + pad * 3)
        if notice_bottom > notice_top:
            cv2.rectangle(
                shown, (0, notice_top),
                (min(width - 1, notice_width + pad * 3), notice_bottom), (75, 35, 0), -1,
            )
            cv2.putText(
                shown, notice, (pad, min(height - pad, notice_top + notice_height + pad)),
                cv2.FONT_HERSHEY_SIMPLEX, notice_scale, (255, 190, 45),
                notice_thickness, cv2.LINE_AA,
            )

    def dashed_line(start, end, color, line_width):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        distance = float(np.linalg.norm(end - start))
        if distance < 1.0:
            return
        direction = (end - start) / distance
        position = 0.0
        dash = max(5.0, line_width * 4.0)
        while position < distance:
            segment_end = min(distance, position + dash)
            a = start + direction * position
            b = start + direction * segment_end
            cv2.line(
                shown, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                color, line_width,
            )
            position += dash * 2.0

    def dashed_rectangle(box, color, line_width):
        x0, y0, x1, y1 = box
        x1, y1 = max(x0, x1 - 1), max(y0, y1 - 1)
        for start, end in (
            ((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
        ):
            dashed_line(start, end, color, line_width)

    for source_index, annotation in enumerate(annotations):
        source_name = str(annotation["source"])
        boxes = annotation.get("boxes", [])
        original_state = bool(annotation.get("original_state", False))
        transformed = bool(annotation.get("transformed", False))
        original_boxes = annotation.get("original_boxes", [])
        color = (
            (255, 45, 45) if original_state
            else ((255, 145, 35) if transformed else palette[source_index % len(palette)])
        )
        for region_index, (x0, y0, x1, y1) in enumerate(boxes, start=1):
            meteor_index = annotation.get("meteor_index")
            suffix = (
                f" #{meteor_index}" if meteor_index is not None
                else (f" #{region_index}" if len(boxes) > 1 else "")
            )
            prefix_parts = []
            if original_state:
                prefix_parts.append("!! ORIGINAL / UNALIGNED !!")
            if transformed:
                prefix_parts.append("!! TRANSFORMED !!")
            prefix = (" ".join(prefix_parts) + " ") if prefix_parts else ""
            label = (prefix + source_name + suffix).encode("ascii", "replace").decode("ascii")
            cv2.rectangle(shown, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, thickness)
            original_box = original_boxes[region_index - 1] if region_index <= len(original_boxes) else None
            if transformed and original_box is not None:
                dashed_rectangle(original_box, (55, 230, 255), max(1, thickness))
                ox0, oy0, ox1, oy1 = original_box
                original_center = ((ox0 + ox1) // 2, (oy0 + oy1) // 2)
                transformed_center = ((x0 + x1) // 2, (y0 + y1) // 2)
                if original_center != transformed_center:
                    cv2.arrowedLine(
                        shown, original_center, transformed_center, (55, 230, 255),
                        max(1, thickness), cv2.LINE_AA, tipLength=0.12,
                    )
            if original_state:
                inset = max(2, thickness * 2)
                cv2.rectangle(
                    shown, (max(0, x0 - inset), max(0, y0 - inset)),
                    (min(width - 1, x1 - 1 + inset), min(height - 1, y1 - 1 + inset)),
                    (255, 230, 40), max(1, thickness),
                )
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            label_x = int(np.clip(x0, 0, max(0, width - text_width - pad * 2)))
            preferred_y = y0 - pad * 2
            if preferred_y - text_height - baseline < 0:
                preferred_y = min(height - baseline - pad, y1 + text_height + pad * 2)
            text_y = int(np.clip(preferred_y, text_height + pad, height - baseline - pad))
            cv2.rectangle(
                shown,
                (label_x, text_y - text_height - pad),
                (min(width - 1, label_x + text_width + pad * 2), min(height - 1, text_y + baseline + pad)),
                (10, 10, 10), -1,
            )
            cv2.putText(
                shown, label, (label_x + pad, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA,
            )
            cv2.line(shown, (x0, y0), (label_x + pad, text_y + baseline), color, thickness)
            label_records.append({
                "label": label, "source": source_name, "box": [x0, y0, x1, y1],
                "original_state": original_state,
                "transformed": transformed,
                "original_box": list(original_box) if original_box is not None else None,
                "transform": annotation.get("transform"),
                "warning": "original source; verify alignment and position" if original_state else None,
            })
    return shown, label_records


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


def estimate_trail_mask_geometry(
    source_preview: np.ndarray, base_preview: np.ndarray,
    start: tuple[int, int], end: tuple[int, int], source_scale: float,
) -> tuple[int, int]:
    """Measure a trail's transverse signal and return full-resolution width/feather.

    Detection lines describe direction and length, not physical thickness.  Sample
    many cross-sections of the positive source/base residual so a long, thin
    meteor stays thin while a genuinely broad head or tail receives more room.
    """
    source = source_preview.astype(np.float32)
    base = base_preview.astype(np.float32)
    if source.shape != base.shape or source.ndim != 3:
        raise ValueError("流星宽度分析要求原图与底图尺寸一致")
    source_luma = source[..., 0] * 0.0722 + source[..., 1] * 0.7152 + source[..., 2] * 0.2126
    base_luma = base[..., 0] * 0.0722 + base[..., 1] * 0.7152 + base[..., 2] * 0.2126
    residual = source_luma - base_luma
    vector = np.asarray((end[0] - start[0], end[1] - start[1]), dtype=np.float32)
    length = float(np.hypot(*vector))
    scale = max(0.001, float(source_scale))
    if length < 2.0:
        return max(12, int(round(7.0 / scale))), max(6, int(round(3.0 / scale)))

    direction = vector / length
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
    half_span = int(np.clip(round(length * 0.10), 12, 42))
    along_count = int(np.clip(round(length / 3.0), 18, 72))
    along = np.linspace(0.08, 0.92, along_count, dtype=np.float32)
    centers = np.asarray(start, dtype=np.float32)[None, :] + along[:, None] * vector[None, :]
    offsets = np.arange(-half_span, half_span + 1, dtype=np.float32)
    sample_points = centers[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    map_x = sample_points[..., 0].astype(np.float32)
    map_y = sample_points[..., 1].astype(np.float32)
    sampled = cv2.remap(
        residual, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1000.0,
    )
    valid = sampled > -999.0
    outer = np.abs(offsets) >= half_span * 0.66
    outer_values = sampled[:, outer]
    outer_valid = valid[:, outer]
    finite_outer = outer_values[outer_valid]
    if finite_outer.size < 20:
        fallback_width = float(np.clip(length * 0.055, 7.0, 20.0))
        return (
            int(np.clip(round(fallback_width / scale), 12, 160)),
            int(np.clip(round(max(3.0, fallback_width * 0.42) / scale), 6, 80)),
        )

    background = float(np.median(finite_outer))
    noise = max(0.35, 1.4826 * float(np.median(np.abs(finite_outer - background))))
    normalized = sampled - background
    normalized[~valid] = 0.0
    profile = np.percentile(normalized, 72.0, axis=0).astype(np.float32)
    profile = cv2.GaussianBlur(profile[None, :], (0, 0), sigmaX=1.15)[0]
    central = np.abs(offsets) <= half_span * 0.72
    central_indices = np.flatnonzero(central)
    peak_index = int(central_indices[np.argmax(profile[central])])
    peak = max(0.0, float(profile[peak_index]))
    threshold = max(noise * 2.6, peak * 0.075, 0.8)
    active = np.isfinite(profile) & (profile >= threshold)
    active = cv2.morphologyEx(
        active.astype(np.uint8)[None, :], cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)
    )[0] > 0
    if not active[peak_index] or peak < noise * 2.2:
        signal_width = float(np.clip(length * 0.045, 5.0, 16.0))
    else:
        left = peak_index
        right = peak_index
        while left > 0 and active[left - 1]:
            left -= 1
        while right + 1 < active.size and active[right + 1]:
            right += 1
        signal_width = float(right - left + 1)

    # A small noise-dependent guard preserves faint edge pixels without carrying
    # the broad rectangular background used by the previous length heuristic.
    guard = float(np.clip(2.0 + noise * 0.18, 2.0, 5.0))
    preview_width = float(np.clip(signal_width + guard * 2.0, 7.0, 36.0))
    preview_feather = float(np.clip(signal_width * 0.38 + guard * 0.55, 3.0, 18.0))
    return (
        int(np.clip(round(preview_width / scale), 12, 180)),
        int(np.clip(round(preview_feather / scale), 6, 90)),
    )


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


def normalize_lsd_lines(lines: np.ndarray | None) -> np.ndarray:
    """Normalize OpenCV LSD output across versions to an (N, 4) array."""
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    values = np.asarray(lines, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        return np.empty((0, 4), dtype=np.float32)
    return values.reshape(-1, 4)


def point_in_padded_bbox(
    x: float, y: float, bbox: tuple[float, float, float, float] | None, padding: float = 0.0
) -> bool:
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    return x0 - padding <= x <= x1 + padding and y0 - padding <= y <= y1 + padding


def content_distance_map(image: np.ndarray) -> np.ndarray:
    """Distance to black projection padding in an aligned source image."""
    signal = np.max(image.astype(np.float32), axis=2)
    valid = (signal > 1.0).astype(np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform(valid, cv2.DIST_L2, 5)


def line_inside_valid_content(
    distance: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> bool:
    """Reject lines running on PTGui's black projection/panorama boundary."""
    height, width = distance.shape
    length = max(1.0, float(np.hypot(end[0] - start[0], end[1] - start[1])))
    samples = max(30, int(length))
    xs = np.linspace(start[0], end[0], samples).clip(0, width - 1).astype(int)
    ys = np.linspace(start[1], end[1], samples).clip(0, height - 1).astype(int)
    # A generous inset is intentional: LSD often finds a second line tens of
    # pixels inside a resampled PTGui boundary, not just the zero-valued edge.
    safe_margin = max(8.0, min(height, width) * 0.035)
    return float(np.mean(distance[ys, xs] >= safe_margin)) >= 0.90


def merge_collinear_candidates(candidates: list[tuple]) -> list[tuple]:
    """Join LSD fragments that describe one continuous physical trail.

    A meteor with a weak middle or a saturated core is commonly returned as
    several collinear pieces.  Comparing only segment centres leaves those
    pieces as separate clickable candidates.  This merges their projected
    intervals while retaining parallel but spatially separate trails.
    """
    merged: list[tuple] = []
    for item in sorted(candidates, reverse=True, key=lambda value: value[0]):
        score, length, angle, midpoint, start, end = item
        joined = False
        for index, kept in enumerate(merged):
            kept_score, kept_length, kept_angle, _kept_midpoint, kept_start, kept_end = kept
            delta = angle - kept_angle
            angle_delta = abs(np.arctan2(np.sin(2 * delta), np.cos(2 * delta))) / 2
            if angle_delta > np.deg2rad(7.0):
                continue
            direction = np.asarray([np.cos(kept_angle), np.sin(kept_angle)], np.float32)
            normal = np.asarray([-direction[1], direction[0]], np.float32)
            first = np.asarray([kept_start, kept_end], np.float32)
            second = np.asarray([start, end], np.float32)
            first_normal = float(np.mean(first @ normal))
            second_normal = float(np.mean(second @ normal))
            perpendicular = abs(first_normal - second_normal)
            maximum_width = max(9.0, min(18.0, min(kept_length, length) * 0.10))
            if perpendicular > maximum_width:
                continue
            first_interval = np.sort(first @ direction)
            second_interval = np.sort(second @ direction)
            axial_gap = max(
                0.0,
                float(max(first_interval[0], second_interval[0]) - min(first_interval[1], second_interval[1])),
            )
            maximum_gap = max(14.0, min(65.0, min(kept_length, length) * 0.55))
            if axial_gap > maximum_gap:
                continue
            points = np.concatenate((first, second), axis=0)
            projections = points @ direction
            center_normal = float(np.median(points @ normal))
            new_start = direction * float(np.min(projections)) + normal * center_normal
            new_end = direction * float(np.max(projections)) + normal * center_normal
            new_start_tuple = tuple(int(round(value)) for value in new_start)
            new_end_tuple = tuple(int(round(value)) for value in new_end)
            new_length = float(np.linalg.norm(new_end - new_start))
            new_midpoint = (
                (new_start_tuple[0] + new_end_tuple[0]) * 0.5,
                (new_start_tuple[1] + new_end_tuple[1]) * 0.5,
            )
            merged[index] = (
                max(score, kept_score), new_length, kept_angle, new_midpoint,
                new_start_tuple, new_end_tuple,
            )
            joined = True
            break
        if not joined:
            merged.append(item)

    # A fragment can bridge two groups, so repeat until the result is stable.
    if len(merged) < len(candidates) and len(merged) > 1:
        second_pass = merge_collinear_candidates(merged)
        if len(second_pass) < len(merged):
            return second_pass
    return sorted(merged, reverse=True, key=lambda value: value[0])


def detect_trails(
    source: np.ndarray, base: np.ndarray, ranked: bool = False,
    valid_region: np.ndarray | None = None,
):
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
    region = None
    if valid_region is not None:
        if valid_region.shape != magnitude.shape:
            region = cv2.resize(
                valid_region.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            region = valid_region > 0
        region = cv2.morphologyEx(
            region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8),
        ) > 0
    content_distance = content_distance_map(source)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    detected_parts = []
    percentile_pairs = ((97.0, 99.90), (93.0, 99.65), (88.0, 99.30), (80.0, 99.00))
    # One partition pass is substantially cheaper than rescanning the complete
    # image for each detector band. Values and detection behavior are unchanged.
    percentile_values = np.percentile(magnitude, [value for pair in percentile_pairs for value in pair])
    for band_index, _pair in enumerate(percentile_pairs):
        low, high = percentile_values[band_index * 2:band_index * 2 + 2]
        enhanced = np.clip((magnitude - low) * 255.0 / max(1.0, high - low), 0, 255).astype(np.uint8)
        if region is None:
            enhanced[int(height * 0.82):] = 0
        else:
            enhanced[~region] = 0
        lines = normalize_lsd_lines(detector.detect(enhanced)[0])
        if len(lines):
            detected_parts.append(lines)
    if not detected_parts:
        return [], 0
    detected = np.concatenate(detected_parts, axis=0)

    # At a 1400px RAW screening proxy, short/faint meteors can be only 14–20px
    # long. The later temporal profile and AI score are safer rejection gates
    # than discarding them here solely by length.
    min_length = max(12.0, max(height, width) * 0.010)
    base_u8 = np.clip(dst, 0, 255).astype(np.uint8)
    structural_edges = cv2.Canny(base_u8, 35, 90)
    structural_edges = cv2.dilate(structural_edges, np.ones((5, 5), np.uint8)) > 0
    bright_limit = float(np.percentile(dst, 97))
    raw_candidates = []
    for raw in detected:
        x1, y1, x2, y2 = (float(v) for v in raw)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        start, end = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        if not line_inside_valid_content(content_distance, start, end):
            continue
        samples = max(30, int(length))
        xs = np.linspace(start[0], end[0], samples).clip(0, width - 1).astype(int)
        ys = np.linspace(start[1], end[1], samples).clip(0, height - 1).astype(int)
        if region is None:
            if float(np.median(ys)) > height * 0.79:
                continue
        else:
            sky_fraction = float(np.mean(region[ys, xs]))
            middle = slice(samples // 4, max(samples // 4 + 1, samples * 3 // 4))
            if sky_fraction < 0.78 or float(np.mean(region[ys[middle], xs[middle]])) < 0.90:
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

    # Merge duplicate LSD edges and separated weak/core/tail fragments into one
    # candidate before classification and user interaction.
    candidates = merge_collinear_candidates(raw_candidates)

    meteors = []
    ranked_meteors = []
    planes = 0
    best_score = candidates[0][0]
    global_profile_threshold = float(np.percentile(magnitude, 99.65))
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
        profile_median = float(np.median(profile))
        profile_threshold = max(
            global_profile_threshold,
            float(profile_median + 6 * np.median(np.abs(profile - profile_median))),
        )
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


def stroke_for_image_crop(
    stroke: Stroke, full_width: int, full_height: int,
    x0: int, y0: int, crop_width: int, crop_height: int,
) -> Stroke:
    """Translate normalized geometry into an ROI without changing pixel transforms."""
    converted = replace(stroke, points=[
        (
            (x * max(1, full_width - 1) - x0) / max(1, crop_width - 1),
            (y * max(1, full_height - 1) - y0) / max(1, crop_height - 1),
        )
        for x, y in stroke.points
    ])
    return converted


def strokes_for_composite_crop(
    strokes: Iterable[Stroke], width: int, height: int, auto_enabled: bool,
) -> tuple[list[Stroke], tuple[int, int, int, int]] | None:
    """Limit full-resolution compositing to the union of source and destination ROIs."""
    ordered = list(strokes)
    positive = [item for item in ordered if not item.erase and item.points]
    if not positive:
        return None
    boxes: list[tuple[int, int, int, int]] = []
    for item in positive:
        prepared = auto_optimized_stroke(item, auto_enabled)
        destination = stroke_annotation_box(prepared, width, height)
        if destination is not None:
            boxes.append(destination)
        original = replace(prepared, points=prepared.points.copy())
        reset_stroke_geometry(original)
        source_box = stroke_annotation_box(original, width, height)
        if source_box is not None:
            boxes.append(source_box)
    if not boxes:
        return None
    margin = max(
        auto_optimized_stroke(item, auto_enabled).width
        + auto_optimized_stroke(item, auto_enabled).feather * 4
        for item in positive
    ) + 8
    x0 = max(0, min(box[0] for box in boxes) - margin)
    y0 = max(0, min(box[1] for box in boxes) - margin)
    x1 = min(width, max(box[2] for box in boxes) + margin)
    y1 = min(height, max(box[3] for box in boxes) + margin)
    relevant: list[Stroke] = []
    for item in ordered:
        if not item.points:
            continue
        if item.erase:
            box = stroke_annotation_box(item, width, height)
            if box is None:
                continue
            ex0, ey0, ex1, ey1 = box
            if not (ex0 < x1 and ex1 > x0 and ey0 < y1 and ey1 > y0):
                continue
        relevant.append(stroke_for_image_crop(
            item, width, height, x0, y0, x1 - x0, y1 - y0
        ))
    return relevant, (x0, y0, x1, y1)


def stroke_annotation_box(stroke: Stroke, width: int, height: int) -> tuple[int, int, int, int] | None:
    points = transformed_stroke_points(stroke, width, height)
    if not points:
        return None
    pixels = np.asarray([
        (x * max(1, width - 1), y * max(1, height - 1)) for x, y in points
    ], dtype=np.float32)
    radius = max(
        2.0,
        (stroke.width * max(0.05, stroke.width_scale) + stroke.feather * 2.0) * 0.55,
    )
    x0 = max(0, int(np.floor(float(pixels[:, 0].min()) - radius)))
    y0 = max(0, int(np.floor(float(pixels[:, 1].min()) - radius)))
    x1 = min(width, int(np.ceil(float(pixels[:, 0].max()) + radius)) + 1)
    y1 = min(height, int(np.ceil(float(pixels[:, 1].max()) + radius)) + 1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def meteor_source_annotations(
    source_name: str, strokes: Iterable[Stroke], width: int, height: int,
    original_state: bool = False, active_mask: np.ndarray | None = None,
    active_mask_origin: tuple[int, int] = (0, 0),
) -> list[dict]:
    """Build per-meteor labels and original references for transformed objects."""
    annotations = []
    meteor_index = 0
    for stroke in strokes:
        if stroke.erase or not stroke.points:
            continue
        meteor_index += 1
        box = stroke_annotation_box(stroke, width, height)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if active_mask is not None:
            origin_x, origin_y = active_mask_origin
            mx0 = max(0, x0 - origin_x)
            my0 = max(0, y0 - origin_y)
            mx1 = min(active_mask.shape[1], x1 - origin_x)
            my1 = min(active_mask.shape[0], y1 - origin_y)
            if mx1 <= mx0 or my1 <= my0 or not np.any(
                active_mask[my0:my1, mx0:mx1] > 0.001
            ):
                continue
        transformed = stroke_is_transformed(stroke)
        original_box = None
        if transformed:
            original = replace(stroke, points=stroke.points.copy())
            reset_stroke_geometry(original)
            original_box = stroke_annotation_box(original, width, height)
        annotations.append({
            "source": source_name,
            "boxes": [box],
            "original_boxes": [original_box] if original_box is not None else [],
            "original_state": original_state or normalized_source_mode(stroke) == "original",
            "source_mode": normalized_source_mode(stroke),
            "transformed": transformed,
            "meteor_index": meteor_index,
            "transform": {
                "offset_x": float(stroke.offset_x), "offset_y": float(stroke.offset_y),
                "rotation": float(stroke.rotation),
                "length_scale": float(stroke.length_scale),
                "width_scale": float(stroke.width_scale),
            } if transformed else None,
        })
    return annotations


def scale_source_annotations(
    annotations: list[dict], scale_x: float, scale_y: float,
) -> list[dict]:
    """Scale source-label boxes for a progressively downsampled preview."""
    scaled = []
    for annotation in annotations:
        item = dict(annotation)
        for field in ("boxes", "original_boxes"):
            item[field] = [
                (
                    int(round(box[0] * scale_x)), int(round(box[1] * scale_y)),
                    int(round(box[2] * scale_x)), int(round(box[3] * scale_y)),
                )
                for box in annotation.get(field, []) if box is not None
            ]
        scaled.append(item)
    return scaled


def transformed_object_crop(
    source: np.ndarray, stroke: Stroke, fast: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
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
        patch, local, (dx1 - dx0, dy1 - dy0),
        flags=cv2.INTER_LINEAR if fast else cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_alpha = cv2.warpAffine(
        alpha, local, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_validity = cv2.warpAffine(
        np.ones(alpha.shape, dtype=np.float32), local, (dx1 - dx0, dy1 - dy0),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_alpha = np.clip(warped_alpha * float(np.clip(stroke.opacity, 0.0, 1.0)), 0.0, 1.0)
    return warped_source, warped_alpha, warped_validity, (dx0, dy0, dx1, dy1)


def transformed_mask_crop(
    source: np.ndarray, strokes: Iterable[Stroke],
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Build only the small affected mask rectangle, preserving paint order."""
    fragments: list[tuple[Stroke, np.ndarray, tuple[int, int, int, int]]] = []
    for stroke in strokes:
        if not stroke.points:
            continue
        transformed = transformed_object_crop(source, stroke, fast=True)
        if transformed is None:
            continue
        _patch, alpha, _validity, box = transformed
        fragments.append((stroke, alpha, box))
    if not fragments:
        return None
    x0 = min(box[0] for _stroke, _alpha, box in fragments)
    y0 = min(box[1] for _stroke, _alpha, box in fragments)
    x1 = max(box[2] for _stroke, _alpha, box in fragments)
    y1 = max(box[3] for _stroke, _alpha, box in fragments)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for stroke, alpha, (fx0, fy0, fx1, fy1) in fragments:
        target = mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
        if stroke.erase:
            target *= 1.0 - alpha
        else:
            np.maximum(target, alpha, out=target)
    return np.clip(mask, 0.0, 1.0), (x0, y0, x1, y1)


def auto_optimized_stroke(stroke: Stroke, enabled: bool = True) -> Stroke:
    """Return a non-destructive mask clone with a safe full-resolution feather."""
    if not enabled or not stroke.auto_blend_enabled or stroke.erase:
        return stroke
    recommended = (
        int(stroke.auto_feather) if stroke.auto_feather is not None
        else max(3, int(round(stroke.width * stroke.width_scale * 0.65)))
    )
    if stroke.feather >= recommended:
        return stroke
    optimized = replace(stroke, points=stroke.points.copy())
    optimized.feather = recommended
    return optimized


def remove_local_background_cast(
    source: np.ndarray, base: np.ndarray, alpha: np.ndarray, strength: float,
    validity: np.ndarray | None = None,
) -> np.ndarray:
    """Remove a smooth RGB sky mismatch estimated outside the meteor mask."""
    amount = float(np.clip(strength / 100.0, 0.0, 1.0))
    if amount <= 0.0:
        return source.astype(np.float32)
    src = source.astype(np.float32)
    dst = base.astype(np.float32)
    difference = src - dst
    height, width = alpha.shape
    valid = np.ones_like(alpha, dtype=bool) if validity is None else validity > 0.98
    background = (alpha < 0.06) & valid
    if int(background.sum()) < 24:
        background = (alpha < 0.20) & valid
    ys, xs = np.nonzero(background)
    if len(xs) < 12:
        return src

    samples = difference[ys, xs]
    center = np.median(samples, axis=0)
    residual_size = np.linalg.norm(samples - center, axis=1)
    cutoff = float(np.percentile(residual_size, 72.0))
    stable = residual_size <= max(cutoff, 1e-6)
    xs, ys, samples = xs[stable], ys[stable], samples[stable]
    if len(xs) < 12:
        correction = np.broadcast_to(center, difference.shape)
    else:
        nx = (xs.astype(np.float32) / max(1, width - 1) - 0.5) * 2.0
        ny = (ys.astype(np.float32) / max(1, height - 1) - 0.5) * 2.0
        design = np.column_stack((np.ones_like(nx), nx, ny))
        grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
        grid_x = (grid_x / max(1, width - 1) - 0.5) * 2.0
        grid_y = (grid_y / max(1, height - 1) - 0.5) * 2.0
        grid = np.stack((np.ones_like(grid_x), grid_x, grid_y), axis=-1)
        correction = np.empty_like(difference)
        for channel in range(3):
            coefficients, *_ = np.linalg.lstsq(design, samples[:, channel], rcond=None)
            predicted = grid @ coefficients
            low, high = np.percentile(samples[:, channel], (8.0, 92.0))
            correction[..., channel] = np.clip(predicted, low, high)
    return src - correction * amount


def analyze_meteor_blend_parameters(
    source: np.ndarray, base: np.ndarray, stroke: Stroke, strength: str = "标准",
) -> dict[str, float | int | str]:
    """Measure one meteor at full resolution and derive non-destructive blend settings."""
    profiles = {
        "保守": {"cleanup": 65.0, "feather": 0.82, "noise": 2.1, "peak_cap": 0.07},
        "标准": {"cleanup": 80.0, "feather": 1.00, "noise": 3.0, "peak_cap": 0.12},
        "强力": {"cleanup": 92.0, "feather": 1.22, "noise": 4.2, "peak_cap": 0.20},
    }
    profile_name = strength if strength in profiles else "标准"
    profile = profiles[profile_name]
    measured = replace(stroke, points=stroke.points.copy())
    measured.auto_blend_enabled = True
    # Auto-detected strokes already carry a transverse-profile feather. Respect
    # that measured value instead of expanding it again from total mask width.
    measured.auto_feather = max(3, int(round(max(3, stroke.feather) * profile["feather"])))
    transformed = transformed_object_crop(source, auto_optimized_stroke(measured, True))
    if transformed is None:
        return {
            "strength": profile_name, "black_point": 0.75,
            "cleanup": profile["cleanup"], "brightness": 100.0,
            "feather": measured.auto_feather, "snr": 0.0,
        }
    source_patch, alpha, validity, (x0, y0, x1, y1) = transformed
    base_patch = base[y0:y1, x0:x1]
    cleaned = remove_local_background_cast(
        source_patch, base_patch, alpha, profile["cleanup"], validity
    )
    positive = np.maximum(cleaned - base_patch.astype(np.float32), 0.0)
    luma = positive[..., 0] * 0.0722 + positive[..., 1] * 0.7152 + positive[..., 2] * 0.2126
    ring = (alpha < 0.04) & (validity > 0.98)
    active = (alpha > 0.10) & (validity > 0.98)
    noise = luma[ring]
    signal = luma[active]
    clip_max = 65535.0 if source.dtype == np.uint16 or base.dtype == np.uint16 else 255.0
    unit = clip_max / 255.0
    if noise.size >= 24:
        center = float(np.median(noise))
        mad = float(np.median(np.abs(noise - center)))
    else:
        center, mad = 0.0, 0.0
    peak = float(np.percentile(signal, 99.0)) if signal.size else unit
    noise_sigma = max(unit * 0.35, 1.4826 * mad)
    snr = max(0.0, (peak - center) / max(noise_sigma, 1e-6))
    proposed_black = max(unit * 0.75, center + profile["noise"] * noise_sigma)
    black = min(proposed_black, max(unit * 0.75, peak * profile["peak_cap"]))
    # Weak meteors receive gentler cleanup and a small compensation for the
    # adaptive black subtraction; strong meteors can tolerate firmer rejection.
    weak_factor = float(np.clip((snr - 3.0) / 12.0, 0.0, 1.0))
    cleanup = profile["cleanup"] - (1.0 - weak_factor) * (10.0 if profile_name != "保守" else 5.0)
    compensation = float(np.clip(100.0 + black / max(peak, unit) * 55.0, 100.0, 118.0))
    return {
        "strength": profile_name,
        "black_point": round(black / unit, 3),
        "cleanup": round(cleanup, 1),
        "brightness": round(compensation, 1),
        "feather": int(measured.auto_feather),
        "snr": round(snr, 2),
    }


def dominant_meteor_signal_gate(
    positive_signal: np.ndarray, alpha: np.ndarray, stroke: Stroke,
    cleanup_strength: float = 70.0,
) -> np.ndarray:
    """Keep the meteor while applying an automatic Lighten/Levels-style cutoff."""
    luminance = (
        positive_signal[..., 0] * 0.0722
        + positive_signal[..., 1] * 0.7152
        + positive_signal[..., 2] * 0.2126
    )
    active = alpha > 0.025
    values = luminance[active]
    if values.size < 12 or float(values.max()) <= 0.0:
        return np.zeros_like(alpha, dtype=np.float32)
    # Use sky outside the actual mask as the noise reference. This also works
    # for hard-edged masks with zero feather, where no intermediate-alpha ring
    # exists at all.
    ring = alpha < 0.025
    noise = luminance[ring]
    if noise.size < 24:
        noise = values
    median = float(np.median(noise))
    mad = float(np.median(np.abs(noise - median)))
    floor = 1.0 if float(values.max()) <= 255.0 else 64.0
    active_median = float(np.median(values))
    active_high = float(np.percentile(values, 97.0))
    has_distinct_highlight = active_high > active_median + floor * 4.0
    noise_limit = median + 3.5 * 1.4826 * mad
    threshold = max(
        floor, noise_limit,
        # A broad hand-painted mask usually contains much more sky than meteor.
        # Treat its median positive luminance as the PS Levels black point so a
        # weak RGB cast cannot survive merely because one channel is brighter.
        active_median + floor if has_distinct_highlight else floor,
        float(np.percentile(values, 72.0)) * 0.85,
    )
    seeds = ((luminance > threshold + floor * 0.01) & (alpha > 0.055)).astype(np.uint8)
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    if count <= 1:
        return np.zeros_like(alpha, dtype=np.float32)
    component_scores = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        component = labels == index
        peak = float(np.percentile(luminance[component], 90.0))
        core_overlap = float(np.mean(alpha[component]))
        cy, cx = np.nonzero(component)
        elongation = 1.0
        if len(cx) >= 5:
            coordinates = np.column_stack((cx, cy)).astype(np.float32)
            covariance = np.cov(coordinates, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if float(eigenvalues[-1]) > 1e-6:
                elongation = float(np.sqrt(
                    eigenvalues[-1] / max(float(eigenvalues[0]), 0.25)
                ))
        # Meteor signal should form a continuous elongated component. Compact
        # stars and block fragments are deliberately disadvantaged even when
        # locally bright.
        shape_weight = 0.55 + min(4.0, elongation) * 0.28
        component_scores.append((
            area * max(1.0, np.sqrt(peak)) * (0.5 + core_overlap) * shape_weight,
            index,
        ))
    if not component_scores:
        return np.zeros_like(alpha, dtype=np.float32)
    dominant_index = max(component_scores)[1]
    dominant = (labels == dominant_index).astype(np.uint8)
    radius = int(np.clip(round(max(2.0, stroke.width * stroke.width_scale) * 0.32), 2, 18))
    kernel_size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    support = cv2.dilate(dominant, kernel).astype(np.float32)
    sigma = max(0.8, radius * 0.45)
    support = cv2.GaussianBlur(support, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if float(support.max()) > 0:
        support /= float(support.max())
    # Reproduce the article's "Lighten + Levels + soft black mask" stage. RGB
    # channel-wise positive residuals alone retain weak magenta/green blocks;
    # one luminance gate removes those blocks without changing meteor colour.
    black = max(
        floor * 0.5, noise_limit,
        active_median if has_distinct_highlight else floor * 0.5,
    )
    white = max(black + floor, threshold)
    levels = np.clip((luminance - black) / (white - black), 0.0, 1.0)
    levels = levels * levels * (3.0 - 2.0 * levels)
    # Cleanup is intentionally stronger in the dark end than a linear slider.
    # At the default 70%, only 2.7% of a below-black-point residual survives
    # (the previous quadratic mapping retained 9%, visible on dark skies).
    amount = 1.0 - (1.0 - float(np.clip(cleanup_strength / 100.0, 0.0, 1.0))) ** 3
    tonal_gate = (1.0 - amount) + amount * levels
    # Alpha is applied later by the compositor. Geometry and tonal rejection
    # belong here; multiplying alpha again would dim the bright core twice.
    return np.clip(support * tonal_gate, 0.0, 1.0)


def auto_signal_needs_fallback(
    positive_signal: np.ndarray, alpha: np.ndarray, clip_max: float,
) -> bool:
    """Cheaply identify an auto-optimized mask whose useful signal collapsed."""
    luminance = (
        positive_signal[..., 0] * 0.0722
        + positive_signal[..., 1] * 0.7152
        + positive_signal[..., 2] * 0.2126
    )
    mask_area = int((alpha > 0.01).sum())
    if mask_area < 24:
        return False
    signal_floor = 1.0 if clip_max <= 255.0 else 64.0
    active = int(((luminance > signal_floor) & (alpha > 0.01)).sum())
    # Normal trails occupy several percent of a painted/feathered mask.  A
    # result below 1.5% is suspicious enough to justify the more expensive
    # conservative extraction and retention comparison.
    return active < max(24, int(round(mask_area * 0.015)))


def adjust_composite_base_exposure(
    composite: np.ndarray, base: np.ndarray, exposure_ev: float,
) -> np.ndarray:
    """Adjust only the clean base while preserving the extracted meteor signal."""
    if abs(float(exposure_ev)) < 1e-6:
        return composite.copy()
    clip_max = 65535.0 if composite.dtype == np.uint16 or base.dtype == np.uint16 else 255.0
    base_float = base.astype(np.float32)
    signal = composite.astype(np.float32) - base_float
    adjusted_base = base_float * float(2.0 ** np.clip(exposure_ev, -4.0, 4.0))
    return np.clip(adjusted_base + signal, 0.0, clip_max).astype(composite.dtype)


def compose_meteor_objects(
    source: np.ndarray,
    base: np.ndarray,
    strokes: Iterable[Stroke],
    match_exposure: bool,
    curve_enabled: bool,
    curve_shadows: float,
    curve_highlights: float,
    blend_mode: str = "natural",
    preserve_brightness: bool = True,
    meteor_brightness: float = 100.0,
    background_cleanup: float = 70.0,
    auto_optimize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    clip_max = 65535.0 if base.dtype == np.uint16 or source.dtype == np.uint16 else 255.0
    # Keep the full frame in its native dtype. Only small meteor patches become
    # float32, avoiding a 3–4× full-frame allocation for every source layer.
    result = base.copy()
    height, width = base.shape[:2]
    union = np.zeros((height, width), np.float16)
    ordered_strokes = list(strokes)
    erase_after_cache: dict[tuple[int, ...], tuple[np.ndarray, tuple[int, int, int, int]] | None] = {}
    for stroke_index, stroke in enumerate(ordered_strokes):
        if stroke.erase:
            continue
        composite_stroke = auto_optimized_stroke(stroke, auto_optimize)
        transformed = transformed_object_crop(source, composite_stroke)
        if transformed is None:
            continue
        source_patch, alpha, validity, (x0, y0, x1, y1) = transformed
        # Mask edits are chronological, like Photoshop: an eraser affects paint
        # that existed before it, while a later brush stroke can paint the area
        # back. The old implementation combined every eraser last, making it
        # impossible to supplement a mask after any erase operation.
        later_eraser_indices = tuple(
            index for index in range(stroke_index + 1, len(ordered_strokes))
            if ordered_strokes[index].erase
        )
        if later_eraser_indices:
            cache_key = later_eraser_indices
            if cache_key not in erase_after_cache:
                erase_after_cache[cache_key] = build_mask_crop(
                    [
                        replace(ordered_strokes[index], erase=False,
                                points=ordered_strokes[index].points.copy())
                        for index in later_eraser_indices
                    ],
                    width, height,
                )
            erased = erase_after_cache[cache_key]
            if erased is not None:
                erase_crop, (ex0, ey0, ex1, ey1) = erased
                ox0, oy0 = max(x0, ex0), max(y0, ey0)
                ox1, oy1 = min(x1, ex1), min(y1, ey1)
                if ox1 > ox0 and oy1 > oy0:
                    alpha[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] *= 1.0 - erase_crop[
                        oy0 - ey0:oy1 - ey0, ox0 - ex0:ox1 - ex0
                    ]
        if not np.any(alpha > 0.001):
            continue
        base_patch = result[y0:y1, x0:x1]
        raw_source = source_patch.astype(np.float32)
        object_cleanup = (
            stroke.background_cleanup_override
            if stroke.background_cleanup_override is not None
            else (
                stroke.auto_cleanup
                if auto_optimize and stroke.auto_blend_enabled and stroke.auto_cleanup is not None
                else background_cleanup
            )
        )
        cleaned_source = remove_local_background_cast(
            raw_source, base_patch, alpha, object_cleanup, validity
        )
        source_float = cleaned_source.copy()
        object_match = match_exposure if stroke.match_exposure_override is None else stroke.match_exposure_override
        if object_match:
            source_float = locally_match(source_float, base_patch, alpha)
        if curve_enabled:
            source_float = apply_meteor_curve(source_float, alpha, curve_shadows, curve_highlights, clip_max)
        # Brightness is applied only to positive meteor signal relative to the
        # destination, so the surrounding sky is not lifted with the meteor.
        object_preserve = (
            preserve_brightness
            if stroke.preserve_brightness_override is None
            else stroke.preserve_brightness_override
        )
        processed_positive = np.maximum(source_float - base_patch, 0.0)
        if object_preserve:
            original_positive = np.maximum(cleaned_source - base_patch, 0.0)
            positive = np.maximum(processed_positive, original_positive)
        else:
            positive = processed_positive
        object_brightness = (
            stroke.brightness_override
            if stroke.brightness_override is not None
            else (
                stroke.auto_brightness
                if auto_optimize and stroke.auto_blend_enabled and stroke.auto_brightness is not None
                else meteor_brightness
            )
        )
        gain = float(np.clip(object_brightness / 100.0, 0.0, 4.0))
        scaled_positive = positive * gain
        object_saturation = 100.0 if stroke.saturation_override is None else stroke.saturation_override
        saturation = float(np.clip(object_saturation / 100.0, 0.0, 3.0))
        signal_luma = (
            scaled_positive[..., 0] * 0.0722
            + scaled_positive[..., 1] * 0.7152
            + scaled_positive[..., 2] * 0.2126
        )[..., None]
        scaled_positive = np.clip(signal_luma + (scaled_positive - signal_luma) * saturation, 0.0, clip_max)
        signal_gate = dominant_meteor_signal_gate(
            scaled_positive, alpha, stroke, object_cleanup
        )
        scaled_positive *= signal_gate[..., None]
        if auto_optimize:
            # Full-resolution export reveals one-level residuals hidden by the
            # Fit-to-window display can hide one-level residuals. Estimate a per-object black point from
            # the corrected outer sky, then subtract it from luminance while
            # preserving meteor colour and strong core/tail structure.
            residual = np.maximum(cleaned_source - base_patch, 0.0)
            residual_luma = (
                residual[..., 0] * 0.0722
                + residual[..., 1] * 0.7152
                + residual[..., 2] * 0.2126
            )
            ring = (alpha < 0.04) & (validity > 0.98)
            noise = residual_luma[ring]
            floor = 0.75 if clip_max <= 255.0 else 192.0
            if (
                stroke.auto_blend_enabled and stroke.auto_black_point is not None
            ):
                black_point = float(stroke.auto_black_point) * clip_max / 255.0
            elif noise.size >= 24:
                noise_center = float(np.median(noise))
                noise_mad = float(np.median(np.abs(noise - noise_center)))
                black_point = max(floor, noise_center + 3.0 * 1.4826 * noise_mad)
            else:
                black_point = floor
            isolated_luma = (
                scaled_positive[..., 0] * 0.0722
                + scaled_positive[..., 1] * 0.7152
                + scaled_positive[..., 2] * 0.2126
            )
            # Soft subtraction is less mechanical than a hard curve cutoff:
            # pixels just above noise fade in smoothly; strong signal loses only
            # a negligible constant and keeps its original colour ratios.
            retained = np.maximum(isolated_luma - black_point, 0.0)
            signal_scale = retained / np.maximum(isolated_luma, 1e-6)
            scaled_positive *= signal_scale[..., None]
        if (
            auto_optimize and stroke.auto_blend_enabled
            and auto_signal_needs_fallback(scaled_positive, alpha, clip_max)
        ):
            # Automatic cleanup is allowed to remove sky residuals, but it must
            # never erase the meteor it was asked to protect.  This matters most
            # after a meteor has been moved: the source sky and destination sky
            # can differ enough that local matching plus a strong cleanup value
            # collapses a long trail to only a few bright pixels.  Build a second,
            # deliberately conservative extraction and use it only when the
            # optimized result has lost nearly all of that coherent signal.
            fallback_cleanup = min(float(object_cleanup), 45.0)
            fallback_cleaned = remove_local_background_cast(
                raw_source, base_patch, alpha, fallback_cleanup, validity
            )
            fallback_source = fallback_cleaned
            if curve_enabled:
                fallback_source = apply_meteor_curve(
                    fallback_source, alpha, curve_shadows, curve_highlights, clip_max
                )
            fallback_processed = np.maximum(fallback_source - base_patch, 0.0)
            if object_preserve:
                fallback_original = np.maximum(fallback_cleaned - base_patch, 0.0)
                fallback_positive = np.maximum(fallback_processed, fallback_original)
            else:
                fallback_positive = fallback_processed
            fallback_positive *= gain
            fallback_luma = (
                fallback_positive[..., 0] * 0.0722
                + fallback_positive[..., 1] * 0.7152
                + fallback_positive[..., 2] * 0.2126
            )[..., None]
            fallback_positive = np.clip(
                fallback_luma + (fallback_positive - fallback_luma) * saturation,
                0.0, clip_max,
            )
            fallback_gate = dominant_meteor_signal_gate(
                fallback_positive, alpha, stroke, fallback_cleanup
            )
            fallback_positive *= fallback_gate[..., None]
            fallback_isolated = (
                fallback_positive[..., 0] * 0.0722
                + fallback_positive[..., 1] * 0.7152
                + fallback_positive[..., 2] * 0.2126
            )
            # Keep a small noise floor even in fallback mode.  The fallback is
            # not a raw paste; it is the conservative end of the same isolated
            # meteor pipeline.
            conservative_black = 0.75 if clip_max <= 255.0 else 192.0
            fallback_retained = np.maximum(fallback_isolated - conservative_black, 0.0)
            fallback_positive *= (
                fallback_retained / np.maximum(fallback_isolated, 1e-6)
            )[..., None]

            optimized_luma = (
                scaled_positive[..., 0] * 0.0722
                + scaled_positive[..., 1] * 0.7152
                + scaled_positive[..., 2] * 0.2126
            )
            fallback_luma = (
                fallback_positive[..., 0] * 0.0722
                + fallback_positive[..., 1] * 0.7152
                + fallback_positive[..., 2] * 0.2126
            )
            signal_floor = 1.0 if clip_max <= 255.0 else 64.0
            fallback_active = (fallback_luma > signal_floor) & (alpha > 0.01)
            optimized_active = (optimized_luma > signal_floor) & (alpha > 0.01)
            fallback_count = int(fallback_active.sum())
            optimized_count = int(optimized_active.sum())
            fallback_energy = float(fallback_luma[fallback_active].sum()) if fallback_count else 0.0
            optimized_energy = float(optimized_luma[optimized_active].sum()) if optimized_count else 0.0
            severe_signal_loss = (
                fallback_count >= 24
                and (
                    optimized_count < max(12, int(round(fallback_count * 0.12)))
                    or optimized_energy < fallback_energy * 0.10
                )
            )
            if severe_signal_loss:
                scaled_positive = fallback_positive
        # Never paste the source sky itself. Only the isolated positive meteor
        # residual is allowed into the clean base, preventing stars, trees and
        # projection seams inside a broad feather from appearing in the result.
        source_float = base_patch + scaled_positive
        object_blend = blend_mode if stroke.blend_mode_override is None else stroke.blend_mode_override
        if object_blend in {"normal", "普通粘贴"}:
            candidate = source_float
        elif object_blend in {"residual", "亮度残差"}:
            positive = np.maximum(source_float - base_patch, 0.0)
            candidate = np.clip(base_patch + positive, 0.0, clip_max)
        else:
            candidate = np.maximum(base_patch, source_float)
        effective_alpha = alpha
        if object_preserve:
            signal_luminance = (
                scaled_positive[..., 0] * 0.0722
                + scaled_positive[..., 1] * 0.7152
                + scaled_positive[..., 2] * 0.2126
            )
            active_signal = signal_luminance[(alpha > 0.01) & (signal_luminance > 0.0)]
            if active_signal.size >= 5:
                low, high = (float(v) for v in np.percentile(active_signal, (55.0, 99.5)))
                if high > low + 1e-6:
                    highlight = np.clip((signal_luminance - low) / (high - low), 0.0, 1.0)
                    highlight = highlight * highlight * (3.0 - 2.0 * highlight)
                else:
                    highlight = (signal_luminance > 0.0).astype(np.float32)
                # Preserve the bright core while leaving the feathered edge and
                # faint tail smooth. This compensates for alpha and resampling loss.
                effective_alpha = alpha + (1.0 - alpha) * highlight
        a = effective_alpha[..., None]
        blended = base_patch * (1.0 - a) + candidate * a
        result[y0:y1, x0:x1] = np.clip(blended, 0, clip_max).astype(base.dtype)
        union[y0:y1, x0:x1] = np.maximum(
            union[y0:y1, x0:x1], alpha.astype(np.float16, copy=False)
        )
    return result, union


def compose_meteor_sources(
    aligned_source: np.ndarray,
    original_source: np.ndarray,
    base: np.ndarray,
    strokes: Iterable[Stroke],
    *settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite one photograph whose meteors may use different pixel sources.

    The aligned and untouched images share the same canvas dimensions, but each
    positive stroke is evaluated only against the source selected for that
    meteor. Erasers are source-local as well, matching what the user saw while
    painting in either source view.
    """
    ordered = list(strokes)
    result = base.copy()
    union = np.zeros(base.shape[:2], dtype=np.float16)
    sources = {"aligned": aligned_source, "original": original_source}
    for mode in ("aligned", "original"):
        selected = [item for item in ordered if normalized_source_mode(item) == mode]
        if not any(not item.erase and item.points for item in selected):
            continue
        result, source_mask = compose_meteor_objects(
            sources[mode], result, selected, *settings
        )
        union = np.maximum(union, source_mask)
    return result, union


class ExactPreviewViewer(tk.Toplevel):
    """Scrollable full-resolution viewer for the export-equivalent composite."""

    def __init__(
        self, parent: tk.Misc, final_image: np.ndarray, labeled_image: np.ndarray,
        initial_mode: str = "blend",
    ) -> None:
        super().__init__(parent)
        self.title("导出级精确预览 — 滚轮缩放，左键拖动")
        self.geometry("1280x820")
        self.minsize(760, 520)
        self.images = {"blend": final_image, "labeled": labeled_image}
        self.mode = tk.StringVar(value="labeled" if initial_mode == "labeled" else "blend")
        self.zoom = 1.0
        self.center_x = final_image.shape[1] / 2.0
        self.center_y = final_image.shape[0] / 2.0
        self.fit_pending = True
        self.drag_start: tuple[int, int, float, float] | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.image_item: int | None = None
        self.zoom_label = tk.StringVar()

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill="x")
        ttk.Radiobutton(
            toolbar, text="最终效果", variable=self.mode, value="blend", command=self._render
        ).pack(side="left")
        ttk.Radiobutton(
            toolbar, text="来源标注", variable=self.mode, value="labeled", command=self._render
        ).pack(side="left", padx=(8, 18))
        ttk.Button(toolbar, text="适合窗口", command=self.fit).pack(side="left")
        ttk.Button(toolbar, text="100%", command=self.actual_size).pack(side="left", padx=4)
        ttk.Button(toolbar, text="−", width=3, command=lambda: self._zoom_by(1 / 1.25)).pack(side="left")
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._zoom_by(1.25)).pack(side="left", padx=(4, 0))
        ttk.Label(toolbar, textvariable=self.zoom_label).pack(side="left", padx=12)
        ttk.Label(
            toolbar, text="鼠标滚轮缩放 · 左键拖动平移 · 双击切换 100%/适合窗口"
        ).pack(side="right")

        self.canvas = tk.Canvas(self, background="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._wheel_steps(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._wheel_steps(event, -1))
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.canvas.bind("<Double-Button-1>", self._toggle_actual)
        self.bind("<KeyPress-0>", lambda _event: self.fit())
        self.bind("<KeyPress-1>", lambda _event: self.actual_size())
        self.bind("<KeyPress-plus>", lambda _event: self._zoom_by(1.25))
        self.bind("<KeyPress-minus>", lambda _event: self._zoom_by(1 / 1.25))
        # An export-quality preview is primarily used to inspect seams, dark
        # residuals and feathering. Open at one image pixel per screen pixel;
        # "适合窗口" remains available when the user wants the whole frame.
        self.after_idle(self.actual_size)

    def _image(self) -> np.ndarray:
        return self.images[self.mode.get()]

    def _fit_scale(self) -> float:
        image = self._image()
        height, width = image.shape[:2]
        return min(
            max(1, self.canvas.winfo_width()) / max(1, width),
            max(1, self.canvas.winfo_height()) / max(1, height),
        )

    def fit(self) -> None:
        image = self._image()
        self.center_x = image.shape[1] / 2.0
        self.center_y = image.shape[0] / 2.0
        self.zoom = self._fit_scale()
        self.fit_pending = False
        self._render()

    def actual_size(self) -> None:
        self.zoom = 1.0
        self.fit_pending = False
        self._clamp_center()
        self._render()

    def _toggle_actual(self, _event=None) -> str:
        if abs(self.zoom - 1.0) < 0.02:
            self.fit()
        else:
            self.actual_size()
        return "break"

    def _view_origin(self) -> tuple[float, float]:
        return (
            self.center_x - self.canvas.winfo_width() / (2.0 * self.zoom),
            self.center_y - self.canvas.winfo_height() / (2.0 * self.zoom),
        )

    def _clamp_center(self) -> None:
        image = self._image()
        height, width = image.shape[:2]
        half_w = self.canvas.winfo_width() / (2.0 * self.zoom)
        half_h = self.canvas.winfo_height() / (2.0 * self.zoom)
        self.center_x = width / 2.0 if half_w >= width / 2.0 else float(np.clip(self.center_x, half_w, width - half_w))
        self.center_y = height / 2.0 if half_h >= height / 2.0 else float(np.clip(self.center_y, half_h, height - half_h))

    def _zoom_by(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        anchor_x, anchor_y = anchor or (canvas_w // 2, canvas_h // 2)
        origin_x, origin_y = self._view_origin()
        image_x = origin_x + anchor_x / self.zoom
        image_y = origin_y + anchor_y / self.zoom
        fit_scale = self._fit_scale()
        self.zoom = float(np.clip(self.zoom * factor, fit_scale * 0.5, 8.0))
        new_origin_x = image_x - anchor_x / self.zoom
        new_origin_y = image_y - anchor_y / self.zoom
        self.center_x = new_origin_x + canvas_w / (2.0 * self.zoom)
        self.center_y = new_origin_y + canvas_h / (2.0 * self.zoom)
        self.fit_pending = False
        self._clamp_center()
        self._render()

    def _wheel(self, event) -> str:
        steps = 1 if event.delta > 0 else -1
        return self._wheel_steps(event, steps)

    def _wheel_steps(self, event, steps: int) -> str:
        self._zoom_by(1.25 ** steps, (int(event.x), int(event.y)))
        return "break"

    def _pan_start(self, event) -> str:
        self.drag_start = (event.x, event.y, self.center_x, self.center_y)
        self.canvas.configure(cursor="fleur")
        return "break"

    def _pan_move(self, event) -> str:
        if self.drag_start is None:
            return "break"
        start_x, start_y, center_x, center_y = self.drag_start
        self.center_x = center_x - (event.x - start_x) / self.zoom
        self.center_y = center_y - (event.y - start_y) / self.zoom
        self._clamp_center()
        self._render()
        return "break"

    def _pan_end(self, _event=None) -> str:
        self.drag_start = None
        self.canvas.configure(cursor="")
        return "break"

    def _on_configure(self, _event=None) -> None:
        if self.fit_pending:
            self.fit()
        else:
            self._clamp_center()
            self._render()

    def _render(self) -> None:
        if not self.winfo_exists():
            return
        image = self._image()
        height, width = image.shape[:2]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self._clamp_center()
        origin_x, origin_y = self._view_origin()
        x0 = max(0, int(np.floor(origin_x)))
        y0 = max(0, int(np.floor(origin_y)))
        x1 = min(width, int(np.ceil(origin_x + canvas_w / self.zoom)))
        y1 = min(height, int(np.ceil(origin_y + canvas_h / self.zoom)))
        if x1 <= x0 or y1 <= y0:
            return
        crop = Image.fromarray(image[y0:y1, x0:x1])
        display_w = max(1, int(round((x1 - x0) * self.zoom)))
        display_h = max(1, int(round((y1 - y0) * self.zoom)))
        if (display_w, display_h) != crop.size:
            resample = Image.Resampling.LANCZOS if self.zoom < 1.0 else Image.Resampling.BICUBIC
            crop = crop.resize((display_w, display_h), resample)
        draw_x = int(round((x0 - origin_x) * self.zoom))
        draw_y = int(round((y0 - origin_y) * self.zoom))
        self.photo = ImageTk.PhotoImage(crop)
        if self.image_item is None or not self.canvas.type(self.image_item):
            self.canvas.delete("all")
            self.image_item = self.canvas.create_image(draw_x, draw_y, anchor="nw", image=self.photo)
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            self.canvas.coords(self.image_item, draw_x, draw_y)
        self.zoom_label.set(f"{self.zoom * 100:.0f}% · {width}×{height}")


class MeteorComposer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_NAME} — {APP_VERSION}")
        self.geometry("1280x820")
        self.minsize(1000, 680)

        self.source_dir = tk.StringVar()
        self.base_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_mode = tk.StringVar(value="combined")
        self.base_selection_summary = tk.StringVar(value="")
        self.brush_width = tk.IntVar(value=18)
        self.eraser_width = tk.IntVar(value=40)
        self.feather = tk.IntVar(value=10)
        self.match_exposure = tk.BooleanVar(value=False)
        self.default_match_exposure = tk.BooleanVar(value=False)
        self.match_exposure_policy = tk.StringVar(value="跟随全局")
        self.curve_enabled = tk.BooleanVar(value=False)
        self.curve_shadows = tk.IntVar(value=15)
        self.curve_highlights = tk.IntVar(value=25)
        self.default_preserve_brightness = tk.BooleanVar(value=True)
        self.default_meteor_brightness = tk.IntVar(value=100)
        self.meteor_brightness = tk.IntVar(value=100)
        self.brightness_override = tk.BooleanVar(value=False)
        self.default_background_cleanup = tk.IntVar(value=70)
        self.auto_optimize = tk.BooleanVar(value=True)
        self.auto_optimize_strength = tk.StringVar(value="标准")
        self.selected_auto_summary = tk.StringVar(value="自动参数：尚未分析")
        self.base_exposure_tenths = tk.IntVar(value=0)
        self.base_exposure_label = tk.StringVar(value="+0.0 EV")
        self.exact_preview_status = tk.StringVar(value="精准预览：自动")
        self.preview_quality_status = tk.StringVar(value="当前画布：原始像素")
        self.selected_object_summary = tk.StringVar(value="尚未选择流星")
        self.selected_override_enabled = tk.BooleanVar(value=False)
        self.selected_brightness = tk.IntVar(value=100)
        self.selected_cleanup = tk.IntVar(value=70)
        self.selected_saturation = tk.IntVar(value=100)
        self.selected_preserve = tk.BooleanVar(value=True)
        self.selected_match = tk.BooleanVar(value=False)
        self.selected_blend = tk.StringVar(value="自然融合")
        self.selected_feather = tk.IntVar(value=10)
        self.selected_source_mode = tk.StringVar(value="自动对齐素材")
        self.loading_selected_adjustments = False
        self.candidate_threshold = tk.IntVar(value=55)
        self.candidate_summary = tk.StringVar(value="当前图尚未分析候选")
        self.ai_model_status = tk.StringVar()
        self.autosave_status = tk.StringVar(value="自动保存：等待更改")
        self.export_tiff = tk.BooleanVar(value=False)
        self.blend_mode = tk.StringVar(value="自然融合")
        self.edit_mode = tk.StringVar(value="paint")
        # A restored/opened project should immediately show its result, not an
        # empty source canvas that requires the user to discover view button 3.
        self.view_mode = tk.StringVar(value="blend")
        self.source_state_label = tk.StringVar(value="当前素材：自动对齐图")
        self.blend_preview_label = tk.StringVar(value="3 融合预览")
        self.source_preview_label = tk.StringVar(value="4 来源标注")
        self.show_mask = tk.BooleanVar(value=True)
        self.h_mask_held = False
        self.status = tk.StringVar(value="请选择输入素材；输出位置可留空自动创建。")

        self.files: list[Path] = []
        self.selected_base_files: list[Path] = []
        self.pairs: dict[str, Path] = {}
        self.pairing_signature: str | None = None
        self.original_sources: dict[str, Path] = {}
        self.use_original_sources: set[str] = set()
        self.alignment_statuses: dict[str, str] = {}
        self.strokes: dict[str, list[Stroke]] = {}
        self.candidates: dict[str, list[Stroke]] = {}
        self.candidate_thresholds: dict[str, int] = {}
        self.adjustment_defaults = {
            "match_exposure": False, "curve_enabled": False,
            "curve_shadows": 15, "curve_highlights": 25,
            "preserve_brightness": True, "meteor_brightness": 100,
            "background_cleanup": 70,
            "auto_optimize": True,
        }
        self.image_adjustments: dict[str, dict] = {}
        self.loading_adjustments = False
        self.setting_candidate_threshold = False
        self.edit_history: dict[str, list[tuple[str, int, object]]] = {}
        self.edit_redo: dict[str, list[tuple[str, int, object]]] = {}
        self.last_edit_key: str | None = None
        self.current_path: Path | None = None
        self.preview_rgb: np.ndarray | None = None
        self.preview_source: np.ndarray | None = None
        self.preview_aligned_source: np.ndarray | None = None
        self.preview_original_source: np.ndarray | None = None
        self.preview_base: np.ndarray | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_image_item: int | None = None
        self.preview_cache: OrderedDict[tuple[str, ...], tuple] = OrderedDict()
        self.base_preview_cache: OrderedDict[tuple[str, int, int, int], np.ndarray] = OrderedDict()
        self.layer_preview_cache: OrderedDict[tuple[str, int, int, int, int], np.ndarray] = OrderedDict()
        self.preview_cache_lock = threading.Lock()
        display_budget, precision_budget, viewport_budget = preview_memory_budgets()
        self.full_display_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self.full_precision_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self.full_display_cache_bytes = 0
        self.full_precision_cache_bytes = 0
        self.full_display_cache_budget = display_budget
        self.full_precision_cache_budget = precision_budget
        self.full_cache_inflight: dict[tuple[str, str, int, int], threading.Event] = {}
        self.full_cache_pinned_paths: set[str] = set()
        self.prefetch_generation = 0
        self.viewport_cache: OrderedDict[tuple, tuple[Image.Image, int]] = OrderedDict()
        self.viewport_cache_bytes = 0
        self.viewport_cache_budget = viewport_budget
        self.preview_frame_serial = 0
        self.preview_mask_overlay: tuple[np.ndarray, tuple[int, int, int, int]] | None = None
        self.preview_request_id = 0
        self.preview_selection_after_id: str | None = None
        self.preview_display_size: tuple[int, int] | None = None
        self.canvas_zoom = 1.0
        self.canvas_fit_mode = True
        self.canvas_preserve_fit_once = False
        self.canvas_last_window_size: tuple[int, int] | None = None
        self.canvas_center_x = 0.0
        self.canvas_center_y = 0.0
        self.canvas_image_shape: tuple[int, int] | None = None
        self.canvas_pan_start: tuple[int, int, float, float] | None = None
        self.canvas_pan_with_left = False
        self.space_pan_held = False
        self.canvas_zoom_label = tk.StringVar(value="适合窗口")
        self.global_preview_rgb: np.ndarray | None = None
        self.global_labeled_preview_rgb: np.ndarray | None = None
        self.global_preview_signature: str | None = None
        self.global_preview_loading_signature: str | None = None
        self.global_preview_pending_signature: str | None = None
        self.global_preview_request_after_id: str | None = None
        self.global_preview_generation = 0
        self.last_incremental_box: tuple[int, int, int, int] | None = None
        self.realtime_label_after_id: str | None = None
        self.shared_base_loading_signature: str | None = None
        self.global_exact_after_id: str | None = None
        self.exact_preview_rgb: np.ndarray | None = None
        self.exact_labeled_preview_rgb: np.ndarray | None = None
        self.exact_preview_full_rgb: np.ndarray | None = None
        self.exact_labeled_preview_full_rgb: np.ndarray | None = None
        self.exact_preview_signature: str | None = None
        self.exact_preview_loading_signature: str | None = None
        self.exact_preview_pending_signature: str | None = None
        self.exact_preview_request_after_id: str | None = None
        self.exact_preview_open_when_ready = False
        self.exact_preview_window: ExactPreviewViewer | None = None
        self.exact_preview_generation = 0
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
        self.last_candidate_hover_time = 0.0
        self.last_candidate_hover_position: tuple[float, float] | None = None
        self.alt_previous_mode: str | None = None
        self.active_action_index = -1
        self.context_stroke_index: int | None = None
        self.context_highlight: int | None = None
        self.selected_object: tuple[str, int] | None = None
        self.object_drag_mode: str | None = None
        self.object_drag_start: tuple[float, float] | None = None
        self.object_drag_original: Stroke | None = None
        self.object_drag_live_source: np.ndarray | None = None
        self.object_drag_live_background: np.ndarray | None = None
        self.object_drag_live_frame: np.ndarray | None = None
        self.object_drag_live_box: tuple[int, int, int, int] | None = None
        self.object_drag_live_full_width = 1
        self.object_drag_live_settings: tuple | None = None
        self.object_drag_last_render = 0.0
        self.object_handle_centers: dict[str, tuple[float, float]] = {}
        self.object_overlay_items: list[int] = []
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
        self.screening_window = None
        self.last_export_path: Path | None = None

        self._build_ui()
        self._bind_shortcuts()
        self._setup_autosave()
        self.after(150, self._poll_queue)
        self.after(350, self._restore_autosave)

    def maximize_for_normal_launch(self) -> None:
        """Use the native maximized state, with a cross-platform fallback."""
        try:
            self.state("zoomed")
            return
        except tk.TclError:
            pass
        try:
            self.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text=APP_NAME, font=("TkDefaultFont", 15, "bold")).pack(side="left")
        ttk.Label(header, text="图片合成工作区").pack(side="left", padx=12)
        ttk.Button(header, text="运行日志", command=lambda: show_runtime_log(self)).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="打开视频动态工作区…", command=self.open_video_workspace).pack(side="right")
        ttk.Button(header, text="2  Siril＋PTGui星空对齐…", command=self.open_alignment_workspace).pack(side="right", padx=(0, 6))
        ttk.Button(header, text="流星批量筛选…", command=self.open_screening_workspace).pack(side="right", padx=(0, 6))
        self.paths_toggle_button = ttk.Button(header, text="收起 1 流星合成功能", command=self._toggle_paths_panel)
        self.paths_toggle_button.pack(side="right", padx=(0, 6))

        paths = ttk.LabelFrame(root, text="1  流星合成功能（源素材只读）", padding=8)
        paths.pack(fill="x")
        self.paths_panel = paths
        mode_row = ttk.Frame(paths)
        mode_row.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))
        ttk.Label(mode_row, text="最终输出方式", width=18).pack(side="left")
        ttk.Radiobutton(
            mode_row, text="拼合到同一张底图（输出一张总图）", variable=self.output_mode,
            value="combined", command=self._output_mode_changed,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row, text="分别输出（按同名底图逐张生成）", variable=self.output_mode,
            value="separate", command=self._output_mode_changed,
        ).pack(side="left", padx=(18, 0))
        self._path_row(paths, 1, "原图 TIFF 文件夹", self.source_dir, self._browse_source)
        ttk.Label(paths, text="干净底图", width=18).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(paths, textvariable=self.base_dir).grid(row=2, column=1, sticky="ew", padx=5)
        base_buttons = ttk.Frame(paths)
        base_buttons.grid(row=2, column=2, sticky="w")
        self.base_files_button = ttk.Button(base_buttons, text="选择一张…", command=self._browse_base_files)
        self.base_files_button.pack(side="left")
        self.base_folder_button = ttk.Button(base_buttons, text="选择文件夹…", command=self._browse_base_folder)
        self.base_folder_button.pack(side="left", padx=(4, 0))
        ttk.Label(paths, textvariable=self.base_selection_summary).grid(row=2, column=4, sticky="w", padx=(4, 0))
        self._path_row(paths, 3, "输出文件夹（可留空）", self.output_dir, self._browse_output)
        ttk.Button(paths, text="只读扫描", command=self.scan_inputs).grid(row=1, column=3, rowspan=3, padx=8, sticky="ns")
        paths.columnconfigure(1, weight=1)
        self._update_output_mode_ui()

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=(10, 6))

        left = ttk.Frame(body, width=310)
        body.add(left, weight=0)
        ttk.Label(left, text="TIFF 素材（单击或上下键立即加载）").pack(anchor="w")
        self.tree = ttk.Treeview(left, columns=("status",), show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="文件")
        self.tree.heading("status", text="蒙版")
        self.tree.column("#0", width=210)
        self.tree.column("status", width=70, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self._tree_selection_changed)
        source_state = ttk.Frame(left)
        source_state.pack(fill="x", pady=(4, 0))
        self.aligned_source_button = ttk.Button(
            source_state, text="自动对齐图", command=lambda: self._set_current_source_state("aligned")
        )
        self.aligned_source_button.pack(side="left", fill="x", expand=True)
        self.original_source_button = ttk.Button(
            source_state, text="原始状态图", command=lambda: self._set_current_source_state("original")
        )
        self.original_source_button.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ttk.Label(left, textvariable=self.source_state_label).pack(anchor="w", pady=(2, 0))

        center = ttk.Frame(body)
        body.add(center, weight=1)
        view_bar = ttk.Frame(center)
        view_bar.pack(fill="x", pady=(0, 2))
        ttk.Label(view_bar, text="查看：").pack(side="left")
        ttk.Radiobutton(view_bar, text="1 当前素材图", variable=self.view_mode, value="source", command=self._view_mode_changed).pack(side="left")
        ttk.Radiobutton(view_bar, text="2 干净 JPG", variable=self.view_mode, value="base", command=self._view_mode_changed).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(view_bar, textvariable=self.blend_preview_label, variable=self.view_mode, value="blend", command=self._view_mode_changed).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(view_bar, textvariable=self.source_preview_label, variable=self.view_mode, value="labeled", command=self._view_mode_changed).pack(side="left", padx=(8, 0))
        ttk.Label(view_bar, text="编辑视图：拖动画蒙版；按住 H 临时隐藏红色").pack(side="right")
        exact_bar = ttk.Frame(center)
        exact_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(exact_bar, textvariable=self.preview_quality_status).pack(side="left")
        ttk.Label(exact_bar, textvariable=self.exact_preview_status).pack(side="left", padx=(8, 0))
        ttk.Button(exact_bar, text="适合窗口", command=self._canvas_fit).pack(side="right", padx=(4, 0))
        ttk.Button(exact_bar, text="100%", command=self._canvas_actual_size).pack(side="right", padx=(4, 0))
        ttk.Button(exact_bar, text="+", width=3, command=lambda: self._canvas_zoom_by(1.25)).pack(side="right", padx=(4, 0))
        ttk.Button(exact_bar, text="−", width=3, command=lambda: self._canvas_zoom_by(1 / 1.25)).pack(side="right")
        ttk.Label(exact_bar, textvariable=self.canvas_zoom_label).pack(side="right", padx=(8, 4))
        self.canvas = tk.Canvas(center, background="#181818", cursor="none", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._canvas_configure)
        self.canvas.bind("<Motion>", self._cursor_motion)
        self.canvas.bind("<Leave>", self._cursor_leave)
        self.canvas.bind("<MouseWheel>", self._canvas_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._canvas_wheel_steps(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._canvas_wheel_steps(event, -1))
        self.canvas.bind("<ButtonPress-2>", self._canvas_pan_start_event)
        self.canvas.bind("<B2-Motion>", self._canvas_pan_move_event)
        self.canvas.bind("<ButtonRelease-2>", self._canvas_pan_end_event)
        # Put left-button editing in its own highest-priority bind tag. On some
        # Windows/Tk combinations NumLock contributes a modifier bit and a more
        # specific widget binding can otherwise swallow ButtonPress while drag
        # and release still arrive, leaving painting impossible.
        pointer_tag = f"MeteorCanvasPointer{id(self.canvas)}"
        self.canvas.bindtags((pointer_tag, *self.canvas.bindtags()))
        self.bind_class(pointer_tag, "<ButtonPress-1>", self._stroke_start)
        self.bind_class(pointer_tag, "<B1-Motion>", self._stroke_move)
        self.bind_class(pointer_tag, "<ButtonRelease-1>", self._stroke_end)
        # A release can be delivered to another widget when rendering takes long
        # enough for the pointer to leave the canvas. Catch it application-wide.
        self.bind_all("<ButtonRelease-1>", self._global_pointer_release, add=True)
        self.bind_all("<KeyPress-space>", self._space_pan_press, add=True)
        self.bind_all("<KeyRelease-space>", self._space_pan_release, add=True)
        self.canvas.bind("<Button-3>", self._show_mask_menu)
        self.canvas.bind("<Shift-Button-3>", self._delete_mask_at_event)
        self.mask_menu = tk.Menu(self, tearoff=0)
        self.mask_menu.add_command(label="删除整条蒙版", command=self._delete_context_stroke)
        self.mask_menu.add_command(label="锁定这条蒙版", command=self._toggle_context_lock)
        self.mask_menu.add_command(label="进入预览直接移动／旋转／拉伸", command=self._transform_context_stroke)
        self.object_menu = tk.Menu(self, tearoff=0)
        self.object_menu.add_command(label="删除这颗流星", command=self._delete_selected_object)
        self.object_menu.add_command(label="一键恢复原始位置／形态", command=self._reset_selected_object)
        self.object_menu.add_command(label="精确变换参数…", command=self._transform_selected_object)
        self.object_menu.add_separator()
        self.object_menu.add_command(label="使用自动对齐素材", command=lambda: self._set_selected_source_mode("aligned"))
        self.object_menu.add_command(label="使用原始素材", command=lambda: self._set_selected_source_mode("original"))

        self.control_notebook = ttk.Notebook(root)
        self.control_notebook.pack(fill="x", pady=(0, 5))

        mask_tools = ttk.Frame(self.control_notebook, padding=8)
        blend_tools = ttk.Frame(self.control_notebook, padding=8)
        selected_tools = ttk.Frame(self.control_notebook, padding=8)
        self.mask_tools_tab = mask_tools
        self.blend_tools_tab = blend_tools
        self.selected_tools_tab = selected_tools
        self.control_notebook.add(mask_tools, text="3  蒙版与候选")
        self.control_notebook.add(blend_tools, text="4  融合与底图")
        self.control_notebook.add(selected_tools, text="5  所选流星")

        ttk.Label(mask_tools, text="当前工具", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mask_tools, text="B ✎ 画笔", variable=self.edit_mode, value="paint", command=self._tool_settings_changed).grid(row=0, column=1, padx=4)
        ttk.Radiobutton(mask_tools, text="E ▱ 橡皮擦", variable=self.edit_mode, value="erase", command=self._tool_settings_changed).grid(row=0, column=2, padx=(0, 12))
        ttk.Label(mask_tools, text="画笔宽度").grid(row=0, column=3)
        ttk.Scale(mask_tools, from_=2, to=100, variable=self.brush_width, orient="horizontal", command=lambda _v: self._tool_settings_changed()).grid(row=0, column=4, sticky="ew", padx=5)
        ttk.Label(mask_tools, textvariable=self.brush_width, width=4).grid(row=0, column=5)
        ttk.Label(mask_tools, text="橡皮擦宽度").grid(row=0, column=6, padx=(12, 0))
        ttk.Scale(mask_tools, from_=2, to=200, variable=self.eraser_width, orient="horizontal", command=lambda _v: self._tool_settings_changed()).grid(row=0, column=7, sticky="ew", padx=5)
        ttk.Label(mask_tools, textvariable=self.eraser_width, width=4).grid(row=0, column=8)
        ttk.Label(mask_tools, text="羽化").grid(row=0, column=9, padx=(12, 0))
        ttk.Scale(mask_tools, from_=0, to=80, variable=self.feather, orient="horizontal", command=lambda _v: self._tool_settings_changed()).grid(row=0, column=10, sticky="ew", padx=5)
        ttk.Label(mask_tools, textvariable=self.feather, width=4).grid(row=0, column=11)

        ttk.Button(mask_tools, text="AI分析当前单张候选", command=self.detect_current_candidates).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Label(mask_tools, text="AI分数阈值").grid(row=1, column=2, pady=(7, 0))
        ttk.Scale(mask_tools, from_=1, to=100, variable=self.candidate_threshold, orient="horizontal", command=self._candidate_threshold_changed).grid(row=1, column=3, columnspan=2, sticky="ew", padx=5, pady=(7, 0))
        ttk.Label(mask_tools, textvariable=self.candidate_threshold, width=4).grid(row=1, column=5, pady=(7, 0))
        ttk.Label(mask_tools, textvariable=self.candidate_summary).grid(row=1, column=6, columnspan=3, sticky="w", padx=(12, 0), pady=(7, 0))
        ttk.Label(mask_tools, textvariable=self.ai_model_status).grid(row=1, column=9, columnspan=3, sticky="e", pady=(7, 0))

        ttk.Button(mask_tools, text="撤销 Ctrl+Z", command=self.undo_stroke).grid(row=2, column=0, pady=(7, 0), sticky="ew")
        ttk.Button(mask_tools, text="清除此图蒙版", command=self.clear_strokes).grid(row=2, column=1, pady=(7, 0), padx=3, sticky="ew")
        ttk.Button(mask_tools, text="自动检测全部", command=self.auto_detect_all).grid(row=2, column=2, pady=(7, 0), padx=3, sticky="ew")
        ttk.Button(mask_tools, text="保存项目", command=self.save_project).grid(row=2, column=3, pady=(7, 0), padx=3, sticky="ew")
        ttk.Button(mask_tools, text="载入项目", command=self.load_project).grid(row=2, column=4, pady=(7, 0), padx=3, sticky="ew")
        ttk.Label(mask_tools, text="绿色虚线=候选；移到线上点击绿色按钮选中。Alt 临时切换画笔/橡皮擦。", foreground="#555555").grid(row=2, column=5, columnspan=7, sticky="e", pady=(7, 0))
        for column in (4, 7, 10):
            mask_tools.columnconfigure(column, weight=1)

        scope_row = ttk.Frame(blend_tools)
        scope_row.pack(fill="x")
        ttk.Label(scope_row, text="作用范围：全局默认", style="Heading.TLabel").pack(side="left")
        ttk.Checkbutton(scope_row, text="局部曝光/颜色匹配", variable=self.default_match_exposure, command=self._match_exposure_default_changed).pack(side="left", padx=(8, 0))
        ttk.Checkbutton(scope_row, text="保持流星亮部", variable=self.default_preserve_brightness, command=self._brightness_default_changed).pack(side="left", padx=(8, 0))
        ttk.Label(scope_row, text="当前图曝光匹配").pack(side="left", padx=(16, 4))
        exposure_policy = ttk.Combobox(scope_row, textvariable=self.match_exposure_policy, state="readonly", width=10, values=("跟随全局", "强制启用", "强制关闭"))
        exposure_policy.pack(side="left")
        exposure_policy.bind("<<ComboboxSelected>>", self._match_exposure_policy_changed)
        ttk.Checkbutton(scope_row, text="导出16位TIFF", variable=self.export_tiff).pack(side="right")
        blend_combo = ttk.Combobox(scope_row, textvariable=self.blend_mode, state="readonly", width=12, values=("自然融合", "亮度残差", "普通粘贴"))
        blend_combo.pack(side="right", padx=(4, 12))
        blend_combo.bind("<<ComboboxSelected>>", lambda _event: self._render_preview())
        ttk.Label(scope_row, text="合成方式").pack(side="right")

        curve_row = ttk.Frame(blend_tools)
        curve_row.pack(fill="x", pady=(7, 0))
        ttk.Checkbutton(curve_row, text="当前图：启用流星曲线", variable=self.curve_enabled, command=self._render_preview).pack(side="left")
        ttk.Label(curve_row, text="暗部压低").pack(side="left", padx=(16, 4))
        ttk.Scale(curve_row, from_=0, to=100, variable=self.curve_shadows, orient="horizontal", command=lambda _v: self._render_preview()).pack(side="left", fill="x", expand=True)
        ttk.Label(curve_row, textvariable=self.curve_shadows, width=4).pack(side="left")
        ttk.Label(curve_row, text="亮部提升").pack(side="left", padx=(16, 4))
        ttk.Scale(curve_row, from_=0, to=100, variable=self.curve_highlights, orient="horizontal", command=lambda _v: self._render_preview()).pack(side="left", fill="x", expand=True)
        ttk.Label(curve_row, textvariable=self.curve_highlights, width=4).pack(side="left")

        brightness_row = ttk.Frame(blend_tools)
        brightness_row.pack(fill="x", pady=(7, 0))
        ttk.Label(brightness_row, text="全局流星亮度%").pack(side="left")
        ttk.Scale(brightness_row, from_=50, to=250, variable=self.default_meteor_brightness, orient="horizontal", command=self._brightness_default_changed).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(brightness_row, textvariable=self.default_meteor_brightness, width=4).pack(side="left")
        ttk.Checkbutton(brightness_row, text="当前图单独设置", variable=self.brightness_override, command=self._brightness_override_changed).pack(side="left", padx=(16, 4))
        self.current_brightness_scale = ttk.Scale(brightness_row, from_=50, to=250, variable=self.meteor_brightness, orient="horizontal", state="disabled", command=self._current_brightness_changed)
        self.current_brightness_scale.pack(side="left", fill="x", expand=True)
        ttk.Label(brightness_row, textvariable=self.meteor_brightness, width=4).pack(side="left")
        ttk.Label(brightness_row, text="全局背景净化").pack(side="left", padx=(16, 4))
        ttk.Scale(brightness_row, from_=0, to=100, variable=self.default_background_cleanup, orient="horizontal", command=self._background_cleanup_default_changed).pack(side="left", fill="x", expand=True)
        ttk.Label(brightness_row, textvariable=self.default_background_cleanup, width=4).pack(side="left")

        base_row = ttk.Frame(blend_tools)
        base_row.pack(fill="x", pady=(7, 0))
        ttk.Label(base_row, text="底图曝光（只改变底图）").pack(side="left")
        ttk.Scale(base_row, from_=-20, to=20, variable=self.base_exposure_tenths, orient="horizontal", command=self._base_exposure_changed).pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(base_row, textvariable=self.base_exposure_label, width=8).pack(side="left")
        ttk.Button(base_row, text="重置底图曝光", command=self._reset_base_exposure).pack(side="left", padx=(8, 0))

        optimize_row = ttk.Frame(blend_tools)
        optimize_row.pack(fill="x", pady=(7, 0))
        ttk.Checkbutton(optimize_row, text="自动优化每颗流星", variable=self.auto_optimize, command=self._auto_optimize_changed).pack(side="left")
        ttk.Combobox(optimize_row, textvariable=self.auto_optimize_strength, state="readonly", width=7, values=("保守", "标准", "强力")).pack(side="left", padx=5)
        ttk.Button(optimize_row, text="自动优化当前流星", command=self.auto_optimize_selected).pack(side="left", padx=3)
        ttk.Button(optimize_row, text="自动优化全部流星", command=self.auto_optimize_all).pack(side="left", padx=3)
        ttk.Label(optimize_row, text="按原尺寸逐颗分析，不修改手绘蒙版", foreground="#555555").pack(side="left", padx=(12, 0))

        ttk.Label(selected_tools, textvariable=self.selected_object_summary, width=20).grid(row=0, column=0, sticky="w")
        independent = ttk.Checkbutton(
            selected_tools, text="单颗独立设置", variable=self.selected_override_enabled,
            command=self._selected_override_changed,
        )
        independent.grid(row=0, column=1, sticky="w", padx=(4, 10))
        self.selected_object_controls = [independent]
        self.reset_selected_button = ttk.Button(
            selected_tools, text="↶ 一键恢复原位", command=self._reset_selected_object,
            state="disabled",
        )
        self.reset_selected_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0), padx=(0, 8))
        source_mode = ttk.Combobox(
            selected_tools, textvariable=self.selected_source_mode, state="readonly", width=13,
            values=("自动对齐素材", "原始素材"),
        )
        source_mode.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0), padx=(0, 8))
        source_mode.bind("<<ComboboxSelected>>", self._selected_source_mode_changed)

        def selected_scale(column: int, label: str, variable, start: int, end: int):
            ttk.Label(selected_tools, text=label).grid(row=0, column=column, sticky="e")
            scale = ttk.Scale(
                selected_tools, from_=start, to=end, variable=variable, orient="horizontal",
                state="disabled", length=80, command=self._selected_adjustment_changed,
            )
            scale.grid(row=0, column=column + 1, sticky="ew", padx=4)
            value = ttk.Label(selected_tools, textvariable=variable, width=4)
            value.grid(row=0, column=column + 2)
            self.selected_object_controls.extend((scale, value))

        selected_scale(2, "亮度%", self.selected_brightness, 50, 250)
        selected_scale(5, "背景净化", self.selected_cleanup, 0, 100)
        selected_scale(8, "饱和度%", self.selected_saturation, 0, 200)
        preserve = ttk.Checkbutton(
            selected_tools, text="保持亮部", variable=self.selected_preserve,
            state="disabled", command=self._selected_adjustment_changed,
        )
        preserve.grid(row=1, column=2, columnspan=2, sticky="w", pady=(5, 0))
        match = ttk.Checkbutton(
            selected_tools, text="局部曝光/颜色匹配", variable=self.selected_match,
            state="disabled", command=self._selected_adjustment_changed,
        )
        match.grid(row=1, column=4, columnspan=3, sticky="w", pady=(5, 0))
        ttk.Label(selected_tools, text="混合方式").grid(row=1, column=7, sticky="e", pady=(5, 0))
        selected_blend = ttk.Combobox(
            selected_tools, textvariable=self.selected_blend, state="disabled", width=11,
            values=("自然融合", "亮度残差", "普通粘贴"),
        )
        selected_blend.grid(row=1, column=8, sticky="w", padx=4, pady=(5, 0))
        selected_blend.bind("<<ComboboxSelected>>", self._selected_adjustment_changed)
        ttk.Label(selected_tools, text="羽化px").grid(row=1, column=9, sticky="e", pady=(5, 0))
        selected_feather = ttk.Scale(
            selected_tools, from_=0, to=120, variable=self.selected_feather,
            orient="horizontal", state="disabled", command=self._selected_adjustment_changed,
        )
        selected_feather.grid(row=1, column=10, sticky="ew", padx=4, pady=(5, 0))
        selected_feather_value = ttk.Label(selected_tools, textvariable=self.selected_feather, width=4)
        selected_feather_value.grid(row=1, column=11, pady=(5, 0))
        self.selected_object_controls.extend((preserve, match, selected_blend, selected_feather, selected_feather_value))
        ttk.Label(selected_tools, textvariable=self.selected_auto_summary).grid(
            row=2, column=0, columnspan=7, sticky="w", pady=(6, 0)
        )
        ttk.Button(
            selected_tools, text="恢复自动值", command=self.restore_selected_auto
        ).grid(row=2, column=7, columnspan=2, sticky="ew", padx=3, pady=(6, 0))
        ttk.Button(
            selected_tools, text="恢复原始融合", command=self.restore_selected_original_blend
        ).grid(row=2, column=9, columnspan=3, sticky="ew", padx=3, pady=(6, 0))
        for column in (3, 6, 9, 10):
            selected_tools.columnconfigure(column, weight=1)

        bottom = ttk.Frame(root)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
        ttk.Label(bottom, textvariable=self.autosave_status).pack(side="left", padx=8)
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=220)
        self.progress.pack(side="left", padx=8)
        ttk.Button(bottom, text="导出合成结果", command=self.export).pack(side="right")
        ttk.Button(bottom, text="打开导出文件夹", command=self._open_output_folder).pack(side="right", padx=(0, 6))
        ttk.Button(bottom, text="快捷键 F1", command=self.show_shortcuts).pack(side="right", padx=(0, 6))

        # Reserve the bottom controls before allowing the canvas to consume the
        # remaining height. Packing the expanding body first can push later
        # controls completely outside an 820px window on Windows display scales.
        body.pack_forget()
        self.control_notebook.pack_forget()
        bottom.pack_forget()
        bottom.pack(side="bottom", fill="x")
        self.control_notebook.pack(side="bottom", fill="x", pady=(0, 5))
        body.pack(fill="both", expand=True, pady=(10, 6))

    def _toggle_paths_panel(self) -> None:
        self._set_paths_panel_visible(not bool(self.paths_panel.winfo_manager()))

    def _set_paths_panel_visible(self, visible: bool) -> None:
        if not visible and self.paths_panel.winfo_manager():
            self.paths_panel.pack_forget()
            self.paths_toggle_button.configure(text="展开 1 流星合成功能")
        elif visible and not self.paths_panel.winfo_manager():
            self.paths_panel.pack(fill="x", after=self.paths_toggle_button.master)
            self.paths_toggle_button.configure(text="收起 1 流星合成功能")
        self.after_idle(self._render_preview)

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
        self._activate_child_workspace(self.video_window, "video_window")

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
        self._activate_child_workspace(self.alignment_window, "alignment_window")

    def open_screening_workspace(self) -> None:
        if self.screening_window is not None:
            try:
                if self.screening_window.winfo_exists():
                    self.screening_window.deiconify()
                    self.screening_window.lift()
                    return
            except tk.TclError:
                pass
        self.screening_window = open_screening_workspace(self, self._load_screening_export)
        self._activate_child_workspace(self.screening_window, "screening_window")

    def _load_screening_export(self, exported_folder: Path) -> None:
        """Carry the concrete timestamped screening result into compositing."""
        folder = Path(exported_folder)
        if not folder.is_dir():
            return
        self.source_dir.set(str(folder))
        source_files = [path for path in folder.iterdir() if path.is_file()]
        tiff_count = sum(path.suffix.lower() in TIFF_SUFFIXES for path in source_files)
        raw_count = sum(path.suffix.lower() in {".arw", ".nef", ".nrw", ".cr2", ".cr3", ".crw"} for path in source_files)
        self._set_paths_panel_visible(True)
        if tiff_count:
            self.status.set(
                f"已自动填入筛选导出目录：{folder}（{tiff_count} 张 TIFF）；"
                "选择干净底图后即可扫描；输出位置可留空并自动创建"
            )
        elif raw_count:
            self.status.set(
                f"已自动填入筛选导出目录：{folder}（{raw_count} 张 RAW）；"
                "RAW 需先运行 Siril＋PTGui 星空对齐，生成 TIFF 后再合成"
            )
        else:
            self.status.set(f"已自动填入筛选导出目录：{folder}")
        self._schedule_autosave()

    def _activate_child_workspace(self, window: tk.Toplevel, attribute: str) -> None:
        """Present a tool workspace as a replacement for the main workspace."""
        def restore_main(event=None) -> None:
            if event is not None and event.widget is not window:
                return
            if getattr(self, attribute, None) is window:
                setattr(self, attribute, None)
            try:
                if self.winfo_exists():
                    self.deiconify()
                    self.lift()
                    self.focus_force()
            except tk.TclError:
                pass

        window.bind("<Destroy>", restore_main, add=True)
        try:
            if self.state() == "zoomed":
                window.state("zoomed")
        except tk.TclError:
            pass
        self.withdraw()
        window.deiconify()
        window.lift()
        window.focus_force()

    def _load_alignment_result(self, result) -> None:
        loaded_items = [(Path(item.output_layer), item.status) for item in result.items if item.output_layer]
        exported = [path for path, _status in loaded_items]
        if not exported or not result.base_layer:
            messagebox.showwarning(APP_NAME, "对齐任务没有可回载的图层")
            return
        base = Path(result.base_layer)
        self.source_dir.set(str(exported[0].parent))
        self.base_dir.set(str(base))
        self.selected_base_files = [base]
        self.output_mode.set("combined")
        self._update_output_mode_ui()
        final_output = Path(result.project_dir) / "final_composite"
        final_output.mkdir(parents=True, exist_ok=True)
        self.output_dir.set(str(final_output))
        self.files = exported
        self.pairs = {str(path): base for path in exported}
        self.pairing_signature = self._base_selection_signature()
        self.original_sources = {
            str(Path(item.output_layer)): Path(item.source)
            for item in result.items
            if item.output_layer and Path(item.output_layer) != Path(item.source)
        }
        self.use_original_sources.clear()
        self.alignment_statuses = {
            str(Path(item.output_layer)): item.status for item in result.items if item.output_layer
        }
        MeteorComposer._restrict_project_to_active_keys(self, self.pairs)
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
            "<KeyPress-4>": lambda: self._set_view_mode("labeled"),
            "<KeyPress-Left>": lambda: self._arrow_action(-1, 0, 1),
            "<KeyPress-Right>": lambda: self._arrow_action(1, 0, 1),
            "<KeyPress-Up>": lambda: self._arrow_action(0, -1, 1),
            "<KeyPress-Down>": lambda: self._arrow_action(0, 1, 1),
            "<Shift-KeyPress-Left>": lambda: self._arrow_action(-1, 0, 10),
            "<Shift-KeyPress-Right>": lambda: self._arrow_action(1, 0, 10),
            "<Shift-KeyPress-Up>": lambda: self._arrow_action(0, -1, 10),
            "<Shift-KeyPress-Down>": lambda: self._arrow_action(0, 1, 10),
            "<KeyPress-plus>": lambda: self._canvas_zoom_by(1.25),
            "<KeyPress-equal>": lambda: self._canvas_zoom_by(1.25),
            "<KeyPress-minus>": lambda: self._canvas_zoom_by(1 / 1.25),
        }
        for sequence, callback in plain.items():
            self.bind_all(sequence, lambda event, fn=callback: self._run_plain_shortcut(event, fn))
        self.bind_all("<Escape>", lambda _e: self._cancel_active_stroke())
        self.bind_all("<Delete>", self._delete_selected_shortcut)
        self.bind_all("<BackSpace>", self._delete_selected_shortcut)
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
                f"<{modifier}-0>": self._canvas_fit,
                f"<{modifier}-1>": self._canvas_actual_size,
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
        if mode not in {"blend", "labeled"}:
            self._clear_object_selection()
        self.view_mode.set(mode)
        self._view_mode_changed()

    def _view_mode_changed(self) -> None:
        mode = self.view_mode.get()
        if mode not in {"blend", "labeled"}:
            self._clear_object_selection()
        if hasattr(self, "control_notebook"):
            target = self.mask_tools_tab if mode == "source" else self.blend_tools_tab
            if mode in {"blend", "labeled"} and self.selected_object is not None:
                target = self.selected_tools_tab
            self.control_notebook.select(target)
        if mode in {"blend", "labeled"} and self._uses_shared_base() and self.preview_base is None:
            self._request_shared_base_preview()
            return
        self._render_preview()

    def _arrow_action(self, dx: int, dy: int, amount: int) -> None:
        if self.view_mode.get() in {"blend", "labeled"} and self.selected_object:
            stroke = self._selected_stroke()
            if stroke is None:
                return
            before = replace(stroke, points=stroke.points.copy())
            stroke.offset_x += dx * amount
            stroke.offset_y += dy * amount
            incremental = self._incremental_selected_object_image(before)
            self._record_object_transform(before)
            if incremental is not None:
                self._commit_incremental_global_preview(incremental, validate=False)
            else:
                self._invalidate_global_preview()
                self._render_preview()
            self.status.set(f"已移动流星 {dx * amount:+d}, {dy * amount:+d} px")
            return
        if dx:
            self._select_relative(dx)

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

最终融合／来源标注
单击流星：选中对象；拖动对象内部：移动
拖动端点／侧边／四角手柄：拉伸或缩放；拖动圆形手柄：旋转
选中后可在“所选流星独立设置”中单独调整亮度、背景净化、饱和度和融合参数
Delete/Backspace：删除所选流星；方向键微移，Shift+方向键移动 10 px
右键所选流星：删除、重置或输入精确变换参数

单张候选
点击“AI分析当前单张候选”，再拖动 AI 分数阈值；阈值越低，加入的候选越多
鼠标靠近候选轨迹：弹出“＋选中”按钮，点击后直接加入并锁定
红色蒙版及候选分数默认显示；按住 H 可临时隐藏

查看与文件
1：原图 TIFF    2：干净 JPG    3：最终融合    4：来源标注
鼠标滚轮或 +/-：以光标为中心缩放；中键拖动或按住空格+左键拖动：平移
Ctrl/Command+0：适合窗口    Ctrl/Command+1：100% 原图像素
拼合到同一张底图：第 3/4 模式汇总全部流星；分别输出：只预览当前图片对
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

    @staticmethod
    def _base_filetypes():
        return [("图片", "*.jpg *.jpeg *.png *.tif *.tiff"), ("所有文件", "*.*")]

    def _browse_base_files(self) -> None:
        if self.output_mode.get() == "combined":
            value = filedialog.askopenfilename(title="选择唯一的干净底图", filetypes=self._base_filetypes())
            values = [value] if value else []
        else:
            values = list(filedialog.askopenfilenames(
                title="选择一张或多张同名底图", filetypes=self._base_filetypes()
            ))
        if not values:
            return
        self.selected_base_files = [Path(value) for value in values]
        if len(values) == 1:
            self.base_dir.set(values[0])
            self.base_selection_summary.set("已选 1 张")
        else:
            parents = {str(Path(value).parent) for value in values}
            self.base_dir.set(next(iter(parents)) if len(parents) == 1 else f"已选择 {len(values)} 张底图")
            self.base_selection_summary.set(f"已选 {len(values)} 张")
        self._update_blend_preview_label()
        self._base_selection_changed()

    def _browse_base_folder(self) -> None:
        if self.output_mode.get() == "combined":
            messagebox.showinfo(APP_NAME, "“拼合到同一张底图”只能选择一张底图照片")
            return
        value = filedialog.askdirectory(title="选择包含同名底图的文件夹")
        if value:
            self.selected_base_files = []
            self.base_dir.set(value)
            self.base_selection_summary.set("按文件夹同名匹配")
            self._base_selection_changed()

    def _update_output_mode_ui(self) -> None:
        combined = self.output_mode.get() == "combined"
        if hasattr(self, "base_files_button"):
            self.base_files_button.configure(text="选择一张…" if combined else "选择图片…")
            self.base_folder_button.configure(state="disabled" if combined else "normal")
        self._update_blend_preview_label()

    def _output_mode_changed(self) -> None:
        combined = self.output_mode.get() == "combined"
        current = Path(self.base_dir.get()).expanduser() if self.base_dir.get().strip() else None
        if combined:
            if len(self.selected_base_files) == 1:
                self.base_dir.set(str(self.selected_base_files[0]))
            elif current is None or not current.is_file():
                self.selected_base_files = []
                self.base_dir.set("")
                self.base_selection_summary.set("请选择唯一底图")
        self._invalidate_base_dependent_state()
        self._update_output_mode_ui()
        self._schedule_autosave()
        self.status.set(
            "输出方式：全部流星拼合为一张总图；请选择一张底图"
            if combined else "输出方式：逐张输出；请选择一张/多张同名底图，或底图文件夹"
        )

    def _browse_output(self) -> None:
        value = filedialog.askdirectory(title="选择输出文件夹（也可留空自动创建）")
        if value:
            self.output_dir.set(value)

    def _open_output_folder(self, path: str | Path | None = None) -> None:
        target = path or self.last_export_path or self.output_dir.get().strip()
        try:
            open_folder(target)
        except Exception as exc:
            show_copyable_error(APP_NAME, str(exc), parent=self)

    def _uses_shared_base(self) -> bool:
        return self.output_mode.get() == "combined"

    def _update_blend_preview_label(self) -> None:
        shared = self._uses_shared_base()
        self.blend_preview_label.set("3 总融合预览" if shared else "3 当前图融合预览")
        self.source_preview_label.set("4 总图来源标注" if shared else "4 当前图来源标注")

    def _base_selection_signature(self) -> str:
        """Identify the clean-base selection represented by the active pair table."""
        return json.dumps({
            "output_mode": self.output_mode.get(),
            "base_dir": self.base_dir.get().strip(),
            "base_files": [str(path) for path in self.selected_base_files],
        }, ensure_ascii=False, sort_keys=True)

    def _invalidate_base_dependent_state(self) -> None:
        """Prevent previews and exports from retaining a previously selected base."""
        self.pairs = {}
        self.pairing_signature = None
        self.preview_base = None
        self.shared_base_loading_signature = None
        self.global_preview_rgb = None
        self.global_labeled_preview_rgb = None
        self.global_preview_signature = None
        self.global_preview_pending_signature = None
        self.global_preview_generation = getattr(self, "global_preview_generation", 0) + 1
        self.exact_preview_rgb = None
        self.exact_labeled_preview_rgb = None
        self.exact_preview_full_rgb = None
        self.exact_labeled_preview_full_rgb = None
        self.exact_preview_signature = None
        self.exact_preview_generation = getattr(self, "exact_preview_generation", 0) + 1
        self.exact_preview_pending_signature = None
        if getattr(self, "exact_preview_request_after_id", None) is not None:
            try:
                self.after_cancel(self.exact_preview_request_after_id)
            except tk.TclError:
                pass
            self.exact_preview_request_after_id = None
        self.exact_preview_status.set("精准预览：底图已变化，将自动更新")
        cache = getattr(self, "preview_cache", None)
        if cache is not None:
            with self.preview_cache_lock:
                cache.clear()
                self.base_preview_cache.clear()
                self.layer_preview_cache.clear()
                self.full_cache_pinned_paths.clear()
                self.prefetch_generation += 1
        if hasattr(self, "viewport_cache"):
            self.viewport_cache.clear()
            self.viewport_cache_bytes = 0
        if self.global_preview_request_after_id is not None:
            try:
                self.after_cancel(self.global_preview_request_after_id)
            except tk.TclError:
                pass
            self.global_preview_request_after_id = None
        if self.exact_preview_window is not None:
            try:
                self.exact_preview_window.destroy()
            except tk.TclError:
                pass
            self.exact_preview_window = None

    def _base_selection_changed(self) -> None:
        self._invalidate_base_dependent_state()
        self._schedule_autosave()
        source = Path(self.source_dir.get()).expanduser()
        if source.is_dir():
            self.scan_inputs(reload_current=True)
        else:
            self.status.set("已选择新底图；选择原图文件夹后会自动建立配对和输出目录")

    def scan_inputs(self, reload_current: bool = True) -> bool:
        previous_current = self.current_path
        try:
            self._clear_object_selection()
            source = Path(self.source_dir.get()).expanduser()
            base_text = self.base_dir.get().strip()
            base_input = Path(base_text).expanduser() if base_text else None
            combined = self.output_mode.get() == "combined"
            if not source.is_dir():
                raise ValueError("请选择有效的 TIFF 文件夹")
            output_text = self.output_dir.get().strip()
            if not output_text:
                output = source / "MeteorStudio_Output"
                self.output_dir.set(str(output))
            else:
                output = Path(output_text).expanduser()
            selected = [path for path in self.selected_base_files if path.is_file()]
            selected_parents = {path.parent.resolve() for path in selected}
            use_selected = bool(selected) and (
                (len(selected) == 1 and base_input == selected[0])
                or (len(selected_parents) == 1 and base_input is not None
                    and base_input.resolve() == next(iter(selected_parents)))
                or base_text.startswith("已选择 ")
            )
            if combined:
                if base_input is None or not base_input.is_file():
                    raise ValueError("“拼合到同一张底图”模式必须选择一张干净底图照片")
                base_files = [base_input]
                self.selected_base_files = [base_input]
            elif use_selected:
                base_files = selected
            elif base_input is not None and base_input.is_dir():
                base_files = sorted(
                    path for path in base_input.iterdir()
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".png"}
                )
                self.selected_base_files = []
            elif base_input is not None and base_input.is_file():
                base_files = [base_input]
                self.selected_base_files = [base_input]
            else:
                raise ValueError("“分别输出”模式请选择一张/多张底图，或包含同名底图的文件夹")
            if not base_files:
                raise ValueError("没有找到可用的干净底图")
            source_files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES)
            if not source_files:
                raise ValueError("原图文件夹中没有 TIFF 文件")
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
            candidates = [
                (path, base_files[0] if combined else base_by_stem.get(path.stem.casefold()))
                for path in source_files
            ]
            missing.extend(path.name for path, base in candidates if base is None)
            candidates = [(path, base) for path, base in candidates if base is not None]
            # Metadata inspection is independent for every pair.  Use several
            # workers and avoid reopening a shared 500 MB base TIFF once per row.
            unique_paths = list(dict.fromkeys(
                [path for pair in candidates for path in pair if path is not None]
            ))
            infos = {}
            if unique_paths:
                info_workers = min(8, max(1, (os.cpu_count() or 4)), len(unique_paths))
                with ThreadPoolExecutor(max_workers=info_workers, thread_name_prefix="meteor-scan") as pool:
                    infos = dict(zip(unique_paths, pool.map(image_info, unique_paths)))
            for path, base in candidates:
                if base is None:
                    continue
                sw, sh, _sdepth, _schannels = infos[path]
                bw, bh, _bdepth, _bchannels = infos[base]
                if (sw, sh) == (bw, bh):
                    valid.append(path)
                    pairs[str(path)] = base
                else:
                    mismatched.append(f"{path.name}: TIFF {sw}×{sh} / JPG {bw}×{bh}")
            self.files = valid
            self.pairs = pairs
            MeteorComposer._discover_alignment_sources(self, source, valid)
            MeteorComposer._restrict_project_to_active_keys(self, pairs)
            self.pairing_signature = self._base_selection_signature()
            self._update_blend_preview_label()
            self.tree.delete(*self.tree.get_children())
            for index, path in enumerate(valid):
                count = len(self.strokes.get(str(path), []))
                prefix = "[原始状态] " if str(path) in self.use_original_sources else ""
                self.tree.insert("", "end", iid=str(index), text=prefix + path.name, values=(count or "—",))
            base_label = "单张共享底图" if combined else f"同名底图 {len(base_files)} 张"
            message = f"TIFF {len(source_files)} 张，{base_label}；成功配对 {len(valid)} 对"
            if missing:
                message += f"；缺少同名 JPG {len(missing)} 张"
            if mismatched:
                message += f"；尺寸不符 {len(mismatched)} 张，已跳过"
            self.status.set(message)
            if valid:
                self._set_paths_panel_visible(False)
                if (
                    combined and getattr(self, "view_mode", None) is not None
                    and self.view_mode.get() in {"blend", "labeled"}
                ):
                    self.after_idle(self._request_shared_base_preview)
            if previous_current in valid:
                selected_index = valid.index(previous_current)
                self.tree.selection_set(str(selected_index))
                self.tree.see(str(selected_index))
                if reload_current:
                    self.load_selected()
            elif previous_current is not None:
                self.current_path = None
                self.preview_source = None
                self.preview_base = None
            elif valid:
                # Always establish a concrete current image.  Combined preview
                # can start from the shared base immediately, while selecting
                # the first row also enables the full-resolution exact worker
                # and prevents the initial black canvas.
                self.tree.selection_set("0")
                self.tree.see("0")
                if reload_current:
                    self.load_selected()
            if hasattr(self, "_schedule_autosave"):
                self._schedule_autosave()
            return True
        except Exception as exc:
            self.pairs = {}
            self.pairing_signature = None
            show_copyable_error(APP_NAME, str(exc), parent=self)
            return False

    def _restrict_project_to_active_keys(self, keys: Iterable[str]) -> None:
        """Make one project contain state for one concrete source batch only."""
        active = {str(key) for key in keys}
        for name in (
            "strokes", "candidates", "candidate_thresholds", "image_adjustments",
            "original_sources", "alignment_statuses", "edit_history", "edit_redo",
            "shift_anchors",
        ):
            values = getattr(self, name, {})
            setattr(self, name, {key: value for key, value in values.items() if key in active})
        self.use_original_sources.intersection_update(active)
        selected = getattr(self, "selected_object", None)
        if selected is not None and selected[0] not in active:
            self._clear_object_selection()

    def _discover_alignment_sources(self, source_dir: Path, valid: list[Path]) -> None:
        """Recover original-image links when an aligned folder is opened directly."""
        manifest_path = source_dir.parent / "alignment_manifest.json"
        if not manifest_path.is_file():
            return
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            by_output = {
                str(Path(item["output_layer"]).resolve()): item
                for item in data.get("items", []) if item.get("output_layer") and item.get("source")
            }
            active_keys = {str(path) for path in valid}
            # Keep mappings belonging to other saved projects, but refresh every
            # entry in the currently scanned folder from its own manifest.
            for path in valid:
                item = by_output.get(str(path.resolve()))
                if item is None:
                    continue
                original = Path(item["source"])
                if original.is_file() and original.resolve() != path.resolve():
                    self.original_sources[str(path)] = original
                    self.alignment_statuses[str(path)] = str(item.get("status", ""))
            self.use_original_sources.intersection_update(
                set(self.original_sources) | (set(self.use_original_sources) - active_keys)
            )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            # A hand-edited/older manifest must never prevent ordinary scanning.
            return

    def _tree_selection_changed(self, _event=None) -> None:
        """Load a row after a short debounce so held arrow keys stay responsive."""
        if self.preview_selection_after_id is not None:
            try:
                self.after_cancel(self.preview_selection_after_id)
            except tk.TclError:
                pass
        self.preview_selection_after_id = self.after(55, self.load_selected)

    def load_selected(self) -> None:
        self.preview_selection_after_id = None
        selection = self.tree.selection()
        if not selection:
            return
        path = self.files[int(selection[0])]
        image_path = self._effective_source_path(path)
        self.full_cache_pinned_paths = {
            str(path), str(image_path), str(self.original_sources.get(str(path), path)),
            str(self.pairs[str(path)]),
        }
        mode = "查看原始图" if str(path) in self.use_original_sources else "查看自动对齐图"
        self.preview_request_id += 1
        request_id = self.preview_request_id
        self.status.set(f"正在加载 {path.name}（{mode}）…")
        self._run_worker(
            self._load_preview_worker, path, image_path, path,
            self.original_sources.get(str(path), path), self.pairs[str(path)],
            self.pairing_signature, request_id,
        )

    def _effective_source_path(self, path: Path) -> Path:
        key = str(path)
        if key in self.use_original_sources:
            return self.original_sources.get(key, path)
        return path

    def _current_source_mode(self, path: Path | None = None) -> str:
        target = path or self.current_path
        return "original" if target is not None and str(target) in self.use_original_sources else "aligned"

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
            mode = "自动对齐图"
        else:
            self.use_original_sources.add(key)
            mode = "原始图"
        status = self.alignment_statuses.get(key, "")
        review = "[需复查] " if "需复查" in status else ""
        self.tree.item(str(index), text=f"[查看{mode}] {review}{path.name}")
        self._schedule_autosave()
        self.load_selected()

    def _set_current_source_state(self, mode: str) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        path = self.files[int(selection[0])]
        key = str(path)
        if mode == "original":
            original = self.original_sources.get(key)
            if original is None or not original.is_file():
                self.status.set("未找到这张对齐图对应的原始状态图")
                return
            self.use_original_sources.add(key)
        else:
            self.use_original_sources.discard(key)
        self._update_source_state_ui(path)
        self._schedule_autosave()
        self.load_selected()

    def _update_source_state_ui(self, path: Path | None) -> None:
        if path is None:
            return
        key = str(path)
        original_available = key in self.original_sources and self.original_sources[key].is_file()
        original_active = key in self.use_original_sources and original_available
        self.source_state_label.set(
            "当前素材：原始状态图（未对齐）" if original_active else "当前素材：自动对齐图"
        )
        if hasattr(self, "original_source_button"):
            self.original_source_button.configure(state="normal" if original_available else "disabled")
            self.aligned_source_button.configure(state="normal")

    def _current_adjustment_values(self) -> dict:
        return {
            "match_exposure": bool(self.match_exposure.get()),
            "curve_enabled": bool(self.curve_enabled.get()),
            "curve_shadows": int(self.curve_shadows.get()),
            "curve_highlights": int(self.curve_highlights.get()),
            "preserve_brightness": bool(self.default_preserve_brightness.get()),
            "meteor_brightness": int(self.meteor_brightness.get()),
            "background_cleanup": int(self.default_background_cleanup.get()),
        }

    def _brightness_default_changed(self, *_args) -> None:
        if self.loading_adjustments:
            return
        preserve = bool(self.default_preserve_brightness.get())
        brightness = int(round(self.default_meteor_brightness.get()))
        if (
            self.adjustment_defaults.get("preserve_brightness") == preserve
            and self.adjustment_defaults.get("meteor_brightness") == brightness
        ):
            return
        self.adjustment_defaults["preserve_brightness"] = preserve
        self.adjustment_defaults["meteor_brightness"] = brightness
        if self.current_path:
            self._load_image_adjustments(str(self.current_path))
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()
        self.status.set(
            f"全局流星亮部保持已{'开启' if self.default_preserve_brightness.get() else '关闭'}；"
            f"默认亮度 {self.default_meteor_brightness.get()}%"
        )

    def _background_cleanup_default_changed(self, *_args) -> None:
        if self.loading_adjustments:
            return
        cleanup = int(round(self.default_background_cleanup.get()))
        if self.adjustment_defaults.get("background_cleanup") == cleanup:
            return
        self.adjustment_defaults["background_cleanup"] = cleanup
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()
        self.status.set(f"全局背景净化强度：{self.default_background_cleanup.get()}")

    def _auto_optimize_changed(self, *_args) -> None:
        if self.loading_adjustments:
            return
        enabled = bool(self.auto_optimize.get())
        self.adjustment_defaults["auto_optimize"] = enabled
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()
        self.status.set(f"逐流星自动融合优化已{'启用' if enabled else '关闭'}")

    def _auto_optimize_references(self, references: list[tuple[str, int]]) -> None:
        tasks = []
        for key, index in references:
            values = self.strokes.get(key, [])
            if 0 <= index < len(values) and not values[index].erase and values[index].points:
                tasks.append((key, index, values[index].points.copy()))
        if not tasks:
            messagebox.showwarning(APP_NAME, "没有可自动优化的流星")
            return
        self.auto_optimize.set(True)
        self.adjustment_defaults["auto_optimize"] = True
        strength = self.auto_optimize_strength.get()
        self.progress["value"] = 0
        self.status.set(f"正在按原始分辨率逐颗分析融合参数（{strength}）…")
        self._run_worker(self._auto_optimize_worker, tasks, strength)

    def auto_optimize_selected(self) -> None:
        if self.selected_object is None or self._selected_stroke() is None:
            messagebox.showwarning(APP_NAME, "请先在最终融合或来源标注视图中选中一颗流星")
            return
        self._auto_optimize_references([self.selected_object])

    def auto_optimize_all(self) -> None:
        references = [
            (key, index)
            for key, values in self.strokes.items()
            for index, stroke in enumerate(values)
            if key in self.pairs and not stroke.erase and stroke.points
        ]
        self._auto_optimize_references(references)

    def _auto_optimize_worker(
        self, tasks: list[tuple[str, int, list[tuple[float, float]]]], strength: str,
    ):
        results = []
        grouped: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {}
        for key, index, points in tasks:
            grouped.setdefault(key, []).append((index, points))
        completed = 0
        for key, entries in grouped.items():
            source_path = Path(key)
            base_path = self.pairs.get(key)
            if base_path is None:
                continue
            base = self._cached_full_image(base_path, True)
            source_cache: dict[str, np.ndarray] = {}
            for index, expected_points in entries:
                values = self.strokes.get(key, [])
                if not (0 <= index < len(values)):
                    continue
                snapshot = replace(values[index], points=expected_points.copy())
                mode = normalized_source_mode(snapshot)
                if mode not in source_cache:
                    selected_path = (
                        self.original_sources.get(key, source_path)
                        if mode == "original" else source_path
                    )
                    source_cache[mode] = self._cached_full_image(selected_path, True)
                source = source_cache[mode]
                if source.shape[:2] != base.shape[:2]:
                    continue
                parameters = analyze_meteor_blend_parameters(source, base, snapshot, strength)
                results.append((key, index, expected_points, parameters))
                completed += 1
                self.work_queue.put((
                    "progress", completed / max(1, len(tasks)) * 95,
                    f"逐流星融合分析 {completed}/{len(tasks)}：{source_path.name}",
                ))
        return "blend_optimized", results, strength, len(tasks)

    def restore_selected_auto(self) -> None:
        stroke = self._selected_stroke()
        if stroke is None:
            return
        if stroke.auto_black_point is None:
            self.auto_optimize_selected()
            return
        before = replace(stroke, points=stroke.points.copy())
        stroke.auto_blend_enabled = True
        stroke.brightness_override = None
        stroke.background_cleanup_override = None
        if self.selected_object:
            self._sync_matching_candidate(self.selected_object[0], stroke)
        incremental = self._incremental_parameter_change_image(before)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
        else:
            self._invalidate_global_preview()
            self._render_preview()
        self._load_selected_object_adjustments()
        self._schedule_autosave()
        self.status.set("已恢复这颗流星的自动融合参数")

    def restore_selected_original_blend(self) -> None:
        stroke = self._selected_stroke()
        if stroke is None:
            return
        before = replace(stroke, points=stroke.points.copy())
        stroke.auto_blend_enabled = False
        if self.selected_object:
            self._sync_matching_candidate(self.selected_object[0], stroke)
        incremental = self._incremental_parameter_change_image(before)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
        else:
            self._invalidate_global_preview()
            self._render_preview()
        self._load_selected_object_adjustments()
        self._schedule_autosave()
        self.status.set("已关闭这颗流星的自动优化，恢复原始蒙版融合")

    def _base_exposure_changed(self, *_args) -> None:
        value = int(round(self.base_exposure_tenths.get()))
        if self.base_exposure_tenths.get() != value:
            self.base_exposure_tenths.set(value)
        ev = value / 10.0
        self.base_exposure_label.set(f"{ev:+.1f} EV")
        if self.loading_adjustments:
            return
        self._schedule_autosave()
        self._render_preview()
        self.status.set(f"底图曝光 {ev:+.1f} EV；流星层亮度保持不变")

    def _reset_base_exposure(self) -> None:
        self.base_exposure_tenths.set(0)
        self._base_exposure_changed()

    def _brightness_override_changed(self, *_args) -> None:
        if self.loading_adjustments or not self.current_path:
            return
        key = str(self.current_path)
        adjustment = self.image_adjustments.setdefault(key, {})
        if self.brightness_override.get():
            adjustment["meteor_brightness"] = int(round(self.meteor_brightness.get()))
        else:
            adjustment.pop("meteor_brightness", None)
        if not adjustment:
            self.image_adjustments.pop(key, None)
        self._load_image_adjustments(key)
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()
        self.status.set(
            f"当前图流星亮度：{'单独设置' if self.brightness_override.get() else '跟随全局'}"
        )

    def _current_brightness_changed(self, *_args) -> None:
        if self.loading_adjustments or not self.current_path or not self.brightness_override.get():
            return
        key = str(self.current_path)
        brightness = int(round(self.meteor_brightness.get()))
        if self.image_adjustments.get(key, {}).get("meteor_brightness") == brightness:
            return
        self.image_adjustments.setdefault(key, {})["meteor_brightness"] = brightness
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()

    def _match_exposure_default_changed(self, *_args) -> None:
        if self.loading_adjustments:
            return
        enabled = bool(self.default_match_exposure.get())
        self.adjustment_defaults["match_exposure"] = enabled
        if self.current_path:
            self._load_image_adjustments(str(self.current_path))
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()
        self.status.set(
            f"全局默认局部曝光匹配已{'启用' if enabled else '关闭'}；单张强制设置保持不变"
        )

    def _match_exposure_policy_changed(self, *_args) -> None:
        if self.loading_adjustments or not self.current_path:
            return
        key = str(self.current_path)
        adjustment = self.image_adjustments.setdefault(key, {})
        policy = self.match_exposure_policy.get()
        if policy == "跟随全局":
            adjustment.pop("match_exposure", None)
        else:
            adjustment["match_exposure"] = policy == "强制启用"
        if not adjustment:
            self.image_adjustments.pop(key, None)
        self._load_image_adjustments(key)
        self._invalidate_global_preview()
        self._schedule_autosave()
        self._render_preview()
        self.status.set(f"当前图局部曝光匹配：{policy}")

    def _current_adjustment_changed(self, *_args) -> None:
        if self.loading_adjustments or not self.current_path:
            return
        key = str(self.current_path)
        adjustment = self.image_adjustments.setdefault(key, {})
        values = self._current_adjustment_values()
        for name in ("curve_enabled", "curve_shadows", "curve_highlights"):
            adjustment[name] = values[name]
        self._invalidate_global_preview()

    def _load_image_adjustments(self, key: str) -> None:
        values = {**self.adjustment_defaults, **self.image_adjustments.get(key, {})}
        explicit_match = self.image_adjustments.get(key, {}).get("match_exposure", None)
        explicit_brightness = self.image_adjustments.get(key, {}).get("meteor_brightness", None)
        self.loading_adjustments = True
        try:
            self.match_exposure.set(bool(values["match_exposure"]))
            self.match_exposure_policy.set(
                "跟随全局" if explicit_match is None
                else ("强制启用" if bool(explicit_match) else "强制关闭")
            )
            self.curve_enabled.set(bool(values["curve_enabled"]))
            self.curve_shadows.set(int(values["curve_shadows"]))
            self.curve_highlights.set(int(values["curve_highlights"]))
            self.default_preserve_brightness.set(bool(self.adjustment_defaults["preserve_brightness"]))
            self.default_meteor_brightness.set(int(self.adjustment_defaults["meteor_brightness"]))
            self.default_background_cleanup.set(int(self.adjustment_defaults["background_cleanup"]))
            self.brightness_override.set(explicit_brightness is not None)
            self.meteor_brightness.set(int(values["meteor_brightness"]))
            self.current_brightness_scale.configure(
                state="normal" if explicit_brightness is not None else "disabled"
            )
        finally:
            self.loading_adjustments = False

    def auto_detect_all(self) -> None:
        if not self.files:
            messagebox.showwarning(APP_NAME, "请先执行只读扫描")
            return
        if any(self.strokes.get(str(path), []) for path in self.files):
            if not messagebox.askyesno(APP_NAME, "AI 自动检测会替换未锁定蒙版；锁定蒙版继续保留。继续吗？"):
                return
        self.progress["value"] = 0
        self.status.set("正在用内置 AI 自动检测并排序流星候选…")
        read_paths = {str(path): self._effective_source_path(path) for path in self.files}
        source_modes = {str(path): self._current_source_mode(path) for path in self.files}
        self._run_worker(
            self._auto_detect_worker, self.files.copy(), self.pairs.copy(), read_paths,
            source_modes,
        )

    def detect_current_candidates(self) -> None:
        if not self.current_path or str(self.current_path) not in self.pairs:
            messagebox.showwarning(APP_NAME, "请先扫描并加载一张图片")
            return
        self.status.set(f"正在用内置 AI 分析当前单张候选：{self.current_path.name}…")
        self.progress["value"] = 10
        self._run_worker(
            self._candidate_worker, self.current_path, self._effective_source_path(self.current_path),
            self.pairs[str(self.current_path)], self._current_source_mode(self.current_path),
        )

    def _candidate_worker(
        self, path: Path, image_path: Path, base_path: Path, source_mode: str,
    ):
        base_preview, _ = detection_preview(self._cached_full_image(base_path, False))
        source_preview, source_scale = detection_preview(self._cached_full_image(image_path, False))
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
            full_width, full_feather = estimate_trail_mask_geometry(
                source_preview, base_preview, start, end, source_scale
            )
            start, end, core_length = expand_trail_segment(start, end, width, height)
            points = [
                (start[0] / max(1, width - 1), start[1] / max(1, height - 1)),
                (end[0] / max(1, width - 1), end[1] / max(1, height - 1)),
            ]
            candidates.append(Stroke(
                points, full_width, full_feather, False, False, int(score),
                source_mode=source_mode,
            ))
        candidates.sort(key=lambda item: item.auto_score or 0, reverse=True)
        return "candidates", path, candidates, planes, source_mode

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
                kept.auto_score == stroke.auto_score and kept.points == stroke.points
                and normalized_source_mode(kept) == normalized_source_mode(stroke)
                for kept in retained
            )
        ]
        # Never reorder manual paint/eraser history. Its order is semantic:
        # paint -> erase -> later paint must keep the later restoration. Automatic
        # candidates go first so existing erasers can still correct them.
        self.strokes[key] = selected + retained
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

    def _auto_detect_worker(
        self, files: list[Path], pairs: dict[str, Path], read_paths: dict[str, Path],
        source_modes: dict[str, str],
    ):
        def analyze(path: Path) -> tuple[str, list[Stroke], int]:
            base_path = pairs[str(path)]
            base_preview, _ = detection_preview(self._cached_full_image(base_path, False))
            source_preview, source_scale = detection_preview(
                self._cached_full_image(read_paths.get(str(path), path), False)
            )
            if source_preview.shape != base_preview.shape:
                source_preview = cv2.resize(source_preview, (base_preview.shape[1], base_preview.shape[0]), interpolation=cv2.INTER_AREA)
            trails, planes = detect_trails(source_preview, base_preview, ranked=True)
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
                full_width, full_feather = estimate_trail_mask_geometry(
                    source_preview, base_preview, start, end, source_scale
                )
                start, end, core_length = expand_trail_segment(start, end, width, height)
                points = [(start[0] / max(1, width - 1), start[1] / max(1, height - 1)),
                          (end[0] / max(1, width - 1), end[1] / max(1, height - 1))]
                strokes.append(Stroke(
                    points, width=full_width, feather=full_feather, auto_score=int(score),
                    source_mode=source_modes.get(str(path), "aligned"),
                ))
            return str(path), strokes, planes

        found: dict[str, list[Stroke]] = {}
        plane_count = 0
        workers = min(max(1, (os.cpu_count() or 4) // 3), 4, max(1, len(files)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="meteor-detect") as pool:
            results = pool.map(analyze, files)
            for index, (key, strokes, planes) in enumerate(results, start=1):
                plane_count += planes
                path = Path(key)
                if strokes:
                    found[key] = strokes
                self.work_queue.put((
                    "progress", index / len(files) * 100,
                    f"并行自动检测 {index}/{len(files)}：{path.name}",
                ))
        return "autodetected", found, plane_count, source_modes

    @staticmethod
    def _file_identity(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def _cached_full_image(self, path: Path, precision: bool = False) -> np.ndarray:
        """Decode once, share between workers, and evict by bytes rather than file count."""
        if not hasattr(self, "full_display_cache") or not hasattr(self, "preview_cache_lock"):
            decoded16 = to_uint16(read_image(path))
            return decoded16 if precision else np.right_shift(decoded16, 8).astype(np.uint8)
        identity = MeteorComposer._file_identity(path)
        cache = self.full_precision_cache if precision else self.full_display_cache
        inflight_key = (("16" if precision else "8"), *identity)
        with self.preview_cache_lock:
            cached = cache.get(identity)
            if cached is not None:
                cache.move_to_end(identity)
                return cached
            event = self.full_cache_inflight.get(inflight_key)
            owner = event is None
            if owner:
                event = threading.Event()
                self.full_cache_inflight[inflight_key] = event
        if not owner:
            event.wait()
            with self.preview_cache_lock:
                cached = cache.get(identity)
                if cached is not None:
                    cache.move_to_end(identity)
                    return cached
            return self._cached_full_image(path, precision)
        try:
            decoded16 = to_uint16(read_image(path))
            decoded = decoded16 if precision else np.right_shift(decoded16, 8).astype(np.uint8)
            with self.preview_cache_lock:
                cache[identity] = decoded
                cache.move_to_end(identity)
                byte_name = "full_precision_cache_bytes" if precision else "full_display_cache_bytes"
                budget = self.full_precision_cache_budget if precision else self.full_display_cache_budget
                setattr(self, byte_name, getattr(self, byte_name) + int(decoded.nbytes))
                # Never evict the currently displayed source/base. Prefetch is
                # opportunistic and gives way to the active working set.
                protected_paths = self.full_cache_pinned_paths
                while getattr(self, byte_name) > budget and len(cache) > 1:
                    victim = next(
                        (key for key in cache if key[0] not in protected_paths), None
                    )
                    if victim is None:
                        break
                    removed = cache.pop(victim)
                    setattr(self, byte_name, getattr(self, byte_name) - int(removed.nbytes))
            return decoded
        finally:
            with self.preview_cache_lock:
                finished = self.full_cache_inflight.pop(inflight_key, None)
                if finished is not None:
                    finished.set()

    def _cached_display_with_precision(self, path: Path) -> np.ndarray:
        """Decode the active image once and retain both 16-bit and display forms."""
        identity = MeteorComposer._file_identity(path)
        with self.preview_cache_lock:
            cached = self.full_display_cache.get(identity)
            if cached is not None:
                self.full_display_cache.move_to_end(identity)
                # The display form may have been prefetched without precision.
                precision_cached = identity in self.full_precision_cache
            else:
                precision_cached = False
        if cached is not None and precision_cached:
            return cached
        precise = MeteorComposer._cached_full_image(self, path, True)
        if cached is not None:
            return cached
        display = np.right_shift(precise, 8).astype(np.uint8)
        with self.preview_cache_lock:
            existing = self.full_display_cache.get(identity)
            if existing is not None:
                self.full_display_cache.move_to_end(identity)
                return existing
            self.full_display_cache[identity] = display
            self.full_display_cache.move_to_end(identity)
            self.full_display_cache_bytes += int(display.nbytes)
            while (
                self.full_display_cache_bytes > self.full_display_cache_budget
                and len(self.full_display_cache) > 1
            ):
                victim = next(
                    (key for key in self.full_display_cache
                     if key[0] not in self.full_cache_pinned_paths), None
                )
                if victim is None:
                    break
                removed = self.full_display_cache.pop(victim)
                self.full_display_cache_bytes -= int(removed.nbytes)
        return display

    def _schedule_neighbor_prefetch(self, current: Path) -> None:
        if current not in self.files:
            return
        self.prefetch_generation += 1
        generation = self.prefetch_generation
        index = self.files.index(current)
        nearby = []
        for distance in (1, 2):
            for neighbor_index in (index + distance, index - distance):
                if 0 <= neighbor_index < len(self.files):
                    nearby.append(self.files[neighbor_index])

        def prefetch() -> None:
            jobs: dict[Path, bool] = {}
            for position, source_path in enumerate(nearby):
                key = str(source_path)
                warm_precision = position < 2
                for path in (
                    source_path,
                    self.original_sources.get(key, source_path),
                    self.pairs.get(key, source_path),
                ):
                    if path.is_file():
                        jobs[path] = jobs.get(path, False) or warm_precision
            work = list(jobs.items())
            workers = min(3, len(work))
            if not workers:
                return
            def load(job):
                path, warm_precision = job
                return (
                    self._cached_display_with_precision(path)
                    if warm_precision else self._cached_full_image(path, False)
                )
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="meteor-prefetch") as pool:
                for _image in pool.map(load, work):
                    if generation != self.prefetch_generation:
                        break

        threading.Thread(target=prefetch, daemon=True).start()

    def _load_preview_worker(
        self, path: Path, image_path: Path, aligned_path: Path,
        original_path: Path, base_path: Path,
        pairing_signature: str | None = None,
        request_id: int = 0,
    ):
        cache_key = tuple(str(value) for target in (
            image_path, aligned_path, original_path, base_path,
        ) for value in self._file_identity(target))
        with self.preview_cache_lock:
            cached = self.preview_cache.get(cache_key)
            if cached is not None:
                self.preview_cache.move_to_end(cache_key)
                return ("preview", request_id, path, *cached, base_path, pairing_signature)

        unique = list(dict.fromkeys((image_path, aligned_path, original_path, base_path)))
        workers = min(4, len(unique))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="meteor-preview") as pool:
            decoded = dict(zip(unique, pool.map(self._cached_display_with_precision, unique)))
        source_preview = decoded[image_path]
        height, width = source_preview.shape[:2]
        aligned_preview = decoded[aligned_path]
        original_preview = decoded[original_path]
        base_preview = decoded[base_path]
        if (
            aligned_preview.shape[:2] != (height, width)
            or original_preview.shape[:2] != (height, width)
        ):
            raise ValueError(f"自动对齐图与原始图尺寸不一致：{path.name}")
        if base_preview.shape[:2] != (height, width):
            raise ValueError(f"尺寸不一致：{path.name} / {base_path.name}")
        preview_size = (width, height)
        with self.preview_cache_lock:
            for layer_path, layer_preview in (
                (aligned_path, aligned_preview), (original_path, original_preview),
            ):
                layer_stat = layer_path.stat()
                layer_key = (
                    str(layer_path), layer_stat.st_mtime_ns, layer_stat.st_size,
                    preview_size[0], preview_size[1],
                )
                self.layer_preview_cache[layer_key] = layer_preview
                self.layer_preview_cache.move_to_end(layer_key)
            while len(self.layer_preview_cache) > 48:
                self.layer_preview_cache.popitem(last=False)
        payload = (
            source_preview, aligned_preview, original_preview, base_preview, (width, height),
        )
        with self.preview_cache_lock:
            self.preview_cache[cache_key] = payload
            self.preview_cache.move_to_end(cache_key)
            while len(self.preview_cache) > 5:
                self.preview_cache.popitem(last=False)
        return ("preview", request_id, path, *payload, base_path, pairing_signature)

    def _cached_layer_preview(
        self, path: Path, width: int, height: int,
    ) -> np.ndarray | None:
        cache = getattr(self, "layer_preview_cache", None)
        lock = getattr(self, "preview_cache_lock", None)
        if cache is None or lock is None:
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size, width, height)
        with lock:
            image = cache.get(key)
            if image is not None:
                cache.move_to_end(key)
                return image
            full_key = (str(path), stat.st_mtime_ns, stat.st_size)
            full_cache = getattr(self, "full_display_cache", {})
            image = full_cache.get(full_key)
            if image is not None and image.shape[:2] == (height, width):
                full_cache.move_to_end(full_key)
                return image
            return None

    def _request_shared_base_preview(self) -> None:
        """Prepare a shared-base composite without requiring a selected source image."""
        if not self._uses_shared_base() or self.preview_base is not None or not self.pairs:
            return
        signature = self.pairing_signature
        if self.shared_base_loading_signature == signature:
            return
        base_path = next(iter(self.pairs.values()))
        self.shared_base_loading_signature = signature
        self.status.set("正在准备总融合预览…")
        self._run_worker(self._load_shared_base_preview_worker, base_path, signature)

    def _load_shared_base_preview_worker(
        self, base_path: Path, pairing_signature: str | None,
    ):
        base_preview = self._cached_full_image(base_path, False)
        height, width = base_preview.shape[:2]
        return "shared_base_preview", base_preview, (width, height), base_path, pairing_signature

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
                item.brightness_override, item.background_cleanup_override,
                item.saturation_override, item.preserve_brightness_override,
                item.match_exposure_override, item.blend_mode_override,
                item.auto_blend_enabled, item.auto_strength, item.auto_black_point,
                item.auto_cleanup, item.auto_brightness,
                (None if item.auto_feather is None else max(1, int(round(item.auto_feather * scale)))),
                source_mode=item.source_mode,
            ))
        live_stroke = self.live_erase_stroke
        if live_stroke is not None:
            scaled.append(self._preview_clone(live_stroke, self.current_dims[0]))
        current_mode = self._current_source_mode()
        visible = [item for item in scaled if normalized_source_mode(item) == current_mode]
        _composed, mask = compose_meteor_objects(
            self.preview_source, self.preview_base, visible,
            bool(self.match_exposure.get()), bool(self.curve_enabled.get()),
            float(self.curve_shadows.get()), float(self.curve_highlights.get()),
            self.blend_mode.get(), bool(self.default_preserve_brightness.get()),
            float(self.meteor_brightness.get()), float(self.default_background_cleanup.get()),
            bool(self.adjustment_defaults.get("auto_optimize", True)),
        )
        return mask

    def _global_preview_state_signature(self) -> str:
        selected = {
            key: [asdict(stroke) for stroke in values]
            for key, values in self.strokes.items() if key in self.pairs and values
        }
        payload = {
            "base": self.base_dir.get(),
            "pairs": {key: str(value) for key, value in sorted(self.pairs.items())},
            "blend_mode": self.blend_mode.get(),
            "strokes": selected,
            "adjustments": {
                key: value for key, value in self.image_adjustments.items() if key in self.pairs
            },
            "defaults": self.adjustment_defaults,
            "shape": tuple(self.preview_base.shape) if self.preview_base is not None else (),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def _exact_preview_state_signature(self) -> str:
        return json.dumps({
            "composite": self._global_preview_state_signature(),
            "output_mode": self.output_mode.get(),
            "current": str(self.current_path) if self.current_path else None,
            "base_exposure_ev": self.base_exposure_tenths.get() / 10.0,
        }, sort_keys=True, ensure_ascii=False)

    def _schedule_automatic_exact_preview(self) -> None:
        if self.preview_base is None or self.current_path is None:
            return
        # Let the fast full-size composite publish visible meteors first. Running
        # both complete pipelines at once doubled disk/memory pressure and could
        # leave the user staring at the base while two workers competed. The
        # global-preview completion event calls render again and starts the exact
        # 16-bit pass immediately afterwards.
        global_signature = self._global_preview_state_signature()
        if self.global_preview_signature != global_signature:
            return
        signature = self._exact_preview_state_signature()
        if signature == self.exact_preview_signature:
            return
        self.exact_preview_pending_signature = signature
        if self.exact_preview_request_after_id is not None:
            try:
                self.after_cancel(self.exact_preview_request_after_id)
            except tk.TclError:
                pass
        self.exact_preview_request_after_id = self.after(320, self._start_automatic_exact_preview)

    def _start_automatic_exact_preview(self) -> None:
        self.exact_preview_request_after_id = None
        signature = self.exact_preview_pending_signature
        if signature is None or signature != self._exact_preview_state_signature():
            return
        if self.exact_preview_loading_signature is not None:
            return
        self.exact_preview_pending_signature = None
        self._begin_exact_preview(signature, False)

    def request_exact_preview(self) -> None:
        """Compatibility entry point; normal operation starts this automatically."""
        if self.preview_base is None or self.current_path is None:
            messagebox.showwarning(APP_NAME, "请先加载一张图片")
            return
        if self.exact_preview_loading_signature is not None:
            self.status.set("原始像素精准预览正在后台更新，请稍候")
            return
        signature = self._exact_preview_state_signature()
        self._begin_exact_preview(signature, False)

    def _begin_exact_preview(self, signature: str, open_when_ready: bool = False) -> None:
        marked = {
            Path(key): [replace(item, points=item.points.copy()) for item in values]
            for key, values in self.strokes.items()
            if key in self.pairs and values
        }
        self.exact_preview_loading_signature = signature
        self.exact_preview_generation = getattr(self, "exact_preview_generation", 0) + 1
        generation = self.exact_preview_generation
        self.exact_preview_open_when_ready = open_when_ready
        self.exact_preview_status.set("精准预览：后台更新中…")
        self.progress["value"] = 0
        self.status.set("正在按原始分辨率和 16 位导出链路更新当前画布…")
        full_height, full_width = self.preview_base.shape[:2]
        quick_scale = min(1.0, 2400.0 / max(1, full_width, full_height))
        quick_shape = (
            max(1, int(round(full_height * quick_scale))),
            max(1, int(round(full_width * quick_scale))),
        )
        self._run_worker(
            self._exact_preview_worker, signature, marked, self.pairs.copy(),
            {key: value.copy() for key, value in self.image_adjustments.items()},
            self.adjustment_defaults.copy(), self.blend_mode.get(),
            {str(path): self.original_sources.get(str(path), path) for path in marked},
            self.output_mode.get(), Path(self.current_path),
            self.base_exposure_tenths.get() / 10.0, quick_shape, generation,
        )

    def open_exact_preview(self) -> None:
        if self.exact_preview_full_rgb is None or self.exact_labeled_preview_full_rgb is None:
            messagebox.showinfo(APP_NAME, "尚未生成可放大的导出级预览，请先点击“生成并打开”。")
            return
        if self.exact_preview_signature != self._exact_preview_state_signature():
            messagebox.showwarning(APP_NAME, "蒙版或融合参数已经变化，请重新生成导出级精确预览。")
            return
        if self.exact_preview_window is not None:
            try:
                if self.exact_preview_window.winfo_exists():
                    self.exact_preview_window.deiconify()
                    self.exact_preview_window.lift()
                    self.exact_preview_window.focus_force()
                    return
            except tk.TclError:
                pass
        viewer = ExactPreviewViewer(
            self, self.exact_preview_full_rgb, self.exact_labeled_preview_full_rgb,
            self.view_mode.get(),
        )
        self.exact_preview_window = viewer

        def clear_window(event=None) -> None:
            if event is None or event.widget is viewer:
                self.exact_preview_window = None

        viewer.bind("<Destroy>", clear_window, add=True)
        viewer.lift()
        viewer.focus_force()

    def _exact_preview_worker(
        self, signature: str, marked: dict[Path, list[Stroke]], pairs: dict[str, Path],
        adjustments: dict[str, dict], adjustment_defaults: dict, blend_mode: str,
        original_paths: dict[str, Path], output_mode: str, current_path: Path,
        base_exposure_ev: float, preview_shape: tuple[int, int], generation: int,
    ):
        selected = list(marked.items()) if output_mode == "combined" else [
            (current_path, marked.get(current_path, []))
        ]
        if output_mode == "combined" and not selected:
            selected = [(current_path, [])]
        if not selected or str(selected[0][0]) not in pairs:
            raise ValueError("找不到当前图片对应的干净底图")
        first_base_path = pairs[str(selected[0][0])]
        clean_base = self._cached_full_image(first_base_path, True)
        result = clean_base.copy()
        height, width = result.shape[:2]
        annotations: list[dict] = []
        included = 0
        last_partial = 0.0
        preview_h, preview_w = preview_shape
        clean_preview16 = cv2.resize(clean_base, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

        def load_selected(entry):
            source_path, strokes = entry
            if not strokes:
                return None
            original_path = original_paths.get(str(source_path), source_path)
            unique_sources = list(dict.fromkeys((source_path, original_path)))
            with ThreadPoolExecutor(
                max_workers=min(2, len(unique_sources)), thread_name_prefix="meteor-exact-pair"
            ) as pool:
                decoded = dict(zip(
                    unique_sources,
                    pool.map(lambda path: self._cached_full_image(path, True), unique_sources),
                ))
            return decoded[source_path], decoded[original_path]

        pipeline_workers = 2 if len(selected) > 1 else 1
        for index, ((source_path, strokes), decoded_pair) in enumerate(
            ordered_prefetch(selected, load_selected, pipeline_workers), start=1
        ):
            if generation != self.exact_preview_generation:
                return "exact_preview_cancelled", signature
            if output_mode != "combined":
                base_path = pairs.get(str(source_path))
                if base_path is None:
                    raise ValueError(f"找不到同名底图配对：{source_path.name}")
                clean_base = self._cached_full_image(base_path, True)
                result = clean_base.copy()
                height, width = result.shape[:2]
                clean_preview16 = cv2.resize(
                    clean_base, (preview_w, preview_h), interpolation=cv2.INTER_AREA
                )
            if not strokes:
                continue
            original_path = original_paths.get(str(source_path), source_path)
            aligned_source, original_source = decoded_pair
            if aligned_source.shape[:2] != (height, width) or original_source.shape[:2] != (height, width):
                raise ValueError(f"尺寸不一致：{source_path.name}")
            adjustment = {**adjustment_defaults, **adjustments.get(str(source_path), {})}
            crop_spec = strokes_for_composite_crop(
                strokes, width, height, bool(adjustment.get("auto_optimize", True))
            )
            if crop_spec is None:
                continue
            cropped_strokes, (x0, y0, x1, y1) = crop_spec
            composed_crop, mask = compose_meteor_sources(
                aligned_source[y0:y1, x0:x1], original_source[y0:y1, x0:x1],
                result[y0:y1, x0:x1], cropped_strokes,
                bool(adjustment["match_exposure"]), bool(adjustment["curve_enabled"]),
                float(adjustment["curve_shadows"]), float(adjustment["curve_highlights"]),
                blend_mode, bool(adjustment.get("preserve_brightness", True)),
                float(adjustment.get("meteor_brightness", 100)),
                float(adjustment.get("background_cleanup", 70)),
                bool(adjustment.get("auto_optimize", True)),
            )
            if np.any(mask > 0.001):
                result[y0:y1, x0:x1] = composed_crop
                included += 1
                annotations.extend(meteor_source_annotations(
                    source_path.stem, strokes, width, height,
                    False, None,
                ))
            now = time.monotonic()
            if index == 1 or index == len(selected) or now - last_partial >= 0.65:
                partial16 = cv2.resize(
                    result, (preview_w, preview_h), interpolation=cv2.INTER_AREA
                )
                partial16 = adjust_composite_base_exposure(
                    partial16, clean_preview16, base_exposure_ev
                )
                partial_rgb = np.right_shift(partial16, 8).astype(np.uint8)
                scaled_annotations = scale_source_annotations(
                    annotations, preview_w / max(1, width), preview_h / max(1, height)
                )
                partial_labeled, _records = annotate_meteor_sources(
                    partial16, scaled_annotations
                )
                self.work_queue.put((
                    "exact_preview_partial", signature, partial_rgb, partial_labeled,
                    index, len(selected), included,
                ))
                last_partial = now
            self.work_queue.put((
                "progress", index / max(1, len(selected)) * 90,
                f"原始像素精准预览 {index}/{len(selected)}：{source_path.name}",
            ))
        result = adjust_composite_base_exposure(result, clean_base, base_exposure_ev)
        labeled, _records = annotate_meteor_sources(result, annotations)
        full_rgb = np.right_shift(result, 8).astype(np.uint8)
        full_labeled_rgb = labeled.astype(np.uint8, copy=False)
        resized16 = cv2.resize(result, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        preview_rgb = np.right_shift(resized16, 8).astype(np.uint8)
        labeled_rgb = cv2.resize(labeled, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        return (
            "exact_preview", signature, preview_rgb, labeled_rgb,
            full_rgb, full_labeled_rgb, included, (width, height),
        )

    def _request_global_preview(self, signature: str) -> None:
        # Coalesce rapid slider/point edits before starting an expensive worker.
        # A running worker remains single-instance and its stale result is discarded.
        if self.preview_base is None or self.global_preview_loading_signature is not None:
            return
        self.global_preview_pending_signature = signature
        if self.global_preview_request_after_id is not None:
            try:
                self.after_cancel(self.global_preview_request_after_id)
            except tk.TclError:
                pass
        self.global_preview_request_after_id = self.after(220, self._start_global_preview_request)

    def _start_global_preview_request(self) -> None:
        self.global_preview_request_after_id = None
        signature = self.global_preview_pending_signature
        self.global_preview_pending_signature = None
        if (
            signature is None or self.preview_base is None
            or self.global_preview_loading_signature is not None
            or signature != self._global_preview_state_signature()
            or signature == self.global_preview_signature
        ):
            return
        marked = {
            Path(key): [Stroke(
                item.points.copy(), item.width, item.feather, item.erase, item.locked,
                item.auto_score, item.offset_x, item.offset_y, item.rotation,
                item.length_scale, item.width_scale, item.opacity,
                item.brightness_override, item.background_cleanup_override,
                item.saturation_override, item.preserve_brightness_override,
                item.match_exposure_override, item.blend_mode_override,
                item.auto_blend_enabled, item.auto_strength, item.auto_black_point,
                item.auto_cleanup, item.auto_brightness, item.auto_feather,
                source_mode=item.source_mode,
            ) for item in values]
            for key, values in self.strokes.items()
            if key in self.pairs and values
        }
        self.global_preview_loading_signature = signature
        self.global_preview_generation = getattr(self, "global_preview_generation", 0) + 1
        generation = self.global_preview_generation
        self.status.set(f"正在生成总融合预览：合成 {len(marked)} 张已标记图片…")
        self._run_worker(
            self._global_preview_worker, signature, generation, self.preview_base.copy(), marked,
            {key: value.copy() for key, value in self.image_adjustments.items()},
            self.adjustment_defaults.copy(), self.blend_mode.get(),
            {str(path): self.original_sources.get(str(path), path) for path in marked},
        )

    def _global_preview_worker(
        self, signature: str, generation: int, base_preview: np.ndarray,
        marked: dict[Path, list[Stroke]], adjustments: dict[str, dict],
        adjustment_defaults: dict, blend_mode: str, original_paths: dict[str, Path],
    ):
        result = base_preview.copy()
        preview_height, preview_width = result.shape[:2]
        total = max(1, len(marked))
        included = 0
        preview_annotations = []
        last_partial = 0.0
        for index, (source_path, strokes) in enumerate(marked.items(), start=1):
            if generation != getattr(self, "global_preview_generation", generation):
                return "global_preview_cancelled", signature
            original_path = original_paths.get(str(source_path), source_path)
            aligned_preview = MeteorComposer._cached_layer_preview(
                self,
                source_path, preview_width, preview_height
            )
            original_preview = MeteorComposer._cached_layer_preview(
                self,
                original_path, preview_width, preview_height
            )
            full_width, full_height, _depth, _channels = image_info(source_path)
            if aligned_preview is None or original_preview is None:
                aligned, original = read_uint16_pair(source_path, original_path)
                full_height, full_width = aligned.shape[:2]
                if original.shape[:2] != (full_height, full_width):
                    raise ValueError(f"自动对齐图与原始图尺寸不一致：{source_path.name}")
                aligned_preview = np.right_shift(cv2.resize(
                    aligned, (preview_width, preview_height), interpolation=cv2.INTER_AREA
                ), 8).astype(np.uint8)
                original_preview = np.right_shift(cv2.resize(
                    original, (preview_width, preview_height), interpolation=cv2.INTER_AREA
                ), 8).astype(np.uint8)
                if hasattr(self, "preview_cache_lock") and hasattr(self, "layer_preview_cache"):
                    with self.preview_cache_lock:
                        for layer_path, layer_image in (
                            (source_path, aligned_preview), (original_path, original_preview),
                        ):
                            layer_stat = layer_path.stat()
                            key = (
                                str(layer_path), layer_stat.st_mtime_ns, layer_stat.st_size,
                                preview_width, preview_height,
                            )
                            self.layer_preview_cache[key] = layer_image
                            self.layer_preview_cache.move_to_end(key)
                        while len(self.layer_preview_cache) > 48:
                            self.layer_preview_cache.popitem(last=False)
            scale_to_preview = preview_width / max(1, full_width)
            scaled = [Stroke(
                item.points, max(1, int(round(item.width * scale_to_preview))),
                max(0, int(round(item.feather * scale_to_preview))), item.erase,
                item.locked, item.auto_score, item.offset_x * scale_to_preview,
                item.offset_y * scale_to_preview, item.rotation, item.length_scale,
                item.width_scale, item.opacity,
                item.brightness_override, item.background_cleanup_override,
                item.saturation_override, item.preserve_brightness_override,
                item.match_exposure_override, item.blend_mode_override,
                item.auto_blend_enabled, item.auto_strength, item.auto_black_point,
                item.auto_cleanup, item.auto_brightness,
                (None if item.auto_feather is None else max(1, int(round(item.auto_feather * scale_to_preview)))),
                source_mode=item.source_mode,
            ) for item in strokes]
            adjustment = {**adjustment_defaults, **adjustments.get(str(source_path), {})}
            result, mask = compose_meteor_sources(
                aligned_preview, original_preview, result, scaled,
                bool(adjustment["match_exposure"]), bool(adjustment["curve_enabled"]),
                float(adjustment["curve_shadows"]), float(adjustment["curve_highlights"]),
                blend_mode, bool(adjustment.get("preserve_brightness", True)),
                float(adjustment.get("meteor_brightness", 100)),
                float(adjustment.get("background_cleanup", 70)),
                bool(adjustment.get("auto_optimize", True)),
            )
            if np.any(mask > 0.001):
                included += 1
                preview_annotations.extend(meteor_source_annotations(
                    source_path.stem, scaled, preview_width, preview_height,
                    False, mask,
                ))
            now = time.monotonic()
            if index == 1 or index == len(marked) or now - last_partial >= 0.18:
                partial = result.copy()
                partial_labeled, _records = annotate_meteor_sources(
                    partial, preview_annotations
                )
                self.work_queue.put((
                    "global_preview_partial", signature, partial, partial_labeled,
                    index, len(marked), included,
                ))
                last_partial = now
            self.work_queue.put((
                "progress", index / total * 100,
                f"正在生成总融合预览 {index}/{len(marked)}：{source_path.name}",
            ))
        labeled, _records = annotate_meteor_sources(result, preview_annotations)
        return "global_preview", signature, result, labeled, included

    def _render_preview(self) -> None:
        mode = self.view_mode.get()
        shared_composite = mode in {"blend", "labeled"} and self._uses_shared_base()
        if self.preview_base is None or (self.preview_source is None and not shared_composite):
            return
        height, width = (
            self.preview_source.shape[:2]
            if self.preview_source is not None else self.preview_base.shape[:2]
        )
        scale_to_preview = width / max(1, self.current_dims[0])
        scaled_strokes = [Stroke(
            item.points, max(1, int(round(item.width * scale_to_preview))),
            max(0, int(round(item.feather * scale_to_preview))), item.erase, item.locked,
            item.auto_score, item.offset_x * scale_to_preview, item.offset_y * scale_to_preview,
            item.rotation, item.length_scale, item.width_scale, item.opacity,
            item.brightness_override, item.background_cleanup_override,
            item.saturation_override, item.preserve_brightness_override,
            item.match_exposure_override, item.blend_mode_override,
            item.auto_blend_enabled, item.auto_strength, item.auto_black_point,
            item.auto_cleanup, item.auto_brightness,
            (None if item.auto_feather is None else max(1, int(round(item.auto_feather * scale_to_preview)))),
            source_mode=item.source_mode,
        ) for item in self.strokes.get(str(self.current_path), [])]
        preview_mask_strokes = scaled_strokes
        live_stroke = self.live_erase_stroke
        if live_stroke is not None:
            preview_mask_strokes = [
                *scaled_strokes, self._preview_clone(live_stroke, self.current_dims[0])
            ]
        composite_mode = mode in {"blend", "labeled"}
        mask_edit_mode = mode == "source"
        self.preview_quality_status.set("当前画布：原始像素")
        self.preview_mask_overlay = None
        if mode == "base":
            shown = adjust_composite_base_exposure(
                self.preview_base, self.preview_base, self.base_exposure_tenths.get() / 10.0
            )
        elif composite_mode:
            exact_signature = self._exact_preview_state_signature()
            exact_full = (
                self.exact_labeled_preview_full_rgb
                if mode == "labeled" else self.exact_preview_full_rgb
            )
            exact_partial = (
                self.exact_labeled_preview_rgb if mode == "labeled" else self.exact_preview_rgb
            )
            exact_current = self.exact_preview_signature == exact_signature and exact_full is not None
            progressive_current = (
                self.exact_preview_loading_signature == exact_signature and exact_partial is not None
            )
            if exact_current:
                shown = exact_full
                self.preview_quality_status.set("当前画布：16 位合成链路 · 原始像素")
                self.exact_preview_status.set(f"精准预览：有效（{shown.shape[1]}×{shown.shape[0]}）")
            elif progressive_current and exact_partial.shape[:2] == self.preview_base.shape[:2]:
                shown = exact_partial
                self.preview_quality_status.set("当前画布：精准预览正在逐张累积…")
            else:
                # The progressive exact frame is intentionally smaller. Showing
                # it on the main canvas changed the image dimensions twice and
                # made every view switch appear to zoom by itself. Keep the
                # full-size fast composite visible until the exact full frame is
                # ready; progress remains visible in the status bar.
                global_signature = self._global_preview_state_signature()
                if self.global_preview_signature != global_signature:
                    self._request_global_preview(global_signature)
                fast = (
                    self.global_labeled_preview_rgb if mode == "labeled"
                    else self.global_preview_rgb
                )
                if mode == "labeled" and fast is None and self.global_preview_rgb is not None:
                    fast, _records = annotate_meteor_sources(
                        self.global_preview_rgb, self._global_annotations_from_state()
                    )
                    self.global_labeled_preview_rgb = fast
                shown = fast if fast is not None else self.preview_base
                self.preview_quality_status.set("当前画布：原始像素 · 后台精准更新中…")
                self._schedule_automatic_exact_preview()
            if not exact_current and not (
                progressive_current and exact_partial.shape[:2] == self.preview_base.shape[:2]
            ):
                shown = adjust_composite_base_exposure(
                    shown, self.preview_base, self.base_exposure_tenths.get() / 10.0
                )
        else:
            shown = self.preview_source
            # The original TIFF remains intact at the old position. For geometrically
            # transformed meteors, also render a non-destructive copy at the new
            # position so the source view explains exactly what will be composited.
            if any(stroke_is_transformed(item) for item in scaled_strokes if not item.erase):
                reference_strokes = []
                for item in scaled_strokes:
                    if normalized_source_mode(item) != self._current_source_mode():
                        continue
                    preview_item = replace(item, points=item.points.copy())
                    preview_item.brightness_override = None
                    preview_item.background_cleanup_override = None
                    preview_item.saturation_override = None
                    preview_item.preserve_brightness_override = None
                    preview_item.match_exposure_override = None
                    preview_item.blend_mode_override = None
                    reference_strokes.append(preview_item)
                shown, _reference_mask = compose_meteor_objects(
                    self.preview_source, shown, reference_strokes,
                    False, False, 0, 0, "普通粘贴", False, 100, 0, False,
                )
        mask_crop = None
        if mask_edit_mode:
            current_mode = self._current_source_mode()
            visible_mask_strokes = [
                item for item in preview_mask_strokes
                if normalized_source_mode(item) == current_mode
            ]
            mask_crop = transformed_mask_crop(self.preview_source, visible_mask_strokes)
        if mask_edit_mode and self.show_mask.get() and mask_crop is not None:
            mask, (mx0, my0, mx1, my1) = mask_crop
            if np.any(mask > 0.001):
                self.preview_mask_overlay = mask_crop
        self._present_preview_image(shown, composite_mode, mask_edit_mode)

    def _canvas_fit_scale(self) -> float:
        if self.preview_rgb is None:
            return 1.0
        height, width = self.preview_rgb.shape[:2]
        return min(
            max(1, self.canvas.winfo_width()) / max(1, width),
            max(1, self.canvas.winfo_height()) / max(1, height),
        )

    def _canvas_view_origin(self) -> tuple[float, float]:
        return (
            self.canvas_center_x - self.canvas.winfo_width() / (2.0 * self.canvas_zoom),
            self.canvas_center_y - self.canvas.winfo_height() / (2.0 * self.canvas_zoom),
        )

    def _clamp_canvas_center(self) -> None:
        if self.preview_rgb is None:
            return
        height, width = self.preview_rgb.shape[:2]
        half_w = self.canvas.winfo_width() / (2.0 * self.canvas_zoom)
        half_h = self.canvas.winfo_height() / (2.0 * self.canvas_zoom)
        self.canvas_center_x = (
            width / 2.0 if half_w >= width / 2.0
            else float(np.clip(self.canvas_center_x, half_w, width - half_w))
        )
        self.canvas_center_y = (
            height / 2.0 if half_h >= height / 2.0
            else float(np.clip(self.canvas_center_y, half_h, height - half_h))
        )

    def _redraw_canvas_only(self) -> None:
        if self.preview_rgb is None:
            return
        mode = self.view_mode.get()
        self._present_preview_image(
            self.preview_rgb, mode in {"blend", "labeled"}, mode == "source"
        )

    def _canvas_fit(self) -> None:
        if self.preview_rgb is None:
            return
        height, width = self.preview_rgb.shape[:2]
        self.canvas_center_x = width / 2.0
        self.canvas_center_y = height / 2.0
        self.canvas_zoom = self._canvas_fit_scale()
        self.canvas_fit_mode = True
        self.canvas_preserve_fit_once = False
        self._redraw_canvas_only()

    def _canvas_actual_size(self) -> None:
        if self.preview_rgb is None:
            return
        self.canvas_zoom = 1.0
        self.canvas_fit_mode = False
        self._clamp_canvas_center()
        self._redraw_canvas_only()

    def _canvas_zoom_by(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        if self.preview_rgb is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        anchor_x, anchor_y = anchor or (canvas_w // 2, canvas_h // 2)
        origin_x, origin_y = self._canvas_view_origin()
        image_x = origin_x + anchor_x / self.canvas_zoom
        image_y = origin_y + anchor_y / self.canvas_zoom
        minimum = min(1.0, self._canvas_fit_scale())
        self.canvas_zoom = float(np.clip(self.canvas_zoom * factor, minimum, 8.0))
        new_origin_x = image_x - anchor_x / self.canvas_zoom
        new_origin_y = image_y - anchor_y / self.canvas_zoom
        self.canvas_center_x = new_origin_x + canvas_w / (2.0 * self.canvas_zoom)
        self.canvas_center_y = new_origin_y + canvas_h / (2.0 * self.canvas_zoom)
        self.canvas_fit_mode = False
        self._clamp_canvas_center()
        self._redraw_canvas_only()

    def _canvas_wheel(self, event) -> str:
        return self._canvas_wheel_steps(event, 1 if event.delta > 0 else -1)

    def _canvas_wheel_steps(self, event, steps: int) -> str:
        self._canvas_zoom_by(1.25 ** steps, (int(event.x), int(event.y)))
        return "break"

    def _canvas_pan_start_event(self, event, with_left: bool = False) -> str:
        if self.preview_rgb is None:
            return "break"
        self.canvas_pan_start = (
            int(event.x), int(event.y), self.canvas_center_x, self.canvas_center_y
        )
        self.canvas_pan_with_left = with_left
        self.canvas.configure(cursor="fleur")
        return "break"

    def _canvas_pan_move_event(self, event) -> str:
        if self.canvas_pan_start is None:
            return "break"
        start_x, start_y, center_x, center_y = self.canvas_pan_start
        self.canvas_center_x = center_x - (event.x - start_x) / self.canvas_zoom
        self.canvas_center_y = center_y - (event.y - start_y) / self.canvas_zoom
        self.canvas_fit_mode = False
        self._clamp_canvas_center()
        self._redraw_canvas_only()
        return "break"

    def _canvas_pan_end_event(self, _event=None) -> str:
        self.canvas_pan_start = None
        self.canvas_pan_with_left = False
        self._update_brush_cursor()
        return "break"

    def _space_pan_press(self, _event=None) -> str | None:
        focused = self.focus_get()
        if focused is not None and focused.winfo_class() in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox"}:
            return None
        hovered = self.winfo_containing(self.winfo_pointerx(), self.winfo_pointery())
        if focused is not self.canvas and hovered is not self.canvas:
            return None
        self.space_pan_held = True
        if hovered is self.canvas:
            self.canvas.configure(cursor="fleur")
        return "break"

    def _space_pan_release(self, _event=None) -> str | None:
        if not self.space_pan_held:
            return None
        self.space_pan_held = False
        if self.canvas_pan_with_left:
            self._canvas_pan_end_event()
        return "break"

    def _canvas_configure(self, _event=None) -> None:
        if self.preview_rgb is None:
            return
        window_size = (max(1, self.winfo_width()), max(1, self.winfo_height()))
        window_changed = (
            self.canvas_last_window_size is None
            or window_size != self.canvas_last_window_size
        )
        self.canvas_last_window_size = window_size
        if self.canvas_fit_mode and window_changed:
            # Only a real top-level resize may change fit zoom. Notebook/status
            # reflow and clicks in blank UI areas must leave the photograph at
            # exactly the same scale.
            height, width = self.preview_rgb.shape[:2]
            self.canvas_zoom = self._canvas_fit_scale()
            self.canvas_center_x = width / 2.0
            self.canvas_center_y = height / 2.0
        elif not self.canvas_fit_mode:
            self._clamp_canvas_center()
        self._redraw_canvas_only()

    def _present_preview_image(
        self, shown: np.ndarray, composite_mode: bool, mask_edit_mode: bool = False
    ) -> None:
        same_frame = shown is self.preview_rgb
        if not same_frame:
            self.preview_frame_serial += 1
        self.preview_rgb = shown
        canvas_w = max(10, self.canvas.winfo_width())
        canvas_h = max(10, self.canvas.winfo_height())
        h, w = self.preview_rgb.shape[:2]
        initial_frame = self.canvas_image_shape is None
        if self.canvas_image_shape != (h, w):
            previous_shape = self.canvas_image_shape
            if previous_shape is not None and not self.canvas_fit_mode:
                old_h, old_w = previous_shape
                self.canvas_center_x = self.canvas_center_x / max(1, old_w) * w
                self.canvas_center_y = self.canvas_center_y / max(1, old_h) * h
                # Preserve the same normalized field of view when a background
                # render changes resolution. View buttons must not zoom or jump.
                self.canvas_zoom *= old_w / max(1, w)
            else:
                self.canvas_center_x = w / 2.0
                self.canvas_center_y = h / 2.0
            self.canvas_image_shape = (h, w)
        preserve_fit = self.canvas_fit_mode and self.canvas_preserve_fit_once
        self.canvas_preserve_fit_once = False
        if self.canvas_fit_mode and initial_frame and not preserve_fit:
            self.canvas_zoom = self._canvas_fit_scale()
            self.canvas_center_x = w / 2.0
            self.canvas_center_y = h / 2.0
        self._clamp_canvas_center()
        origin_x, origin_y = self._canvas_view_origin()
        crop_x0 = max(0, int(np.floor(origin_x)))
        crop_y0 = max(0, int(np.floor(origin_y)))
        crop_x1 = min(w, int(np.ceil(origin_x + canvas_w / self.canvas_zoom)))
        crop_y1 = min(h, int(np.ceil(origin_y + canvas_h / self.canvas_zoom)))
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            return
        dw = max(1, int(round((crop_x1 - crop_x0) * self.canvas_zoom)))
        dh = max(1, int(round((crop_y1 - crop_y0) * self.canvas_zoom)))
        viewport_key = (
            self.preview_frame_serial, crop_x0, crop_y0, crop_x1, crop_y1, dw, dh
        )
        cached_view = self.viewport_cache.get(viewport_key)
        if cached_view is not None:
            image = cached_view[0]
            self.viewport_cache.move_to_end(viewport_key)
        else:
            image = Image.fromarray(self.preview_rgb[crop_y0:crop_y1, crop_x0:crop_x1])
            if image.size != (dw, dh):
                resample = (
                    Image.Resampling.LANCZOS if self.canvas_zoom < 1.0
                    else Image.Resampling.BICUBIC
                )
                image = image.resize((dw, dh), resample)
            byte_size = max(1, dw * dh * 3)
            self.viewport_cache[viewport_key] = (image, byte_size)
            self.viewport_cache.move_to_end(viewport_key)
            self.viewport_cache_bytes += byte_size
            while self.viewport_cache_bytes > self.viewport_cache_budget and len(self.viewport_cache) > 1:
                _old_key, (_old_image, old_size) = self.viewport_cache.popitem(last=False)
                self.viewport_cache_bytes -= old_size
        if mask_edit_mode and self.show_mask.get() and self.preview_mask_overlay is not None:
            mask, (mx0, my0, mx1, my1) = self.preview_mask_overlay
            ix0, iy0 = max(crop_x0, mx0), max(crop_y0, my0)
            ix1, iy1 = min(crop_x1, mx1), min(crop_y1, my1)
            if ix1 > ix0 and iy1 > iy0:
                sx = dw / max(1, crop_x1 - crop_x0)
                sy = dh / max(1, crop_y1 - crop_y0)
                dx0 = max(0, int(round((ix0 - crop_x0) * sx)))
                dy0 = max(0, int(round((iy0 - crop_y0) * sy)))
                dx1 = min(dw, int(round((ix1 - crop_x0) * sx)))
                dy1 = min(dh, int(round((iy1 - crop_y0) * sy)))
                if dx1 > dx0 and dy1 > dy0:
                    alpha = mask[iy0 - my0:iy1 - my0, ix0 - mx0:ix1 - mx0]
                    alpha = cv2.resize(
                        alpha, (dx1 - dx0, dy1 - dy0), interpolation=cv2.INTER_AREA
                    )
                    alpha = (np.clip(alpha, 0.0, 1.0) * 0.55)[..., None]
                    display = np.asarray(image).copy()
                    region = display[dy0:dy1, dx0:dx1].astype(np.float32)
                    red = np.empty_like(region)
                    red[:] = (255, 35, 25)
                    display[dy0:dy1, dx0:dx1] = np.clip(
                        region * (1.0 - alpha) + red * alpha, 0, 255
                    ).astype(np.uint8)
                    image = Image.fromarray(display)
        draw_x = int(round((crop_x0 - origin_x) * self.canvas_zoom))
        draw_y = int(round((crop_y0 - origin_y) * self.canvas_zoom))
        full_x0 = draw_x - crop_x0 * self.canvas_zoom
        full_y0 = draw_y - crop_y0 * self.canvas_zoom
        self.display_box = (
            full_x0, full_y0, full_x0 + w * self.canvas_zoom,
            full_y0 + h * self.canvas_zoom,
        )
        reusable = False
        if (
            self.preview_photo is not None
            and self.preview_image_item is not None
            and self.preview_display_size == (dw, dh)
        ):
            try:
                reusable = bool(self.canvas.type(self.preview_image_item))
            except tk.TclError:
                reusable = False
        if reusable:
            # Keep the canvas image item alive. Deleting and recreating it made a
            # tiny local edit look like the entire photograph flashed/repainted.
            for item in self.canvas.find_all():
                if item != self.preview_image_item:
                    self.canvas.delete(item)
            self.preview_photo.paste(image)
            self.canvas.coords(self.preview_image_item, draw_x, draw_y)
        else:
            self.canvas.delete("all")
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_image_item = self.canvas.create_image(
                draw_x, draw_y, anchor="nw", image=self.preview_photo, tags=("preview_image",)
            )
            self.preview_display_size = (dw, dh)
        self.canvas_zoom_label.set(f"{self.canvas_zoom * 100:.0f}%")
        self.context_highlight = None
        self.hover_candidate_items = []
        self.hover_candidate_index = None
        if composite_mode:
            self._draw_selected_object_overlay()
        if mask_edit_mode and self.show_mask.get():
            self._draw_candidate_guides()
            self._draw_mask_annotations()
            self._draw_transform_reference_guides()
        self.cursor_items = []
        self._update_brush_cursor()
        if mask_edit_mode and self.cursor_position is not None and not self.active_points:
            self._update_candidate_hover(*self.cursor_position)

    def _draw_candidate_guides(self) -> None:
        """Keep every selectable AI candidate visible before it is chosen."""
        if not self.current_path:
            return
        x0, y0, x1, y1 = self.display_box
        threshold = self.candidate_thresholds.get(
            str(self.current_path), int(self.candidate_threshold.get())
        )
        for index, stroke in enumerate(self.candidates.get(str(self.current_path), [])):
            if stroke.locked or not stroke.points:
                continue
            transformed = transformed_stroke_points(stroke, *self.current_dims)
            points = [
                (x0 + px * (x1 - x0), y0 + py * (y1 - y0))
                for px, py in transformed
            ]
            score = int(stroke.auto_score or 0)
            color = "#55f29a" if score >= threshold else "#43c985"
            tags = ("candidate_guide", f"candidate_{index}")
            if len(points) == 1:
                px, py = points[0]
                self.canvas.create_oval(
                    px - 5, py - 5, px + 5, py + 5,
                    outline=color, width=2, dash=(4, 3), tags=tags,
                )
            else:
                coords = [coordinate for point in points for coordinate in point]
                self.canvas.create_line(
                    *coords, fill=color, width=2, dash=(7, 4),
                    capstyle="round", tags=tags,
                )
            px, py = points[len(points) // 2]
            self.canvas.create_text(
                px, py - 8, text=str(score), fill=color,
                font=("TkDefaultFont", 8, "bold"), tags=tags,
            )

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

    def _draw_transform_reference_guides(self) -> None:
        """Show old geometry and transformed geometry together in the source view."""
        if not self.current_path:
            return
        x0, y0, x1, y1 = self.display_box
        for stroke in self.strokes.get(str(self.current_path), []):
            if stroke.erase or not stroke.points or not stroke_is_transformed(stroke):
                continue
            original = replace(stroke, points=stroke.points.copy())
            reset_stroke_geometry(original)
            original_points = transformed_stroke_points(original, *self.current_dims)
            moved_points = transformed_stroke_points(stroke, *self.current_dims)
            original_canvas = [
                (x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in original_points
            ]
            moved_canvas = [
                (x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in moved_points
            ]
            if len(original_canvas) == 1:
                ox, oy = original_canvas[0]
                self.canvas.create_oval(
                    ox - 7, oy - 7, ox + 7, oy + 7, outline="#55e6ff",
                    width=2, dash=(5, 4), tags=("transform_reference",),
                )
            else:
                self.canvas.create_line(
                    *(coordinate for point in original_canvas for coordinate in point),
                    fill="#55e6ff", width=3, dash=(8, 5), capstyle="round",
                    tags=("transform_reference",),
                )
            if len(moved_canvas) == 1:
                mx, my = moved_canvas[0]
                self.canvas.create_oval(
                    mx - 7, my - 7, mx + 7, my + 7, outline="#ff9635",
                    width=3, tags=("transform_reference",),
                )
            else:
                self.canvas.create_line(
                    *(coordinate for point in moved_canvas for coordinate in point),
                    fill="#ff9635", width=3, capstyle="round",
                    tags=("transform_reference",),
                )
            old_center = np.mean(np.asarray(original_canvas, dtype=np.float32), axis=0)
            new_center = np.mean(np.asarray(moved_canvas, dtype=np.float32), axis=0)
            if float(np.linalg.norm(new_center - old_center)) > 3.0:
                self.canvas.create_line(
                    float(old_center[0]), float(old_center[1]),
                    float(new_center[0]), float(new_center[1]),
                    fill="#55e6ff", width=2, arrow="last", dash=(5, 4),
                    tags=("transform_reference",),
                )
            self.canvas.create_text(
                float(new_center[0]), float(new_center[1]) - 16,
                text="[已人工变换] 橙=变换后  青虚线=原位置",
                fill="#ffb35c", font=("TkDefaultFont", 10, "bold"),
                tags=("transform_reference",),
            )

    def _tool_settings_changed(self) -> None:
        self._update_brush_cursor()

    def _cursor_motion(self, event) -> None:
        self.cursor_position = (float(event.x), float(event.y))
        self._update_brush_cursor()
        if self.view_mode.get() in {"blend", "labeled"}:
            self._clear_candidate_hover()
            self._update_object_cursor(event.x, event.y)
            return
        if self.view_mode.get() != "source":
            self._clear_candidate_hover()
            return
        if self.active_points:
            self._clear_candidate_hover()
            return
        now = time.monotonic()
        previous = self.last_candidate_hover_position
        if (
            previous is not None
            and now - self.last_candidate_hover_time < 0.03
            and np.hypot(event.x - previous[0], event.y - previous[1]) < 5.0
        ):
            return
        self.last_candidate_hover_time = now
        self.last_candidate_hover_position = (float(event.x), float(event.y))
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
        # Keep the button alive while the pointer crosses the small gap from the
        # candidate line to the button. Previously Motion cleared it before the
        # subsequent click could reach the button area handled by the canvas.
        if self.hover_candidate_items:
            button_bbox = self.canvas.bbox("candidate_pick")
            if point_in_padded_bbox(canvas_x, canvas_y, button_bbox, padding=16.0):
                return
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
        # The canvas widget owns the click. Binding the same press to both the
        # item tag and the canvas caused ordering differences across Tk builds:
        # one handler could redraw/delete the item before the other ran.

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
        incremental = None
        if matching is not None:
            matching.locked = True
        else:
            values = self.strokes.setdefault(key, [])
            insert_at = next((i for i, stroke in enumerate(values) if stroke.erase), len(values))
            values.insert(insert_at, candidate)
            # The cached combined preview predates this object. Rebuild only its
            # footprint (and intersecting objects), treating the new stroke as
            # both the affected footprint and the current object to include.
            incremental = self._incremental_recomposed_object_image(
                (key, insert_at), candidate, include_selected=True
            )
        score = candidate.auto_score or 0
        self.edit_history.pop(key, None)
        self.edit_redo.pop(key, None)
        self._clear_candidate_hover()
        self.show_mask.set(True)
        self._update_candidate_summary(key)
        self._update_tree_status()
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
            self.status.set(f"已选中并锁定 {score} 分候选；对应局部已即时融合")
        else:
            self._render_preview()
            self.status.set(f"已选中并锁定 {score} 分候选；调阈值或清除蒙版时都会保留")
        self._schedule_autosave()
        return "break"

    def _update_brush_cursor(self) -> None:
        for item in self.cursor_items:
            self.canvas.delete(item)
        self.cursor_items = []
        if (self.cursor_position is None or self.preview_source is None
                or self.view_mode.get() != "source"):
            self.canvas.configure(cursor="arrow")
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

    def _editable_object_refs(self) -> list[tuple[str, int]]:
        if self._uses_shared_base():
            sources = (
                (key, values) for key, values in self.strokes.items() if key in self.pairs
            )
        elif self.current_path:
            key = str(self.current_path)
            sources = ((key, self.strokes.get(key, [])),)
        else:
            sources = ()
        return [
            (key, index)
            for key, values in sources
            for index, stroke in enumerate(values)
            if not stroke.erase and stroke.points
        ]

    def _selected_stroke(self) -> Stroke | None:
        if self.selected_object is None:
            return None
        key, index = self.selected_object
        if key not in self.pairs:
            self.selected_object = None
            return None
        values = self.strokes.get(key, [])
        if not (0 <= index < len(values)) or values[index].erase:
            self.selected_object = None
            return None
        return values[index]

    def _object_canvas_geometry(
        self, reference: tuple[str, int], stroke_override: Stroke | None = None
    ) -> dict | None:
        key, index = reference
        values = self.strokes.get(key, [])
        if stroke_override is None:
            if not (0 <= index < len(values)):
                return None
            stroke = values[index]
        else:
            stroke = stroke_override
        if stroke.erase or not stroke.points:
            return None
        x0, y0, x1, y1 = self.display_box
        full_width, full_height = self.current_dims
        normalized = transformed_stroke_points(stroke, full_width, full_height)
        points = np.asarray([
            (x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in normalized
        ], dtype=np.float64)
        if not len(points):
            return None
        center = points.mean(axis=0)
        direction = points[-1] - points[0] if len(points) > 1 else np.asarray([1.0, 0.0])
        length = float(np.hypot(direction[0], direction[1]))
        axis = direction / length if length > 1e-6 else np.asarray([1.0, 0.0])
        normal = np.asarray([-axis[1], axis[0]])
        along = (points - center) @ axis
        across = (points - center) @ normal
        display_scale = (x1 - x0) / max(1, full_width)
        half_length = max(8.0, float(np.max(np.abs(along))) + 4.0)
        half_width = max(
            7.0,
            float(np.max(np.abs(across))) + stroke.width * stroke.width_scale * display_scale * 0.5 + 4.0,
        )
        corners = [
            center - axis * half_length - normal * half_width,
            center + axis * half_length - normal * half_width,
            center + axis * half_length + normal * half_width,
            center - axis * half_length + normal * half_width,
        ]
        # Resize handles live outside the meteor body.  Previously eight handles
        # sat directly on a thin trail and used a 12 px hit radius, so an ordinary
        # move near the head/tail was frequently interpreted as a scale gesture.
        handle_length = half_length + 18.0
        handle_width = half_width + 18.0
        handles = {
            "length_start": center - axis * handle_length,
            "length_end": center + axis * handle_length,
            "width_neg": center - normal * handle_width,
            "width_pos": center + normal * handle_width,
            "scale_nw": center - axis * handle_length - normal * handle_width,
            "scale_ne": center + axis * handle_length - normal * handle_width,
            "scale_se": center + axis * handle_length + normal * handle_width,
            "scale_sw": center - axis * handle_length + normal * handle_width,
            "rotate": center - normal * (half_width + 42.0),
        }
        return {
            "points": points, "center": center, "axis": axis, "normal": normal,
            "half_length": half_length, "half_width": half_width, "corners": corners,
            "handles": handles, "display_scale": max(1e-9, display_scale),
        }

    def _set_selected_controls_state(self, has_selection: bool, independent: bool = False) -> None:
        for index, widget in enumerate(self.selected_object_controls):
            if index == 0:
                state = "normal" if has_selection else "disabled"
            elif not has_selection or not independent:
                state = "disabled"
            else:
                state = "readonly" if isinstance(widget, ttk.Combobox) else "normal"
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _load_selected_object_adjustments(self) -> None:
        stroke = self._selected_stroke()
        self.loading_selected_adjustments = True
        try:
            if stroke is None or self.selected_object is None:
                self.selected_object_summary.set("尚未选择流星")
                self.selected_auto_summary.set("自动参数：尚未分析")
                self.selected_override_enabled.set(False)
                self._set_selected_controls_state(False)
                self.reset_selected_button.configure(state="disabled")
                self.selected_source_mode.set("自动对齐素材")
                return
            key, index = self.selected_object
            image_values = {**self.adjustment_defaults, **self.image_adjustments.get(key, {})}
            independent = any(value is not None for value in (
                stroke.brightness_override, stroke.background_cleanup_override,
                stroke.saturation_override, stroke.preserve_brightness_override,
                stroke.match_exposure_override, stroke.blend_mode_override,
            ))
            ordinal = 1 + sum(
                1 for item in self.strokes.get(key, [])[:index]
                if not item.erase and item.points
            )
            transform_suffix = " · 已人工变换" if stroke_is_transformed(stroke) else ""
            source_suffix = " · 原始素材" if normalized_source_mode(stroke) == "original" else " · 自动对齐素材"
            self.selected_object_summary.set(f"{Path(key).stem} · 流星 {ordinal}{source_suffix}{transform_suffix}")
            self.selected_source_mode.set(
                "原始素材" if normalized_source_mode(stroke) == "original" else "自动对齐素材"
            )
            if not stroke.auto_blend_enabled:
                self.selected_auto_summary.set("自动参数：已关闭（使用原始/手动融合）")
            elif stroke.auto_black_point is None:
                self.selected_auto_summary.set("自动参数：动态标准值（建议执行原尺寸分析）")
            else:
                self.selected_auto_summary.set(
                    f"自动参数[{stroke.auto_strength}]：黑场 {stroke.auto_black_point:.2f} · "
                    f"净化 {stroke.auto_cleanup:.0f} · 亮度 {stroke.auto_brightness:.0f}% · "
                    f"有效羽化 {stroke.auto_feather}px"
                )
            self.reset_selected_button.configure(
                state="normal" if stroke_is_transformed(stroke) else "disabled"
            )
            self.selected_override_enabled.set(independent)
            self.selected_brightness.set(int(round(
                image_values.get("meteor_brightness", 100)
                if stroke.brightness_override is None else stroke.brightness_override
            )))
            self.selected_cleanup.set(int(round(
                image_values.get("background_cleanup", 70)
                if stroke.background_cleanup_override is None else stroke.background_cleanup_override
            )))
            self.selected_saturation.set(int(round(
                100 if stroke.saturation_override is None else stroke.saturation_override
            )))
            self.selected_preserve.set(bool(
                image_values.get("preserve_brightness", True)
                if stroke.preserve_brightness_override is None else stroke.preserve_brightness_override
            ))
            self.selected_match.set(bool(
                image_values.get("match_exposure", False)
                if stroke.match_exposure_override is None else stroke.match_exposure_override
            ))
            self.selected_blend.set(
                self.blend_mode.get() if stroke.blend_mode_override is None else stroke.blend_mode_override
            )
            self.selected_feather.set(int(stroke.feather))
            self._set_selected_controls_state(True, independent)
        finally:
            self.loading_selected_adjustments = False

    def _selected_source_mode_changed(self, _event=None) -> None:
        if self.loading_selected_adjustments:
            return
        mode = "original" if self.selected_source_mode.get() == "原始素材" else "aligned"
        self._set_selected_source_mode(mode)

    def _set_selected_source_mode(self, mode: str) -> None:
        stroke = self._selected_stroke()
        if stroke is None or self.selected_object is None:
            return
        key, index = self.selected_object
        mode = "original" if mode == "original" else "aligned"
        if mode == "original" and key not in self.original_sources:
            self.status.set("这张图片没有独立的原始素材；当前图层本身就是原始状态")
            self.loading_selected_adjustments = True
            self.selected_source_mode.set("自动对齐素材")
            self.loading_selected_adjustments = False
            return
        if normalized_source_mode(stroke) == mode:
            return
        before = replace(stroke, points=stroke.points.copy())
        stroke.source_mode = mode
        after = replace(stroke, points=stroke.points.copy())
        self._record_edit(key, ("transform", index, (before, after)))
        incremental = self._incremental_parameter_change_image(before)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
        else:
            self._invalidate_global_preview()
            self._render_preview()
        self._update_tree_status_for_key(key)
        self._load_selected_object_adjustments()
        self._schedule_autosave()
        label = "原始素材" if mode == "original" else "自动对齐素材"
        self.status.set(f"当前流星已切换为{label}；其他流星保持原来的来源不变")

    def _selected_override_changed(self, *_args) -> None:
        if self.loading_selected_adjustments:
            return
        stroke = self._selected_stroke()
        if stroke is None:
            self._load_selected_object_adjustments()
            return
        before = replace(stroke, points=stroke.points.copy())
        if self.selected_override_enabled.get():
            stroke.brightness_override = float(self.selected_brightness.get())
            stroke.background_cleanup_override = float(self.selected_cleanup.get())
            stroke.saturation_override = float(self.selected_saturation.get())
            stroke.preserve_brightness_override = bool(self.selected_preserve.get())
            stroke.match_exposure_override = bool(self.selected_match.get())
            stroke.blend_mode_override = self.selected_blend.get()
        else:
            stroke.brightness_override = None
            stroke.background_cleanup_override = None
            stroke.saturation_override = None
            stroke.preserve_brightness_override = None
            stroke.match_exposure_override = None
            stroke.blend_mode_override = None
        key, _index = self.selected_object
        self._sync_matching_candidate(key, stroke)
        self._set_selected_controls_state(True, self.selected_override_enabled.get())
        incremental = self._incremental_parameter_change_image(before)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
        else:
            self._invalidate_global_preview()
            self._render_preview()
        self._schedule_autosave()
        self.status.set(
            "所选流星已启用独立融合参数"
            if self.selected_override_enabled.get() else "所选流星已恢复跟随当前图片"
        )

    def _selected_adjustment_changed(self, *_args) -> None:
        if self.loading_selected_adjustments or not self.selected_override_enabled.get():
            return
        stroke = self._selected_stroke()
        if stroke is None or self.selected_object is None:
            return
        desired = (
            float(self.selected_brightness.get()), float(self.selected_cleanup.get()),
            float(self.selected_saturation.get()), bool(self.selected_preserve.get()),
            bool(self.selected_match.get()), self.selected_blend.get(),
            int(round(self.selected_feather.get())),
        )
        current = (
            stroke.brightness_override, stroke.background_cleanup_override,
            stroke.saturation_override, stroke.preserve_brightness_override,
            stroke.match_exposure_override, stroke.blend_mode_override, stroke.feather,
        )
        if current == desired:
            return
        before = replace(stroke, points=stroke.points.copy())
        (
            stroke.brightness_override, stroke.background_cleanup_override,
            stroke.saturation_override, stroke.preserve_brightness_override,
            stroke.match_exposure_override, stroke.blend_mode_override, stroke.feather,
        ) = desired
        key, _index = self.selected_object
        self._sync_matching_candidate(key, stroke)
        incremental = self._incremental_parameter_change_image(before)
        if incremental is not None:
            self._commit_incremental_global_preview(
                incremental, validate=False, realtime=True,
                dirty_box=self.last_incremental_box,
            )
        else:
            self._invalidate_global_preview()
            self._render_preview()
        self._schedule_autosave()

    def _draw_selected_object_overlay(self) -> None:
        for item in self.object_overlay_items:
            self.canvas.delete(item)
        self.object_overlay_items = []
        self.object_handle_centers = {}
        if self.view_mode.get() not in {"blend", "labeled"} or self.selected_object is None:
            self._load_selected_object_adjustments()
            return
        stroke = self._selected_stroke()
        if stroke is None:
            self._load_selected_object_adjustments()
            return
        geometry = self._object_canvas_geometry(self.selected_object)
        if geometry is None:
            return
        corners = geometry["corners"]
        polygon = [coordinate for point in corners for coordinate in point]
        color = "#ffb000" if stroke.locked else "#00e5ff"
        self.object_overlay_items.append(self.canvas.create_polygon(
            *polygon, fill="", outline=color, width=2, dash=(6, 3), tags=("object_overlay",)
        ))
        rotate = geometry["handles"]["rotate"]
        top = (corners[0] + corners[1]) * 0.5
        self.object_overlay_items.append(self.canvas.create_line(
            top[0], top[1], rotate[0], rotate[1], fill=color, width=2, tags=("object_overlay",)
        ))
        self.object_handle_centers = {
            name: (float(point[0]), float(point[1])) for name, point in geometry["handles"].items()
        }
        for name, (hx, hy) in self.object_handle_centers.items():
            radius = 7 if name == "rotate" else 5
            shape = self.canvas.create_oval if name == "rotate" else self.canvas.create_rectangle
            self.object_overlay_items.append(shape(
                hx - radius, hy - radius, hx + radius, hy + radius,
                fill="#181818", outline=color, width=2, tags=("object_overlay",),
            ))
        key, _index = self.selected_object
        original_warning = "  [原始状态/未对齐]" if key in self.use_original_sources else ""
        transformed_warning = "  [已人工变换]" if stroke_is_transformed(stroke) else ""
        label = Path(key).stem + original_warning + transformed_warning + ("  [已锁定]" if stroke.locked else "")
        label_x, label_y = corners[0]
        self.object_overlay_items.append(self.canvas.create_text(
            label_x, label_y - 12, text=label, anchor="sw", fill=color,
            font=("TkDefaultFont", 10, "bold"), tags=("object_overlay",),
        ))
        self._load_selected_object_adjustments()

    def _object_handle_at(self, canvas_x: float, canvas_y: float) -> str | None:
        best = None
        best_distance = 7.5
        for name, (hx, hy) in self.object_handle_centers.items():
            distance = float(np.hypot(canvas_x - hx, canvas_y - hy))
            if distance < best_distance:
                best, best_distance = name, distance
        return best

    def _find_object_near(self, canvas_x: float, canvas_y: float) -> tuple[str, int] | None:
        x0, y0, x1, y1 = self.display_box
        if not (x0 <= canvas_x <= x1 and y0 <= canvas_y <= y1):
            return None
        best_reference = None
        best_distance = float("inf")
        for reference in reversed(self._editable_object_refs()):
            geometry = self._object_canvas_geometry(reference)
            if geometry is None:
                continue
            points = geometry["points"]
            if len(points) == 1:
                distance = float(np.hypot(*(np.asarray((canvas_x, canvas_y)) - points[0])))
            else:
                distance = min(
                    self._point_segment_distance(canvas_x, canvas_y, *start, *end)
                    for start, end in zip(points, points[1:])
                )
            edge_distance = distance - geometry["half_width"]
            if edge_distance <= 12.0 and edge_distance < best_distance:
                best_reference, best_distance = reference, edge_distance
        return best_reference

    def _update_object_cursor(self, canvas_x: float, canvas_y: float) -> None:
        handle = self._object_handle_at(canvas_x, canvas_y)
        if handle == "rotate":
            cursor = "exchange"
        elif handle is not None:
            cursor = "sizing"
        elif self._find_object_near(canvas_x, canvas_y) is not None:
            cursor = "fleur"
        else:
            cursor = "arrow"
        try:
            self.canvas.configure(cursor=cursor)
        except tk.TclError:
            self.canvas.configure(cursor="crosshair" if handle else "arrow")

    def _object_pointer_start(self, event):
        handle = self._object_handle_at(event.x, event.y)
        reference = self.selected_object if handle else self._find_object_near(event.x, event.y)
        if reference is None:
            self._clear_object_selection()
            self.status.set("单击一颗流星即可选中；选中后可拖动或使用变换手柄")
            return "break"
        self.selected_object = reference
        if hasattr(self, "control_notebook"):
            self.control_notebook.select(self.selected_tools_tab)
        stroke = self._selected_stroke()
        self._draw_selected_object_overlay()
        if stroke is None:
            return "break"
        self.object_drag_mode = handle or "move"
        self.object_drag_start = (float(event.x), float(event.y))
        self.object_drag_original = replace(stroke, points=stroke.points.copy())
        self._prepare_live_object_drag(reference, self.object_drag_original)
        self.status.set("拖动中：流星内容与蒙版同步移动；松开后只精确更新旧位置和新位置")
        return "break"

    def _preview_clone(self, stroke: Stroke, full_width: int) -> Stroke:
        scale = self.preview_base.shape[1] / max(1, full_width)
        return Stroke(
            stroke.points.copy(), max(1, int(round(stroke.width * scale))),
            max(0, int(round(stroke.feather * scale))), stroke.erase, stroke.locked,
            stroke.auto_score, stroke.offset_x * scale, stroke.offset_y * scale,
            stroke.rotation, stroke.length_scale, stroke.width_scale, stroke.opacity,
            stroke.brightness_override, stroke.background_cleanup_override,
            stroke.saturation_override, stroke.preserve_brightness_override,
            stroke.match_exposure_override, stroke.blend_mode_override,
            stroke.auto_blend_enabled, stroke.auto_strength, stroke.auto_black_point,
            stroke.auto_cleanup, stroke.auto_brightness,
            (None if stroke.auto_feather is None else max(1, int(round(stroke.auto_feather * scale)))),
            source_mode=stroke.source_mode,
        )

    def _object_composite_settings(self, key: str) -> tuple:
        adjustment = {**self.adjustment_defaults, **self.image_adjustments.get(key, {})}
        return (
            bool(adjustment["match_exposure"]), bool(adjustment["curve_enabled"]),
            float(adjustment["curve_shadows"]), float(adjustment["curve_highlights"]),
            self.blend_mode.get(), bool(adjustment.get("preserve_brightness", True)),
            float(adjustment.get("meteor_brightness", 100)),
            float(adjustment.get("background_cleanup", 70)),
            bool(adjustment.get("auto_optimize", True)),
        )

    def _prepare_live_object_drag(self, reference: tuple[str, int], original: Stroke) -> None:
        self.object_drag_live_source = None
        self.object_drag_live_background = None
        self.object_drag_live_frame = None
        self.object_drag_live_box = None
        self.object_drag_live_settings = None
        if self.preview_base is None:
            return
        key, selected_index = reference
        try:
            selected_mode = normalized_source_mode(original)
            if self.current_path is not None and key == str(self.current_path):
                selected_preview = (
                    self.preview_original_source if selected_mode == "original"
                    else self.preview_aligned_source
                )
                source_preview = (
                    selected_preview.copy() if selected_preview is not None
                    else self.preview_source.copy()
                )
                aligned_preview = (
                    self.preview_aligned_source if self.preview_aligned_source is not None
                    else self.preview_source
                )
                original_preview = (
                    self.preview_original_source if self.preview_original_source is not None
                    else self.preview_source
                )
                full_width = self.current_dims[0]
            else:
                preview_size = (self.preview_base.shape[1], self.preview_base.shape[0])
                aligned_path = Path(key)
                original_path = self.original_sources.get(key, aligned_path)
                full_width = image_info(
                    original_path if selected_mode == "original" else aligned_path
                )[0]
                aligned_preview = self._cached_layer_preview(
                    aligned_path, preview_size[0], preview_size[1]
                )
                original_preview = self._cached_layer_preview(
                    original_path, preview_size[0], preview_size[1]
                )
                # Never decode a 300–500 MB TIFF synchronously inside ButtonPress.
                # Total-preview generation fills this cache.  If a very old layer
                # was evicted, drag the lightweight outline now and calculate the
                # accurate pixels after release instead of freezing the pointer.
                if aligned_preview is None or original_preview is None:
                    return
                source_preview = original_preview if selected_mode == "original" else aligned_preview
            settings = self._object_composite_settings(key)
            if self._uses_shared_base():
                cached = self.global_preview_rgb
                if cached is None:
                    return
                background = cached.copy()
                old_scaled = auto_optimized_stroke(
                    self._preview_clone(original, full_width),
                    bool(self.adjustment_defaults.get("auto_optimize", True)),
                )
                old_crop = transformed_object_crop(self.preview_base, old_scaled, fast=True)
                if old_crop is not None:
                    _old_patch, old_alpha, _validity, (ox0, oy0, ox1, oy1) = old_crop
                    removal = np.clip(old_alpha * 4.0, 0.0, 1.0)[..., None]
                    destination = background[oy0:oy1, ox0:ox1].astype(np.float32)
                    clean = self.preview_base[oy0:oy1, ox0:ox1].astype(np.float32)
                    background[oy0:oy1, ox0:ox1] = np.clip(
                        destination * (1.0 - removal) + clean * removal, 0, 255
                    ).astype(np.uint8)
            else:
                if self.current_path is None or key != str(self.current_path):
                    return
                others = [
                    self._preview_clone(item, full_width)
                    for index, item in enumerate(self.strokes.get(key, []))
                    if index != selected_index
                ]
                background, _mask = compose_meteor_sources(
                    aligned_preview, original_preview, self.preview_base, others, *settings
                )
            self.object_drag_live_source = source_preview
            self.object_drag_live_background = background
            self.object_drag_live_frame = background.copy()
            self.object_drag_live_full_width = full_width
            self.object_drag_live_settings = settings
            self.object_drag_last_render = 0.0
        except Exception:
            # Live feedback is optional; the accurate render on mouse release
            # remains available even if a source file cannot be read here.
            self.object_drag_live_source = None
            self.object_drag_live_background = None
            self.object_drag_live_settings = None

    def _render_live_object_drag(self, force: bool = False) -> None:
        if (
            self.object_drag_live_source is None
            or self.object_drag_live_background is None
            or self.object_drag_live_settings is None
        ):
            return
        now = time.monotonic()
        if not force and now - self.object_drag_last_render < 0.025:
            return
        stroke = self._selected_stroke()
        if stroke is None:
            return
        self.object_drag_last_render = now
        scaled = auto_optimized_stroke(
            self._preview_clone(stroke, self.object_drag_live_full_width),
            bool(self.adjustment_defaults.get("auto_optimize", True)),
        )
        transformed_source = transformed_object_crop(
            self.object_drag_live_source, scaled, fast=True
        )
        transformed_base = transformed_object_crop(self.preview_base, scaled, fast=True)
        if transformed_source is None or transformed_base is None:
            self._draw_selected_object_overlay()
            return
        source_patch, alpha, validity, box = transformed_source
        base_patch, _base_alpha, _base_validity, base_box = transformed_base
        if base_box != box:
            self._draw_selected_object_overlay()
            return
        x0, y0, x1, y1 = box
        shown = self.object_drag_live_frame
        if shown is None:
            shown = self.object_drag_live_background.copy()
            self.object_drag_live_frame = shown
        if self.object_drag_live_box is not None:
            px0, py0, px1, py1 = self.object_drag_live_box
            shown[py0:py1, px0:px1] = self.object_drag_live_background[py0:py1, px0:px1]
        destination = shown[y0:y1, x0:x1].astype(np.float32)
        # The drag frame is deliberately a lightweight positive-residual preview:
        # it retains the complete core and faint tail and avoids the expensive
        # cleanup/continuity analysis on every mouse event.  Mouse-up performs the
        # exact compositor once.
        positive = np.maximum(
            source_patch.astype(np.float32) - base_patch.astype(np.float32), 0.0
        )
        live_alpha = np.clip(alpha * validity, 0.0, 1.0)[..., None]
        shown[y0:y1, x0:x1] = np.clip(
            destination + positive * live_alpha, 0, 255
        ).astype(np.uint8)
        self.object_drag_live_box = box
        self.preview_frame_serial += 1
        self._present_preview_image(shown, True, False)

    def _clear_live_object_drag(self) -> None:
        self.object_drag_live_source = None
        self.object_drag_live_background = None
        self.object_drag_live_frame = None
        self.object_drag_live_box = None
        self.object_drag_live_settings = None
        self.object_drag_last_render = 0.0

    def _incremental_selected_object_image(self, before: Stroke) -> np.ndarray | None:
        if self.selected_object is None:
            return None
        return self._incremental_recomposed_object_image(
            self.selected_object, before, include_selected=True
        )

    def _incremental_parameter_change_image(self, before: Stroke) -> np.ndarray | None:
        adjusted = self._incremental_adjusted_object_image(before)
        if adjusted is not None:
            return adjusted
        return self._incremental_selected_object_image(before)

    def _incremental_adjusted_object_image(
        self, before: Stroke, reference: tuple[str, int] | None = None,
    ) -> np.ndarray | None:
        """Apply one object's parameter delta without rebuilding overlapping meteors.

        Brightness/cleanup/feather sliders do not change scene topology.  Rebuilding
        a clean ROI forced every overlapping source layer back into memory and could
        fall through to a full-project render.  Isolated before/after composites
        provide the selected object's pixel delta while preserving every other
        already-composited meteor in the visible canvas.
        """
        self.last_incremental_box = None
        reference = reference or self.selected_object
        if reference is None or self.preview_base is None:
            return None
        key, index = reference
        values = self.strokes.get(key, [])
        if not (0 <= index < len(values)) or key not in self.pairs:
            return None
        current = values[index]
        # global_preview_rgb is always the unexposed composite. Prefer it even
        # in per-image output mode: exact_preview_full_rgb may already contain
        # the user's base-exposure display adjustment, and adding an unexposed
        # meteor delta to that image would produce the wrong pixels.
        cached = self.global_preview_rgb
        if cached is None:
            if abs(self.base_exposure_tenths.get()) >= 1:
                return None
            cached = self.exact_preview_full_rgb if self.exact_preview_full_rgb is not None else self.preview_rgb
        if cached is None or cached.shape[:2] != self.preview_base.shape[:2]:
            return None
        try:
            preview_h, preview_w = self.preview_base.shape[:2]
            if self.current_path is not None and key == str(self.current_path):
                full_width = self.current_dims[0]
                aligned = self.preview_aligned_source if self.preview_aligned_source is not None else self.preview_source
                original = self.preview_original_source if self.preview_original_source is not None else self.preview_source
            else:
                aligned_path = Path(key)
                original_path = self.original_sources.get(key, aligned_path)
                full_width = image_info(self._effective_source_path(aligned_path))[0]
                aligned = self._cached_layer_preview(aligned_path, preview_w, preview_h)
                original = self._cached_layer_preview(original_path, preview_w, preview_h)
                # Selecting an old layer can outlive the rolling preview cache.
                # Decode only that layer once; never launch a full composite merely
                # because the user moved a per-meteor slider.
                if aligned is None:
                    aligned = self._cached_full_image(aligned_path, False)
                if original is None:
                    original = self._cached_full_image(original_path, False)
            if aligned is None or original is None:
                return None
            if aligned.shape[:2] != (preview_h, preview_w):
                aligned = cv2.resize(aligned, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            if original.shape[:2] != (preview_h, preview_w):
                original = cv2.resize(original, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

            auto_enabled = bool(self.adjustment_defaults.get("auto_optimize", True))
            old_scaled = auto_optimized_stroke(self._preview_clone(before, full_width), auto_enabled)
            new_scaled = auto_optimized_stroke(self._preview_clone(current, full_width), auto_enabled)
            boxes = []
            for prepared in (old_scaled, new_scaled):
                box = stroke_annotation_box(prepared, preview_w, preview_h)
                if box is not None:
                    boxes.append(box)
            if not boxes:
                return cached.copy()
            margin = max(old_scaled.feather, new_scaled.feather, 3) * 2 + 6
            x0 = max(0, min(box[0] for box in boxes) - margin)
            y0 = max(0, min(box[1] for box in boxes) - margin)
            x1 = min(preview_w, max(box[2] for box in boxes) + margin)
            y1 = min(preview_h, max(box[3] for box in boxes) + margin)
            self.last_incremental_box = (x0, y0, x1, y1)
            crop_w, crop_h = x1 - x0, y1 - y0
            old_local = stroke_for_image_crop(old_scaled, preview_w, preview_h, x0, y0, crop_w, crop_h)
            new_local = stroke_for_image_crop(new_scaled, preview_w, preview_h, x0, y0, crop_w, crop_h)
            base_crop = self.preview_base[y0:y1, x0:x1]
            settings = self._object_composite_settings(key)
            old_layer, _ = compose_meteor_sources(
                aligned[y0:y1, x0:x1], original[y0:y1, x0:x1],
                base_crop.copy(), [old_local], *settings,
            )
            new_layer, _ = compose_meteor_sources(
                aligned[y0:y1, x0:x1], original[y0:y1, x0:x1],
                base_crop.copy(), [new_local], *settings,
            )
            # This cached image is the committed composite state. Updating its
            # small ROI in place avoids copying an 8K frame on every slider tick.
            result = cached
            destination = result[y0:y1, x0:x1].astype(np.int16)
            delta = new_layer.astype(np.int16) - old_layer.astype(np.int16)
            result[y0:y1, x0:x1] = np.clip(destination + delta, 0, 255).astype(np.uint8)
            return result
        except Exception:
            self.last_incremental_delete_error = traceback.format_exc()
            return None

    def _incremental_deleted_object_image(
        self, reference: tuple[str, int], deleted: Stroke
    ) -> np.ndarray | None:
        return self._incremental_recomposed_object_image(
            reference, deleted, include_selected=False
        )

    def _incremental_recomposed_object_image(
        self, reference: tuple[str, int], before: Stroke,
        include_selected: bool,
    ) -> np.ndarray | None:
        """Exactly rebuild the union of one object's old and new footprints."""
        self.last_incremental_delete_error = None
        self.last_incremental_box = None
        shared = self._uses_shared_base()
        cached_display = self.global_preview_rgb
        if shared and cached_display is None:
            cached_display = (
                self.exact_preview_full_rgb
                if self.exact_preview_full_rgb is not None else self.preview_rgb
            )
        elif not shared:
            cached_display = (
                self.exact_preview_full_rgb
                if self.exact_preview_full_rgb is not None else self.preview_rgb
            )
        if (
            self.preview_base is None
            or cached_display is None
            or reference[0] not in self.pairs
            or (not shared and (
                self.current_path is None or reference[0] != str(self.current_path)
            ))
        ):
            return None
        key, deleted_index = reference
        try:
            if self.current_path is not None and key == str(self.current_path):
                full_width = self.current_dims[0]
            else:
                full_width = image_info(self._effective_source_path(Path(key)))[0]
            auto_enabled = bool(self.adjustment_defaults.get("auto_optimize", True))
            scaled_before = auto_optimized_stroke(
                self._preview_clone(before, full_width), auto_enabled
            )
            transformed = transformed_object_crop(self.preview_base, scaled_before)
            if transformed is None:
                return cached_display.copy()
            _patch, _old_alpha, _validity, old_box = transformed
            preview_h, preview_w = self.preview_base.shape[:2]
            affected_boxes = [old_box]

            values_for_key = self.strokes.get(key, [])
            if include_selected and 0 <= deleted_index < len(values_for_key):
                current = values_for_key[deleted_index]
                transformed_current = transformed_object_crop(
                    self.preview_base, auto_optimized_stroke(
                        self._preview_clone(current, full_width), auto_enabled
                    )
                )
                if transformed_current is not None:
                    _new_patch, _new_alpha, _new_validity, new_box = transformed_current
                    affected_boxes.append(new_box)
            x0 = min(box[0] for box in affected_boxes)
            y0 = min(box[1] for box in affected_boxes)
            x1 = max(box[2] for box in affected_boxes)
            y1 = max(box[3] for box in affected_boxes)
            self.last_incremental_box = (x0, y0, x1, y1)

            # Gather only objects whose footprints intersect the deleted one.
            # They are recomposited in original source order over a clean base;
            # the rest of the cached sky is never touched.
            intersecting: list[tuple[str, int, list[Stroke]]] = []
            for other_key, values in self.strokes.items():
                if other_key not in self.pairs:
                    continue
                if not shared and other_key != key:
                    continue
                try:
                    other_width = (
                        self.current_dims[0]
                        if self.current_path is not None and other_key == str(self.current_path)
                        else image_info(self._effective_source_path(Path(other_key)))[0]
                    )
                except Exception:
                    return None
                relevant_positive_indices = []
                for other_index, other in enumerate(values):
                    if (
                        ((other_key, other_index) == (key, deleted_index)
                         and not include_selected)
                        or other.erase or not other.points
                    ):
                        continue
                    box = stroke_annotation_box(
                        auto_optimized_stroke(
                            self._preview_clone(other, other_width), auto_enabled
                        ), preview_w, preview_h
                    )
                    if box is None:
                        continue
                    ox0, oy0, ox1, oy1 = box
                    if ox0 < x1 and ox1 > x0 and oy0 < y1 and oy1 > y0:
                        relevant_positive_indices.append(other_index)
                if relevant_positive_indices:
                    # Preserve chronological erase semantics for the relevant
                    # positive objects, but retain only erasers that intersect
                    # this ROI; distant erasers may normalize outside the crop.
                    relevant = []
                    for item_index, item in enumerate(values):
                        if item_index in relevant_positive_indices:
                            relevant.append(item)
                            continue
                        if not item.erase or not item.points:
                            continue
                        erase_box = stroke_annotation_box(
                            self._preview_clone(item, other_width), preview_w, preview_h
                        )
                        if erase_box is None:
                            continue
                        ex0, ey0, ex1, ey1 = erase_box
                        if ex0 < x1 and ex1 > x0 and ey0 < y1 and ey1 > y0:
                            relevant.append(item)
                    intersecting.append((other_key, other_width, relevant))

            image = cached_display.copy()
            replacement = self.preview_base[y0:y1, x0:x1].copy()
            crop_h, crop_w = replacement.shape[:2]
            for other_key, other_width, relevant in intersecting:
                if self.current_path is not None and other_key == str(self.current_path):
                    aligned_preview = (
                        self.preview_aligned_source if self.preview_aligned_source is not None
                        else self.preview_source
                    )
                    original_preview = (
                        self.preview_original_source if self.preview_original_source is not None
                        else self.preview_source
                    )
                else:
                    aligned_path = Path(other_key)
                    original_path = self.original_sources.get(other_key, aligned_path)
                    aligned_preview = self._cached_layer_preview(
                        aligned_path, preview_w, preview_h
                    )
                    original_preview = self._cached_layer_preview(
                        original_path, preview_w, preview_h
                    )
                    # Mouse-up must never block on full TIFF decoding.  A cache
                    # miss falls back to the normal background preview worker.
                    if aligned_preview is None or original_preview is None:
                        return None
                if aligned_preview is None or original_preview is None:
                    return None
                scaled_relevant = [
                    self._preview_clone(item, other_width) for item in relevant
                ]
                cropped_relevant = [
                    stroke_for_image_crop(
                        item, preview_w, preview_h, x0, y0, crop_w, crop_h
                    ) for item in scaled_relevant
                ]
                replacement, _mask = compose_meteor_sources(
                    aligned_preview[y0:y1, x0:x1],
                    original_preview[y0:y1, x0:x1],
                    replacement, cropped_relevant,
                    *self._object_composite_settings(other_key),
                )
            image[y0:y1, x0:x1] = replacement
            return image
        except Exception:
            self.last_incremental_delete_error = traceback.format_exc()
            return None

    def _object_pointer_move(self, event):
        stroke = self._selected_stroke()
        original = self.object_drag_original
        start = self.object_drag_start
        if stroke is None or original is None or start is None or self.selected_object is None:
            return "break"
        geometry = self._object_canvas_geometry(self.selected_object, original)
        if geometry is None:
            return "break"
        cursor = np.asarray((float(event.x), float(event.y)))
        start_point = np.asarray(start)
        center = geometry["center"]
        mode = self.object_drag_mode or "move"
        if mode != "move" and float(np.hypot(*(cursor - start_point))) < 4.0:
            return "break"
        if mode == "move":
            delta = (cursor - start_point) / geometry["display_scale"]
            stroke.offset_x = original.offset_x + float(delta[0])
            stroke.offset_y = original.offset_y + float(delta[1])
        elif mode == "rotate":
            first = float(np.arctan2(start_point[1] - center[1], start_point[0] - center[0]))
            current = float(np.arctan2(cursor[1] - center[1], cursor[0] - center[0]))
            delta = np.rad2deg(current - first)
            stroke.rotation = original.rotation + float((delta + 180.0) % 360.0 - 180.0)
        elif mode.startswith("length"):
            start_distance = abs(float((start_point - center) @ geometry["axis"]))
            current_distance = abs(float((cursor - center) @ geometry["axis"]))
            stroke.length_scale = max(0.05, original.length_scale * current_distance / max(3.0, start_distance))
        elif mode.startswith("width"):
            start_distance = abs(float((start_point - center) @ geometry["normal"]))
            current_distance = abs(float((cursor - center) @ geometry["normal"]))
            stroke.width_scale = max(0.05, original.width_scale * current_distance / max(3.0, start_distance))
        elif mode.startswith("scale"):
            start_distance = float(np.hypot(*(start_point - center)))
            current_distance = float(np.hypot(*(cursor - center)))
            ratio = current_distance / max(3.0, start_distance)
            stroke.length_scale = max(0.05, original.length_scale * ratio)
            stroke.width_scale = max(0.05, original.width_scale * ratio)
        self._render_live_object_drag()
        if self.object_drag_live_source is None:
            self._draw_selected_object_overlay()
        return "break"

    def _object_pointer_end(self, _event=None):
        stroke = self._selected_stroke()
        before = self.object_drag_original
        live_visual = (
            self.preview_rgb
            if self.object_drag_live_source is not None and self.preview_rgb is not None
            else None
        )
        incremental = None
        if stroke is not None and before is not None and stroke != before:
            incremental = self._incremental_selected_object_image(before)
        self.object_drag_mode = None
        self.object_drag_start = None
        self.object_drag_original = None
        self._clear_live_object_drag()
        if stroke is not None and before is not None and stroke != before:
            self._record_object_transform(before)
            if incremental is not None:
                self._commit_incremental_global_preview(incremental, validate=False)
                self.status.set("流星局部已精确更新；未重绘画面其他区域")
            else:
                if self._uses_shared_base() and live_visual is not None:
                    # Keep the last responsive drag frame visible while the exact
                    # background worker catches up; never flash back to the old spot.
                    self.global_preview_rgb = live_visual
                    self.global_labeled_preview_rgb = live_visual
                self._invalidate_global_preview()
                self._render_preview()
                self.status.set("流星变换已应用；最终效果与来源标注已同步更新")
        else:
            self._draw_selected_object_overlay()
        return "break"

    def _record_object_transform(self, before: Stroke) -> None:
        if self.selected_object is None:
            return
        key, index = self.selected_object
        stroke = self._selected_stroke()
        if stroke is None:
            return
        self._sync_matching_candidate(key, stroke)
        after = replace(stroke, points=stroke.points.copy())
        self._record_edit(key, ("transform", index, (before, after)))

    def _matching_candidate(self, key: str, stroke: Stroke) -> tuple[int, Stroke] | None:
        for index, candidate in enumerate(self.candidates.get(key, [])):
            if candidate.auto_score == stroke.auto_score and candidate.points == stroke.points:
                return index, candidate
        return None

    def _sync_matching_candidate(self, key: str, stroke: Stroke) -> None:
        match = self._matching_candidate(key, stroke)
        if match is None:
            return
        _index, candidate = match
        candidate.offset_x = stroke.offset_x
        candidate.offset_y = stroke.offset_y
        candidate.rotation = stroke.rotation
        candidate.length_scale = stroke.length_scale
        candidate.width_scale = stroke.width_scale
        candidate.opacity = stroke.opacity
        candidate.feather = stroke.feather
        candidate.brightness_override = stroke.brightness_override
        candidate.background_cleanup_override = stroke.background_cleanup_override
        candidate.saturation_override = stroke.saturation_override
        candidate.preserve_brightness_override = stroke.preserve_brightness_override
        candidate.match_exposure_override = stroke.match_exposure_override
        candidate.blend_mode_override = stroke.blend_mode_override
        candidate.auto_blend_enabled = stroke.auto_blend_enabled
        candidate.auto_strength = stroke.auto_strength
        candidate.auto_black_point = stroke.auto_black_point
        candidate.auto_cleanup = stroke.auto_cleanup
        candidate.auto_brightness = stroke.auto_brightness
        candidate.auto_feather = stroke.auto_feather
        candidate.source_mode = stroke.source_mode

    def _invalidate_global_preview(self) -> None:
        self.global_preview_signature = None
        self.global_preview_generation = getattr(self, "global_preview_generation", 0) + 1
        self.exact_preview_generation = getattr(self, "exact_preview_generation", 0) + 1
        if self.global_exact_after_id is not None:
            try:
                self.after_cancel(self.global_exact_after_id)
            except tk.TclError:
                pass
            self.global_exact_after_id = None

    def _global_annotations_from_state(self) -> list[dict]:
        if self.preview_base is None:
            return []
        preview_height, preview_width = self.preview_base.shape[:2]
        annotations = []
        for key, values in self.strokes.items():
            if key not in self.pairs:
                continue
            if not any(not item.erase and item.points for item in values):
                continue
            try:
                if self.current_path is not None and key == str(self.current_path):
                    full_width = self.current_dims[0]
                else:
                    full_width = image_info(self._effective_source_path(Path(key)))[0]
            except Exception:
                full_width = self.current_dims[0]
            scaled = [self._preview_clone(item, full_width) for item in values]
            annotations.extend(meteor_source_annotations(
                Path(key).stem, scaled, preview_width, preview_height,
                key in self.use_original_sources,
            ))
        return annotations

    def _commit_incremental_global_preview(
        self, image: np.ndarray, validate: bool = False,
        realtime: bool = False, dirty_box: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Commit the one-object live result without blanking/rebuilding the full sky."""
        self.global_preview_rgb = image
        self.global_labeled_preview_rgb = None
        if self.view_mode.get() == "labeled" and not realtime:
            self.global_labeled_preview_rgb, _records = annotate_meteor_sources(
                image, self._global_annotations_from_state()
            )
        signature = self._global_preview_state_signature()
        self.global_preview_signature = signature
        # Supersede an older full-composite pass. It will stop at the next
        # source boundary instead of finishing stale work and repainting twice.
        self.global_preview_generation = getattr(self, "global_preview_generation", 0) + 1
        # The main canvas stores an 8-bit display of the 16-bit exact result.
        # A one-object recomposition already updates that display at full source
        # resolution, so promote it to the current exact-view state instead of
        # invalidating and recomputing every meteor twice. Export still performs
        # the complete 16-bit pipeline from source files.
        if (
            self.preview_base is not None
            and image.shape[:2] == self.preview_base.shape[:2]
        ):
            exposure_ev = self.base_exposure_tenths.get() / 10.0
            if abs(exposure_ev) < 1e-6:
                exact_image = image
            elif (
                dirty_box is not None
                and self.exact_preview_full_rgb is not None
                and self.exact_preview_full_rgb.shape == image.shape
            ):
                # The prior exact frame is still correct outside the edited
                # footprint. Re-expose only the changed pixels instead of
                # recompositing (or even re-exposing) the complete 8K canvas.
                exact_image = self.exact_preview_full_rgb.copy()
                x0, y0, x1, y1 = dirty_box
                exact_image[y0:y1, x0:x1] = adjust_composite_base_exposure(
                    image[y0:y1, x0:x1], self.preview_base[y0:y1, x0:x1], exposure_ev
                )
            else:
                exact_image = adjust_composite_base_exposure(
                    image, self.preview_base, exposure_ev
                )
            self.exact_preview_full_rgb = exact_image
            self.exact_preview_rgb = exact_image
            if realtime:
                # Labels are a derived diagnostic view. Do not regenerate the
                # full labeled frame for every slider tick; it will be rebuilt
                # only if the user actually opens that view.
                self.exact_labeled_preview_full_rgb = None
                self.exact_labeled_preview_rgb = None
            else:
                annotations = self._global_annotations_from_state()
                labeled, _records = annotate_meteor_sources(exact_image, annotations)
                self.exact_labeled_preview_full_rgb = labeled.astype(np.uint8, copy=False)
                self.exact_labeled_preview_rgb = self.exact_labeled_preview_full_rgb
            self.exact_preview_signature = self._exact_preview_state_signature()
            self.exact_preview_pending_signature = None
            if self.exact_preview_request_after_id is not None:
                try:
                    self.after_cancel(self.exact_preview_request_after_id)
                except tk.TclError:
                    pass
                self.exact_preview_request_after_id = None
            self.exact_preview_status.set("精准预览：当前流星已局部更新")
            # Stop an older full-frame worker at its next meteor boundary. Its
            # stale partial/final frames must never repaint this local result.
            self.exact_preview_generation = getattr(self, "exact_preview_generation", 0) + 1
        display_image = self.exact_preview_full_rgb if self.exact_preview_full_rgb is not None else image
        if realtime and self.view_mode.get() == "labeled" and dirty_box is not None:
            labeled = self.preview_rgb
            if labeled is None or labeled.shape != image.shape:
                labeled = image.copy()
            x0, y0, x1, y1 = dirty_box
            labeled[y0:y1, x0:x1] = display_image[y0:y1, x0:x1]
            self.global_labeled_preview_rgb = labeled
            self.exact_labeled_preview_full_rgb = labeled
            self.exact_labeled_preview_rgb = labeled
            display_image = labeled
            self._schedule_realtime_label_refresh()
        if not (
            realtime and dirty_box is not None
            and self._paste_realtime_preview_patch(display_image, dirty_box)
        ):
            self._render_preview()
        if validate:
            self._schedule_global_exact_validation(signature)

    def _paste_realtime_preview_patch(
        self, image: np.ndarray, dirty_box: tuple[int, int, int, int],
    ) -> bool:
        """Paste one changed image ROI into the existing Tk photo without a redraw."""
        if (
            self.view_mode.get() not in {"blend", "labeled"}
            or self.preview_photo is None
            or self.preview_rgb is None
            or image.shape != self.preview_rgb.shape
            or self.preview_display_size is None
        ):
            return False
        try:
            height, width = image.shape[:2]
            canvas_w = max(10, self.canvas.winfo_width())
            canvas_h = max(10, self.canvas.winfo_height())
            origin_x, origin_y = self._canvas_view_origin()
            crop_x0 = max(0, int(np.floor(origin_x)))
            crop_y0 = max(0, int(np.floor(origin_y)))
            crop_x1 = min(width, int(np.ceil(origin_x + canvas_w / self.canvas_zoom)))
            crop_y1 = min(height, int(np.ceil(origin_y + canvas_h / self.canvas_zoom)))
            x0, y0, x1, y1 = dirty_box
            # Include a couple of display pixels so resampling cannot leave a seam.
            pad = max(2, int(np.ceil(2.0 / max(self.canvas_zoom, 1e-6))))
            x0, y0 = max(crop_x0, x0 - pad), max(crop_y0, y0 - pad)
            x1, y1 = min(crop_x1, x1 + pad), min(crop_y1, y1 + pad)
            if x1 <= x0 or y1 <= y0:
                self.preview_rgb = image
                return True
            dx0 = int(round((x0 - crop_x0) * self.canvas_zoom))
            dy0 = int(round((y0 - crop_y0) * self.canvas_zoom))
            dx1 = int(round((x1 - crop_x0) * self.canvas_zoom))
            dy1 = int(round((y1 - crop_y0) * self.canvas_zoom))
            if dx1 <= dx0 or dy1 <= dy0:
                return False
            patch = Image.fromarray(image[y0:y1, x0:x1])
            if patch.size != (dx1 - dx0, dy1 - dy0):
                patch = patch.resize(
                    (dx1 - dx0, dy1 - dy0),
                    Image.Resampling.LANCZOS if self.canvas_zoom < 1.0 else Image.Resampling.BILINEAR,
                )
            self.preview_rgb = image
            self.preview_frame_serial += 1
            self.viewport_cache.clear()
            patch_photo = ImageTk.PhotoImage(patch, master=self.canvas)
            self.canvas.tk.call(
                str(self.preview_photo), "copy", str(patch_photo),
                "-from", 0, 0, dx1 - dx0, dy1 - dy0,
                "-to", dx0, dy0,
            )
            self._draw_selected_object_overlay()
            return True
        except (tk.TclError, ValueError):
            return False

    def _schedule_realtime_label_refresh(self) -> None:
        if self.realtime_label_after_id is not None:
            try:
                self.after_cancel(self.realtime_label_after_id)
            except tk.TclError:
                pass

        def refresh() -> None:
            self.realtime_label_after_id = None
            if self.global_preview_rgb is None or self.view_mode.get() != "labeled":
                return
            labeled, _records = annotate_meteor_sources(
                self.global_preview_rgb, self._global_annotations_from_state()
            )
            self.global_labeled_preview_rgb = labeled
            self.exact_labeled_preview_full_rgb = labeled
            self.exact_labeled_preview_rgb = labeled
            self._render_preview()

        self.realtime_label_after_id = self.after(420, refresh)

    def _schedule_global_exact_validation(self, signature: str) -> None:
        if self.global_exact_after_id is not None:
            try:
                self.after_cancel(self.global_exact_after_id)
            except tk.TclError:
                pass

        def validate() -> None:
            self.global_exact_after_id = None
            if (
                not self._uses_shared_base()
                or signature != self._global_preview_state_signature()
                or self.global_preview_signature != signature
            ):
                return
            # Keep the incremental image visible while one exact background
            # verification is coalesced and calculated off the UI thread.
            self.global_preview_signature = None
            self._render_preview()

        self.global_exact_after_id = self.after(1200, validate)

    def _clear_object_selection(self) -> None:
        self._clear_live_object_drag()
        self.selected_object = None
        self.object_drag_mode = None
        self.object_drag_start = None
        self.object_drag_original = None
        for item in self.object_overlay_items:
            self.canvas.delete(item)
        self.object_overlay_items = []
        self.object_handle_centers = {}
        if hasattr(self, "selected_object_controls"):
            self._load_selected_object_adjustments()

    def _delete_selected_shortcut(self, event):
        widget_class = event.widget.winfo_class() if event.widget else ""
        if widget_class in {"Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox"}:
            return None
        return self._delete_selected_object()

    def _delete_selected_object(self):
        if self.view_mode.get() not in {"blend", "labeled"} or self.selected_object is None:
            return None
        key, index = self.selected_object
        values = self.strokes.get(key, [])
        if not (0 <= index < len(values)):
            self._clear_object_selection()
            return "break"
        before = replace(values[index], points=values[index].points.copy())
        incremental = self._incremental_deleted_object_image((key, index), before)
        stroke = values.pop(index)
        candidate_match = self._matching_candidate(key, stroke)
        if candidate_match is None:
            self._record_edit(key, ("delete", index, stroke))
        else:
            candidate_index, candidate = candidate_match
            self.candidates[key].pop(candidate_index)
            self._record_edit(key, ("delete_object", index, (stroke, candidate_index, candidate)))
        self._clear_object_selection()
        self._update_tree_status_for_key(key)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
            self.status.set("已即时删除这颗流星；未重算其他区域，可用 Ctrl/Command+Z 恢复")
        else:
            self._invalidate_global_preview()
            self._render_preview()
            self.status.set("局部合成缓存不可用，正在精确更新；可用 Ctrl/Command+Z 恢复")
        return "break"

    def _reset_selected_object(self) -> None:
        stroke = self._selected_stroke()
        if stroke is None:
            self.status.set("没有可重置的流星")
            return
        before = replace(stroke, points=stroke.points.copy())
        reset_stroke_geometry(stroke)
        if stroke != before:
            incremental = self._incremental_selected_object_image(before)
            self._record_object_transform(before)
            if incremental is not None:
                self._commit_incremental_global_preview(incremental, validate=False)
            else:
                self._invalidate_global_preview()
                self._render_preview()
        self.status.set("已恢复这颗流星的原始位置和形态；蒙版及融合参数保持不变")

    def _transform_selected_object(self) -> None:
        if self.selected_object is not None:
            self._open_transform_dialog(*self.selected_object)

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
            transformed = transformed_stroke_points(stroke, *self.current_dims)
            points = [(x0 + px * (x1 - x0), y0 + py * (y1 - y0)) for px, py in transformed]
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
        if self.view_mode.get() in {"blend", "labeled"}:
            reference = self._find_object_near(event.x, event.y)
            if reference is not None:
                self.selected_object = reference
            if self._selected_stroke() is None:
                self.status.set("此处附近没有可编辑的流星")
                self._draw_selected_object_overlay()
                return "break"
            self._draw_selected_object_overlay()
            try:
                self.object_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.object_menu.grab_release()
            return "break"
        if self.view_mode.get() != "source":
            self.status.set("干净 JPG 仅用于检查底图；请在 1 原图 TIFF 中编辑蒙版")
            return "break"
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
        if self.view_mode.get() in {"blend", "labeled"}:
            reference = self._find_object_near(event.x, event.y)
            if reference is not None:
                self.selected_object = reference
                return self._delete_selected_object()
            return "break"
        if self.view_mode.get() != "source":
            return "break"
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
        self.last_edit_key = key
        self._schedule_autosave()

    @staticmethod
    def _resolve_transform_action_index(
        values: list[Stroke], recorded_index: int, expected: Stroke
    ) -> int | None:
        """Find the transformed object even if later insertions shifted its index."""
        if 0 <= recorded_index < len(values) and values[recorded_index] == expected:
            return recorded_index
        return next((i for i, item in enumerate(values) if item == expected), None)

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
        key = str(self.current_path)
        index = self.context_stroke_index
        values = self.strokes.get(key, [])
        if not (0 <= index < len(values)) or values[index].erase:
            self.context_stroke_index = None
            return
        self.context_stroke_index = None
        if self.context_highlight is not None:
            self.canvas.delete(self.context_highlight)
            self.context_highlight = None
        self.selected_object = (key, index)
        self.view_mode.set("blend")
        self._render_preview()
        self._load_selected_object_adjustments()
        self._draw_selected_object_overlay()
        scope = "总融合预览" if self._uses_shared_base() else "当前图融合预览"
        self.status.set(
            f"已进入{scope}并选中流星：拖动主体移动，拖动手柄旋转或拉伸；"
            "右键可输入精确数值"
        )

    def _open_transform_dialog(self, object_key: str, index: int) -> None:
        values = self.strokes.get(object_key, [])
        if not (0 <= index < len(values)) or values[index].erase:
            return
        stroke = values[index]
        before = replace(stroke, points=stroke.points.copy())
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
                    reset_stroke_geometry(stroke)
                else:
                    stroke.offset_x = float(variables["offset_x"].get())
                    stroke.offset_y = float(variables["offset_y"].get())
                    stroke.rotation = float(variables["rotation"].get())
                    stroke.length_scale = max(0.05, float(variables["length_scale"].get()))
                    stroke.width_scale = max(0.05, float(variables["width_scale"].get()))
                    stroke.opacity = float(np.clip(float(variables["opacity"].get()) / 100.0, 0.0, 1.0))
            except (ValueError, tk.TclError):
                show_copyable_error(APP_NAME, "变换参数无效", parent=dialog)
                return
            self.context_stroke_index = None
            incremental = None
            if stroke != before:
                if self.selected_object == (object_key, index):
                    incremental = self._incremental_selected_object_image(before)
                self._sync_matching_candidate(object_key, stroke)
                after = replace(stroke, points=stroke.points.copy())
                self._record_edit(object_key, ("transform", index, (before, after)))
            if incremental is not None:
                self._commit_incremental_global_preview(incremental, validate=False)
            else:
                self._invalidate_global_preview()
                self._render_preview()
            dialog.destroy()

        ttk.Button(buttons, text="恢复真实位置", command=lambda: apply_values(True)).pack(side="left")
        ttk.Button(buttons, text="取消", command=dialog.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="应用", command=apply_values).pack(side="left")

    def _stroke_start(self, event) -> None:
        if not self.current_path:
            return
        if getattr(self, "space_pan_held", False):
            return self._canvas_pan_start_event(event, with_left=True)
        if self.view_mode.get() in {"blend", "labeled"}:
            return self._object_pointer_start(event)
        if self.view_mode.get() != "source":
            self.status.set("干净 JPG 是只读检查视图；请切换到 1 原图 TIFF 绘制蒙版")
            return "break"
        # Defensive Photoshop-style transaction boundary. If Windows/Tk lost the
        # previous mouse-up while the preview was repainting, never replace that
        # live stroke: commit it before beginning the next one.
        if getattr(self, "active_points", []):
            self._commit_active_stroke()
        # Alt/Option key-up can be swallowed by the OS menu/focus handling. If
        # the next mouse-down no longer carries Mod1, restore the visible tool
        # before deciding whether this stroke paints or erases.
        alt_is_physically_held = bool(int(getattr(event, "state", 0)) & 0x0008)
        if getattr(self, "alt_previous_mode", None) is not None and not alt_is_physically_held:
            self._restore_alt_tool()
        # The floating candidate button is handled only by the canvas, so selecting a
        # candidate and painting can never compete for the same press.
        if self.hover_candidate_items and point_in_padded_bbox(
            float(event.x), float(event.y), self.canvas.bbox("candidate_pick"), padding=2.0
        ):
            self._pick_hover_candidate(event)
            return "break"
        point = self._event_normalized(event)
        if point is None:
            return
        self._clear_candidate_hover()
        anchor = self.shift_anchors.get(str(self.current_path))
        self.active_shift_line = bool(event.state & 0x0001) and anchor is not None
        self.active_points = [anchor, point] if self.active_shift_line else [point]
        self.active_tool_mode = self.edit_mode.get()
        self.active_tool_width = self._tool_width()
        self.active_tool_feather = int(self.feather.get())
        self.active_canvas_line = None
        self.cursor_position = (float(event.x), float(event.y))
        try:
            self.canvas.grab_set()
        except (tk.TclError, AttributeError):
            pass
        values = self.strokes.setdefault(str(self.current_path), [])
        self.active_action_index = len(values)
        if self.active_tool_mode == "erase":
            self.live_erase_stroke = Stroke(
                self.active_points.copy(), self.active_tool_width,
                self.active_tool_feather, True,
                source_mode=self._current_source_mode(),
            )
            self._refresh_live_mask(force=True)
        else:
            self.live_erase_stroke = None
            # Paint behaves like a regular bitmap brush: draw a lightweight
            # foreground trace while dragging, then union it into the mask once
            # on mouse-up.  A Shift connection is visible immediately.
            if self.active_shift_line:
                self._draw_active_stroke()
        # Left-button painting owns this event completely. Do not let it fall
        # through to a second canvas binding that can reinterpret the same
        # press using platform-specific modifier state.
        return "break"

    def _draw_active_stroke(self) -> None:
        if not self.active_points:
            return
        x0, y0, x1, y1 = self.display_box
        coords = []
        for x, y in self.active_points:
            coords.extend((x0 + x * (x1 - x0), y0 + y * (y1 - y0)))
        color = "#64d2ff" if self.active_tool_mode == "erase" else "#ff3b30"
        active_width = self.active_tool_width or self._tool_width()
        shown_width = max(3, int(active_width * (x1 - x0) / max(1, self.current_dims[0])))
        if len(coords) == 2:
            radius = shown_width / 2.0
            oval_coords = (
                coords[0] - radius, coords[1] - radius,
                coords[0] + radius, coords[1] + radius,
            )
            if self.active_canvas_line and self.canvas.type(self.active_canvas_line) == "oval":
                self.canvas.coords(self.active_canvas_line, *oval_coords)
                self.canvas.itemconfigure(self.active_canvas_line, fill=color)
            else:
                if self.active_canvas_line:
                    self.canvas.delete(self.active_canvas_line)
                self.active_canvas_line = self.canvas.create_oval(
                    *oval_coords, fill=color, outline="",
                )
        else:
            if self.active_canvas_line and self.canvas.type(self.active_canvas_line) == "line":
                self.canvas.coords(self.active_canvas_line, *coords)
                self.canvas.itemconfigure(
                    self.active_canvas_line, fill=color, width=shown_width
                )
            else:
                if self.active_canvas_line:
                    self.canvas.delete(self.active_canvas_line)
                self.active_canvas_line = self.canvas.create_line(
                    *coords, fill=color, width=shown_width,
                    capstyle="round", joinstyle="round",
                )

    def _refresh_live_mask(self, force: bool = False) -> None:
        if self.live_erase_stroke is None:
            return
        now = time.monotonic()
        if force or now - self.last_live_render >= 0.012:
            self.last_live_render = now
            # Erasing uses the same lightweight foreground trace as painting.
            # Rebuilding the feathered full-frame mask on every pointer event made
            # the brush visibly lag; the actual local mask is committed on release.
            self._draw_active_stroke()

    def _tool_width(self) -> int:
        return int(self.eraser_width.get() if self.edit_mode.get() == "erase" else self.brush_width.get())

    def _stroke_move(self, event) -> None:
        if getattr(self, "canvas_pan_with_left", False):
            return self._canvas_pan_move_event(event)
        if self.object_drag_mode is not None:
            return self._object_pointer_move(event)
        if not self.active_points:
            return
        point = self._event_normalized(event)
        if point is None:
            return
        self.cursor_position = (float(event.x), float(event.y))
        if self.active_shift_line:
            self.active_points[-1] = point
        else:
            x0, y0, x1, y1 = self.display_box
            last = self.active_points[-1]
            screen_distance = np.hypot(
                (point[0] - last[0]) * (x1 - x0),
                (point[1] - last[1]) * (y1 - y0),
            )
            if screen_distance < 1.5:
                return "break"
            self.active_points.append(point)
        if self.live_erase_stroke is not None:
            self.live_erase_stroke.points = self.active_points.copy()
            self._refresh_live_mask()
        else:
            self._draw_active_stroke()
        return "break"

    def _stroke_end(self, event) -> None:
        if getattr(self, "canvas_pan_with_left", False):
            return self._canvas_pan_end_event(event)
        if self.object_drag_mode is not None:
            return self._object_pointer_end(event)
        if not self.active_points or not self.current_path:
            return
        point = self._event_normalized(event)
        if point is not None:
            if self.active_shift_line:
                self.active_points[-1] = point
            elif not self.active_points or np.hypot(point[0] - self.active_points[-1][0], point[1] - self.active_points[-1][1]) > 1e-6:
                self.active_points.append(point)
        self._commit_active_stroke()
        return "break"

    def _global_pointer_release(self, event) -> None:
        """Finish a canvas gesture even when mouse-up lands outside the canvas."""
        if (not self.active_points and self.object_drag_mode is None
                and not getattr(self, "canvas_pan_with_left", False)):
            return
        proxy = type("CanvasReleaseEvent", (), {})()
        proxy.x = int(event.x_root - self.canvas.winfo_rootx())
        proxy.y = int(event.y_root - self.canvas.winfo_rooty())
        self._stroke_end(proxy)

    def _commit_active_stroke(self) -> bool:
        if not self.active_points or not self.current_path:
            return False
        live_stroke = self.live_erase_stroke
        if live_stroke is not None:
            live_stroke.points = self.active_points.copy()
            stroke = live_stroke
            values = self.strokes.setdefault(str(self.current_path), [])
            self.active_action_index = len(values)
            values.append(stroke)
        else:
            stroke = Stroke(
                self.active_points.copy(), self.active_tool_width or self._tool_width(),
                self.active_tool_feather, False,
                source_mode=self._current_source_mode(),
            )
            values = self.strokes.setdefault(str(self.current_path), [])
            self.active_action_index = len(values)
            values.append(stroke)
        key = str(self.current_path)
        reference = (key, self.active_action_index)
        self._record_edit(key, ("add", self.active_action_index, stroke))
        # The new paint/eraser stroke changes only its own footprint. Updating
        # that region here keeps the combined preview cache exact and avoids a
        # full composite the next time the user switches to the blend view.
        incremental = None
        if not stroke.erase:
            # A new brush stroke is an isolated object delta: its "before"
            # state is simply the same object with no painted points. This is
            # exact for the affected ROI and does not require decoding every
            # overlapping meteor layer from the rolling cache.
            incremental = self._incremental_adjusted_object_image(
                replace(stroke, points=[]), reference
            )
        if incremental is None:
            incremental = self._incremental_recomposed_object_image(
                reference, stroke, include_selected=True
            )
        self.shift_anchors[str(self.current_path)] = self.active_points[-1]
        committed_mode = "橡皮擦" if stroke.erase else "画笔"
        if self.active_canvas_line:
            self.canvas.delete(self.active_canvas_line)
        self.active_points = []
        self.active_canvas_line = None
        self.active_shift_line = False
        self.active_tool_mode = None
        self.active_tool_width = 0
        self.active_tool_feather = 0
        self.live_erase_stroke = None
        self.active_action_index = -1
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass
        self._update_tree_status()
        self.show_mask.set(True)
        if incremental is not None:
            self._commit_incremental_global_preview(
                incremental, validate=False, dirty_box=self.last_incremental_box
            )
        else:
            self._render_preview()
        positive_count = sum(
            not item.erase for item in self.strokes.get(str(self.current_path), [])
        )
        self.status.set(
            f"{committed_mode}已提交并局部即时融合；当前累计 {positive_count} 笔画笔。"
            "红色区域为保留范围，按住 H 可临时隐藏"
        )
        self._schedule_autosave()
        return True

    def undo_stroke(self) -> None:
        if not self.current_path:
            return
        key = (
            self.last_edit_key
            if self.view_mode.get() in {"blend", "labeled"} and self.last_edit_key
            else str(self.current_path)
        )
        values = self.strokes.setdefault(key, [])
        history = self.edit_history.setdefault(key, [])
        incremental = None
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
            elif kind == "delete_object":
                stroke, candidate_index, candidate = payload
                values.insert(min(index, len(values)), stroke)
                pool = self.candidates.setdefault(key, [])
                pool.insert(min(candidate_index, len(pool)), candidate)
            elif kind == "clear":
                before, _after = payload
                values[:] = list(before)
            elif kind == "transform":
                before, after = payload
                target_index = self._resolve_transform_action_index(values, index, after)
                if target_index is None:
                    history.append(action)
                    self.status.set("无法定位要撤销的流星；已保留蒙版，没有执行删除")
                    return
                previous = replace(values[target_index], points=values[target_index].points.copy())
                values[target_index] = replace(before, points=before.points.copy())
                self._sync_matching_candidate(key, values[target_index])
                if self.selected_object == (key, index):
                    self.selected_object = (key, target_index)
                incremental = self._incremental_recomposed_object_image(
                    (key, target_index), previous, include_selected=True
                )
        else:
            self.status.set("当前没有可撤销的操作；蒙版保持不变")
            return
        self.edit_redo.setdefault(key, []).append(action)
        self._restore_shift_anchor()
        self._update_tree_status_for_key(key)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
        else:
            self._invalidate_global_preview()
            self._render_preview()
        self._schedule_autosave()

    def redo_stroke(self) -> None:
        if not self.current_path:
            return
        key = (
            self.last_edit_key
            if self.view_mode.get() in {"blend", "labeled"} and self.last_edit_key
            else str(self.current_path)
        )
        if not self.edit_redo.get(key):
            return
        action = self.edit_redo[key].pop()
        kind, index, payload = action
        values = self.strokes.setdefault(key, [])
        incremental = None
        if kind == "add":
            values.insert(min(index, len(values)), payload)
        elif kind == "delete":
            if 0 <= index < len(values) and values[index] is payload:
                values.pop(index)
            elif payload in values:
                values.remove(payload)
        elif kind == "delete_object":
            stroke, _candidate_index, candidate = payload
            if 0 <= index < len(values) and values[index] is stroke:
                values.pop(index)
            elif stroke in values:
                values.remove(stroke)
            pool = self.candidates.get(key, [])
            if candidate in pool:
                pool.remove(candidate)
        elif kind == "clear":
            _before, after = payload
            values[:] = list(after)
        elif kind == "transform":
            before, after = payload
            target_index = self._resolve_transform_action_index(values, index, before)
            if target_index is None:
                self.edit_redo.setdefault(key, []).append(action)
                self.status.set("无法定位要重做的流星；已保留当前蒙版")
                return
            previous = replace(values[target_index], points=values[target_index].points.copy())
            values[target_index] = replace(after, points=after.points.copy())
            self._sync_matching_candidate(key, values[target_index])
            if self.selected_object == (key, index):
                self.selected_object = (key, target_index)
            incremental = self._incremental_recomposed_object_image(
                (key, target_index), previous, include_selected=True
            )
        self.edit_history.setdefault(key, []).append(action)
        self._restore_shift_anchor()
        self._update_tree_status_for_key(key)
        if incremental is not None:
            self._commit_incremental_global_preview(incremental, validate=False)
        else:
            self._invalidate_global_preview()
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
        if self.object_drag_mode is not None and self.object_drag_original is not None and self.selected_object:
            key, index = self.selected_object
            values = self.strokes.get(key, [])
            if 0 <= index < len(values):
                values[index] = replace(self.object_drag_original, points=self.object_drag_original.points.copy())
            self.object_drag_mode = None
            self.object_drag_start = None
            self.object_drag_original = None
            self._clear_live_object_drag()
            self._render_preview()
            self.status.set("已取消当前流星变换")
            return "break"
        if self.selected_object is not None:
            self._clear_object_selection()
            self.status.set("已取消选择")
            return "break"
        live_stroke = self.live_erase_stroke
        if self.current_path and live_stroke is not None:
            key = str(self.current_path)
            values = self.strokes.get(key, [])
            if values and values[-1] is live_stroke:
                values.pop()
            elif live_stroke in values:
                values.remove(live_stroke)
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
        try:
            self.canvas.grab_release()
        except tk.TclError:
            pass
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
        self._update_tree_status_for_key(str(self.current_path))

    def _update_tree_status_for_key(self, key: str) -> None:
        for index, path in enumerate(self.files):
            if str(path) == key and self.tree.exists(str(index)):
                meteors = [
                    item for item in self.strokes.get(key, [])
                    if not item.erase and item.points
                ]
                original = sum(normalized_source_mode(item) == "original" for item in meteors)
                aligned = len(meteors) - original
                detail = (
                    f"{len(meteors)}（对{aligned}/原{original}）"
                    if meteors and original else (str(len(meteors)) if meteors else "—")
                )
                self.tree.set(str(index), "status", detail)

    def _setup_autosave(self) -> None:
        variables = (
            self.source_dir, self.base_dir, self.output_dir, self.brush_width,
            self.eraser_width, self.feather, self.default_match_exposure, self.curve_enabled,
            self.curve_shadows, self.curve_highlights, self.export_tiff, self.blend_mode,
            self.output_mode, self.default_preserve_brightness,
            self.default_meteor_brightness, self.meteor_brightness,
            self.default_background_cleanup,
            self.base_exposure_tenths,
            self.auto_optimize, self.auto_optimize_strength,
        )
        for variable in variables:
            variable.trace_add("write", lambda *_args: self._schedule_autosave())
        for variable in (
            self.curve_enabled, self.curve_shadows, self.curve_highlights
        ):
            variable.trace_add("write", self._current_adjustment_changed)
        self.autosave_suspended = False

    def _project_data(self) -> dict:
        active = set(self.pairs) or {str(path) for path in self.files}
        return {
            "version": PROJECT_VERSION,
            "source_dir": self.source_dir.get(),
            "base_dir": self.base_dir.get(),
            "base_files": [str(path) for path in self.selected_base_files],
            "output_dir": self.output_dir.get(),
            "output_mode": self.output_mode.get(),
            "match_exposure": self.default_match_exposure.get(),
            "curve_enabled": self.curve_enabled.get(),
            "curve_shadows": self.curve_shadows.get(),
            "curve_highlights": self.curve_highlights.get(),
            "preserve_brightness": self.default_preserve_brightness.get(),
            "meteor_brightness": self.default_meteor_brightness.get(),
            "background_cleanup": self.default_background_cleanup.get(),
            "base_exposure_ev": self.base_exposure_tenths.get() / 10.0,
            "auto_optimize": self.auto_optimize.get(),
            "auto_optimize_strength": self.auto_optimize_strength.get(),
            "adjustment_defaults": self.adjustment_defaults,
            "image_adjustments": {
                key: value for key, value in self.image_adjustments.items() if key in active
            },
            "export_tiff": self.export_tiff.get(),
            "blend_mode": self.blend_mode.get(),
            "brush_width": self.brush_width.get(),
            "eraser_width": self.eraser_width.get(),
            "feather": self.feather.get(),
            "candidate_thresholds": {
                key: value for key, value in self.candidate_thresholds.items() if key in active
            },
            "original_sources": {
                key: str(value) for key, value in self.original_sources.items() if key in active
            },
            "use_original_sources": sorted(self.use_original_sources & active),
            "alignment_statuses": {
                key: value for key, value in self.alignment_statuses.items() if key in active
            },
            "candidates": {
                key: [asdict(item) for item in value]
                for key, value in self.candidates.items() if key in active
            },
            "strokes": {
                key: [asdict(item) for item in value]
                for key, value in self.strokes.items() if key in active
            },
        }

    def _apply_project_data(self, data: dict) -> None:
        self.autosave_suspended = True
        self.loading_adjustments = True
        try:
            self._clear_object_selection()
            self.last_edit_key = None
            self.current_path = None
            self.preview_source = None
            self.preview_base = None
            self.source_dir.set(data.get("source_dir", ""))
            self.base_dir.set(data.get("base_dir", ""))
            saved_base = Path(data.get("base_dir", "")).expanduser()
            inferred_mode = "combined" if saved_base.is_file() else "separate"
            self.output_mode.set(data.get("output_mode", inferred_mode))
            self.selected_base_files = [Path(value) for value in data.get("base_files", [])]
            if not self.selected_base_files and saved_base.is_file():
                self.selected_base_files = [saved_base]
            self.base_selection_summary.set(
                f"已选 {len(self.selected_base_files)} 张" if self.selected_base_files
                else ("按文件夹同名匹配" if saved_base.is_dir() else "")
            )
            self._update_output_mode_ui()
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
                "preserve_brightness": bool(data.get("preserve_brightness", True)),
                "meteor_brightness": int(data.get("meteor_brightness", 100)),
                "background_cleanup": int(data.get("background_cleanup", 70)),
                "auto_optimize": bool(data.get("auto_optimize", True)),
            }
            self.adjustment_defaults = {
                **legacy_defaults, **data.get("adjustment_defaults", {})
            }
            self.default_match_exposure.set(bool(self.adjustment_defaults["match_exposure"]))
            self.default_preserve_brightness.set(bool(self.adjustment_defaults["preserve_brightness"]))
            self.default_meteor_brightness.set(int(self.adjustment_defaults["meteor_brightness"]))
            self.default_background_cleanup.set(int(self.adjustment_defaults["background_cleanup"]))
            self.auto_optimize.set(bool(self.adjustment_defaults.get("auto_optimize", True)))
            self.auto_optimize_strength.set(str(data.get("auto_optimize_strength", "标准")))
            self.base_exposure_tenths.set(int(round(float(data.get("base_exposure_ev", 0.0)) * 10.0)))
            self._base_exposure_changed()
            self.brightness_override.set(False)
            self.meteor_brightness.set(int(self.adjustment_defaults["meteor_brightness"]))
            self.current_brightness_scale.configure(state="disabled")
            self.image_adjustments = {}
            for key, value in data.get("image_adjustments", {}).items():
                normalized = {
                    "curve_enabled": bool(value.get("curve_enabled", self.adjustment_defaults["curve_enabled"])),
                    "curve_shadows": int(value.get("curve_shadows", self.adjustment_defaults["curve_shadows"])),
                    "curve_highlights": int(value.get("curve_highlights", self.adjustment_defaults["curve_highlights"])),
                }
                # Absence means "follow global". Preserve it instead of turning
                # the current global value into a permanent per-image override.
                if "match_exposure" in value:
                    normalized["match_exposure"] = bool(value["match_exposure"])
                if "meteor_brightness" in value:
                    normalized["meteor_brightness"] = int(np.clip(value["meteor_brightness"], 50, 250))
                self.image_adjustments[key] = normalized
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

            def decode_strokes(values, default_source_mode="aligned"):
                return [Stroke(
                    points=[tuple(p) for p in item["points"]], width=item["width"],
                    feather=item["feather"], erase=bool(item.get("erase", False)),
                    locked=bool(item.get("locked", False)), auto_score=item.get("auto_score"),
                    offset_x=float(item.get("offset_x", 0.0)), offset_y=float(item.get("offset_y", 0.0)),
                    rotation=float(item.get("rotation", 0.0)),
                    length_scale=float(item.get("length_scale", 1.0)),
                    width_scale=float(item.get("width_scale", 1.0)),
                    opacity=float(item.get("opacity", 1.0)),
                    brightness_override=(
                        None if item.get("brightness_override") is None
                        else float(item["brightness_override"])
                    ),
                    background_cleanup_override=(
                        None if item.get("background_cleanup_override") is None
                        else float(item["background_cleanup_override"])
                    ),
                    saturation_override=(
                        None if item.get("saturation_override") is None
                        else float(item["saturation_override"])
                    ),
                    preserve_brightness_override=item.get("preserve_brightness_override"),
                    match_exposure_override=item.get("match_exposure_override"),
                    blend_mode_override=item.get("blend_mode_override"),
                    auto_blend_enabled=bool(item.get("auto_blend_enabled", True)),
                    auto_strength=str(item.get("auto_strength", "标准")),
                    auto_black_point=(
                        None if item.get("auto_black_point") is None
                        else float(item["auto_black_point"])
                    ),
                    auto_cleanup=(
                        None if item.get("auto_cleanup") is None else float(item["auto_cleanup"])
                    ),
                    auto_brightness=(
                        None if item.get("auto_brightness") is None else float(item["auto_brightness"])
                    ),
                    auto_feather=(
                        None if item.get("auto_feather") is None else int(item["auto_feather"])
                    ),
                    source_mode=(
                        "original" if item.get("source_mode", default_source_mode) == "original"
                        else "aligned"
                    ),
                ) for item in values]

            self.candidates = {
                key: decode_strokes(
                    values, "original" if key in self.use_original_sources else "aligned"
                ) for key, values in data.get("candidates", {}).items()
            }
            self.strokes = {
                key: decode_strokes(
                    values, "original" if key in self.use_original_sources else "aligned"
                ) for key, values in data.get("strokes", {}).items()
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
            if source.is_dir() and (base.is_dir() or base.is_file()):
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
        if (
            self.pairing_signature != self._base_selection_signature()
            or not self.pairs
        ):
            self.status.set("底图选择已变化，正在重新建立导出配对…")
            if not self.scan_inputs(reload_current=True):
                return
        marked = {
            Path(key): [Stroke(
                item.points.copy(), item.width, item.feather, item.erase, item.locked, item.auto_score,
                item.offset_x, item.offset_y, item.rotation, item.length_scale,
                item.width_scale, item.opacity,
                item.brightness_override, item.background_cleanup_override,
                item.saturation_override, item.preserve_brightness_override,
                item.match_exposure_override, item.blend_mode_override,
                item.auto_blend_enabled, item.auto_strength, item.auto_black_point,
                item.auto_cleanup, item.auto_brightness, item.auto_feather,
                source_mode=item.source_mode,
            ) for item in value]
            for key, value in self.strokes.items()
            if key in self.pairs and value
        }
        if not marked:
            messagebox.showwarning(APP_NAME, "还没有检测或标记任何流星蒙版")
            return
        source = Path(self.source_dir.get())
        base_dir = Path(self.base_dir.get())
        output_text = self.output_dir.get().strip()
        if not output_text:
            output = source / "MeteorStudio_Output"
            self.output_dir.set(str(output))
        else:
            output = Path(output_text)
        combined = self.output_mode.get() == "combined"
        export_summary = (
            f"将把 {len(marked)} 张图片中的全部流星累计拼合到一张底图。\n"
            if combined else f"将分别处理 {len(marked)} 组同名 TIFF/底图。\n"
        )
        learning_choice = messagebox.askyesnocancel(
            APP_NAME + " — 导出与 AI 学习",
            export_summary +
            f"源素材只读，结果写入新的运行目录：\n{output}\n\n"
            "是否在导出完成后，用本次最终蒙版继续训练本地 AI？\n\n"
            "是：导出并自动学习\n否：仅导出\n取消：停止",
        )
        if learning_choice is None:
            return
        self.progress["value"] = 0
        self.status.set("开始导出…")
        self._release_large_caches_for_export()
        self._run_worker(
            self._export_worker, output, marked, self.pairs.copy(),
            {key: value.copy() for key, value in self.image_adjustments.items()},
            self.adjustment_defaults.copy(),
            bool(self.export_tiff.get()), bool(learning_choice), self.blend_mode.get(),
            {str(path): self.original_sources.get(str(path), path) for path in marked},
            self.output_mode.get(), self.base_exposure_tenths.get() / 10.0,
        )

    def _release_large_caches_for_export(self) -> None:
        """Free decoded source caches before OpenCV starts the 16-bit export.

        An exact 8K preview plus the rolling TIFF cache can occupy several GB.
        Keeping those decoded source frames while the exporter creates its own
        16-bit working buffers caused OpenCV allocation failures at the first
        meteor. The visible canvas/photo remains intact; only reloadable source
        caches and redundant labeled copies are discarded.
        """
        self.exact_preview_generation += 1
        self.global_preview_generation += 1
        with self.preview_cache_lock:
            self.full_display_cache.clear()
            self.full_precision_cache.clear()
            self.full_display_cache_bytes = 0
            self.full_precision_cache_bytes = 0
            self.preview_cache.clear()
            self.preview_cache_bytes = 0
            self.viewport_cache.clear()
            self.viewport_cache_bytes = 0
        self.global_labeled_preview_rgb = None
        self.exact_labeled_preview_rgb = None
        self.exact_labeled_preview_full_rgb = None
        gc.collect()

    def _export_worker(
        self, output_dir: Path, marked: dict[Path, list[Stroke]], pairs: dict[str, Path],
        adjustments: dict[str, dict], adjustment_defaults: dict,
        export_tiff: bool, learn_after_export: bool, blend_mode: str,
        original_paths: dict[str, Path], output_mode: str, base_exposure_ev: float = 0.0,
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
        combined = output_mode == "combined"
        combined_result = None
        combined_clean_base = None
        combined_base_path = None
        combined_outputs: list[str] = []
        combined_annotations: list[dict] = []
        source_label_records: list[dict] = []
        for index, (source_path, strokes) in enumerate(marked.items(), start=1):
            adjustment = {**adjustment_defaults, **adjustments.get(str(source_path), {})}
            match = bool(adjustment["match_exposure"])
            curve_enabled = bool(adjustment["curve_enabled"])
            curve_shadows = float(adjustment["curve_shadows"])
            curve_highlights = float(adjustment["curve_highlights"])
            if str(source_path) not in pairs:
                raise ValueError(f"找不到同名底图配对：{source_path.name}")
            base_path = pairs[str(source_path)]
            if combined and combined_result is not None:
                base = combined_result
            else:
                # Export owns a bounded working set. Do not feed decoded 16-bit
                # frames back into the interactive preview cache: doing so can
                # retain roughly a gigabyte while OpenCV allocates blend buffers.
                base = to_uint16(read_image(base_path))
                if combined:
                    combined_base_path = base_path
                    combined_clean_base = base.copy()
            height, width = base.shape[:2]
            original_path = original_paths.get(str(source_path), source_path)
            unique_sources = list(dict.fromkeys((source_path, original_path)))
            with ThreadPoolExecutor(
                max_workers=min(2, len(unique_sources)), thread_name_prefix="meteor-export"
            ) as pool:
                decoded = dict(zip(
                    unique_sources,
                    pool.map(
                        lambda path: to_uint16(read_image(path)),
                        unique_sources,
                    ),
                ))
            aligned16, original16 = decoded[source_path], decoded[original_path]
            if aligned16.shape[:2] != (height, width) or original16.shape[:2] != (height, width):
                raise ValueError(f"尺寸不一致：{source_path.name}")
            crop_spec = strokes_for_composite_crop(
                strokes, width, height, bool(adjustment.get("auto_optimize", True))
            )
            if crop_spec is None:
                continue
            cropped_strokes, (x0, y0, x1, y1) = crop_spec
            result = base if combined and combined_result is not None else base.copy()
            try:
                composed_crop, mask_crop = compose_meteor_sources(
                    aligned16[y0:y1, x0:x1], original16[y0:y1, x0:x1],
                    result[y0:y1, x0:x1], cropped_strokes, match, curve_enabled,
                    curve_shadows, curve_highlights, blend_mode,
                    bool(adjustment.get("preserve_brightness", True)),
                    float(adjustment.get("meteor_brightness", 100)),
                    float(adjustment.get("background_cleanup", 70)),
                    bool(adjustment.get("auto_optimize", True)),
                )
            except cv2.error as exc:
                raise RuntimeError(
                    f"OpenCV 合成失败：{source_path.name}，局部区域 "
                    f"{x1 - x0}×{y1 - y0}。\n{exc}"
                ) from exc
            if not np.any(mask_crop > 0.001):
                continue
            result[y0:y1, x0:x1] = composed_crop
            local_boxes = meteor_mask_boxes(mask_crop)
            source_boxes = [
                (bx0 + x0, by0 + y0, bx1 + x0, by1 + y0)
                for bx0, by0, bx1, by1 in local_boxes
            ]
            item_annotations = meteor_source_annotations(
                source_path.stem, strokes, width, height,
                False, mask_crop, (x0, y0),
            )
            if combined:
                combined_annotations.extend(item_annotations)
            # Allocate one full mask, not both float16 and uint16 copies. On an
            # 8K frame this removes an 80+ MB peak before PNG compression.
            mask_full = np.zeros((height, width), dtype=np.uint16)
            mask_full[y0:y1, x0:x1] = np.clip(
                mask_crop.astype(np.float32) * 65535.0, 0, 65535
            ).astype(np.uint16)
            mask_path = masks_dir / f"{source_path.stem}_mask.png"
            try:
                ok, encoded = cv2.imencode(".png", mask_full)
            except cv2.error as exc:
                raise RuntimeError(
                    f"OpenCV 蒙版编码失败：{source_path.name}（{width}×{height}）。\n{exc}"
                ) from exc
            if not ok:
                raise IOError(f"蒙版保存失败：{mask_path.name}")
            encoded.tofile(mask_path)
            outputs = []
            if combined:
                combined_result = result
            else:
                result = adjust_composite_base_exposure(result, base, base_exposure_ev)
                jpg_path = jpg_dir / f"{source_path.stem}.jpg"
                atomic_write_jpeg(jpg_path, result)
                outputs = [str(jpg_path)]
                annotated, item_label_records = annotate_meteor_sources(result, item_annotations)
                labeled_path = jpg_dir / f"{source_path.stem}_来源标注.jpg"
                atomic_write_jpeg(labeled_path, annotated.astype(np.uint16) * 257)
                outputs.append(str(labeled_path))
                source_label_records.extend(item_label_records)
                if export_tiff:
                    tif_path = tif_dir / f"{source_path.stem}.tif"
                    atomic_write_tiff(tif_path, result)
                    outputs.append(str(tif_path))
            report.append({
                "source_tiff": str(source_path), "original_tiff": str(original_path),
                "workspace_layer": str(source_path),
                "original_state": all(
                    normalized_source_mode(item) == "original"
                    for item in strokes if not item.erase
                ),
                "mixed_source_modes": sorted({
                    normalized_source_mode(item) for item in strokes if not item.erase
                }),
                "clean_jpg": str(base_path),
                "strokes": len(strokes), "mask": str(mask_path),
                "local_match": match, "meteor_curve": curve_enabled,
                "meteor_highlight_protection": bool(adjustment.get("preserve_brightness", True)),
                "meteor_brightness_percent": float(adjustment.get("meteor_brightness", 100)),
                "background_cleanup": float(adjustment.get("background_cleanup", 70)),
                "base_exposure_ev": float(base_exposure_ev),
                "curve_shadows": curve_shadows, "curve_highlights": curve_highlights,
                "blend_mode": blend_mode,
                "per_meteor_overrides": [
                    {
                        "meteor": meteor_index,
                        "brightness_percent": stroke.brightness_override,
                        "background_cleanup": stroke.background_cleanup_override,
                        "saturation_percent": stroke.saturation_override,
                        "preserve_brightness": stroke.preserve_brightness_override,
                        "match_exposure": stroke.match_exposure_override,
                        "blend_mode": stroke.blend_mode_override,
                        "feather": stroke.feather,
                        "source_mode": normalized_source_mode(stroke),
                        "auto_blend": {
                            "enabled": stroke.auto_blend_enabled,
                            "strength": stroke.auto_strength,
                            "black_point_8bit": stroke.auto_black_point,
                            "cleanup": stroke.auto_cleanup,
                            "brightness_percent": stroke.auto_brightness,
                            "effective_feather": stroke.auto_feather,
                        },
                        "manually_transformed": stroke_is_transformed(stroke),
                        "transform": {
                            "offset_x": float(stroke.offset_x),
                            "offset_y": float(stroke.offset_y),
                            "rotation": float(stroke.rotation),
                            "length_scale": float(stroke.length_scale),
                            "width_scale": float(stroke.width_scale),
                        },
                    }
                    for meteor_index, stroke in enumerate(
                        (item for item in strokes if not item.erase and item.points), start=1
                    )
                ],
                "source_regions": [list(box) for box in source_boxes],
                "outputs": outputs,
            })
            del aligned16, original16, mask_full
            if not combined:
                del base, result
            self.work_queue.put(("progress", index / total * 90, f"正在合成 {index}/{total}：{source_path.name}"))
        if (combined and combined_result is not None and combined_base_path is not None
                and combined_clean_base is not None):
            combined_result = adjust_composite_base_exposure(
                combined_result, combined_clean_base, base_exposure_ev
            )
            output_stem = f"{combined_base_path.stem}_全部流星"
            jpg_path = jpg_dir / f"{output_stem}.jpg"
            atomic_write_jpeg(jpg_path, combined_result)
            combined_outputs = [str(jpg_path)]
            if export_tiff:
                tif_path = tif_dir / f"{output_stem}.tif"
                atomic_write_tiff(tif_path, combined_result)
                combined_outputs.append(str(tif_path))
            annotated, source_label_records = annotate_meteor_sources(
                combined_result, combined_annotations
            )
            labeled_path = jpg_dir / f"{output_stem}_来源标注.jpg"
            atomic_write_jpeg(labeled_path, annotated.astype(np.uint16) * 257)
            combined_outputs.append(str(labeled_path))
            for entry in report:
                entry["outputs"] = combined_outputs.copy()
        manifest = {
            "version": PROJECT_VERSION,
            "per_image_adjustments": True,
            "meteor_highlight_protection": True,
            "adjustment_defaults": adjustment_defaults,
            "base_exposure_ev": float(base_exposure_ev),
            "export_tiff": export_tiff,
            "output_mode": output_mode,
            "final_outputs": combined_outputs if combined else [],
            "source_labels": source_label_records,
            "items": report,
        }
        (run_dir / "processing_report.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return "exported", run_dir, len(report), learn_after_export, marked, pairs, original_paths, output_mode

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
                "screening_feedback_path": (
                    __import__("meteor_screening").screening_feedback_file_path()
                ),
                "screening_preview": __import__("meteor_screening").screening_preview,
                "temporal_reference": __import__("meteor_screening").temporal_reference,
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
                if kind == "shared_base_preview":
                    _, base_preview, dims, base_path, pairing_signature = item
                    if self.shared_base_loading_signature == pairing_signature:
                        self.shared_base_loading_signature = None
                    if (
                        pairing_signature != self.pairing_signature
                        or base_path not in self.pairs.values()
                    ):
                        continue
                    self.preview_base = base_preview
                    self.current_dims = dims
                    self.full_cache_pinned_paths = {str(base_path)}
                    if self.view_mode.get() in {"blend", "labeled"} and self._uses_shared_base():
                        self._render_preview()
                elif kind == "preview":
                    (
                        _, request_id, path, source_preview, aligned_preview, original_preview,
                        base_preview, dims, base_path, pairing_signature,
                    ) = item
                    if (
                        request_id != self.preview_request_id
                        or
                        pairing_signature != self.pairing_signature
                        or self.pairs.get(str(path)) != base_path
                    ):
                        continue
                    self.current_path = path
                    self.preview_source = source_preview
                    self.preview_aligned_source = aligned_preview
                    self.preview_original_source = original_preview
                    self.preview_base = base_preview
                    self.current_dims = dims
                    self.full_cache_pinned_paths = {
                        str(path), str(self._effective_source_path(path)),
                        str(self.original_sources.get(str(path), path)), str(base_path),
                    }
                    self._update_source_state_ui(path)
                    key = str(path)
                    self._load_image_adjustments(key)
                    self.setting_candidate_threshold = True
                    self.candidate_threshold.set(self.candidate_thresholds.get(key, 55))
                    self.setting_candidate_threshold = False
                    self._update_candidate_summary(key)
                    self.status.set(f"已加载 {path.name}；可拖动画笔，或单击起点后 Shift+单击终点画直线。")
                    self._render_preview()
                    self._schedule_neighbor_prefetch(path)
                elif kind == "global_preview_partial":
                    _, signature, image, labeled_image, completed, total, included = item
                    if signature != self._global_preview_state_signature():
                        continue
                    self.global_preview_rgb = image
                    self.global_labeled_preview_rgb = labeled_image
                    self.progress["value"] = completed / max(1, total) * 100
                    self.status.set(
                        f"总融合预览正在累积 {completed}/{total}：已加入 {included} 张流星素材"
                    )
                    if self.view_mode.get() in {"blend", "labeled"}:
                        self._render_preview()
                elif kind == "global_preview":
                    _, signature, image, labeled_image, included = item
                    if self.global_preview_loading_signature == signature:
                        self.global_preview_loading_signature = None
                    current_signature = self._global_preview_state_signature()
                    if signature == current_signature:
                        self.global_preview_signature = signature
                        self.global_preview_rgb = image
                        self.global_labeled_preview_rgb = labeled_image
                        self.progress["value"] = 100
                        self.status.set(f"总融合预览完成：已合成 {included} 张图片中的全部流星")
                        if self.view_mode.get() in {"blend", "labeled"}:
                            self._render_preview()
                    elif self.view_mode.get() in {"blend", "labeled"}:
                        # The controls changed while this worker was running.
                        # Discard its stale image and schedule exactly one render
                        # for the newest state instead of starting overlapping jobs.
                        self.after_idle(self._render_preview)
                elif kind == "global_preview_cancelled":
                    _, signature = item
                    if self.global_preview_loading_signature == signature:
                        self.global_preview_loading_signature = None
                    if self.view_mode.get() in {"blend", "labeled"}:
                        self.after_idle(self._render_preview)
                elif kind == "exact_preview_cancelled":
                    _, signature = item
                    if self.exact_preview_loading_signature == signature:
                        self.exact_preview_loading_signature = None
                    if self.exact_preview_signature != self._exact_preview_state_signature():
                        self.after_idle(self._schedule_automatic_exact_preview)
                elif kind == "exact_preview_partial":
                    _, signature, image, labeled_image, completed, total, included = item
                    if signature != self._exact_preview_state_signature():
                        continue
                    self.exact_preview_rgb = image
                    self.exact_labeled_preview_rgb = labeled_image
                    self.progress["value"] = completed / max(1, total) * 90
                    self.exact_preview_status.set(f"精准预览：逐张累积 {completed}/{total}")
                    self.status.set(
                        f"精确预览正在生成 {completed}/{total}：已加入 {included} 张流星素材"
                    )
                    if self.view_mode.get() in {"blend", "labeled"}:
                        self._render_preview()
                elif kind == "exact_preview":
                    (
                        _, signature, image, labeled_image,
                        full_image, full_labeled_image, included, full_dims,
                    ) = item
                    if self.exact_preview_loading_signature == signature:
                        self.exact_preview_loading_signature = None
                    if signature == self._exact_preview_state_signature():
                        self.exact_preview_rgb = image
                        self.exact_labeled_preview_rgb = labeled_image
                        self.exact_preview_full_rgb = full_image
                        self.exact_labeled_preview_full_rgb = full_labeled_image
                        self.exact_preview_signature = signature
                        if abs(self.base_exposure_tenths.get()) < 1:
                            self.global_preview_rgb = full_image
                            self.global_labeled_preview_rgb = full_labeled_image
                            self.global_preview_signature = self._global_preview_state_signature()
                        self.progress["value"] = 100
                        self.exact_preview_status.set(
                            f"精准预览：有效（原图 {full_dims[0]}×{full_dims[1]}）"
                        )
                        self.status.set(
                            f"精准预览完成：按原始尺寸、16 位合成了 {included} 张流星素材"
                        )
                        self._render_preview()
                        if self.exact_preview_open_when_ready:
                            self.open_exact_preview()
                    else:
                        if self.exact_preview_signature != self._exact_preview_state_signature():
                            self.exact_preview_status.set("精准预览：参数已变化，正在重新计算…")
                    self.exact_preview_open_when_ready = False
                    if self.exact_preview_signature != self._exact_preview_state_signature():
                        self.after_idle(self._schedule_automatic_exact_preview)
                elif kind == "blend_optimized":
                    _, results, strength, requested = item
                    applied = 0
                    local_change = None
                    for key, index, expected_points, parameters in results:
                        values = self.strokes.get(key, [])
                        if not (0 <= index < len(values)) or values[index].points != expected_points:
                            continue
                        stroke = values[index]
                        before = replace(stroke, points=stroke.points.copy())
                        stroke.auto_blend_enabled = True
                        stroke.auto_strength = str(parameters["strength"])
                        stroke.auto_black_point = float(parameters["black_point"])
                        stroke.auto_cleanup = float(parameters["cleanup"])
                        stroke.auto_brightness = float(parameters["brightness"])
                        stroke.auto_feather = int(parameters["feather"])
                        self._sync_matching_candidate(key, stroke)
                        applied += 1
                        if requested == 1:
                            local_change = ((key, index), before)
                    self.progress["value"] = 100
                    incremental = None
                    if applied == 1 and local_change is not None:
                        reference, before = local_change
                        incremental = self._incremental_recomposed_object_image(
                            reference, before, include_selected=True
                        )
                    if incremental is not None:
                        self._commit_incremental_global_preview(incremental, validate=False)
                    elif applied:
                        self._invalidate_global_preview()
                        self._render_preview()
                    self._load_selected_object_adjustments()
                    self._update_tree_status()
                    self._schedule_autosave()
                    self.status.set(
                        f"逐流星自动融合优化完成（{strength}）：已更新 {applied}/{requested} 颗；"
                        "当前画布将自动按原始像素更新"
                    )
                elif kind == "progress":
                    _, value, text = item
                    self.progress["value"] = value
                    self.status.set(text)
                elif kind == "autodetected":
                    _, found, plane_count, detected_modes = item
                    updated: dict[str, list[Stroke]] = {
                        key: values.copy() for key, values in self.strokes.items()
                    }
                    for key in {str(path) for path in self.files}:
                        mode = detected_modes.get(key, "aligned")
                        preserved = [
                            stroke for stroke in updated.get(key, [])
                            if normalized_source_mode(stroke) != mode or stroke.locked
                        ]
                        detected = found.get(key, [])
                        updated[key] = detected + [
                            stroke for stroke in preserved
                            if not any(
                                existing.points == stroke.points
                                and normalized_source_mode(existing) == normalized_source_mode(stroke)
                                for existing in detected
                            )
                        ]
                    self.strokes = updated
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
                    _, path, candidates, plane_count, analyzed_mode = item
                    key = str(path)
                    retained_candidates = [
                        stroke for stroke in self.candidates.get(key, [])
                        if normalized_source_mode(stroke) != analyzed_mode
                    ]
                    self.candidates[key] = retained_candidates + candidates
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
                    _, run_dir, count, learn_after_export, marked, pairs, original_paths, output_mode = item
                    self.last_export_path = Path(run_dir)
                    self.progress["value"] = 100
                    self.status.set(f"导出完成：{run_dir}")
                    result_summary = (
                        f"已累计处理 {count} 张流星素材，输出 1 张总合成图。"
                        if output_mode == "combined" else f"已分别输出 {count} 张合成图。"
                    )
                    if learn_after_export:
                        should_open = messagebox.askyesno(
                            APP_NAME,
                            f"导出完成，{result_summary}\n\n结果目录：\n{run_dir}\n\n"
                            "现在开始在后台验证并训练个性化 AI。导出结果不受训练成败影响。\n\n"
                            "是否打开导出文件夹？",
                        )
                        if should_open:
                            self._open_output_folder(run_dir)
                        self.progress["value"] = 0
                        self.status.set("正在从本次最终蒙版学习…")
                        learning_marked: dict[Path, list[Stroke]] = {}
                        learning_pairs: dict[str, Path] = {}
                        for path, strokes in marked.items():
                            for mode in ("aligned", "original"):
                                selected = [
                                    stroke for stroke in strokes
                                    if normalized_source_mode(stroke) == mode
                                ]
                                if not any(not stroke.erase for stroke in selected):
                                    continue
                                actual = (
                                    original_paths.get(str(path), path)
                                    if mode == "original" else path
                                )
                                learning_marked.setdefault(actual, []).extend(selected)
                                learning_pairs[str(actual)] = pairs[str(path)]
                        self._run_worker(self._learn_worker, learning_marked, learning_pairs)
                    else:
                        if messagebox.askyesno(
                            APP_NAME,
                            f"导出完成，{result_summary}\n\n结果目录：\n{run_dir}\n\n是否打开导出文件夹？",
                        ):
                            self._open_output_folder(run_dir)
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
                    show_copyable_error(
                        APP_NAME + " — AI 学习失败",
                        f"照片已经正常导出，但 AI 学习没有完成：\n{text}\n\n原模型保持不变。",
                        parent=self,
                        details=details,
                    )
                elif kind == "error":
                    _, text, details = item
                    self.shared_base_loading_signature = None
                    if self.exact_preview_loading_signature is not None:
                        self.exact_preview_loading_signature = None
                        self.exact_preview_status.set("精准预览：后台更新失败")
                    self.status.set("处理失败")
                    show_copyable_error(APP_NAME, text, parent=self, details=details)
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)


if __name__ == "__main__":
    smoke_project = os.environ.get("METEOR_INTERACTION_SMOKE_PROJECT")
    editable_smoke_report = os.environ.get("METEOR_EDITABLE_SMOKE_REPORT")
    if editable_smoke_report:
        from editable_composite_smoke import run_smoke

        application = MeteorComposer()
        try:
            smoke_result = run_smoke(application)
            Path(editable_smoke_report).write_text(
                json.dumps(smoke_result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            application.destroy()
    elif smoke_project:
        from gui_interaction_smoke import run_smoke

        application = MeteorComposer()
        try:
            smoke_result = run_smoke(application, Path(smoke_project))
            report_path = Path(os.environ.get("METEOR_INTERACTION_SMOKE_REPORT", "interaction-smoke.json"))
            report_path.write_text(json.dumps(smoke_result, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            application.destroy()
    else:
        application = MeteorComposer()
        application.after_idle(application.maximize_for_normal_launch)
        application.mainloop()
