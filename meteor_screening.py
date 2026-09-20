"""Temporal meteor screening without a pre-existing clean base image."""

from __future__ import annotations

import json
import hashlib
import os
import queue
import shutil
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import psutil
import rawpy
import tifffile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk
from platform_utils import open_folder
from error_dialog import append_runtime_log, show_copyable_error, show_runtime_log
from meteor_detection import (
    ML_FEATURE_NAMES,
    calibrate_secondary_candidate_scores,
    candidate_feature_vector,
    predict_gradient_boosting,
    prepare_ml_maps,
)
from background_tasks import BackgroundTaskScheduler, CancellationToken
from ui_navigation import keep_tree_row_in_navigation_runway


RAW_SUFFIXES = {".arw", ".nef", ".nrw", ".cr2", ".cr3", ".crw"}
IMAGE_SUFFIXES = RAW_SUFFIXES | {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
SCREENING_ALGORITHM_VERSION = "2026-09-04-all-candidate-ranking-v3"


def screening_memory_budgets(
    *, available_memory: int | None = None, total_memory: int | None = None,
) -> tuple[int, int]:
    """Use the largest safe preview cache supported by the current machine."""
    memory = psutil.virtual_memory()
    available = int(
        available_memory if available_memory is not None else memory.available
    )
    total = int(total_memory if total_memory is not None else memory.total)
    available = max(0, min(available, total))

    # Preserve enough RAM for the OS, Tk rendering and transient LibRaw/OpenCV
    # arrays, then devote most of the remaining headroom to reusable previews.
    # The total-memory cap prevents a temporarily idle machine from letting one
    # workspace monopolize RAM; the available-memory term adapts to other apps.
    system_reserve = max(2 << 30, int(total * 0.18))
    cache_headroom = max(0, available - system_reserve)
    full_ceiling = min(16 << 30, int(total * 0.50))
    full = int(np.clip(cache_headroom * 0.72, 384 << 20, full_ceiling))
    proxy = int(np.clip(available * 0.04, 64 << 20, 512 << 20))
    return proxy, full


def screening_worker_counts(
    frame_count: int,
    *,
    logical_cpus: int | None = None,
    physical_cpus: int | None = None,
    available_memory: int | None = None,
) -> tuple[int, int]:
    """Choose frame/decode concurrency without nested OpenCV oversubscription."""
    logical = max(1, int(logical_cpus or os.cpu_count() or 4))
    physical = max(
        1, int(physical_cpus or psutil.cpu_count(logical=False) or max(1, logical // 2))
    )
    available = int(
        available_memory
        if available_memory is not None
        else psutil.virtual_memory().available
    )
    # A frame analysis temporarily owns several float/gray/registered arrays.
    # Budget roughly 512 MiB per concurrent frame and leave about one quarter
    # of the physical cores to Tk, the OS and other workspaces.  This arithmetic
    # is deliberately platform-neutral: psutil supplies the same measurements
    # on Windows, macOS and Linux, with the logical-core fallback covering
    # platforms that do not expose physical topology.
    memory_cap = max(1, available // (512 << 20))
    cpu_cap = max(1, min(logical - 1, int(physical * 0.75)))
    analysis = min(max(1, int(frame_count)), 16, memory_cap, cpu_cap)
    # LibRaw decoding is memory- and I/O-heavy. Keep it independent from the
    # larger OpenCV analysis pool so extra analysis concurrency cannot create
    # a burst of ten simultaneous full RAW demosaics.
    decode = min(4, max(1, analysis // 3))
    return analysis, decode


def trim_array_cache(cache: OrderedDict[str, np.ndarray], budget: int) -> int:
    total = sum(int(image.nbytes) for image in cache.values())
    while total > budget and len(cache) > 1:
        _path, removed = cache.popitem(last=False)
        total -= int(removed.nbytes)
    return total


def preview_window(paths: list[str], current: str, radius: int = 2) -> list[str]:
    """Current-first window with most capacity assigned to later list items."""
    try:
        center = paths.index(current)
    except ValueError:
        return [current] if current else []
    radius = max(0, int(radius))
    target = min(len(paths), 1 + radius * 2)
    if target <= 1:
        return [current]

    # Sequential review overwhelmingly moves to the following photo. Keep
    # roughly one quarter of the nominal backward radius for quick reversal,
    # and spend the rest on a contiguous forward runway. Near either boundary,
    # transfer unused capacity to the side that still has photos.
    backward = min(center, max(1, radius // 4))
    forward = min(len(paths) - center - 1, target - 1 - backward)
    remaining = target - 1 - forward - backward
    if remaining:
        extra_backward = min(center - backward, remaining)
        backward += extra_backward
        remaining -= extra_backward
    if remaining:
        forward += min(len(paths) - center - 1 - forward, remaining)

    return (
        [current]
        + paths[center + 1:center + 1 + forward]
        + list(reversed(paths[center - backward:center]))
    )


@dataclass
class ScreeningCandidate:
    start: tuple[int, int]
    end: tuple[int, int]
    score: int
    legacy_score: float = 0.0
    # Candidate-level feedback is deliberately separate from the image-level
    # keep/export decision. Only these explicit labels may train the AI.
    label: str = ""
    manual: bool = False
    features: list[float] = field(default_factory=list)


@dataclass
class ScreeningResult:
    path: str
    candidates: list[ScreeningCandidate]
    score: int
    plane_count: int = 0
    temporal_hits: int = 0
    note: str = ""
    analysis_width: int = 0
    analysis_height: int = 0


@dataclass(frozen=True)
class ScreeningSource:
    """One physical capture with its canonical original and fast analysis proxy."""

    path: Path
    proxy_path: Path


def discover_screening_sources(folder: Path) -> list[ScreeningSource]:
    """Collapse same-stem RAW/JPG pairs while preserving the original RAW path."""
    groups: dict[str, list[Path]] = {}
    for path in folder.iterdir():
        # macOS AppleDouble sidecars such as ``._DSC0001.ARW`` only contain
        # Finder metadata. Their borrowed extension made LibRaw try (and fail)
        # to decode one bogus RAW for every real photo on copied drives.
        if (
            path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            groups.setdefault(path.stem.casefold(), []).append(path)
    priority = {
        ".arw": 0, ".nef": 1, ".nrw": 2, ".cr3": 3, ".cr2": 4, ".crw": 5,
        ".tif": 10, ".tiff": 11, ".jpg": 20, ".jpeg": 21, ".png": 22,
    }
    sources = []
    for paths in groups.values():
        canonical = min(paths, key=lambda item: (priority.get(item.suffix.lower(), 99), item.name.casefold()))
        jpeg = next(
            (item for item in paths if item.suffix.lower() in {".jpg", ".jpeg"}),
            None,
        )
        sources.append(ScreeningSource(canonical, jpeg or canonical))
    return sources


def _normalize_image(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"不支持的图片形状：{array.shape}")
    return array[..., :3]


def read_screening_image(path: Path, max_dimension: int | None = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in RAW_SUFFIXES:
        # Decode only into memory for analysis/preview. The source RAW is never
        # rewritten; selected files are later copied byte-for-byte.
        with rawpy.imread(str(path)) as raw:
            full_resolution = max_dimension is None
            array = raw.postprocess(
                half_size=not full_resolution,
                demosaic_algorithm=(
                    rawpy.DemosaicAlgorithm.AHD if full_resolution
                    else rawpy.DemosaicAlgorithm.LINEAR
                ),
                use_camera_wb=True,
                use_auto_wb=False,
                no_auto_bright=False,
                output_bps=8,
            )
    elif suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        with Image.open(path) as image:
            # JPEG decoders can create a reduced image directly instead of
            # allocating and decoding every full-resolution pixel first.
            if max_dimension and suffix in {".jpg", ".jpeg"}:
                image.draft("RGB", (max_dimension, max_dimension))
            array = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    array = _normalize_image(np.asarray(array))
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        values *= 255.0 / max(1.0, maximum)
    else:
        finite = values[np.isfinite(values)]
        high = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        if high <= 1.5:
            values *= 255.0
        else:
            values *= 255.0 / max(1.0, high)
    return np.nan_to_num(values).clip(0, 255).astype(np.uint8)


def screening_preview(path: Path, max_dimension: int = 1400) -> np.ndarray:
    rgb = read_screening_image(path, max_dimension)
    height, width = rgb.shape[:2]
    scale = min(1.0, max_dimension / max(1, width, height))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return rgb


def screening_fast_preview(
    path: Path, max_dimension: int = 1400, fallback_path: Path | None = None,
) -> np.ndarray:
    """Decode a screening proxy without demosaicing the complete RAW frame."""
    if path.suffix.lower() not in RAW_SUFFIXES:
        return screening_preview(path, max_dimension)
    try:
        with rawpy.imread(str(path)) as raw:
            flip = int(raw.sizes.flip)
            thumbnail = raw.extract_thumb()
        if thumbnail.format == rawpy.ThumbFormat.JPEG:
            encoded = np.frombuffer(thumbnail.data, dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("RAW 内嵌 JPEG 预览无法解码")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = _normalize_image(np.asarray(thumbnail.data))
            if rgb.dtype != np.uint8:
                rgb = _normalize_image(rgb).clip(0, 255).astype(np.uint8)
        if flip == 3:
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        elif flip == 5:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif flip == 6:
            rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        height, width = rgb.shape[:2]
        scale = min(1.0, max_dimension / max(1, width, height))
        if scale < 1.0:
            rgb = cv2.resize(
                rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(rgb)
    except Exception:
        # Some cameras omit the embedded preview. A camera-generated same-stem
        # JPEG is the next-cheapest faithful proxy; only then demosaic RAW.
        if fallback_path is not None and fallback_path != path:
            return screening_preview(fallback_path, max_dimension)
        return screening_preview(path, max_dimension)


def capture_sort_key(path: Path) -> tuple[str, str]:
    """Prefer EXIF capture time and fall back to the stable filename order."""
    captured = ""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            captured = str(exif.get(36867) or exif.get(306) or "")
    except Exception:
        pass
    if not captured and path.suffix.lower() in {".tif", ".tiff"}:
        try:
            with tifffile.TiffFile(path) as tif:
                tags = tif.pages[0].tags
                captured = str(tags[306].value) if 306 in tags else ""
                nested = tags.get("ExifTag")
                if nested is not None and isinstance(nested.value, dict):
                    captured = str(nested.value.get("DateTimeOriginal") or captured)
        except Exception:
            pass
    return captured or "9999", path.name.casefold()


def screening_feedback_file_path() -> Path:
    # The screening and compositing workspaces intentionally share one model
    # data directory, so explicit screening labels can improve the same AI.
    from meteor_composer import user_model_file_path
    return user_model_file_path().parent / "screening_candidate_feedback.json"


def screening_diagnostics_file_path() -> Path:
    """Developer-readable false-positive/false-negative case history."""
    return screening_feedback_file_path().with_name("screening_diagnostic_cases.json")


def save_screening_diagnostic_event(record: dict, path: Path | None = None) -> Path:
    """Append one reviewed failure case without touching any source image."""
    destination = path or screening_diagnostics_file_path()
    records: list[dict] = []
    try:
        if destination.is_file():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                records = [item for item in payload if isinstance(item, dict)]
    except (OSError, ValueError, TypeError):
        records = []
    records.append(record)
    records = records[-10000:]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".writing.json")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def confirmed_track_item(
    source_path: Path, exported_name: str, result_data: dict,
    preview_size: tuple[int, int] | None = None,
) -> dict | None:
    """Build a normalized, cross-workspace track record for one photograph."""
    confirmed = [
        candidate for candidate in result_data.get("candidates", [])
        if candidate.get("label") == "meteor"
    ]
    if not confirmed:
        return None
    width = int(result_data.get("analysis_width", 0))
    height = int(result_data.get("analysis_height", 0))
    if (width <= 1 or height <= 1) and preview_size is not None:
        width, height = map(int, preview_size)
    if width <= 1 or height <= 1:
        return None
    tracks = []
    for candidate in confirmed:
        start = [int(value) for value in candidate["start"]]
        end = [int(value) for value in candidate["end"]]
        tracks.append({
            "start": start,
            "end": end,
            "start_normalized": [start[0] / (width - 1), start[1] / (height - 1)],
            "end_normalized": [end[0] / (width - 1), end[1] / (height - 1)],
            "score": int(candidate.get("score", 100)),
            "manual": bool(candidate.get("manual", False)),
            "locked": True,
        })
    return {
        "source_path": str(source_path),
        "source_name": source_path.name,
        "exported_name": exported_name,
        "analysis_size": {"width": width, "height": height},
        "tracks": tracks,
    }


def screening_autosave_file_path() -> Path:
    from meteor_composer import autosave_file_path
    return autosave_file_path().parent / "screening_autosave.json"


def screening_analysis_checkpoint_file_path() -> Path:
    """Persistent in-progress batch state, separate from reviewed UI state."""
    return screening_autosave_file_path().with_name("screening_analysis_checkpoint.json")


def screening_sequence_signature(sources: list[ScreeningSource]) -> str:
    """Fingerprint ordered inputs without reading or modifying image contents."""
    digest = hashlib.sha256()
    for source in sources:
        for role, path in (("source", source.path), ("proxy", source.proxy_path)):
            if role == "proxy" and path == source.path:
                continue
            try:
                stat = path.stat()
                metadata = f"{stat.st_size}:{stat.st_mtime_ns}"
            except OSError:
                metadata = "missing"
            normalized = os.path.normcase(os.path.abspath(path))
            digest.update(f"{role}\0{normalized}\0{metadata}\n".encode("utf-8"))
    return digest.hexdigest()


def screening_candidate_from_json(payload: dict) -> ScreeningCandidate:
    return ScreeningCandidate(
        tuple(int(value) for value in payload["start"]),
        tuple(int(value) for value in payload["end"]),
        int(payload.get("score", 0)),
        legacy_score=float(payload.get("legacy_score", 0.0)),
        label=str(payload.get("label", "")),
        manual=bool(payload.get("manual", False)),
        features=[float(value) for value in payload.get("features", [])],
    )


def screening_result_from_json(payload: dict) -> ScreeningResult:
    candidates = [
        screening_candidate_from_json(item) for item in payload.get("candidates", [])
    ]
    result = ScreeningResult(
        str(payload.get("path", "")), candidates, int(payload.get("score", 0)),
        plane_count=int(payload.get("plane_count", 0)),
        temporal_hits=int(payload.get("temporal_hits", 0)),
        note=str(payload.get("note", "")),
        analysis_width=int(payload.get("analysis_width", 0)),
        analysis_height=int(payload.get("analysis_height", 0)),
    )
    suppressed = suppress_dense_temporal_clutter(result)
    if suppressed:
        result.note = f"连续多帧云层/结构已抑制 {suppressed} 条候选"
    return result


def load_screening_checkpoint(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if payload.get("format") == "meteor-screening-checkpoint-v1" else None
    except (OSError, ValueError, TypeError):
        return None


def write_screening_checkpoint(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_screening_feedback(records: list[dict]) -> Path | None:
    if not records:
        return None
    path = screening_feedback_file_path()
    existing: dict[str, dict] = {}
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing = {str(item["id"]): item for item in payload if isinstance(item, dict) and "id" in item}
    except (OSError, ValueError, TypeError):
        existing = {}
    for record in records:
        existing[str(record["id"])] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(
        json.dumps(list(existing.values()), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def estimate_neighbor_transform(neighbor: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate a conservative temporary neighbor-to-current similarity transform."""
    if neighbor.shape != current.shape:
        raise ValueError("相邻照片尺寸不一致")
    first = cv2.cvtColor(neighbor, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)

    # A small phase-correlation pass is much cheaper than SIFT. Tripod
    # sequences normally have strong correlation and sub-pixel movement, so
    # avoid extracting thousands of features when registration would later be
    # discarded anyway. Low-confidence or moving frames still use full SIFT.
    quick_scale = min(1.0, 480.0 / max(first.shape))
    quick_size = (max(32, round(first.shape[1] * quick_scale)), max(32, round(first.shape[0] * quick_scale)))
    quick_first = cv2.resize(first, quick_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    quick_second = cv2.resize(second, quick_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    quick_first -= cv2.GaussianBlur(quick_first, (0, 0), 3.0)
    quick_second -= cv2.GaussianBlur(quick_second, (0, 0), 3.0)
    quick_shift, quick_response = cv2.phaseCorrelate(quick_first, quick_second)
    quick_displacement = float(np.hypot(*quick_shift) / max(quick_scale, 1e-6))
    if quick_response >= 0.55 and quick_displacement < 0.8:
        return identity, quick_displacement

    # High-pass images emphasize stars and suppress broad exposure/cloud changes.
    first_hp = cv2.subtract(first, cv2.GaussianBlur(first, (0, 0), 5.0))
    second_hp = cv2.subtract(second, cv2.GaussianBlur(second, (0, 0), 5.0))
    detector = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.015, edgeThreshold=8)
    kp1, des1 = detector.detectAndCompute(first_hp, None)
    kp2, des2 = detector.detectAndCompute(second_hp, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return identity, 0.0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [first_match for first_match, second_match in matches if first_match.distance < 0.72 * second_match.distance]
    if len(good) < 8:
        return identity, 0.0
    source = np.asarray([kp1[item.queryIdx].pt for item in good], np.float32)
    target = np.asarray([kp2[item.trainIdx].pt for item in good], np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5,
        maxIters=5000, confidence=0.995, refineIters=20,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < 7:
        return identity, 0.0
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = float(np.hypot(a, b))
    angle = float(np.degrees(np.arctan2(b, a)))
    height, width = current.shape[:2]
    shift = float(np.hypot(matrix[0, 2], matrix[1, 2]))
    if not 0.975 <= scale <= 1.025 or abs(angle) > 2.5 or shift > max(width, height) * 0.08:
        return identity, 0.0
    projected = cv2.transform(source[:, None, :], matrix)[:, 0, :]
    displacement = float(np.median(np.linalg.norm(projected - source, axis=1)))
    if displacement < 1.2:
        return identity, displacement
    return matrix.astype(np.float32), displacement


def temporal_reference(current: np.ndarray, neighbors: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Create a meteor-free reference from registered neighboring photographs."""
    if not neighbors:
        raise ValueError("至少需要一张相邻照片")
    height, width = current.shape[:2]
    registered = []
    displacements = []
    for neighbor in neighbors:
        if neighbor.shape != current.shape:
            continue
        matrix, displacement = estimate_neighbor_transform(neighbor, current)
        aligned = cv2.warpAffine(
            neighbor, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        registered.append(aligned)
        displacements.append(displacement)
    if not registered:
        raise ValueError("没有尺寸一致的相邻照片")
    reference = np.median(np.stack(registered, axis=0), axis=0)
    return np.clip(reference, 0, 255).astype(np.uint8), float(np.median(displacements))


def estimate_star_sky_mask(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Estimate the star-bearing sky above a possibly uneven foreground."""
    if current.shape[:2] != reference.shape[:2]:
        raise ValueError("当前照片与相邻参考图尺寸不一致")
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # The temporal reference excludes the current frame and is therefore
    # already meteor-free. Deriving the horizon from it also averages moving
    # cloud detail. Taking min(current, reference) made detailed RAW clouds in
    # DSC06006 look like bottom-connected terrain (22% sky), while the same
    # neighbor median alone gives a stable geometric sky around 87%.
    gray = gray_reference
    height, width = gray.shape

    background = cv2.GaussianBlur(gray, (0, 0), 2.2)
    peaks = gray - background
    median = float(np.median(peaks))
    mad = float(np.median(np.abs(peaks - median)))
    threshold = max(median + 3.2 * max(0.7, mad), float(np.percentile(peaks, 96.5)))
    local_maximum = gray >= cv2.dilate(gray, np.ones((5, 5), np.uint8))
    star_seeds = local_maximum & (peaks >= threshold)

    rows, columns = 18, 28
    y_edges = np.linspace(0, height, rows + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, columns + 1, dtype=np.int32)
    ys, xs = np.nonzero(star_seeds)
    cell_y = np.clip(np.searchsorted(y_edges, ys, side="right") - 1, 0, rows - 1)
    cell_x = np.clip(np.searchsorted(x_edges, xs, side="right") - 1, 0, columns - 1)
    star_density = np.zeros((rows, columns), np.float32)
    np.add.at(star_density, (cell_y, cell_x), 1.0)
    star_density = cv2.GaussianBlur(star_density, (3, 3), 0.65)

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    gradient_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    edges = cv2.Canny(gray_u8, 28, 75).astype(np.float32) / 255.0

    def cell_means(values: np.ndarray) -> np.ndarray:
        result = np.zeros((rows, columns), np.float32)
        for row in range(rows):
            for column in range(columns):
                block = values[y_edges[row]:y_edges[row + 1], x_edges[column]:x_edges[column + 1]]
                result[row, column] = float(np.mean(block)) if block.size else 0.0
        return result

    edge_density = cell_means(edges)
    texture = cell_means(gradient)
    luminance = cell_means(gray)

    def robust_unit(values: np.ndarray) -> np.ndarray:
        low, high = (float(value) for value in np.percentile(values, (15.0, 90.0)))
        return np.clip((values - low) / max(1e-5, high - low), 0.0, 1.0)

    star_score = robust_unit(star_density)
    edge_score = robust_unit(edge_density)
    texture_score = robust_unit(texture)
    luminance_score = robust_unit(luminance)
    vertical_prior = np.linspace(0.20, -0.08, rows, dtype=np.float32)[:, None]
    sky_score = 1.45 * star_score - 0.90 * edge_score - 0.35 * texture_score + vertical_prior

    # Choose one smooth but non-flat horizon across the frame. This follows
    # mountains and roofs instead of assuming that the bottom N% is ground.
    data_cost = np.zeros((columns, rows + 1), np.float32)
    for column in range(columns):
        values = sky_score[:, column]
        for boundary in range(2, rows + 1):
            sky_cost = -float(np.mean(values[:boundary]))
            ground_cost = float(np.mean(values[boundary:])) if boundary < rows else 0.10
            boundary_row = min(boundary, rows - 1)
            boundary_edge = float(max(
                edge_score[max(0, boundary_row - 1), column],
                edge_score[boundary_row, column],
            ))
            # A real foreground silhouette normally creates an abrupt cell-to-
            # cell luminance transition even when the ground is brighter than
            # the sky. Cloud texture alone has many edges but much smaller
            # coherent vertical jumps. This prevents a cloud top from becoming
            # the horizon in detailed RAW demosaics such as DSC06006.
            luminance_jump = abs(float(
                luminance_score[boundary_row, column]
                - luminance_score[max(0, boundary_row - 1), column]
            ))
            data_cost[column, boundary] = (
                sky_cost + ground_cost
                - 0.42 * boundary_edge
                - 1.10 * luminance_jump
            )
        data_cost[column, :2] = 50.0

    path_cost = np.full_like(data_cost, np.inf)
    previous = np.zeros_like(data_cost, dtype=np.int16)
    path_cost[0] = data_cost[0]
    heights = np.arange(rows + 1, dtype=np.float32)
    for column in range(1, columns):
        for boundary in range(2, rows + 1):
            transitions = path_cost[column - 1] + 0.055 * np.abs(heights - boundary)
            best = int(np.argmin(transitions))
            path_cost[column, boundary] = data_cost[column, boundary] + transitions[best]
            previous[column, boundary] = best
    horizon = np.zeros(columns, dtype=np.int16)
    horizon[-1] = int(np.argmin(path_cost[-1]))
    for column in range(columns - 1, 0, -1):
        horizon[column - 1] = previous[column, horizon[column]]

    coarse_x = (x_edges[:-1] + x_edges[1:] - 1) * 0.5
    pixel_horizon = np.interp(np.arange(width), coarse_x, y_edges[np.clip(horizon, 0, rows)])
    # Keep the uncertain boundary cell out of the detection region. A meteor
    # crossing the horizon is ambiguous; admitting lamps, grass and buildings
    # creates far more false positives than this conservative half-cell inset.
    pixel_horizon -= max(3.0, height / rows * 0.45)
    mask = np.zeros((height, width), np.uint8)
    for x, bottom in enumerate(pixel_horizon):
        mask[:int(np.clip(round(bottom), 1, height)), x] = 255
    return mask


def mark_temporal_repeats(results: list[ScreeningResult], image_shape: tuple[int, int]) -> None:
    """Flag repeating, similarly directed trails typical of planes or satellites."""
    height, width = image_shape
    diagonal = float(np.hypot(width, height))
    repeat_data: list[tuple[int, list[int]]] = []
    for index, result in enumerate(results):
        hits = 0
        candidate_hits: list[int] = []
        for candidate in result.candidates:
            candidate_repeat_frames = 0
            sx, sy = candidate.start
            ex, ey = candidate.end
            angle = float(np.arctan2(ey - sy, ex - sx))
            direction = np.asarray((np.cos(angle), np.sin(angle)), np.float32)
            normal = np.asarray((-direction[1], direction[0]), np.float32)
            candidate_length = float(np.hypot(ex - sx, ey - sy))
            center = np.asarray(((sx + ex) * 0.5, (sy + ey) * 0.5), np.float32)
            for other_index in range(max(0, index - 2), min(len(results), index + 3)):
                if other_index == index:
                    continue
                broad_matched = False
                strict_matched = False
                for other in results[other_index].candidates:
                    osx, osy = other.start
                    oex, oey = other.end
                    other_angle = float(np.arctan2(oey - osy, oex - osx))
                    delta = abs(np.arctan2(np.sin(2 * (angle - other_angle)), np.cos(2 * (angle - other_angle)))) / 2
                    other_center = np.asarray(((osx + oex) * 0.5, (osy + oey) * 0.5), np.float32)
                    center_delta = other_center - center
                    distance = float(np.linalg.norm(center_delta))
                    perpendicular = abs(float(np.dot(center_delta, normal)))
                    other_length = float(np.hypot(oex - osx, oey - osy))
                    length_ratio = other_length / max(1.0, candidate_length)
                    broad = (
                        delta < np.deg2rad(9)
                        and diagonal * 0.015 < distance < diagonal * 0.38
                    )
                    broad_matched = broad_matched or broad
                    if (
                        broad
                        and diagonal * 0.01 < distance < diagonal * 0.28
                        and perpendicular < max(18.0, diagonal * 0.045)
                        and 0.32 <= length_ratio <= 3.1
                    ):
                        strict_matched = True
                    if broad_matched and strict_matched:
                        break
                hits += int(broad_matched)
                candidate_repeat_frames += int(strict_matched)
            candidate_hits.append(candidate_repeat_frames)
        repeat_data.append((hits, candidate_hits))

    # Measure the complete immutable sequence first. Removing a weak trail from
    # frame N while still measuring frame N+1 otherwise erases its previous-frame
    # evidence, so only the first item in a moving sequence gets suppressed.
    for result, (hits, candidate_hits) in zip(results, repeat_data):
        result.temporal_hits = hits
        had_candidates = bool(result.candidates)
        repeat_counts = {
            id(candidate): repeat_count
            for candidate, repeat_count in zip(result.candidates, candidate_hits)
        }
        suppressed = suppress_dense_temporal_clutter(result, candidate_hits)
        repeated_low = suppress_repeating_low_confidence_tracks(
            result,
            candidate_hits=[repeat_counts.get(id(candidate), 0) for candidate in result.candidates],
        )
        retained_top = max((candidate.score for candidate in result.candidates), default=0)
        if had_candidates:
            result.score = retained_top
            if hits >= 2 and retained_top:
                result.score = max(0, retained_top - 25)
        if repeated_low:
            result.note = f"连续多帧低置信飞机/卫星轨迹已自动排除 {repeated_low} 条"
        elif suppressed:
            result.note = f"连续多帧云层/结构已抑制 {suppressed} 条候选"
        elif hits >= 2:
            result.note = "疑似连续飞机/卫星，请复查"
        elif result.plane_count:
            result.note = "检测到断续灯迹，请复查"


def _candidate_looks_like_luminous_transient(candidate: ScreeningCandidate) -> bool:
    """Protect one-frame positive light trails from dense cloud-edge cleanup."""
    features = candidate.features
    if candidate.manual or candidate.label == "meteor":
        return True
    if len(features) < 32:
        return candidate.score >= 55
    positive_fraction = float(features[29])
    negative_fraction = float(features[30])
    signed_mean = float(features[31])
    return (
        candidate.score >= 55
        or (
            positive_fraction >= 0.82
            and negative_fraction <= 0.18
            and signed_mean >= 1.5
        )
    )


def suppress_dense_temporal_clutter(
    result: ScreeningResult, candidate_hits: list[int] | None = None,
) -> int:
    """Remove dense moving cloud/structure edges while retaining transient light."""
    total = len(result.candidates)
    if total < 6 or result.temporal_hits < int(np.ceil(total * 1.5)):
        return 0
    kept = []
    for index, candidate in enumerate(result.candidates):
        repeats = candidate_hits[index] if candidate_hits is not None else 2
        luminous = _candidate_looks_like_luminous_transient(candidate)
        removable = candidate.label != "not_meteor" and (
            not luminous
            or repeats >= 3 and luminous and candidate.score < 50
        )
        if not removable:
            kept.append(candidate)
    removed = total - len(kept)
    if not removed:
        return 0
    result.candidates[:] = kept
    retained_top = max((candidate.score for candidate in kept), default=0)
    result.score = min(int(result.score), int(retained_top))
    return removed


def suppress_repeating_low_confidence_tracks(
    result: ScreeningResult, candidate_hits: list[int] | None = None,
    *, score_cutoff: int = 50,
) -> int:
    """Drop weak multi-frame aircraft/satellite tracks from the candidate list.

    A meteor is normally a one-frame transient. A weak track that moves with
    the same direction through several adjacent exposures is useful evidence
    for rejection, but keeping it as a visible meteor candidate is misleading.
    Explicit user labels and manual marks are always preserved. When loading an
    older checkpoint without per-candidate repeat counts, apply the conservative
    rule only to a single-candidate result.
    """
    total = len(result.candidates)
    if not total:
        return 0
    if candidate_hits is None:
        if result.temporal_hits < 2 or total != 1:
            return 0
        candidate_hits = [result.temporal_hits]
    kept = []
    for index, candidate in enumerate(result.candidates):
        repeats = candidate_hits[index] if index < len(candidate_hits) else 0
        removable = (
            repeats >= 2
            and candidate.score < score_cutoff
            and not candidate.manual
            and not candidate.label
        )
        if not removable:
            kept.append(candidate)
    removed = total - len(kept)
    if removed:
        result.candidates[:] = kept
        retained_top = max((candidate.score for candidate in kept), default=0)
        result.score = min(int(result.score), int(retained_top))
    return removed


class MeteorScreeningWindow(tk.Toplevel):
    def __init__(
        self, master: tk.Misc,
        return_callback: Callable[[Path], None] | None = None,
    ):
        super().__init__(master)
        self.return_callback = return_callback
        self.title("流星批量筛选（无需底图）")
        self.geometry("1280x820")
        self.minsize(980, 650)
        self.source_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.last_export_dir = tk.StringVar()
        # User-facing sensitivity is intentionally the inverse of the internal
        # score cutoff: moving right always means “show me more candidates”.
        self.sensitivity = tk.IntVar(value=42)
        self.sensitivity_hint = tk.StringVar(value="标准 · 平衡漏检和误选")
        self.analysis_mode = tk.StringVar(value="标准两阶段")
        self.filter_name = tk.StringVar()
        self.filter_status = tk.StringVar(value="全部状态")
        self.filter_label = tk.StringVar(value="全部标签")
        self.filter_score_min = tk.IntVar(value=0)
        self.filter_score_max = tk.IntVar(value=100)
        self.filter_summary = tk.StringVar(value="显示 0/0")
        self.filter_after_id: str | None = None
        self.filtered_result_indices: list[int] = []
        self.status = tk.StringVar(value="选择连续拍摄照片文件夹；原文件只读。")
        self.summary = tk.StringVar(value="尚未分析")
        self.files: list[Path] = []
        self.results: list[ScreeningResult] = []
        self.decisions: dict[str, str] = {}
        self.decision_sources: dict[str, str] = {}
        self.active_candidates: dict[str, int] = {}
        self.selected_candidates: dict[str, set[int]] = {}
        self.candidate_multi_select_mode = tk.BooleanVar(value=False)
        proxy_budget, full_budget = screening_memory_budgets()
        self.preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.preview_cache_bytes = 0
        self.preview_cache_budget = proxy_budget
        self.proxy_preview_loading: set[str] = set()
        self.proxy_preview_active_path: str | None = None
        self.full_preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.full_preview_cache_bytes = 0
        self.full_preview_cache_budget = full_budget
        self.full_preview_loading: set[str] = set()
        self.full_preview_active_path: str | None = None
        self.full_preview_window_paths: list[str] = []
        self.analysis_generation = 0
        self.analysis_running = False
        self.export_running = False
        self.use_original_preview = tk.BooleanVar(value=False)
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_render_rgb: np.ndarray | None = None
        self.preview_path: str | None = None
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_drag_start: tuple[int, int] | None = None
        self.preview_drag_origin: tuple[float, float] | None = None
        self.preview_dragged = False
        self.preview_resize_after: str | None = None
        self.show_candidate_marks = tk.BooleanVar(value=True)
        self.candidate_marks_temporarily_hidden = False
        self.manual_mark_mode = False
        self.manual_mark_start: tuple[int, int] | None = None
        self.manual_mark_label = tk.StringVar(value="手动标记漏检流星")
        self.candidate_status = tk.StringVar(value="点击候选标记后，可逐条确认；照片保留不会自动训练AI。")
        self.autosave_status = tk.StringVar(value="自动保存：等待修改")
        self.autosave_after_id: str | None = None
        self._restoring_autosave = False
        self.work_queue: queue.Queue = queue.Queue()
        self.background_tasks = BackgroundTaskScheduler(
            max_workers=3, thread_name_prefix="meteor-screening"
        )
        self._build_ui()
        for variable in (
            self.filter_name, self.filter_status, self.filter_label,
            self.filter_score_min, self.filter_score_max,
        ):
            variable.trace_add("write", self._filters_changed)
        self.source_dir.trace_add("write", self._source_directory_changed)
        self.output_dir.trace_add("write", lambda *_args: self._schedule_autosave())
        self.protocol("WM_DELETE_WINDOW", self._close_window)
        self._restore_autosave()
        self.after(120, self._poll_queue)
        self.after(350, self._resume_analysis_checkpoint)

    def destroy(self) -> None:
        from ui_navigation import cancel_widget_timers
        cancel_widget_timers(self)
        scheduler = getattr(self, "background_tasks", None)
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            self.background_tasks = None
        super().destroy()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="流星批量筛选", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="与流星合成共用同一本地模型 · 支持主流RAW/TIFF/JPG/PNG · 原图只读").pack(side="left", padx=12)
        ttk.Button(header, text="返回流星合成功能", command=self._return_to_composer).pack(side="right")
        ttk.Button(header, text="运行日志", command=lambda: show_runtime_log(self)).pack(side="right", padx=(0, 6))

        settings = ttk.LabelFrame(root, text="筛选设置", padding=8)
        settings.pack(fill="x")
        self._path_row(settings, 0, "连续照片文件夹", self.source_dir, self._browse_source)
        self._path_row(settings, 1, "筛选结果保存位置", self.output_dir, self._browse_output)
        ttk.Label(settings, text="筛选灵敏度").grid(row=2, column=0, sticky="w", pady=(7, 0))
        sensitivity_row = ttk.Frame(settings)
        sensitivity_row.grid(row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=(7, 0))
        ttk.Label(sensitivity_row, text="候选更少").pack(side="left")
        ttk.Scale(
            sensitivity_row, from_=5, to=95, variable=self.sensitivity, orient="horizontal",
            command=self._sensitivity_changed,
        ).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Label(sensitivity_row, text="候选更多").pack(side="left")
        ttk.Label(sensitivity_row, textvariable=self.sensitivity_hint, width=22).pack(side="left", padx=(10, 0))
        ttk.Label(sensitivity_row, text="分析模式").pack(side="left", padx=(12, 4))
        analysis_mode = ttk.Combobox(
            sensitivity_row, textvariable=self.analysis_mode, state="readonly", width=10,
            values=("快速预筛", "标准两阶段", "最高精度"),
        )
        analysis_mode.pack(side="left")
        analysis_mode.bind("<<ComboboxSelected>>", lambda _event: self._schedule_autosave())
        self.analyze_button = ttk.Button(settings, text="开始分析", command=self.analyze)
        self.analyze_button.grid(row=0, column=4, rowspan=2, padx=(10, 0), sticky="nsew")
        self.export_button = ttk.Button(settings, text="导出已选流星照片", command=self.copy_selected)
        self.export_button.grid(row=2, column=4, padx=(10, 0), pady=(7, 0), sticky="ew")
        ttk.Label(settings, text="最近实际导出位置").grid(row=3, column=0, sticky="w", pady=(7, 0))
        ttk.Entry(settings, textvariable=self.last_export_dir, state="readonly").grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=6, pady=(7, 0),
        )
        ttk.Button(settings, text="打开文件夹", command=self._open_export_folder).grid(
            row=3, column=4, padx=(10, 0), pady=(7, 0), sticky="ew",
        )
        settings.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, pady=8)
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)
        filters = ttk.LabelFrame(left, text="列表筛选", padding=5)
        filters.pack(fill="x", pady=(0, 5))
        ttk.Label(filters, text="照片名").grid(row=0, column=0, sticky="w")
        self.filter_name_entry = ttk.Entry(
            filters, textvariable=self.filter_name, width=14,
        )
        self.filter_name_entry.grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(4, 7),
        )
        ttk.Label(filters, text="状态").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.filter_status_combo = ttk.Combobox(
            filters, textvariable=self.filter_status, state="readonly", width=9,
            values=(
                "全部状态", "自动保留", "自动排除", "人工保留", "人工排除",
                "候选保留", "候选排除",
            ),
        )
        self.filter_status_combo.grid(
            row=1, column=1, sticky="w", padx=(4, 7), pady=(5, 0),
        )
        ttk.Label(filters, text="标签").grid(row=1, column=2, sticky="w", pady=(5, 0))
        self.filter_label_combo = ttk.Combobox(
            filters, textvariable=self.filter_label, state="readonly", width=13,
            values=(
                "全部标签", "有候选", "无候选", "未确认候选",
                "已确认流星", "已确认误选", "需人工复核", "飞机/卫星",
            ),
        )
        self.filter_label_combo.grid(
            row=1, column=3, columnspan=2, sticky="ew", padx=(4, 0), pady=(5, 0),
        )
        ttk.Label(filters, text="评分").grid(row=2, column=0, sticky="w", pady=(5, 0))
        score_range = ttk.Frame(filters)
        score_range.grid(row=2, column=1, sticky="w", padx=(4, 7), pady=(5, 0))
        ttk.Spinbox(
            score_range, from_=0, to=100, width=4,
            textvariable=self.filter_score_min,
        ).pack(side="left")
        ttk.Label(score_range, text="–").pack(side="left", padx=2)
        ttk.Spinbox(
            score_range, from_=0, to=100, width=4,
            textvariable=self.filter_score_max,
        ).pack(side="left")
        ttk.Button(filters, text="清空", command=self._reset_filters).grid(
            row=0, column=4, sticky="e",
        )
        ttk.Label(filters, textvariable=self.filter_summary).grid(
            row=2, column=2, columnspan=3, sticky="e", pady=(5, 0),
        )
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(3, weight=1)
        tree_frame = ttk.Frame(left)
        columns = ("decision", "score", "candidates", "note")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="照片")
        self.tree.heading("decision", text="状态")
        self.tree.heading("score", text="流星评分")
        self.tree.heading("candidates", text="候选")
        self.tree.heading("note", text="提示")
        self.tree.column("#0", width=180)
        self.tree.column("decision", width=90, anchor="center")
        self.tree.column("score", width=65, anchor="center")
        self.tree.column("candidates", width=55, anchor="center")
        self.tree.column("note", width=120)
        self.tree.tag_configure("manual_accept", background="#c8f2d0", foreground="#125a28")
        self.tree.tag_configure("auto_accept", background="#e5f5e8", foreground="#245c31")
        self.tree.tag_configure("manual_reject", background="#f7d2d2", foreground="#812323")
        self.tree.tag_configure("auto_reject", background="#f5eeee", foreground="#6b5555")
        self.tree.tag_configure("candidate_accept", background="#d8ecff", foreground="#174f7a")
        self.tree.tag_configure("candidate_reject", background="#eadcf7", foreground="#5f3478")
        self.tree.tag_configure("warning", background="#ffe8bd", foreground="#744600")
        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        legend = ttk.Frame(left)
        legend.pack(fill="x", pady=(0, 4))
        legend_items = (
            ("自动保留", "#e5f5e8"), ("人工保留", "#c8f2d0"),
            ("自动排除", "#f5eeee"), ("人工排除", "#f7d2d2"),
            ("候选保留", "#d8ecff"), ("候选排除", "#eadcf7"),
            ("飞机/卫星", "#ffe8bd"),
        )
        for index, (text, color) in enumerate(legend_items):
            item = ttk.Frame(legend)
            item.grid(row=index // 4, column=index % 4, sticky="w", padx=(0, 8), pady=1)
            tk.Label(item, text="  ", background=color, relief="solid", borderwidth=1).pack(side="left", padx=(0, 2))
            ttk.Label(item, text=text).pack(side="left")
        tree_frame.pack(fill="both", expand=True)
        horizontal = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        horizontal.pack(side="bottom", fill="x")
        self.tree.configure(xscrollcommand=horizontal.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._show_selected)
        self.tree.bind("<Double-1>", lambda _event: self.accept_selected())

        preview_tools = ttk.Frame(right)
        preview_tools.pack(fill="x", pady=(0, 5))
        ttk.Button(preview_tools, text="适应窗口", command=self._fit_preview).pack(side="left")
        ttk.Button(preview_tools, text="1:1", command=self._actual_size_preview).pack(side="left", padx=(5, 0))
        ttk.Checkbutton(
            preview_tools, text="显示候选标记", variable=self.show_candidate_marks,
            command=self._candidate_marks_changed,
        ).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(
            preview_tools, text="原图精细预览", variable=self.use_original_preview,
            command=self._original_preview_changed,
        ).pack(side="left", padx=(10, 0))
        ttk.Label(preview_tools, text="滚轮缩放 · 拖动平移 · H 原图", style="Muted.TLabel").pack(side="left", padx=10)

        self.canvas = tk.Canvas(right, background="#151515", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._canvas_configured)
        self.canvas.bind("<MouseWheel>", self._canvas_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._canvas_wheel(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._canvas_wheel(event, -1))
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)
        self.canvas.bind("<Delete>", self._remove_selected_candidates)

        candidate_actions = ttk.Frame(right)
        candidate_actions.pack(fill="x", pady=(6, 0))
        candidate_labels = ttk.Frame(candidate_actions)
        candidate_labels.pack(fill="x")
        ttk.Label(candidate_labels, text="候选判断：").pack(side="left")
        ttk.Button(
            candidate_labels, text="✓ 选中是流星",
            command=lambda: self._label_candidate("meteor"),
        ).pack(side="left")
        ttk.Button(
            candidate_labels, text="✕ 选中不是流星",
            command=lambda: self._label_candidate("not_meteor"),
        ).pack(side="left", padx=6)
        ttk.Button(
            candidate_labels, text="清除判断（保留候选）",
            command=lambda: self._label_candidate(""),
        ).pack(side="left")
        manual_actions = ttk.Frame(candidate_actions)
        manual_actions.pack(fill="x", pady=(5, 0))
        ttk.Button(
            manual_actions, textvariable=self.manual_mark_label,
            command=self._toggle_manual_mark,
        ).pack(side="left")
        ttk.Button(
            manual_actions, text="打开误判记录",
            command=self._open_diagnostic_folder,
        ).pack(side="right")
        candidate_selection = ttk.Frame(candidate_actions)
        candidate_selection.pack(fill="x", pady=(5, 0))
        ttk.Checkbutton(
            candidate_selection, text="多选", variable=self.candidate_multi_select_mode,
            command=self._candidate_selection_mode_changed,
        ).pack(side="left")
        self.select_all_candidates_button = ttk.Button(
            candidate_selection, text="全选", command=self._select_all_candidates,
        )
        self.select_all_candidates_button.pack(side="left", padx=(6, 0))
        self.remove_selected_candidates_button = ttk.Button(
            candidate_selection, text="清除选中候选",
            command=self._remove_selected_candidates,
        )
        self.remove_selected_candidates_button.pack(side="left", padx=6)
        self.remove_all_candidates_button = ttk.Button(
            candidate_selection, text="清除全部候选",
            command=self._remove_all_candidates,
        )
        self.remove_all_candidates_button.pack(side="left")
        ttk.Label(
            candidate_selection, text="Delete 清除选中",
        ).pack(side="left", padx=(10, 0))
        ttk.Label(right, textvariable=self.candidate_status).pack(fill="x", pady=(3, 0))

        image_actions = ttk.Frame(right)
        image_actions.pack(fill="x", pady=(5, 0))
        ttk.Label(image_actions, text="当前照片：").pack(side="left")
        ttk.Button(image_actions, text="✓ 保留这张照片", command=self.accept_selected).pack(side="left")
        ttk.Button(image_actions, text="✕ 排除这张照片", command=self.reject_selected).pack(side="left", padx=6)
        ttk.Button(image_actions, text="恢复自动判断", command=self.reset_selected).pack(side="left")
        ttk.Label(image_actions, textvariable=self.summary).pack(side="right")

        bottom = ttk.Frame(root)
        bottom.pack(side="bottom", fill="x", before=body)
        ttk.Label(bottom, textvariable=self.status).pack(side="left", fill="x", expand=True)
        ttk.Label(bottom, textvariable=self.autosave_status).pack(side="right", padx=(8, 10))
        self.progress = ttk.Progressbar(bottom, length=260, mode="determinate")
        self.progress.pack(side="right")
        self.bind("<KeyPress-h>", self._hide_candidate_marks)
        self.bind("<KeyRelease-h>", self._show_candidate_marks)
        self.bind("<KeyPress-H>", self._hide_candidate_marks)
        self.bind("<KeyRelease-H>", self._show_candidate_marks)
        self.bind("<Escape>", self._cancel_manual_mark)

    @staticmethod
    def _path_row(parent, row: int, label: str, variable: tk.StringVar, command: Callable) -> None:
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        ttk.Button(parent, text="选择…", command=command).grid(row=row, column=3, sticky="ew")

    def _autosave_payload(self) -> dict:
        score_min, score_max = self._filter_score_bounds()
        return {
            "format": "meteor-screening-autosave-v1",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": self.source_dir.get(),
            "output_dir": self.output_dir.get(),
            "last_export_dir": self.last_export_dir.get(),
            "sensitivity": int(self.sensitivity.get()),
            "analysis_mode": self.analysis_mode.get(),
            "use_original_preview": bool(self.use_original_preview.get()),
            "filters": {
                "name": self.filter_name.get(),
                "status": self.filter_status.get(),
                "label": self.filter_label.get(),
                "score_min": score_min,
                "score_max": score_max,
            },
            "decisions": self.decisions,
            "decision_sources": self.decision_sources,
            "results": [asdict(result) for result in self.results],
        }

    def _schedule_autosave(self, _event=None) -> None:
        if self._restoring_autosave:
            return
        if self.autosave_after_id is not None:
            try:
                self.after_cancel(self.autosave_after_id)
            except tk.TclError:
                pass
        self.autosave_status.set("自动保存：等待写入…")
        self.autosave_after_id = self.after(900, self._save_autosave)

    def _save_autosave(self) -> bool:
        self.autosave_after_id = None
        try:
            path = screening_autosave_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".writing.json")
            temporary.write_text(
                json.dumps(self._autosave_payload(), ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary, path)
            self.autosave_status.set("自动保存：" + datetime.now().strftime("%H:%M:%S"))
            return True
        except Exception as exc:
            self.autosave_status.set(f"自动保存失败：{exc}")
            return False

    @staticmethod
    def _candidate_from_json(payload: dict) -> ScreeningCandidate:
        return screening_candidate_from_json(payload)

    def _restore_autosave(self) -> None:
        path = screening_autosave_file_path()
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != "meteor-screening-autosave-v1":
                return
            restored = []
            for item in payload.get("results", []):
                source_path = Path(str(item.get("path", "")))
                if not source_path.is_file():
                    continue
                restored_result = ScreeningResult(
                    str(source_path),
                    [self._candidate_from_json(value) for value in item.get("candidates", [])],
                    int(item.get("score", 0)),
                    plane_count=int(item.get("plane_count", 0)),
                    temporal_hits=int(item.get("temporal_hits", 0)),
                    note=str(item.get("note", "")),
                    analysis_width=int(item.get("analysis_width", 0)),
                    analysis_height=int(item.get("analysis_height", 0)),
                )
                suppressed = suppress_dense_temporal_clutter(restored_result)
                repeated_low = suppress_repeating_low_confidence_tracks(restored_result)
                if repeated_low:
                    restored_result.note = (
                        f"连续多帧低置信飞机/卫星轨迹已自动排除 {repeated_low} 条"
                    )
                elif suppressed:
                    restored_result.note = (
                        f"连续多帧云层/结构已抑制 {suppressed} 条候选"
                    )
                restored.append(restored_result)
            self._restoring_autosave = True
            self.source_dir.set(str(payload.get("source_dir", "")))
            self.output_dir.set(str(payload.get("output_dir", "")))
            restored_export = Path(str(payload.get("last_export_dir", ""))).expanduser()
            self.last_export_dir.set(str(restored_export) if restored_export.is_dir() else "")
            self.sensitivity.set(int(payload.get("sensitivity", 42)))
            restored_mode = str(payload.get("analysis_mode", "标准两阶段"))
            self.analysis_mode.set(
                restored_mode if restored_mode in {"快速预筛", "标准两阶段", "最高精度"}
                else "标准两阶段"
            )
            self.use_original_preview.set(bool(payload.get("use_original_preview", False)))
            # Candidate marks are a review aid, not a persistent project choice.
            # Always enter the screening workspace with them visible; otherwise
            # one old unchecked autosave makes later analyses look as if no
            # meteors were detected.  The user can still hide them for the
            # current session, or hold H for a temporary clean-image view.
            self.show_candidate_marks.set(True)
            restored_filters = payload.get("filters", {})
            self.filter_name.set(str(restored_filters.get("name", "")))
            status_filter = str(restored_filters.get("status", "全部状态"))
            self.filter_status.set(status_filter if status_filter in {
                "全部状态", "自动保留", "自动排除", "人工保留", "人工排除",
                "候选保留", "候选排除",
            } else "全部状态")
            label_filter = str(restored_filters.get("label", "全部标签"))
            self.filter_label.set(label_filter if label_filter in {
                "全部标签", "有候选", "无候选", "未确认候选",
                "已确认流星", "已确认误选", "需人工复核", "飞机/卫星",
            } else "全部标签")
            try:
                restored_score_min = int(restored_filters.get("score_min", 0))
                restored_score_max = int(restored_filters.get("score_max", 100))
            except (TypeError, ValueError):
                restored_score_min, restored_score_max = 0, 100
            self.filter_score_min.set(int(np.clip(restored_score_min, 0, 100)))
            self.filter_score_max.set(int(np.clip(restored_score_max, 0, 100)))
            valid_paths = {item.path for item in restored}
            self.decisions = {
                str(key): str(value) for key, value in payload.get("decisions", {}).items()
                if str(key) in valid_paths and value in {"accept", "reject"}
            }
            restored_sources = payload.get("decision_sources", {})
            self.decision_sources = {
                key: str(restored_sources.get(key, "manual"))
                for key in self.decisions
                if restored_sources.get(key, "manual") in {"manual", "candidate"}
            }
            self.results = restored
            self.files = [Path(item.path) for item in restored]
            self._restoring_autosave = False
            self._sensitivity_changed()
            if self.results:
                self._refresh_tree(False)
                self._show_selected()
                self.status.set(
                    f"已恢复上次筛选结果：{len(self.results)} 张 · "
                    f"保存于 {payload.get('saved_at', '未知时间')}"
                )
            self.autosave_status.set("自动保存：已恢复")
        except Exception as exc:
            self._restoring_autosave = False
            self.autosave_status.set(f"自动保存恢复失败：{exc}")

    def _close_window(self) -> None:
        if (self.analysis_running or self.export_running) and not messagebox.askyesno(
            "流星批量筛选正在运行",
            "后台分析或导出尚未完成。关闭会取消后续处理；已经复制的文件会保留。确定关闭吗？",
            parent=self,
        ):
            return
        if self.autosave_after_id is not None:
            try:
                self.after_cancel(self.autosave_after_id)
            except tk.TclError:
                pass
            self.autosave_after_id = None
        if self.filter_after_id is not None:
            try:
                self.after_cancel(self.filter_after_id)
            except tk.TclError:
                pass
            self.filter_after_id = None
        self._save_autosave()
        self.analysis_generation += 1
        self.background_tasks.shutdown(wait=False)
        self.destroy()

    def _return_to_composer(self) -> None:
        text = self.last_export_dir.get().strip()
        exported = Path(text).expanduser() if text else None
        callback = self.return_callback
        master = self.master
        self._close_window()
        if not self.winfo_exists():
            if exported is not None and exported.is_dir() and callback is not None:
                callback(exported)
            if hasattr(master, "show_composite_workspace"):
                master.show_composite_workspace()

    def _browse_source(self) -> None:
        value = filedialog.askdirectory(title="选择连续拍摄照片文件夹", parent=self)
        if value:
            self.source_dir.set(value)
            if not self.output_dir.get().strip():
                self.output_dir.set(str(Path(value) / "MeteorStudio_Output"))

    def _source_directory_changed(self, *_args) -> None:
        self._schedule_autosave()
        scheduler = getattr(self, "background_tasks", None)
        if scheduler is None:
            return
        scheduler.cancel("proxy_preview")
        scheduler.cancel("full_preview")
        self.proxy_preview_loading.clear()
        self.proxy_preview_active_path = None
        self.full_preview_loading.clear()
        self.full_preview_active_path = None
        self.full_preview_window_paths = []
        if self.analysis_running:
            self.analysis_generation += 1
            scheduler.cancel("analysis")
            self.analysis_running = False
            self.analyze_button.configure(state="normal", text="开始分析")
            self.status.set("照片文件夹已变化，旧分析任务已取消")

    def _browse_output(self) -> None:
        value = filedialog.askdirectory(title="选择筛选结果保存位置", parent=self)
        if value:
            self.output_dir.set(value)

    def _resume_analysis_checkpoint(self) -> None:
        """Automatically continue an interrupted batch after reopening the workspace."""
        if self.analysis_running:
            return
        checkpoint = load_screening_checkpoint(screening_analysis_checkpoint_file_path())
        if not checkpoint:
            return
        source = Path(self.source_dir.get().strip()).expanduser()
        if (
            not source.is_dir()
            or checkpoint.get("source_dir") != os.path.normcase(os.path.abspath(source))
            or checkpoint.get("analysis_mode") != self.analysis_mode.get()
            or checkpoint.get("algorithm_version") != SCREENING_ALGORITHM_VERSION
        ):
            return
        self.status.set("检测到未完成的分析断点，正在自动继续…")
        self.analyze()

    def analyze(self) -> None:
        if self.analysis_running:
            self.status.set("已有分析任务正在运行，请等待完成")
            return
        source = Path(self.source_dir.get().strip()).expanduser()
        if not source.is_dir():
            show_copyable_error("流星批量筛选", "请选择有效的照片文件夹", parent=self)
            return
        # Directory enumeration and capture-time sorting open image metadata.
        # A real shoot can contain 1000+ RAW/TIFF files on a removable drive,
        # so doing this before submitting the worker freezes Tk for minutes.
        self.files = []
        self.results = []
        self.filtered_result_indices = []
        self.filter_summary.set("显示 0/0")
        self.decisions.clear()
        self.decision_sources.clear()
        self.active_candidates.clear()
        self.selected_candidates.clear()
        self.preview_cache.clear()
        self.preview_cache_bytes = 0
        self.proxy_preview_loading.clear()
        self.proxy_preview_active_path = None
        self.full_preview_cache.clear()
        self.full_preview_cache_bytes = 0
        self.full_preview_loading.clear()
        self.full_preview_active_path = None
        self.full_preview_window_paths = []
        self.background_tasks.cancel("proxy_preview")
        self.background_tasks.cancel("full_preview")
        self.tree.delete(*self.tree.get_children())
        self.progress["value"] = 0
        self.status.set("正在后台扫描照片和读取拍摄时间…")
        self.analysis_generation += 1
        generation = self.analysis_generation
        analysis_mode = self.analysis_mode.get()
        self.analysis_running = True
        self.analyze_button.configure(state="disabled", text=f"{analysis_mode}…")
        self.background_tasks.submit(
            "analysis",
            lambda token: self._analyze_worker(
                source, generation, str(source), token, analysis_mode,
                screening_analysis_checkpoint_file_path(),
            ),
            replace=True,
        )

    def _analyze_worker(
        self, source: Path, generation: int, source_signature: str,
        token: CancellationToken, analysis_mode: str = "标准两阶段",
        checkpoint_path: Path | None = None,
    ) -> None:
        try:
            from meteor_composer import detect_trails, load_meteor_ranker
            self.work_queue.put((
                "analysis_progress", generation, source_signature, 0,
                "正在后台扫描照片文件…",
            ))
            discovered = discover_screening_sources(source)
            if token.cancelled:
                return
            if len(discovered) < 3:
                raise ValueError("至少需要 3 张连续照片；推荐 5 张以上")
            decorated = []
            for index, item in enumerate(discovered, start=1):
                if token.cancelled:
                    return
                decorated.append((capture_sort_key(item.proxy_path), item))
                if index == 1 or index == len(discovered) or index % 25 == 0:
                    self.work_queue.put((
                        "analysis_progress", generation, source_signature,
                        index / len(discovered) * 4.0,
                        f"整理拍摄序列 {index}/{len(discovered)}：{item.path.name}",
                    ))
            decorated.sort(key=lambda item: item[0])
            sources = [item for _sort_key, item in decorated]
            files = [item.path for item in sources]
            proxy_files = [item.proxy_path for item in sources]
            sequence_signature = screening_sequence_signature(sources)
            normalized_source = os.path.normcase(os.path.abspath(source))
            model = load_meteor_ranker()
            cache: OrderedDict[str, np.ndarray] = OrderedDict()
            analysis_workers, decode_workers = screening_worker_counts(len(files))
            cache_limit = analysis_workers + 8
            results_by_index: dict[int, ScreeningResult] = {}
            decode_errors: list[tuple[str, str]] = []
            analysis_errors: dict[str, str] = {}
            refine_errors: dict[str, str] = {}
            failed_paths: set[str] = set()
            first_preview: np.ndarray | None = None
            shape = (1, 1)
            previous_cv_threads = cv2.getNumThreads()
            refined_paths: set[str] = set()
            checkpoint_stage = ""

            checkpoint = load_screening_checkpoint(checkpoint_path)
            if checkpoint and (
                checkpoint.get("source_dir") == normalized_source
                and checkpoint.get("analysis_mode") == analysis_mode
                and checkpoint.get("sequence_signature") == sequence_signature
                and checkpoint.get("algorithm_version") == SCREENING_ALGORITHM_VERSION
            ):
                file_indices = {str(path): index for index, path in enumerate(files)}
                for item in checkpoint.get("results", []):
                    result = screening_result_from_json(item)
                    suppress_repeating_low_confidence_tracks(result)
                    index = file_indices.get(result.path)
                    if index is not None:
                        results_by_index[index] = result
                decode_errors = [
                    (str(path), str(message))
                    for path, message in checkpoint.get("decode_errors", [])
                    if str(path) in file_indices
                ]
                failed_paths = {path for path, _message in decode_errors}
                analysis_errors = {
                    str(path): str(message)
                    for path, message in checkpoint.get("analysis_errors", [])
                    if str(path) in file_indices
                }
                refine_errors = {
                    str(path): str(message)
                    for path, message in checkpoint.get("refine_errors", [])
                    if str(path) in file_indices
                }
                refined_paths = {
                    str(path) for path in checkpoint.get("refined_paths", [])
                    if str(path) in file_indices
                }
                saved_shape = checkpoint.get("image_shape", [1, 1])
                if isinstance(saved_shape, list) and len(saved_shape) == 2:
                    shape = (max(1, int(saved_shape[0])), max(1, int(saved_shape[1])))
                checkpoint_stage = str(checkpoint.get("stage", "first_pass"))
                resume_progress = (
                    96.0 if checkpoint_stage in {"refine", "completed"}
                    else 4.0 + len(results_by_index) / max(1, len(files)) * 92.0
                )
                self.work_queue.put((
                    "analysis_progress", generation, source_signature, resume_progress,
                    f"已恢复分析断点：首轮 {len(results_by_index)}/{len(files)}，"
                    f"精确复核 {len(refined_paths)} 张",
                ))

            def save_checkpoint(stage: str) -> None:
                write_screening_checkpoint(checkpoint_path, {
                    "format": "meteor-screening-checkpoint-v1",
                    "algorithm_version": SCREENING_ALGORITHM_VERSION,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "source_dir": normalized_source,
                    "analysis_mode": analysis_mode,
                    "sequence_signature": sequence_signature,
                    "stage": stage,
                    "image_shape": [int(shape[0]), int(shape[1])],
                    "results": [
                        asdict(results_by_index[index]) for index in sorted(results_by_index)
                    ],
                    "decode_errors": decode_errors,
                    "analysis_errors": list(analysis_errors.items()),
                    "refine_errors": list(refine_errors.items()),
                    "refined_paths": sorted(refined_paths),
                })

            def load_first_pass(index: int) -> np.ndarray:
                if analysis_mode == "最高精度":
                    return screening_preview(files[index])
                if files[index].suffix.lower() in RAW_SUFFIXES:
                    return screening_fast_preview(
                        files[index], fallback_path=proxy_files[index],
                    )
                return screening_fast_preview(proxy_files[index])

            def analyze_frame(
                index: int, frame_cache: OrderedDict[str, np.ndarray],
            ) -> ScreeningResult | None:
                if token.cancelled:
                    return None
                path = files[index]
                current = frame_cache.get(str(path))
                if current is None:
                    return None
                neighbor_indices = [
                    other for other in range(max(0, index - 3), min(len(files), index + 4))
                    if other != index
                ]
                neighbors = []
                for other in neighbor_indices:
                    neighbor = frame_cache.get(str(files[other]))
                    if neighbor is not None and neighbor.shape == current.shape:
                        neighbors.append(neighbor)
                reference, displacement = temporal_reference(current, neighbors)
                sky_mask = estimate_star_sky_mask(current, reference)
                trails, planes = detect_trails(
                    current, reference, ranked=True, valid_region=sky_mask,
                )
                # Always retain the shared model's feature vector. It is used
                # only if the user explicitly labels this exact candidate.
                maps = prepare_ml_maps(current, reference) if trails else None
                measured = []
                # Geometry ranking is only a proposal stage. Classify every
                # bounded detector proposal (at most 60), otherwise a faint
                # meteor behind twelve strong cloud edges never reaches AI.
                for start, end, legacy_score in trails:
                    features = (
                        candidate_feature_vector(maps, start, end, legacy_score)
                        if maps is not None else np.empty(0, dtype=np.float32)
                    )
                    if model is not None and features.size:
                        score = int(round(100 * predict_gradient_boosting(
                            features, model
                        )))
                    else:
                        score = int(legacy_score)
                    measured.append((
                        int(np.clip(score, 0, 100)), start, end,
                        float(legacy_score), features,
                    ))
                calibrated = calibrate_secondary_candidate_scores([
                    (score, start, end, legacy_score)
                    for score, start, end, legacy_score, _features in measured
                ])
                candidates = []
                for (score, start, end, legacy_score), measured_item in zip(calibrated, measured):
                    features = measured_item[4]
                    candidates.append(ScreeningCandidate(
                        start, end, int(np.clip(score, 0, 100)), legacy_score=float(legacy_score),
                        features=features.tolist(),
                    ))
                candidates.sort(key=lambda item: item.score, reverse=True)
                top = candidates[0].score if candidates else 0
                sky_fraction = float(np.mean(sky_mask > 0))
                note_parts = [f"星空区域 {sky_fraction:.0%}"]
                if displacement >= 1.2:
                    note_parts.append(f"相邻星点配准 {displacement:.1f}px")
                note = "；".join(note_parts)
                if token.cancelled:
                    return None
                return ScreeningResult(
                    str(path), candidates, top, planes, note=note,
                    analysis_width=int(current.shape[1]),
                    analysis_height=int(current.shape[0]),
                )

            try:
                # OpenCV otherwise creates a full CPU-sized worker pool inside
                # every frame. Give each frame one OpenCV thread and parallelize
                # independent frames instead; this provides much higher batch
                # throughput and avoids severe nested-thread oversubscription.
                cv2.setNumThreads(1)
                with (
                    ThreadPoolExecutor(max_workers=decode_workers, thread_name_prefix="meteor-decode") as decoder,
                    ThreadPoolExecutor(max_workers=analysis_workers, thread_name_prefix="meteor-analyze") as analyzer,
                ):
                    for batch_start in range(0, len(files), analysis_workers):
                        if token.cancelled:
                            return
                        if checkpoint_stage in {"refine", "completed"}:
                            break
                        batch_end = min(len(files), batch_start + analysis_workers)
                        pending_indices = [
                            index for index in range(batch_start, batch_end)
                            if index not in results_by_index
                            and str(files[index]) not in failed_paths
                        ]
                        if not pending_indices:
                            continue
                        required = range(max(0, batch_start - 3), min(len(files), batch_end + 3))
                        missing = [
                            index for index in required
                            if str(files[index]) not in cache and str(files[index]) not in failed_paths
                        ]
                        decode_futures = {
                            decoder.submit(load_first_pass, index): index for index in missing
                        }
                        for decoded_count, future in enumerate(
                            as_completed(decode_futures), start=1
                        ):
                            index = decode_futures[future]
                            path_text = str(files[index])
                            try:
                                image = future.result()
                            except Exception as exc:
                                failed_paths.add(path_text)
                                decode_errors.append((
                                    path_text, f"{type(exc).__name__}: {exc}"
                                ))
                                self.work_queue.put((
                                    "analysis_progress", generation, source_signature,
                                    4.0 + batch_start / len(files) * 92.0,
                                    f"跳过无法读取的文件：{files[index].name}；继续分析其余照片",
                                ))
                                continue
                            cache[str(files[index])] = image
                            self.work_queue.put((
                                "analysis_progress", generation, source_signature,
                                4.0 + batch_start / len(files) * 92.0,
                                f"读取分析帧 {batch_start + decoded_count}/{len(files)}："
                                f"{files[index].name}",
                            ))

                        analysis_futures = {
                            analyzer.submit(analyze_frame, index, cache): index
                            for index in pending_indices
                            if str(files[index]) in cache
                        }
                        batch_results: dict[int, ScreeningResult] = {}
                        batch_failures: dict[int, Exception] = {}
                        for finished_count, future in enumerate(
                            as_completed(analysis_futures), start=1
                        ):
                            index = analysis_futures[future]
                            try:
                                result = future.result()
                            except Exception as exc:
                                # Do not abort the shoot for an opaque native
                                # OpenCV failure. Wait for the concurrent wave
                                # to drain, then retry this frame synchronously.
                                batch_failures[index] = exc
                                result = None
                            if result is None and token.cancelled:
                                return
                            if result is not None:
                                batch_results[index] = result
                            completed = batch_start + finished_count
                            self.work_queue.put((
                                "analysis_progress", generation, source_signature,
                                4.0 + completed / len(files) * 92.0,
                                f"并行分析 {completed}/{len(files)}（{analysis_workers} 线程）",
                            ))
                        for index, first_error in batch_failures.items():
                            if token.cancelled:
                                return
                            try:
                                result = analyze_frame(index, cache)
                            except Exception as retry_error:
                                path_text = str(files[index])
                                analysis_errors[path_text] = (
                                    f"首次：{type(first_error).__name__}: {first_error}\n"
                                    f"串行重试：{type(retry_error).__name__}: {retry_error}"
                                )
                                # Failure must be fail-safe: keep the photo for
                                # manual review instead of silently discarding a
                                # possible meteor or terminating the whole batch.
                                result = ScreeningResult(
                                    path_text, [], 100, 0,
                                    note="OpenCV 分析失败；已默认保留，必须人工复核",
                                )
                            if result is not None:
                                batch_results[index] = result
                        if token.cancelled:
                            return
                        for index in range(batch_start, batch_end):
                            result = batch_results.get(index)
                            if result is None:
                                continue
                            results_by_index[index] = result
                            shape = cache[result.path].shape[:2]
                            result.analysis_height = int(shape[0])
                            result.analysis_width = int(shape[1])
                            if first_preview is None:
                                first_preview = cache[result.path]

                        # Keep enough overlap for the next temporal batch and a
                        # small UI preview LRU, without retaining the whole shoot.
                        while len(cache) > cache_limit:
                            cache.popitem(last=False)
                        trim_array_cache(cache, self.preview_cache_budget)
                        save_checkpoint("first_pass")

                    refined_count = 0
                    if (
                        analysis_mode == "标准两阶段"
                        and checkpoint_stage != "completed"
                        and not token.cancelled
                    ):
                        candidate_indices = [
                            index for index, result in sorted(results_by_index.items())
                            if result.candidates or result.plane_count
                        ]
                        candidate_indices = [
                            index for index in candidate_indices
                            if str(files[index]) not in refined_paths
                        ]
                        precise_cache: OrderedDict[str, np.ndarray] = OrderedDict()
                        candidate_total = len(candidate_indices)
                        save_checkpoint("refine")
                        if candidate_total:
                            self.work_queue.put((
                                "analysis_progress", generation, source_signature, 96.0,
                                f"首轮完成，正在用原始素材精确复核 {candidate_total} 张候选…",
                            ))
                        for group_start in range(0, candidate_total, analysis_workers):
                            if token.cancelled:
                                return
                            core = candidate_indices[group_start:group_start + analysis_workers]
                            required = sorted({
                                neighbor
                                for index in core
                                for neighbor in range(max(0, index - 3), min(len(files), index + 4))
                                if str(files[neighbor]) not in failed_paths
                            })
                            missing = [
                                index for index in required
                                if str(files[index]) not in precise_cache
                                and str(files[index]) not in refine_errors
                            ]
                            decode_futures = {
                                decoder.submit(screening_preview, files[index]): index
                                for index in missing
                            }
                            for future in as_completed(decode_futures):
                                index = decode_futures[future]
                                path_text = str(files[index])
                                try:
                                    precise_cache[path_text] = future.result()
                                except Exception as exc:
                                    refine_errors[path_text] = f"{type(exc).__name__}: {exc}"

                            refinement_futures = {
                                analyzer.submit(analyze_frame, index, precise_cache): index
                                for index in core
                                if str(files[index]) in precise_cache
                            }
                            refinement_failures: dict[int, Exception] = {}
                            for future in as_completed(refinement_futures):
                                index = refinement_futures[future]
                                try:
                                    result = future.result()
                                except Exception as exc:
                                    refinement_failures[index] = exc
                                    result = None
                                if result is None and token.cancelled:
                                    return
                                if result is not None:
                                    results_by_index[index] = result
                                    refined_count += 1
                            for index, first_error in refinement_failures.items():
                                if token.cancelled:
                                    return
                                path_text = str(files[index])
                                try:
                                    result = analyze_frame(index, precise_cache)
                                except Exception as retry_error:
                                    refine_errors[path_text] = (
                                        f"首次：{type(first_error).__name__}: {first_error}\n"
                                        f"串行重试：{type(retry_error).__name__}: {retry_error}"
                                    )
                                    # Keep the valid fast-pass result.
                                    continue
                                if result is not None:
                                    results_by_index[index] = result
                                    refined_count += 1
                            completed = min(group_start + len(core), candidate_total)
                            self.work_queue.put((
                                "analysis_progress", generation, source_signature,
                                96.0 + completed / max(1, candidate_total) * 3.5,
                                f"原始素材精确复核 {completed}/{candidate_total}",
                            ))

                            # Preserve the current temporal neighborhood only.
                            # Candidate groups can be far apart in a long shoot,
                            # so keeping every demosaiced proxy wastes hundreds
                            # of megabytes without improving later results.
                            keep = {str(files[index]) for index in required[-(analysis_workers + 6):]}
                            for cached_path in list(precise_cache):
                                if cached_path not in keep:
                                    precise_cache.pop(cached_path, None)
                            trim_array_cache(precise_cache, self.preview_cache_budget)
                            refined_paths.update(str(files[index]) for index in core)
                            save_checkpoint("refine")
            finally:
                cv2.setNumThreads(previous_cv_threads)
            results = [
                results_by_index[index] for index in sorted(results_by_index)
            ]
            while len(cache) > 12:
                cache.popitem(last=False)
            trim_array_cache(cache, self.preview_cache_budget)
            if len(results) < 3:
                raise ValueError(
                    f"可读取的连续照片不足 3 张；发现 {len(files)} 个素材，"
                    f"其中 {len(decode_errors)} 个无法读取"
                )
            first_index = min(results_by_index)
            first_path = results_by_index[first_index].path
            if first_path not in cache and not token.cancelled:
                try:
                    first_preview = load_first_pass(first_index)
                    cache[first_path] = first_preview
                    shape = first_preview.shape[:2]
                except Exception:
                    first_preview = None
            if checkpoint_stage != "completed":
                mark_temporal_repeats(results, shape)
            save_checkpoint("completed")
            # The UI selects the first result after completion. Keep that proxy
            # ready so completion cannot synchronously decode a large RAW/TIFF
            # on the Tk thread after a long analysis.
            if first_path not in cache and first_preview is not None and not token.cancelled:
                cache[first_path] = first_preview
                cache.move_to_end(first_path)
                trim_array_cache(cache, self.preview_cache_budget)
            if not token.cancelled:
                self.work_queue.put((
                    "analysis_finished", generation, source_signature,
                    results, cache, decode_errors, {
                        "analysis_mode": analysis_mode,
                        "paired_count": sum(
                            source_item.proxy_path != source_item.path for source_item in sources
                        ),
                        "refined_count": len(refined_paths),
                        "analysis_errors": list(analysis_errors.items()),
                        "refine_errors": list(refine_errors.items()),
                    },
                ))
        except Exception as exc:
            import traceback
            if not token.cancelled:
                self.work_queue.put((
                    "analysis_error", generation, source_signature, str(exc), traceback.format_exc()
                ))

    def _automatic_decision(self, result: ScreeningResult) -> bool:
        return result.score >= self._score_cutoff()

    def _score_cutoff(self) -> int:
        return int(np.clip(100 - int(self.sensitivity.get()), 5, 95))

    def _effective_decision(self, result: ScreeningResult) -> bool:
        decision = self.decisions.get(result.path)
        return decision == "accept" if decision else self._automatic_decision(result)

    def _decision_label(self, result: ScreeningResult) -> str:
        manual = self.decisions.get(result.path)
        if self.decision_sources.get(result.path) == "candidate":
            return "候选保留" if manual == "accept" else "候选排除"
        if manual == "accept":
            return "人工保留"
        if manual == "reject":
            return "人工排除"
        return "自动保留" if self._automatic_decision(result) else "自动排除"

    def _decision_tag(self, result: ScreeningResult) -> str:
        if "飞机/卫星" in result.note:
            return "warning"
        manual = self.decisions.get(result.path)
        if self.decision_sources.get(result.path) == "candidate":
            return "candidate_accept" if manual == "accept" else "candidate_reject"
        if manual == "accept":
            return "manual_accept"
        if manual == "reject":
            return "manual_reject"
        return "auto_accept" if self._automatic_decision(result) else "auto_reject"

    def _result_matches_filters(self, result: ScreeningResult) -> bool:
        name_query = self.filter_name.get().strip().casefold()
        if name_query and name_query not in Path(result.path).name.casefold():
            return False
        status_filter = self.filter_status.get()
        if status_filter != "全部状态" and self._decision_label(result) != status_filter:
            return False
        low, high = self._filter_score_bounds()
        if not low <= int(result.score) <= high:
            return False
        label_filter = self.filter_label.get()
        labels = {candidate.label for candidate in result.candidates}
        if label_filter == "有候选" and not result.candidates:
            return False
        if label_filter == "无候选" and result.candidates:
            return False
        if label_filter == "未确认候选" and not any(
            not candidate.label for candidate in result.candidates
        ):
            return False
        if label_filter == "已确认流星" and "meteor" not in labels:
            return False
        if label_filter == "已确认误选" and "not_meteor" not in labels:
            return False
        if label_filter == "需人工复核" and not (
            "复核" in result.note or "分析失败" in result.note
        ):
            return False
        if label_filter == "飞机/卫星" and "飞机/卫星" not in result.note:
            return False
        return True

    def _filter_score_bounds(self) -> tuple[int, int]:
        try:
            first = int(self.filter_score_min.get())
            second = int(self.filter_score_max.get())
        except (TypeError, ValueError, tk.TclError):
            first, second = 0, 100
        return tuple(sorted((
            int(np.clip(first, 0, 100)),
            int(np.clip(second, 0, 100)),
        )))

    def _filters_changed(self, *_args) -> None:
        if self._restoring_autosave:
            return
        if self.filter_after_id is not None:
            try:
                self.after_cancel(self.filter_after_id)
            except tk.TclError:
                pass
        self.filter_after_id = self.after(120, self._apply_filters)

    def _reset_filters(self) -> None:
        self.filter_name.set("")
        self.filter_status.set("全部状态")
        self.filter_label.set("全部标签")
        self.filter_score_min.set(0)
        self.filter_score_max.set(100)
        if self.filter_after_id is not None:
            try:
                self.after_cancel(self.filter_after_id)
            except tk.TclError:
                pass
            self.filter_after_id = None
        self._apply_filters()

    def _apply_filters(self, preserve_selection: bool = True) -> None:
        self.filter_after_id = None
        selected_path = None
        selection = self.tree.selection()
        if preserve_selection and selection:
            index = int(selection[0])
            if 0 <= index < len(self.results):
                selected_path = self.results[index].path
        indices = [
            index for index, result in enumerate(self.results)
            if self._result_matches_filters(result)
        ]
        self.filtered_result_indices = indices
        self.tree.delete(*self.tree.get_children())
        for index in indices:
            result = self.results[index]
            self.tree.insert(
                "", "end", iid=str(index), text=Path(result.path).name,
                values=(
                    self._decision_label(result), result.score,
                    len(result.candidates), result.note or "—",
                ),
                tags=(self._decision_tag(result),),
            )
        chosen = next(
            (index for index in indices if self.results[index].path == selected_path),
            indices[0] if indices else None,
        )
        if chosen is not None:
            iid = str(chosen)
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
        else:
            self.preview_path = None
            self.preview_render_rgb = None
            self.preview_photo = None
            self.canvas.delete("all")
            self.candidate_status.set("当前筛选条件下没有照片。")
        self.filter_summary.set(f"显示 {len(indices)}/{len(self.results)}")
        kept = sum(self._effective_decision(item) for item in self.results)
        manual = sum(
            self.decision_sources.get(path, "manual") == "manual"
            for path in self.decisions
        )
        linked = sum(
            self.decision_sources.get(path) == "candidate" for path in self.decisions
        )
        exact = sum(
            candidate.label in {"meteor", "not_meteor"}
            for item in self.results for candidate in item.candidates
        )
        self.summary.set(
            f"显示 {len(indices)}/{len(self.results)} · 保留 {kept} · "
            f"照片手动 {manual} · 候选联动 {linked} · 候选确认 {exact}"
        )
        self._schedule_autosave()

    def _refresh_tree(self, preserve_selection: bool = True) -> None:
        self._apply_filters(preserve_selection)

    def _visible_preview_paths(self) -> list[str]:
        indices = self.filtered_result_indices
        if not indices and self.results:
            return []
        return [self.results[index].path for index in indices]

    def _sensitivity_changed(self, _value=None) -> None:
        value = int(self.sensitivity.get())
        if value < 30:
            self.sensitivity_hint.set("严格 · 误选更少")
        elif value > 60:
            self.sensitivity_hint.set("宽松 · 尽量不漏")
        else:
            self.sensitivity_hint.set("标准 · 平衡漏检和误选")
        if self.results:
            self._refresh_tree()
            self._show_selected()
        self._schedule_autosave()

    def _selected_result(self) -> ScreeningResult | None:
        selection = self.tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        return self.results[index] if 0 <= index < len(self.results) else None

    def _store_proxy_preview(self, path: str, image: np.ndarray) -> None:
        previous = self.preview_cache.pop(path, None)
        if previous is not None:
            self.preview_cache_bytes -= int(previous.nbytes)
        self.preview_cache[path] = image
        self.preview_cache_bytes += int(image.nbytes)
        self.preview_cache.move_to_end(path)
        while self.preview_cache_bytes > self.preview_cache_budget and len(self.preview_cache) > 1:
            _old_path, removed = self.preview_cache.popitem(last=False)
            self.preview_cache_bytes -= int(removed.nbytes)

    def _store_full_preview(self, path: str, image: np.ndarray) -> None:
        previous = self.full_preview_cache.pop(path, None)
        if previous is not None:
            self.full_preview_cache_bytes -= int(previous.nbytes)
        self.full_preview_cache[path] = image
        self.full_preview_cache_bytes += int(image.nbytes)
        self.full_preview_cache.move_to_end(path)
        protected = set(self.full_preview_window_paths)
        while (
            self.full_preview_cache_bytes > self.full_preview_cache_budget
            and len(self.full_preview_cache) > 1
        ):
            removable = [
                cached for cached in self.full_preview_cache
                if cached != self.preview_path and cached not in protected
            ]
            if not removable:
                removable = [
                    cached for cached in reversed(self.full_preview_window_paths)
                    if cached in self.full_preview_cache and cached != self.preview_path
                ]
            if not removable:
                break
            removed = self.full_preview_cache.pop(removable[0])
            self.full_preview_cache_bytes -= int(removed.nbytes)

    def _show_selected(self, _event=None) -> None:
        selection = self.tree.selection()
        if selection:
            # Scroll first, before cache lookup/RAW decoding, so keyboard review
            # never outruns the visible list position.
            keep_tree_row_in_navigation_runway(self.tree, selection[0])
        result = self._selected_result()
        if result is None:
            return
        if self.preview_path != result.path:
            self.preview_path = result.path
            self.preview_zoom = 1.0
            self.preview_pan_x = self.preview_pan_y = 0.0
            self.manual_mark_mode = False
            self.manual_mark_start = None
            self.manual_mark_label.set("手动标记漏检流星")
            if result.candidates:
                self.active_candidates.setdefault(result.path, 0)
        image = self.preview_cache.get(result.path)
        self._ensure_proxy_previews()
        if image is None:
            # Keep the last fully rendered frame until the newly selected one
            # is ready. Clearing the canvas here produced a black flash every
            # time keyboard navigation outran RAW/TIFF decoding. Canvas-to-image
            # editing is already guarded by the selected frame's missing cache,
            # so the retained pixels are display-only and cannot be written to
            # the wrong photo.
            suffix = "（暂时保留上一张画面）" if self.preview_render_rgb is not None else ""
            self.status.set(
                f"正在后台读取预览：{Path(result.path).name}…{suffix}"
            )
            return
        self.preview_cache.move_to_end(result.path)
        self._rebuild_preview_overlay()

    def _ensure_proxy_previews(self) -> None:
        current = self.preview_path
        if not current:
            return
        paths = self._visible_preview_paths()
        suffix = Path(current).suffix.lower()
        # Embedded RAW previews and JPEGs are cheap enough for a broad
        # navigation runway. TIFF must be fully read before downscaling, so a
        # smaller window avoids speculative I/O overwhelming foreground work.
        radius = 4 if suffix in {".tif", ".tiff"} else 8
        window = preview_window(paths, current, radius)
        missing = [
            path for path in window
            if path not in self.preview_cache
        ]
        if not missing:
            return
        if (
            self.proxy_preview_active_path == current
            and all(path in self.proxy_preview_loading for path in missing)
        ):
            return
        self.proxy_preview_active_path = current
        self.proxy_preview_loading = set(missing)
        generation = self.analysis_generation
        source_signature = str(Path(self.source_dir.get().strip()).expanduser())

        def worker(token: CancellationToken):
            for path in missing:
                if token.cancelled:
                    break
                try:
                    image = screening_fast_preview(Path(path), 1400)
                except Exception as exc:
                    self.work_queue.put((
                        "proxy_preview_error", generation, source_signature, path, str(exc),
                    ))
                    continue
                # The completed decode is still useful if navigation changed
                # while LibRaw was inside a non-interruptible native call.
                self.work_queue.put((
                    "proxy_preview", generation, source_signature, path, image,
                ))
            return None

        self.background_tasks.submit(
            "proxy_preview", worker, replace=True, retain_current=False,
        )

    def _original_preview_changed(self) -> None:
        self.preview_zoom = 1.0
        self.preview_pan_x = self.preview_pan_y = 0.0
        if self.use_original_preview.get():
            self._ensure_original_preview()
        self._rebuild_preview_overlay()
        self._schedule_autosave()

    def _candidate_marks_changed(self) -> None:
        self._rebuild_preview_overlay()
        self._schedule_autosave()

    def _ensure_original_preview(self) -> None:
        current = self.preview_path
        if not current:
            return
        paths = self._visible_preview_paths()
        current_image = self.full_preview_cache.get(current)
        estimated_bytes = (
            int(current_image.nbytes) if current_image is not None else 128 << 20
        )
        capacity = int(np.clip(
            self.full_preview_cache_budget // max(1, estimated_bytes), 3, 65,
        ))
        # Preload a wide navigation runway on high-memory machines. Completed
        # frames remain in the larger LRU beyond this window, so reviewing a
        # sequence back and forth normally becomes entirely memory-backed.
        radius = max(1, min(12, (capacity - 1) // 2))
        window = preview_window(paths, current, radius)
        missing = [path for path in window if path not in self.full_preview_cache]
        self.full_preview_window_paths = window
        if not missing:
            return
        if (
            self.full_preview_active_path == current
            and all(path in self.full_preview_loading for path in missing)
        ):
            return
        self.full_preview_active_path = current
        self.full_preview_loading = set(missing)
        self.status.set(f"正在读取原图精细预览：{Path(current).name}…")
        generation = self.analysis_generation
        source_signature = str(Path(self.source_dir.get().strip()).expanduser())

        def worker(token: CancellationToken):
            for path in missing:
                if token.cancelled:
                    break
                try:
                    image = read_screening_image(Path(path), None)
                except Exception as exc:
                    self.work_queue.put((
                        "full_preview_error", generation, source_signature, path, str(exc),
                    ))
                    continue
                self.work_queue.put((
                    "full_preview", generation, source_signature, path, image,
                ))
            return None

        self.background_tasks.submit(
            "full_preview", worker, replace=True, retain_current=False,
        )

    def _source_preview(self) -> np.ndarray | None:
        if not self.preview_path:
            return None
        if self.use_original_preview.get():
            original = self.full_preview_cache.get(self.preview_path)
            if original is not None:
                self.full_preview_cache.move_to_end(self.preview_path)
                return original
            self._ensure_original_preview()
        return self.preview_cache.get(self.preview_path)

    @staticmethod
    def _draw_candidate_marker(
        image: np.ndarray, candidate: ScreeningCandidate, color: tuple[int, int, int], active: bool,
        coordinate_scale: float = 1.0, selected: bool = False,
    ) -> None:
        start = np.asarray(candidate.start, dtype=np.float32)
        end = np.asarray(candidate.end, dtype=np.float32)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 2:
            return
        normal = np.asarray((-direction[1], direction[0]), np.float32) / length
        offset = max(7.0, 7.0 * coordinate_scale)
        thickness = max(1, round((2 if active else 1) * coordinate_scale))
        if selected:
            highlight = (70, 235, 255)
            highlight_thickness = max(thickness + 3, round(5 * coordinate_scale))
            for sign in (-1.0, 1.0):
                delta = normal * offset * sign
                a = tuple(np.round(start + delta).astype(int))
                b = tuple(np.round(end + delta).astype(int))
                cv2.line(
                    image, a, b, highlight, highlight_thickness, cv2.LINE_AA,
                )
        for sign in (-1.0, 1.0):
            delta = normal * offset * sign
            a = tuple(np.round(start + delta).astype(int))
            b = tuple(np.round(end + delta).astype(int))
            cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
        # Short end caps make the selected extent visible while leaving the
        # meteor core and tail unobscured between the two guide lines.
        for point in (start, end):
            a = tuple(np.round(point - normal * offset).astype(int))
            b = tuple(np.round(point + normal * offset).astype(int))
            cv2.line(image, a, b, color, thickness, cv2.LINE_AA)
        if selected:
            badge = start - normal * max(18.0, 18.0 * coordinate_scale)
            center = tuple(np.round(badge).astype(int))
            radius = max(7, round(8 * coordinate_scale))
            cv2.circle(image, center, radius, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(image, center, radius, (70, 235, 255), max(2, round(2 * coordinate_scale)), cv2.LINE_AA)
            p1 = tuple(np.round(badge + np.asarray((-radius * 0.45, 0.0))).astype(int))
            p2 = tuple(np.round(badge + np.asarray((-radius * 0.10, radius * 0.38))).astype(int))
            p3 = tuple(np.round(badge + np.asarray((radius * 0.50, -radius * 0.40))).astype(int))
            cv2.line(image, p1, p2, (70, 235, 255), max(2, round(2 * coordinate_scale)), cv2.LINE_AA)
            cv2.line(image, p2, p3, (70, 235, 255), max(2, round(2 * coordinate_scale)), cv2.LINE_AA)
        midpoint = (start + end) * 0.5 + normal * max(14.0, 14.0 * coordinate_scale)
        label = f"{candidate.score}%"
        if candidate.label == "meteor":
            label = "流星 " + label
        elif candidate.label == "not_meteor":
            label = "误选 " + label
        elif candidate.score < 20:
            label = "LOW " + label
        else:
            label = "CANDIDATE " + label
        origin = tuple(np.round(midpoint).astype(int))
        font_scale = max(0.52, 0.52 * coordinate_scale)
        cv2.putText(
            image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
            max(3, round(4 * coordinate_scale)), cv2.LINE_AA,
        )
        cv2.putText(
            image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
            max(1, round(coordinate_scale)), cv2.LINE_AA,
        )

    def _rebuild_preview_overlay(self) -> None:
        source = self._source_preview()
        result = self._selected_result()
        if source is None or result is None:
            return
        image = source.copy()
        analysis = self.preview_cache.get(result.path)
        if analysis is None:
            return
        scale_x = source.shape[1] / max(1, analysis.shape[1])
        scale_y = source.shape[0] / max(1, analysis.shape[0])
        coordinate_scale = float((scale_x + scale_y) * 0.5)
        marks_visible = self.show_candidate_marks.get() and not self.candidate_marks_temporarily_hidden
        active_index = self.active_candidates.get(result.path, 0)
        selected_indices = self.selected_candidates.get(result.path, set())
        if marks_visible:
            for index, candidate in enumerate(result.candidates):
                if candidate.label == "meteor":
                    color = (55, 230, 95)
                elif candidate.label == "not_meteor":
                    color = (235, 90, 80)
                elif candidate.score < 20:
                    color = (145, 155, 165)
                else:
                    color = (255, 190, 45)
                display_candidate = ScreeningCandidate(
                    (round(candidate.start[0] * scale_x), round(candidate.start[1] * scale_y)),
                    (round(candidate.end[0] * scale_x), round(candidate.end[1] * scale_y)),
                    candidate.score, legacy_score=candidate.legacy_score,
                    label=candidate.label, manual=candidate.manual,
                )
                self._draw_candidate_marker(
                    image, display_candidate, color,
                    index == active_index, coordinate_scale,
                    selected=index in selected_indices,
                )
            if self.manual_mark_start is not None:
                display_start = (
                    round(self.manual_mark_start[0] * scale_x),
                    round(self.manual_mark_start[1] * scale_y),
                )
                cv2.circle(
                    image, display_start, max(8, round(8 * coordinate_scale)),
                    (70, 210, 255), max(2, round(2 * coordinate_scale)), cv2.LINE_AA,
                )
        self.preview_render_rgb = image
        self._render_canvas()
        if result.candidates and 0 <= active_index < len(result.candidates):
            candidate = result.candidates[active_index]
            if candidate.label == "meteor":
                state = "已确认是流星"
            elif candidate.label == "not_meteor":
                state = "已确认是误选"
            elif candidate.score < 20:
                state = "低置信候选，并非已确认流星"
            else:
                state = "未确认候选，并非已确认流星"
            selected_note = (
                f" · 已选择 {len(selected_indices)} 条" if selected_indices else ""
            )
            self.candidate_status.set(
                f"第 {active_index + 1}/{len(result.candidates)} 条 · "
                f"AI {candidate.score}% · {state}{selected_note}"
            )
        else:
            self.candidate_status.set("没有自动候选；可用“手动标记漏检流星”补充。")
        if self.use_original_preview.get() and result.path in self.full_preview_cache:
            quality = f"原图 {source.shape[1]}×{source.shape[0]}"
        elif self.use_original_preview.get():
            quality = "原图载入中（暂显快速预览）"
        else:
            quality = "快速预览"
        self.status.set(
            f"{Path(result.path).name} · {self._decision_label(result)} · "
            f"{quality} · AI流星可能性 {result.score}% · 照片判断不会作为候选训练标签"
        )

    def _view_geometry(self) -> tuple[float, float, float, int, int] | None:
        if self.preview_render_rgb is None:
            return None
        height, width = self.preview_render_rgb.shape[:2]
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        fit = min(canvas_width / max(1, width), canvas_height / max(1, height))
        scale = max(0.01, fit * self.preview_zoom)
        scaled_width, scaled_height = width * scale, height * scale
        max_pan_x = max(0.0, (scaled_width - canvas_width) * 0.5)
        max_pan_y = max(0.0, (scaled_height - canvas_height) * 0.5)
        self.preview_pan_x = float(np.clip(self.preview_pan_x, -max_pan_x, max_pan_x))
        self.preview_pan_y = float(np.clip(self.preview_pan_y, -max_pan_y, max_pan_y))
        left = canvas_width * 0.5 + self.preview_pan_x - scaled_width * 0.5
        top = canvas_height * 0.5 + self.preview_pan_y - scaled_height * 0.5
        return scale, left, top, canvas_width, canvas_height

    def _render_canvas(self) -> None:
        geometry = self._view_geometry()
        if geometry is None or self.preview_render_rgb is None:
            return
        scale, left, top, canvas_width, canvas_height = geometry
        height, width = self.preview_render_rgb.shape[:2]
        x0 = int(np.clip(np.floor((0.0 - left) / scale), 0, width - 1))
        y0 = int(np.clip(np.floor((0.0 - top) / scale), 0, height - 1))
        x1 = int(np.clip(np.ceil((canvas_width - left) / scale), x0 + 1, width))
        y1 = int(np.clip(np.ceil((canvas_height - top) / scale), y0 + 1, height))
        crop = self.preview_render_rgb[y0:y1, x0:x1]
        draw_width = max(1, round((x1 - x0) * scale))
        draw_height = max(1, round((y1 - y0) * scale))
        interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
        shown = cv2.resize(crop, (draw_width, draw_height), interpolation=interpolation)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(shown))
        self.canvas.delete("all")
        self.canvas.create_image(left + x0 * scale, top + y0 * scale, image=self.preview_photo, anchor="nw")

    def _canvas_configured(self, _event=None) -> None:
        if self.preview_resize_after is not None:
            try:
                self.after_cancel(self.preview_resize_after)
            except tk.TclError:
                pass
        self.preview_resize_after = self.after(30, self._finish_canvas_resize)

    def _finish_canvas_resize(self) -> None:
        self.preview_resize_after = None
        self._render_canvas()

    def _fit_preview(self) -> None:
        self.preview_zoom = 1.0
        self.preview_pan_x = self.preview_pan_y = 0.0
        self._render_canvas()

    def _actual_size_preview(self) -> None:
        source = self._source_preview()
        if source is None:
            return
        self.canvas.update_idletasks()
        fit = min(
            max(1, self.canvas.winfo_width()) / source.shape[1],
            max(1, self.canvas.winfo_height()) / source.shape[0],
        )
        self.preview_zoom = float(np.clip(1.0 / max(fit, 1e-6), 1.0, 16.0))
        self.preview_pan_x = self.preview_pan_y = 0.0
        self._render_canvas()

    def _canvas_wheel(self, event, direction: int | None = None):
        if self.preview_render_rgb is None:
            return "break"
        if direction is None:
            direction = 1 if event.delta > 0 else -1
        old_geometry = self._view_geometry()
        if old_geometry is None:
            return "break"
        factor = 1.18 if direction > 0 else 1.0 / 1.18
        old_zoom = self.preview_zoom
        self.preview_zoom = float(np.clip(old_zoom * factor, 1.0, 16.0))
        actual_factor = self.preview_zoom / max(old_zoom, 1e-6)
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        old_center_x = canvas_width * 0.5 + self.preview_pan_x
        old_center_y = canvas_height * 0.5 + self.preview_pan_y
        new_center_x = event.x - (event.x - old_center_x) * actual_factor
        new_center_y = event.y - (event.y - old_center_y) * actual_factor
        self.preview_pan_x = new_center_x - canvas_width * 0.5
        self.preview_pan_y = new_center_y - canvas_height * 0.5
        self._render_canvas()
        return "break"

    def _canvas_to_image(self, x: float, y: float) -> tuple[int, int] | None:
        geometry = self._view_geometry()
        source = self._source_preview()
        analysis = self.preview_cache.get(self.preview_path or "")
        if geometry is None or source is None or analysis is None:
            return None
        scale, left, top, _cw, _ch = geometry
        ix = int(round((x - left) / scale))
        iy = int(round((y - top) / scale))
        if 0 <= ix < source.shape[1] and 0 <= iy < source.shape[0]:
            return (
                int(np.clip(round(ix * analysis.shape[1] / source.shape[1]), 0, analysis.shape[1] - 1)),
                int(np.clip(round(iy * analysis.shape[0] / source.shape[0]), 0, analysis.shape[0] - 1)),
            )
        return None

    def _canvas_press(self, event) -> None:
        self.canvas.focus_set()
        self.preview_drag_start = (event.x, event.y)
        self.preview_drag_origin = (self.preview_pan_x, self.preview_pan_y)
        self.preview_dragged = False

    def _canvas_drag(self, event) -> None:
        if self.preview_drag_start is None or self.preview_drag_origin is None:
            return
        dx = event.x - self.preview_drag_start[0]
        dy = event.y - self.preview_drag_start[1]
        if abs(dx) + abs(dy) >= 4:
            self.preview_dragged = True
        if self.preview_zoom > 1.0:
            self.preview_pan_x = self.preview_drag_origin[0] + dx
            self.preview_pan_y = self.preview_drag_origin[1] + dy
            self._render_canvas()

    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        delta = end - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 1e-6:
            return float(np.linalg.norm(point - start))
        amount = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + amount * delta)))

    def _canvas_release(self, event) -> None:
        dragged = self.preview_dragged
        self.preview_drag_start = None
        self.preview_drag_origin = None
        self.preview_dragged = False
        if dragged:
            return
        point = self._canvas_to_image(event.x, event.y)
        result = self._selected_result()
        if point is None or result is None:
            return
        if self.manual_mark_mode:
            if self.manual_mark_start is None:
                self.manual_mark_start = point
                self.manual_mark_label.set("点击流星终点（Esc取消）")
            else:
                if np.hypot(point[0] - self.manual_mark_start[0], point[1] - self.manual_mark_start[1]) >= 6:
                    result.candidates.append(ScreeningCandidate(
                        self.manual_mark_start, point, 100, legacy_score=100.0,
                        label="meteor", manual=True,
                    ))
                    result.score = max(result.score, 100)
                    self.active_candidates[result.path] = len(result.candidates) - 1
                    self.selected_candidates[result.path] = {
                        len(result.candidates) - 1
                    }
                    self._sync_photo_decision_from_candidates(result)
                    self._record_screening_diagnostic(
                        result, len(result.candidates) - 1,
                        "false_negative_manual_mark",
                    )
                self.manual_mark_mode = False
                self.manual_mark_start = None
                self.manual_mark_label.set("手动标记漏检流星")
                self._refresh_tree()
                self._schedule_autosave()
            self._rebuild_preview_overlay()
            return
        if not result.candidates:
            return
        image_point = np.asarray(point, np.float32)
        distances = [
            self._point_segment_distance(
                image_point, np.asarray(candidate.start, np.float32), np.asarray(candidate.end, np.float32)
            ) for candidate in result.candidates
        ]
        index = int(np.argmin(distances))
        geometry = self._view_geometry()
        source = self._source_preview()
        analysis = self.preview_cache.get(result.path)
        if geometry and source is not None and analysis is not None:
            analysis_to_display = geometry[0] * source.shape[1] / max(1, analysis.shape[1])
            tolerance = 18.0 / max(analysis_to_display, 1e-6)
        else:
            tolerance = 18.0
        if distances[index] <= tolerance:
            self.active_candidates[result.path] = index
            selected = self.selected_candidates.setdefault(result.path, set())
            if self.candidate_multi_select_mode.get():
                if index in selected:
                    selected.remove(index)
                else:
                    selected.add(index)
                if not selected:
                    self.selected_candidates.pop(result.path, None)
            else:
                self.selected_candidates[result.path] = {index}
            self._rebuild_preview_overlay()

    def _toggle_manual_mark(self) -> None:
        self.manual_mark_mode = not self.manual_mark_mode
        self.manual_mark_start = None
        self.manual_mark_label.set("点击流星起点" if self.manual_mark_mode else "手动标记漏检流星")
        self._rebuild_preview_overlay()

    def _cancel_manual_mark(self, _event=None) -> None:
        if not self.manual_mark_mode and self.manual_mark_start is None:
            return
        self.manual_mark_mode = False
        self.manual_mark_start = None
        self.manual_mark_label.set("手动标记漏检流星")
        self._rebuild_preview_overlay()

    def _active_candidate(self) -> ScreeningCandidate | None:
        result = self._selected_result()
        if result is None or not result.candidates:
            return None
        index = self.active_candidates.get(result.path, 0)
        return result.candidates[index] if 0 <= index < len(result.candidates) else None

    def _candidate_indices_for_action(self, result: ScreeningResult) -> list[int]:
        selected = sorted(
            index for index in self.selected_candidates.get(result.path, set())
            if 0 <= index < len(result.candidates)
        )
        if selected:
            return selected
        active = self.active_candidates.get(result.path, 0)
        return [active] if 0 <= active < len(result.candidates) else []

    def _candidate_selection_mode_changed(self) -> None:
        if not self.candidate_multi_select_mode.get():
            result = self._selected_result()
            if result is not None:
                selected = self.selected_candidates.get(result.path, set())
                if len(selected) > 1:
                    active = self.active_candidates.get(result.path, min(selected))
                    self.selected_candidates[result.path] = {
                        active if active in selected else min(selected)
                    }
        self._rebuild_preview_overlay()

    def _select_all_candidates(self) -> None:
        result = self._selected_result()
        if result is None or not result.candidates:
            return
        self.selected_candidates[result.path] = set(range(len(result.candidates)))
        self.active_candidates[result.path] = 0
        self.candidate_multi_select_mode.set(True)
        self._rebuild_preview_overlay()

    @staticmethod
    def _recalculate_candidate_score(result: ScreeningResult) -> None:
        score = max((candidate.score for candidate in result.candidates), default=0)
        if result.temporal_hits >= 2:
            score = max(0, score - 25)
        result.score = int(np.clip(score, 0, 100))

    def _remove_candidate_indices(
        self, result: ScreeningResult, indices: set[int],
    ) -> int:
        valid = {
            index for index in indices if 0 <= index < len(result.candidates)
        }
        if not valid:
            return 0
        result.candidates[:] = [
            candidate for index, candidate in enumerate(result.candidates)
            if index not in valid
        ]
        self.selected_candidates.pop(result.path, None)
        if result.candidates:
            self.active_candidates[result.path] = 0
        else:
            self.active_candidates.pop(result.path, None)
        self._recalculate_candidate_score(result)
        self._sync_photo_decision_from_candidates(result)
        return len(valid)

    def _finish_candidate_removal(self, result: ScreeningResult, removed: int) -> None:
        if not removed:
            return
        self._refresh_tree()
        self._show_selected()
        self._schedule_autosave()
        self.status.set(
            f"已从 {Path(result.path).name} 清除 {removed} 条候选；原始照片未修改"
        )

    def _remove_selected_candidates(self, _event=None):
        result = self._selected_result()
        if result is None:
            return "break"
        selected = self.selected_candidates.get(result.path, set())
        if not selected:
            messagebox.showinfo(
                "清除候选", "请先点击候选，或使用“全选”。", parent=self,
            )
            return "break"
        removed = self._remove_candidate_indices(result, set(selected))
        self._finish_candidate_removal(result, removed)
        return "break"

    def _remove_all_candidates(self) -> None:
        result = self._selected_result()
        if result is None or not result.candidates:
            return
        removed = self._remove_candidate_indices(
            result, set(range(len(result.candidates))),
        )
        self._finish_candidate_removal(result, removed)

    def _sync_photo_decision_from_candidates(self, result: ScreeningResult) -> None:
        labels = [candidate.label for candidate in result.candidates]
        if any(label == "meteor" for label in labels):
            self.decisions[result.path] = "accept"
            self.decision_sources[result.path] = "candidate"
        elif labels and all(label == "not_meteor" for label in labels):
            self.decisions[result.path] = "reject"
            self.decision_sources[result.path] = "candidate"
        elif self.decision_sources.get(result.path) == "candidate":
            self.decisions.pop(result.path, None)
            self.decision_sources.pop(result.path, None)

    def _label_candidate(self, label: str) -> None:
        result = self._selected_result()
        if result is None:
            return
        indices = self._candidate_indices_for_action(result)
        if not indices:
            messagebox.showinfo("候选确认", "请先在预览中点击一条候选标记", parent=self)
            return
        for index in indices:
            result.candidates[index].label = label
            if label == "not_meteor":
                self._record_screening_diagnostic(
                    result, index, "false_positive_rejected_candidate",
                )
            elif label == "meteor" and result.candidates[index].score < self._score_cutoff():
                self._record_screening_diagnostic(
                    result, index, "false_negative_under_ranked_candidate",
                )
        self._sync_photo_decision_from_candidates(result)
        self._refresh_tree()
        self._rebuild_preview_overlay()
        self._schedule_autosave()
        action = {
            "meteor": "标记为流星",
            "not_meteor": "标记为误选",
            "": "清除判断（候选仍保留）",
        }[label]
        self.status.set(f"已对 {len(indices)} 条候选执行：{action}")

    def _record_screening_diagnostic(
        self, result: ScreeningResult, candidate_index: int, error_type: str,
    ) -> Path | None:
        """Persist enough state for a developer to reproduce one wrong decision."""
        try:
            candidate = result.candidates[candidate_index]
            result_index = next(
                (index for index, item in enumerate(self.results) if item.path == result.path),
                -1,
            )
            neighbor_paths = []
            if result_index >= 0:
                neighbor_paths = [
                    str(self.files[index])
                    for index in range(
                        max(0, result_index - 3), min(len(self.files), result_index + 4),
                    )
                    if index != result_index
                ]
            recorded_at = datetime.now().isoformat(timespec="milliseconds")
            feature_values = [float(value) for value in candidate.features]
            named_features = {
                name: feature_values[index]
                for index, name in enumerate(ML_FEATURE_NAMES)
                if index < len(feature_values)
            }
            event_identity = (
                f"{recorded_at}|{error_type}|{result.path}|{candidate_index}|"
                f"{candidate.start}|{candidate.end}"
            )
            record = {
                "id": hashlib.sha256(event_identity.encode("utf-8")).hexdigest(),
                "recorded_at": recorded_at,
                "error_type": error_type,
                "suspected_stage": (
                    "candidate_generation"
                    if error_type == "false_negative_manual_mark"
                    else "candidate_ranking_or_suppression"
                    if error_type == "false_negative_under_ranked_candidate"
                    else "candidate_generation_or_classification"
                ),
                "algorithm_version": SCREENING_ALGORITHM_VERSION,
                "analysis_mode": self.analysis_mode.get(),
                "sensitivity": int(self.sensitivity.get()),
                "score_cutoff": int(self._score_cutoff()),
                "source_path": result.path,
                "neighbor_paths": neighbor_paths,
                "source_files_are_read_only_references": True,
                "result_snapshot": {
                    "score": int(result.score),
                    "plane_count": int(result.plane_count),
                    "temporal_hits": int(result.temporal_hits),
                    "note": result.note,
                },
                "reviewed_candidate": {
                    "candidate_index": int(candidate_index),
                    "start": list(candidate.start),
                    "end": list(candidate.end),
                    "score": int(candidate.score),
                    "legacy_score": float(candidate.legacy_score),
                    "manual": bool(candidate.manual),
                    "user_label": candidate.label,
                    "features": feature_values,
                    "named_features": named_features,
                },
                "all_candidates": [asdict(item) for item in result.candidates],
            }
            return save_screening_diagnostic_event(record)
        except Exception as exc:
            append_runtime_log("保存筛选误判诊断记录失败", str(exc))
            return None

    def _open_diagnostic_folder(self) -> None:
        path = screening_diagnostics_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        open_folder(path.parent)
        self.status.set(f"误判诊断记录：{path}")

    def _hide_candidate_marks(self, _event=None):
        if not self.candidate_marks_temporarily_hidden:
            self.candidate_marks_temporarily_hidden = True
            self._rebuild_preview_overlay()

    def _show_candidate_marks(self, _event=None):
        if self.candidate_marks_temporarily_hidden:
            self.candidate_marks_temporarily_hidden = False
            self._rebuild_preview_overlay()

    def _set_decision(self, value: str | None) -> None:
        result = self._selected_result()
        if result is None:
            return
        if value is None:
            self.decisions.pop(result.path, None)
            self.decision_sources.pop(result.path, None)
        else:
            self.decisions[result.path] = value
            self.decision_sources[result.path] = "manual"
        self._refresh_tree()
        self._show_selected()
        self._schedule_autosave()

    def accept_selected(self) -> None:
        self._set_decision("accept")

    def reject_selected(self) -> None:
        self._set_decision("reject")

    def reset_selected(self) -> None:
        self._set_decision(None)

    def copy_selected(self) -> None:
        if self.export_running:
            self.status.set("导出任务正在运行，请等待完成")
            return
        if not self.results:
            messagebox.showwarning("流星批量筛选", "请先完成分析", parent=self)
            return
        source = Path(self.source_dir.get().strip()).expanduser()
        output_text = self.output_dir.get().strip()
        output = (
            Path(output_text).expanduser()
            if output_text else source / "MeteorStudio_Output"
        )
        if not output_text:
            self.output_dir.set(str(output))
        output.mkdir(parents=True, exist_ok=True)
        if not output.is_dir():
            show_copyable_error("流星批量筛选", "请选择有效的筛选结果保存位置", parent=self)
            return
        selected = [item for item in self.results if self._effective_decision(item)]
        if not selected:
            messagebox.showwarning("流星批量筛选", "当前没有保留的照片", parent=self)
            return
        run_dir = output / ("meteor_selected_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        run_dir.mkdir(parents=True, exist_ok=False)
        explicit_feedback = []
        result_indices = {item.path: index for index, item in enumerate(self.results)}
        for result in self.results:
            index = result_indices[result.path]
            neighbor_paths = [
                str(self.files[other])
                for other in range(max(0, index - 3), min(len(self.files), index + 4))
                if other != index
            ]
            for candidate_index, candidate in enumerate(result.candidates):
                if candidate.label not in {"meteor", "not_meteor"}:
                    continue
                identity = (
                    f"{result.path}|{candidate.start[0]},{candidate.start[1]}|"
                    f"{candidate.end[0]},{candidate.end[1]}"
                )
                explicit_feedback.append({
                    "id": identity,
                    "source_path": result.path,
                    "neighbor_paths": neighbor_paths,
                    "candidate_index": candidate_index,
                    "start": list(candidate.start), "end": list(candidate.end),
                    "label": 1 if candidate.label == "meteor" else 0,
                    "legacy": float(candidate.legacy_score) / 100.0,
                    "features": candidate.features,
                    "manual": bool(candidate.manual),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })
        selected_snapshot = [
            (
                Path(result.path), asdict(result), self._decision_label(result),
                self.decision_sources.get(result.path, "automatic"),
            )
            for result in selected
        ]
        # Legacy autosaves may not contain per-result analysis dimensions. The
        # in-memory proxy cache already has the exact screening coordinate size;
        # reuse it instead of reopening and decoding every copied RAW merely to
        # normalize two candidate endpoints for the JSON sidecar.
        fallback_preview_size = next((
            (int(result.analysis_width), int(result.analysis_height))
            for result in self.results
            if result.analysis_width > 1 and result.analysis_height > 1
        ), None)
        if fallback_preview_size is None:
            fallback_preview_size = next((
            (int(image.shape[1]), int(image.shape[0]))
            for image in self.preview_cache.values()
            if image is not None and image.ndim >= 2
            ), None)
        report_header = {
            "source_folder": str(source),
            "screening_sensitivity": int(self.sensitivity.get()),
            "sensitivity_description": self.sensitivity_hint.get(),
            "internal_score_cutoff": self._score_cutoff(),
            "selected_count": len(selected),
            "candidate_feedback_count": len(explicit_feedback),
            "learning_rule": "Only explicitly labeled candidates train the AI; image keep/reject decisions never do.",
        }
        self.export_running = True
        self.export_button.configure(state="disabled", text="导出中…")
        self.status.set(f"正在后台导出 {len(selected)} 张原图…")
        self.background_tasks.submit(
            "export",
            lambda token: self._copy_selected_worker(
                run_dir, selected_snapshot, explicit_feedback, report_header, token,
                fallback_preview_size,
            ),
            replace=True,
        )

    def _copy_selected_worker(
        self, run_dir: Path, selected: list[tuple[Path, dict, str, str]],
        explicit_feedback: list[dict], report_header: dict, token: CancellationToken,
        fallback_preview_size: tuple[int, int] | None = None,
    ) -> None:
        try:
            feedback_path = save_screening_feedback(explicit_feedback)
            report = []
            confirmed_track_items = []
            sizes = []
            for source_path, _item_data, _decision, _decision_source in selected:
                try:
                    sizes.append(int(source_path.stat().st_size))
                except OSError:
                    sizes.append(0)
            total_bytes = sum(sizes)
            copied_bytes = 0
            total_gib = total_bytes / float(1 << 30)
            self.work_queue.put((
                "export_progress", 0.0,
                f"准备复制 {len(selected)} 张原始文件，共 {total_gib:.2f} GiB；不会重新分析或解码",
            ))
            for index, (source_path, item_data, decision, decision_source) in enumerate(selected, start=1):
                if token.cancelled:
                    return
                destination = run_dir / source_path.name
                shutil.copy2(source_path, destination)
                copied_bytes += sizes[index - 1]
                track_item = confirmed_track_item(
                    source_path, destination.name, item_data, fallback_preview_size,
                )
                if track_item is not None:
                    confirmed_track_items.append(track_item)
                report.append({
                    **item_data, "copied_to": str(destination),
                    "decision": decision, "decision_source": decision_source,
                })
                self.work_queue.put((
                    "export_progress", index / max(1, len(selected)) * 100,
                    f"正在复制原图 {index}/{len(selected)} · "
                    f"{copied_bytes / float(1 << 30):.2f}/{total_gib:.2f} GiB：{source_path.name}",
                ))
            tracks_path = run_dir / "confirmed_meteor_tracks.json"
            tracks_path.write_text(json.dumps({
                "format": "meteor-confirmed-tracks-v1",
                "algorithm_version": SCREENING_ALGORITHM_VERSION,
                "coordinate_space": "normalized_source_image",
                "items": confirmed_track_items,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            payload = {
                **report_header,
                "candidate_feedback_file": str(feedback_path) if feedback_path else None,
                "confirmed_meteor_tracks_file": str(tracks_path),
                "confirmed_meteor_track_count": sum(
                    len(item["tracks"]) for item in confirmed_track_items
                ),
                "items": report,
            }
            (run_dir / "screening_report.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            if not token.cancelled:
                self.work_queue.put((
                    "screening_exported", run_dir, len(selected), len(explicit_feedback)
                ))
        except Exception as exc:
            import traceback
            if not token.cancelled:
                self.work_queue.put(("screening_export_error", str(exc), traceback.format_exc()))

    def _open_export_folder(self) -> None:
        path = self.last_export_dir.get().strip() or self.output_dir.get().strip()
        try:
            open_folder(path)
        except Exception as exc:
            show_copyable_error("打开文件夹", str(exc), parent=self)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.work_queue.get_nowait()
                if item[0] == "analysis_progress":
                    _, generation, source_signature, value, text = item
                    if (
                        generation != self.analysis_generation
                        or source_signature != str(Path(self.source_dir.get().strip()).expanduser())
                    ):
                        continue
                    self.progress["value"] = value
                    self.status.set(text)
                elif item[0] == "export_progress":
                    _, value, text = item
                    self.progress["value"] = value
                    self.status.set(text)
                elif item[0] == "analysis_finished":
                    (
                        _, generation, source_signature, results, preview_cache,
                        decode_errors, analysis_stats,
                    ) = item
                    if generation != self.analysis_generation:
                        continue
                    self.analysis_running = False
                    self.analyze_button.configure(state="normal", text="开始分析")
                    if source_signature != str(Path(self.source_dir.get().strip()).expanduser()):
                        self.status.set("分析期间照片文件夹已变化，旧结果已丢弃")
                        continue
                    self.results = results
                    self.files = [Path(result.path) for result in results]
                    self.preview_cache = preview_cache
                    self.preview_cache_bytes = trim_array_cache(
                        self.preview_cache, self.preview_cache_budget
                    )
                    self.progress["value"] = 100
                    self._refresh_tree(False)
                    self._show_selected()
                    kept = sum(self._effective_decision(result) for result in self.results)
                    skipped_note = ""
                    if decode_errors:
                        skipped_note = f"；跳过 {len(decode_errors)} 个无法读取文件（详情见运行日志）"
                        append_runtime_log(
                            f"流星批量筛选跳过 {len(decode_errors)} 个无法读取文件",
                            "\n".join(
                                f"{path}\n  {message}" for path, message in decode_errors
                            ),
                        )
                    analysis_errors = analysis_stats.get("analysis_errors", [])
                    if analysis_errors:
                        skipped_note += (
                            f"；{len(analysis_errors)} 张 OpenCV 分析重试后仍失败，"
                            "已默认保留并要求人工复核（详情见运行日志）"
                        )
                        append_runtime_log(
                            f"流星批量筛选分析失败 {len(analysis_errors)} 张",
                            "\n".join(
                                f"{path}\n  {message}" for path, message in analysis_errors
                            ),
                        )
                    refine_errors = analysis_stats.get("refine_errors", [])
                    if refine_errors:
                        skipped_note += (
                            f"；{len(refine_errors)} 张原始素材精确复核失败，"
                            "已保留快速预筛结果（详情见运行日志）"
                        )
                        append_runtime_log(
                            f"流星批量筛选精确复核失败 {len(refine_errors)} 张",
                            "\n".join(
                                f"{path}\n  {message}" for path, message in refine_errors
                            ),
                        )
                    mode_note = f"{analysis_stats.get('analysis_mode', '标准两阶段')}"
                    paired_count = int(analysis_stats.get("paired_count", 0))
                    refined_count = int(analysis_stats.get("refined_count", 0))
                    if paired_count:
                        mode_note += f"，合并 {paired_count} 组同名 RAW/JPG"
                    if refined_count:
                        mode_note += f"，精确复核 {refined_count} 张"
                    self.status.set(
                        f"分析完成：{len(self.results)} 张中自动保留 {kept} 张，"
                        f"请逐张复核（{mode_note}）{skipped_note}"
                    )
                    # Persist the completed UI state before removing the more
                    # granular worker checkpoint. If autosave itself fails, the
                    # completed checkpoint remains recoverable on next launch.
                    if self._save_autosave():
                        try:
                            screening_analysis_checkpoint_file_path().unlink(missing_ok=True)
                        except OSError as exc:
                            append_runtime_log("清理筛选分析断点失败", str(exc))
                elif item[0] == "proxy_preview":
                    _, generation, source_signature, path, image = item
                    if (
                        generation != self.analysis_generation
                        or source_signature != str(Path(self.source_dir.get().strip()).expanduser())
                        or path not in {str(value) for value in self.files}
                    ):
                        self.proxy_preview_loading.discard(path)
                        continue
                    self.proxy_preview_loading.discard(path)
                    self._store_proxy_preview(path, image)
                    if path == self.preview_path:
                        self._rebuild_preview_overlay()
                elif item[0] == "proxy_preview_error":
                    _, generation, source_signature, path, message = item
                    self.proxy_preview_loading.discard(path)
                    if (
                        generation == self.analysis_generation
                        and source_signature == str(Path(self.source_dir.get().strip()).expanduser())
                        and path == self.preview_path
                    ):
                        self.status.set(f"预览读取失败：{message}")
                elif item[0] == "full_preview":
                    _, generation, source_signature, path, image = item
                    if (
                        generation != self.analysis_generation
                        or source_signature != str(Path(self.source_dir.get().strip()).expanduser())
                        or path not in {str(value) for value in self.files}
                    ):
                        self.full_preview_loading.discard(path)
                        if self.full_preview_active_path == path:
                            self.full_preview_active_path = None
                        continue
                    self.full_preview_loading.discard(path)
                    self._store_full_preview(path, image)
                    if path == self.preview_path and self.use_original_preview.get():
                        self.preview_zoom = 1.0
                        self.preview_pan_x = self.preview_pan_y = 0.0
                        self._rebuild_preview_overlay()
                        self.status.set(
                            f"原图精细预览：{Path(path).name} · "
                            f"{image.shape[1]}×{image.shape[0]} · 滚轮放大或点击 1:1"
                        )
                elif item[0] == "full_preview_error":
                    _, generation, source_signature, path, message = item
                    if (
                        generation != self.analysis_generation
                        or source_signature != str(Path(self.source_dir.get().strip()).expanduser())
                    ):
                        self.full_preview_loading.discard(path)
                        if self.full_preview_active_path == path:
                            self.full_preview_active_path = None
                        continue
                    self.full_preview_loading.discard(path)
                    if path == self.preview_path:
                        self.use_original_preview.set(False)
                        self._rebuild_preview_overlay()
                        self.status.set(f"原图精细预览读取失败：{message}")
                elif item[0] == "analysis_error":
                    _, generation, _source_signature, message, details = item
                    if generation != self.analysis_generation:
                        continue
                    self.analysis_running = False
                    self.analyze_button.configure(state="normal", text="开始分析")
                    self.status.set("分析失败：" + message)
                    show_copyable_error(
                        "流星批量筛选", message, parent=self, details=details
                    )
                elif item[0] == "screening_exported":
                    _, run_dir, count, feedback_count = item
                    self.export_running = False
                    self.export_button.configure(state="normal", text="导出已选流星照片")
                    self.progress["value"] = 100
                    self.last_export_dir.set(str(run_dir))
                    self._save_autosave()
                    feedback_note = f"；保存 {feedback_count} 条候选级模型反馈" if feedback_count else ""
                    self.status.set(f"已导出 {count} 张；实际导出位置：{run_dir}{feedback_note}")
                    if messagebox.askyesno(
                        "流星批量筛选",
                        f"已导出 {count} 张流星照片。{feedback_note}\n\n"
                        f"实际导出位置：\n{run_dir}\n\n"
                        "返回流星合成功能时会自动填入这个文件夹。\n\n是否现在打开？",
                        parent=self,
                    ):
                        self._open_export_folder()
                elif item[0] == "screening_export_error":
                    _, message, details = item
                    self.export_running = False
                    self.export_button.configure(state="normal", text="导出已选流星照片")
                    self.status.set("导出失败：" + message)
                    show_copyable_error(
                        "流星批量筛选 — 导出失败", message, parent=self, details=details
                    )
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(120, self._poll_queue)


def open_screening_workspace(
    master: tk.Misc, return_callback: Callable[[Path], None] | None = None,
) -> MeteorScreeningWindow:
    return MeteorScreeningWindow(master, return_callback)
