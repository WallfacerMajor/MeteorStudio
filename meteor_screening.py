"""Temporal meteor screening without a pre-existing clean base image."""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import rawpy
import tifffile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk
from platform_utils import open_folder
from error_dialog import show_copyable_error, show_runtime_log


RAW_SUFFIXES = {".arw", ".nef", ".nrw", ".cr2", ".cr3", ".crw"}
IMAGE_SUFFIXES = RAW_SUFFIXES | {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


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


def screening_autosave_file_path() -> Path:
    from meteor_composer import autosave_file_path
    return autosave_file_path().parent / "screening_autosave.json"


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
    gray_current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # A transient meteor must not create its own valid sky region.
    gray = np.minimum(gray_current, gray_reference)
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

    def robust_unit(values: np.ndarray) -> np.ndarray:
        low, high = (float(value) for value in np.percentile(values, (15.0, 90.0)))
        return np.clip((values - low) / max(1e-5, high - low), 0.0, 1.0)

    star_score = robust_unit(star_density)
    edge_score = robust_unit(edge_density)
    texture_score = robust_unit(texture)
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
            boundary_edge = float(edge_score[min(boundary, rows - 1), column])
            data_cost[column, boundary] = sky_cost + ground_cost - 0.42 * boundary_edge
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
    for index, result in enumerate(results):
        hits = 0
        for candidate in result.candidates:
            sx, sy = candidate.start
            ex, ey = candidate.end
            angle = float(np.arctan2(ey - sy, ex - sx))
            center = np.asarray(((sx + ex) * 0.5, (sy + ey) * 0.5), np.float32)
            for other_index in range(max(0, index - 2), min(len(results), index + 3)):
                if other_index == index:
                    continue
                matched = False
                for other in results[other_index].candidates:
                    osx, osy = other.start
                    oex, oey = other.end
                    other_angle = float(np.arctan2(oey - osy, oex - osx))
                    delta = abs(np.arctan2(np.sin(2 * (angle - other_angle)), np.cos(2 * (angle - other_angle)))) / 2
                    other_center = np.asarray(((osx + oex) * 0.5, (osy + oey) * 0.5), np.float32)
                    distance = float(np.linalg.norm(center - other_center))
                    if delta < np.deg2rad(9) and diagonal * 0.015 < distance < diagonal * 0.38:
                        matched = True
                        break
                hits += int(matched)
        result.temporal_hits = hits
        if hits >= 2:
            result.score = max(0, result.score - 25)
            result.note = "疑似连续飞机/卫星，请复查"
        elif result.plane_count:
            result.note = "检测到断续灯迹，请复查"


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
        self.status = tk.StringVar(value="选择连续拍摄照片文件夹；原文件只读。")
        self.summary = tk.StringVar(value="尚未分析")
        self.files: list[Path] = []
        self.results: list[ScreeningResult] = []
        self.decisions: dict[str, str] = {}
        self.decision_sources: dict[str, str] = {}
        self.active_candidates: dict[str, int] = {}
        self.preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.full_preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.full_preview_cache_bytes = 0
        self.full_preview_cache_budget = 768 << 20
        self.full_preview_loading: set[str] = set()
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
        self._build_ui()
        self.source_dir.trace_add("write", lambda *_args: self._schedule_autosave())
        self.output_dir.trace_add("write", lambda *_args: self._schedule_autosave())
        self.protocol("WM_DELETE_WINDOW", self._close_window)
        self._restore_autosave()
        self.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="流星批量筛选", font=("TkDefaultFont", 15, "bold")).pack(side="left")
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
        columns = ("decision", "score", "candidates", "note")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="照片")
        self.tree.heading("decision", text="状态")
        self.tree.heading("score", text="流星评分")
        self.tree.heading("candidates", text="候选")
        self.tree.heading("note", text="提示")
        self.tree.column("#0", width=260)
        self.tree.column("decision", width=90, anchor="center")
        self.tree.column("score", width=65, anchor="center")
        self.tree.column("candidates", width=55, anchor="center")
        self.tree.column("note", width=180)
        self.tree.tag_configure("manual_accept", background="#c8f2d0", foreground="#125a28")
        self.tree.tag_configure("auto_accept", background="#e5f5e8", foreground="#245c31")
        self.tree.tag_configure("manual_reject", background="#f7d2d2", foreground="#812323")
        self.tree.tag_configure("auto_reject", background="#f5eeee", foreground="#6b5555")
        self.tree.tag_configure("candidate_accept", background="#d8ecff", foreground="#174f7a")
        self.tree.tag_configure("candidate_reject", background="#eadcf7", foreground="#5f3478")
        self.tree.tag_configure("warning", background="#ffe8bd", foreground="#744600")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
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
        ttk.Label(preview_tools, text="滚轮缩放 · 拖动查看 · 按住 H 看无标记原图").pack(side="left", padx=10)

        self.canvas = tk.Canvas(right, background="#151515", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._canvas_configured)
        self.canvas.bind("<MouseWheel>", self._canvas_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._canvas_wheel(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._canvas_wheel(event, -1))
        self.canvas.bind("<ButtonPress-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)

        candidate_actions = ttk.Frame(right)
        candidate_actions.pack(fill="x", pady=(6, 0))
        ttk.Label(candidate_actions, text="当前候选：").pack(side="left")
        ttk.Button(candidate_actions, text="✓ 这条是流星", command=lambda: self._label_candidate("meteor")).pack(side="left")
        ttk.Button(candidate_actions, text="✕ 这条不是流星", command=lambda: self._label_candidate("not_meteor")).pack(side="left", padx=6)
        ttk.Button(candidate_actions, text="清除候选判断", command=lambda: self._label_candidate("")).pack(side="left")
        ttk.Button(candidate_actions, textvariable=self.manual_mark_label, command=self._toggle_manual_mark).pack(side="left", padx=(12, 0))
        ttk.Label(right, textvariable=self.candidate_status).pack(fill="x", pady=(3, 0))

        image_actions = ttk.Frame(right)
        image_actions.pack(fill="x", pady=(5, 0))
        ttk.Label(image_actions, text="当前照片：").pack(side="left")
        ttk.Button(image_actions, text="✓ 保留这张照片", command=self.accept_selected).pack(side="left")
        ttk.Button(image_actions, text="✕ 排除这张照片", command=self.reject_selected).pack(side="left", padx=6)
        ttk.Button(image_actions, text="恢复自动判断", command=self.reset_selected).pack(side="left")
        ttk.Label(image_actions, textvariable=self.summary).pack(side="right")

        bottom = ttk.Frame(root)
        bottom.pack(fill="x")
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
        return {
            "format": "meteor-screening-autosave-v1",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "source_dir": self.source_dir.get(),
            "output_dir": self.output_dir.get(),
            "last_export_dir": self.last_export_dir.get(),
            "sensitivity": int(self.sensitivity.get()),
            "use_original_preview": bool(self.use_original_preview.get()),
            "show_candidate_marks": bool(self.show_candidate_marks.get()),
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

    def _save_autosave(self) -> None:
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
        except Exception as exc:
            self.autosave_status.set(f"自动保存失败：{exc}")

    @staticmethod
    def _candidate_from_json(payload: dict) -> ScreeningCandidate:
        return ScreeningCandidate(
            tuple(int(value) for value in payload["start"]),
            tuple(int(value) for value in payload["end"]),
            int(payload.get("score", 0)),
            legacy_score=float(payload.get("legacy_score", 0.0)),
            label=str(payload.get("label", "")),
            manual=bool(payload.get("manual", False)),
            features=[float(value) for value in payload.get("features", [])],
        )

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
                restored.append(ScreeningResult(
                    str(source_path),
                    [self._candidate_from_json(value) for value in item.get("candidates", [])],
                    int(item.get("score", 0)),
                    plane_count=int(item.get("plane_count", 0)),
                    temporal_hits=int(item.get("temporal_hits", 0)),
                    note=str(item.get("note", "")),
                ))
            self._restoring_autosave = True
            self.source_dir.set(str(payload.get("source_dir", "")))
            self.output_dir.set(str(payload.get("output_dir", "")))
            restored_export = Path(str(payload.get("last_export_dir", ""))).expanduser()
            self.last_export_dir.set(str(restored_export) if restored_export.is_dir() else "")
            self.sensitivity.set(int(payload.get("sensitivity", 42)))
            self.use_original_preview.set(bool(payload.get("use_original_preview", False)))
            self.show_candidate_marks.set(bool(payload.get("show_candidate_marks", True)))
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
        if self.autosave_after_id is not None:
            try:
                self.after_cancel(self.autosave_after_id)
            except tk.TclError:
                pass
            self.autosave_after_id = None
        self._save_autosave()
        self.destroy()

    def _return_to_composer(self) -> None:
        exported = Path(self.last_export_dir.get().strip()).expanduser()
        if exported.is_dir() and self.return_callback is not None:
            self.return_callback(exported)
        self._close_window()

    def _browse_source(self) -> None:
        value = filedialog.askdirectory(title="选择连续拍摄照片文件夹", parent=self)
        if value:
            self.source_dir.set(value)
            if not self.output_dir.get().strip():
                self.output_dir.set(str(Path(value) / "MeteorStudio_Output"))

    def _browse_output(self) -> None:
        value = filedialog.askdirectory(title="选择筛选结果保存位置", parent=self)
        if value:
            self.output_dir.set(value)

    def analyze(self) -> None:
        if self.analysis_running:
            self.status.set("已有分析任务正在运行，请等待完成")
            return
        source = Path(self.source_dir.get().strip()).expanduser()
        if not source.is_dir():
            show_copyable_error("流星批量筛选", "请选择有效的照片文件夹", parent=self)
            return
        files = sorted(
            (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=capture_sort_key,
        )
        if len(files) < 3:
            show_copyable_error("流星批量筛选", "至少需要 3 张连续照片；推荐 5 张以上", parent=self)
            return
        self.files = files
        self.results = []
        self.decisions.clear()
        self.decision_sources.clear()
        self.active_candidates.clear()
        self.preview_cache.clear()
        self.full_preview_cache.clear()
        self.full_preview_cache_bytes = 0
        self.full_preview_loading.clear()
        self.tree.delete(*self.tree.get_children())
        self.progress["value"] = 0
        self.status.set(f"正在分析 {len(files)} 张照片；必要时自动配准相邻星点…")
        self.analysis_generation += 1
        generation = self.analysis_generation
        self.analysis_running = True
        self.analyze_button.configure(state="disabled", text="分析中…")
        threading.Thread(
            target=self._analyze_worker, args=(files, generation, str(source)), daemon=True,
            name=f"meteor-screening-{generation}",
        ).start()

    def _analyze_worker(self, files: list[Path], generation: int, source_signature: str) -> None:
        try:
            from meteor_composer import (
                calibrate_secondary_candidate_scores, candidate_feature_vector,
                detect_trails, load_meteor_ranker,
                predict_gradient_boosting, prepare_ml_maps,
            )
            model = load_meteor_ranker()
            cache: OrderedDict[str, np.ndarray] = OrderedDict()
            cpu_count = os.cpu_count() or 4
            analysis_workers = min(len(files), max(2, min(10, cpu_count // 2)))
            decode_workers = min(4, analysis_workers)
            cache_limit = analysis_workers + 8
            results: list[ScreeningResult] = []
            shape = (1, 1)
            previous_cv_threads = cv2.getNumThreads()

            def analyze_frame(index: int) -> ScreeningResult:
                path = files[index]
                current = cache[str(path)]
                neighbor_indices = [
                    other for other in range(max(0, index - 3), min(len(files), index + 4))
                    if other != index
                ]
                neighbors = [
                    cache[str(files[other])] for other in neighbor_indices
                    if cache[str(files[other])].shape == current.shape
                ]
                reference, displacement = temporal_reference(current, neighbors)
                sky_mask = estimate_star_sky_mask(current, reference)
                trails, planes = detect_trails(
                    current, reference, ranked=True, valid_region=sky_mask,
                )
                # Always retain the shared model's feature vector. It is used
                # only if the user explicitly labels this exact candidate.
                maps = prepare_ml_maps(current, reference) if trails else None
                measured = []
                for start, end, legacy_score in trails[:12]:
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
                return ScreeningResult(str(path), candidates, top, planes, note=note)

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
                        batch_end = min(len(files), batch_start + analysis_workers)
                        required = range(max(0, batch_start - 3), min(len(files), batch_end + 3))
                        missing = [index for index in required if str(files[index]) not in cache]
                        decoded = decoder.map(screening_preview, (files[index] for index in missing))
                        for index, image in zip(missing, decoded):
                            cache[str(files[index])] = image

                        batch_results = list(analyzer.map(analyze_frame, range(batch_start, batch_end)))
                        for completed, result in enumerate(batch_results, start=batch_start + 1):
                            results.append(result)
                            shape = cache[result.path].shape[:2]
                            self.work_queue.put((
                                "analysis_progress", generation, source_signature,
                                completed / len(files) * 92,
                                f"并行分析 {completed}/{len(files)}（{analysis_workers} 线程）",
                            ))

                        # Keep enough overlap for the next temporal batch and a
                        # small UI preview LRU, without retaining the whole shoot.
                        while len(cache) > cache_limit:
                            cache.popitem(last=False)
            finally:
                cv2.setNumThreads(previous_cv_threads)
            while len(cache) > 12:
                cache.popitem(last=False)
            mark_temporal_repeats(results, shape)
            self.work_queue.put(("analysis_finished", generation, source_signature, results, cache))
        except Exception as exc:
            import traceback
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

    def _refresh_tree(self, preserve_selection: bool = True) -> None:
        selected_path = None
        selection = self.tree.selection()
        if preserve_selection and selection:
            selected_path = self.results[int(selection[0])].path
        self.tree.delete(*self.tree.get_children())
        for index, result in enumerate(self.results):
            self.tree.insert(
                "", "end", iid=str(index), text=Path(result.path).name,
                values=(self._decision_label(result), result.score, len(result.candidates), result.note or "—"),
                tags=(self._decision_tag(result),),
            )
        if self.results:
            chosen = next((str(i) for i, item in enumerate(self.results) if item.path == selected_path), "0")
            self.tree.selection_set(chosen)
            self.tree.see(chosen)
        kept = sum(self._effective_decision(item) for item in self.results)
        manual = sum(self.decision_sources.get(path, "manual") == "manual" for path in self.decisions)
        linked = sum(self.decision_sources.get(path) == "candidate" for path in self.decisions)
        exact = sum(candidate.label in {"meteor", "not_meteor"} for item in self.results for candidate in item.candidates)
        self.summary.set(
            f"保留 {kept}/{len(self.results)} · 照片手动 {manual} · 候选联动 {linked} · 候选确认 {exact}"
        )

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

    def _show_selected(self, _event=None) -> None:
        result = self._selected_result()
        if result is None:
            return
        try:
            # Candidate coordinates are measured on the same 1400px analysis
            # proxy, so the overlay remains pixel-accurate in this preview.
            image = self.preview_cache.get(result.path)
            if image is None:
                image = screening_preview(Path(result.path), 1400)
                self.preview_cache[result.path] = image
                while len(self.preview_cache) > 12:
                    self.preview_cache.popitem(last=False)
            else:
                self.preview_cache.move_to_end(result.path)
            if self.preview_path != result.path:
                self.preview_path = result.path
                self.preview_zoom = 1.0
                self.preview_pan_x = self.preview_pan_y = 0.0
                self.manual_mark_mode = False
                self.manual_mark_start = None
                self.manual_mark_label.set("手动标记漏检流星")
                if result.candidates:
                    self.active_candidates.setdefault(result.path, 0)
            self._rebuild_preview_overlay()
        except Exception as exc:
            self.status.set(f"预览失败：{exc}")

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
        path = self.preview_path
        if not path or path in self.full_preview_cache or path in self.full_preview_loading:
            return
        self.full_preview_loading.add(path)
        self.status.set(f"正在读取原图精细预览：{Path(path).name}…")

        def worker() -> None:
            try:
                image = read_screening_image(Path(path), None)
                self.work_queue.put(("full_preview", path, image))
            except Exception as exc:
                self.work_queue.put(("full_preview_error", path, str(exc)))

        threading.Thread(target=worker, daemon=True, name="meteor-full-preview").start()

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
        coordinate_scale: float = 1.0,
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
        midpoint = (start + end) * 0.5 + normal * max(14.0, 14.0 * coordinate_scale)
        label = f"{candidate.score}%"
        if candidate.label == "meteor":
            label = "流星 " + label
        elif candidate.label == "not_meteor":
            label = "误选 " + label
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
        if marks_visible:
            for index, candidate in enumerate(result.candidates):
                if candidate.label == "meteor":
                    color = (55, 230, 95)
                elif candidate.label == "not_meteor":
                    color = (235, 90, 80)
                else:
                    color = (255, 190, 45)
                display_candidate = ScreeningCandidate(
                    (round(candidate.start[0] * scale_x), round(candidate.start[1] * scale_y)),
                    (round(candidate.end[0] * scale_x), round(candidate.end[1] * scale_y)),
                    candidate.score, legacy_score=candidate.legacy_score,
                    label=candidate.label, manual=candidate.manual,
                )
                self._draw_candidate_marker(
                    image, display_candidate, color, index == active_index, coordinate_scale,
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
            state = {"meteor": "已确认是流星", "not_meteor": "已确认是误选"}.get(candidate.label, "尚未确认")
            self.candidate_status.set(f"第 {active_index + 1}/{len(result.candidates)} 条 · AI {candidate.score}% · {state}")
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
                    self._sync_photo_decision_from_candidates(result)
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
        candidate = self._active_candidate()
        if result is None or candidate is None:
            messagebox.showinfo("候选确认", "请先在预览中点击一条候选标记", parent=self)
            return
        candidate.label = label
        self._sync_photo_decision_from_candidates(result)
        self._refresh_tree()
        self._rebuild_preview_overlay()
        self._schedule_autosave()

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
        threading.Thread(
            target=self._copy_selected_worker,
            args=(run_dir, selected_snapshot, explicit_feedback, report_header),
            daemon=True, name="meteor-screening-export",
        ).start()

    def _copy_selected_worker(
        self, run_dir: Path, selected: list[tuple[Path, dict, str, str]],
        explicit_feedback: list[dict], report_header: dict,
    ) -> None:
        try:
            feedback_path = save_screening_feedback(explicit_feedback)
            report = []
            for index, (source_path, item_data, decision, decision_source) in enumerate(selected, start=1):
                destination = run_dir / source_path.name
                shutil.copy2(source_path, destination)
                report.append({
                    **item_data, "copied_to": str(destination),
                    "decision": decision, "decision_source": decision_source,
                })
                self.work_queue.put((
                    "export_progress", index / max(1, len(selected)) * 100,
                    f"正在导出原图 {index}/{len(selected)}：{source_path.name}",
                ))
            payload = {
                **report_header,
                "candidate_feedback_file": str(feedback_path) if feedback_path else None,
                "items": report,
            }
            (run_dir / "screening_report.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            self.work_queue.put((
                "screening_exported", run_dir, len(selected), len(explicit_feedback)
            ))
        except Exception as exc:
            import traceback
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
                    _, generation, source_signature, results, preview_cache = item
                    if generation != self.analysis_generation:
                        continue
                    self.analysis_running = False
                    self.analyze_button.configure(state="normal", text="开始分析")
                    if source_signature != str(Path(self.source_dir.get().strip()).expanduser()):
                        self.status.set("分析期间照片文件夹已变化，旧结果已丢弃")
                        continue
                    self.results = results
                    self.preview_cache = preview_cache
                    self.progress["value"] = 100
                    self._refresh_tree(False)
                    self._show_selected()
                    kept = sum(self._effective_decision(result) for result in self.results)
                    self.status.set(f"分析完成：{len(self.results)} 张中自动保留 {kept} 张，请逐张复核")
                    self._schedule_autosave()
                elif item[0] == "full_preview":
                    _, path, image = item
                    self.full_preview_loading.discard(path)
                    previous = self.full_preview_cache.pop(path, None)
                    if previous is not None:
                        self.full_preview_cache_bytes -= int(previous.nbytes)
                    self.full_preview_cache[path] = image
                    self.full_preview_cache_bytes += int(image.nbytes)
                    self.full_preview_cache.move_to_end(path)
                    while (
                        self.full_preview_cache_bytes > self.full_preview_cache_budget
                        and len(self.full_preview_cache) > 1
                    ):
                        _old_path, removed = self.full_preview_cache.popitem(last=False)
                        self.full_preview_cache_bytes -= int(removed.nbytes)
                    if path == self.preview_path and self.use_original_preview.get():
                        self.preview_zoom = 1.0
                        self.preview_pan_x = self.preview_pan_y = 0.0
                        self._rebuild_preview_overlay()
                        self.status.set(
                            f"原图精细预览：{Path(path).name} · "
                            f"{image.shape[1]}×{image.shape[0]} · 滚轮放大或点击 1:1"
                        )
                elif item[0] == "full_preview_error":
                    _, path, message = item
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
