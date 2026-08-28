from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
Progress = Callable[[float, str], None]
PANORAMA_PROJECTIONS = {"rectilinear", "mercator", "equirectangular", "stereographic"}


@dataclass
class AlignmentItem:
    source: str
    status: str = "等待"
    control_points: int = 0
    median_error: float | None = None
    message: str = ""
    layer_index: int | None = None
    output_layer: str | None = None
    focal_length: float | None = None
    focal_source: str = ""
    sky_coverage: float | None = None


@dataclass(frozen=True)
class ImageLensInfo:
    focal_length: float
    sensor_diagonal: float
    equivalent_35mm: float | None
    source: str


@dataclass
class AlignmentResult:
    project_dir: str
    project_file: str
    base_layer: str | None
    duplicate_base_layer: str | None
    items: list[AlignmentItem]
    ptgui_log: str
    siril_version: str
    ptgui_path: str
    siril_path: str
    laboratory: bool = False
    projection: str = "rectilinear"
    canvas_scale: float = 1.0


def default_ptgui_path() -> Path | None:
    candidates = []
    if sys.platform == "win32":
        candidates += [Path(r"C:\Program Files\PTGui\PTGui.exe")]
    elif sys.platform == "darwin":
        candidates += [Path("/Applications/PTGui Pro.app/Contents/MacOS/PTGui Pro")]
    return next((p for p in candidates if p.is_file()), None)


def default_siril_path() -> Path | None:
    candidates = []
    if sys.platform == "win32":
        candidates += [
            Path(r"C:\Program Files\Siril\bin\siril-cli.exe"),
            Path(r"C:\Program Files\Siril\siril-cli.exe"),
        ]
    elif sys.platform == "darwin":
        candidates += [Path("/Applications/Siril.app/Contents/MacOS/siril-cli")]
    resolved = shutil.which("siril-cli")
    if resolved:
        candidates.append(Path(resolved))
    return next((p for p in candidates if p.is_file()), None)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def _read_rgb8(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            data = tifffile.imread(path)
        except (ValueError, RuntimeError):
            with Image.open(path) as image:
                return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        if data.ndim == 3 and data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4):
            data = np.moveaxis(data, 0, -1)
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        data = data[..., :3]
        if data.dtype == np.uint8:
            return data
        if data.dtype == np.uint16:
            return np.right_shift(data, 8).astype(np.uint8)
        finite = data[np.isfinite(data)]
        high = float(np.percentile(finite, 99.9)) if finite.size else 1.0
        scale = 255.0 / max(high, 1e-6)
        return np.nan_to_num(data * scale).clip(0, 255).astype(np.uint8)
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def _exif_number(value) -> float | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            denominator = float(value[1])
            number = float(value[0]) / denominator
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return number if math.isfinite(number) and number > 0 else None
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            number = float(value.numerator) / float(value.denominator)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None
    return number if math.isfinite(number) and number > 0 else None


def read_lens_info(
    path: Path,
    fallback_focal: float = 14.0,
    fallback_sensor_diagonal: float = 43.2666,
) -> ImageLensInfo:
    """Read per-image focal data without assuming the batch used one zoom setting."""
    focal = None
    equivalent = None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            focal = _exif_number(exif.get(37386))  # FocalLength
            equivalent = _exif_number(exif.get(41989))  # FocalLengthIn35mmFilm
    except (OSError, ValueError, TypeError):
        pass
    # Lightroom/Photoshop TIFFs commonly retain camera EXIF in tag 34665 while
    # exposing only IFD0 through Pillow. tifffile resolves that child IFD into
    # a dictionary, so read it before declaring the metadata missing.
    if path.suffix.lower() in {".tif", ".tiff"} and (focal is None or equivalent is None):
        try:
            with tifffile.TiffFile(path) as tif:
                tag = tif.pages[0].tags.get("ExifTag")
                nested = tag.value if tag is not None and isinstance(tag.value, dict) else {}
                focal = focal or _exif_number(nested.get("FocalLength"))
                equivalent = equivalent or _exif_number(nested.get("FocalLengthIn35mmFilm"))
        except (OSError, ValueError, TypeError, KeyError, tifffile.TiffFileError):
            pass
    sensor = float(fallback_sensor_diagonal)
    if focal is not None and equivalent is not None:
        # EXIF's 35 mm equivalent provides a crop-factor estimate for this
        # particular file and is more useful than a batch-wide sensor guess.
        sensor = 43.2666 * focal / equivalent
    if focal is not None:
        return ImageLensInfo(float(focal), float(sensor), equivalent, "EXIF")
    if equivalent is not None:
        inferred = equivalent * sensor / 43.2666
        return ImageLensInfo(float(inferred), float(sensor), equivalent, "EXIF 35mm等效")
    return ImageLensInfo(float(fallback_focal), float(sensor), None, "兜底值（EXIF缺失）")


def make_sky_proxy(
    source: Path,
    destination: Path,
    scale: float = 0.25,
    sky_fraction: float | None = None,
) -> tuple[int, int]:
    rgb = _read_rgb8(source).copy()
    height, width = rgb.shape[:2]
    proxy = cv2.resize(rgb, (max(32, round(width * scale)), max(32, round(height * scale))), interpolation=cv2.INTER_AREA)
    # ``sky_fraction`` is retained only as a legacy/fallback safety cap. The
    # normal pipeline passes None and builds a different star-driven mask for
    # every image after Siril has measured its PSF catalogue.
    if sky_fraction is not None:
        sky_bottom = int(proxy.shape[0] * float(np.clip(sky_fraction, 0.35, 1.0)))
        fade = max(8, round(proxy.shape[0] * 0.025))
        start = max(0, sky_bottom - fade)
        weights = np.linspace(1.0, 0.0, sky_bottom - start, dtype=np.float32)[:, None, None]
        proxy[start:sky_bottom] = np.clip(proxy[start:sky_bottom].astype(np.float32) * weights, 0, 255).astype(np.uint8)
        proxy[sky_bottom:] = 0
    # Siril needs the original point-spread profile.  Background subtraction is
    # deliberately left to its star finder; sharpening here turns stars into
    # rings and makes the PSF rejection unreliable.
    proxy_rgb = proxy
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(proxy_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"无法生成星点代理图：{destination}")
    encoded.tofile(destination)
    return width, height


def filter_sky_stars(stars: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Keep the dense astronomical field and reject isolated ground/lamp PSFs."""
    if len(stars) < 6:
        return stars
    height, width = shape
    columns, rows = 20, 14
    cell_x = np.clip((stars[:, 0] / max(1, width) * columns).astype(int), 0, columns - 1)
    cell_y = np.clip((stars[:, 1] / max(1, height) * rows).astype(int), 0, rows - 1)
    counts = np.zeros((rows, columns), dtype=np.float32)
    np.add.at(counts, (cell_y, cell_x), 1)
    density = cv2.boxFilter(counts, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    positive = density[density > 0]
    threshold = max(3.0, float(np.percentile(positive, 28.0))) if positive.size else 3.0
    cells = (density >= threshold).astype(np.uint8)
    cells = cv2.morphologyEx(cells, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(cells, 8)
    best = None
    best_score = -1.0
    for label in range(1, component_count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        center_y = float(centroids[label][1]) / max(1, rows - 1)
        score = area * (1.25 if center_y < 0.72 else 0.65)
        if score > best_score:
            best, best_score = label, score
    if best is None:
        return stars
    accepted_cells = cv2.dilate((labels == best).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    keep = accepted_cells[cell_y, cell_x] > 0
    filtered = stars[keep]
    # Never let an aggressive density estimate destroy an otherwise usable
    # sparse frame. The geometric matcher remains the final acceptance gate.
    return filtered if len(filtered) >= 6 else stars


def make_star_sky_mask(shape: tuple[int, int], stars: np.ndarray, radius: int | None = None) -> np.ndarray:
    height, width = shape
    radius = radius or max(14, round(min(width, height) / 55))
    mask = _star_mask(shape, stars, radius=radius)
    # Join nearby star islands but retain large holes around foreground objects.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, radius // 2 * 2 + 1),) * 2)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def make_ptgui_sky_proxy(
    source: Path,
    destination: Path,
    proxy_mask: np.ndarray,
) -> None:
    rgb = _read_rgb8(source).copy()
    mask = cv2.resize(proxy_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    feather = cv2.GaussianBlur(mask, (0, 0), max(2.0, min(rgb.shape[:2]) / 900.0)).astype(np.float32) / 255.0
    rgb = np.clip(rgb.astype(np.float32) * feather[:, :, None], 0, 255).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exif_bytes = None
    try:
        with Image.open(source) as image:
            exif = image.getexif()
            exif_bytes = exif.tobytes() if len(exif) else None
    except (OSError, ValueError, TypeError):
        pass
    save_options = {"quality": 90, "subsampling": 0}
    if exif_bytes:
        save_options["exif"] = exif_bytes
    Image.fromarray(rgb, "RGB").save(destination, **save_options)


def _script_path_text(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def siril_find_stars(siril: Path, proxy: Path, work_dir: Path, name: str, max_stars: int = 1600) -> tuple[np.ndarray, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    list_name = f"{name}_stars.lst"
    script = work_dir / f"{name}_findstars.ssf"
    script.write_text(
        "\n".join(
            [
                "requires 1.2.0",
                f'load "{_script_path_text(proxy)}"',
                "setfindstar reset -sigma=1.0 -roundness=0.20 -relax=on -radius=5",
                f"findstar -out={list_name} -maxstars={max_stars}",
                "close",
                "",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(siril), "-o", "-d", str(work_dir), "-s", str(script)],
        cwd=work_dir,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Siril星点检测失败：{proxy.name}\n{completed.stdout[-1200:]}")
    star_file = work_dir / list_name
    if not star_file.is_file():
        raise RuntimeError(f"Siril没有生成星表：{proxy.name}")
    proxy_image = cv2.imdecode(np.fromfile(proxy, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if proxy_image is None:
        raise RuntimeError(f"无法读取Siril星点代理：{proxy.name}")
    proxy_height = proxy_image.shape[0]
    rows = []
    for line in star_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) < 16:
            continue
        try:
            x, y = float(columns[5]), float(columns[6])
            fwhm_x, fwhm_y = float(columns[7]), float(columns[8])
            rmse, saturated = float(columns[12]), int(columns[14])
        except ValueError:
            continue
        # On low-contrast early frames Siril can still locate real stars while
        # reporting a comparatively large PSF fit RMSE.  Keep those candidates
        # for geometric verification; RANSAC and the final reprojection error
        # remain the actual acceptance gate.
        if saturated or not (0.5 <= fwhm_x <= 18 and 0.5 <= fwhm_y <= 18) or rmse > 40.0:
            continue
        # Siril/FITS uses a lower-left Y origin while OpenCV and PTGui image
        # coordinates use an upper-left origin. Normalize at the boundary so
        # masks, SIFT support checks, and PTGui control points agree.
        rows.append((x, proxy_height - 1.0 - y))
    # Three PSF-confirmed stars are sufficient to prove that the sky region is
    # usable.  The actual transform still requires at least six independent
    # RANSAC-filtered SIFT correspondences below, so this does not weaken the
    # geometric acceptance rule on sparse early frames.
    if len(rows) < 3:
        raise RuntimeError(f"Siril仅找到{len(rows)}个可用星点：{proxy.name}")
    return np.asarray(rows, dtype=np.float32), completed.stdout


def _star_mask(shape: tuple[int, int], stars: np.ndarray, radius: int = 12) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y in stars:
        cv2.circle(mask, (round(float(x)), round(float(y))), radius, 255, -1, cv2.LINE_AA)
    return mask


def _distributed_pairs(source: np.ndarray, target: np.ndarray, width: int, height: int, limit: int = 48) -> list[tuple[np.ndarray, np.ndarray]]:
    cells: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for src, dst in zip(source, target):
        cell = (min(5, int(dst[0] / max(1, width) * 6)), min(3, int(dst[1] / max(1, height) * 4)))
        cells.setdefault(cell, []).append((src, dst))
    result = []
    while len(result) < limit and any(cells.values()):
        for key in sorted(cells):
            if cells[key]:
                result.append(cells[key].pop(0))
                if len(result) >= limit:
                    break
    return result


def match_star_pairs(
    meteor_proxy: Path,
    base_proxy: Path,
    meteor_stars: np.ndarray,
    base_stars: np.ndarray,
    full_scale: float,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float]:
    meteor = cv2.imdecode(np.fromfile(meteor_proxy, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    base = cv2.imdecode(np.fromfile(base_proxy, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if meteor is None or base is None:
        raise RuntimeError("无法读取星点代理图")
    def prepare(image: np.ndarray) -> np.ndarray:
        background = cv2.GaussianBlur(image, (0, 0), 12)
        return cv2.normalize(cv2.subtract(image, background), None, 0, 255, cv2.NORM_MINMAX)

    meteor_features, base_features = prepare(meteor), prepare(base)
    sift = cv2.SIFT_create(nfeatures=16000, contrastThreshold=0.006, edgeThreshold=15, sigma=1.2)
    meteor_feature_mask = make_star_sky_mask(meteor.shape, meteor_stars, radius=max(12, round(min(meteor.shape) / 70)))
    base_feature_mask = make_star_sky_mask(base.shape, base_stars, radius=max(12, round(min(base.shape) / 70)))
    mk, md = sift.detectAndCompute(meteor_features, meteor_feature_mask)
    bk, bd = sift.detectAndCompute(base_features, base_feature_mask)
    if md is None or bd is None:
        raise RuntimeError("Siril星区内没有足够的可匹配特征")
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(md, bd, k=2)
    good = [first for first, second in candidates if first.distance < 0.78 * second.distance]
    if len(good) < 4:
        raise RuntimeError(f"仅找到{len(good)}组候选星点")
    source = np.float32([mk[m.queryIdx].pt for m in good])
    target = np.float32([bk[m.trainIdx].pt for m in good])
    transform, inliers = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 2.5, maxIters=150000, confidence=0.999)
    if transform is None or inliers is None:
        raise RuntimeError("无法建立可靠星空变换")
    keep = inliers.ravel().astype(bool)
    source, target = source[keep], target[keep]
    if len(source) < 4:
        raise RuntimeError(f"稳健筛选后仅剩{len(source)}组星点")
    # Prefer matches supported by Siril's PSF star catalogue. It is a
    # validator, not the sole matcher: very faint stars may be absent from the
    # catalogue but remain valid SIFT correspondences.
    def nearest_distance(points: np.ndarray, catalogue: np.ndarray) -> np.ndarray:
        if not len(catalogue):
            return np.full(len(points), np.inf, np.float32)
        result = np.full(len(points), np.inf, np.float32)
        for start in range(0, len(catalogue), 256):
            block = catalogue[start:start + 256]
            result = np.minimum(result, np.sqrt(((points[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)).min(axis=1))
        return result

    support_radius = max(10.0, min(base.shape) / 80.0)
    supported = (
        (nearest_distance(source, meteor_stars) <= support_radius)
        & (nearest_distance(target, base_stars) <= support_radius)
    )
    # Siril is a sky/PSF validator, not the sole correspondence detector. Two
    # differently processed frames may expose different subsets of faint stars;
    # in that case retain the already mask-limited, MAGSAC-consistent matches.
    if int(supported.sum()) >= 4:
        source, target = source[supported], target[supported]
    projected = cv2.perspectiveTransform(source[:, None, :], transform)[:, 0, :]
    errors = np.linalg.norm(projected - target, axis=1)
    median_error = float(np.median(errors) / full_scale)
    distributed = _distributed_pairs(source, target, base.shape[1], base.shape[0])
    return [(src / full_scale, dst / full_scale) for src, dst in distributed], median_error


def alignment_solution_quality(control_points: int, median_error: float) -> tuple[bool, bool, str]:
    """Classify a projective star solution without discarding usable 4–5 point fits."""
    if control_points < 4:
        return False, True, f"控制点不足：单应性对齐至少需要4组，当前{control_points}组"
    if median_error > 8.0:
        return False, True, f"星点误差过大：{median_error:.2f}px，{control_points}组"
    review = control_points < 6 or median_error > 3.0
    reasons = []
    if control_points < 6:
        reasons.append(f"仅{control_points}组控制点")
    if median_error > 3.0:
        reasons.append(f"星点误差 {median_error:.2f}px")
    message = "、".join(reasons)
    return True, review, message


def _run(command: list[str], cwd: Path, log_file: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    output = completed.stdout or ""
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n" + output + "\n")
    if completed.returncode != 0:
        raise RuntimeError(f"命令失败（{completed.returncode}）：{Path(command[0]).name}\n{output[-1600:]}")
    return output


def _link_all_to_first_lens(project: dict) -> None:
    if not project.get("globallenses"):
        return
    project["globallenses"] = project["globallenses"][:1]
    lens = project["globallenses"][0]
    for group in project["imagegroups"]:
        group["globallens"] = 0
    lens["lens"]["optimizerflags"].update({"fov": False, "a": False, "b": False, "c": False, "fisheyefactor": False})
    lens["shift"]["optimizerflags"].update({"longside": False, "shortside": False})
    lens["shear"]["optimizerflags"].update({"hshear": False, "vshear": False})


def configure_blog_project(
    project_file: Path,
    base_points: Iterable[tuple[float, float]],
    matched: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]],
    calibration_project: Path | None = None,
    coordinate_scale: float = 1.0,
    optimize_lens: bool = False,
) -> None:
    data = json.loads(project_file.read_text(encoding="utf-8"))
    project = data["project"]
    if len(project["imagegroups"]) < 3:
        raise RuntimeError("PTGui工程缺少对齐参考图或流星图")
    _link_all_to_first_lens(project)
    if calibration_project is not None:
        calibrated = json.loads(calibration_project.read_text(encoding="utf-8"))["project"]
        project["globallenses"] = json.loads(json.dumps(calibrated["globallenses"][:1]))
        _link_all_to_first_lens(project)
    for index, group in enumerate(project["imagegroups"]):
        flags = index >= 2
        group["position"]["optimizerflags"].update({"yaw": flags, "pitch": flags, "roll": flags})
        group["linkable"]["position"]["optimizerflags"].update({"yaw": flags, "pitch": flags, "roll": flags})
        if index < 2:
            for key in ("yaw", "pitch", "roll"):
                group["position"]["params"][key] = 0
                group["linkable"]["position"]["params"][key] = 0
    control_points = []
    for x, y in list(base_points)[:24]:
        x, y = x * coordinate_scale, y * coordinate_scale
        control_points.append({"t": 0, "0": [0, 0, round(x), round(y)], "1": [1, 0, round(x), round(y)]})
    for image_index, pairs in matched:
        for source, target in pairs:
            control_points.append({
                "t": 0,
                "0": [0, 0, round(float(target[0]) * coordinate_scale), round(float(target[1]) * coordinate_scale)],
                "1": [image_index, 0, round(float(source[0]) * coordinate_scale), round(float(source[1]) * coordinate_scale)],
            })
    project["controlpoints"] = control_points
    align = project["projectsettings"]["alignsettings"]
    align.update(
        {
            "generatecp": True,
            "optimize": True,
            "roughlyalign": True,
            "straighten": False,
            "fit": False,
            "chooseprojection": False,
            "optimizeexposure": False,
            "isprealigned": False,
        }
    )
    project["projectsettings"]["batchstitchersettings"].update({"align": True, "stitch": False, "stitchonlyifcontrolpoints": True})
    project["optimizer"]["simplemodesettings"].update(
        {"anchorimagegroup": 0, "optimizefov": False, "optimizelens": "none"}
    )
    # Use PTGui's advanced optimizer flags so the user-provided focal length
    # remains an anchor.  Letting focal length float across a very long time
    # span is under-constrained and can converge to a visually plausible but
    # physically impossible lens.
    project["optimizer"]["simplemode"] = False
    lens = project["globallenses"][0]
    lens["lens"]["optimizerflags"].update(
        {"fov": False, "a": optimize_lens, "b": optimize_lens, "c": optimize_lens, "fisheyefactor": False}
    )
    lens["shift"]["optimizerflags"].update({"longside": optimize_lens, "shortside": optimize_lens})
    # Control points are already supplied by MeteorStudio. Mark CP generation
    # complete so PTGui's batch optimizer does not add ground/border matches.
    project["autocpdone"] = True
    project["hasbeenoptimized"] = False
    project_file.write_text(json.dumps(data, ensure_ascii=False, indent="\t") + "\n", encoding="utf-8")


def transfer_proxy_solution(
    full_project_file: Path,
    proxy_project_file: Path,
    base_points: Iterable[tuple[float, float]],
    matched: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]],
) -> None:
    full_data = json.loads(full_project_file.read_text(encoding="utf-8"))
    proxy_data = json.loads(proxy_project_file.read_text(encoding="utf-8"))
    full = full_data["project"]
    proxy = proxy_data["project"]
    if len(full["imagegroups"]) != len(proxy["imagegroups"]):
        raise RuntimeError("代理工程与完整工程的图层数量不同")
    # Preserve the lens profile solved from the sky-only batch, not merely the
    # yaw/pitch/roll values.  Without a/b/c and optical-centre shift, stars near
    # the edge can miss by 5–10 pixels after the final full-resolution export.
    full["globallenses"] = json.loads(json.dumps(proxy["globallenses"][:1]))
    _link_all_to_first_lens(full)
    for index, group in enumerate(full["imagegroups"]):
        solved = proxy["imagegroups"][index]["position"]["params"]
        for target in (group["position"]["params"], group["linkable"]["position"]["params"]):
            target.update({key: solved[key] for key in ("yaw", "pitch", "roll")})
        group["position"]["optimizerflags"].update({"yaw": False, "pitch": False, "roll": False})
        group["linkable"]["position"]["optimizerflags"].update({"yaw": False, "pitch": False, "roll": False})
    control_points = []
    for x, y in list(base_points)[:24]:
        control_points.append({"t": 0, "0": [0, 0, round(x), round(y)], "1": [1, 0, round(x), round(y)]})
    for image_index, pairs in matched:
        for source, target in pairs:
            control_points.append({
                "t": 0,
                "0": [0, 0, round(float(target[0])), round(float(target[1]))],
                "1": [image_index, 0, round(float(source[0])), round(float(source[1]))],
            })
    full["controlpoints"] = control_points
    full["autocpdone"] = True
    full["hasbeenoptimized"] = True
    full["projectsettings"]["alignsettings"].update({"generatecp": False, "optimize": False, "optimizeexposure": False})
    full["projectsettings"]["batchstitchersettings"].update({"align": False, "stitch": False})
    full_project_file.write_text(json.dumps(full_data, ensure_ascii=False, indent="\t") + "\n", encoding="utf-8")


def configure_single_star_project(
    project_file: Path,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    base_lens: ImageLensInfo,
    meteor_lens: ImageLensInfo,
    image_width: int,
    image_height: int,
    initial_position: dict[str, float] | None = None,
    panorama_projection: str = "rectilinear",
    canvas_scale: float = 1.0,
) -> None:
    """Configure one base + one meteor project using sky-only control points."""
    data = json.loads(project_file.read_text(encoding="utf-8"))
    project = data["project"]
    if len(project["imagegroups"]) != 2:
        raise RuntimeError("单张PTGui工程必须包含对齐参考图和一张流星图")
    _set_independent_lenses(
        project,
        [base_lens, meteor_lens],
        image_width,
        image_height,
        optimize_distortion=len(pairs) >= 16,
        panorama_projection=panorama_projection,
        canvas_scale=canvas_scale,
    )
    for index, group in enumerate(project["imagegroups"]):
        movable = index == 1
        group["position"]["optimizerflags"].update({"yaw": movable, "pitch": movable, "roll": movable})
        group["linkable"]["position"]["optimizerflags"].update({"yaw": movable, "pitch": movable, "roll": movable})
        if index == 0:
            for target in (group["position"]["params"], group["linkable"]["position"]["params"]):
                target.update({"yaw": 0, "pitch": 0, "roll": 0})
        elif initial_position is not None:
            for target in (group["position"]["params"], group["linkable"]["position"]["params"]):
                target.update({key: float(initial_position[key]) for key in ("yaw", "pitch", "roll")})
    project["controlpoints"] = [
        {
            "t": 0,
            "0": [0, 0, round(float(target[0])), round(float(target[1]))],
            "1": [1, 0, round(float(source[0])), round(float(source[1]))],
        }
        for source, target in pairs
    ]
    project["projectsettings"]["alignsettings"].update(
        {
            "generatecp": True,
            "optimize": True,
            "roughlyalign": initial_position is None,
            "straighten": False,
            "fit": False,
            "chooseprojection": False,
            "optimizeexposure": False,
            "isprealigned": initial_position is not None,
        }
    )
    project["projectsettings"]["batchstitchersettings"].update(
        {"align": True, "stitch": False, "stitchonlyifcontrolpoints": True}
    )
    project["optimizer"]["simplemodesettings"].update(
        {"anchorimagegroup": 0, "optimizefov": False, "optimizelens": "none"}
    )
    project["optimizer"]["simplemode"] = False
    project["autocpdone"] = True
    project["hasbeenoptimized"] = False
    project_file.write_text(json.dumps(data, ensure_ascii=False, indent="\t") + "\n", encoding="utf-8")


def transfer_single_solution(full_project_file: Path, proxy_project_file: Path) -> None:
    full_data = json.loads(full_project_file.read_text(encoding="utf-8"))
    proxy = json.loads(proxy_project_file.read_text(encoding="utf-8"))["project"]
    full = full_data["project"]
    if len(full["imagegroups"]) != 2 or len(proxy["imagegroups"]) != 2:
        raise RuntimeError("单张代理工程与原图工程结构不一致")
    full["globallenses"] = json.loads(json.dumps(proxy["globallenses"]))
    for index, group in enumerate(full["imagegroups"]):
        group["globallens"] = int(proxy["imagegroups"][index].get("globallens", index))
        solved = proxy["imagegroups"][index]["position"]["params"]
        for target in (group["position"]["params"], group["linkable"]["position"]["params"]):
            target.update({key: solved[key] for key in ("yaw", "pitch", "roll")})
        group["position"]["optimizerflags"].update({"yaw": False, "pitch": False, "roll": False})
        group["linkable"]["position"]["optimizerflags"].update({"yaw": False, "pitch": False, "roll": False})
    full["controlpoints"] = json.loads(json.dumps(proxy["controlpoints"]))
    full["autocpdone"] = True
    full["hasbeenoptimized"] = True
    full["projectsettings"]["alignsettings"].update({"generatecp": False, "optimize": False, "optimizeexposure": False})
    full["projectsettings"]["batchstitchersettings"].update({"align": False, "stitch": False})
    full_project_file.write_text(json.dumps(full_data, ensure_ascii=False, indent="\t") + "\n", encoding="utf-8")


def configure_layer_export(project_file: Path, output_base: Path) -> None:
    data = json.loads(project_file.read_text(encoding="utf-8"))
    project = data["project"]
    project["outputcomponents"].update({"ldrpanorama": False, "ldrlayers": True, "ldrblendplanes": False})
    project["panoramaparams"]["outputfile"] = str(output_base)
    project["panoramaparams"]["fileformat"] = "tiff"
    project["panoramaparams"]["tiffparams"].update({"datatype": "u16", "compression": "deflate", "alphatype": "unassociated"})
    project["panoramaparams"]["outputcrop"] = [0, 0, 1, 1]
    project["projectsettings"]["alignsettings"].update({"generatecp": False, "optimize": False, "optimizeexposure": False})
    project["projectsettings"]["batchstitchersettings"].update({"align": False, "stitch": True, "stitchonlyifcontrolpoints": True})
    project_file.write_text(json.dumps(data, ensure_ascii=False, indent="\t") + "\n", encoding="utf-8")


def _set_independent_lenses(
    project: dict,
    lens_infos: list[ImageLensInfo],
    image_width: int,
    image_height: int,
    optimize_distortion: bool = False,
    panorama_projection: str = "rectilinear",
    canvas_scale: float = 1.0,
) -> None:
    if len(project.get("imagegroups", [])) != len(lens_infos):
        raise RuntimeError("镜头信息数量与PTGui图层数量不一致")
    existing = project.get("globallenses") or []
    if not existing:
        raise RuntimeError("PTGui工程缺少镜头模型")
    templates = []
    for index, info in enumerate(lens_infos):
        source = existing[min(index, len(existing) - 1)]
        lens_group = json.loads(json.dumps(source))
        params = lens_group["lens"]["params"]
        params.update(
            {
                "projection": "rectilinear",
                "focallength": float(info.focal_length),
                "sensordiagonal": float(info.sensor_diagonal),
            }
        )
        flags = lens_group["lens"]["optimizerflags"]
        flags.update(
            {
                "fov": False,
                "a": bool(optimize_distortion and index > 0),
                "b": bool(optimize_distortion and index > 0),
                "c": bool(optimize_distortion and index > 0),
                "fisheyefactor": False,
            }
        )
        lens_group["shift"]["optimizerflags"].update({"longside": False, "shortside": False})
        lens_group["shear"]["optimizerflags"].update({"hshear": False, "vshear": False})
        templates.append(lens_group)
        project["imagegroups"][index]["globallens"] = index
    project["globallenses"] = templates
    base_lens = lens_infos[0]
    aspect = image_width / max(1, image_height)
    sensor_height = base_lens.sensor_diagonal / math.sqrt(aspect * aspect + 1.0)
    sensor_width = sensor_height * aspect
    hfov = math.degrees(2.0 * math.atan(sensor_width / (2.0 * base_lens.focal_length)))
    vfov = math.degrees(2.0 * math.atan(sensor_height / (2.0 * base_lens.focal_length)))
    projection = panorama_projection if panorama_projection in PANORAMA_PROJECTIONS else "rectilinear"
    scale = float(np.clip(canvas_scale, 1.0, 1.8))
    maximum_hfov = 165.0 if projection == "rectilinear" else (300.0 if projection == "stereographic" else 360.0)
    maximum_vfov = 165.0 if projection in {"rectilinear", "stereographic"} else 175.0
    project["panoramaparams"].update({
        "projection": projection,
        "hfov": min(maximum_hfov, hfov * scale),
        "vfov": min(maximum_vfov, vfov * scale),
        "outputcrop": [0, 0, 1, 1],
    })
    project["outputsize"].update({"mode": "relative", "fractionofoptimumsize": 1})


def configure_input_lenses(
    project_file: Path,
    lens_infos: list[ImageLensInfo],
    image_width: int,
    image_height: int,
    panorama_projection: str = "rectilinear",
    canvas_scale: float = 1.0,
) -> None:
    data = json.loads(project_file.read_text(encoding="utf-8"))
    project = data["project"]
    _set_independent_lenses(
        project, lens_infos, image_width, image_height, optimize_distortion=False,
        panorama_projection=panorama_projection, canvas_scale=canvas_scale,
    )
    project_file.write_text(json.dumps(data, ensure_ascii=False, indent="\t") + "\n", encoding="utf-8")


def configure_input_lens(
    project_file: Path,
    focal_length: float,
    sensor_diagonal: float,
    image_width: int,
    image_height: int,
) -> None:
    """Backward-compatible wrapper; new code should pass per-image lens data."""
    data = json.loads(project_file.read_text(encoding="utf-8"))
    count = len(data["project"].get("imagegroups", []))
    info = ImageLensInfo(float(focal_length), float(sensor_diagonal), None, "兼容参数")
    configure_input_lenses(project_file, [info] * count, image_width, image_height)


def _distributed_base_points(stars: np.ndarray, scale: float, width: int, height: int) -> list[tuple[float, float]]:
    full = stars / scale
    pairs = _distributed_pairs(full, full, width, height, limit=24)
    return [(float(dst[0]), float(dst[1])) for _, dst in pairs]


def run_alignment_pipeline(
    base: Path,
    meteor_files: list[Path],
    output_root: Path,
    ptgui: Path,
    siril: Path,
    progress: Progress | None = None,
    sky_fraction: float | None = None,
    focal_length: float = 14.0,
    sensor_diagonal: float = 43.2666,
    export_layers: bool = True,
    laboratory: bool = False,
    panorama_projection: str = "rectilinear",
    canvas_scale: float = 1.0,
) -> AlignmentResult:
    if not base.is_file() or not meteor_files:
        raise ValueError("请选择对齐参考图和至少一张流星原图")
    for executable, label in ((ptgui, "PTGui"), (siril, "Siril")):
        if not executable.is_file():
            raise FileNotFoundError(f"找不到{label}：{executable}")
    if panorama_projection not in PANORAMA_PROJECTIONS:
        raise ValueError(f"不支持的实验投影：{panorama_projection}")
    canvas_scale = float(np.clip(canvas_scale, 1.0, 1.8))
    output_root.mkdir(parents=True, exist_ok=True)
    task_name = (
        f"MeteorStudio_PTGui_Lab_{panorama_projection}_{round(canvas_scale * 100)}"
        if laboratory else "MeteorStudio_PTGui"
    )
    task_dir = output_root / task_name
    counter = 2
    while task_dir.exists():
        task_dir = output_root / f"{task_name}_v{counter}"
        counter += 1
    proxy_dir = task_dir / "cache" / "sky_proxies"
    mask_dir = task_dir / "cache" / "sky_masks"
    ptgui_proxy_dir = task_dir / "cache" / "ptgui_full_proxies"
    siril_dir = task_dir / "cache" / "siril_stars"
    layer_dir = task_dir / "aligned_layers"
    reference_dir = task_dir / "aligned_reference"
    project_dir = task_dir / "ptgui_project"
    for folder in (proxy_dir, mask_dir, ptgui_proxy_dir, siril_dir, layer_dir, reference_dir, project_dir):
        folder.mkdir(parents=True, exist_ok=True)
    log_file = task_dir / "pipeline.log"
    manifest_file = task_dir / "alignment_manifest.json"
    scale = 0.25
    notify = progress or (lambda _value, _text: None)
    base_lens = read_lens_info(base, focal_length, sensor_diagonal)
    notify(1, f"生成对齐参考图星点代理（{base_lens.focal_length:.1f}mm，{base_lens.source}）…")
    base_proxy = proxy_dir / "base_sky.png"
    # Automatic mode always examines the full frame. A non-None value remains
    # available only for callers explicitly requesting the legacy safety cap.
    base_width, base_height = make_sky_proxy(base, base_proxy, scale, sky_fraction)
    base_stars, siril_log = siril_find_stars(siril, base_proxy, siril_dir, "base")
    base_proxy_image = cv2.imdecode(np.fromfile(base_proxy, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if base_proxy_image is None:
        raise RuntimeError("无法读取对齐参考图星点代理")
    base_stars = filter_sky_stars(base_stars, base_proxy_image.shape)
    base_sky_mask = make_star_sky_mask(base_proxy_image.shape, base_stars)
    cv2.imencode(".png", base_sky_mask)[1].tofile(mask_dir / "base_sky_mask.png")
    base_ptgui_proxy = ptgui_proxy_dir / "base_sky_full.jpg"
    make_ptgui_sky_proxy(base, base_ptgui_proxy, base_sky_mask)
    items: list[AlignmentItem] = []
    matched: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]] = []
    accepted: list[Path] = []
    accepted_proxies: list[Path] = []
    accepted_lenses: list[ImageLensInfo] = []
    for sequence, source in enumerate(meteor_files, start=1):
        item = AlignmentItem(str(source))
        try:
            lens_info = read_lens_info(source, focal_length, sensor_diagonal)
            item.focal_length = lens_info.focal_length
            item.focal_source = lens_info.source
            notify(
                4 + sequence / len(meteor_files) * 45,
                f"Siril寻找星点 {sequence}/{len(meteor_files)}：{source.name}（{lens_info.focal_length:.1f}mm）",
            )
            proxy = proxy_dir / f"{source.stem}_sky.png"
            width, height = make_sky_proxy(source, proxy, scale, sky_fraction)
            if (width, height) != (base_width, base_height):
                raise ValueError(f"尺寸{width}×{height}与对齐参考图{base_width}×{base_height}不同")
            stars, star_log = siril_find_stars(siril, proxy, siril_dir, source.stem)
            siril_log += "\n" + star_log
            proxy_image = cv2.imdecode(np.fromfile(proxy, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if proxy_image is None:
                raise RuntimeError("无法读取当前星点代理")
            stars = filter_sky_stars(stars, proxy_image.shape)
            sky_mask = make_star_sky_mask(proxy_image.shape, stars)
            item.sky_coverage = float(np.count_nonzero(sky_mask) / sky_mask.size)
            cv2.imencode(".png", sky_mask)[1].tofile(mask_dir / f"{source.stem}_sky_mask.png")
            pairs, median_error = match_star_pairs(proxy, base_proxy, stars, base_stars, scale)
            item.control_points = len(pairs)
            item.median_error = median_error
            accepted_solution, needs_review, quality_message = alignment_solution_quality(
                len(pairs), median_error,
            )
            if not accepted_solution:
                raise RuntimeError(quality_message)
            # A 3–8 px solution is still valuable as a basic placement, but it
            # should never be presented as production-ready.  Export it through
            # the same independent PTGui path and let the user fine-tune from
            # that starting point instead of beginning again from zero.
            if needs_review:
                item.status = "需复查"
                item.message = f"已保留基础对齐；{quality_message}，建议100%检查后微调"
            else:
                item.status = "可对齐"
            lens_note = f"焦距 {lens_info.focal_length:.1f}mm（{lens_info.source}），自动星点有效区 {item.sky_coverage * 100:.1f}%"
            item.message = f"{item.message}；{lens_note}" if item.message else lens_note
            item.layer_index = len(accepted) + 2
            accepted.append(source)
            ptgui_proxy = ptgui_proxy_dir / f"{source.stem}_sky_full.jpg"
            make_ptgui_sky_proxy(source, ptgui_proxy, sky_mask)
            accepted_proxies.append(ptgui_proxy)
            accepted_lenses.append(lens_info)
            matched.append((item.layer_index, pairs))
        except Exception as exc:
            item.status = "需处理"
            item.message = str(exc)
        items.append(item)
        manifest_file.write_text(json.dumps({"items": [asdict(x) for x in items]}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not accepted:
        for item in items:
            item.status = "已输出（原始状态）"
            item.output_layer = item.source
            item.message = (item.message + "；" if item.message else "") + "自动对齐无法建立，已保留原始素材供手动放置"
        result = AlignmentResult(
            str(task_dir), "", str(base), None, items, str(log_file),
            siril_log.splitlines()[1] if len(siril_log.splitlines()) > 1 else "Siril",
            str(ptgui), str(siril), laboratory, panorama_projection, canvas_scale,
        )
        manifest_file.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
        notify(100, "无可靠对齐；已输出全部原始状态")
        return result
    notify(52, "按每张EXIF焦距创建独立镜头模型…")
    sky_projects = project_dir / "sky_projects"
    ready_projects = project_dir / "ready_projects"
    sky_projects.mkdir(exist_ok=True)
    ready_projects.mkdir(exist_ok=True)
    base_layer = None
    duplicate_layer = None
    first_ready_project: Path | None = None
    previous_position: dict[str, float] | None = None
    by_source = {item.source: item for item in items}
    for index, (source, proxy, source_lens, (_old_layer_index, pairs)) in enumerate(
        zip(accepted, accepted_proxies, accepted_lenses, matched), start=1
    ):
        item = by_source[str(source)]
        needs_review = item.status == "需复查"
        try:
            notify(58 + index / len(accepted) * 34, f"PTGui单张对齐 {index}/{len(accepted)}：{source.name}")
            sky_project = sky_projects / f"{source.stem}_sky.pts"
            _run([str(ptgui), "-createproject", str(base_ptgui_proxy), str(proxy), "-output", str(sky_project)], project_dir, log_file)
            configure_single_star_project(
                sky_project, pairs, base_lens, source_lens, base_width, base_height,
                initial_position=previous_position if needs_review else None,
                panorama_projection=panorama_projection, canvas_scale=canvas_scale,
            )
            _run([str(ptgui), "-stitchnogui", str(sky_project)], project_dir, log_file)
            optimized = json.loads(sky_project.read_text(encoding="utf-8"))["project"]
            if not optimized.get("hasbeenoptimized"):
                raise RuntimeError("PTGui没有完成单张优化")
            solved_position = optimized["imagegroups"][1]["position"]["params"]
            previous_position = {key: float(solved_position[key]) for key in ("yaw", "pitch", "roll")}
            ready_project = ready_projects / f"{source.stem}_READY.pts"
            _run([str(ptgui), "-createproject", str(base), str(source), "-output", str(ready_project)], project_dir, log_file)
            configure_input_lenses(
                ready_project, [base_lens, source_lens], base_width, base_height,
                panorama_projection=panorama_projection, canvas_scale=canvas_scale,
            )
            transfer_single_solution(ready_project, sky_project)
            first_ready_project = first_ready_project or ready_project
            if export_layers:
                output_base = layer_dir / f"{source.stem}_aligned.tif"
                configure_layer_export(ready_project, output_base)
                _run([str(ptgui), "-stitchnogui", str(ready_project)], project_dir, log_file)
                generated_base = layer_dir / f"{source.stem}_aligned0000.tif"
                generated_meteor = layer_dir / f"{source.stem}_aligned0001.tif"
                if not generated_base.is_file() or not generated_meteor.is_file():
                    raise RuntimeError("PTGui缺少单张导出图层")
                if base_layer is None:
                    base_target = reference_dir / "alignment_reference.tif"
                    os.replace(generated_base, base_target)
                    base_layer = str(base_target)
                else:
                    generated_base.unlink()
                target = layer_dir / f"{source.stem}.tif"
                os.replace(generated_meteor, target)
                item.output_layer = str(target)
                item.status = "已导出（需复查）" if needs_review else "已导出"
        except Exception as exc:
            item.status = "需处理"
            item.message = str(exc)
    project_file = first_ready_project or project_dir
    # A per-image failure must not remove the photograph from the workflow.
    # Keep the read-only original as the editable fallback; successfully
    # aligned items still retain the same original path for the Studio reset.
    for item in items:
        if export_layers and item.output_layer is None:
            previous = item.message
            item.status = "已输出（原始状态）"
            item.output_layer = item.source
            item.message = (previous + "；" if previous else "") + "已保留原始素材，可在MeteorStudio中手动放置"
    if export_layers and base_layer is None:
        raise RuntimeError("没有任何单张PTGui工程成功导出")
    result = AlignmentResult(
        str(task_dir), str(project_file), base_layer, duplicate_layer, items,
        str(log_file), siril_log.splitlines()[1] if len(siril_log.splitlines()) > 1 else "Siril",
        str(ptgui), str(siril), laboratory, panorama_projection, canvas_scale,
    )
    manifest_file.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    notify(100, "对齐图层已返回MeteorStudio")
    return result
