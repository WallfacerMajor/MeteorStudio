"""Original, streaming experimental image operations; no upstream code copied."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image, ImageOps

MODES = {
    "trails": ("星轨叠加", "逐像素取亮。按列表顺序叠加固定机位照片；飞机、热像素也会保留。"),
    "mean": ("已对齐降噪", "平均叠加。请先完成星空对齐，并使用相同尺寸、曝光及色彩处理的照片。"),
    "quality": ("批量画质体检", "输出清晰度、背景亮度和过曝比例。仅比较同机位同曝光素材；不自动删除照片。"),
}
SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}


def read_pixels(path):
    path = Path(path)
    if path.suffix.lower() not in SUFFIXES:
        raise ValueError(f"暂不支持此格式：{path.name}，请先转换为 TIFF / PNG / JPG")
    if path.suffix.lower() in {".tif", ".tiff"}:
        data = tifffile.imread(path)
    elif path.suffix.lower() == ".png":
        data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"无法读取：{path.name}")
        if data.ndim == 3 and data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    else:
        with Image.open(path) as im:
            data = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    if data.ndim == 2:
        data = np.repeat(data[..., None], 3, axis=2)
    if data.ndim != 3 or data.shape[2] != 3 or data.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"{path.name}：需要 8 / 16 位 RGB 或灰度图像")
    return data.astype(np.uint16) * 257 if data.dtype == np.uint8 else data


def quality_metrics(image):
    # Fixed-size analysis prevents full-resolution temporary float images.
    h, w = image.shape[:2]
    scale = min(1, 1600 / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small.astype(np.float32) / 65535, cv2.COLOR_RGB2GRAY)
    clipped = sum(np.count_nonzero(np.max(image[y:y + 128], axis=2) >= 65500) for y in range(0, h, 128))
    return dict(width=w, height=h, analysis_scale=scale,
                sharpness=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                background=float(np.median(gray)),
                clipped_fraction=float(clipped / (w * h)))


def run_experiment(paths, destination, mode, token, progress=lambda *_: None):
    if mode not in MODES:
        raise ValueError("未知实验")
    paths = [Path(p).resolve() for p in paths]
    if len(set(paths)) != len(paths) or len(paths) < (1 if mode == "quality" else 2):
        raise ValueError("体检至少需要一张，叠加至少需要两张不重复照片")
    if not str(destination).strip():
        raise ValueError("请选择输出目录")
    for path in paths:
        token.raise_if_cancelled()
        if not path.is_file():
            raise ValueError(f"素材不存在：{path}")
        if path.suffix.lower() not in SUFFIXES:
            raise ValueError(f"暂不支持此格式：{path.name}")
    destination = Path(destination).expanduser().resolve()
    if any(destination == p.parent or p.parent in destination.parents for p in paths):
        raise ValueError("请选择素材文件夹之外的输出目录")
    token.raise_if_cancelled()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / datetime.now().strftime("实验室_%Y%m%d_%H%M%S_%f")
    output.mkdir()
    manifest = {"mode": mode, "sources": [str(p) for p in paths], "status": "running", "algorithm_version": 1}
    def save_manifest():
        temporary = output / "experiment.json.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output / "experiment.json")
    save_manifest()
    accumulator = None
    records = []
    try:
        for index, path in enumerate(paths):
            token.raise_if_cancelled()
            progress(index, f"读取 {index + 1}/{len(paths)} · {path.name}")
            pixels = read_pixels(path)
            token.raise_if_cancelled()
            if mode == "quality":
                records.append(dict(file=str(path), **quality_metrics(pixels)))
            else:
                if accumulator is None:
                    accumulator = pixels.astype(np.float32) if mode == "mean" else pixels.copy()
                elif pixels.shape != accumulator.shape:
                    raise ValueError(f"尺寸不一致：{path.name}；请先对齐，程序不会裁切原图")
                elif mode == "trails":
                    np.maximum(accumulator, pixels, out=accumulator)
                else:
                    # Running mean, row-bounded scratch space; memory independent of frame count.
                    for y in range(0, pixels.shape[0], 128):
                        token.raise_if_cancelled()
                        region = accumulator[y:y + 128]
                        region += (pixels[y:y + 128].astype(np.float32) - region) / (index + 1)
            del pixels
        token.raise_if_cancelled()
        if mode == "quality":
            with (output / "quality.partial.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                # Prevent spreadsheet formula execution through imported filenames.
                writer.writerows({**r, "file": "'" + r["file"] if r["file"].startswith(("=", "+", "-", "@")) else r["file"]} for r in records)
            token.raise_if_cancelled()
            (output / "quality.partial.csv").rename(output / "quality.csv")
        else:
            progress(len(paths), "写入完整分辨率 16 位 TIFF")
            if mode == "mean":
                converted = np.empty(accumulator.shape, dtype=np.uint16)
                for y in range(0, accumulator.shape[0], 128):
                    token.raise_if_cancelled()
                    converted[y:y + 128] = np.rint(accumulator[y:y + 128]).astype(np.uint16)
                accumulator = converted
            tifffile.imwrite(output / "result.partial.tif", accumulator, photometric="rgb")
            token.raise_if_cancelled()
            (output / "result.partial.tif").rename(output / "result.tif")
        manifest["status"] = "complete"
        progress(len(paths), "处理完成")
        return output
    except Exception as exc:
        manifest.update(status="cancelled" if token.cancelled else "failed", error=str(exc))
        raise
    finally:
        save_manifest()
