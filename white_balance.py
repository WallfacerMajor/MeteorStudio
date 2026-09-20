"""Relative white balance in linear sRGB; original implementation."""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image, ImageCms
from laboratory import read_pixels

RAW_SUFFIXES = {".arw", ".nef", ".nrw", ".cr2", ".cr3", ".crw", ".dng", ".raf", ".orf", ".rw2"}
LUMA = np.array([.2126, .7152, .0722])


def decode_srgb(values):
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def encode_srgb(values):
    return np.where(values <= .0031308, values * 12.92, 1.055 * np.maximum(values, 0) ** (1 / 2.4) - .055)


def validate_settings(settings):
    if not isinstance(settings, dict) or settings.get("version", 1) != 1:
        raise ValueError("不支持的白平衡设置")
    try:
        warmth, tint = float(settings.get("warmth", 0)), float(settings.get("tint", 0))
        neutral = np.asarray(settings.get("neutral", [1, 1, 1]), dtype=float)
        strength = float(settings.get("neutral_strength", 100))
    except (TypeError, ValueError) as exc:
        raise ValueError("白平衡参数无效") from exc
    if not np.isfinite([warmth, tint]).all() or max(abs(warmth), abs(tint)) > 100:
        raise ValueError("冷暖和色调须在 -100 到 100 之间")
    if neutral.shape != (3,) or not np.isfinite(neutral).all() or np.any(neutral < .125) or np.any(neutral > 8):
        raise ValueError("中性点增益无效，请重新取样")
    if not np.isfinite(strength) or not 0 <= strength <= 100:
        raise ValueError("校准强度须在 0 到 100 之间")
    baseline = settings.get("raw_baseline", "camera")
    if baseline not in ("camera", "daylight"):
        raise ValueError("未知 RAW 解码基准")
    equipment = settings.get("equipment", {})
    if not isinstance(equipment, dict) or any(not isinstance(v, str) or len(v) > 300 for v in equipment.values()):
        raise ValueError("设备预设内容无效")
    return dict(version=1, warmth=warmth, tint=tint, neutral=neutral.tolist(), neutral_strength=strength,
                raw_baseline=baseline, equipment={k: equipment.get(k, "") for k in ("camera", "modification", "filter", "reference")})


def gains_for(settings):
    settings = validate_settings(settings)
    w, t = settings["warmth"] / 100, settings["tint"] / 100
    # Relative controls, intentionally not labelled as absolute kelvin.
    gain = np.asarray(settings["neutral"]) ** (settings["neutral_strength"] / 100) * np.exp2([w + t * .25, -t * .5, -w + t * .25])
    return gain / np.dot(gain, LUMA)


def make_lut(settings):
    linear = decode_srgb(np.arange(65536, dtype=np.float64) / 65535)
    adjusted = np.minimum(linear[:, None] * gains_for(settings), 1)
    return np.rint(encode_srgb(adjusted) * 65535).clip(0, 65535).astype(np.uint16)


def apply_lut(image, lut):
    result = np.empty_like(image)
    for channel in range(3):
        result[..., channel] = lut[image[..., channel], channel]
    return result


def sample_neutral(image, x, y, radius=7):
    h, w = image.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("请在照片内取样")
    patch = image[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)].reshape(-1, 3)
    valid = patch[(patch.min(axis=1) > 512) & (patch.max(axis=1) < 64500)]
    if len(valid) < max(1, len(patch) // 3):
        raise ValueError("该区域太暗或已过曝，请选择应当为灰色／白色的中性区域")
    rgb = np.median(decode_srgb(valid.astype(float) / 65535), axis=0)
    gains = np.dot(rgb, LUMA) / rgb
    validate_settings({"neutral": gains})
    return gains.tolist()


def read_source(path, raw_baseline="camera"):
    if raw_baseline not in ("camera", "daylight"):
        raise ValueError("未知 RAW 解码基准")
    path = Path(path)
    if path.suffix.lower() in RAW_SUFFIXES:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            # Camera WB is the baseline; controls remain a relative RGB edit.
            linear = raw.postprocess(use_camera_wb=raw_baseline == "camera", use_auto_wb=False, output_color=rawpy.ColorSpace.sRGB,
                                     gamma=(1, 1), output_bps=16, no_auto_bright=True)
        lut = np.rint(encode_srgb(np.arange(65536) / 65535) * 65535).astype(np.uint16)
        for y in range(0, linear.shape[0], 128):
            linear[y:y+128] = lut[linear[y:y+128]]
        return linear
    profile = None
    orientation = 1
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            profile = tags[34675].value if 34675 in tags else None
            orientation = tags[274].value if 274 in tags else 1
    else:
        with Image.open(path) as im:
            profile = im.info.get("icc_profile")
            if path.suffix.lower() == ".png":
                orientation = im.getexif().get(274, 1)
    if profile:
        description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(profile))).lower()
        if "srgb" not in description:
            raise ValueError("当前白平衡使用 sRGB 工作空间，请先将素材转换为 sRGB；不会静默改变其他 ICC 色彩空间")
    pixels = read_pixels(path)
    transforms = {2: lambda a: a[:, ::-1], 3: lambda a: a[::-1, ::-1], 4: lambda a: a[::-1],
                  5: lambda a: a.transpose(1, 0, 2), 6: lambda a: np.rot90(a, -1),
                  7: lambda a: a.transpose(1, 0, 2)[::-1, ::-1], 8: lambda a: np.rot90(a)}
    return np.ascontiguousarray(transforms.get(orientation, lambda a: a)(pixels))


def make_pyramid(image, token):
    levels = [image]
    while max(levels[-1].shape[:2]) > 1000:
        token.raise_if_cancelled()
        h, w = levels[-1].shape[:2]
        levels.append(cv2.resize(levels[-1], (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA))
    for level in levels:
        level.flags.writeable = False
    return levels


def render_view(levels, settings, zoom, center, size, original=False):
    """Render only a screen-sized viewport; never transform an entire 8K frame."""
    h, w = levels[0].shape[:2]
    cw, ch = size
    left, top = center[0] - cw / (2 * zoom), center[1] - ch / (2 * zoom)
    x0, y0 = max(0, int(np.floor(left))), max(0, int(np.floor(top)))
    x1, y1 = min(w, int(np.ceil(left + cw / zoom))), min(h, int(np.ceil(top + ch / zoom)))
    if x1 <= x0 or y1 <= y0:
        return None
    level = min(len(levels) - 1, max(0, int(np.floor(np.log2(1 / zoom)))))
    image = levels[level]
    sx, sy = image.shape[1] / w, image.shape[0] / h
    # Fractional crop through an affine map preserves the exact viewport at
    # every pyramid level, including odd dimensions and comparison toggles.
    out_w, out_h = max(1, round((x1-x0)*zoom)), max(1, round((y1-y0)*zoom))
    matrix = np.array([[sx / zoom, 0, x0 * sx], [0, sy / zoom, y0 * sy]], np.float32)
    view = cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    if not original:
        view = apply_lut(view, make_lut(settings))
    clipped = float(np.count_nonzero(np.max(view, axis=2) == 65535) / (out_w * out_h))
    return ((view.astype(np.uint32) + 128) // 257).astype(np.uint8), ((x0-left)*zoom, (y0-top)*zoom), clipped


def export_image(source, destination, settings, token, progress=lambda *_: None):
    settings = validate_settings(settings)
    if not str(destination).strip():
        raise ValueError("请选择输出目录")
    source, destination = Path(source).resolve(), Path(destination).expanduser().resolve()
    if destination == source.parent or source.parent in destination.parents:
        raise ValueError("请选择原片文件夹之外的输出目录")
    token.raise_if_cancelled()
    progress(0, "重新读取原片…")
    pixels = read_source(source, settings["raw_baseline"])
    token.raise_if_cancelled()
    destination.mkdir(parents=True, exist_ok=True)
    folder = destination / datetime.now().strftime("白平衡_%Y%m%d_%H%M%S_%f")
    folder.mkdir()
    manifest = dict(source=str(source), settings=settings, color_space="sRGB", status="running")
    def record():
        temp = folder / "white_balance.json.tmp"
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(folder / "white_balance.json")
    record()
    output = None
    try:
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        output = tifffile.memmap(folder / "result.partial.tif", shape=pixels.shape, dtype=np.uint16,
                                photometric="rgb", extratags=[(34675, "B", len(profile), profile, False)])
        lut = make_lut(settings)
        for y in range(0, pixels.shape[0], 128):
            token.raise_if_cancelled()
            output[y:y+128] = apply_lut(pixels[y:y+128], lut)
            progress(min(99, round(100 * (y+128) / pixels.shape[0])), "写入 16 位 TIFF…")
        output.flush()
        output._mmap.close()
        output = None
        token.raise_if_cancelled()
        (folder / "result.partial.tif").rename(folder / "result.tif")
        manifest["status"] = "complete"
        progress(100, "导出完成")
        return folder
    except Exception as exc:
        manifest.update(status="cancelled" if token.cancelled else "failed", error=str(exc))
        raise
    finally:
        if output is not None:
            output._mmap.close()
        record()


def export_batch(sources, destination, settings, token, progress=lambda *_: None):
    """Apply one frozen calibration to every source; report partial failures."""
    settings = validate_settings(settings)
    sources = list(dict.fromkeys(Path(p).resolve() for p in sources))
    if not sources or not str(destination).strip():
        raise ValueError("请选择批量素材和输出目录")
    destination = Path(destination).expanduser().resolve()
    for source in sources:
        if destination == source.parent or source.parent in destination.parents:
            raise ValueError("批量输出目录必须位于所有素材目录之外")
        if not source.is_file():
            raise ValueError(f"素材不存在：{source}")
    token.raise_if_cancelled()
    destination.mkdir(parents=True, exist_ok=True)
    folder = destination / datetime.now().strftime("改机批量_%Y%m%d_%H%M%S_%f")
    folder.mkdir()
    report = dict(settings=settings, sources=[str(p) for p in sources], status="running", items=[])
    def record():
        temporary = folder / "batch.json.tmp"
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(folder / "batch.json")
    record()
    try:
        for index, source in enumerate(sources):
            token.raise_if_cancelled()
            try:
                output = export_image(source, folder, settings, token,
                    lambda percent, message: progress((index + percent/100) * 100/len(sources), f"{index+1}/{len(sources)} · {source.name} · {message}"))
                named_output = folder / f"{index+1:04d}_{source.stem[:80]}"
                # Only rename our newly-created output directory, never input.
                if output.resolve().parent != folder.resolve() or named_output.resolve().parent != folder.resolve():
                    raise ValueError("输出目录越界")
                output.rename(named_output)
                output = named_output
                report["items"].append(dict(source=str(source), output=str(output), status="complete"))
            except Exception as exc:
                report["items"].append(dict(source=str(source), status="cancelled" if token.cancelled else "failed", error=str(exc)))
                if token.cancelled:
                    raise
            record()
            progress((index + 1) * 100 / len(sources), f"已处理 {index+1}/{len(sources)} 张；失败项已记录")
        report["status"] = "complete_with_errors" if any(item["status"] != "complete" for item in report["items"]) else "complete"
        return folder
    except Exception:
        report["status"] = "cancelled" if token.cancelled else "failed"
        raise
    finally:
        record()
