"""Directional additive background correction, independently implemented."""
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import ImageCms
from white_balance import decode_srgb, encode_srgb, read_source


DIRECTIONS = {'底部': 'bottom', '顶部': 'top', '左侧': 'left', '右侧': 'right'}


def validate(settings):
    if not isinstance(settings, dict):
        raise ValueError('无效的光污染设置')
    data = dict(settings)
    if data.get('direction') not in DIRECTIONS.values():
        raise ValueError('未知光污染方向')
    for key, low, high in (('start', 0, .8), ('falloff', .5, 3), ('strength', 0, 1.5)):
        value = float(data[key])
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError('光污染参数超出范围')
        data[key] = value
    rectangles = np.asarray(data.get('protected', []), dtype=float)
    if rectangles.size and (rectangles.ndim != 2 or rectangles.shape[1] != 4 or not np.isfinite(rectangles).all() or np.any(rectangles < 0) or np.any(rectangles > 1) or np.any(rectangles[:, 2:] <= rectangles[:, :2])):
        raise ValueError('保护区域无效')
    data['protected'] = rectangles.tolist()
    if len(data['protected']) > 100:
        raise ValueError('保护区域最多 100 个')
    return data


def field(x, y, settings):
    direction = settings['direction']
    distance = {'bottom': y, 'top': 1-y, 'left': 1-x, 'right': x}[direction]
    return np.clip((distance-settings['start'])/(1-settings['start']), 0, 1) ** settings['falloff']


def protection(x, y, rectangles):
    weight = np.ones(np.broadcast_shapes(x.shape, y.shape), dtype=np.float32)
    for x0, y0, x1, y1 in rectangles:
        dx = np.maximum(np.maximum(x0-x, x-x1), 0)
        dy = np.maximum(np.maximum(y0-y, y-y1), 0)
        distance = np.sqrt(dx*dx+dy*dy)
        fade = np.clip(distance/.025, 0, 1)
        weight *= fade*fade*(3-2*fade)
    return weight


def estimate(image, settings, token):
    settings = validate(settings)
    h, w = image.shape[:2]
    scale = min(1., 900/max(h, w))
    small = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    linear = decode_srgb(small.astype(np.float32)/65535)
    sh, sw = linear.shape[:2]
    samples, weights = [], []
    for yy in np.array_split(np.arange(sh), min(24, sh)):
        token.raise_if_cancelled()
        for xx in np.array_split(np.arange(sw), min(24, sw)):
            x, y = float(xx.mean()/max(1, sw-1)), float(yy.mean()/max(1, sh-1))
            # Reject any cell touching a protected region, not only its center.
            if any(xx[-1]/max(1, sw-1) >= a and xx[0]/max(1, sw-1) <= c and yy[-1]/max(1, sh-1) >= b and yy[0]/max(1, sh-1) <= d for a,b,c,d in settings['protected']):
                continue
            patch = linear[yy[0]:yy[-1]+1, xx[0]:xx[-1]+1].reshape(-1, 3)
            valid = (patch.min(axis=1) > .0003) & (patch.max(axis=1) < .8)
            if valid.mean() < .8:
                continue
            samples.append(np.percentile(patch[valid], 30, axis=0))
            weights.append(float(field(np.array(x), np.array(y), settings)))
    weights, samples = np.asarray(weights), np.asarray(samples)
    if len(weights) < 20 or np.ptp(weights) < .5:
        raise ValueError('可用背景区域不足：需要同时保留较暗天空和靠近光污染方向的天空供估计')
    design = np.column_stack((np.ones(len(weights)), weights))
    keep = np.ones(len(weights), dtype=bool)
    for _ in range(6):
        token.raise_if_cancelled()
        if keep.sum() < 20 or np.ptp(weights[keep]) < .5:
            raise ValueError('背景样本不可靠，请调整保护区域或方向')
        coefficients = np.linalg.lstsq(design[keep], samples[keep], rcond=None)[0]
        residual = samples - design @ coefficients
        center = np.median(residual[keep], axis=0)
        spread = np.maximum(np.median(np.abs(residual[keep]-center), axis=0)*1.4826, .00015)
        keep = np.all((residual-center > -3*spread) & (residual-center < 2*spread), axis=1)
    amplitude = np.clip(coefficients[1], 0, .5)
    amplitude[amplitude < .0002] = 0
    return dict(amplitude=amplitude.tolist(), samples=int(keep.sum()))


def correct(image, settings, model, full_shape=None, origin=(0, 0), step=1):
    settings = validate(settings)
    amplitude = np.asarray(model['amplitude'], dtype=float)
    if amplitude.shape != (3,) or not np.isfinite(amplitude).all() or np.any(amplitude < 0) or np.any(amplitude > .5):
        raise ValueError('无效的光污染模型')
    if settings['strength'] == 0 or not amplitude.any():
        return image.copy()
    h, w = full_shape or image.shape[:2]
    x = (origin[0]+np.arange(image.shape[1])[None, :]*step)/max(1, w-1)
    y = (origin[1]+np.arange(image.shape[0])[:, None]*step)/max(1, h-1)
    amount = field(x, y, settings)*protection(x, y, settings['protected'])*settings['strength']
    background = amount[..., None]*amplitude
    linear = decode_srgb(image.astype(np.float32)/65535)
    # Keep dark pixels from being driven below zero; use one scale for all channels.
    limiter = np.minimum(1., np.min(np.divide(linear*.98, np.maximum(background, 1e-12)), axis=2))
    adjusted = np.maximum(0, linear-background*limiter[..., None])
    output = np.rint(encode_srgb(adjusted)*65535).clip(0, 65535).astype(np.uint16)
    untouched = np.broadcast_to(amount, image.shape[:2]) == 0
    output[untouched] = image[untouched]
    return output


def export_image(source, destination, settings, model, token, progress):
    settings = validate(settings)
    source = Path(source).resolve()
    if not str(destination).strip():
        raise ValueError('请选择输出目录')
    destination = Path(destination).expanduser().resolve()
    if destination == source.parent or source.parent in destination.parents:
        raise ValueError('输出目录必须位于原片文件夹之外')
    token.raise_if_cancelled()
    pixels = read_source(source)
    token.raise_if_cancelled()
    folder = destination / datetime.now().strftime('光污染校正_%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    report = dict(source=str(source), settings=settings, model=model, color_space='sRGB', status='running')
    def record():
        temporary = folder / 'light_pollution.json.tmp'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(folder / 'light_pollution.json')
    record()
    output = None
    try:
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        output = tifffile.memmap(folder/'result.partial.tif', shape=pixels.shape, dtype=np.uint16,
                                photometric='rgb', extratags=[(34675, 'B', len(profile), profile, False)])
        for y in range(0, len(pixels), 128):
            token.raise_if_cancelled()
            output[y:y+128] = correct(pixels[y:y+128], settings, model, pixels.shape[:2], (0, y))
            progress(min(99, (y+128)*100/len(pixels)), '写入 16 位光污染校正结果…')
        output.flush()
        output._mmap.close()
        output = None
        token.raise_if_cancelled()
        (folder/'result.partial.tif').rename(folder/'result.tif')
        report['status'] = 'complete'
        progress(100, '导出完成')
        return folder
    except Exception as exc:
        report.update(status='cancelled' if token.cancelled else 'failed', error=str(exc))
        raise
    finally:
        if output is not None:
            output._mmap.close()
        record()
