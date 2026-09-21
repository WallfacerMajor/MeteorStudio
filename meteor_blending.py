"""Local meteor layer operations, shared by previews and full-depth export."""
import cv2
import numpy as np

BLEND_MODES = ('自然融合', '滤色', '线性减淡（添加）', '亮度残差', '普通粘贴')


def blend_kind(mode):
    if mode in ('滤色', 'screen'):
        return 'screen'
    if mode in ('线性减淡（添加）', '线性减淡(添加)', 'add', 'linear_dodge'):
        return 'add'
    return 'legacy'


def blend_signal(destination, signal, alpha, maximum, mode):
    """Blend a black-background meteor layer, keeping alpha outside the blend."""
    dest = destination.astype(np.float32)
    layer = np.clip(signal, 0, maximum)
    if blend_kind(mode) == 'screen':
        blended = dest + (maximum-dest)*layer/maximum
    else:
        blended = np.minimum(dest+layer, maximum)
    return np.clip(dest+(blended-dest)*alpha[..., None], 0, maximum)


def remove_mask_stars(patch, alpha, points, width, strength):
    """Repair compact off-axis stars; leave the meteor's centreline untouched.

    No learned model/download. Work only on the local source patch before warp.
    Stars crossing the protected core deliberately remain ambiguous.
    """
    amount = float(np.clip(strength/100., 0, 1))
    if amount == 0 or min(patch.shape[:2]) < 5 or len(points) < 2:
        return patch
    maximum = 65535. if patch.dtype == np.uint16 else 255.
    rgb = patch.astype(np.float32)/maximum
    gray = rgb.max(axis=2)
    radius = max(2, min(12, round(width*.22)))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1)))
    high = np.maximum(gray-background, 0)
    samples = high[alpha > .01]
    if samples.size < 16:
        return patch
    median = float(np.median(samples))
    noise = float(np.median(np.abs(samples-median)))
    candidates = ((high > max(.003, median+6*1.4826*noise)) & (alpha>.001)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    stars = np.zeros(gray.shape, np.uint8)
    for index in range(1, count):
        x,y,w,h,area = stats[index]
        if area < 2 or area > np.pi*(radius*2)**2:
            continue
        yy,xx = np.where(labels[y:y+h,x:x+w] == index)
        cov = np.cov(np.array([xx,yy], dtype=np.float32))
        eig = np.linalg.eigvalsh(cov)
        if eig[-1]/max(.3, eig[0]) > 4.0:
            continue
        stars[y:y+h,x:x+w][labels[y:y+h,x:x+w] == index] = 255
    stars = cv2.dilate(stars, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    protected = np.zeros_like(stars)
    cv2.polylines(protected, [np.int32(points)], False, 255, max(3, round(width*.22)), cv2.LINE_8)
    stars[(protected>0) | (alpha<=.001)] = 0
    if not np.any(stars):
        return patch
    repaired = np.stack([cv2.inpaint(rgb[...,c], stars, max(2, radius), cv2.INPAINT_NS)
                         for c in range(3)], axis=2)
    return np.clip((rgb+(repaired-rgb)*amount)*maximum, 0, maximum).astype(patch.dtype)
