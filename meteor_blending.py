"""Local meteor layer operations, shared by previews and full-depth export."""
import cv2
import numpy as np
import hashlib
import threading
from collections import OrderedDict

# Keep legacy names readable in saved projects, but offer only blend operations
# with distinct behavior in the current isolated-signal compositor.
BLEND_MODES = ('线性减淡（添加）', '滤色')


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


# Cache only local source patches, never an entire composite. The key hashes
# pixels and geometry so replaced files, undo and edited masks cannot reuse
# another meteor's repair. Strength is deliberately excluded: it is just a mix.
_star_repairs = OrderedDict()
_star_repairs_lock = threading.Lock()
_star_repairs_bytes = 0
_STAR_REPAIR_BUDGET = 64 << 20


def remove_mask_stars(patch, alpha, points, width, strength):
    """Reuse the expensive star repair while the user adjusts its mix strength."""
    amount = float(np.clip(strength/100., 0, 1))
    if amount == 0 or min(patch.shape[:2]) < 5 or len(points) < 2:
        return patch
    digest=hashlib.blake2b(digest_size=20)
    for array in (patch,alpha,np.asarray(points,dtype=np.float64)):
        contiguous=np.ascontiguousarray(array)
        digest.update(str((contiguous.shape,contiguous.dtype.str)).encode())
        digest.update(memoryview(contiguous).cast('B'))
    digest.update(float(width).hex().encode())
    key=digest.digest()
    missing=object()
    with _star_repairs_lock:
        repaired=_star_repairs.get(key,missing)
        if repaired is not missing:_star_repairs.move_to_end(key)
    if repaired is missing:
        repaired=_repair_mask_stars(patch,alpha,points,width)
        size=0 if repaired is None else repaired.nbytes
        global _star_repairs_bytes
        if size<=_STAR_REPAIR_BUDGET:
            with _star_repairs_lock:
                previous=_star_repairs.pop(key,None)
                if previous is not None:_star_repairs_bytes-=previous.nbytes
                _star_repairs[key]=repaired;_star_repairs_bytes+=size
                while _star_repairs_bytes>_STAR_REPAIR_BUDGET or len(_star_repairs)>32:
                    _,removed=_star_repairs.popitem(last=False)
                    if removed is not None:_star_repairs_bytes-=removed.nbytes
    if repaired is None:return patch
    maximum=65535. if patch.dtype==np.uint16 else 255.
    rgb=patch.astype(np.float32)/maximum
    return np.clip((rgb+(repaired-rgb)*amount)*maximum,0,maximum).astype(patch.dtype)


def _repair_mask_stars(patch, alpha, points, width):
    """Detect and inpaint off-axis stars independently of the mix strength."""
    maximum = 65535. if patch.dtype == np.uint16 else 255.
    rgb = patch.astype(np.float32)/maximum
    gray = rgb.max(axis=2)
    radius = max(2, min(12, round(width*.22)))
    background = cv2.morphologyEx(gray, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1)))
    high = np.maximum(gray-background, 0)
    samples = high[alpha > .01]
    if samples.size < 16:
        return None
    median = float(np.median(samples))
    noise = float(np.median(np.abs(samples-median)))
    candidates = ((high > max(.003, median+6*1.4826*noise)) & (alpha>.001)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    stars = np.zeros(gray.shape, np.uint8)
    protected = np.zeros_like(stars)
    path = np.asarray(points, dtype=np.float32).copy()
    # Protect the whole emission corridor, including uncertain/off-centre masks
    # and disconnected weak tail segments beyond the marked endpoints.
    for index, neighbour in ((0,1),(-1,-2)):
        direction=path[index]-path[neighbour]
        length=float(np.linalg.norm(direction))
        if length>0:path[index]+=direction/length*max(3,width*.5)
    cv2.polylines(protected,[np.int32(np.rint(path))],False,255,max(5,round(width*.4)),cv2.LINE_8)
    for index in range(1, count):
        x,y,w,h,area = stats[index]
        if area < 2 or area > np.pi*(radius*2)**2:
            continue
        component = labels[y:y+h,x:x+w] == index
        if np.any(protected[y:y+h,x:x+w][component]):
            continue
        yy,xx = np.where(component)
        cov = np.cov(np.array([xx,yy], dtype=np.float32))
        eig = np.linalg.eigvalsh(cov)
        if eig[-1]/max(.3, eig[0]) > 2.5:
            continue
        stars[y:y+h,x:x+w][labels[y:y+h,x:x+w] == index] = 255
    stars = cv2.dilate(stars, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    stars[(protected>0) | (alpha<=.001)] = 0
    if not np.any(stars):
        return None
    repaired = np.stack([cv2.inpaint(rgb[...,c], stars, max(2, radius), cv2.INPAINT_NS)
                         for c in range(3)], axis=2)
    return repaired
