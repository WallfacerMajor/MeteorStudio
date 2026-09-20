"""Conservative spatial reference suggestions, not astronomical color calibration."""
import cv2
import numpy as np
from white_balance import sample_neutral


def suggest_neutral_points(image, token, limit=5):
    """Rank smooth patches. Uniform casts are allowed; intrinsic neutrality is unknown.

    Analyze a bounded thumbnail, then validate gains against original pixels.
    Never label a patch as sky, dust, nebula, or a known Galactic location.
    """
    h, w = image.shape[:2]
    scale = min(1., 1000 / max(h, w))
    small = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) / 65535
    sh, sw = small.shape[:2]
    radius = max(5, round(min(sh, sw)*.018))
    ranked = []
    for y in range(radius, sh-radius, radius*2):
        token.raise_if_cancelled()
        for x in range(radius, sw-radius, radius*2):
            patch = small[y-radius:y+radius+1, x-radius:x+radius+1].reshape(-1, 3)
            valid = (patch.min(axis=1) > .025) & (patch.max(axis=1) < .94)
            if valid.mean() < .98:
                continue
            median = np.median(patch, axis=0)
            # Reject strong texture, stars, spatial color changes and extreme casts.
            residual = np.max(np.abs(patch-median) / np.maximum(median, .04), axis=1)
            roughness = float(np.percentile(residual, 90))
            if roughness > .10 or (residual > .3).mean() > .005 or median.max()/median.min() > 2.5:
                continue
            ox, oy = min(w-1, round(x/scale)), min(h-1, round(y/scale))
            try:
                gains = sample_neutral(image, ox, oy)
            except ValueError:
                continue
            ranked.append((roughness, oy, ox, gains))
    chosen = []
    for roughness, y, x, gains in sorted(ranked):
        if all(((x-p['x'])/w)**2 + ((y-p['y'])/h)**2 > .20**2 for p in chosen):
            chosen.append(dict(x=x, y=y, gains=gains, roughness=roughness))
            if len(chosen) >= limit:
                break
    token.raise_if_cancelled()
    return chosen
