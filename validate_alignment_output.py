"""Measure residual star displacement in PTGui-exported alignment layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile


def read_sky(path: Path, scale: float = 0.25, sky_fraction: float = 0.604) -> np.ndarray:
    image = tifffile.imread(path)
    rgb = image[..., :3].astype(np.float32)
    if image.shape[-1] > 3:
        alpha = image[..., 3] > 0
        rgb[~alpha] = 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = gray[: round(gray.shape[0] * sky_fraction)]
    background = cv2.GaussianBlur(gray, (0, 0), 12)
    high_pass = np.maximum(gray - background, 0)
    return cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def measure(base: np.ndarray, layer: np.ndarray, full_scale: float = 0.25) -> tuple[int, float, float]:
    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.006, edgeThreshold=15, sigma=1.2)
    key_a, desc_a = sift.detectAndCompute(layer, None)
    key_b, desc_b = sift.detectAndCompute(base, None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("not enough features")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in matches if first.distance < 0.75 * second.distance]
    source = np.float32([key_a[m.queryIdx].pt for m in good])
    target = np.float32([key_b[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 1.5, maxIters=150000, confidence=0.999)
    if matrix is None or mask is None:
        raise RuntimeError("no robust transform")
    keep = mask.ravel().astype(bool)
    source, target = source[keep], target[keep]
    residual = np.linalg.norm(source - target, axis=1) / full_scale
    return len(source), float(np.median(residual)), float(np.percentile(residual, 90))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    args = parser.parse_args()
    base = read_sky(args.project_dir / "aligned_reference" / "clean_base_aligned.tif")
    diagnostics = args.project_dir / "alignment_diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for layer_path in sorted((args.project_dir / "aligned_layers").glob("*.tif")):
        layer = read_sky(layer_path)
        count, median, p90 = measure(base, layer)
        print(f"{layer_path.name}: {count} inliers, median={median:.2f}px, p90={p90:.2f}px")
        overlay = np.zeros((*base.shape, 3), np.uint8)
        overlay[..., 2] = base
        overlay[..., 1] = layer
        cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 96])[1].tofile(
            diagnostics / f"{layer_path.stem}_red_base_green_meteor.jpg"
        )


if __name__ == "__main__":
    main()
