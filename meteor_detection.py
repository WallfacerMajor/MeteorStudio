"""Pure candidate feature extraction and ranking helpers.

This module has no Tk, filesystem, or project-state dependencies, allowing the
screening and composition workflows to share one scoring implementation.
"""

from __future__ import annotations

import cv2
import numpy as np


ML_FEATURE_NAMES = [
    "legacy_score", "length_ratio", "mid_x", "mid_y", "abs_horizontal", "abs_vertical",
    "center_mean", "center_median", "center_q90", "center_q99", "center_std",
    "background_mean", "background_q90", "contrast_mean", "contrast_q90",
    "source_contrast", "base_contrast", "inner_outer_ratio", "peak_mean_ratio",
    "smoothness", "gradient_std", "first_second_ratio", "middle_edge_ratio",
    "active95", "active99", "active997", "runs95", "runs99", "runs997",
    "positive_fraction", "negative_fraction", "signed_mean", "signed_q90", "signed_q10",
]


def count_true_runs(values: np.ndarray) -> int:
    padded = np.pad(values.astype(np.int8), (1, 1))
    return int(np.count_nonzero(np.diff(padded) == 1))


def prepare_ml_maps(source: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, ...]:
    src = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dst = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    s2, s98 = np.percentile(src, (2, 98))
    b2, b98 = np.percentile(dst, (2, 98))
    mapped = (src - s2) * float((b98 - b2) / max(5.0, s98 - s2)) + b2
    difference = mapped - dst
    sigma = max(12.0, min(src.shape) / 45.0)
    residual = difference - cv2.GaussianBlur(difference, (0, 0), sigmaX=sigma, sigmaY=sigma)
    magnitude = np.abs(residual)
    limits = np.percentile(magnitude, (95.0, 99.0, 99.7)).astype(np.float32)
    return src, dst, residual, magnitude, limits


def candidate_feature_vector(
    maps: tuple[np.ndarray, ...],
    start: tuple[int, int],
    end: tuple[int, int],
    legacy_score: float,
) -> np.ndarray:
    src, dst, residual, magnitude, limits = maps
    height, width = magnitude.shape
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    length = max(1.0, float(np.hypot(dx, dy)))
    direction = np.array((dx / length, dy / length), dtype=np.float32)
    normal = np.array((-direction[1], direction[0]), dtype=np.float32)
    samples = int(np.clip(round(length * 1.25), 48, 256))
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    center_points = (
        np.array(start, np.float32)[None, :]
        + t[:, None] * np.array((dx, dy), np.float32)[None, :]
    )

    def profile(image: np.ndarray, offset: float) -> np.ndarray:
        points = center_points + normal[None, :] * offset
        xs = points[:, 0].clip(0, width - 1).astype(np.int32)
        ys = points[:, 1].clip(0, height - 1).astype(np.int32)
        return image[ys, xs]

    inner = np.max(np.stack([profile(magnitude, offset) for offset in (-2, 0, 2)]), axis=0)
    outer = np.mean(
        np.stack([profile(magnitude, offset) for offset in (-20, -16, 16, 20)]), axis=0
    )
    source_inner = profile(src, 0)
    source_outer = np.mean(np.stack([profile(src, -16), profile(src, 16)]), axis=0)
    base_inner = profile(dst, 0)
    base_outer = np.mean(np.stack([profile(dst, -16), profile(dst, 16)]), axis=0)
    signed = profile(residual, 0)
    smoothness = float(np.mean(np.abs(np.diff(inner))) / max(1e-3, np.std(inner)))
    gradient_std = float(np.std(np.diff(inner)) / max(1e-3, np.mean(inner)))
    half = max(1, len(inner) // 2)
    first_second = float((np.mean(inner[:half]) + 1e-3) / (np.mean(inner[half:]) + 1e-3))
    edge_count = max(1, len(inner) // 5)
    edge_mean = float((np.mean(inner[:edge_count]) + np.mean(inner[-edge_count:])) * 0.5)
    middle_mean = (
        float(np.mean(inner[edge_count:-edge_count]))
        if len(inner) > edge_count * 2 else float(np.mean(inner))
    )
    active = [inner > float(limit) for limit in limits]
    features = [
        float(legacy_score) / 100.0,
        length / max(width, height),
        (start[0] + end[0]) * 0.5 / max(1, width - 1),
        (start[1] + end[1]) * 0.5 / max(1, height - 1),
        abs(direction[0]), abs(direction[1]),
        float(np.mean(inner)), float(np.median(inner)), float(np.percentile(inner, 90)),
        float(np.percentile(inner, 99)), float(np.std(inner)),
        float(np.mean(outer)), float(np.percentile(outer, 90)),
        float(np.mean(inner) - np.mean(outer)),
        float(np.percentile(inner, 90) - np.percentile(outer, 90)),
        float(np.mean(source_inner - source_outer)), float(np.mean(base_inner - base_outer)),
        float((np.mean(inner) + 1e-3) / (np.mean(outer) + 1e-3)),
        float((np.max(inner) + 1e-3) / (np.mean(inner) + 1e-3)),
        smoothness, gradient_std, first_second,
        float((middle_mean + 1e-3) / (edge_mean + 1e-3)),
        *(float(np.mean(values)) for values in active),
        *(float(count_true_runs(values)) / max(1, len(values)) for values in active),
        float(np.mean(signed > 0)), float(np.mean(signed < 0)), float(np.mean(signed)),
        float(np.percentile(signed, 90)), float(np.percentile(signed, 10)),
    ]
    return np.nan_to_num(
        np.asarray(features, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6
    )


def predict_gradient_boosting(features: np.ndarray, model: dict) -> float:
    raw = float(model["base_raw"])
    for tree in model["trees"]:
        node = 0
        while tree["left"][node] != -1:
            if float(features[tree["feature"][node]]) <= tree["threshold"][node]:
                node = tree["left"][node]
            else:
                node = tree["right"][node]
        raw += float(model["learning_rate"]) * tree["value"][node]
    return float(1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0))))


def calibrate_secondary_candidate_scores(
    candidates: list[tuple[int, tuple[int, int], tuple[int, int], float]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], float]]:
    """Keep a weak short residual available without outranking an obvious meteor."""
    if len(candidates) < 2:
        return candidates

    def length(item) -> float:
        _score, start, end, _legacy = item
        return float(np.hypot(end[0] - start[0], end[1] - start[1]))

    anchor = max(candidates, key=lambda item: (float(item[3]), length(item)))
    anchor_length = length(anchor)
    anchor_legacy = float(anchor[3])
    if anchor_legacy < 85.0 or anchor_length < 1.0:
        return candidates

    calibrated = []
    for item in candidates:
        score, start, end, legacy = item
        item_length = length(item)
        if (
            item is not anchor
            and float(legacy) < 55.0
            and item_length < anchor_length * 0.38
        ):
            score = min(int(score), 49)
        calibrated.append((int(score), start, end, float(legacy)))
    return calibrated
