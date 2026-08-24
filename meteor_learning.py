from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


PARAMETERS = {
    "n_estimators": 120,
    "learning_rate": 0.04,
    "max_depth": 2,
    "min_samples_leaf": 10,
    "subsample": 0.8,
}


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    result = np.ones(len(y), dtype=np.float64)
    result[y == 1] = len(y) / (2.0 * positives)
    result[y == 0] = len(y) / (2.0 * negatives)
    return result


def top1_accuracy(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> tuple[int, int, float]:
    hit = total = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        if not np.any(y[indices] == 1):
            continue
        total += 1
        hit += int(y[indices[np.argmax(scores[indices])]] == 1)
    return hit, total, hit / max(1, total)


def export_model(model: GradientBoostingClassifier, feature_names: list[str], metrics: dict) -> dict:
    trees = []
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append({
            "left": tree.children_left.astype(int).tolist(),
            "right": tree.children_right.astype(int).tolist(),
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.astype(float).tolist(),
            "value": tree.value[:, 0, 0].astype(float).tolist(),
        })
    base_raw = float(model._raw_predict_init(np.zeros((1, len(feature_names)), np.float32))[0, 0])
    return {
        "format": "meteor-gradient-boosting-v1",
        "feature_names": feature_names,
        "base_raw": base_raw,
        "learning_rate": float(model.learning_rate),
        "trees": trees,
        "metrics": metrics,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def build_feedback_dataset(
    marked: dict[Path, list], pairs: dict[str, Path], toolkit: dict,
    progress: Callable[[float, str], None],
) -> dict[str, np.ndarray]:
    rows, labels, groups, legacy, metadata = [], [], [], [], []
    total = max(1, len(marked))
    for index, (source_path, strokes) in enumerate(marked.items(), start=1):
        base_path = pairs.get(str(source_path))
        if base_path is None or not source_path.is_file() or not base_path.is_file():
            continue
        source, scale = toolkit["detection_preview"](toolkit["read_image"](source_path))
        base, _ = toolkit["detection_preview"](toolkit["read_image"](base_path))
        if source.shape != base.shape:
            base = cv2.resize(base, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_AREA)
        height, width = source.shape[:2]
        scaled = [toolkit["Stroke"](
            item.points, max(1, round(item.width * scale)), max(0, round(item.feather * scale)),
            item.erase, item.locked, item.auto_score,
        ) for item in strokes]
        ground_truth = np.zeros((height, width), dtype=np.float32)
        built = toolkit["build_mask_crop"](scaled, width, height)
        if built is not None:
            crop, (x0, y0, x1, y1) = built
            ground_truth[y0:y1, x0:x1] = crop
        dilated = cv2.dilate((ground_truth > 0.12).astype(np.uint8), np.ones((25, 25), np.uint8))
        candidates, _planes = toolkit["detect_trails"](source, base, ranked=True)
        maps = toolkit["prepare_ml_maps"](source, base)
        for candidate_index, (start, end, old_score) in enumerate(candidates):
            samples = max(20, int(np.hypot(end[0] - start[0], end[1] - start[1])))
            xs = np.linspace(start[0], end[0], samples).clip(0, width - 1).astype(np.int32)
            ys = np.linspace(start[1], end[1], samples).clip(0, height - 1).astype(np.int32)
            overlap = float(np.mean(dilated[ys, xs] > 0))
            rows.append(toolkit["candidate_feature_vector"](maps, start, end, old_score))
            labels.append(int(overlap >= 0.18))
            groups.append(source_path.name)
            legacy.append(old_score / 100.0)
            metadata.append(f"{source_path.name}:{candidate_index}")
        progress(index / total * 70.0, f"AI 学习样本 {index}/{total}：{source_path.name}")
    return {
        "x": np.asarray(rows, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.int8),
        "groups": np.asarray(groups),
        "legacy": np.asarray(legacy, dtype=np.float32),
        "metadata": np.asarray(metadata),
    }


def learn_from_feedback(
    marked: dict[Path, list], pairs: dict[str, Path], toolkit: dict,
    base_dataset_path: Path, current_model: dict, user_model_path: Path,
    progress: Callable[[float, str], None],
) -> dict:
    feedback = build_feedback_dataset(marked, pairs, toolkit, progress)
    if len(feedback["y"]) < 30 or len(np.unique(feedback["y"])) < 2:
        raise ValueError("有效训练反馈不足：至少需要同时包含正候选和负候选")
    loaded = np.load(base_dataset_path, allow_pickle=False)
    base = {key: loaded[key] for key in loaded.files}
    replaced_groups = set(feedback["groups"].tolist())
    keep = np.asarray([group not in replaced_groups for group in base["groups"]], dtype=bool)
    dataset = {
        key: np.concatenate((base[key][keep], feedback[key]))
        for key in ("x", "y", "groups", "legacy", "metadata")
    }
    x, y, groups = dataset["x"], dataset["y"], dataset["groups"]
    if len(np.unique(groups)) < 5:
        raise ValueError("可用于按图片验证的样本不足 5 张")
    scores = np.zeros(len(y), dtype=np.float64)
    splitter = GroupKFold(n_splits=5)
    for fold, (train, valid) in enumerate(splitter.split(x, y, groups), start=1):
        model = GradientBoostingClassifier(random_state=42, **PARAMETERS)
        model.fit(x[train], y[train], sample_weight=balanced_weights(y[train]))
        scores[valid] = model.predict_proba(x[valid])[:, 1]
        progress(70 + fold * 4, f"AI 留图验证 {fold}/5")
    hit, total, top1 = top1_accuracy(y, scores, groups)
    validation = {
        "auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "top1_hit": hit, "top1_total": total, "top1_accuracy": top1,
    }
    previous = current_model.get("metrics", {}).get("cross_validation", {})
    previous_ap = float(previous.get("average_precision", 0.0))
    previous_top1 = float(previous.get("top1_accuracy", 0.0))
    accepted = (
        validation["average_precision"] >= previous_ap - 0.03
        and validation["top1_accuracy"] >= previous_top1 - 0.03
    )
    report = {
        "accepted": accepted,
        "feedback_samples": int(len(feedback["y"])),
        "feedback_positive": int(feedback["y"].sum()),
        "combined_samples": int(len(y)),
        "validation": validation,
        "previous_validation": {
            "average_precision": previous_ap, "top1_accuracy": previous_top1,
        },
    }
    if not accepted:
        progress(100, "新模型验证未通过，继续使用原模型")
        return report
    final_model = GradientBoostingClassifier(random_state=42, **PARAMETERS)
    final_model.fit(x, y, sample_weight=balanced_weights(y))
    metrics = {
        "cross_validation": validation,
        "samples": int(len(y)), "positives": int(y.sum()),
        "recommended_threshold": 55,
        "learned_at": datetime.now().isoformat(timespec="seconds"),
    }
    payload = export_model(final_model, toolkit["ML_FEATURE_NAMES"], metrics)
    backup_dir = user_model_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / ("meteor_ranker_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    _atomic_json(backup, current_model)
    _atomic_json(user_model_path, payload)
    report["model_path"] = str(user_model_path)
    report["backup_path"] = str(backup)
    progress(100, "个性化 AI 模型已通过验证并保存")
    return report
