"""Promote a validated local MeteorStudio model into release resources."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from meteor_composer import ML_FEATURE_NAMES, user_dataset_file_path, user_model_file_path


def main() -> None:
    root = Path(__file__).resolve().parent
    source_model = user_model_file_path()
    if not source_model.is_file():
        raise SystemExit(f"找不到个性化模型：{source_model}")
    payload = json.loads(source_model.read_text(encoding="utf-8"))
    if payload.get("format") != "meteor-gradient-boosting-v1":
        raise SystemExit("个性化模型格式不受支持")
    if payload.get("feature_names") != ML_FEATURE_NAMES:
        raise SystemExit("个性化模型特征版本与当前代码不一致")
    metrics = payload.get("metrics", {})
    if int(metrics.get("samples", 0)) < 100 or int(metrics.get("positives", 0)) < 20:
        raise SystemExit("模型样本不足，拒绝晋升为内置模型")

    bundled_model = root / "meteor_ranker.json"
    shutil.copy2(source_model, bundled_model)
    source_dataset = user_dataset_file_path()
    if source_dataset.is_file():
        shutil.copy2(source_dataset, root / "candidate_dataset.npz")
        dataset_note = f"并同步累计数据集：{source_dataset}"
    else:
        dataset_note = "本机尚无累计数据集；仅晋升模型"
    print(
        f"已晋升内置模型：{metrics.get('samples')} 样本，"
        f"{metrics.get('positives')} 正样本，训练时间 {metrics.get('learned_at', '未知')}\n"
        f"{dataset_note}"
    )


if __name__ == "__main__":
    main()
