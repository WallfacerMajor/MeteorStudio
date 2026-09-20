"""Train an isolated challenger with photo-group holdout; never replace user AI."""
from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

from meteor_detection import ML_FEATURE_NAMES, candidate_feature_vector, prepare_ml_maps
from meteor_learning import PARAMETERS, balanced_weights, export_model, build_screening_feedback_dataset
from meteor_screening import screening_preview, temporal_reference


def sequence_group(value: str) -> str:
    # Group short sequence blocks rather than individual candidates. This
    # reduces adjacent-frame leakage; it is not a different-night benchmark.
    name = value.replace('\\', '/').rsplit('/', 1)[-1]
    match = re.search(r'([A-Za-z]+)(\d+)', name)
    return f'{match[1].upper()}:{int(match[2]) // 20}' if match else name.casefold()


def metrics(y, scores):
    predicted = scores >= 0.55
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'average_precision': float(average_precision_score(y, scores)),
            'precision_at_55': float(precision_score(y, predicted, zero_division=0)),
            'recall_at_55': float(recall_score(y, predicted, zero_division=0)),
            'false_positives': int(fp), 'misses': int(fn), 'true_positives': int(tp),
            'true_negatives': int(tn)}


def train(x, y, params):
    model = GradientBoostingClassifier(random_state=42, **params)
    model.fit(x, y, sample_weight=balanced_weights(y))
    return model


@threadpool_limits.wrap(limits=2)
def run(base_path: Path, feedback_path: Path | None, output: Path):
    if output.exists():
        raise ValueError('Choose a new output directory; existing experiments are never overwritten')
    output.mkdir(parents=True)
    cv2.setNumThreads(min(2, os.cpu_count() or 1))
    with np.load(base_path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    decoded = 0
    @lru_cache(maxsize=24)
    def preview(path):
        nonlocal decoded
        image = screening_preview(path)
        decoded += 1
        if decoded % 16 == 0:
            print(f'Reconstructed {decoded} source/neighbor previews', flush=True)
        return image

    feedback = build_screening_feedback_dataset({
        'ML_FEATURE_NAMES': ML_FEATURE_NAMES, 'screening_feedback_path': feedback_path,
        'screening_preview': preview,
        'temporal_reference': temporal_reference, 'prepare_ml_maps': prepare_ml_maps,
        'candidate_feature_vector': candidate_feature_vector,
    }) if feedback_path is not None else {key: value[:0] for key, value in data.items()}
    print(f'Usable explicit feedback: {len(feedback["y"])}', flush=True)
    data = {key: np.concatenate((data[key], feedback[key])) for key in data}
    np.savez_compressed(output / 'training_dataset.npz', **data)
    x, y = data['x'], data['y']
    if x.shape[1] != len(ML_FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('Invalid feature schema or nonfinite training data')
    groups = np.asarray([sequence_group(str(value)) for value in data['groups']])
    train_ids, holdout = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=6021).split(x, y, groups))
    if len(np.unique(y[holdout])) != 2 or len(np.unique(y[train_ids])) != 2:
        raise ValueError('Both labels required in training and holdout')
    candidates = [dict(PARAMETERS),
                  dict(PARAMETERS, n_estimators=220, max_depth=2, min_samples_leaf=7),
                  dict(PARAMETERS, n_estimators=180, max_depth=3, min_samples_leaf=10)]
    validation = []
    for params in candidates:
        scores = np.zeros(len(train_ids))
        for fitting, checking in GroupKFold(4).split(x[train_ids], y[train_ids], groups[train_ids]):
            if len(np.unique(y[train_ids[fitting]])) != 2:
                raise ValueError('Training fold has only one label')
            model = train(x[train_ids[fitting]], y[train_ids[fitting]], params)
            scores[checking] = model.predict_proba(x[train_ids[checking]])[:, 1]
        result = metrics(y[train_ids], scores)
        validation.append(result)
        print(params, result, flush=True)
    # Holdout is never used to choose hyperparameters or thresholds.
    winner = max(range(len(candidates)), key=lambda i: validation[i]['average_precision'])
    baseline = train(x[train_ids], y[train_ids], candidates[0])
    challenger = train(x[train_ids], y[train_ids], candidates[winner])
    old = metrics(y[holdout], baseline.predict_proba(x[holdout])[:, 1])
    new = metrics(y[holdout], challenger.predict_proba(x[holdout])[:, 1])
    accepted = (winner != 0 and new['average_precision'] > old['average_precision']
                and new['false_positives'] <= old['false_positives'] and new['misses'] <= old['misses'])
    report = {'status': 'candidate_passed' if accepted else 'not_promoted',
              'baseline_definition': 'Current training recipe refitted on the same training split; not a claim of superiority to the installed personalized artifact',
              'samples': len(y), 'positives': int(y.sum()), 'explicit_feedback': len(feedback['y']),
              'sequence_groups': len(np.unique(groups)), 'holdout_samples': len(holdout),
              'holdout_groups': sorted(set(groups[holdout].tolist())),
              'holdout_indices': holdout.tolist(), 'parameters': candidates[winner],
              'inner_validation': validation, 'baseline_holdout': old, 'challenger_holdout': new,
              'threshold': 55, 'user_model_modified': False}
    final = train(x, y, candidates[winner])
    payload = export_model(final, ML_FEATURE_NAMES, {'samples': len(y), 'positives': int(y.sum()),
                          'recommended_threshold': 55, 'upgrade_validation': report})
    (output / 'meteor_ranker_candidate.json').write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    (output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'training_dataset.npz', **data)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('candidate_dataset.npz'))
    parser.add_argument('--feedback', type=Path,
                        help='Omit when --base is an already reconstructed training_dataset.npz')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.feedback, args.output)
