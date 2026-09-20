import tempfile
import unittest
import queue
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meteor_learning import build_feedback_dataset
from model_upgrade import sequence_group, metrics


class ModelUpgradeTests(unittest.TestCase):
    def test_exported_challenger_preserves_predictor_scores(self):
        from model_upgrade import train
        from meteor_learning import export_model
        from meteor_detection import ML_FEATURE_NAMES, predict_gradient_boosting
        x = np.random.default_rng(8).normal(size=(50, 34)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int8)
        model = train(x, y, {'n_estimators': 5, 'max_depth': 2})
        payload = export_model(model, ML_FEATURE_NAMES, {})
        np.testing.assert_allclose(
            [predict_gradient_boosting(row, payload) for row in x],
            model.predict_proba(x)[:, 1], atol=1e-7,
        )

    def test_screening_classifies_candidates_beyond_old_twelve_limit(self):
        from meteor_screening import MeteorScreeningWindow
        from background_tasks import CancellationToken
        image = np.full((90, 140, 3), 20, np.uint8)
        seen = []
        def features(_maps, _start, _end, score):
            seen.append(score)
            return np.zeros(34, np.float32)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for index in range(3):
                (source / f'frame_{index}.jpg').touch()
            work_queue = queue.Queue()
            window = SimpleNamespace(work_queue=work_queue, preview_cache_budget=32 << 20)
            proposals = [((10, i + 20), (70, i + 20), i + 1) for i in range(13)]
            with (patch('meteor_screening.capture_sort_key', side_effect=lambda p: ('', p.name)),
                  patch('meteor_screening.screening_preview', return_value=image),
                  patch('meteor_composer.load_meteor_ranker', return_value=None),
                  patch('meteor_composer.detect_trails', return_value=(proposals, 0)),
                  patch('meteor_screening.candidate_feature_vector', side_effect=features)):
                MeteorScreeningWindow._analyze_worker(window, source, 1, str(source),
                                                      CancellationToken('test', 1), '最高精度')
            messages = list(work_queue.queue)
            self.assertFalse(any(item[0] == 'analysis_error' for item in messages), messages)
        self.assertIn(13, seen)

    def test_adjacent_and_cross_workflow_names_share_validation_group(self):
        self.assertEqual(sequence_group('DSC06021.tif'), sequence_group('screening:C:\\photos\\DSC06026.ARW'))
        self.assertNotEqual(sequence_group('DSC06021.tif'), sequence_group('DSC06061.tif'))

    def test_metrics_include_false_positives_and_misses_not_only_top1(self):
        result = metrics(np.array([1, 1, 0, 0]), np.array([.9, .4, .6, .1]))
        self.assertEqual(result['false_positives'], 1)
        self.assertEqual(result['misses'], 1)

    def test_unmarked_candidates_are_not_negative_training_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            source, base = Path(folder) / 'a.tif', Path(folder) / 'b.tif'
            source.touch()
            base.touch()
            image = np.zeros((100, 100, 3), np.uint8)
            mask = np.zeros((100, 100), np.float32)
            mask[10:20, 10:40] = 1
            stroke = SimpleNamespace(points=[(.1, .15), (.4, .15)], width=3, feather=1,
                                     erase=False, locked=False, auto_score=None)
            toolkit = {
                'read_image': lambda _: image, 'detection_preview': lambda _: (image, 1),
                'Stroke': lambda *args: args,
                'build_mask_crop': lambda *_: (mask, (0, 0, 100, 100)),
                'detect_trails': lambda *_, **kw: ([((10, 15), (40, 15), 90), ((60, 80), (90, 80), 80)], 0),
                'prepare_ml_maps': lambda *_: None,
                'candidate_feature_vector': lambda *_: np.zeros(34, np.float32),
            }
            result = build_feedback_dataset({source: [stroke]}, {str(source): base}, toolkit, lambda *_: None)
            self.assertEqual(result['y'].tolist(), [1])
            stroke.auto_score = 90
            result = build_feedback_dataset({source: [stroke]}, {str(source): base}, toolkit, lambda *_: None)
            self.assertEqual(len(result['y']), 0)
