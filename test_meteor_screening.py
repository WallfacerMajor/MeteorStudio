import json
import os
import queue
import tempfile
import tkinter as tk
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from meteor_screening import (
    IMAGE_SUFFIXES, RAW_SUFFIXES, SCREENING_ALGORITHM_VERSION,
    MeteorScreeningWindow, ScreeningCandidate, ScreeningResult,
    discover_screening_sources,
    mark_temporal_repeats,
    estimate_star_sky_mask,
    temporal_reference,
    screening_memory_budgets,
    screening_worker_counts,
    save_screening_diagnostic_event,
    confirmed_track_item,
    load_screening_checkpoint,
    write_screening_checkpoint,
    preview_window,
    suppress_dense_temporal_clutter,
    suppress_repeating_low_confidence_tracks,
    trim_array_cache,
)
from meteor_composer import detect_trails
from meteor_learning import build_screening_feedback_dataset
from background_tasks import CancellationToken


class TemporalReferenceTests(unittest.TestCase):
    def test_confirmed_track_export_keeps_only_user_confirmed_meteors(self):
        item = confirmed_track_item(Path("D:/shoot/frame.arw"), "frame.arw", {
            "analysis_width": 101, "analysis_height": 51,
            "candidates": [
                {"start": [10, 5], "end": [90, 45], "score": 88, "label": "meteor"},
                {"start": [1, 1], "end": [2, 2], "score": 77, "label": "not_meteor"},
                {"start": [3, 3], "end": [4, 4], "score": 66, "label": ""},
            ],
        })
        self.assertIsNotNone(item)
        self.assertEqual(len(item["tracks"]), 1)
        self.assertEqual(item["tracks"][0]["start_normalized"], [0.1, 0.1])
        self.assertTrue(item["tracks"][0]["locked"])

    def test_export_worker_writes_confirmed_track_sidecar_next_to_photos(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.arw"
            source.write_bytes(b"original")
            run_dir = root / "export"
            run_dir.mkdir()
            result = ScreeningResult(
                str(source), [ScreeningCandidate(
                    (10, 5), (90, 45), 92, label="meteor", manual=True,
                )], 92, analysis_width=101, analysis_height=51,
            )
            window = SimpleNamespace(work_queue=queue.Queue())
            with patch("meteor_screening.save_screening_feedback", return_value=None):
                MeteorScreeningWindow._copy_selected_worker(
                    window, run_dir,
                    [(source, __import__("dataclasses").asdict(result), "候选保留", "candidate")],
                    [], {}, CancellationToken("export", 1),
                )
            payload = json.loads(
                (run_dir / "confirmed_meteor_tracks.json").read_text(encoding="utf-8")
            )
        self.assertEqual(payload["format"], "meteor-confirmed-tracks-v1")
        self.assertEqual(len(payload["items"][0]["tracks"]), 1)

    def test_export_legacy_track_uses_cached_proxy_size_without_decoding_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "legacy.arw"
            source.write_bytes(b"original")
            run_dir = root / "export"
            run_dir.mkdir()
            result = ScreeningResult(
                str(source), [ScreeningCandidate(
                    (140, 94), (700, 470), 92, label="meteor", manual=True,
                )], 92,
            )
            window = SimpleNamespace(work_queue=queue.Queue())
            with (
                patch("meteor_screening.save_screening_feedback", return_value=None),
                patch(
                    "meteor_screening.screening_fast_preview",
                    side_effect=AssertionError("RAW decode leaked into export"),
                ),
            ):
                MeteorScreeningWindow._copy_selected_worker(
                    window, run_dir,
                    [(source, __import__("dataclasses").asdict(result), "候选保留", "candidate")],
                    [], {}, CancellationToken("export", 1), (1400, 940),
                )
            payload = json.loads(
                (run_dir / "confirmed_meteor_tracks.json").read_text(encoding="utf-8")
            )
        track = payload["items"][0]["tracks"][0]
        self.assertAlmostEqual(track["start_normalized"][0], 140 / 1399)
        self.assertAlmostEqual(track["start_normalized"][1], 94 / 939)

    def test_diagnostic_events_append_without_copying_source_material(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "diagnostic.json"
            source = Path(folder) / "frame.arw"
            source.write_bytes(b"read-only-original")
            before = source.read_bytes()
            save_screening_diagnostic_event({
                "id": "false-positive", "source_path": str(source),
                "error_type": "false_positive_rejected_candidate",
            }, destination)
            save_screening_diagnostic_event({
                "id": "false-negative", "source_path": str(source),
                "error_type": "false_negative_manual_mark",
            }, destination)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            after = source.read_bytes()
        self.assertEqual([item["id"] for item in payload], [
            "false-positive", "false-negative",
        ])
        self.assertEqual(after, before)

    def test_rejecting_candidate_immediately_records_false_positive(self):
        result = ScreeningResult(
            "D:/shoot/frame_3.arw",
            [ScreeningCandidate((10, 20), (90, 35), 81, features=[0.81] * 34)],
            81,
        )
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        window._selected_result = MagicMock(return_value=result)
        window._candidate_indices_for_action = MagicMock(return_value=[0])
        window._record_screening_diagnostic = MagicMock()
        window._sync_photo_decision_from_candidates = MagicMock()
        window._refresh_tree = MagicMock()
        window._rebuild_preview_overlay = MagicMock()
        window._schedule_autosave = MagicMock()
        window.status = SimpleNamespace(set=MagicMock())
        window._score_cutoff = MagicMock(return_value=50)
        MeteorScreeningWindow._label_candidate(window, "not_meteor")
        window._record_screening_diagnostic.assert_called_once_with(
            result, 0, "false_positive_rejected_candidate",
        )

    def test_diagnostic_record_contains_replay_context_and_named_features(self):
        result = ScreeningResult(
            "D:/shoot/frame_3.arw",
            [ScreeningCandidate((10, 20), (90, 35), 20, features=[0.2] * 34)],
            20, plane_count=2, temporal_hits=1, note="星空区域 82%",
        )
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        window.results = [
            ScreeningResult("D:/shoot/frame_2.arw", [], 0), result,
            ScreeningResult("D:/shoot/frame_4.arw", [], 0),
        ]
        window.files = [Path(item.path) for item in window.results]
        window.analysis_mode = SimpleNamespace(get=lambda: "标准两阶段")
        window.sensitivity = SimpleNamespace(get=lambda: 42)
        window._score_cutoff = lambda: 58
        with patch("meteor_screening.save_screening_diagnostic_event") as save:
            MeteorScreeningWindow._record_screening_diagnostic(
                window, result, 0, "false_negative_under_ranked_candidate",
            )
        record = save.call_args.args[0]
        self.assertEqual(record["neighbor_paths"], [
            str(Path("D:/shoot/frame_2.arw")), str(Path("D:/shoot/frame_4.arw")),
        ])
        self.assertEqual(record["score_cutoff"], 58)
        self.assertEqual(record["result_snapshot"]["note"], "星空区域 82%")
        self.assertEqual(len(record["reviewed_candidate"]["named_features"]), 34)

    def test_analysis_skips_one_unreadable_raw_and_finishes_remaining_sequence(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for name in ("frame_0.arw", "frame_1.arw", "broken.arw", "frame_3.arw"):
                (source / name).touch()
            work_queue = queue.Queue()
            window = SimpleNamespace(
                work_queue=work_queue,
                preview_cache_budget=32 << 20,
            )

            def preview(path: Path, _max_dimension: int = 1400):
                if path.name == "broken.arw":
                    raise RuntimeError("Unsupported file format or not RAW file")
                image = np.full((90, 140, 3), 12, dtype=np.uint8)
                cv2.circle(image, (30 + int(path.stem[-1]) * 5, 25), 1, (210, 210, 210), -1)
                return image

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", return_value=([], 0)),
            ):
                MeteorScreeningWindow._analyze_worker(
                    window, source, 1, str(source), CancellationToken("analysis", 1),
                    "最高精度",
                )
        messages = []
        while not work_queue.empty():
            messages.append(work_queue.get_nowait())
        self.assertFalse(any(item[0] == "analysis_error" for item in messages))
        finished = next(item for item in messages if item[0] == "analysis_finished")
        self.assertEqual(len(finished[3]), 3)
        self.assertEqual(len(finished[5]), 1)
        self.assertTrue(finished[5][0][0].endswith("broken.arw"))

    def test_analyze_click_dispatches_directory_metadata_scan_without_reading_files(self):
        with tempfile.TemporaryDirectory() as folder:
            scheduler = MagicMock()
            window = SimpleNamespace(
                analysis_running=False,
                source_dir=SimpleNamespace(get=lambda: folder),
                files=[Path("old.jpg")], results=[object()],
                decisions={"old": "accept"}, decision_sources={"old": "manual"},
                active_candidates={"old": 0},
                selected_candidates={"old": {0}},
                preview_cache=OrderedDict((('old', np.zeros((1, 1, 3), np.uint8)),)),
                preview_cache_bytes=3,
                proxy_preview_loading={"old"}, proxy_preview_active_path="old",
                full_preview_cache=OrderedDict((('old', np.zeros((1, 1, 3), np.uint8)),)),
                full_preview_cache_bytes=3,
                full_preview_loading={"old"}, full_preview_active_path="old",
                full_preview_window_paths=["old"],
                background_tasks=scheduler,
                tree=SimpleNamespace(
                    get_children=MagicMock(return_value=("old",)), delete=MagicMock(),
                ),
                progress={"value": 99}, status=SimpleNamespace(set=MagicMock()),
                analysis_mode=SimpleNamespace(get=lambda: "标准两阶段"),
                filtered_result_indices=[0],
                filter_summary=SimpleNamespace(set=MagicMock()),
                analysis_generation=4, analyze_button=SimpleNamespace(configure=MagicMock()),
            )
            with patch(
                "meteor_screening.capture_sort_key",
                side_effect=AssertionError("metadata read leaked onto Tk caller"),
            ):
                MeteorScreeningWindow.analyze(window)
        self.assertTrue(window.analysis_running)
        self.assertEqual(window.files, [])
        scheduler.submit.assert_called_once()
        self.assertEqual(scheduler.submit.call_args.args[0], "analysis")

    def test_discovery_collapses_same_stem_raw_jpeg_and_keeps_raw_as_original(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for name in (
                "DSC0001.ARW", "DSC0001.JPG", "DSC0002.NEF",
                "._DSC0001.ARW", "note.txt",
            ):
                (source / name).touch()
            discovered = sorted(discover_screening_sources(source), key=lambda item: item.path.name)
        self.assertEqual(len(discovered), 2)
        self.assertEqual(discovered[0].path.name, "DSC0001.ARW")
        self.assertEqual(discovered[0].proxy_path.name, "DSC0001.JPG")
        self.assertEqual(discovered[1].path.name, "DSC0002.NEF")
        self.assertEqual(discovered[1].proxy_path.name, "DSC0002.NEF")

    def test_standard_mode_refines_candidates_and_temporal_neighbors_from_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for index in range(7):
                (source / f"frame_{index}.arw").touch()
            work_queue = queue.Queue()
            window = SimpleNamespace(work_queue=work_queue, preview_cache_budget=32 << 20)
            fast_calls = []
            precise_calls = []

            def image_for(path: Path):
                index = int(path.stem.rsplit("_", 1)[1])
                return np.full((72, 108, 3), 12 + index, dtype=np.uint8)

            def fast_preview(
                path: Path, _max_dimension: int = 1400, fallback_path: Path | None = None,
            ):
                fast_calls.append(path.name)
                return image_for(path)

            def precise_preview(path: Path, _max_dimension: int = 1400):
                precise_calls.append(path.name)
                return image_for(path)

            def detect(current, _reference, **_kwargs):
                # The fourth first-pass frame is the sole candidate. Its
                # precise review must load itself and all ±3 temporal frames.
                if int(round(float(np.mean(current)))) == 15:
                    return [((12, 36), (82, 34), 80)], 0
                return [], 0

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_fast_preview", side_effect=fast_preview),
                patch("meteor_screening.screening_preview", side_effect=precise_preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=detect),
            ):
                MeteorScreeningWindow._analyze_worker(
                    window, source, 1, str(source), CancellationToken("analysis", 1),
                    "标准两阶段",
                )
        messages = []
        while not work_queue.empty():
            messages.append(work_queue.get_nowait())
        self.assertFalse(any(item[0] == "analysis_error" for item in messages))
        finished = next(item for item in messages if item[0] == "analysis_finished")
        self.assertEqual(len(fast_calls), 7)
        self.assertEqual(sorted(precise_calls), [f"frame_{index}.arw" for index in range(7)])
        self.assertEqual(finished[6]["refined_count"], 1)

    def test_transient_opencv_frame_failure_is_retried_after_parallel_wave(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for index in range(3):
                (source / f"frame_{index}.jpg").touch()
            work_queue = queue.Queue()
            window = SimpleNamespace(work_queue=work_queue, preview_cache_budget=32 << 20)
            attempts = {13: 0}

            def preview(path: Path, _max_dimension: int = 1400, **_kwargs):
                index = int(path.stem.rsplit("_", 1)[1])
                return np.full((60, 90, 3), 12 + index, dtype=np.uint8)

            def detect(current, _reference, **_kwargs):
                marker = int(round(float(np.mean(current))))
                if marker == 13:
                    attempts[13] += 1
                    if attempts[13] == 1:
                        raise cv2.error("synthetic transient C++ failure")
                return [], 0

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=detect),
            ):
                MeteorScreeningWindow._analyze_worker(
                    window, source, 1, str(source), CancellationToken("analysis", 1),
                    "快速预筛",
                )
        messages = []
        while not work_queue.empty():
            messages.append(work_queue.get_nowait())
        self.assertFalse(any(item[0] == "analysis_error" for item in messages))
        finished = next(item for item in messages if item[0] == "analysis_finished")
        self.assertEqual(len(finished[3]), 3)
        self.assertEqual(attempts[13], 2)
        self.assertEqual(finished[6]["analysis_errors"], [])

    def test_persistent_opencv_frame_failure_is_kept_for_manual_review(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for index in range(3):
                (source / f"frame_{index}.jpg").touch()
            work_queue = queue.Queue()
            window = SimpleNamespace(work_queue=work_queue, preview_cache_budget=32 << 20)

            def preview(path: Path, _max_dimension: int = 1400, **_kwargs):
                index = int(path.stem.rsplit("_", 1)[1])
                return np.full((60, 90, 3), 12 + index, dtype=np.uint8)

            def detect(current, _reference, **_kwargs):
                if int(round(float(np.mean(current)))) == 13:
                    raise cv2.error("synthetic persistent C++ failure")
                return [], 0

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=detect),
            ):
                MeteorScreeningWindow._analyze_worker(
                    window, source, 1, str(source), CancellationToken("analysis", 1),
                    "快速预筛",
                )
        messages = []
        while not work_queue.empty():
            messages.append(work_queue.get_nowait())
        self.assertFalse(any(item[0] == "analysis_error" for item in messages))
        finished = next(item for item in messages if item[0] == "analysis_finished")
        failed = next(result for result in finished[3] if result.path.endswith("frame_1.jpg"))
        self.assertEqual(failed.score, 100)
        self.assertIn("必须人工复核", failed.note)
        self.assertEqual(len(finished[6]["analysis_errors"]), 1)

    def test_analysis_checkpoint_resumes_batches_and_invalidates_on_source_change(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for index in range(8):
                (source / f"frame_{index}.jpg").touch()
            checkpoint_path = source / "checkpoint.json"

            def preview(path: Path, _max_dimension: int = 1400, **_kwargs):
                index = int(path.stem.rsplit("_", 1)[1])
                return np.full((60, 90, 3), 12 + index, dtype=np.uint8)

            first_queue = queue.Queue()
            first_token = CancellationToken("analysis", 1)

            def stop_after_first_batch(path: Path, payload: dict):
                write_screening_checkpoint(path, payload)
                if payload["stage"] == "first_pass" and len(payload["results"]) == 4:
                    first_token.cancel()

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_worker_counts", return_value=(4, 1)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", return_value=([], 0)),
                patch("meteor_screening.write_screening_checkpoint", side_effect=stop_after_first_batch),
            ):
                MeteorScreeningWindow._analyze_worker(
                    SimpleNamespace(work_queue=first_queue, preview_cache_budget=32 << 20),
                    source, 1, str(source), first_token, "快速预筛", checkpoint_path,
                )
            saved = load_screening_checkpoint(checkpoint_path)
            self.assertIsNotNone(saved)
            self.assertEqual(len(saved["results"]), 4)

            resumed_calls = 0

            def resumed_detect(_current, _reference, **_kwargs):
                nonlocal resumed_calls
                resumed_calls += 1
                return [], 0

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_worker_counts", return_value=(4, 1)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=resumed_detect),
            ):
                MeteorScreeningWindow._analyze_worker(
                    SimpleNamespace(work_queue=queue.Queue(), preview_cache_budget=32 << 20),
                    source, 2, str(source), CancellationToken("analysis", 2),
                    "快速预筛", checkpoint_path,
                )
            self.assertEqual(resumed_calls, 4)
            self.assertEqual(load_screening_checkpoint(checkpoint_path)["stage"], "completed")

            (source / "frame_0.jpg").write_bytes(b"changed")
            invalidated_calls = 0

            def invalidated_detect(_current, _reference, **_kwargs):
                nonlocal invalidated_calls
                invalidated_calls += 1
                return [], 0

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_worker_counts", return_value=(4, 1)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=invalidated_detect),
            ):
                MeteorScreeningWindow._analyze_worker(
                    SimpleNamespace(work_queue=queue.Queue(), preview_cache_budget=32 << 20),
                    source, 3, str(source), CancellationToken("analysis", 3),
                    "快速预筛", checkpoint_path,
                )
            self.assertEqual(invalidated_calls, 8)

    def test_workspace_reopen_automatically_starts_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            checkpoint_path = source / "checkpoint.json"
            write_screening_checkpoint(checkpoint_path, {
                "format": "meteor-screening-checkpoint-v1",
                "algorithm_version": SCREENING_ALGORITHM_VERSION,
                "source_dir": os.path.normcase(os.path.abspath(source)),
                "analysis_mode": "标准两阶段",
                "stage": "first_pass",
            })
            window = SimpleNamespace(
                analysis_running=False,
                source_dir=SimpleNamespace(get=lambda: str(source)),
                analysis_mode=SimpleNamespace(get=lambda: "标准两阶段"),
                status=SimpleNamespace(set=MagicMock()),
                analyze=MagicMock(),
            )
            with patch(
                "meteor_screening.screening_analysis_checkpoint_file_path",
                return_value=checkpoint_path,
            ):
                MeteorScreeningWindow._resume_analysis_checkpoint(window)
        window.analyze.assert_called_once_with()
        window.status.set.assert_called_once_with("检测到未完成的分析断点，正在自动继续…")

    def test_standard_checkpoint_resumes_only_unfinished_refinement(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for index in range(8):
                (source / f"frame_{index}.jpg").touch()
            checkpoint_path = source / "checkpoint.json"

            def preview(path: Path, _max_dimension: int = 1400, **_kwargs):
                index = int(path.stem.rsplit("_", 1)[1])
                return np.full((72, 108, 3), 12 + index, dtype=np.uint8)

            def candidate(_current, _reference, **_kwargs):
                return [((12, 36), (82, 34), 80)], 0

            first_token = CancellationToken("analysis", 1)

            def stop_after_refine_group(path: Path, payload: dict):
                write_screening_checkpoint(path, payload)
                if payload["stage"] == "refine" and len(payload["refined_paths"]) == 4:
                    first_token.cancel()

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_worker_counts", return_value=(4, 1)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_screening.screening_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=candidate),
                patch("meteor_screening.write_screening_checkpoint", side_effect=stop_after_refine_group),
            ):
                MeteorScreeningWindow._analyze_worker(
                    SimpleNamespace(work_queue=queue.Queue(), preview_cache_budget=32 << 20),
                    source, 1, str(source), first_token, "标准两阶段", checkpoint_path,
                )
            saved = load_screening_checkpoint(checkpoint_path)
            self.assertEqual(saved["stage"], "refine")
            self.assertEqual(len(saved["refined_paths"]), 4)

            resumed_calls = 0

            def resumed_candidate(_current, _reference, **_kwargs):
                nonlocal resumed_calls
                resumed_calls += 1
                return [((12, 36), (82, 34), 80)], 0

            with (
                patch("meteor_screening.capture_sort_key", side_effect=lambda path: ("", path.name)),
                patch("meteor_screening.screening_worker_counts", return_value=(4, 1)),
                patch("meteor_screening.screening_fast_preview", side_effect=preview),
                patch("meteor_screening.screening_preview", side_effect=preview),
                patch("meteor_composer.load_meteor_ranker", return_value=None),
                patch("meteor_composer.detect_trails", side_effect=resumed_candidate),
            ):
                MeteorScreeningWindow._analyze_worker(
                    SimpleNamespace(work_queue=queue.Queue(), preview_cache_budget=32 << 20),
                    source, 2, str(source), CancellationToken("analysis", 2),
                    "标准两阶段", checkpoint_path,
                )
            self.assertEqual(resumed_calls, 4)
            completed = load_screening_checkpoint(checkpoint_path)
            self.assertEqual(completed["stage"], "completed")
            self.assertEqual(len(completed["refined_paths"]), 8)

    def test_screening_cache_budgets_are_bounded_and_adaptive(self):
        low_proxy, low_full = screening_memory_budgets(
            available_memory=3 << 30, total_memory=8 << 30,
        )
        high_proxy, high_full = screening_memory_budgets(
            available_memory=24 << 30, total_memory=32 << 30,
        )
        self.assertGreaterEqual(low_proxy, 64 << 20)
        self.assertLessEqual(high_proxy, 512 << 20)
        self.assertGreaterEqual(low_full, 384 << 20)
        self.assertGreater(high_full, low_full)
        self.assertLessEqual(high_full, 16 << 30)

    def test_screening_workers_scale_with_cpu_but_remain_memory_bounded(self):
        self.assertEqual(
            screening_worker_counts(
                100, logical_cpus=20, physical_cpus=14,
                available_memory=8 << 30,
            ),
            (10, 3),
        )
        self.assertEqual(
            screening_worker_counts(
                100, logical_cpus=20, physical_cpus=14,
                available_memory=1 << 30,
            ),
            (2, 1),
        )
        self.assertEqual(
            screening_worker_counts(
                3, logical_cpus=20, physical_cpus=14,
                available_memory=8 << 30,
            ),
            (3, 1),
        )
        self.assertEqual(
            screening_worker_counts(
                100, logical_cpus=32, physical_cpus=24,
                available_memory=32 << 30,
            ),
            (16, 4),
        )
        self.assertEqual(
            screening_worker_counts(
                100, logical_cpus=4, physical_cpus=2,
                available_memory=8 << 30,
            ),
            (1, 1),
        )

    def test_proxy_cache_is_trimmed_by_bytes(self):
        cache = OrderedDict(
            (str(index), np.zeros((100, 100, 3), dtype=np.uint8))
            for index in range(4)
        )
        total = trim_array_cache(cache, 65_000)
        self.assertLessEqual(total, 65_000)
        self.assertEqual(list(cache), ["2", "3"])

    def test_preview_window_is_current_first_and_biases_next_item(self):
        paths = [f"frame_{index}" for index in range(7)]
        self.assertEqual(
            preview_window(paths, "frame_3", 2),
            ["frame_3", "frame_4", "frame_5", "frame_6", "frame_2"],
        )

    def test_uncached_selection_keeps_last_frame_without_decoding_on_tk_handler(self):
        result = ScreeningResult("frame.arw", [], 0)
        retained = np.ones((2, 2, 3), np.uint8)
        retained_photo = object()
        window = SimpleNamespace(
            _selected_result=lambda: result,
            preview_path=None, preview_zoom=2.0, preview_pan_x=3.0, preview_pan_y=4.0,
            manual_mark_mode=False, manual_mark_start=None,
            manual_mark_label=SimpleNamespace(set=MagicMock()),
            active_candidates={}, preview_cache=OrderedDict(),
            _ensure_proxy_previews=MagicMock(),
            preview_render_rgb=retained, preview_photo=retained_photo,
            tree=SimpleNamespace(selection=MagicMock(return_value=())),
            canvas=SimpleNamespace(delete=MagicMock()), status=SimpleNamespace(set=MagicMock()),
        )
        with patch(
            "meteor_screening.screening_preview",
            side_effect=AssertionError("RAW decode leaked onto Tk handler"),
        ):
            MeteorScreeningWindow._show_selected(window)
        window._ensure_proxy_previews.assert_called_once_with()
        window.canvas.delete.assert_not_called()
        self.assertIs(window.preview_render_rgb, retained)
        self.assertIs(window.preview_photo, retained_photo)
        self.assertIn("暂时保留上一张画面", window.status.set.call_args.args[0])

    def test_original_preview_prefetches_adaptive_navigation_window(self):
        paths = [Path(f"frame_{index}.arw") for index in range(11)]
        work_queue = queue.Queue()

        class ImmediateScheduler:
            def submit(self, channel, worker, **_kwargs):
                self.channel = channel
                worker(CancellationToken(channel, 1))

        scheduler = ImmediateScheduler()
        window = SimpleNamespace(
            preview_path=str(paths[5]), files=paths,
            full_preview_cache=OrderedDict(), full_preview_loading=set(),
            full_preview_cache_budget=1536 << 20,
            full_preview_active_path=None, full_preview_window_paths=[],
            status=SimpleNamespace(set=MagicMock()), analysis_generation=2,
            source_dir=SimpleNamespace(get=lambda: "folder"),
            work_queue=work_queue, background_tasks=scheduler,
            _visible_preview_paths=lambda: [str(path) for path in paths],
        )
        with patch(
            "meteor_screening.read_screening_image",
            return_value=np.zeros((8, 12, 3), np.uint8),
        ):
            MeteorScreeningWindow._ensure_original_preview(window)
        queued = []
        while not work_queue.empty():
            queued.append(work_queue.get_nowait()[3])
        self.assertEqual(scheduler.channel, "full_preview")
        self.assertEqual(
            queued,
            [str(paths[index]) for index in (5, 6, 7, 8, 9, 10, 4, 3, 2, 1, 0)],
        )

    def test_raw_proxy_prefetch_window_expands_to_seventeen_frames(self):
        paths = [Path(f"frame_{index}.arw") for index in range(21)]
        work_queue = queue.Queue()

        class ImmediateScheduler:
            def submit(self, channel, worker, **_kwargs):
                worker(CancellationToken(channel, 1))

        window = SimpleNamespace(
            preview_path=str(paths[10]), files=paths,
            preview_cache=OrderedDict(), proxy_preview_loading=set(),
            proxy_preview_active_path=None, analysis_generation=2,
            source_dir=SimpleNamespace(get=lambda: "folder"),
            work_queue=work_queue, background_tasks=ImmediateScheduler(),
            _visible_preview_paths=lambda: [str(path) for path in paths],
        )
        with patch(
            "meteor_screening.screening_fast_preview",
            return_value=np.zeros((8, 12, 3), np.uint8),
        ):
            MeteorScreeningWindow._ensure_proxy_previews(window)
        queued = []
        while not work_queue.empty():
            queued.append(work_queue.get_nowait()[3])
        self.assertEqual(len(queued), 17)
        self.assertEqual(queued[:5], [str(paths[index]) for index in (10, 11, 12, 13, 14)])

    def test_preview_prefetch_only_reads_filtered_visible_photos(self):
        paths = [Path(f"frame_{index}.arw") for index in range(8)]
        visible = [str(paths[index]) for index in (1, 3, 5, 7)]
        work_queue = queue.Queue()

        class ImmediateScheduler:
            def submit(self, _channel, worker, **_kwargs):
                worker(CancellationToken("proxy_preview", 1))

        window = SimpleNamespace(
            preview_path=str(paths[3]), files=paths,
            preview_cache=OrderedDict(), proxy_preview_loading=set(),
            proxy_preview_active_path=None, analysis_generation=2,
            source_dir=SimpleNamespace(get=lambda: "folder"),
            work_queue=work_queue, background_tasks=ImmediateScheduler(),
            _visible_preview_paths=lambda: visible,
        )
        with patch(
            "meteor_screening.screening_fast_preview",
            return_value=np.zeros((8, 12, 3), np.uint8),
        ):
            MeteorScreeningWindow._ensure_proxy_previews(window)
        queued = []
        while not work_queue.empty():
            queued.append(work_queue.get_nowait()[3])
        self.assertEqual(queued, [visible[index] for index in (1, 2, 3, 0)])

    def test_name_status_score_and_candidate_label_filters_combine(self):
        variables = {
            "filter_name": "080",
            "filter_status": "人工保留",
            "filter_label": "已确认流星",
            "filter_score_min": 70,
            "filter_score_max": 90,
            "sensitivity": 42,
        }
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        for name, value in variables.items():
            setattr(window, name, SimpleNamespace(get=lambda value=value: value))
        window.decisions = {"D:/shoot/DSC08083.ARW": "accept"}
        window.decision_sources = {"D:/shoot/DSC08083.ARW": "manual"}
        matching = ScreeningResult(
            "D:/shoot/DSC08083.ARW",
            [ScreeningCandidate((1, 2), (20, 10), 80, label="meteor")],
            84,
        )
        self.assertTrue(window._result_matches_filters(matching))
        self.assertFalse(window._result_matches_filters(ScreeningResult(
            "D:/shoot/DSC08033.ARW", matching.candidates, 84,
        )))
        self.assertFalse(window._result_matches_filters(ScreeningResult(
            matching.path,
            [ScreeningCandidate((1, 2), (20, 10), 80, label="not_meteor")],
            84,
        )))
        self.assertFalse(window._result_matches_filters(ScreeningResult(
            matching.path, matching.candidates, 55,
        )))

    def test_filter_controls_and_keyboard_navigation_use_visible_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"Tk display unavailable: {exc}")
            root.withdraw()
            autosave = Path(folder) / "missing-autosave.json"
            with patch("meteor_screening.screening_autosave_file_path", return_value=autosave):
                window = MeteorScreeningWindow(root)
            try:
                window.results = [
                    ScreeningResult("D:/shoot/DSC08031.ARW", [], 20),
                    ScreeningResult("D:/shoot/DSC08032.ARW", [], 72),
                    ScreeningResult("D:/shoot/DSC08033.ARW", [], 88),
                    ScreeningResult("D:/shoot/DSC09001.ARW", [], 95),
                ]
                window.files = [Path(result.path) for result in window.results]
                window._show_selected = MagicMock()
                window.filter_name_entry.event_generate("<Button-1>", x=4, y=4)
                window.filter_name_entry.insert(0, "080")
                window.filter_score_min.set(70)
                window.update()
                debounce_finished = tk.BooleanVar(master=window, value=False)
                window.after(180, lambda: debounce_finished.set(True))
                window.wait_variable(debounce_finished)
                self.assertEqual(window.tree.get_children(), ("1", "2"))
                self.assertEqual(window.filter_summary.get(), "显示 2/4")
                window.tree.selection_set("1")
                window.tree.focus("1")
                window.tree.focus_force()
                window.tree.event_generate("<Down>")
                window.update()
                self.assertEqual(window.tree.selection(), ("2",))
            finally:
                for after_id in window.tk.call("after", "info"):
                    window.after_cancel(after_id)
                window.destroy()
                root.destroy()

    def test_real_tk_candidate_marks_start_visible_even_if_old_autosave_hid_them(self):
        with tempfile.TemporaryDirectory() as folder:
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"Tk display unavailable: {exc}")
            root.withdraw()
            autosave = Path(folder) / "screening-autosave.json"
            autosave.write_text(json.dumps({
                "format": "meteor-screening-autosave-v1",
                "show_candidate_marks": False,
                "results": [],
            }), encoding="utf-8")
            with patch("meteor_screening.screening_autosave_file_path", return_value=autosave):
                window = MeteorScreeningWindow(root)
            try:
                window.update()
                self.assertTrue(window.show_candidate_marks.get())
            finally:
                for after_id in window.tk.call("after", "info"):
                    window.after_cancel(after_id)
                window.destroy()
                root.destroy()

    def test_real_tk_rapid_uncached_navigation_never_clears_canvas(self):
        with tempfile.TemporaryDirectory() as folder:
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"Tk display unavailable: {exc}")
            root.withdraw()
            autosave = Path(folder) / "missing-autosave.json"
            with patch("meteor_screening.screening_autosave_file_path", return_value=autosave):
                window = MeteorScreeningWindow(root)
            try:
                first = "D:/shoot/frame_001.ARW"
                second = "D:/shoot/frame_002.ARW"
                window.results = [
                    ScreeningResult(first, [], 65),
                    ScreeningResult(second, [], 70),
                ]
                window.files = [Path(first), Path(second)]
                window.preview_cache[first] = np.full((80, 120, 3), 32, np.uint8)
                window.preview_cache_bytes = window.preview_cache[first].nbytes
                window.background_tasks.submit = MagicMock()
                window._apply_filters(False)
                window._show_selected()
                window.update()
                retained_render = window.preview_render_rgb
                retained_photo = window.preview_photo
                retained_items = window.canvas.find_all()
                self.assertIsNotNone(retained_render)
                self.assertTrue(retained_items)

                window.tree.focus_force()
                window.tree.event_generate("<Down>")
                window.update()
                delayed_events_done = tk.BooleanVar(master=window, value=False)
                window.after(180, lambda: delayed_events_done.set(True))
                window.wait_variable(delayed_events_done)

                self.assertEqual(window.tree.selection(), ("1",))
                self.assertIs(window.preview_render_rgb, retained_render)
                self.assertIs(window.preview_photo, retained_photo)
                self.assertEqual(window.canvas.find_all(), retained_items)
                self.assertIn("暂时保留上一张画面", window.status.get())
            finally:
                for after_id in window.tk.call("after", "info"):
                    window.after_cancel(after_id)
                window.destroy()
                root.destroy()

    def test_candidate_multi_remove_recalculates_score_and_candidate_decision(self):
        path = "frame.arw"
        result = ScreeningResult(path, [
            ScreeningCandidate((1, 1), (20, 1), 91, label="meteor"),
            ScreeningCandidate((1, 5), (20, 5), 63, label="not_meteor"),
            ScreeningCandidate((1, 9), (20, 9), 77),
        ], 91)
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        window.selected_candidates = {path: {0, 2}}
        window.active_candidates = {path: 2}
        window.decisions = {path: "accept"}
        window.decision_sources = {path: "candidate"}
        removed = window._remove_candidate_indices(result, {0, 2})
        self.assertEqual(removed, 2)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].score, 63)
        self.assertEqual(result.score, 63)
        self.assertEqual(window.decisions[path], "reject")
        self.assertEqual(window.decision_sources[path], "candidate")
        self.assertEqual(window.active_candidates[path], 0)
        self.assertNotIn(path, window.selected_candidates)

    def test_remove_all_candidates_clears_candidate_decision_and_score(self):
        path = "frame.arw"
        result = ScreeningResult(
            path, [ScreeningCandidate((1, 1), (20, 1), 91, label="meteor")], 91,
        )
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        window.selected_candidates = {path: {0}}
        window.active_candidates = {path: 0}
        window.decisions = {path: "accept"}
        window.decision_sources = {path: "candidate"}
        self.assertEqual(window._remove_candidate_indices(result, {0}), 1)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.score, 0)
        self.assertNotIn(path, window.decisions)
        self.assertNotIn(path, window.decision_sources)

    def test_real_tk_canvas_multi_select_and_clear_selected_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            try:
                root = tk.Tk()
            except tk.TclError as exc:
                self.skipTest(f"Tk display unavailable: {exc}")
            root.withdraw()
            autosave = Path(folder) / "missing-autosave.json"
            with patch("meteor_screening.screening_autosave_file_path", return_value=autosave):
                window = MeteorScreeningWindow(root)
            try:
                path = "D:/shoot/frame_multi.ARW"
                result = ScreeningResult(path, [
                    ScreeningCandidate((30, 40), (100, 40), 82),
                    ScreeningCandidate((180, 100), (260, 100), 76),
                ], 82)
                window.results = [result]
                window.files = [Path(path)]
                window.preview_cache[path] = np.full((150, 300, 3), 30, np.uint8)
                window.preview_cache_bytes = window.preview_cache[path].nbytes
                window.background_tasks.submit = MagicMock()
                window._apply_filters(False)
                window._show_selected()
                window.update()
                window.candidate_multi_select_mode.set(True)

                def click_image(x: float, y: float) -> None:
                    scale, left, top, _cw, _ch = window._view_geometry()
                    canvas_x = round(left + x * scale)
                    canvas_y = round(top + y * scale)
                    window.canvas.event_generate(
                        "<ButtonPress-1>", x=canvas_x, y=canvas_y,
                    )
                    window.canvas.event_generate(
                        "<ButtonRelease-1>", x=canvas_x, y=canvas_y,
                    )
                    window.update()

                click_image(65, 40)
                click_image(220, 100)
                self.assertEqual(window.selected_candidates[path], {0, 1})
                self.assertIn("已选择 2 条", window.candidate_status.get())
                click_image(220, 100)
                self.assertEqual(window.selected_candidates[path], {0})
                window.select_all_candidates_button.invoke()
                window.update()
                self.assertEqual(window.selected_candidates[path], {0, 1})
                window.remove_selected_candidates_button.invoke()
                window.update()
                self.assertEqual(result.candidates, [])
                self.assertEqual(result.score, 0)
                self.assertEqual(window.tree.item("0", "values")[2], "0")
            finally:
                for after_id in window.tk.call("after", "info"):
                    window.after_cancel(after_id)
                window.destroy()
                root.destroy()

    def test_full_preview_cache_keeps_current_when_window_exceeds_budget(self):
        window = SimpleNamespace(
            full_preview_cache=OrderedDict(), full_preview_cache_bytes=0,
            full_preview_cache_budget=9,
            full_preview_window_paths=["2", "3", "1", "4", "0"],
            preview_path="2",
        )
        image = np.zeros((1, 1, 3), np.uint8)
        for path in window.full_preview_window_paths:
            MeteorScreeningWindow._store_full_preview(window, path, image.copy())
        self.assertIn("2", window.full_preview_cache)
        self.assertLessEqual(window.full_preview_cache_bytes, 9)

    def test_common_sony_nikon_canon_raw_formats_are_accepted(self):
        expected = {".arw", ".nef", ".nrw", ".cr2", ".cr3", ".crw"}
        self.assertTrue(expected.issubset(RAW_SUFFIXES))
        self.assertTrue(expected.issubset(IMAGE_SUFFIXES))

    def test_neighbor_median_excludes_current_frame_meteor(self):
        base = np.full((180, 280, 3), 12, dtype=np.uint8)
        for x, y in ((30, 25), (80, 42), (140, 65), (210, 38), (245, 92), (55, 115)):
            cv2.circle(base, (x, y), 2, (190, 190, 190), -1, cv2.LINE_AA)
        current = base.copy()
        cv2.line(current, (48, 130), (230, 82), (245, 248, 255), 4, cv2.LINE_AA)
        reference, displacement = temporal_reference(current, [base.copy() for _ in range(4)])
        self.assertLess(displacement, 1.2)
        self.assertLess(int(reference[108, 135].max()), 40)
        self.assertGreater(int(current[108, 135].max()), 180)

    def test_dynamic_star_sky_mask_excludes_high_uneven_landscape(self):
        rng = np.random.default_rng(8)
        height, width = 360, 560
        reference = np.full((height, width, 3), 12, dtype=np.uint8)
        horizon = np.asarray([
            220 + 42 * np.sin(x / 70.0) + (32 if 215 < x < 320 else 0)
            for x in range(width)
        ], dtype=int)
        for x, bottom in enumerate(horizon):
            texture = rng.integers(0, 55, size=(height - bottom, 3), dtype=np.uint8)
            reference[bottom:, x] = 34 + texture
        for _ in range(520):
            x = int(rng.integers(5, width - 5))
            y = int(rng.integers(5, max(6, horizon[x] - 5)))
            value = int(rng.integers(145, 245))
            cv2.circle(reference, (x, y), 1, (value, value, value), -1, cv2.LINE_AA)
        current = reference.copy()
        cv2.line(current, (70, 90), (230, 125), (240, 245, 255), 3, cv2.LINE_AA)
        cv2.line(current, (340, 300), (520, 325), (245, 245, 245), 3, cv2.LINE_AA)
        mask = estimate_star_sky_mask(current, reference)
        self.assertGreater(float(np.mean(mask[60:150] > 0)), 0.92)
        self.assertLess(float(np.mean(mask[310:350] > 0)), 0.10)
        trails, _ = detect_trails(current, reference, ranked=True, valid_region=mask)
        self.assertTrue(any((start[1] + end[1]) / 2 < 180 for start, end, _score in trails))
        self.assertFalse(any((start[1] + end[1]) / 2 > 280 for start, end, _score in trails))

    def test_current_frame_cloud_texture_cannot_become_false_horizon(self):
        rng = np.random.default_rng(6006)
        height, width = 360, 560
        reference = np.full((height, width, 3), 42, dtype=np.uint8)
        for _ in range(700):
            x = int(rng.integers(4, width - 4))
            y = int(rng.integers(4, 292))
            value = int(rng.integers(125, 235))
            cv2.circle(reference, (x, y), 1, (value, value, value), -1, cv2.LINE_AA)
        reference[305:] = 9
        current = reference.copy()
        cloud = rng.integers(0, 95, size=(155, width, 1), dtype=np.uint8)
        current[95:250] = np.clip(
            current[95:250].astype(np.int16) + cloud.astype(np.int16), 0, 255,
        ).astype(np.uint8)
        mask = estimate_star_sky_mask(current, reference)
        self.assertGreater(float(np.mean(mask > 0)), 0.70)
        self.assertLess(float(np.mean(mask[325:] > 0)), 0.05)

    def test_temporal_repeat_marks_probable_aircraft_sequence(self):
        results = []
        for index in range(3):
            candidate = ScreeningCandidate(
                (20 + index * 25, 60), (100 + index * 25, 70), 82
            )
            results.append(ScreeningResult(f"frame_{index}.tif", [candidate], 82))
        mark_temporal_repeats(results, (180, 300))
        self.assertGreaterEqual(results[1].temporal_hits, 2)
        self.assertLess(results[1].score, 82)
        self.assertIn("飞机/卫星", results[1].note)

    def test_7246_style_weak_moving_track_is_not_kept_as_meteor_candidate(self):
        results = []
        tracks = (
            ((402, 113), (429, 101), 36),
            ((435, 99), (462, 89), 38),
            ((469, 87), (488, 79), 18),
        )
        for index, (start, end, score) in enumerate(tracks):
            candidate = ScreeningCandidate(start, end, score)
            results.append(ScreeningResult(f"DSC0724{4 + index}.ARW", [candidate], score))
        mark_temporal_repeats(results, (935, 1400))
        self.assertEqual([result.candidates for result in results], [[], [], []])
        self.assertEqual([result.score for result in results], [0, 0, 0])
        self.assertTrue(all("自动排除" in result.note for result in results))

    def test_old_single_candidate_checkpoint_gets_same_repeat_cleanup(self):
        candidate = ScreeningCandidate((469, 87), (488, 79), 18)
        result = ScreeningResult(
            "DSC07246.ARW", [candidate], 0, temporal_hits=2,
            note="疑似连续飞机/卫星，请复查",
        )
        removed = suppress_repeating_low_confidence_tracks(result)
        self.assertEqual(removed, 1)
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.score, 0)

    def test_repeat_cleanup_preserves_high_confidence_and_explicit_labels(self):
        candidates = [
            ScreeningCandidate((10, 10), (80, 20), 72),
            ScreeningCandidate((20, 30), (90, 40), 18, label="meteor"),
            ScreeningCandidate((30, 50), (100, 60), 12, manual=True),
        ]
        result = ScreeningResult("protected.ARW", candidates.copy(), 72, temporal_hits=6)
        removed = suppress_repeating_low_confidence_tracks(result, [3, 3, 3])
        self.assertEqual(removed, 0)
        self.assertEqual(result.candidates, candidates)

    def test_dense_temporal_cloud_edges_are_suppressed_but_positive_meteor_remains(self):
        cloud_features = [0.0] * 34
        cloud_features[29] = 0.48
        cloud_features[30] = 0.52
        cloud_features[31] = -2.0
        meteor_features = [0.0] * 34
        meteor_features[29] = 0.95
        meteor_features[30] = 0.05
        meteor_features[31] = 8.0
        candidates = [
            ScreeningCandidate(
                (20 + index * 15, 50), (60 + index * 15, 54), 32,
                features=cloud_features.copy(),
            )
            for index in range(7)
        ]
        meteor = ScreeningCandidate(
            (40, 110), (180, 85), 48, features=meteor_features,
        )
        candidates.append(meteor)
        result = ScreeningResult(
            "cloudy.arw", candidates, 48, temporal_hits=24,
        )
        removed = suppress_dense_temporal_clutter(
            result, [3, 4, 3, 4, 2, 3, 4, 2],
        )
        self.assertEqual(removed, 7)
        self.assertEqual(result.candidates, [meteor])
        self.assertEqual(result.score, 48)

    def test_candidate_guides_do_not_cover_meteor_center(self):
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        candidate = ScreeningCandidate((30, 60), (210, 60), 88)
        MeteorScreeningWindow._draw_candidate_marker(image, candidate, (255, 190, 45), True)
        self.assertTrue(np.all(image[60, 120] == 0))
        self.assertGreater(int(image[53, 120].max()), 0)

    def test_selected_candidate_has_strong_cyan_outline_and_check_badge(self):
        image = np.zeros((140, 260, 3), dtype=np.uint8)
        candidate = ScreeningCandidate((40, 70), (210, 70), 88)
        MeteorScreeningWindow._draw_candidate_marker(
            image, candidate, (255, 190, 45), True, selected=True,
        )
        cyan = np.all(image == np.asarray((70, 235, 255), np.uint8), axis=2)
        self.assertGreater(int(np.count_nonzero(cyan)), 100)

    def test_learning_uses_only_explicit_candidate_labels(self):
        feature_names = [f"f{index}" for index in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feedback.json"
            path.write_text(json.dumps([
                {"id": "exact", "source_path": "a.arw", "label": 1,
                 "features": [1, 2, 3, 4], "legacy": 0.8},
                {"id": "image-only", "source_path": "b.arw", "decision": "accept",
                 "features": [4, 3, 2, 1]},
            ]), encoding="utf-8")
            data = build_screening_feedback_dataset({
                "ML_FEATURE_NAMES": feature_names,
                "screening_feedback_path": path,
            })
        self.assertEqual(data["x"].shape, (1, 4))
        self.assertEqual(data["y"].tolist(), [1])

    def test_candidate_labels_drive_photo_decision_only_when_conclusive(self):
        window = MeteorScreeningWindow.__new__(MeteorScreeningWindow)
        window.decisions = {}
        window.decision_sources = {}
        first = ScreeningCandidate((10, 10), (80, 40), 70, label="not_meteor")
        second = ScreeningCandidate((20, 70), (100, 50), 65)
        result = ScreeningResult("frame.arw", [first, second], 70)
        window._sync_photo_decision_from_candidates(result)
        self.assertNotIn(result.path, window.decisions)
        second.label = "not_meteor"
        window._sync_photo_decision_from_candidates(result)
        self.assertEqual(window.decisions[result.path], "reject")
        self.assertEqual(window.decision_sources[result.path], "candidate")
        second.label = ""
        window._sync_photo_decision_from_candidates(result)
        self.assertNotIn(result.path, window.decisions)
        second.label = "meteor"
        window._sync_photo_decision_from_candidates(result)
        self.assertEqual(window.decisions[result.path], "accept")


if __name__ == "__main__":
    unittest.main()
