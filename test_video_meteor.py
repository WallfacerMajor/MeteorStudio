from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from video_meteor import (
    EventSettings,
    EventClip,
    ResidualLayer,
    VideoEvent,
    VideoStroke,
    analyze_video,
    build_residual_layer,
    effect_strength,
    frame_candidates,
    normalize_lsd_lines,
    prepare_layers,
    probe_video,
    render_video,
    sample_clip,
    build_stroke_mask,
)


DEFAULTS = EventSettings(
    effect_mode="慢放并淡出",
    local_speed=20.0,
    hold_seconds=0.24,
    fade_seconds=0.76,
    brightness=1.0,
    mask_width=16,
    mask_feather=8,
    curve="渐慢",
    curve_start=1.8,
    curve_mid=0.8,
    curve_end=0.25,
)


class VideoMeteorTests(unittest.TestCase):
    def test_lsd_output_shape_is_cross_platform(self) -> None:
        expected = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(
            normalize_lsd_lines(expected.reshape(2, 1, 4)), expected
        )
        np.testing.assert_array_equal(normalize_lsd_lines(expected), expected)
        np.testing.assert_array_equal(
            normalize_lsd_lines(expected[0]), expected[:1]
        )
        self.assertEqual(normalize_lsd_lines(None).shape, (0, 4))

    def test_line_detection_and_residual(self) -> None:
        background = np.zeros((180, 320, 3), dtype=np.uint8)
        target = background.copy()
        cv2.line(target, (70, 55), (250, 110), (210, 225, 245), 3, cv2.LINE_AA)
        candidates = frame_candidates(target, background, 1.0, 10.0)
        self.assertTrue(candidates)
        event = VideoEvent(4, 100, [candidates[0][1]], accepted=True)
        layer = build_residual_layer(event, target, background, DEFAULTS)
        self.assertIsNotNone(layer)
        self.assertGreater(float(np.max(layer.residual)), 20.0)

    def test_effect_curve(self) -> None:
        self.assertEqual(effect_strength(1, 25.0, DEFAULTS), 1.0)
        self.assertGreater(effect_strength(12, 25.0, DEFAULTS), 0.0)
        self.assertEqual(effect_strength(40, 25.0, DEFAULTS), 0.0)

    def test_paint_and_erase_mask(self) -> None:
        strokes = [
            VideoStroke([[0.1, 0.5], [0.9, 0.5]], 18, 0),
            VideoStroke([[0.5, 0.35], [0.5, 0.65]], 28, 0, erase=True),
        ]
        mask = build_stroke_mask(strokes, 200, 100, DEFAULTS)
        self.assertGreater(mask[50, 30], 0.8)
        self.assertLess(mask[50, 100], 0.1)

    def test_multiframe_clip_is_retimed(self) -> None:
        patch = np.ones((2, 2, 3), dtype=np.float32)
        first = ResidualLayer(10, 0, 0, 2, 2, patch, DEFAULTS, 0)
        second = ResidualLayer(11, 0, 0, 2, 2, patch * 2, DEFAULTS, 1)
        clip = EventClip(10, [first, second], DEFAULTS)
        self.assertIs(sample_clip(clip, 0, 25.0)[0][0], first)
        self.assertTrue(sample_clip(clip, 5, 25.0))
        self.assertTrue(any(layer is second for layer, _weight in sample_clip(clip, 9, 25.0)))

    def test_clean_video_analysis_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.avi"
            clean_path = root / "clean.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            source_writer = cv2.VideoWriter(str(source_path), fourcc, 25.0, (320, 180))
            clean_writer = cv2.VideoWriter(str(clean_path), fourcc, 25.0, (320, 180))
            rng = np.random.default_rng(7)
            stars = [(int(x), int(y)) for x, y in rng.integers([5, 5], [315, 150], size=(70, 2))]
            for index in range(20):
                clean = np.full((180, 320, 3), 20, dtype=np.uint8)
                for x, y in stars:
                    cv2.circle(clean, (x, y), 1, (145, 145, 145), -1)
                source = clean.copy()
                if index == 8:
                    cv2.line(source, (75, 55), (255, 115), (210, 225, 245), 3, cv2.LINE_AA)
                clean_writer.write(clean)
                source_writer.write(source)
            clean_writer.release()
            source_writer.release()

            info, events = analyze_video(source_path, clean_path, 10.0, lambda *_args: None)
            event = max(events, key=lambda item: item.score)
            self.assertEqual(event.frame, 8)
            event.accepted = True
            info, clips = prepare_layers(source_path, clean_path, [event], DEFAULTS, lambda *_args: None)
            self.assertEqual(clips[0].start_frame, 8)
            self.assertTrue(sample_clip(clips[0], 2, info.fps))
            output = render_video(source_path, root / "output.mp4", info, clips, 1.0, False, lambda *_args: None)
            exported = probe_video(output)
            self.assertEqual(exported.frames, 20)
            self.assertEqual((exported.width, exported.height), (320, 180))
            slowed = render_video(
                source_path, root / "background_75.mp4", info, clips, 0.75, False,
                lambda *_args: None, "平衡文件 CRF 18",
            )
            slowed_info = probe_video(slowed)
            self.assertEqual(slowed_info.frames, 26)
            self.assertGreater(slowed_info.duration, exported.duration)


if __name__ == "__main__":
    unittest.main()
