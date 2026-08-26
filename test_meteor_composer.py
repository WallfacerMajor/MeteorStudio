import unittest
import tempfile
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
import tifffile

from meteor_composer import (
    MeteorComposer, Stroke, content_distance_map, line_inside_valid_content,
    normalize_lsd_lines, point_in_padded_bbox, compose_meteor_objects,
    remove_local_background_cast, annotate_meteor_sources,
    meteor_source_annotations, stroke_is_transformed, reset_stroke_geometry,
    adjust_composite_base_exposure,
    analyze_meteor_blend_parameters,
)


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class NormalizeLsdLinesTests(unittest.TestCase):
    def test_accepts_opencv_nested_shape(self):
        lines = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.float32)
        result = normalize_lsd_lines(lines)
        self.assertEqual(result.shape, (2, 4))
        np.testing.assert_array_equal(result[1], [5, 6, 7, 8])

    def test_accepts_flat_line_shape(self):
        lines = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        result = normalize_lsd_lines(lines)
        self.assertEqual(result.shape, (2, 4))

    def test_accepts_single_flat_line(self):
        result = normalize_lsd_lines(np.array([1, 2, 3, 4], dtype=np.float32))
        self.assertEqual(result.shape, (1, 4))

    def test_none_is_empty(self):
        self.assertEqual(normalize_lsd_lines(None).shape, (0, 4))


class ProjectionBoundaryTests(unittest.TestCase):
    def test_rejects_line_on_black_projection_edge(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        image[20:180, 60:260] = 35
        distance = content_distance_map(image)
        self.assertFalse(line_inside_valid_content(distance, (61, 35), (61, 150)))

    def test_keeps_line_inside_real_image_content(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        image[20:180, 60:260] = 35
        distance = content_distance_map(image)
        self.assertTrue(line_inside_valid_content(distance, (120, 70), (210, 110)))


class ExposureScopeTests(unittest.TestCase):
    @staticmethod
    def fake(adjustments, global_default):
        return SimpleNamespace(
            adjustment_defaults={
                "match_exposure": global_default, "curve_enabled": False,
                "curve_shadows": 15, "curve_highlights": 25,
                "preserve_brightness": True, "meteor_brightness": 100,
                "background_cleanup": 70,
            },
            image_adjustments=adjustments,
            loading_adjustments=False,
            match_exposure=FakeVar(), match_exposure_policy=FakeVar(),
            curve_enabled=FakeVar(), curve_shadows=FakeVar(), curve_highlights=FakeVar(),
            default_preserve_brightness=FakeVar(), default_meteor_brightness=FakeVar(),
            default_background_cleanup=FakeVar(),
            brightness_override=FakeVar(), meteor_brightness=FakeVar(),
            current_brightness_scale=SimpleNamespace(configure=lambda **_kwargs: None),
        )

    def test_current_image_can_follow_global_default(self):
        fake = self.fake({"a.tif": {"curve_enabled": True}}, True)
        MeteorComposer._load_image_adjustments(fake, "a.tif")
        self.assertTrue(fake.match_exposure.get())
        self.assertEqual(fake.match_exposure_policy.get(), "跟随全局")

    def test_current_image_can_force_disable_against_global(self):
        fake = self.fake({"a.tif": {"match_exposure": False}}, True)
        MeteorComposer._load_image_adjustments(fake, "a.tif")
        self.assertFalse(fake.match_exposure.get())
        self.assertEqual(fake.match_exposure_policy.get(), "强制关闭")

    def test_current_image_can_force_enable_against_global(self):
        fake = self.fake({"a.tif": {"match_exposure": True}}, False)
        MeteorComposer._load_image_adjustments(fake, "a.tif")
        self.assertTrue(fake.match_exposure.get())
        self.assertEqual(fake.match_exposure_policy.get(), "强制启用")

    def test_current_image_brightness_can_override_global(self):
        fake = self.fake({"a.tif": {"meteor_brightness": 135}}, False)
        MeteorComposer._load_image_adjustments(fake, "a.tif")
        self.assertTrue(fake.brightness_override.get())
        self.assertEqual(fake.meteor_brightness.get(), 135)

    def test_current_image_brightness_follows_global_when_absent(self):
        fake = self.fake({"a.tif": {}}, False)
        fake.adjustment_defaults["meteor_brightness"] = 118
        MeteorComposer._load_image_adjustments(fake, "a.tif")
        self.assertFalse(fake.brightness_override.get())
        self.assertEqual(fake.meteor_brightness.get(), 118)


class CandidateButtonGeometryTests(unittest.TestCase):
    def test_button_hit_area(self):
        bbox = (100, 50, 200, 80)
        self.assertTrue(point_in_padded_bbox(150, 65, bbox))
        self.assertFalse(point_in_padded_bbox(90, 65, bbox))

    def test_padding_bridges_pointer_gap(self):
        bbox = (100, 50, 200, 80)
        self.assertTrue(point_in_padded_bbox(90, 65, bbox, padding=16))
        self.assertFalse(point_in_padded_bbox(80, 65, bbox, padding=16))

    def test_candidate_button_press_does_not_start_brush(self):
        picked = []
        fake = SimpleNamespace(
            current_path="image.tif",
            view_mode=SimpleNamespace(get=lambda: "source"),
            hover_candidate_items=[1, 2],
            canvas=SimpleNamespace(bbox=lambda _tag: (100, 50, 200, 80)),
            _pick_hover_candidate=lambda event: picked.append(event),
        )
        event = SimpleNamespace(x=150, y=65)
        result = MeteorComposer._stroke_start(fake, event)
        self.assertEqual(result, "break")
        self.assertEqual(picked, [event])

    def test_press_outside_button_starts_manual_brush(self):
        cleared = []
        fake_var = lambda value: SimpleNamespace(get=lambda: value)
        fake = SimpleNamespace(
            current_path="image.tif",
            view_mode=SimpleNamespace(get=lambda: "source"),
            hover_candidate_items=[1, 2],
            canvas=SimpleNamespace(bbox=lambda _tag: (100, 50, 200, 80)),
            _pick_hover_candidate=lambda _event: self.fail("candidate should not be picked"),
            _event_normalized=lambda _event: (0.25, 0.5),
            _clear_candidate_hover=lambda: cleared.append(True),
            shift_anchors={},
            active_shift_line=False,
            active_points=[],
            edit_mode=fake_var("paint"),
            _tool_width=lambda: 18,
            feather=fake_var(10),
            active_tool_mode=None,
            active_tool_width=0,
            active_tool_feather=0,
            active_canvas_line=None,
            cursor_position=None,
            strokes={},
            active_action_index=-1,
            live_erase_stroke=None,
            _refresh_live_mask=lambda force=False: None,
        )
        result = MeteorComposer._stroke_start(fake, SimpleNamespace(x=20, y=20, state=0))
        self.assertEqual(result, "break")
        self.assertEqual(fake.active_points, [(0.25, 0.5)])
        self.assertEqual(fake.active_tool_mode, "paint")
        self.assertEqual(fake.strokes["image.tif"], [])
        self.assertIsNone(fake.live_erase_stroke)
        self.assertEqual(cleared, [True])

    def test_blend_preview_routes_press_to_object_editor(self):
        events = []
        fake = SimpleNamespace(
            current_path="image.tif",
            view_mode=SimpleNamespace(get=lambda: "blend"),
            _object_pointer_start=lambda event: events.append(event) or "break",
        )
        event = SimpleNamespace(x=10, y=10, state=0)
        result = MeteorComposer._stroke_start(fake, event)
        self.assertEqual(result, "break")
        self.assertEqual(events, [event])

    def test_labeled_preview_routes_press_to_object_editor(self):
        events = []
        fake = SimpleNamespace(
            current_path="image.tif",
            view_mode=FakeVar("labeled"),
            _object_pointer_start=lambda event: events.append(event) or "break",
        )
        event = SimpleNamespace(x=10, y=10, state=0)
        result = MeteorComposer._stroke_start(fake, event)
        self.assertEqual(result, "break")
        self.assertEqual(events, [event])

    def test_preview_scope_follows_explicit_output_mode(self):
        shared = SimpleNamespace(output_mode=FakeVar("combined"))
        paired = SimpleNamespace(output_mode=FakeVar("separate"))
        self.assertTrue(MeteorComposer._uses_shared_base(shared))
        self.assertFalse(MeteorComposer._uses_shared_base(paired))

    def test_unlocked_candidates_are_drawn_and_locked_candidates_are_hidden(self):
        created = []
        canvas = SimpleNamespace(
            create_line=lambda *args, **kwargs: created.append(("line", args, kwargs)),
            create_oval=lambda *args, **kwargs: created.append(("oval", args, kwargs)),
            create_text=lambda *args, **kwargs: created.append(("text", args, kwargs)),
        )
        candidate = Stroke([(0.1, 0.2), (0.8, 0.7)], 20, 10, auto_score=46)
        locked = Stroke([(0.2, 0.3), (0.7, 0.6)], 20, 10, locked=True, auto_score=80)
        fake = SimpleNamespace(
            current_path="image.tif",
            display_box=(0, 0, 1000, 500),
            current_dims=(2000, 1000),
            candidates={"image.tif": [candidate, locked]},
            candidate_thresholds={"image.tif": 55},
            candidate_threshold=SimpleNamespace(get=lambda: 55),
            canvas=canvas,
        )
        MeteorComposer._draw_candidate_guides(fake)
        self.assertEqual([item[0] for item in created], ["line", "text"])
        self.assertEqual(created[1][2]["text"], "46")


class MeteorBrightnessTests(unittest.TestCase):
    def test_per_meteor_analyzer_derives_full_resolution_parameters(self):
        base = np.full((180, 320, 3), (7000, 8200, 9800), dtype=np.uint16)
        source = base.copy()
        source[55:125, 35:285] = np.clip(
            source[55:125, 35:285].astype(np.int32) + np.asarray((900, 180, 1100)),
            0, 65535,
        ).astype(np.uint16)
        cv2.line(source, (55, 105), (265, 75), (51000, 55000, 61000), 7, cv2.LINE_AA)
        stroke = Stroke([(45 / 319, 108 / 179), (275 / 319, 72 / 179)], 24, 0)
        conservative = analyze_meteor_blend_parameters(source, base, stroke, "保守")
        strong = analyze_meteor_blend_parameters(source, base, stroke, "强力")
        self.assertEqual(conservative["strength"], "保守")
        self.assertEqual(strong["strength"], "强力")
        self.assertGreater(strong["feather"], conservative["feather"])
        self.assertGreaterEqual(strong["cleanup"], conservative["cleanup"])
        self.assertGreater(strong["black_point"], 0)
        self.assertGreater(strong["snr"], 3)

    def test_automatic_object_blend_reduces_hard_edge_residual(self):
        base = np.full((110, 200, 3), (32, 38, 48), dtype=np.uint8)
        source = base.copy()
        source[35:76, 22:178] = np.clip(
            source[35:76, 22:178].astype(np.int16) + np.asarray((8, 2, 10)), 0, 255
        ).astype(np.uint8)
        cv2.line(source, (38, 57), (162, 53), (205, 220, 238), 4, cv2.LINE_AA)
        stroke = Stroke([(30 / 199, 58 / 109), (170 / 199, 52 / 109)], 28, 0)
        plain, _ = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25,
            "自然融合", True, 100, 0, False,
        )
        optimized, _ = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25,
            "自然融合", True, 100, 0, True,
        )
        sky = np.s_[39:46, 75:125]
        plain_error = np.abs(plain[sky].astype(np.int16) - base[sky].astype(np.int16)).max()
        optimized_error = np.abs(optimized[sky].astype(np.int16) - base[sky].astype(np.int16)).max()
        self.assertLess(optimized_error, plain_error)
        self.assertGreater(int(optimized[51:61, 85:115].max()), 170)

    def test_base_exposure_preserves_meteor_signal(self):
        base = np.full((40, 60, 3), 20, dtype=np.uint8)
        composite = base.copy()
        composite[18:22, 20:40] = 100
        adjusted = adjust_composite_base_exposure(composite, base, 1.0)
        self.assertEqual(int(adjusted[0, 0, 0]), 40)
        self.assertEqual(int(adjusted[20, 30, 0] - adjusted[0, 0, 0]), 80)

    def test_highlight_preservation_compensates_feather_loss(self):
        base = np.full((120, 180, 3), 20, dtype=np.uint8)
        source = base.copy()
        cv2.line(source, (35, 70), (145, 45), (190, 205, 220), 3, cv2.LINE_AA)
        stroke = Stroke([(35 / 179, 70 / 119), (145 / 179, 45 / 119)], 4, 22)
        plain, _ = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25, "自然融合", False, 100
        )
        preserved, _ = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25, "自然融合", True, 100
        )
        plain_signal = int(np.maximum(plain.astype(np.int16) - base, 0).sum())
        preserved_signal = int(np.maximum(preserved.astype(np.int16) - base, 0).sum())
        self.assertGreater(preserved_signal, plain_signal)

    def test_brightness_gain_changes_only_positive_signal(self):
        base = np.full((80, 120, 3), 40, dtype=np.uint8)
        source = base.copy()
        source[35:42, 25:95] = 140
        stroke = Stroke([(25 / 119, 38 / 79), (95 / 119, 38 / 79)], 10, 4)
        normal, _ = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25, "自然融合", True, 100
        )
        boosted, _ = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25, "自然融合", True, 150
        )
        self.assertGreater(int(boosted.max()), int(normal.max()))
        self.assertTrue(np.array_equal(boosted[0, 0], base[0, 0]))

    def test_background_cleanup_removes_colored_halo(self):
        base = np.full((100, 180, 3), (35, 40, 50), dtype=np.uint8)
        source = np.clip(base.astype(np.int16) + np.asarray((28, 4, 18)), 0, 255).astype(np.uint8)
        cv2.line(source, (35, 52), (145, 52), (210, 220, 235), 4, cv2.LINE_AA)
        stroke = Stroke([(35 / 179, 52 / 99), (145 / 179, 52 / 99)], 7, 22)
        alpha = np.zeros(base.shape[:2], dtype=np.float32)
        cv2.line(alpha, (35, 52), (145, 52), 1.0, 7, cv2.LINE_AA)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=8, sigmaY=8)
        cleaned = remove_local_background_cast(source, base, alpha, 100)
        sky = (alpha > 0.01) & (alpha < 0.15) & (np.abs(np.indices(alpha.shape)[0] - 52) > 8)
        plain_error = np.abs(source.astype(np.int16) - base.astype(np.int16))[sky].mean()
        cleaned_error = np.abs(cleaned - base.astype(np.float32))[sky].mean()
        self.assertLess(cleaned_error, plain_error * 0.45)

    def test_levels_style_cleanup_rejects_weak_colored_blocks(self):
        base = np.full((110, 200, 3), (32, 38, 48), dtype=np.uint8)
        source = base.copy()
        # Simulate the faint coloured rectangle left by an aligned source layer.
        source[38:73, 25:176] = np.clip(
            source[38:73, 25:176].astype(np.int16) + np.asarray((7, 1, 9)),
            0, 255,
        ).astype(np.uint8)
        cv2.line(source, (40, 56), (160, 56), (205, 220, 238), 4, cv2.LINE_AA)
        stroke = Stroke([(30 / 199, 56 / 109), (170 / 199, 56 / 109)], 28, 12)
        result, _mask = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25,
            "自然融合", True, 100, 85,
        )
        weak_block_error = np.abs(
            result[42:48, 70:130].astype(np.int16)
            - base[42:48, 70:130].astype(np.int16)
        )
        self.assertLessEqual(int(weak_block_error.max()), 1)
        self.assertGreater(int(result[54:59, 90:110].max()), 170)

    def test_default_cleanup_hides_dark_sky_block(self):
        base = np.full((110, 200, 3), (32, 38, 48), dtype=np.uint8)
        source = base.copy()
        source[38:73, 25:176] = np.clip(
            source[38:73, 25:176].astype(np.int16) + np.asarray((7, 1, 9)), 0, 255
        ).astype(np.uint8)
        cv2.line(source, (40, 56), (160, 56), (205, 220, 238), 4, cv2.LINE_AA)
        stroke = Stroke([(30 / 199, 56 / 109), (170 / 199, 56 / 109)], 28, 12)
        result, _mask = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25,
            "自然融合", True, 100, 70,
        )
        weak_block_error = np.abs(
            result[42:48, 70:130].astype(np.int16) - base[42:48, 70:130].astype(np.int16)
        )
        self.assertLessEqual(int(weak_block_error.max()), 1)
        self.assertGreater(int(result[54:59, 90:110].max()), 170)

    def test_two_meteors_can_use_different_brightness(self):
        base = np.full((100, 180, 3), 20, dtype=np.uint8)
        source = base.copy()
        cv2.line(source, (25, 28), (75, 28), (150, 150, 150), 5, cv2.LINE_AA)
        cv2.line(source, (105, 70), (155, 70), (150, 150, 150), 5, cv2.LINE_AA)
        dim = Stroke([(25 / 179, 28 / 99), (75 / 179, 28 / 99)], 10, 4, brightness_override=60)
        bright = Stroke([(105 / 179, 70 / 99), (155 / 179, 70 / 99)], 10, 4, brightness_override=160)
        result, _ = compose_meteor_objects(
            source, base, [dim, bright], False, False, 15, 25,
            "自然融合", True, 100, 0,
        )
        self.assertGreater(int(result[65:76, 100:160].max()), int(result[23:34, 20:80].max()))

    def test_unrelated_star_inside_feather_is_not_pasted(self):
        base = np.full((100, 190, 3), 25, dtype=np.uint8)
        source = base.copy()
        cv2.line(source, (30, 48), (155, 48), (175, 185, 205), 5, cv2.LINE_AA)
        cv2.circle(source, (92, 65), 4, (250, 250, 250), -1, cv2.LINE_AA)
        stroke = Stroke([(30 / 189, 48 / 99), (155 / 189, 48 / 99)], 8, 26)
        result, mask = compose_meteor_objects(
            source, base, [stroke], False, False, 15, 25,
            "自然融合", True, 100, 0,
        )
        self.assertGreater(float(mask[65, 92]), 0.01)
        self.assertGreater(int(result[48, 92].max()), 150)
        self.assertLess(int(result[65, 92].max()), 35)

    def test_paint_after_eraser_restores_mask_in_stroke_order(self):
        base = np.zeros((90, 160, 3), dtype=np.uint8)
        source = base.copy()
        cv2.line(source, (20, 45), (140, 45), (220, 220, 220), 7, cv2.LINE_AA)
        first = Stroke([(20 / 159, 45 / 89), (140 / 159, 45 / 89)], 14, 4)
        erased = Stroke([(70 / 159, 45 / 89), (100 / 159, 45 / 89)], 24, 3, erase=True)
        supplement = Stroke([(78 / 159, 45 / 89), (92 / 159, 45 / 89)], 10, 2)
        _erased_result, erased_mask = compose_meteor_objects(
            source, base, [first, erased], False, False, 15, 25,
            "自然融合", True, 100, 0,
        )
        _result, mask = compose_meteor_objects(
            source, base, [first, erased, supplement], False, False, 15, 25,
            "自然融合", True, 100, 0,
        )
        self.assertLess(float(erased_mask[45, 85]), 0.10)
        self.assertGreater(float(mask[45, 85]), 0.45)
        self.assertGreater(float(mask[45, 35]), 0.45)


class PreviewCoalescingTests(unittest.TestCase):
    def test_running_global_preview_blocks_overlapping_worker(self):
        fake = SimpleNamespace(
            preview_base=np.zeros((5, 5, 3), dtype=np.uint8),
            global_preview_loading_signature="older-state",
        )
        MeteorComposer._request_global_preview(fake, "newer-state")
        self.assertEqual(fake.global_preview_loading_signature, "older-state")


class CandidateThresholdOrderingTests(unittest.TestCase):
    def test_threshold_refresh_preserves_manual_paint_erase_paint_order(self):
        key = "image.tif"
        first = Stroke([(0.1, 0.5), (0.8, 0.5)], 20, 3)
        erased = Stroke([(0.4, 0.5), (0.6, 0.5)], 24, 2, erase=True)
        restored = Stroke([(0.48, 0.5), (0.52, 0.5)], 12, 1)
        fake = SimpleNamespace(
            candidate_thresholds={key: 55},
            candidate_threshold=FakeVar(55),
            strokes={key: [first, erased, restored]},
            candidates={key: []},
            current_path=Path(key),
            _update_candidate_summary=lambda _key: None,
        )
        MeteorComposer._apply_candidate_threshold(fake, key)
        self.assertEqual(fake.strokes[key], [first, erased, restored])


class SourceAnnotationTests(unittest.TestCase):
    def test_original_state_uses_explicit_warning_label(self):
        image = np.zeros((180, 320, 3), dtype=np.uint16)
        annotated, records = annotate_meteor_sources(image, [{
            "source": "DSC01234", "boxes": [(80, 70, 180, 100)],
            "original_state": True,
        }])
        self.assertTrue(records[0]["original_state"])
        self.assertIn("ORIGINAL / UNALIGNED", records[0]["label"])
        self.assertIsNotNone(records[0]["warning"])
        self.assertGreater(int(annotated.max()), 0)

    def test_transformed_meteor_records_original_position_and_geometry(self):
        image = np.zeros((180, 320, 3), dtype=np.uint16)
        stroke = Stroke(
            [(0.20, 0.55), (0.45, 0.45)], 16, 4,
            offset_x=45, offset_y=-12, rotation=18, length_scale=1.35,
        )
        annotations = meteor_source_annotations("DSC04567", [stroke], 320, 180)
        annotated, records = annotate_meteor_sources(image, annotations)
        self.assertTrue(records[0]["transformed"])
        self.assertIn("TRANSFORMED", records[0]["label"])
        self.assertIsNotNone(records[0]["original_box"])
        self.assertNotEqual(records[0]["box"], records[0]["original_box"])
        self.assertGreater(int(annotated.max()), 0)

    def test_geometry_reset_keeps_mask_opacity_and_adjustments(self):
        stroke = Stroke(
            [(0.1, 0.2), (0.5, 0.4)], 22, 7,
            offset_x=40, offset_y=10, rotation=25,
            length_scale=1.7, width_scale=0.8, opacity=0.6,
            brightness_override=135,
        )
        self.assertTrue(stroke_is_transformed(stroke))
        reset_stroke_geometry(stroke)
        self.assertFalse(stroke_is_transformed(stroke))
        self.assertEqual((stroke.width, stroke.feather), (22, 7))
        self.assertEqual(stroke.opacity, 0.6)
        self.assertEqual(stroke.brightness_override, 135)


class ExportModeTests(unittest.TestCase):
    @staticmethod
    def write_tiff(path: Path, image: np.ndarray):
        tifffile.imwrite(path, image, photometric="rgb")

    def test_combined_mode_accumulates_sources_into_one_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / "base.tif"
            first = root / "first.tif"
            second = root / "second.tif"
            image = np.zeros((80, 120, 3), dtype=np.uint16)
            self.write_tiff(base, image)
            source1 = image.copy(); source1[15:25, 10:50] = 50000
            source2 = image.copy(); source2[50:60, 70:110] = 50000
            self.write_tiff(first, source1); self.write_tiff(second, source2)
            original_first = root / "first_original.tif"
            self.write_tiff(original_first, source1)
            marked = {
                first: [Stroke([(0.08, 0.25), (0.42, 0.25)], 14, 0)],
                second: [Stroke([(0.58, 0.70), (0.92, 0.70)], 14, 0)],
            }
            fake = SimpleNamespace(work_queue=queue.Queue())
            result = MeteorComposer._export_worker(
                fake, root / "out", marked, {str(first): base, str(second): base}, {},
                {"match_exposure": False, "curve_enabled": False,
                 "curve_shadows": 15, "curve_highlights": 25},
                False, False, "普通粘贴",
                {str(first): original_first, str(second): second}, "combined",
            )
            run_dir = Path(result[1])
            outputs = list((run_dir / "final_jpg").glob("*.jpg"))
            self.assertEqual(len(outputs), 2)
            clean_output = next(path for path in outputs if "来源标注" not in path.name)
            labeled_output = next(path for path in outputs if "来源标注" in path.name)
            final = cv2.imdecode(np.fromfile(clean_output, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertGreater(int(final[20, 25].max()), 20)
            self.assertGreater(int(final[55, 90].max()), 20)
            labeled = cv2.imdecode(np.fromfile(labeled_output, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertFalse(np.array_equal(final, labeled))
            report = json.loads((run_dir / "processing_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["output_mode"], "combined")
            self.assertEqual(len(report["items"]), 2)
            self.assertGreaterEqual(len(report["source_labels"]), 2)
            self.assertEqual({item["source"] for item in report["source_labels"]}, {"first", "second"})
            first_labels = [item for item in report["source_labels"] if item["source"] == "first"]
            self.assertTrue(first_labels and all(item["original_state"] for item in first_labels))
            self.assertTrue(report["items"][0]["original_state"])
            self.assertFalse(report["items"][1]["original_state"])
            self.assertEqual(report["items"][0]["outputs"], report["items"][1]["outputs"])

    def test_separate_mode_writes_unique_source_names(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.full((40, 60, 3), 20000, dtype=np.uint16)
            sources, bases, marked, pairs = [], [], {}, {}
            for name in ("a", "b"):
                source, base = root / f"{name}.tif", root / f"{name}.png"
                self.write_tiff(source, image)
                cv2.imwrite(str(base), np.zeros((40, 60, 3), dtype=np.uint16))
                sources.append(source); bases.append(base)
                marked[source] = [Stroke([(0.1, 0.5), (0.9, 0.5)], 10, 0)]
                pairs[str(source)] = base
            fake = SimpleNamespace(work_queue=queue.Queue())
            result = MeteorComposer._export_worker(
                fake, root / "out", marked, pairs, {},
                {"match_exposure": False, "curve_enabled": False,
                 "curve_shadows": 15, "curve_highlights": 25},
                False, False, "普通粘贴", {str(path): path for path in sources}, "separate",
            )
            names = sorted(path.name for path in (Path(result[1]) / "final_jpg").glob("*.jpg"))
            self.assertEqual(names, ["a.jpg", "a_来源标注.jpg", "b.jpg", "b_来源标注.jpg"])

    def test_global_preview_returns_clean_and_labeled_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "DSC01000.tif"
            image = np.zeros((80, 120, 3), dtype=np.uint16)
            image[30:42, 20:95] = 52000
            self.write_tiff(source, image)
            fake = SimpleNamespace(work_queue=queue.Queue())
            result = MeteorComposer._global_preview_worker(
                fake, "signature", np.zeros((80, 120, 3), dtype=np.uint8),
                {source: [Stroke([(0.15, 0.45), (0.82, 0.45)], 14, 2)]}, {},
                {"match_exposure": False, "curve_enabled": False,
                 "curve_shadows": 15, "curve_highlights": 25},
                "普通粘贴", {str(source): source},
            )
            kind, signature, clean, labeled, included = result
            self.assertEqual((kind, signature, included), ("global_preview", "signature", 1))
            self.assertFalse(np.array_equal(clean, labeled))



if __name__ == "__main__":
    unittest.main()
