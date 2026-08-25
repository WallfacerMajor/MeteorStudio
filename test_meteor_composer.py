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
    normalize_lsd_lines, point_in_padded_bbox,
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
            },
            image_adjustments=adjustments,
            loading_adjustments=False,
            match_exposure=FakeVar(), match_exposure_policy=FakeVar(),
            curve_enabled=FakeVar(), curve_shadows=FakeVar(), curve_highlights=FakeVar(),
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
        )
        result = MeteorComposer._stroke_start(fake, SimpleNamespace(x=20, y=20, state=0))
        self.assertIsNone(result)
        self.assertEqual(fake.active_points, [(0.25, 0.5)])
        self.assertEqual(fake.active_tool_mode, "paint")
        self.assertEqual(cleared, [True])

    def test_blend_preview_blocks_mask_painting(self):
        messages = []
        fake = SimpleNamespace(
            current_path="image.tif",
            view_mode=SimpleNamespace(get=lambda: "blend"),
            status=SimpleNamespace(set=lambda message: messages.append(message)),
        )
        result = MeteorComposer._stroke_start(fake, SimpleNamespace(x=10, y=10, state=0))
        self.assertEqual(result, "break")
        self.assertIn("融合预览", messages[0])

    def test_labeled_preview_blocks_mask_painting(self):
        messages = []
        fake = SimpleNamespace(
            current_path="image.tif",
            view_mode=FakeVar("labeled"),
            status=SimpleNamespace(set=lambda message: messages.append(message)),
        )
        result = MeteorComposer._stroke_start(fake, SimpleNamespace(x=10, y=10, state=0))
        self.assertEqual(result, "break")
        self.assertIn("来源标注", messages[0])

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
            marked = {
                first: [Stroke([(0.08, 0.25), (0.42, 0.25)], 14, 0)],
                second: [Stroke([(0.58, 0.70), (0.92, 0.70)], 14, 0)],
            }
            fake = SimpleNamespace(work_queue=queue.Queue())
            result = MeteorComposer._export_worker(
                fake, root / "out", marked, {str(first): base, str(second): base}, {},
                {"match_exposure": False, "curve_enabled": False,
                 "curve_shadows": 15, "curve_highlights": 25},
                False, False, "普通粘贴", {str(first): first, str(second): second}, "combined",
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
            self.assertEqual(names, ["a.jpg", "b.jpg"])

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
