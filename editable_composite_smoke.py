"""Real Tk smoke test for direct editing in final/labeled composite views."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


def run_smoke(app) -> dict:
    from meteor_composer import (
        ExactPreviewViewer, Stroke, adjust_composite_base_exposure,
        compose_meteor_objects, transformed_stroke_points,
    )

    app.geometry("1280x820+10000+10000")
    app.update_idletasks()
    for after_id in app.tk.call("after", "info"):
        app.after_cancel(after_id)
    app._schedule_autosave = lambda: None
    app.autosave_suspended = True

    tab_names = [
        app.control_notebook.tab(index, "text")
        for index in range(app.control_notebook.index("end"))
    ]
    if tab_names != ["3  蒙版与候选", "4  融合与底图", "5  所选流星", "6  操作历史"]:
        raise AssertionError(f"Unexpected workspace tabs: {tab_names}")
    required_controls = {
        "B ✎ 画笔", "E ▱ 橡皮擦", "本地模型分析当前单张", "自动检测全部",
        "保存项目", "载入项目", "自动优化当前流星", "自动优化全部流星",
        "重置底图曝光", "恢复自动值", "恢复原始融合", "导出合成结果",
        "↶ 撤销", "↷ 重做", "历史记录",
    }
    available_controls = set()
    pending = [app]
    while pending:
        widget = pending.pop()
        pending.extend(widget.winfo_children())
        try:
            text = widget.cget("text")
        except Exception:
            continue
        if text:
            available_controls.add(str(text))
    missing_controls = required_controls - available_controls
    if missing_controls:
        raise AssertionError(f"UI reorganization hid controls: {sorted(missing_controls)}")
    removed_exact_controls = {
        "生成并打开导出级精确预览", "打开已生成预览",
    } & available_controls
    if removed_exact_controls:
        raise AssertionError(f"Obsolete exact-preview controls remain: {sorted(removed_exact_controls)}")
    app._set_paths_panel_visible(True)
    app.update_idletasks()
    app._toggle_paths_panel()
    app.update_idletasks()
    if app.paths_panel.winfo_manager():
        raise AssertionError("Material panel did not collapse")
    app._toggle_paths_panel()
    app.update_idletasks()
    if not app.paths_panel.winfo_manager():
        raise AssertionError("Material panel could not be restored")
    # Editing normally happens with the completed material setup collapsed,
    # which leaves enough canvas room for precise object interaction.
    app._toggle_paths_panel()
    app.update_idletasks()
    app.view_mode.set("source")
    app._view_mode_changed()
    if app.control_notebook.select() != str(app.mask_tools_tab):
        raise AssertionError("Source view did not expose mask controls")
    app.view_mode.set("base")
    app._view_mode_changed()
    if app.control_notebook.select() != str(app.blend_tools_tab):
        raise AssertionError("Base view did not expose blend controls")

    width, height = 1200, 800
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:] = (10, 15, 28)
    source = base.copy()
    cv2.line(source, (300, 420), (760, 300), (235, 245, 255), 8, cv2.LINE_AA)
    key = str(Path("synthetic_meteor.tif"))
    app.current_path = Path(key)
    app.files = [app.current_path]
    app.preview_source = source
    app.preview_base = base
    app.preview_rgb = source
    app.current_dims = (width, height)
    meteor = Stroke(
        [(300 / width, 420 / height), (760 / width, 300 / height)],
        32, 12, locked=True, auto_score=91,
    )
    app.strokes = {key: [meteor]}
    app.candidates = {key: [replace(meteor, points=meteor.points.copy())]}
    # The production editor only exposes objects belonging to the active
    # source/base pairing.  Keep the smoke fixture equivalent to a scanned
    # current batch so stale autosave masks cannot accidentally pass it.
    app.pairs = {key: Path("synthetic_base.tif")}
    app.output_mode.set("separate")
    app.canvas.configure(width=900, height=560)
    app.update_idletasks()
    app.view_mode.set("base")
    app._render_preview()
    if not np.array_equal(app.preview_rgb, base):
        raise AssertionError("Clean base view was contaminated by mask or annotations")

    app.view_mode.set("source")
    app._render_preview()
    app.update()
    # Match the ordinary application entry point instead of testing only an
    # off-screen fixed geometry. Windows performs another layout settlement
    # when the real workspace enters its native maximized state.
    app.geometry("1280x820+0+0")
    app.update_idletasks()
    app.maximize_for_normal_launch()
    app.update_idletasks()
    app.update()
    app._canvas_fit()
    x0, y0, x1, y1 = app.display_box
    before_manual = len(app.strokes[key])
    first_manual = None
    for start_ratio, end_ratio in (((0.12, 0.18), (0.28, 0.24)), ((0.66, 0.68), (0.82, 0.73))):
        sx = int(x0 + (x1 - x0) * start_ratio[0])
        sy = int(y0 + (y1 - y0) * start_ratio[1])
        ex = int(x0 + (x1 - x0) * end_ratio[0])
        ey = int(y0 + (y1 - y0) * end_ratio[1])
        app.canvas.event_generate("<ButtonPress-1>", x=sx, y=sy)
        app.canvas.event_generate("<B1-Motion>", x=ex, y=ey, state=0x0100)
        app.canvas.event_generate("<ButtonRelease-1>", x=ex, y=ey)
        app.update()
        if first_manual is None:
            first_manual = app.strokes[key][-1]
    if len(app.strokes[key]) != before_manual + 2:
        raise AssertionError("Second manual stroke replaced the previous stroke")
    if first_manual not in app.strokes[key]:
        raise AssertionError("First manual stroke disappeared after supplementing the mask")

    second_start = (int(x0 + (x1 - x0) * 0.70), int(y0 + (y1 - y0) * 0.69))
    second_end = (int(x0 + (x1 - x0) * 0.78), int(y0 + (y1 - y0) * 0.72))
    sample_x = int(round(((second_start[0] + second_end[0]) * 0.5 - x0) / max(1, x1 - x0) * (width - 1)))
    sample_y = int(round(((second_start[1] + second_end[1]) * 0.5 - y0) / max(1, y1 - y0) * (height - 1)))
    app._set_edit_mode("erase")
    app.canvas.event_generate("<ButtonPress-1>", x=second_start[0], y=second_start[1])
    app.canvas.event_generate("<B1-Motion>", x=second_end[0], y=second_end[1], state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=second_end[0], y=second_end[1])
    app.update()
    erased_value = float(app._preview_mask()[sample_y, sample_x])
    app._set_edit_mode("paint")
    app.canvas.event_generate("<ButtonPress-1>", x=second_start[0], y=second_start[1])
    app.canvas.event_generate("<B1-Motion>", x=second_end[0], y=second_end[1], state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=second_end[0], y=second_end[1])
    app.update()
    restored_value = float(app._preview_mask()[sample_y, sample_x])
    if restored_value <= erased_value + 0.20:
        raise AssertionError(
            f"Painting after erasing did not restore the mask: {erased_value:.3f} -> {restored_value:.3f}; "
            f"strokes={[(item.erase, len(item.points), item.width) for item in app.strokes[key]]}"
        )

    # Regression: click an anchor, Shift-click a straight segment, then press a
    # new ordinary stroke. The committed Shift segment must remain in the mask
    # even before the new mouse press is released.
    app._set_edit_mode("paint")
    shift_sequence_start = len(app.strokes[key])
    anchor_canvas = (int(x0 + (x1 - x0) * 0.34), int(y0 + (y1 - y0) * 0.78))
    shift_end_canvas = (int(x0 + (x1 - x0) * 0.57), int(y0 + (y1 - y0) * 0.66))
    app.canvas.event_generate("<ButtonPress-1>", x=anchor_canvas[0], y=anchor_canvas[1])
    app.canvas.event_generate("<ButtonRelease-1>", x=anchor_canvas[0], y=anchor_canvas[1])
    app.canvas.event_generate(
        "<ButtonPress-1>", x=shift_end_canvas[0], y=shift_end_canvas[1], state=0x0001
    )
    app.canvas.event_generate(
        "<ButtonRelease-1>", x=shift_end_canvas[0], y=shift_end_canvas[1], state=0x0001
    )
    app.update()
    shift_mid_canvas = (
        (anchor_canvas[0] + shift_end_canvas[0]) // 2,
        (anchor_canvas[1] + shift_end_canvas[1]) // 2,
    )
    shift_mid_pixel = (
        int(round((shift_mid_canvas[0] - x0) / max(1, x1 - x0) * (width - 1))),
        int(round((shift_mid_canvas[1] - y0) / max(1, y1 - y0) * (height - 1))),
    )
    before_next_press = float(app._preview_mask()[shift_mid_pixel[1], shift_mid_pixel[0]])
    next_canvas = (int(x0 + (x1 - x0) * 0.20), int(y0 + (y1 - y0) * 0.82))
    app.canvas.event_generate("<ButtonPress-1>", x=next_canvas[0], y=next_canvas[1])
    next_end_canvas = (next_canvas[0] + 18, next_canvas[1] - 8)
    app.canvas.event_generate("<B1-Motion>", x=next_end_canvas[0], y=next_end_canvas[1])
    app.update()
    live_mask = app._preview_mask()
    during_next_press = float(live_mask[shift_mid_pixel[1], shift_mid_pixel[0]])
    if app.active_canvas_line is None:
        raise AssertionError("Dragging paint did not create the foreground brush trace")
    if before_next_press < 0.35 or during_next_press < 0.35:
        raise AssertionError(
            f"Shift segment disappeared when the next paint stroke began: "
            f"{before_next_press:.3f} -> {during_next_press:.3f}"
        )
    app.canvas.event_generate("<ButtonRelease-1>", x=next_end_canvas[0], y=next_end_canvas[1])
    app.update()
    if len(app.strokes[key]) != shift_sequence_start + 3:
        raise AssertionError("Click/Shift-click/next-click did not accumulate three paint strokes")

    # Regression: a stale/spurious Control modifier must never reinterpret a
    # left-button brush press as "delete nearest mask". Left paint owns the
    # complete event sequence; whole-mask deletion is right-click only.
    modifier_start = len(app.strokes[key])
    protected_mask = app._preview_mask().copy()
    ctrl_start = (next_end_canvas[0] + 8, next_end_canvas[1] + 5)
    ctrl_end = (ctrl_start[0] + 12, ctrl_start[1] + 3)
    app.canvas.event_generate("<ButtonPress-1>", x=ctrl_start[0], y=ctrl_start[1], state=0x0004)
    app.canvas.event_generate("<B1-Motion>", x=ctrl_end[0], y=ctrl_end[1], state=0x0004)
    app.canvas.event_generate("<ButtonRelease-1>", x=ctrl_end[0], y=ctrl_end[1], state=0x0004)
    app.update()
    if len(app.strokes[key]) != modifier_start + 1:
        raise AssertionError("Modified left click deleted/replaced a nearby paint stroke")
    after_modifier_mask = app._preview_mask()
    if np.any(after_modifier_mask + 1e-6 < protected_mask):
        raise AssertionError("Modified nearby paint reduced the previously committed mask")

    # Simulate the real failure mode: mouse-up from the first stroke is lost
    # while the preview repaints. Starting stroke two must auto-commit stroke one.
    missed_release_start = len(app.strokes[key])
    first_start = (int(x0 + (x1 - x0) * 0.12), int(y0 + (y1 - y0) * 0.38))
    first_end = (int(x0 + (x1 - x0) * 0.25), int(y0 + (y1 - y0) * 0.34))
    second_start = (int(x0 + (x1 - x0) * 0.72), int(y0 + (y1 - y0) * 0.25))
    second_end = (int(x0 + (x1 - x0) * 0.84), int(y0 + (y1 - y0) * 0.30))
    app.canvas.event_generate("<ButtonPress-1>", x=first_start[0], y=first_start[1])
    app.canvas.event_generate("<B1-Motion>", x=first_end[0], y=first_end[1], state=0x0100)
    app.update()
    app.canvas.event_generate("<ButtonPress-1>", x=second_start[0], y=second_start[1])
    app.update()
    if len(app.strokes[key]) != missed_release_start + 1:
        raise AssertionError("Second mouse-down did not preserve/commit the first live stroke")
    app.canvas.event_generate("<B1-Motion>", x=second_end[0], y=second_end[1], state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=second_end[0], y=second_end[1])
    app.update()
    if len(app.strokes[key]) != missed_release_start + 2:
        raise AssertionError("Two physical paint gestures did not produce two committed strokes")
    if app.canvas.grab_current() is not None:
        raise AssertionError("Brush mouse capture was not released after committing")

    # Adjacent/overlapping/crossing paint must be monotonic: a later PAINT
    # gesture may add pixels but can never remove pixels from an earlier one.
    close_start_count = len(app.strokes[key])
    close_a = (
        (int(x0 + (x1 - x0) * 0.40), int(y0 + (y1 - y0) * 0.52)),
        (int(x0 + (x1 - x0) * 0.62), int(y0 + (y1 - y0) * 0.52)),
    )
    close_b = (
        (int(x0 + (x1 - x0) * 0.46), int(y0 + (y1 - y0) * 0.54)),
        (int(x0 + (x1 - x0) * 0.68), int(y0 + (y1 - y0) * 0.54)),
    )
    cross = (
        (int(x0 + (x1 - x0) * 0.54), int(y0 + (y1 - y0) * 0.46)),
        (int(x0 + (x1 - x0) * 0.54), int(y0 + (y1 - y0) * 0.60)),
    )
    for start_point, end_point in (close_a, close_b, cross):
        app.canvas.event_generate("<ButtonPress-1>", x=start_point[0], y=start_point[1])
        app.canvas.event_generate("<B1-Motion>", x=end_point[0], y=end_point[1], state=0x0100)
        app.canvas.event_generate("<ButtonRelease-1>", x=end_point[0], y=end_point[1])
        app.update()
        if app.strokes[key][-1].erase:
            raise AssertionError("Paint tool unexpectedly committed a nearby stroke as an eraser")
        if len(app.strokes[key]) == close_start_count + 1:
            first_only_canvas = (
                int(x0 + (x1 - x0) * 0.42), int(y0 + (y1 - y0) * 0.52)
            )
            first_only_pixel = (
                int(round((first_only_canvas[0] - x0) / max(1, x1 - x0) * (width - 1))),
                int(round((first_only_canvas[1] - y0) / max(1, y1 - y0) * (height - 1))),
            )
            close_first_value = float(
                app._preview_mask()[first_only_pixel[1], first_only_pixel[0]]
            )
    close_final_value = float(app._preview_mask()[first_only_pixel[1], first_only_pixel[0]])
    if close_first_value < 0.35 or close_final_value + 0.01 < close_first_value:
        raise AssertionError(
            f"Nearby paint removed earlier mask pixels: {close_first_value:.3f} -> {close_final_value:.3f}"
        )
    if len(app.strokes[key]) != close_start_count + 3:
        raise AssertionError("Adjacent/overlapping/crossing paint strokes did not all accumulate")

    # Single-image transformation must use the same direct canvas editor as the
    # combined preview; the source-view context action must not open a modal.
    app.output_mode.set("separate")
    app.view_mode.set("source")
    app.context_stroke_index = 0
    children_before_direct_edit = set(app.winfo_children())
    app._transform_context_stroke()
    app.update()
    if app.view_mode.get() != "blend" or app.selected_object != (key, 0):
        raise AssertionError("Single-image transform did not enter direct blend editing")
    if set(app.winfo_children()) != children_before_direct_edit:
        raise AssertionError("Single-image transform unexpectedly opened a modal window")
    if not app.object_overlay_items or not app.object_handle_centers:
        raise AssertionError("Single-image transform did not show direct manipulation handles")

    app.view_mode.set("labeled")
    app._render_preview()
    app.update()

    selected = (key, 0)
    geometry = app._object_canvas_geometry(selected)
    if geometry is None:
        raise AssertionError("Object geometry missing")
    center = geometry["center"]
    before = replace(app.strokes[key][0], points=app.strokes[key][0].points.copy())
    separate_drag_global_calls = 0
    real_separate_drag_invalidate = app._invalidate_global_preview
    def count_separate_drag_global():
        nonlocal separate_drag_global_calls
        separate_drag_global_calls += 1
        return real_separate_drag_invalidate()
    app._invalidate_global_preview = count_separate_drag_global
    separate_committed_frame = app.preview_rgb
    drag_press_started = time.perf_counter()
    app.canvas.event_generate("<ButtonPress-1>", x=int(center[0]), y=int(center[1]))
    app.update()
    drag_press_elapsed = time.perf_counter() - drag_press_started
    if app.object_drag_live_background is not separate_committed_frame:
        raise AssertionError("Drag start copied the complete committed frame instead of borrowing it")
    if app.object_drag_live_frame is not None:
        raise AssertionError("Drag start allocated a second complete live frame")
    app.canvas.event_generate("<B1-Motion>", x=int(center[0] + 42), y=int(center[1] + 21), state=0x0100)
    app.update()
    moving = app.strokes[key][0]
    original_midpoint = np.mean(
        np.asarray([(x * (width - 1), y * (height - 1)) for x, y in moving.points]), axis=0
    )
    moved_midpoint = original_midpoint + np.asarray([moving.offset_x, moving.offset_y])
    mx, my = int(round(moved_midpoint[0])), int(round(moved_midpoint[1]))
    ox, oy = int(round(original_midpoint[0])), int(round(original_midpoint[1]))
    live_patches = app.object_drag_live_last_patches
    if not live_patches:
        raise AssertionError("Live drag did not produce a sparse display patch")
    live_patch_bytes = sum(patch.nbytes for patch, _box in live_patches)
    largest_live_patch = max(patch.nbytes for patch, _box in live_patches)
    if (
        largest_live_patch >= separate_committed_frame.nbytes // 2
        or live_patch_bytes >= separate_committed_frame.nbytes * 3 // 4
    ):
        raise AssertionError(
            f"Live drag patch unexpectedly approached full-frame size: "
            f"patches={live_patch_bytes}, frame={separate_committed_frame.shape}/{separate_committed_frame.nbytes}"
        )
    if drag_press_elapsed > 0.25:
        raise AssertionError(f"Drag press feedback was not immediate: {drag_press_elapsed:.3f}s")
    def live_peak(px: int, py: int, patches) -> int:
        for patch, (px0, py0, px1, py1) in reversed(patches):
            if px0 <= px < px1 and py0 <= py < py1:
                local_x, local_y = px - px0, py - py0
                return int(patch[
                    max(0, local_y - 6):local_y + 7,
                    max(0, local_x - 6):local_x + 7,
                ].max())
        return 0
    live_new_peak = live_peak(mx, my, live_patches)
    live_old_peak = live_peak(ox, oy, live_patches)
    if live_new_peak < 140 or live_old_peak > 100:
        raise AssertionError(
            f"Live drag moved only the mask, not meteor pixels: old={live_old_peak}, new={live_new_peak}"
        )
    app.canvas.event_generate("<ButtonRelease-1>", x=int(center[0] + 42), y=int(center[1] + 21))
    app.update()
    app._invalidate_global_preview = real_separate_drag_invalidate
    if separate_drag_global_calls:
        raise AssertionError("Single-image drag unnecessarily invalidated the full composite")
    moved = app.strokes[key][0]
    if moved.offset_x == before.offset_x and moved.offset_y == before.offset_y:
        raise AssertionError("Move did not update offsets")
    if app.candidates[key][0].offset_x != moved.offset_x:
        raise AssertionError("Candidate transform was not synchronized")

    combined_image, _combined_mask = compose_meteor_objects(
        source, base, app.strokes[key], False, False, 15, 25,
        "自然融合", True, 100, 70,
    )
    app.output_mode.set("combined")
    app.global_preview_rgb = combined_image
    app.global_labeled_preview_rgb = combined_image.copy()
    app.global_preview_signature = app._global_preview_state_signature()
    app.view_mode.set("blend")
    app._render_preview()
    app.update()
    geometry = app._object_canvas_geometry(selected)
    shared_center = geometry["center"]
    shared_before = replace(app.strokes[key][0], points=app.strokes[key][0].points.copy())
    app.canvas.event_generate("<ButtonPress-1>", x=int(shared_center[0]), y=int(shared_center[1]))
    app.canvas.event_generate(
        "<B1-Motion>", x=int(shared_center[0] + 28), y=int(shared_center[1] + 14), state=0x0100
    )
    app.update()
    shared_moving = app.strokes[key][0]
    shared_midpoint = np.mean(
        np.asarray([(x * (width - 1), y * (height - 1)) for x, y in shared_moving.points]), axis=0
    )
    shared_new = shared_midpoint + np.asarray([shared_moving.offset_x, shared_moving.offset_y])
    sx, sy = int(round(shared_new[0])), int(round(shared_new[1]))
    shared_patches = app.object_drag_live_last_patches
    if not shared_patches:
        raise AssertionError("Combined drag did not produce a sparse display patch")
    shared_peak = live_peak(sx, sy, shared_patches)
    if shared_peak < 140 or shared_moving.offset_x == shared_before.offset_x:
        raise AssertionError("Combined preview did not move meteor pixels during drag")
    app.canvas.event_generate(
        "<ButtonRelease-1>", x=int(shared_center[0] + 28), y=int(shared_center[1] + 14)
    )
    app.update()
    if app.global_preview_signature != app._global_preview_state_signature():
        raise AssertionError("Combined edit discarded the incremental preview cache")
    if app.global_exact_after_id is not None:
        raise AssertionError("Exact local transform unnecessarily scheduled a full validation")

    # Undo/redo and one-click restore after a drag must use the same exact local
    # footprint rebuild, with no global invalidation or delayed validation.
    transform_global_calls = 0
    transform_validation_calls = 0
    real_transform_invalidate = app._invalidate_global_preview
    real_transform_validation = app._schedule_global_exact_validation
    def count_transform_global():
        nonlocal transform_global_calls
        transform_global_calls += 1
        return real_transform_invalidate()
    def count_transform_validation(_signature):
        nonlocal transform_validation_calls
        transform_validation_calls += 1
    app._invalidate_global_preview = count_transform_global
    app._schedule_global_exact_validation = count_transform_validation
    transform_count_before_undo = len(app.strokes[key])
    app.undo_stroke()
    if len(app.strokes[key]) != transform_count_before_undo:
        raise AssertionError("Undo after dragging deleted the transformed mask")
    undone_transform = app.strokes[key][0]
    if (
        undone_transform.points != shared_before.points
        or undone_transform.offset_x != shared_before.offset_x
        or undone_transform.offset_y != shared_before.offset_y
    ):
        raise AssertionError("Undo after dragging did not restore the same mask geometry")
    app.redo_stroke()
    app.selected_object = selected

    # Every operation that changes only the selected meteor must keep the local
    # composite cache.  These used to invalidate the whole project and launch a
    # second exact pass even though the one-object compositor already had enough
    # information to update the affected footprint.
    selected_global_calls = 0
    real_selected_invalidate = app._invalidate_global_preview
    def count_selected_global():
        nonlocal selected_global_calls
        selected_global_calls += 1
        return real_selected_invalidate()
    app._invalidate_global_preview = count_selected_global
    app.restore_selected_original_blend()
    app.restore_selected_auto()
    app.original_sources[key] = Path(key)
    app.preview_aligned_source = source
    app.preview_original_source = source
    app._set_selected_source_mode("original")
    app._set_selected_source_mode("aligned")
    app._invalidate_global_preview = real_selected_invalidate
    if selected_global_calls:
        raise AssertionError(
            f"Selected-meteor restore/source operations invalidated globally {selected_global_calls} times"
        )
    app._reset_selected_object()
    app.undo_stroke()
    app._invalidate_global_preview = real_transform_invalidate
    app._schedule_global_exact_validation = real_transform_validation
    if transform_global_calls or transform_validation_calls:
        raise AssertionError(
            f"Transform restore triggered global work: invalidate={transform_global_calls}, "
            f"validation={transform_validation_calls}"
        )
    if app.global_preview_signature != app._global_preview_state_signature():
        raise AssertionError("Transform undo/redo/restore did not retain exact local cache")

    # A missing/stale undo record must be a no-op. The legacy fallback used to
    # guess that the final stroke was an add and silently remove the mask.
    saved_history = app.edit_history.pop(key, None)
    no_history_count = len(app.strokes[key])
    app.undo_stroke()
    if len(app.strokes[key]) != no_history_count:
        raise AssertionError("Undo without history deleted an existing mask")
    if saved_history is not None:
        app.edit_history[key] = saved_history

    # Adding a manual mask or accepting a detected candidate must update only
    # the affected footprint. Neither path may invalidate or asynchronously
    # rebuild the complete combined preview.
    app.view_mode.set("source")
    app._render_preview()
    app.update()
    mask_add_global_calls = 0
    mask_add_validation_calls = 0
    real_mask_add_invalidate = app._invalidate_global_preview
    real_mask_add_validation = app._schedule_global_exact_validation
    def count_mask_add_global():
        nonlocal mask_add_global_calls
        mask_add_global_calls += 1
        return real_mask_add_invalidate()
    def count_mask_add_validation(_signature):
        nonlocal mask_add_validation_calls
        mask_add_validation_calls += 1
    app._invalidate_global_preview = count_mask_add_global
    app._schedule_global_exact_validation = count_mask_add_validation
    x0, y0, x1, y1 = app.display_box
    before_local_add = len(app.strokes[key])
    local_start = (int(x0 + (x1 - x0) * 0.86), int(y0 + (y1 - y0) * 0.15))
    local_end = (int(x0 + (x1 - x0) * 0.92), int(y0 + (y1 - y0) * 0.19))
    app._set_edit_mode("paint")
    app.canvas.event_generate("<ButtonPress-1>", x=local_start[0], y=local_start[1])
    app.canvas.event_generate("<B1-Motion>", x=local_end[0], y=local_end[1], state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=local_end[0], y=local_end[1])
    app.update()
    if len(app.strokes[key]) != before_local_add + 1:
        raise AssertionError("Manual mask addition was not committed")
    if app.global_preview_signature != app._global_preview_state_signature():
        raise AssertionError("Manual mask addition discarded the combined preview cache")

    # A locally committed brush stroke is already pixel-exact.  Even with a
    # non-zero clean-base exposure, mouse-up must update only that ROI and mark
    # the exact state current rather than scheduling a duplicate full pass.
    app.base_exposure_tenths.set(5)
    app.exact_preview_full_rgb = adjust_composite_base_exposure(
        app.global_preview_rgb, app.preview_base, 0.5
    )
    app.exact_preview_rgb = app.exact_preview_full_rgb
    app.exact_preview_signature = app._exact_preview_state_signature()
    exact_schedule_calls = 0
    real_exact_schedule = app._schedule_automatic_exact_preview
    def count_exact_schedule():
        nonlocal exact_schedule_calls
        exact_schedule_calls += 1
    app._schedule_automatic_exact_preview = count_exact_schedule
    exposure_start = (int(x0 + (x1 - x0) * 0.08), int(y0 + (y1 - y0) * 0.11))
    exposure_end = (exposure_start[0] + 34, exposure_start[1] + 13)
    app.canvas.event_generate("<ButtonPress-1>", x=exposure_start[0], y=exposure_start[1])
    app.canvas.event_generate("<B1-Motion>", x=exposure_end[0], y=exposure_end[1], state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=exposure_end[0], y=exposure_end[1])
    app.update()
    app._schedule_automatic_exact_preview = real_exact_schedule
    if exact_schedule_calls:
        raise AssertionError("Manual stroke scheduled a duplicate exact-preview pass")
    if app.exact_preview_signature != app._exact_preview_state_signature():
        raise AssertionError("Manual stroke did not promote its local result to exact state")
    app.base_exposure_tenths.set(0)

    candidate = Stroke([(0.74, 0.12), (0.80, 0.15)], 24, 9, auto_score=73)
    app.candidates[key].append(candidate)
    x0, y0, x1, y1 = app.display_box
    cx, cy = round(x0 + 0.77 * (x1 - x0)), round(y0 + 0.135 * (y1 - y0))
    app.canvas.event_generate("<Motion>", x=cx, y=cy)
    app.update()
    button_box = app.canvas.bbox("candidate_pick")
    if button_box is None:
        raise AssertionError("Candidate hover did not expose the real pick button")
    bx, by = (button_box[0] + button_box[2]) // 2, (button_box[1] + button_box[3]) // 2
    pick_started = time.perf_counter()
    app.canvas.event_generate("<ButtonPress-1>", x=bx, y=by)
    app.canvas.event_generate("<ButtonRelease-1>", x=bx, y=by)
    app.update()
    pick_elapsed = time.perf_counter() - pick_started
    print(f"Candidate pick response: {pick_elapsed:.3f}s", flush=True)
    deadline = time.monotonic() + 1.4
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.01)
    app._invalidate_global_preview = real_mask_add_invalidate
    app._schedule_global_exact_validation = real_mask_add_validation
    if candidate not in app.strokes[key] or not candidate.locked:
        raise AssertionError("Candidate click did not add and lock its mask")
    if mask_add_global_calls or mask_add_validation_calls:
        raise AssertionError(
            f"Mask addition triggered global work: invalidate={mask_add_global_calls}, "
            f"validation={mask_add_validation_calls}"
        )
    if app.global_preview_signature != app._global_preview_state_signature():
        raise AssertionError(
            "Candidate mask addition did not retain exact local cache: "
            + str(getattr(app, "last_incremental_delete_error", None))
        )
    app.strokes[key] = [item for item in app.strokes[key] if item is not candidate]
    app.candidates[key] = [item for item in app.candidates[key] if item is not candidate]
    app.global_preview_rgb, _candidate_cleanup_mask = compose_meteor_objects(
        source, base, app.strokes[key], False, False, 15, 25,
        "自然融合", True, 100, 70,
    )
    app.global_labeled_preview_rgb = app.global_preview_rgb.copy()
    app.global_preview_signature = app._global_preview_state_signature()
    app.output_mode.set("separate")
    app.view_mode.set("labeled")
    app._render_preview()
    app.update()

    geometry = app._object_canvas_geometry(selected)
    end = geometry["handles"]["length_end"]
    axis = geometry["axis"]
    old_length_scale = moved.length_scale
    app.canvas.event_generate("<ButtonPress-1>", x=int(end[0]), y=int(end[1]))
    app.canvas.event_generate(
        "<B1-Motion>", x=int(end[0] + axis[0] * 35), y=int(end[1] + axis[1] * 35), state=0x0100
    )
    app.canvas.event_generate(
        "<ButtonRelease-1>", x=int(end[0] + axis[0] * 35), y=int(end[1] + axis[1] * 35)
    )
    app.update()
    if app.strokes[key][0].length_scale <= old_length_scale:
        raise AssertionError("Length handle did not stretch meteor")

    geometry = app._object_canvas_geometry(selected)
    side = geometry["handles"]["width_pos"]
    normal = geometry["normal"]
    old_width_scale = app.strokes[key][0].width_scale
    app.canvas.event_generate("<ButtonPress-1>", x=int(side[0]), y=int(side[1]))
    app.canvas.event_generate(
        "<B1-Motion>", x=int(side[0] + normal[0] * 25), y=int(side[1] + normal[1] * 25), state=0x0100
    )
    app.canvas.event_generate(
        "<ButtonRelease-1>", x=int(side[0] + normal[0] * 25), y=int(side[1] + normal[1] * 25)
    )
    app.update()
    if app.strokes[key][0].width_scale <= old_width_scale:
        raise AssertionError("Width handle did not stretch meteor")

    geometry = app._object_canvas_geometry(selected)
    rotate = geometry["handles"]["rotate"]
    center = geometry["center"]
    vector = rotate - center
    angle = np.deg2rad(24.0)
    rotated = center + np.asarray([
        vector[0] * np.cos(angle) - vector[1] * np.sin(angle),
        vector[0] * np.sin(angle) + vector[1] * np.cos(angle),
    ])
    old_rotation = app.strokes[key][0].rotation
    app.canvas.event_generate("<ButtonPress-1>", x=int(rotate[0]), y=int(rotate[1]))
    app.canvas.event_generate("<B1-Motion>", x=int(rotated[0]), y=int(rotated[1]), state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=int(rotated[0]), y=int(rotated[1]))
    app.update()
    if abs(app.strokes[key][0].rotation - old_rotation) < 10.0:
        raise AssertionError("Rotation handle did not rotate meteor")

    app.selected_object = selected
    app._load_selected_object_adjustments()
    # Toggling one meteor's independent controls must remain an ROI update. It
    # must not launch a fast full rebuild followed by a second exact rebuild.
    app.exact_preview_full_rgb = app.global_preview_rgb.copy()
    app.exact_labeled_preview_full_rgb = app.global_preview_rgb.copy()
    app.exact_preview_signature = app._exact_preview_state_signature()
    full_override_rebuilds = []
    real_request_override_global = app._request_global_preview
    real_begin_override_exact = app._begin_exact_preview
    app._request_global_preview = lambda signature: full_override_rebuilds.append(("global", signature))
    app._begin_exact_preview = lambda signature, open_when_ready=False: full_override_rebuilds.append(("exact", signature))
    app.selected_override_enabled.set(True)
    app._selected_override_changed()
    app.update()
    if full_override_rebuilds:
        raise AssertionError(
            f"Per-meteor override launched full rebuilds: {len(full_override_rebuilds)}; "
            f"incremental_error={app.last_incremental_delete_error}"
        )
    if app.exact_preview_signature != app._exact_preview_state_signature():
        raise AssertionError("Per-meteor override did not commit the local exact-view state")
    app._request_global_preview = real_request_override_global
    app._begin_exact_preview = real_begin_override_exact
    # Reproduce a busy total composite containing an overlapping source whose
    # layer cache is unavailable. Per-meteor sliders must still use the selected
    # object's before/after pixel delta and never rebuild every source.
    overlap_key = str(Path("evicted_overlapping_source.tif"))
    app.output_mode.set("combined")
    app.view_mode.set("blend")
    app._render_preview()
    app.update()
    slider_canvas_item = app.preview_image_item
    app.pairs[overlap_key] = Path("evicted_overlapping_base.tif")
    app.strokes[overlap_key] = [replace(app.strokes[key][0], points=app.strokes[key][0].points.copy())]
    slider_full_rebuilds = []
    slider_invalidations = 0
    real_slider_request = app._request_global_preview
    real_slider_exact = app._begin_exact_preview
    real_slider_invalidate = app._invalidate_global_preview
    real_slider_render = app._render_preview
    real_slider_auto_exact = app._schedule_automatic_exact_preview
    real_slider_validation = app._schedule_global_exact_validation
    slider_render_calls = 0
    def count_slider_invalidate():
        nonlocal slider_invalidations
        slider_invalidations += 1
        return real_slider_invalidate()
    def count_slider_render():
        nonlocal slider_render_calls
        slider_render_calls += 1
        return real_slider_render()
    app._request_global_preview = lambda signature: slider_full_rebuilds.append(("global", signature))
    app._begin_exact_preview = lambda signature, open_when_ready=False: slider_full_rebuilds.append(("exact", signature))
    app._invalidate_global_preview = count_slider_invalidate
    app._render_preview = count_slider_render
    app._schedule_automatic_exact_preview = lambda: slider_full_rebuilds.append(("auto_exact", ""))
    app._schedule_global_exact_validation = lambda signature: slider_full_rebuilds.append(("validation", signature))
    delayed_fires = []
    app.global_preview_request_after_id = app.after(
        360, lambda: delayed_fires.append("global")
    )
    app.exact_preview_request_after_id = app.after(
        430, lambda: delayed_fires.append("exact")
    )
    app.global_exact_after_id = app.after(
        1220, lambda: delayed_fires.append("validation")
    )
    app.selected_brightness.set(137)
    app.selected_cleanup.set(84)
    app.selected_saturation.set(112)
    app.selected_match.set(True)
    app.selected_feather.set(19)
    app._selected_adjustment_changed()
    # Cover both delayed pipelines (automatic exact at 320 ms and the former
    # global validation at 1200 ms). The previous immediate-only test restored
    # these spies before the user-visible duplicate jobs could begin.
    deadline = time.monotonic() + 1.45
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    app._request_global_preview = real_slider_request
    app._begin_exact_preview = real_slider_exact
    app._invalidate_global_preview = real_slider_invalidate
    app._render_preview = real_slider_render
    app._schedule_automatic_exact_preview = real_slider_auto_exact
    app._schedule_global_exact_validation = real_slider_validation
    app.pairs.pop(overlap_key, None)
    app.strokes.pop(overlap_key, None)
    app.output_mode.set("separate")
    if slider_invalidations or slider_full_rebuilds:
        raise AssertionError(
            f"Per-meteor slider rebuilt the composite: invalidations={slider_invalidations}, "
            f"workers={slider_full_rebuilds}"
        )
    if delayed_fires:
        raise AssertionError(
            f"Per-meteor slider did not cancel deferred full-frame jobs: {delayed_fires}"
        )
    if any((
        app.global_preview_request_after_id,
        app.exact_preview_request_after_id,
        app.global_exact_after_id,
    )):
        raise AssertionError("Per-meteor slider left a delayed full-preview job queued")
    if slider_render_calls or app.preview_image_item != slider_canvas_item:
        raise AssertionError(
            f"Per-meteor slider redrew the viewport instead of pasting its ROI: "
            f"renders={slider_render_calls}"
        )
    adjusted = app.strokes[key][0]
    if (
        adjusted.brightness_override != 137
        or adjusted.background_cleanup_override != 84
        or adjusted.saturation_override != 112
        or adjusted.match_exposure_override is not True
        or adjusted.feather != 19
    ):
        raise AssertionError("Per-meteor adjustments were not stored")
    candidate = app.candidates[key][0]
    if candidate.brightness_override != 137 or candidate.background_cleanup_override != 84:
        raise AssertionError("Per-meteor candidate adjustments were not synchronized")

    adjusted.auto_blend_enabled = True
    adjusted.auto_strength = "强力"
    adjusted.auto_black_point = 1.75
    adjusted.auto_cleanup = 91.0
    adjusted.auto_brightness = 108.0
    adjusted.auto_feather = 26

    project_data = json.loads(json.dumps(app._project_data()))
    app._apply_project_data(project_data)
    restored = app.strokes[key][0]
    if (
        restored.brightness_override != 137
        or restored.background_cleanup_override != 84
        or restored.saturation_override != 112
        or restored.match_exposure_override is not True
        or restored.feather != 19
        or restored.auto_strength != "强力"
        or restored.auto_black_point != 1.75
        or restored.auto_cleanup != 91.0
        or restored.auto_brightness != 108.0
        or restored.auto_feather != 26
    ):
        raise AssertionError("Per-meteor adjustments did not survive project reload")
    app.current_path = Path(key)
    app.selected_object = selected
    app.preview_source = source
    app.preview_base = base
    app.current_dims = (width, height)

    transformed_before_reset = replace(restored, points=restored.points.copy())
    original_midpoint = np.mean(
        np.asarray([(x * (width - 1), y * (height - 1)) for x, y in restored.points]), axis=0
    )
    transformed_points = np.asarray([
        (x * (width - 1), y * (height - 1))
        for x, y in transformed_stroke_points(restored, width, height)
    ])
    transformed_midpoint = transformed_points.mean(axis=0)
    app.view_mode.set("source")
    app.show_mask.set(True)
    app._render_preview()
    app.update()
    stable_canvas_image = app.preview_image_item
    app._render_preview()
    app.update()
    if app.preview_image_item != stable_canvas_image:
        raise AssertionError("A same-size local edit recreated the full canvas image")
    tx, ty = int(round(transformed_midpoint[0])), int(round(transformed_midpoint[1]))
    ox, oy = int(round(original_midpoint[0])), int(round(original_midpoint[1]))
    if int(app.preview_rgb[max(0, ty - 8):ty + 9, max(0, tx - 8):tx + 9].max()) < 120:
        raise AssertionError("Source view did not show the transformed meteor copy")
    if int(app.preview_rgb[max(0, oy - 8):oy + 9, max(0, ox - 8):ox + 9].max()) < 120:
        raise AssertionError("Source view did not preserve the meteor at its original position")
    if not app.canvas.find_withtag("transform_reference"):
        raise AssertionError("Source view did not show original/transformed reference guides")

    app.view_mode.set("labeled")
    app.selected_object = selected
    app._reset_selected_object()
    app.update()
    reset = app.strokes[key][0]
    if any((reset.offset_x, reset.offset_y, reset.rotation)) or reset.length_scale != 1 or reset.width_scale != 1:
        raise AssertionError("One-click restore did not clear geometric transforms")
    if reset.brightness_override != transformed_before_reset.brightness_override or reset.feather != transformed_before_reset.feather:
        raise AssertionError("One-click restore changed mask or per-meteor adjustments")
    app.undo_stroke()
    app.update()
    if app.strokes[key][0].offset_x != transformed_before_reset.offset_x:
        raise AssertionError("Undo did not restore the transform after one-click reset")
    app.selected_object = selected

    render_calls = 0
    real_render_preview = app._render_preview
    def count_render_calls():
        nonlocal render_calls
        render_calls += 1
    app._render_preview = count_render_calls
    app._load_selected_object_adjustments()
    app.update_idletasks()
    app._render_preview = real_render_preview
    if render_calls:
        raise AssertionError("Loading selected controls triggered a render loop")

    # Seed the exact shared-preview cache as it exists when the user deletes
    # from the visible final composite.
    # Give this regression its own non-overlapping meteor so another accumulated
    # brush stroke cannot legitimately keep the same source pixels visible.
    cv2.line(source, (1010, 700), (1140, 635), (250, 250, 255), 10, cv2.LINE_AA)
    history_meteor = Stroke(
        [(1010 / width, 700 / height), (1140 / width, 635 / height)],
        30, 11, locked=True, auto_score=97,
    )
    history_candidate = replace(history_meteor, points=history_meteor.points.copy())
    app.strokes[key].append(history_meteor)
    app.candidates[key].append(history_candidate)
    app.selected_object = (key, len(app.strokes[key]) - 1)
    delete_cached, _delete_mask = compose_meteor_objects(
        source, base, app.strokes[key], *app._object_composite_settings(key),
    )
    app.output_mode.set("combined")
    app.global_preview_rgb = delete_cached.copy()
    app.global_labeled_preview_rgb = delete_cached.copy()
    app.global_preview_signature = app._global_preview_state_signature()
    count = len(app.strokes[key])
    candidate_count = len(app.candidates[key])
    full_rebuild_calls = 0
    exact_validation_calls = 0
    history_worker_calls = []
    real_invalidate = app._invalidate_global_preview
    real_exact_validation = app._schedule_global_exact_validation
    real_request_global = app._request_global_preview
    real_auto_exact = app._schedule_automatic_exact_preview
    real_begin_exact = app._begin_exact_preview
    real_scheduler_submit = app.background_tasks.submit
    def count_full_rebuild():
        nonlocal full_rebuild_calls
        full_rebuild_calls += 1
        return real_invalidate()
    def count_exact_validation(_signature):
        nonlocal exact_validation_calls
        exact_validation_calls += 1
    def count_history_submit(channel, _function, **_kwargs):
        if channel in {"global_preview_task", "exact_preview_task"}:
            history_worker_calls.append(channel)
            return None
        return real_scheduler_submit(channel, _function, **_kwargs)
    def click_history_button(button):
        app.update_idletasks()
        x = max(1, button.winfo_width() // 2)
        y = max(1, button.winfo_height() // 2)
        button.event_generate("<Enter>", x=x, y=y)
        button.event_generate("<ButtonPress-1>", x=x, y=y)
        button.event_generate("<ButtonRelease-1>", x=x, y=y)
        app.update()
    app._invalidate_global_preview = count_full_rebuild
    app._schedule_global_exact_validation = count_exact_validation
    app._request_global_preview = lambda signature: history_worker_calls.append(
        ("request_global", signature)
    )
    app._schedule_automatic_exact_preview = lambda: history_worker_calls.append(
        ("schedule_exact", "")
    )
    app._begin_exact_preview = lambda signature, open_when_ready=False: history_worker_calls.append(
        ("begin_exact", signature, open_when_ready)
    )
    app.background_tasks.submit = count_history_submit
    app._delete_selected_object()
    deleted_pixels = app.global_preview_rgb.copy()
    if len(app.strokes[key]) != count - 1 or len(app.candidates[key]) != candidate_count - 1:
        raise AssertionError("Delete did not remove the locked candidate object")
    if full_rebuild_calls or exact_validation_calls:
        raise AssertionError(
            f"Isolated delete scheduled a global rebuild: "
            f"invalidate={full_rebuild_calls}, validation={exact_validation_calls}; "
            f"error={getattr(app, 'last_incremental_delete_error', None)}"
        )
    if app.global_preview_signature != app._global_preview_state_signature():
        raise AssertionError("Incremental delete did not commit the new composite state")
    if np.array_equal(deleted_pixels, delete_cached):
        raise AssertionError("Delete changed mask metadata but not the meteor pixels")

    # Exercise the actual visible buttons. Undo must restore both the candidate
    # metadata and its pixels; redo must remove both again. Keep every full-frame
    # entry point monitored beyond the old 320/1200 ms deferred windows.
    click_history_button(app.undo_button)
    if len(app.strokes[key]) != count or len(app.candidates[key]) != candidate_count:
        raise AssertionError("Undo did not restore object and candidate metadata")
    undo_difference = np.abs(
        app.global_preview_rgb.astype(np.int16) - delete_cached.astype(np.int16)
    )
    if int(undo_difference.max()) > 1:
        max_y, max_x, max_channel = np.unravel_index(
            int(np.argmax(undo_difference)), undo_difference.shape
        )
        raise AssertionError(
            f"Undo restored the mask but not its meteor pixels (max delta {undo_difference.max()} "
            f"at {(max_x, max_y, max_channel)}, dirty={app.last_incremental_box})"
        )
    click_history_button(app.redo_button)
    if len(app.strokes[key]) != count - 1 or len(app.candidates[key]) != candidate_count - 1:
        raise AssertionError("Redo did not remove object and candidate metadata")
    redo_difference = np.abs(
        app.global_preview_rgb.astype(np.int16) - deleted_pixels.astype(np.int16)
    )
    if int(redo_difference.max()) > 1:
        raise AssertionError(
            f"Redo changed the mask but did not remove its meteor pixels (max delta {redo_difference.max()})"
        )
    click_history_button(app.undo_button)
    deadline = time.monotonic() + 1.45
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    app._invalidate_global_preview = real_invalidate
    app._schedule_global_exact_validation = real_exact_validation
    app._request_global_preview = real_request_global
    app._schedule_automatic_exact_preview = real_auto_exact
    app._begin_exact_preview = real_begin_exact
    app.background_tasks.submit = real_scheduler_submit
    if full_rebuild_calls or exact_validation_calls or history_worker_calls:
        raise AssertionError(
            "Undo/redo launched full-frame work: "
            f"invalidate={full_rebuild_calls}, validation={exact_validation_calls}, "
            f"workers={history_worker_calls}"
        )
    if any((
        app.global_preview_request_after_id,
        app.exact_preview_request_after_id,
        app.global_exact_after_id,
    )):
        raise AssertionError("Undo/redo left a delayed full-preview job queued")

    app.strokes[key] = [item for item in app.strokes[key] if item is not history_meteor]
    app.candidates[key] = [item for item in app.candidates[key] if item is not history_candidate]
    delete_cached, _delete_mask = compose_meteor_objects(
        source, base, app.strokes[key], *app._object_composite_settings(key),
    )

    # Deletion must also remain local when the fast global cache was evicted but
    # the exact canvas currently visible to the user is still available.
    app.selected_object = selected
    app.global_preview_rgb = None
    app.exact_preview_full_rgb = delete_cached.copy()
    exact_cache_delete_global_calls = 0
    real_exact_cache_invalidate = app._invalidate_global_preview
    def count_exact_cache_delete_global():
        nonlocal exact_cache_delete_global_calls
        exact_cache_delete_global_calls += 1
        return real_exact_cache_invalidate()
    app._invalidate_global_preview = count_exact_cache_delete_global
    app._delete_selected_object()
    app._invalidate_global_preview = real_exact_cache_invalidate
    if exact_cache_delete_global_calls:
        raise AssertionError("Delete ignored the visible exact cache and rebuilt the full composite")
    app.undo_stroke()

    app.view_mode.set("source")
    app._render_preview()
    app.update()

    def click_and_assert_viewport(widget, x: int, y: int, label: str) -> None:
        before = (
            round(float(app.canvas_zoom), 12),
            round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
            round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
            tuple(round(float(value), 4) for value in app.display_box),
            app.canvas.winfo_width(), app.canvas.winfo_height(),
            app.canvas_image_shape, app.preview_request_id,
        )
        widget.event_generate("<ButtonPress-1>", x=x, y=y)
        widget.event_generate("<ButtonRelease-1>", x=x, y=y)
        deadline = time.monotonic() + 1.45
        while time.monotonic() < deadline:
            app.update_idletasks()
            app.update()
            time.sleep(0.01)
        after = (
            round(float(app.canvas_zoom), 12),
            round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
            round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
            tuple(round(float(value), 4) for value in app.display_box),
            app.canvas.winfo_width(), app.canvas.winfo_height(),
            app.canvas_image_shape, app.preview_request_id,
        )
        if after != before:
            raise AssertionError(f"{label} changed the viewport: {before} -> {after}")

    # Exact user-reported point 1: the empty header strip immediately to the
    # right of "流星合成工作区", before the right-aligned workspace buttons.
    title_right = app.workspace_title_label.winfo_rootx() + app.workspace_title_label.winfo_width()
    controls_left = app.paths_toggle_button.winfo_rootx()
    if controls_left - title_right < 6:
        raise AssertionError("No real header blank space exists at the reported coordinate")
    header_root_x = (title_right + controls_left) // 2
    header_root_y = app.header_panel.winfo_rooty() + app.header_panel.winfo_height() // 2
    header_target = app.header_panel
    header_local_x = header_root_x - header_target.winfo_rootx()
    header_local_y = header_root_y - header_target.winfo_rooty()
    for child in header_target.winfo_children():
        if (
            child.winfo_x() <= header_local_x < child.winfo_x() + child.winfo_width()
            and child.winfo_y() <= header_local_y < child.winfo_y() + child.winfo_height()
        ):
            raise AssertionError(f"Reported header blank coordinate overlaps {child}")
    click_and_assert_viewport(
        header_target,
        header_local_x,
        header_local_y,
        "Clicking blank space beside 流星合成工作区",
    )

    # Exact user-reported point 2: blank padding directly below 载入项目.
    load_center_x = app.load_project_button.winfo_rootx() + app.load_project_button.winfo_width() // 2
    load_bottom = app.load_project_button.winfo_rooty() + app.load_project_button.winfo_height()
    tools_bottom = app.mask_tools_tab.winfo_rooty() + app.mask_tools_tab.winfo_height()
    lower_root_y = min(tools_bottom - 2, load_bottom + max(1, (tools_bottom - load_bottom) // 2))
    lower_target = app.mask_tools_tab
    if lower_root_y <= load_bottom:
        raise AssertionError("No real blank padding exists below 载入项目")
    click_and_assert_viewport(
        lower_target,
        load_center_x - lower_target.winfo_rootx(),
        lower_root_y - lower_target.winfo_rooty(),
        "Clicking blank space below 载入项目",
    )
    app.state("normal")
    app.geometry("1280x820+10000+10000")
    app.update_idletasks()
    app.update()

    # A preview worker may replace a quick frame with the full-resolution
    # version shortly after an otherwise unrelated click.  In fit mode that
    # replacement must preserve the *visible* field of view; keeping the same
    # numeric pixels-to-screen zoom would make the photograph suddenly grow.
    full_frame = app.preview_rgb.copy()
    quick_frame = cv2.resize(
        full_frame,
        (max(2, full_frame.shape[1] // 2), max(2, full_frame.shape[0] // 2)),
        interpolation=cv2.INTER_AREA,
    )
    app._present_preview_image(quick_frame, False, True)
    app._canvas_fit()
    quick_box = tuple(float(value) for value in app.display_box)
    app._present_preview_image(full_frame, False, True)
    app.update()
    full_box = tuple(float(value) for value in app.display_box)
    if any(abs(current - expected) > 1.5 for current, expected in zip(full_box, quick_box)):
        raise AssertionError(
            f"Full-resolution preview replacement changed visible fit: {quick_box} -> {full_box}"
        )

    app._canvas_fit()
    main_fit_zoom = app.canvas_zoom
    app.control_notebook.select(app.selected_tools_tab)
    app.update()
    if abs(app.canvas_zoom - main_fit_zoom) > 1e-9:
        raise AssertionError("Clicking a non-image panel changed fit-view zoom")
    app.control_notebook.select(app.mask_tools_tab)
    app.update()
    if abs(app.canvas_zoom - main_fit_zoom) > 1e-9:
        raise AssertionError("Returning to a non-image panel changed fit-view zoom")
    blank_x = max(1, app.mask_tools_tab.winfo_width() - 3)
    blank_y = max(1, app.mask_tools_tab.winfo_height() - 3)
    root_x = app.mask_tools_tab.winfo_rootx() + blank_x
    root_y = app.mask_tools_tab.winfo_rooty() + blank_y
    click_target = app.winfo_containing(root_x, root_y) or app.mask_tools_tab
    local_x = root_x - click_target.winfo_rootx()
    local_y = root_y - click_target.winfo_rooty()
    before_blank_click = (
        app.canvas_fit_mode,
        tuple(round(float(value), 4) for value in app.display_box),
        app.canvas.winfo_width(), app.canvas.winfo_height(),
        round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
        round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
    )
    click_target.event_generate("<ButtonPress-1>", x=local_x, y=local_y)
    click_target.event_generate("<ButtonRelease-1>", x=local_x, y=local_y)
    # Let focus, Configure and delayed preview callbacks run.  The old test only
    # sampled immediately and missed the user-visible jump after the click.
    deadline = time.monotonic() + 1.45
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    after_blank_click = (
        app.canvas_fit_mode,
        tuple(round(float(value), 4) for value in app.display_box),
        app.canvas.winfo_width(), app.canvas.winfo_height(),
        round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
        round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
    )
    if after_blank_click != before_blank_click:
        raise AssertionError(
            f"Blank panel click changed visible viewport: {before_blank_click} -> {after_blank_click}"
        )
    if abs(app.canvas_zoom - main_fit_zoom) > 1e-9:
        raise AssertionError("Clicking blank space inside the bottom panel changed zoom")
    # Panel layout changes may alter how much of the photograph is visible, but
    # they are not preview zoom controls and must never rewrite pixel scale.
    before_paths_toggle_zoom = float(app.canvas_zoom)
    before_paths_toggle_allocation = (app.canvas.winfo_width(), app.canvas.winfo_height())
    app._toggle_paths_panel()
    app.update_idletasks()
    app.update()
    after_paths_toggle_allocation = (app.canvas.winfo_width(), app.canvas.winfo_height())
    if after_paths_toggle_allocation == before_paths_toggle_allocation:
        raise AssertionError("Path-panel toggle did not exercise a real canvas allocation change")
    if abs(app.canvas_zoom - before_paths_toggle_zoom) > 1e-9:
        raise AssertionError(
            "Opening the upper material panel changed preview zoom without using a preview control: "
            f"{before_paths_toggle_zoom} -> {app.canvas_zoom}"
        )
    upper_x = max(1, app.paths_panel.winfo_width() - 4)
    upper_y = max(1, app.paths_panel.winfo_height() - 4)
    upper_root_x = app.paths_panel.winfo_rootx() + upper_x
    upper_root_y = app.paths_panel.winfo_rooty() + upper_y
    upper_target = app.winfo_containing(upper_root_x, upper_root_y) or app.paths_panel
    upper_local_x = upper_root_x - upper_target.winfo_rootx()
    upper_local_y = upper_root_y - upper_target.winfo_rooty()
    before_upper_click = (
        round(float(app.canvas_zoom), 12),
        round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
        round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
        tuple(round(float(value), 4) for value in app.display_box),
        app.canvas.winfo_width(), app.canvas.winfo_height(),
        app.preview_request_id,
    )
    upper_target.event_generate("<ButtonPress-1>", x=upper_local_x, y=upper_local_y)
    upper_target.event_generate("<ButtonRelease-1>", x=upper_local_x, y=upper_local_y)
    deadline = time.monotonic() + 1.45
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    after_upper_click = (
        round(float(app.canvas_zoom), 12),
        round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
        round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
        tuple(round(float(value), 4) for value in app.display_box),
        app.canvas.winfo_width(), app.canvas.winfo_height(),
        app.preview_request_id,
    )
    if after_upper_click != before_upper_click:
        raise AssertionError(
            f"Clicking upper-panel blank space changed the viewport: "
            f"{before_upper_click} -> {after_upper_click}"
        )
    # Reproduce the reported layout with the TIFF/material path panel expanded,
    # one selected file row, and blank Treeview space below it.
    app._canvas_fit()
    app.tree.unbind("<<TreeviewSelect>>")
    app.tree.delete(*app.tree.get_children())
    app.tree.insert("", "end", iid="0", text="synthetic_meteor.tif", values=("已锁定",))
    app.tree.selection_set("0")
    app.update_idletasks()
    app.update()
    app.tree.bind("<<TreeviewSelect>>", app._tree_selection_changed)
    app.update_idletasks()
    tree_x = max(2, app.tree.winfo_width() // 2)
    tree_y = max(2, app.tree.winfo_height() - 4)
    if app.tree.identify_row(tree_y):
        raise AssertionError("TIFF material-list test coordinate unexpectedly hits a file row")
    before_tree_blank = (
        app.canvas_fit_mode,
        round(float(app.canvas_zoom), 12),
        tuple(round(float(value), 4) for value in app.display_box),
        app.canvas.winfo_width(), app.canvas.winfo_height(),
        round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
        round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
        app.canvas_image_shape,
    )
    before_tree_request = app.preview_request_id
    app.tree.event_generate("<ButtonPress-1>", x=tree_x, y=tree_y)
    app.tree.event_generate("<ButtonRelease-1>", x=tree_x, y=tree_y)
    deadline = time.monotonic() + 1.45
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    after_tree_blank = (
        app.canvas_fit_mode,
        round(float(app.canvas_zoom), 12),
        tuple(round(float(value), 4) for value in app.display_box),
        app.canvas.winfo_width(), app.canvas.winfo_height(),
        round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
        round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
        app.canvas_image_shape,
    )
    if after_tree_blank != before_tree_blank:
        raise AssertionError(
            "Clicking blank TIFF material-list space changed the visible viewport: "
            f"{before_tree_blank} -> {after_tree_blank}"
        )
    if app.tree.selection() != ("0",):
        raise AssertionError("Blank TIFF-list click cleared the current file selection")
    if app.preview_request_id != before_tree_request or app.preview_selection_after_id is not None:
        raise AssertionError("Blank TIFF-list click enqueued a redundant preview load")
    app._toggle_paths_panel()
    app.update_idletasks()
    app.update()
    app._canvas_fit()
    main_fit_zoom = app.canvas_zoom
    x0, y0, x1, y1 = app.display_box
    canvas_w, canvas_h = app.canvas.winfo_width(), app.canvas.winfo_height()
    blank_canvas_points = [
        (2, 2), (max(2, canvas_w - 3), 2),
        (2, max(2, canvas_h - 3)), (max(2, canvas_w - 3), max(2, canvas_h - 3)),
    ]
    outside = next(
        ((x, y) for x, y in blank_canvas_points if not (x0 <= x <= x1 and y0 <= y <= y1)),
        None,
    )
    if outside is not None:
        app.canvas.event_generate("<ButtonPress-1>", x=outside[0], y=outside[1])
        app.canvas.event_generate("<ButtonRelease-1>", x=outside[0], y=outside[1])
        app.status.set("空白区域点击不应改变图像缩放")
        app.update()
        app._canvas_configure()
        app._canvas_configure()
        if abs(app.canvas_zoom - main_fit_zoom) > 1e-9:
            raise AssertionError("Clicking outside the displayed photograph changed zoom")
    main_center = (app.canvas_center_x, app.canvas_center_y)
    zoom_started = time.perf_counter()
    app._canvas_zoom_by(1.25, (app.canvas.winfo_width() // 3, app.canvas.winfo_height() // 3))
    app.update_idletasks()
    app.update()
    zoom_elapsed = time.perf_counter() - zoom_started
    if app.canvas_zoom <= main_fit_zoom or app.preview_photo is None:
        raise AssertionError("Main mask canvas did not zoom without recompositing")
    if app.last_viewport_render_source != "cached-ancestor":
        raise AssertionError(
            f"Interactive zoom reread the source instead of reusing the visible viewport: "
            f"{app.last_viewport_render_source}"
        )
    if zoom_elapsed > 0.20:
        raise AssertionError(f"Interactive zoom frame was not immediate: {zoom_elapsed:.3f}s")
    pan_event = type("PanEvent", (), {"x": 300, "y": 260})()
    app._canvas_pan_start_event(pan_event)
    pan_event.x, pan_event.y = 350, 290
    app._canvas_pan_move_event(pan_event)
    app._canvas_pan_end_event(pan_event)
    if (app.canvas_center_x, app.canvas_center_y) == main_center:
        raise AssertionError("Main mask canvas did not pan")
    switched_zoom = app.canvas_zoom
    switched_center = (app.canvas_center_x, app.canvas_center_y)
    for mode in ("base", "blend", "labeled", "source"):
        app.view_mode.set(mode)
        app._render_preview()
        app.update()
        if abs(app.canvas_zoom - switched_zoom) > 1e-9:
            raise AssertionError(f"View switch changed canvas zoom in {mode} mode")
        if any(
            abs(current - expected) > 1e-6
            for current, expected in zip(
                (app.canvas_center_x, app.canvas_center_y), switched_center
            )
        ):
            raise AssertionError(f"View switch changed canvas center in {mode} mode")
    app._canvas_actual_size()

    exact_viewer = ExactPreviewViewer(app, source, source.copy(), "blend")
    exact_viewer.geometry("900x620+10000+10000")
    exact_viewer.update()
    exact_viewer.fit()
    fit_zoom = exact_viewer.zoom
    exact_viewer.geometry("1000x700+10000+10000")
    exact_viewer.update()
    resized_fit_zoom = exact_viewer.zoom
    if not exact_viewer.fit_mode or resized_fit_zoom <= fit_zoom:
        raise AssertionError("Exact-preview fit mode did not follow a real window resize")
    exact_viewer.actual_size()
    exact_viewer.center_x = source.shape[1] / 2.0
    exact_viewer.center_y = source.shape[0] / 2.0
    exact_viewer._render()
    reusable_photo = exact_viewer.photo
    exact_viewer.center_x += 12
    exact_viewer._render()
    if exact_viewer.photo is not reusable_photo:
        raise AssertionError("Exact-preview pan recreated an unchanged-size viewport image")
    exact_viewer._zoom_by(1.25)
    exact_viewer.mode.set("labeled")
    exact_viewer._render()
    exact_viewer.update()
    if fit_zoom >= 1.0 or exact_viewer.zoom <= 1.0 or exact_viewer.photo is None:
        raise AssertionError("Full-resolution viewer did not fit, zoom, and render")
    exact_viewer.destroy()

    # Exercise the visible undo/redo controls and select a version through the
    # real Treeview row bindings. A new canvas edit after that jump must discard
    # the abandoned redo branch.
    app.view_mode.set("source")
    app._view_mode_changed()
    app.update()
    history_key = str(app.current_path)
    app.strokes[history_key] = []
    app.edit_history[history_key] = []
    app.edit_redo.pop(history_key, None)
    history_first = Stroke([(0.20, 0.30), (0.25, 0.34)], 12, 4)
    history_second = Stroke([(0.35, 0.42), (0.40, 0.46)], 12, 4)
    app.strokes[history_key].append(history_first)
    app._record_edit(history_key, ("add", 0, history_first))
    app.strokes[history_key].append(history_second)
    app._record_edit(history_key, ("add", 1, history_second))
    app.update_idletasks()
    history_button_x = max(1, app.history_button.winfo_width() // 2)
    history_button_y = max(1, app.history_button.winfo_height() // 2)
    app.history_button.event_generate("<Enter>", x=history_button_x, y=history_button_y)
    app.history_button.event_generate(
        "<ButtonPress-1>", x=history_button_x, y=history_button_y
    )
    app.history_button.event_generate(
        "<ButtonRelease-1>", x=history_button_x, y=history_button_y
    )
    app.update()
    if app.control_notebook.select() != str(app.history_tools_tab):
        raise AssertionError("Visible history button did not open the operation-history panel")
    app._refresh_history_ui(history_key)
    app.update_idletasks()
    app.update()
    if str(app.undo_button.cget("state")) == "disabled" or len(app.history_tree.get_children()) != 3:
        raise AssertionError("Visible history controls did not expose the current two-step timeline")
    row_box = app.history_tree.bbox("1")
    if not row_box:
        raise AssertionError("History version row is not visible/clickable")
    row_x = row_box[0] + max(2, row_box[2] // 2)
    row_y = row_box[1] + max(2, row_box[3] // 2)
    app.history_tree.event_generate("<ButtonPress-1>", x=row_x, y=row_y)
    app.history_tree.event_generate("<ButtonRelease-1>", x=row_x, y=row_y)
    app.update_idletasks()
    app.update()
    if len(app.strokes[history_key]) != 1 or len(app.edit_redo.get(history_key, [])) != 1:
        raise AssertionError("Clicking a history version did not restore that version")

    app.control_notebook.select(app.mask_tools_tab)
    app._clear_candidate_hover()
    app.update()
    x0, y0, x1, y1 = app.display_box
    draw_x = int(round(x0 + (x1 - x0) * 0.62))
    draw_y = int(round(y0 + (y1 - y0) * 0.58))
    app.canvas.event_generate("<ButtonPress-1>", x=draw_x, y=draw_y)
    app.canvas.event_generate("<B1-Motion>", x=draw_x + 22, y=draw_y + 10, state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=draw_x + 22, y=draw_y + 10)
    app.update_idletasks()
    app.update()
    if app.edit_redo.get(history_key):
        raise AssertionError("A new edit after history navigation kept the abandoned future branch")
    branch_count = len(app.strokes[history_key])
    app.undo_button.invoke()
    app.update()
    if len(app.strokes[history_key]) != branch_count - 1 or str(app.redo_button.cget("state")) == "disabled":
        raise AssertionError("Visible undo button did not move backward or enable redo")
    app.redo_button.invoke()
    app.update()
    if len(app.strokes[history_key]) != branch_count:
        raise AssertionError("Visible redo button did not restore the undone edit")

    app.open_video_workspace()
    app.update()
    if app.state() != "withdrawn" or app.video_window is None:
        raise AssertionError("Opening video workspace did not replace the main workspace")
    video_scheduler = app.video_window.background_tasks
    app.video_window.destroy()
    app.update()
    if not video_scheduler.closed:
        raise AssertionError("Closing video workspace left its background scheduler running")
    if app.state() == "withdrawn" or app.video_window is not None:
        raise AssertionError("Closing video workspace did not restore the main workspace")

    app.open_alignment_workspace()
    app.update()
    if app.state() != "withdrawn" or app.alignment_window is None:
        raise AssertionError("Opening alignment workspace did not replace the main workspace")
    alignment_window = app.alignment_window
    with tempfile.TemporaryDirectory() as alignment_folder:
        reference = Path(alignment_folder) / "reference_without_exif.png"
        cv2.imwrite(str(reference), np.zeros((40, 60, 3), dtype=np.uint8))
        alignment_window._refresh_reference_focal_ui(reference)
        app.update_idletasks()
        app.update()
        if str(alignment_window.reference_focal_input.cget("state")) == "disabled":
            raise AssertionError("Reference focal input stayed disabled for an EXIF-free base")
        if alignment_window.reference_focal_length.get().strip():
            raise AssertionError("EXIF-free base silently received a guessed focal length")
        if "请" not in alignment_window.reference_focal_status.get():
            raise AssertionError("EXIF-free base did not visibly request a focal length")
        alignment_window.reference_focal_length.set("20")
        focal, focal_source = alignment_window._reference_focal(reference)
        if focal != 20.0 or "用户填写" not in focal_source:
            raise AssertionError("User-entered reference focal length was not accepted")
    alignment_scheduler = alignment_window.background_tasks
    alignment_window.destroy()
    app.update()
    if not alignment_scheduler.closed:
        raise AssertionError("Closing alignment workspace left its background scheduler running")
    if app.state() == "withdrawn" or app.alignment_window is not None:
        raise AssertionError("Closing alignment workspace did not restore the main workspace")
    app.open_screening_workspace()
    app.update()
    if app.state() != "withdrawn" or app.screening_window is None:
        raise AssertionError("Opening screening workspace did not replace the main workspace")
    # Reproduce the start-button freeze: capture-time sorting deliberately
    # blocks for each file. A real click must return before that metadata work,
    # because discovery/sorting now belongs to the submitted background task.
    import meteor_screening as screening_module
    screening_window = app.screening_window
    original_screening_source = screening_window.source_dir.get()
    real_capture_sort_key = screening_module.capture_sort_key
    real_screening_submit = screening_window.background_tasks.submit
    submitted_analysis = []
    with tempfile.TemporaryDirectory() as screening_folder:
        for index in range(3):
            Path(screening_folder, f"frame_{index}.jpg").touch()
        screening_module.capture_sort_key = lambda path: (
            time.sleep(0.20) or ("", path.name)
        )
        screening_window.background_tasks.submit = (
            lambda channel, function, **kwargs: submitted_analysis.append(
                (channel, function, kwargs)
            )
        )
        screening_window._restoring_autosave = True
        screening_window.source_dir.set(screening_folder)
        screening_window._restoring_autosave = False
        app.update_idletasks()
        analyze_x = max(1, screening_window.analyze_button.winfo_width() // 2)
        analyze_y = max(1, screening_window.analyze_button.winfo_height() // 2)
        analyze_started = time.perf_counter()
        screening_window.analyze_button.event_generate("<Enter>", x=analyze_x, y=analyze_y)
        screening_window.analyze_button.event_generate(
            "<ButtonPress-1>", x=analyze_x, y=analyze_y
        )
        screening_window.analyze_button.event_generate(
            "<ButtonRelease-1>", x=analyze_x, y=analyze_y
        )
        app.update()
        analyze_click_elapsed = time.perf_counter() - analyze_started
        if analyze_click_elapsed > 0.15:
            raise AssertionError(
                f"Screening start button blocked Tk during metadata scan: {analyze_click_elapsed:.3f}s"
            )
        if not submitted_analysis or submitted_analysis[0][0] != "analysis":
            raise AssertionError("Screening start did not dispatch metadata scanning to background work")
        if not screening_window.analysis_running:
            raise AssertionError("Screening start did not enter the visible background-running state")
        screening_window.analysis_generation += 1
        screening_window.analysis_running = False
        screening_window.analyze_button.configure(state="normal", text="开始分析")
        screening_window._restoring_autosave = True
        screening_window.source_dir.set(original_screening_source)
        screening_window._restoring_autosave = False
    screening_module.capture_sort_key = real_capture_sort_key
    screening_window.background_tasks.submit = real_screening_submit
    screening_scheduler = app.screening_window.background_tasks
    app.screening_window.destroy()
    app.update()
    if not screening_scheduler.closed:
        raise AssertionError("Closing screening workspace left its background scheduler running")
    if app.state() == "withdrawn" or app.screening_window is not None:
        raise AssertionError("Closing screening workspace did not restore the main workspace")
    return {
        "editable_composite": "passed", "move": "passed", "stretch": "passed",
        "clean_base_has_no_mask": "passed",
        "consecutive_manual_strokes": "passed",
        "paint_after_erase_restores": "passed",
        "shift_then_click_accumulates": "passed",
        "modified_left_click_cannot_delete": "passed",
        "lost_release_preserves_previous_stroke": "passed",
        "nearby_overlapping_paint_is_monotonic": "passed",
        "direct_single_image_transform": "passed",
        "live_drag_moves_meteor_pixels": "passed",
        "combined_live_drag_moves_pixels": "passed",
        "drag_start_borrows_full_frame": "passed",
        "live_drag_uses_sparse_patch": "passed",
        "drag_press_latency": "passed",
        "incremental_shared_preview": "passed",
        "instant_transform_restore": "passed",
        "drag_undo_preserves_mask": "passed",
        "empty_undo_preserves_mask": "passed",
        "instant_mask_add": "passed",
        "manual_stroke_needs_no_exact_rebuild": "passed",
        "instant_candidate_add": "passed",
        "stable_canvas_image": "passed",
        "source_transform_reference": "passed",
        "one_click_geometry_restore": "passed",
        "rotate": "passed", "per_meteor_adjustments": "passed",
        "per_meteor_override_is_local": "passed",
        "per_meteor_slider_is_realtime": "passed",
        "selected_restore_and_source_are_local": "passed",
        "separate_drag_is_local": "passed",
        "project_roundtrip": "passed", "delete_undo": "passed",
        "instant_isolated_delete": "passed",
        "delete_uses_visible_exact_cache": "passed",
        "no_render_loop": "passed",
        "single_workspace_navigation": "passed",
        "exact_preview_viewer": "passed",
        "exact_preview_resize_keeps_fit": "passed",
        "exact_preview_pan_reuses_image": "passed",
        "main_canvas_zoom_pan": "passed",
        "interactive_zoom_reuses_viewport": "passed",
        "interactive_zoom_latency": "passed",
        "view_switch_preserves_zoom": "passed",
        "panel_click_preserves_zoom": "passed",
        "blank_panel_click_preserves_zoom": "passed",
        "header_workspace_blank_click_preserves_viewport": "passed",
        "below_load_project_blank_click_preserves_viewport": "passed",
        "blank_tiff_list_click_preserves_viewport": "passed",
        "outside_image_click_preserves_zoom": "passed",
        "workspace_tabs": "passed",
        "all_controls_reachable": "passed",
        "collapsible_material_panel": "passed",
        "visible_undo_redo_buttons": "passed",
        "selectable_history_versions": "passed",
        "history_branch_truncation": "passed",
        "history_limit_100": "passed",
        "reference_focal_required_without_exif": "passed",
        "screening_start_is_nonblocking": "passed",
        "child_workspace_tasks_close_cleanly": "passed",
    }


def run_real_pointer_smoke(app) -> dict:
    """Drive the two reported blank areas with the real Windows mouse cursor."""
    import ctypes

    if not hasattr(ctypes, "windll"):
        raise RuntimeError("Real pointer smoke requires Windows")
    user32 = ctypes.windll.user32
    app._schedule_autosave = lambda: None
    app.autosave_suspended = True
    app.geometry("1280x820+0+0")
    app.deiconify()
    app.update_idletasks()
    app.update()
    app.maximize_for_normal_launch()
    app.update_idletasks()
    app.update()

    height, width = 800, 1200
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (12, 18, 30)
    cv2.line(image, (280, 430), (810, 285), (235, 245, 255), 8, cv2.LINE_AA)
    app.current_path = Path("real_pointer_meteor.tif")
    app.files = [app.current_path]
    app.preview_source = image
    app.preview_base = image.copy()
    app.preview_rgb = image
    app.current_dims = (width, height)
    app.canvas_image_shape = None
    app.view_mode.set("source")
    app.control_notebook.select(app.mask_tools_tab)
    app._present_preview_image(image, False, True)
    app._canvas_fit()
    app.update_idletasks()
    app.update()

    try:
        app.lift()
        app.focus_force()
        user32.SetForegroundWindow(int(app.winfo_id()))
    except Exception:
        pass
    app.update()

    def snapshot() -> tuple:
        return (
            round(float(app.canvas_zoom), 12),
            round(app.canvas_center_x / app.preview_rgb.shape[1], 8),
            round(app.canvas_center_y / app.preview_rgb.shape[0], 8),
            tuple(round(float(value), 4) for value in app.display_box),
            app.canvas.winfo_width(), app.canvas.winfo_height(),
            app.canvas_image_shape, app.preview_request_id,
            getattr(app, "canvas_zoom_reason", "unknown"),
        )

    def real_click_and_check(screen_x: int, screen_y: int, expected, label: str) -> None:
        if expected is None:
            raise AssertionError(f"{label}: test coordinate is outside the application")
        if not user32.SetCursorPos(int(screen_x), int(screen_y)):
            raise AssertionError(f"{label}: SetCursorPos failed")
        app.update_idletasks()
        app.update()
        pointer_x, pointer_y = app.winfo_pointerxy()
        hit = app.winfo_containing(pointer_x, pointer_y)
        if hit is not expected:
            raise AssertionError(
                f"{label}: real cursor hit {hit}, expected {expected}; "
                f"pointer=({pointer_x},{pointer_y}) requested=({screen_x},{screen_y})"
            )
        before = snapshot()
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        time.sleep(0.04)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        deadline = time.monotonic() + 1.55
        while time.monotonic() < deadline:
            app.update_idletasks()
            app.update()
            time.sleep(0.01)
        after = snapshot()
        if after != before:
            raise AssertionError(f"{label}: viewport changed: {before} -> {after}")

    title_right = app.workspace_title_label.winfo_rootx() + app.workspace_title_label.winfo_width()
    controls_left = app.paths_toggle_button.winfo_rootx()
    header_x = (title_right + controls_left) // 2
    header_y = app.header_panel.winfo_rooty() + app.header_panel.winfo_height() // 2
    real_click_and_check(header_x, header_y, app.header_panel, "真实鼠标点击工作区标题旁空白")

    load_x = app.load_project_button.winfo_rootx() + app.load_project_button.winfo_width() // 2
    load_bottom = app.load_project_button.winfo_rooty() + app.load_project_button.winfo_height()
    tools_bottom = app.mask_tools_tab.winfo_rooty() + app.mask_tools_tab.winfo_height()
    below_y = min(tools_bottom - 2, load_bottom + max(1, (tools_bottom - load_bottom) // 2))
    below_hit = app.winfo_containing(load_x, below_y)
    if below_hit is app.load_project_button:
        raise AssertionError("真实鼠标测试坐标仍落在载入项目按钮上")
    real_click_and_check(load_x, below_y, below_hit, "真实鼠标点击载入项目下方空白")

    return {
        "real_windows_pointer": "passed",
        "header_workspace_blank": "passed",
        "below_load_project_blank": "passed",
        "delay_seconds_each": 1.55,
    }


def main() -> None:
    from meteor_composer import MeteorComposer

    app = MeteorComposer()
    try:
        print(json.dumps(run_smoke(app), ensure_ascii=False))
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
