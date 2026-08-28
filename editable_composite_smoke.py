"""Real Tk smoke test for direct editing in final/labeled composite views."""

from __future__ import annotations

import json
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
    if tab_names != ["3  蒙版与候选", "4  融合与底图", "5  所选流星"]:
        raise AssertionError(f"Unexpected workspace tabs: {tab_names}")
    required_controls = {
        "B ✎ 画笔", "E ▱ 橡皮擦", "AI分析当前单张候选", "自动检测全部",
        "保存项目", "载入项目", "自动优化当前流星", "自动优化全部流星",
        "重置底图曝光", "恢复自动值", "恢复原始融合", "导出合成结果",
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
    app.canvas.event_generate("<ButtonPress-1>", x=int(center[0]), y=int(center[1]))
    app.canvas.event_generate("<B1-Motion>", x=int(center[0] + 42), y=int(center[1] + 21), state=0x0100)
    app.update()
    moving = app.strokes[key][0]
    original_midpoint = np.mean(
        np.asarray([(x * (width - 1), y * (height - 1)) for x, y in moving.points]), axis=0
    )
    moved_midpoint = original_midpoint + np.asarray([moving.offset_x, moving.offset_y])
    mx, my = int(round(moved_midpoint[0])), int(round(moved_midpoint[1]))
    ox, oy = int(round(original_midpoint[0])), int(round(original_midpoint[1]))
    live_new_peak = int(app.preview_rgb[max(0, my - 6):my + 7, max(0, mx - 6):mx + 7].max())
    live_old_peak = int(app.preview_rgb[max(0, oy - 6):oy + 7, max(0, ox - 6):ox + 7].max())
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
    shared_peak = int(app.preview_rgb[max(0, sy - 6):sy + 7, max(0, sx - 6):sx + 7].max())
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
    app.hover_candidate_index = len(app.candidates[key]) - 1
    app._pick_hover_candidate()
    app.update()
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
    app.selected_brightness.set(137)
    app.selected_cleanup.set(84)
    app.selected_saturation.set(112)
    app.selected_match.set(True)
    app.selected_feather.set(19)
    app._selected_adjustment_changed()
    app.update()
    app._request_global_preview = real_slider_request
    app._begin_exact_preview = real_slider_exact
    app._invalidate_global_preview = real_slider_invalidate
    app._render_preview = real_slider_render
    app.pairs.pop(overlap_key, None)
    app.strokes.pop(overlap_key, None)
    app.output_mode.set("separate")
    if slider_invalidations or slider_full_rebuilds:
        raise AssertionError(
            f"Per-meteor slider rebuilt the composite: invalidations={slider_invalidations}, "
            f"workers={slider_full_rebuilds}"
        )
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
    delete_cached, _delete_mask = compose_meteor_objects(
        source, base, app.strokes[key], False, False, 15, 25,
        "自然融合", True, 100, 70,
    )
    app.output_mode.set("combined")
    app.global_preview_rgb = delete_cached
    app.global_labeled_preview_rgb = delete_cached.copy()
    app.global_preview_signature = app._global_preview_state_signature()
    count = len(app.strokes[key])
    full_rebuild_calls = 0
    exact_validation_calls = 0
    real_invalidate = app._invalidate_global_preview
    real_exact_validation = app._schedule_global_exact_validation
    def count_full_rebuild():
        nonlocal full_rebuild_calls
        full_rebuild_calls += 1
        return real_invalidate()
    def count_exact_validation(_signature):
        nonlocal exact_validation_calls
        exact_validation_calls += 1
    app._invalidate_global_preview = count_full_rebuild
    app._schedule_global_exact_validation = count_exact_validation
    app._delete_selected_object()
    app._invalidate_global_preview = real_invalidate
    app._schedule_global_exact_validation = real_exact_validation
    if len(app.strokes[key]) != count - 1 or app.candidates[key]:
        raise AssertionError("Delete did not remove the locked candidate object")
    if full_rebuild_calls or exact_validation_calls:
        raise AssertionError(
            f"Isolated delete scheduled a global rebuild: "
            f"invalidate={full_rebuild_calls}, validation={exact_validation_calls}; "
            f"error={getattr(app, 'last_incremental_delete_error', None)}"
        )
    if app.global_preview_signature != app._global_preview_state_signature():
        raise AssertionError("Incremental delete did not commit the new composite state")
    app.undo_stroke()
    if len(app.strokes[key]) != count or len(app.candidates[key]) != 1:
        raise AssertionError("Undo did not restore object and candidate metadata")

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
    app.mask_tools_tab.event_generate("<ButtonPress-1>", x=blank_x, y=blank_y)
    app.mask_tools_tab.event_generate("<ButtonRelease-1>", x=blank_x, y=blank_y)
    app.update()
    if abs(app.canvas_zoom - main_fit_zoom) > 1e-9:
        raise AssertionError("Clicking blank space inside the bottom panel changed zoom")
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
    app._canvas_zoom_by(1.25, (app.canvas.winfo_width() // 3, app.canvas.winfo_height() // 3))
    if app.canvas_zoom <= main_fit_zoom or app.preview_photo is None:
        raise AssertionError("Main mask canvas did not zoom without recompositing")
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
    exact_viewer.actual_size()
    exact_viewer._zoom_by(1.25)
    exact_viewer.mode.set("labeled")
    exact_viewer._render()
    exact_viewer.update()
    if fit_zoom >= 1.0 or exact_viewer.zoom <= 1.0 or exact_viewer.photo is None:
        raise AssertionError("Full-resolution viewer did not fit, zoom, and render")
    exact_viewer.destroy()

    app.open_video_workspace()
    app.update()
    if app.state() != "withdrawn" or app.video_window is None:
        raise AssertionError("Opening video workspace did not replace the main workspace")
    app.video_window.destroy()
    app.update()
    if app.state() == "withdrawn" or app.video_window is not None:
        raise AssertionError("Closing video workspace did not restore the main workspace")

    app.open_alignment_workspace()
    app.update()
    if app.state() != "withdrawn" or app.alignment_window is None:
        raise AssertionError("Opening alignment workspace did not replace the main workspace")
    app.alignment_window.destroy()
    app.update()
    if app.state() == "withdrawn" or app.alignment_window is not None:
        raise AssertionError("Closing alignment workspace did not restore the main workspace")
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
        "main_canvas_zoom_pan": "passed",
        "view_switch_preserves_zoom": "passed",
        "panel_click_preserves_zoom": "passed",
        "blank_panel_click_preserves_zoom": "passed",
        "outside_image_click_preserves_zoom": "passed",
        "workspace_tabs": "passed",
        "all_controls_reachable": "passed",
        "collapsible_material_panel": "passed",
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
