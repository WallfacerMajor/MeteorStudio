"""Real Tk canvas smoke test using an existing MeteorStudio project (read-only)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def run_smoke(app, project_path: Path) -> dict:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    data["output_dir"] = str(Path.cwd() / "smoke_output")
    if not data.get("candidates"):
        raise RuntimeError("Project has no analyzed candidates")

    app.geometry("1280x820+10000+10000")
    app.update_idletasks()
    for after_id in app.tk.call("after", "info"):
        app.after_cancel(after_id)
    app._schedule_autosave = lambda: None
    app.autosave_suspended = True
    app._apply_project_data(data)
    app.autosave_suspended = True
    app.scan_inputs()

    candidate_key = next(iter(app.candidates))
    path = Path(candidate_key)
    if path not in app.files:
        raise RuntimeError(f"Candidate image was not scanned: {path}")
    source, base, dims = None, None, None
    result = app._load_preview_worker(path, app._effective_source_path(path), app.pairs[candidate_key])
    _, loaded_path, source, base, dims, _base_path, _pairing_signature = result
    app.current_path = loaded_path
    app.preview_source = source
    app.preview_base = base
    app.preview_rgb = source
    app.current_dims = dims
    app.canvas.configure(width=900, height=560)
    app.update_idletasks()
    app._render_preview()
    app.update_idletasks()

    # Move onto an unlocked candidate to create the real floating canvas button.
    requested_index = os.environ.get("METEOR_SMOKE_CANDIDATE_INDEX")
    candidate_index = (
        int(requested_index) if requested_index is not None
        else next(i for i, item in enumerate(app.candidates[candidate_key]) if not item.locked)
    )
    candidate = app.candidates[candidate_key][candidate_index]
    x0, y0, x1, y1 = app.display_box
    px, py = candidate.points[len(candidate.points) // 2]
    cx = int(x0 + px * (x1 - x0))
    cy = int(y0 + py * (y1 - y0))
    app.canvas.event_generate("<Motion>", x=cx, y=cy)
    app.update()
    button_bbox = app.canvas.bbox("candidate_pick")
    if button_bbox is None:
        raise AssertionError("Candidate button did not appear")
    bx = int((button_bbox[0] + button_bbox[2]) / 2)
    by = int((button_bbox[1] + button_bbox[3]) / 2)
    app.canvas.event_generate("<Motion>", x=bx, y=by)
    app.canvas.event_generate("<ButtonPress-1>", x=bx, y=by)
    app.canvas.event_generate("<ButtonRelease-1>", x=bx, y=by)
    app.update()
    if not candidate.locked:
        raise AssertionError("Candidate click did not lock the candidate")
    if not any(item.locked and item.points == candidate.points for item in app.strokes[candidate_key]):
        raise AssertionError("Candidate click did not add/lock a mask stroke")
    selected_mask = app._preview_mask()
    if not np.any(selected_mask > 0.05):
        raise AssertionError("Candidate click produced no visible mask pixels")
    output_dir_text = os.environ.get("METEOR_SMOKE_OUTPUT")
    output_dir = Path(output_dir_text) if output_dir_text else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(app.preview_rgb).save(output_dir / "sample_01_candidate_selected.png")
        Image.fromarray(np.clip(selected_mask * 255, 0, 255).astype(np.uint8)).save(
            output_dir / "sample_01_candidate_mask.png"
        )

    # Draw through the actual ButtonPress/B1-Motion/ButtonRelease bindings.
    before = len(app.strokes[candidate_key])
    sx = int(x0 + (x1 - x0) * 0.60)
    sy = int(y0 + (y1 - y0) * 0.45)
    ex = min(x1 - 5, sx + 100)
    ey = min(y1 - 5, sy + 45)
    app.canvas.event_generate("<Motion>", x=sx, y=sy)
    app.canvas.event_generate("<ButtonPress-1>", x=sx, y=sy)
    for amount in range(1, 6):
        x = int(sx + (ex - sx) * amount / 5)
        y = int(sy + (ey - sy) * amount / 5)
        app.canvas.event_generate("<B1-Motion>", x=x, y=y, state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=ex, y=ey)
    app.update()
    after = len(app.strokes[candidate_key])
    if after != before + 1:
        raise AssertionError(f"Manual brush did not add exactly one stroke: {before} -> {after}")
    if len(app.strokes[candidate_key][-1].points) < 2:
        raise AssertionError("Manual brush stroke contains too few points")
    manual_mask = app._preview_mask()
    if np.count_nonzero(manual_mask > 0.05) <= np.count_nonzero(selected_mask > 0.05):
        raise AssertionError("Manual brush did not add visible mask pixels")
    if output_dir is not None:
        Image.fromarray(app.preview_rgb).save(output_dir / "sample_02_manual_added.png")
        Image.fromarray(np.clip(manual_mask * 255, 0, 255).astype(np.uint8)).save(
            output_dir / "sample_02_combined_mask.png"
        )
        (output_dir / "sample_interaction_demo.json").write_text(
            json.dumps(app._project_data(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # In the final/labeled preview the same manual stroke must behave as an
    # editable meteor object rather than starting another paint stroke.
    app.output_mode.set("separate")
    app.view_mode.set("labeled")
    app._render_preview()
    app.update()
    object_index = len(app.strokes[candidate_key]) - 1
    geometry = app._object_canvas_geometry((candidate_key, object_index))
    if geometry is None:
        raise AssertionError("Editable object geometry was not created")
    center = geometry["center"]
    ox, oy = app.strokes[candidate_key][object_index].offset_x, app.strokes[candidate_key][object_index].offset_y
    app.canvas.event_generate("<ButtonPress-1>", x=int(center[0]), y=int(center[1]))
    app.canvas.event_generate("<B1-Motion>", x=int(center[0] + 36), y=int(center[1] + 18), state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=int(center[0] + 36), y=int(center[1] + 18))
    app.update()
    moved = app.strokes[candidate_key][object_index]
    if moved.offset_x == ox and moved.offset_y == oy:
        raise AssertionError("Dragging selected meteor did not update its transform")
    before_delete = len(app.strokes[candidate_key])
    app._delete_selected_object()
    if len(app.strokes[candidate_key]) != before_delete - 1:
        raise AssertionError("Delete did not remove selected meteor object")
    app.undo_stroke()
    if len(app.strokes[candidate_key]) != before_delete:
        raise AssertionError("Undo did not restore deleted meteor object")

    result = {
        "candidate_button": "passed",
        "candidate_locked": candidate.auto_score,
        "manual_brush": "passed",
        "manual_points": len(app.strokes[candidate_key][-1].points),
        "editable_composite": "passed",
        "mask_pixels_before": int(np.count_nonzero(selected_mask > 0.05)),
        "mask_pixels_after": int(np.count_nonzero(manual_mask > 0.05)),
        "image": candidate_key,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    return result


def main() -> None:
    from meteor_composer import MeteorComposer

    project_path = Path(os.environ["METEOR_SMOKE_PROJECT"])
    app = MeteorComposer()
    try:
        result = run_smoke(app, project_path)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
