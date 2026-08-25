"""Real Tk smoke test for direct editing in final/labeled composite views."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


def run_smoke(app) -> dict:
    from meteor_composer import Stroke

    app.geometry("1280x820+10000+10000")
    app.update_idletasks()
    for after_id in app.tk.call("after", "info"):
        app.after_cancel(after_id)
    app._schedule_autosave = lambda: None
    app.autosave_suspended = True

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
    app.output_mode.set("separate")
    app.view_mode.set("labeled")
    app.canvas.configure(width=900, height=560)
    app.update_idletasks()
    app._render_preview()
    app.update()

    selected = (key, 0)
    geometry = app._object_canvas_geometry(selected)
    if geometry is None:
        raise AssertionError("Object geometry missing")
    center = geometry["center"]
    before = replace(app.strokes[key][0], points=app.strokes[key][0].points.copy())
    app.canvas.event_generate("<ButtonPress-1>", x=int(center[0]), y=int(center[1]))
    app.canvas.event_generate("<B1-Motion>", x=int(center[0] + 42), y=int(center[1] + 21), state=0x0100)
    app.canvas.event_generate("<ButtonRelease-1>", x=int(center[0] + 42), y=int(center[1] + 21))
    app.update()
    moved = app.strokes[key][0]
    if moved.offset_x == before.offset_x and moved.offset_y == before.offset_y:
        raise AssertionError("Move did not update offsets")
    if app.candidates[key][0].offset_x != moved.offset_x:
        raise AssertionError("Candidate transform was not synchronized")

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

    count = len(app.strokes[key])
    app._delete_selected_object()
    if len(app.strokes[key]) != count - 1 or app.candidates[key]:
        raise AssertionError("Delete did not remove the locked candidate object")
    app.undo_stroke()
    if len(app.strokes[key]) != count or len(app.candidates[key]) != 1:
        raise AssertionError("Undo did not restore object and candidate metadata")
    return {
        "editable_composite": "passed", "move": "passed", "stretch": "passed",
        "rotate": "passed", "delete_undo": "passed",
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
