"""Real Tk pointer-event regression for toolbox navigation and responsive scan."""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageGrab
from tkinter import ttk


def pump(app, seconds):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        app.update()
        time.sleep(0.01)


def click(app, widget):
    widget.winfo_toplevel().attributes('-topmost', True)
    widget.winfo_toplevel().lift()
    app.update()
    reveal_control(app, widget)
    x = widget.winfo_rootx() + widget.winfo_width() // 2
    y = widget.winfo_rooty() + widget.winfo_height() // 2
    if 0 <= x < app.winfo_screenwidth() and 0 <= y < app.winfo_screenheight():
        assert app.winfo_containing(x, y) == widget, f'Control obscured: {widget}; hit {app.winfo_containing(x, y)}'
    widget.event_generate("<ButtonPress-1>", x=widget.winfo_width() // 2, y=widget.winfo_height() // 2)
    widget.event_generate("<ButtonRelease-1>", x=widget.winfo_width() // 2, y=widget.winfo_height() // 2)
    pump(app, 0.15)


def reveal_control(app, widget):
    widget.winfo_toplevel().attributes('-topmost', True)
    widget.winfo_toplevel().lift()
    app.update()
    owners = []
    branch = widget
    ancestor = widget.master
    while ancestor is not None:
        if isinstance(getattr(ancestor, 'master', None), ttk.Notebook):
            notebook = ancestor.master
            wanted = notebook.index(ancestor)
            if notebook.select() != str(ancestor):
                for x in range(2, notebook.winfo_width(), 4):
                    try:
                        hit = notebook.index(f'@{x},10')
                    except Exception:
                        continue
                    if hit == wanted:
                        notebook.event_generate('<ButtonPress-1>', x=x, y=10)
                        notebook.event_generate('<ButtonRelease-1>', x=x, y=10)
                        app.update()
                        break
        if hasattr(ancestor, '_inspector_canvas') and branch in getattr(ancestor, '_inspector_scrolled_widgets', (branch,)):
            owners.append(ancestor._inspector_canvas)
        branch = ancestor
        ancestor = getattr(ancestor, 'master', None)
    for canvas in reversed(owners):
        for _ in range(160):
            app.update()
            top = canvas.winfo_rooty()
            bottom = top + canvas.winfo_height()
            y = widget.winfo_rooty()
            if top <= y and y + widget.winfo_height() <= bottom:
                break
            canvas.event_generate('<Button-4>' if y < top else '<Button-5>')
        else:
            raise AssertionError(f'Cannot scroll to control: {widget}')


def widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from widgets(child)


def capture(window, name):
    target = os.environ.get("NIGHTSCAPE_SCREENSHOTS")
    if target:
        folder = Path(target)
        folder.mkdir(parents=True, exist_ok=True)
        window.update()
        x, y = window.winfo_rootx(), window.winfo_rooty()
        desktop = ImageGrab.grab()
        scale = desktop.width / window.winfo_screenwidth()
        desktop.crop(tuple(round(value * scale) for value in (x, y, x + window.winfo_width(), y + window.winfo_height()))).save(folder / name)


def run_smoke(app):
    app.attributes("-topmost", True)
    app.state("normal")
    app.geometry("1280x820+20+20")
    app.lift()
    app.focus_force()
    assert app.edit_inspector.winfo_exists()
    app.show_toolbox()
    pump(app, 0.4)
    app.lift()
    app.focus_force()
    pump(app, 0.3)
    assert app.toolbox_home.winfo_ismapped()
    capture(app, "toolbox.png")
    assert set(app.toolbox_home.tool_buttons) == {"meteor", "control_points", "color", "laboratory"}
    from software_settings_smoke import exercise_settings
    settings_checks = exercise_settings(app)
    click(app, app.toolbox_home.tool_buttons["color"])
    click(app, app.toolbox_home.tool_buttons["white_balance"])
    from white_balance_smoke import exercise_white_balance
    white_balance_checks = exercise_white_balance(app, app.white_balance_window)
    app.white_balance_window._request_close()
    pump(app, 1.4)
    assert app.white_balance_window is None
    click(app, app.toolbox_home.tool_buttons['light_pollution'])
    from light_pollution_smoke import exercise_light_pollution
    light_pollution_checks = exercise_light_pollution(app, app.light_pollution_window)
    app.light_pollution_window._request_close()
    pump(app, 1.4)
    assert app.light_pollution_window is None
    app.show_toolbox()
    click(app, app.toolbox_home.tool_buttons["laboratory"])
    assert set(app.toolbox_home.tool_buttons) == {"trails", "mean", "quality"}
    click(app, app.toolbox_home.tool_buttons["mean"])
    lab = app.laboratory_window
    from laboratory_smoke import exercise_laboratory
    laboratory_checks = exercise_laboratory(app, lab)
    with tempfile.TemporaryDirectory() as lab_folder:
        import numpy as np
        import tifffile
        source = Path(lab_folder) / "source"
        source.mkdir()
        paths = [source / f"{i}.tif" for i in range(2)]
        for path in paths:
            tifffile.imwrite(path, np.full((48, 64, 3), 12345, np.uint16), photometric="rgb")
        with patch("laboratory_workspace.filedialog.askopenfilenames", return_value=tuple(map(str, paths))):
            click(app, lab.add_button)
        lab.destination.set(str(Path(lab_folder) / "output"))
        click(app, lab.start_button)
        pump(app, 1.6)
        assert not lab.busy and lab.result
        assert np.all(tifffile.imread(lab.result / "result.tif") == 12345)
        capture(lab, "laboratory.png")
    lab._request_close()
    pump(app, 1.4)
    app.show_toolbox()
    click(app, app.toolbox_home.tool_buttons["control_points"])
    assert app._toolbox_path == ("control_points",)
    capture(app, "control-points-submenu.png")
    click(app, app.toolbox_home.tool_buttons["control_points"])
    window = app.alignment_window
    assert app.state() == "withdrawn" and window.control_points_only.get()
    pump(app, 0.3)
    capture(window, "control-points.png")
    heartbeat = []
    errors = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        inputs = root / "input"
        inputs.mkdir()
        base = root / "base.png"
        Image.new("RGB", (120, 80)).save(base)
        Image.new("RGB", (120, 80)).save(inputs / "frame.png")
        from alignment_workspace import read_lens_info
        def slow_read(*args):
            time.sleep(0.25)
            return read_lens_info(*args)
        window.base_path.set(str(base))
        window.meteor_dir.set(str(inputs))
        window.reference_focal_length.set("35")
        window.ptgui_path.set("")
        window.siril_path.set("")
        window.after(350, lambda: heartbeat.append(time.monotonic()))
        with patch("alignment_workspace.read_lens_info", side_effect=slow_read), patch("alignment_workspace.show_copyable_error", side_effect=lambda *a, **k: errors.append(a)):
            click(app, window.scan_button)
            assert window.running and window.scan_button.instate(["disabled"])
            pump(app, 0.5)
            assert heartbeat and not window.running and len(window.items) == 1, errors
            assert window.run_button.instate(["!disabled"])
            with patch('toolbox.require_software', return_value=None) as setup, patch('alignment_workspace.run_alignment_pipeline') as pipeline:
                click(app, window.run_button)
                setup.assert_called_once_with(window, ('ptgui', 'siril'))
                assert not pipeline.called and not window.running
            # A changed folder cannot launch the old scan's images.
            other = root / "other"
            other.mkdir()
            Image.new("RGB", (120, 80)).save(other / "different.png")
            window.meteor_dir.set(str(other))
            with patch("alignment_workspace.run_alignment_pipeline") as pipeline:
                click(app, window.run_button)
                assert not pipeline.called and errors
    back = next(w for w in widgets(window) if isinstance(w, ttk.Button) and w.cget("text") == "← 返回控制点生成")
    click(app, back)
    pump(app, 1.4)
    assert app.alignment_window is None and app.toolbox_home.winfo_ismapped()
    assert app._toolbox_path == ("control_points",)
    back = next(w for w in widgets(app.toolbox_home) if isinstance(w, ttk.Button) and w.cget("text") == "← 返回上级")
    click(app, back)
    assert app._toolbox_path == ()
    click(app, app.toolbox_home.tool_buttons["meteor"])
    assert set(app.toolbox_home.tool_buttons) == {"screening", "alignment", "composite", "video"}
    capture(app, "meteor-submenu.png")
    click(app, app.toolbox_home.tool_buttons["composite"])
    assert app.composite_panel.winfo_ismapped() and not app.toolbox_home.winfo_ismapped()
    capture(app, "composite.png")
    from meteor_screening import MeteorScreeningWindow
    from video_meteor import VideoMeteorWindow
    for key, cls, restore, save, attr in (
        ("screening", MeteorScreeningWindow, "_restore_autosave", "_save_autosave", "screening_window"),
        ("video", VideoMeteorWindow, "_restore_autosave", "_write_autosave", "video_window"),
    ):
        app.show_toolbox()
        click(app, app.toolbox_home.tool_buttons["meteor"])
        with patch.object(cls, restore), patch.object(cls, save):
            click(app, app.toolbox_home.tool_buttons[key])
            child = getattr(app, attr)
            assert child.edit_inspector.winfo_rootx() >= child.canvas.winfo_rootx() + child.canvas.winfo_width()
            assert child.canvas.winfo_width() >= 150 and child.canvas.winfo_height() >= 200
            if key == "screening":
                status, label = child.filter_status_combo, child.filter_label_combo
                assert status.winfo_rootx() + status.winfo_width() <= label.winfo_rootx()
                child.last_export_dir.set("")
                child.return_callback = lambda path: (_ for _ in ()).throw(AssertionError("Empty export became current directory"))
            capture(child, f"{attr}.png")
            child.geometry('980x650' if key == 'screening' else '1120x720')
            pump(app, 1.4)
            assert child.edit_inspector.winfo_rootx() >= child.canvas.winfo_rootx() + child.canvas.winfo_width()
            assert child.canvas.winfo_width() >= 150 and child.canvas.winfo_height() >= 200
            target = child.export_button if key == 'screening' else next(w for w in widgets(child) if isinstance(w, ttk.Button) and w.cget('text') == '导出动态流星视频')
            reveal_control(app, target)
            assert child.winfo_containing(target.winfo_rootx()+target.winfo_width()//2, target.winfo_rooty()+target.winfo_height()//2) == target
            capture(child, f"{attr}-small.png")
            commands = set(child._tclCommands or ())
            owned_timers = set()
            for timer in app.tk.splitlist(app.tk.call("after", "info")):
                script, _ = app.tk.call("after", "info", timer)
                if set(app.tk.splitlist(script)) & commands:
                    owned_timers.add(timer)
            back = next(w for w in widgets(child) if isinstance(w, ttk.Button) and w.cget("text") in {"返回流星合成功能", "返回流星合成工作区"})
            click(app, back)
            # Tkinter command names contain id(callback). After destruction,
            # another workspace may reuse that Python address and method name.
            # Tcl timer IDs, unlike callback addresses, are never recycled.
            remaining = set(app.tk.splitlist(app.tk.call("after", "info")))
            assert not (owned_timers & remaining), (attr, owned_timers & remaining)
            pump(app, 1.4)
            assert getattr(app, attr) is None and app.composite_panel.winfo_ismapped()
    return {**settings_checks, **white_balance_checks, **light_pollution_checks, **laboratory_checks, "hierarchical_categories": "passed", "submenu_parent_navigation": "passed", "toolbox_navigation": "passed", "control_points_entry": "passed", "scan_nonblocking": "passed", "scan_inputs_disabled": "passed", "stale_scan_prevented": "passed", "return_and_delayed_close": "passed", "composite_navigation": "passed", "screening_video_timer_cleanup": "passed", "screening_filters_do_not_overlap": "passed", "empty_export_never_imports_cwd": "passed"}


if __name__ == "__main__":
    from meteor_composer import MeteorComposer
    with patch.object(MeteorComposer, "_restore_autosave"):
        app = MeteorComposer()
    try:
        print(json.dumps(run_smoke(app)))
    finally:
        app.destroy()
