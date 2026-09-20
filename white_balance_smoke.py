"""Real pointer regressions also executed inside the packaged application."""
import hashlib
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from white_balance import apply_lut, make_lut


def exercise_white_balance(root, window):
    from toolbox_smoke import click, pump, capture
    def wait_for(predicate):
        limit = time.monotonic() + 12
        while not predicate() and time.monotonic() < limit:
            pump(root, .05)
        assert predicate(), window.status.get()
        pump(root, 1.4)
    def reveal(widget):
        for _ in range(100):
            top = window.control_canvas.winfo_rooty()
            bottom = top + window.control_canvas.winfo_height()
            y = widget.winfo_rooty()
            if top <= y and y + widget.winfo_height() <= bottom:
                return
            window.control_canvas.event_generate("<Button-4>" if y < top else "<Button-5>")
            pump(root, .015)
        raise AssertionError("Control is not reachable")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source"
        source.mkdir()
        path = source / "night.tif"
        y, x = np.mgrid[:1200, :1800]
        pixels = np.stack([3000+y*9, 3500+y*7, 8000+y*4], axis=-1).astype(np.uint16)
        rng = np.random.default_rng(71)
        for xx, yy in zip(rng.integers(10, 1790, 180), rng.integers(10, 1190, 180)):
            pixels[yy-1:yy+2, xx-1:xx+2] = [45000, 48000, 55000]
        pixels[540:660, 840:960] = [24000, 18000, 12000]
        tifffile.imwrite(path, pixels, photometric="rgb")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch("white_balance_workspace.filedialog.askopenfilename", return_value=str(path)):
            click(root, window.open_button)
        wait_for(lambda: window.levels is not None and window.photo is not None and not window.busy)
        assert window.control_canvas.winfo_rootx() >= window.canvas.winfo_rootx() + window.canvas.winfo_width()
        assert np.array_equal(window.levels[0], pixels)
        click(root, window.actual_button)
        pump(root, .4)
        cw, ch = window.canvas.winfo_width(), window.canvas.winfo_height()
        window.canvas.event_generate("<ButtonPress-1>", x=cw//2, y=ch//2)
        window.canvas.event_generate("<B1-Motion>", x=cw//2-35, y=ch//2-20)
        window.canvas.event_generate("<ButtonRelease-1>", x=cw//2-35, y=ch//2-20)
        pump(root, .4)
        view = (window.zoom, window.center, window.canvas.bbox(window.image_item), window.canvas.winfo_width(), window.canvas.winfo_height())
        scale = window.sliders[0]
        scale.event_generate("<ButtonPress-1>", x=round(scale.winfo_width()*.8), y=scale.winfo_height()//2)
        scale.event_generate("<B1-Motion>", x=round(scale.winfo_width()*.85), y=scale.winfo_height()//2)
        scale.event_generate("<ButtonRelease-1>", x=round(scale.winfo_width()*.85), y=scale.winfo_height()//2)
        pump(root, 1.5)
        assert abs(window.warmth.get()) > .1
        assert view == (window.zoom, window.center, window.canvas.bbox(window.image_item), window.canvas.winfo_width(), window.canvas.winfo_height())
        click(root, window.compare_button)
        pump(root, 1.5)
        assert window.original.get() and "原图" in window.info.get()
        assert view[:2] == (window.zoom, window.center)
        click(root, window.compare_button)
        click(root, window.fit_button)
        click(root, window.pick_button)
        window.canvas.event_generate("<ButtonPress-1>", x=window.canvas.winfo_width()//2, y=window.canvas.winfo_height()//2)
        window.canvas.event_generate("<ButtonRelease-1>", x=window.canvas.winfo_width()//2, y=window.canvas.winfo_height()//2)
        pump(root, 1.5)
        assert window.neutral != [1, 1, 1] and not window.picker.get()
        balanced = apply_lut(pixels[600:601,900:901], make_lut(window.settings()))[0,0].astype(int)
        assert balanced.max() - balanced.min() <= 2
        window.equipment["camera"].set("Test camera")
        window.equipment["modification"].set("Hα 增强")
        window.equipment["filter"].set("UV/IR cut")
        window.equipment["reference"].set("灰卡 / 同一光学组合")
        settings = window.settings()
        config = Path(folder) / "settings.json"
        reveal(window.save_button)
        with patch("white_balance_workspace.filedialog.asksaveasfilename", return_value=str(config)):
            click(root, window.save_button)
        reveal(window.reset_button)
        click(root, window.reset_button)
        assert window.neutral == [1, 1, 1] and window.warmth.get() == 0
        reveal(window.load_button)
        with patch("white_balance_workspace.filedialog.askopenfilename", return_value=str(config)):
            click(root, window.load_button)
        assert window.settings() == settings
        window.destination.set(str(Path(folder) / "out"))
        click(root, window.export_button)
        wait_for(lambda: window.result is not None and not window.busy)
        result = tifffile.imread(window.result / "result.tif")
        assert result.dtype == np.uint16 and result.shape == pixels.shape
        np.testing.assert_array_equal(result, apply_lut(pixels, make_lut(settings)))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        # Preset loading enables reuse, and opening another photo must not
        # discard equipment identifiers or the sampled calibration.
        second = source / "second.tif"
        tifffile.imwrite(second, pixels, photometric="rgb")
        with patch("white_balance_workspace.filedialog.askopenfilename", return_value=str(second)):
            click(root, window.open_button)
        wait_for(lambda: window.source == second and not window.busy)
        assert window.settings() == settings and window.keep_settings.get()
        previous = window.result
        broken = source / "broken.tif"
        broken.write_bytes(b"invalid")
        logged = []
        with patch("white_balance_workspace.append_runtime_log", side_effect=lambda *args: logged.append(args)):
            with patch("white_balance_workspace.filedialog.askopenfilenames", return_value=(str(path), str(second), str(broken))):
                click(root, window.batch_button)
            wait_for(lambda: not window.busy and window.result != previous)
        batch = json.loads((window.result / "batch.json").read_text(encoding="utf-8"))
        assert batch["status"] == "complete_with_errors" and len(batch["items"]) == 3
        assert logged and "broken.tif" in logged[0][0] and "成功 2 / 3" in window.status.get()
        assert float(window.progress["value"]) == 100
        assert batch["settings"] == settings
        for item in batch["items"]:
            if item["status"] == "complete":
                np.testing.assert_array_equal(tifffile.imread(Path(item["output"]) / "result.tif"), result)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        capture(window, "white-balance.png")
        window.state("normal")
        window.geometry("850x600")
        pump(root, 1.5)
        for _ in range(20):
            window.control_canvas.event_generate("<Button-5>")
        pump(root, .3)
        top = window.control_canvas.winfo_rooty()
        bottom = top + window.control_canvas.winfo_height()
        assert top <= window.load_button.winfo_rooty() < bottom
        assert window.export_button.winfo_ismapped() and window.canvas.winfo_height() > 100
        assert window.control_canvas.winfo_rootx() >= window.canvas.winfo_rootx() + window.canvas.winfo_width()
        capture(window, "white-balance-small.png")
    return {"modified_camera_preset": "passed", "modified_batch_consistency": "passed", "wb_viewport_stable": "passed", "wb_original_compare": "passed", "wb_neutral_sample": "passed", "wb_settings_roundtrip": "passed", "wb_16bit_export_readonly": "passed", "wb_small_window_controls": "passed"}
