"""Pointer/keyboard regression shared by source tests and packaged smoke."""
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile


def exercise_laboratory(root, window):
    from toolbox_smoke import click, pump
    from laboratory import read_pixels
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source"
        source.mkdir()
        paths = [source / f"frame_{i}.tif" for i in range(3)]
        for i, path in enumerate(paths):
            tifffile.imwrite(path, np.full((48, 64, 3), 1000 + 100 * i, np.uint16), photometric="rgb")
        window.destination.set(str(Path(folder) / "out"))
        assert window.start_button.instate(["disabled"])
        with patch("laboratory_workspace.filedialog.askopenfilenames", return_value=tuple(map(str, paths + paths))):
            click(root, window.add_button)
        assert len(window.paths) == 3
        # Select a real row and remove it using the visible button.
        row = window.files.get_children()[1]
        x, y, width, height = window.files.bbox(row)
        window.files.event_generate("<ButtonPress-1>", x=x + 30, y=y + height // 2)
        window.files.event_generate("<ButtonRelease-1>", x=x + 30, y=y + height // 2)
        pump(root, .1)
        click(root, window.remove_button)
        assert window.paths == [paths[0], paths[2]]
        assert all(path.exists() for path in paths)
        started, release = threading.Event(), threading.Event()
        def slow_read(path):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test read was not released")
            return read_pixels(path)
        try:
            with patch("laboratory.read_pixels", side_effect=slow_read):
                click(root, window.start_button)
                assert started.wait(2)
                assert window.add_button.instate(["disabled"])
                assert window.remove_button.instate(["disabled"])
                window.files.focus_force()
                window.files.event_generate("<Delete>")
                pump(root, .1)
                assert len(window.paths) == 2
                click(root, window.cancel_button)
                assert window.busy and window.start_button.instate(["disabled"])
                release.set()
                pump(root, 1.6)
                assert not window.busy and window.result is None
        finally:
            release.set()
        click(root, window.start_button)
        # Delayed packets from the cancelled generation must never unlock or
        # replace a newer result, even if they arrive after the next Start.
        window.events.put((window.run_id - 1, "done", Path(folder) / "stale"))
        window.events.put((window.run_id - 1, "progress", (0, "stale")))
        pump(root, 1.6)
        assert not window.busy and window.result and window.result.name != "stale"
        np.testing.assert_array_equal(tifffile.imread(window.result / "result.tif"), np.full((48, 64, 3), 1100, np.uint16))
        assert "stale" not in window.status.get()
        last_result = window.result
        # Preserve the previous successful output after a recoverable failure.
        window.destination.set(str(source))
        errors = []
        with patch("laboratory_workspace.show_copyable_error", side_effect=lambda *args, **kwargs: errors.append(args)):
            click(root, window.start_button)
            pump(root, 1.6)
        assert errors and not window.busy
        assert window.result == last_result and not window.open_button.instate(["disabled"])
        click(root, window.clear_button)
        assert not window.paths and window.start_button.instate(["disabled"])
        assert all(path.exists() for path in paths)
    return {"lab_remove_readonly": "passed", "lab_cancel_restart": "passed", "lab_stale_events": "passed", "lab_previous_result_preserved": "passed"}
