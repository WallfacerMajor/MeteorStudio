"""Integration check using installed Siril/PTGui and disposable synthetic stars."""
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import tifffile
from PIL import Image, TiffImagePlugin

from meteor_composer import MeteorComposer
from toolbox_smoke import click, pump


def run():
    root = Path("review-artifacts/external-alignment").resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "input"
    source.mkdir(exist_ok=True)
    rng = np.random.default_rng(412)
    plane = rng.normal(12, 1, (1200, 1600)).astype(np.float32)
    yy, xx = np.mgrid[-20:21, -20:21]
    for _ in range(300):
        x, y = int(rng.integers(30, 1570)), int(rng.integers(30, 1170))
        sigma = rng.uniform(4, 6)
        plane[y-20:y+21, x-20:x+21] += rng.uniform(80, 190) * np.exp(-(xx*xx + yy*yy) / (2*sigma*sigma))
    plane = np.clip(plane, 0, 255).astype(np.uint8)
    base = root / "reference.jpg"
    exif = Image.Exif()
    exif[37386] = TiffImagePlugin.IFDRational(35, 1)
    Image.fromarray(plane).convert("RGB").save(base, quality=98, exif=exif)
    for index, shift in enumerate((8, 15)):
        moved = cv2.warpAffine(plane, np.float32([[1, 0, shift], [0, 1, 3]]), (1600, 1200), borderValue=8)
        Image.fromarray(moved).convert("RGB").save(source / f"frame{index}.jpg", quality=98, exif=exif)
    files = [base, *source.glob("*.jpg")]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with patch.object(MeteorComposer, "_restore_autosave"):
        app = MeteorComposer()
    errors = []
    reports = []
    try:
        app.open_control_points_workspace()
        window = app.alignment_window
        window.base_path.set(str(base))
        window.meteor_dir.set(str(source))
        window.output_dir.set(str(root / "output"))
        with patch("alignment_workspace.messagebox.askyesno", return_value=False), patch("alignment_workspace.show_copyable_error", side_effect=lambda *a, **k: errors.append(str(a))):
            click(app, window.scan_button)
            deadline = time.monotonic() + 30
            while window.running and time.monotonic() < deadline:
                pump(app, 0.05)
            assert not errors and len(window.items) == 2, errors
            for export in (False, True):
                window.control_points_only.set(not export)
                click(app, window.run_button)
                deadline = time.monotonic() + 150
                while window.running and time.monotonic() < deadline:
                    pump(app, 0.1)
                assert not window.running and not errors, (window.status.get(), errors)
                result = window.last_result
                assert result is not None
                projects = list((Path(result.project_dir) / "ptgui_project" / "ready_projects").glob("*.pts"))
                assert len(projects) == 2, [(i.status, i.message) for i in result.items]
                for project in projects:
                    data = json.loads(project.read_text(encoding="utf-8"))["project"]
                    assert len(data["imagegroups"]) == 2 and len(data["controlpoints"]) >= 4
                if export:
                    for item in result.items:
                        assert item.status.startswith("已导出"), item.message
                        with tifffile.TiffFile(item.output_layer) as tif:
                            assert tif.pages[0].dtype == np.uint16
                else:
                    assert all(item.output_layer is None for item in result.items)
                    assert window.load_button.instate(["disabled"])
                reports.append({"mode": "layers" if export else "control_points", "projects": len(projects), "control_points": [item.control_points for item in result.items], "statuses": [item.status for item in result.items], "output": result.project_dir})
                print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
        assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (root / "report.json").write_text(json.dumps({"input_hashes_unchanged": True, "runs": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        app.destroy()


if __name__ == "__main__":
    run()
