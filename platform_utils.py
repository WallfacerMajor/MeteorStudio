from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def open_folder(path: str | Path) -> Path:
    """Open a directory in the native file manager and return that directory."""
    target = Path(path).expanduser()
    folder = target if target.is_dir() else target.parent
    if not folder.is_dir():
        raise FileNotFoundError(f"文件夹不存在：{folder}")
    if sys.platform == "win32":
        os.startfile(str(folder))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(folder)])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
    return folder
