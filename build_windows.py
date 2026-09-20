"""Reproducible Windows packaging entry point for MeteorStudio."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HIDDEN_IMPORTS = (
    "background_tasks", "meteor_detection", "meteor_learning", "video_meteor",
    "alignment_workspace", "ptgui_pipeline", "meteor_screening",
    "preview_viewer", "gui_interaction_smoke", "editable_composite_smoke",
)
EXCLUDED_MODULES = (
    "torch", "torchvision", "transformers", "triton", "llvmlite", "numba",
    "onnx", "onnxruntime", "matplotlib", "timm", "tokenizers",
)


def pyinstaller_arguments(ffmpeg: str | None) -> list[str]:
    arguments = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--windowed", "--name", "MeteorStudio",
        "--icon", str(ROOT / "build" / "app-icons" / "nightscape.ico"),
    ]
    for module in HIDDEN_IMPORTS:
        arguments.extend(("--hidden-import", module))
    for module in EXCLUDED_MODULES:
        arguments.extend(("--exclude-module", module))
    arguments.extend((
        "--add-data", "meteor_ranker.json;.",
        "--add-data", "candidate_dataset.npz;.",
    ))
    if ffmpeg:
        arguments.extend(("--add-binary", f"{ffmpeg};."))
    arguments.append("meteor_composer.py")
    return arguments


def main() -> int:
    if sys.platform != "win32":
        raise SystemExit("build_windows.py 只能在 Windows 上运行")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "pyinstaller"],
        cwd=ROOT, check=True,
    )
    ffmpeg = shutil.which("ffmpeg")
    from app_icon import build_icons
    build_icons(ROOT / "build" / "app-icons")
    if ffmpeg is None:
        print("警告：未找到 FFmpeg，视频导出将要求目标电脑自行安装 FFmpeg。")
    subprocess.run(pyinstaller_arguments(ffmpeg), cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
