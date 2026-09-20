# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
sys.path.insert(0, SPECPATH)
from app_icon import build_icons
icon_dir = build_icons(Path(SPECPATH) / 'build' / 'app-icons')


a = Analysis(
    ['meteor_composer.py'],
    pathex=[],
    binaries=[('C:/Users/meijie/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.1-full_build/bin/ffmpeg.EXE', '.')],
    datas=[('meteor_ranker.json', '.'), ('candidate_dataset.npz', '.')],
    hiddenimports=['background_tasks', 'meteor_detection', 'meteor_learning', 'video_meteor', 'alignment_workspace', 'ptgui_pipeline', 'meteor_screening', 'preview_viewer', 'gui_interaction_smoke', 'editable_composite_smoke'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'transformers', 'triton', 'llvmlite', 'numba', 'onnx', 'onnxruntime', 'matplotlib', 'timm', 'tokenizers'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MeteorStudio',
    icon=str(icon_dir / 'nightscape.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeteorStudio',
)
