# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
import sys
sys.path.insert(0, SPECPATH)
from app_icon import build_icons
icon_dir = build_icons(Path(SPECPATH) / 'build' / 'app-icons')
star_root = Path(SPECPATH) / 'experiments' / 'star_reduction_compare'
star_binary = star_root / 'dist' / 'StarReductionCompare'
if not star_binary.is_dir():
    raise RuntimeError('Build experiments/star_reduction_compare/StarReductionCompare.spec first')
star_data = [(str(star_binary), 'star_reduction')]


a = Analysis(
    ['meteor_composer.py'],
    pathex=[],
    binaries=[('C:/Users/meijie/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-7.1-full_build/bin/ffmpeg.EXE', '.')],
    datas=[('meteor_ranker.json', '.'), ('candidate_dataset.npz', '.')]
          + collect_data_files('setuptools._vendor.jaraco.text') + star_data,
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
    manifest=str(Path(SPECPATH) / 'windows.manifest'),
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
