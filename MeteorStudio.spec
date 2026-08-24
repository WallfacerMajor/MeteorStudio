# -*- mode: python ; coding: utf-8 -*-

import shutil


ffmpeg_path = shutil.which('ffmpeg')


a = Analysis(
    ['meteor_composer.py'],
    pathex=[],
    binaries=[(ffmpeg_path, '.')] if ffmpeg_path else [],
    datas=[('meteor_ranker.json', '.'), ('candidate_dataset.npz', '.')],
    hiddenimports=['meteor_learning', 'video_meteor', 'alignment_workspace', 'ptgui_pipeline'],
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
