# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

datas = []
datas += collect_data_files('setuptools._vendor.jaraco.text')
datas += copy_metadata('numpy')
datas += copy_metadata('pillow')
datas += copy_metadata('tifffile')


a = Analysis(
    ['app.py'],
    pathex=[str(__import__('pathlib').Path(SPECPATH).parents[1])],
    binaries=[],
    datas=datas,
    hiddenimports=['app'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'torch', 'scipy', 'pandas', 'IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StarReductionCompare',
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
    name='StarReductionCompare',
)
