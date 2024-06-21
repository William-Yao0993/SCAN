# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 10)
from PyInstaller.utils.hooks import collect_data_files
import ultralytics
ultra_files = collect_data_files('ultralytics')
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), ('models', 'models')] + ultra_files,
    hiddenimports=['ultralytics','openpyxl.cell._writer',],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['sklearn.externals.joblib'],
    noarchive=True,
)
pyz = PYZ(a.pure)
splash = Splash(
    'splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_default = 'initializing...',
    text_color = 'black',
    text_pos=(240,665),
    text_size=15,
    minify_script=True,
    always_on_top=False,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name='SCAN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SCAN',
)
