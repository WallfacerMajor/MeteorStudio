"""Build the separately licensed laboratory companion on the current platform."""
from pathlib import Path
import shutil
import subprocess
import sys


def build():
    root = Path(__file__).resolve().parent / 'experiments' / 'star_reduction_compare'
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', 'StarReductionCompare.spec'], cwd=root, check=True)
    source = root / 'dist' / 'StarReductionCompare' / 'source'
    source.mkdir(parents=True, exist_ok=True)
    for name in ('app.py', 'engine.py', 'smoke.py', 'test_engine.py', 'StarReductionCompare.spec', 'COPYING', 'THIRD_PARTY.md', 'README.md'):
        shutil.copy2(root / name, source / name)
    shutil.copy2(root.parents[1] / 'runtime_log.py', source / 'runtime_log.py')


if __name__ == '__main__':
    build()
