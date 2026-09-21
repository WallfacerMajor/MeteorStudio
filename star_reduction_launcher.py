"""Run the separately licensed comparison application as a laboratory tool.

The GPL comparison program remains a separate executable, with its source and
license shipped beside it. Neither its engine nor the DSA expression is imported
into the toolbox process.
"""
import os
from pathlib import Path
import subprocess
import sys


def command():
    if getattr(sys, 'frozen', False):
        directory = Path(sys._MEIPASS) / 'star_reduction'
        executable = directory / ('StarReductionCompare.exe' if os.name == 'nt' else 'StarReductionCompare')
        if not executable.is_file():
            raise FileNotFoundError('缩星组件缺失，请使用完整的软件目录')
        return [str(executable), '--toolbox']
    return [sys.executable, str(Path(__file__).parent / 'experiments/star_reduction_compare/app.py'), '--toolbox']


def open_star_reduction(app):
    from toolbox import SoftwareRegistry
    from error_dialog import show_copyable_error
    previous = getattr(app, 'star_reduction_process', None)
    if previous is not None and previous.poll() is None:
        return
    env = os.environ.copy()
    siril = SoftwareRegistry().resolve('siril')
    if siril:
        env['METEOR_STAR_SIRIL'] = str(siril)
    try:
        process = subprocess.Popen(command(), env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    except OSError as exc:
        show_copyable_error('缩星', str(exc), parent=app)
        return
    app.star_reduction_process = process
    app.withdraw()
    def poll():
        code = process.poll()
        if code is None:
            app.after(150, poll)
            return
        app.star_reduction_process = None
        app.show_toolbox()
        app.deiconify()
        app.lift()
        if code:
            show_copyable_error('缩星', f'缩星组件退出，错误代码 {code}。请检查组件运行日志。', parent=app)
    app.after(150, poll)
