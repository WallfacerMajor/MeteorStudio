"""Exercise the laboratory entry and the real separate Siril comparison tool."""
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch


def run_smoke(app):
    from toolbox_smoke import click, pump
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        app.autosave_path = root / 'autosave.json'
        app.show_toolbox()
        # A packaged process launched with SW_HIDE needs a fresh mapping before
        # native hit-testing; wait for the actual window, not merely Tk layout.
        app.withdraw(); app.update_idletasks(); app.deiconify()
        app.geometry('1200x850+0+0'); app.lift(); pump(app, .8)
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.WinDLL('user32')
            user32.GetAncestor.argtypes = (wintypes.HWND, wintypes.UINT)
            user32.GetAncestor.restype = wintypes.HWND
            user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
            user32.ShowWindow(user32.GetAncestor(app.winfo_id(), 2), 9)
            app.focus_force(); pump(app, .4)
        button = app.toolbox_home.tool_buttons['star_reduction']
        assert button._action_key == 'star_reduce'
        report = root / 'comparison.json'
        fixtures = Path(os.environ.get('STAR_COMPARE_SMOKE_ROOT', 'experiments/star_reduction_compare')).resolve()
        log_path=root/'runtime.log'
        with patch.dict(os.environ, STAR_COMPARE_SMOKE_REPORT=str(report), STAR_COMPARE_SMOKE_ROOT=str(fixtures), METEOR_RUNTIME_LOG=str(log_path)):
            from runtime_log import append_runtime_log
            append_runtime_log('工具箱共享日志回归')
            try:
                click(app, button)
            except AssertionError as exc:
                from PIL import ImageGrab
                output = Path(os.environ.get('METEOR_STAR_LAB_SMOKE_REPORT', 'review-artifacts/star-debug.json'))
                ImageGrab.grab((app.winfo_rootx(), app.winfo_rooty(), app.winfo_rootx()+app.winfo_width(), app.winfo_rooty()+app.winfo_height())).save(output.with_suffix('.png'))
                raise AssertionError((str(exc), 'window', app.state(), app.geometry(), 'button', button.winfo_ismapped(), button.winfo_rootx(), button.winfo_rooty(), button.winfo_width(), button.winfo_height(), 'home', app.toolbox_home.winfo_ismapped(), 'focus', str(app.focus_get()))) from exc
        process = app.star_reduction_process
        assert process is not None and app.state() == 'withdrawn'
        app.open_star_reduction_workspace()
        assert app.star_reduction_process is process, 'Duplicate instance created'
        deadline = time.monotonic() + 120
        while app.star_reduction_process is not None:
            pump(app, .1)
            assert time.monotonic() < deadline, 'Comparison timed out'
        assert process.returncode == 0, report.read_text(encoding='utf-8') if report.exists() else process.returncode
        result = json.loads(report.read_text(encoding='utf-8'))
        assert all(value == 'passed' for value in result.values()), result
        assert app.state() != 'withdrawn' and app.toolbox_home.winfo_ismapped()
        with patch.dict(os.environ,METEOR_RUNTIME_LOG=str(log_path)):
            from runtime_log import show_runtime_log
            panel=show_runtime_log(app);pump(app,.6)
            text=panel.text.get('1.0','end')
            assert '工具箱共享日志回归' in text and '缩星保存完成' in text
            if os.environ.get('METEOR_STAR_LAB_SMOKE_REPORT'):
                Path(os.environ['METEOR_STAR_LAB_SMOKE_REPORT']).with_suffix('.log').write_text(text,encoding='utf-8')
            click(app,panel.hide_button);assert not panel.winfo_ismapped()
        result['shared_cross_process_log']='passed'
        result.update(laboratory_entry='passed', single_workspace='passed', return_to_toolbox='passed')
        return result


if __name__ == '__main__':
    from meteor_composer import MeteorComposer
    with patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        app = MeteorComposer()
    try:
        print(json.dumps(run_smoke(app), ensure_ascii=False))
    finally:
        app.destroy()
