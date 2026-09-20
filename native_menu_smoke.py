"""Windows native popup rendering regression, with real posted menu events."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import threading
import time
import tkinter as tk
from PIL import ImageGrab


def run_native_menu_smoke(root):
    if sys.platform != 'win32':
        return {'native_menu': 'not_applicable'}
    from toolbox import tool_menu_button, settings_menu_button
    from tkinter import ttk
    user = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def menus():
        found = []
        def visit(hwnd, _):
            pid = wintypes.DWORD()
            user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            name = ctypes.create_unicode_buffer(128)
            user.GetClassNameW(hwnd, name, 128)
            if pid.value == os.getpid() and name.value == '#32768' and user.IsWindowVisible(hwnd):
                found.append(hwnd)
            return True
        user.EnumWindows(callback_type(visit), 0)
        return found
    def key(hwnd, code):
        user.PostMessageW(hwnd, 0x100, code, 0)
        user.PostMessageW(hwnd, 0x101, code, 0)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill='x')
    tools = tool_menu_button(frame, root)
    tools.pack(side='left')
    settings = settings_menu_button(frame)
    settings.pack(side='left')
    context = tk.Menu(frame, tearoff=False)
    context.add_command(label='锁定蒙版')
    context.add_command(label='删除蒙版')
    probe = ttk.Label(frame, text='右键蒙版测试')
    probe.pack(side='right')
    probe.bind('<Button-3>', lambda e: context.tk_popup(probe.winfo_rootx(), probe.winfo_rooty()+probe.winfo_height()))
    root.attributes('-topmost', True)
    root.update()
    screen_width = root.winfo_screenwidth()
    def capture(rect):
        desktop = ImageGrab.grab()
        scale = desktop.width / screen_width
        return desktop.crop(tuple(round(v*scale) for v in (rect.left, rect.top, rect.right, rect.bottom)))
    output = Path(os.environ.get('NATIVE_MENU_SCREENSHOTS', 'review-artifacts/native-menus'))
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for label, button, down in (('tools', tools, 1), ('settings', settings, 2), ('context', probe, 0)):
        failures = []
        def observer():
            try:
                deadline = time.monotonic()+5
                while not menus() and time.monotonic() < deadline:
                    time.sleep(.05)
                handles = menus()
                assert handles, 'Native popup did not open'
                time.sleep(.3)
                hwnd = handles[0]
                rect = wintypes.RECT()
                user.GetWindowRect(hwnd, ctypes.byref(rect))
                picture = capture(rect)
                picture.save(output/f'{label}.png')
                interior = picture.crop((8, 8, picture.width-8, picture.height-8)).convert('RGB')
                # This runner uses the normal Windows light menu palette.
                assert sum(1 for rgb in interior.getdata() if max(rgb)<150) > 60, 'Popup text is not visible'
                for _ in range(down):
                    key(hwnd, 0x28)
                    time.sleep(.12)
                if down:
                    key(hwnd, 0x27)
                    time.sleep(.4)
                    handles = menus()
                    assert len(handles) >= 2, 'Submenu did not open'
                for index, handle in enumerate(handles):
                    user.GetWindowRect(handle, ctypes.byref(rect))
                    capture(rect).save(output/f'{label}-cascade-{index}.png')
                reports[label] = 'passed'
            except BaseException as exc:
                failures.append(str(exc))
            finally:
                for _ in range(5):
                    for hwnd in menus():
                        key(hwnd, 0x1b)
                    time.sleep(.1)
        worker = threading.Thread(target=observer, daemon=True)
        worker.start()
        number = 3 if label == 'context' else 1
        button.event_generate(f'<ButtonPress-{number}>', x=button.winfo_width()//2, y=button.winfo_height()//2)
        button.event_generate(f'<ButtonRelease-{number}>', x=button.winfo_width()//2, y=button.winfo_height()//2)
        while worker.is_alive():
            root.update()
            time.sleep(.02)
        assert not failures, failures
        if root.grab_current():
            root.grab_release()
    frame.destroy()
    return reports


if __name__ == '__main__':
    from ui_theme import apply_theme
    root = tk.Tk()
    root.geometry('600x320+40+40')
    root.navigate_tool = lambda *args: None
    apply_theme(root)
    try:
        print(run_native_menu_smoke(root))
    finally:
        root.destroy()
