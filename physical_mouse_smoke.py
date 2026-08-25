"""Drive a packaged MeteorStudio instance with real Windows cursor input."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

from PIL import ImageGrab


ROOT = Path(__file__).resolve().parent
STATE = ROOT / "physical_mouse_proof" / "state.json"
USER32 = ctypes.windll.user32
USER32.SetProcessDPIAware()
SW_RESTORE = 9
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = [("type", wintypes.DWORD), ("data", INPUT_UNION)]


def send_mouse(flags: int) -> None:
    value = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, flags, 0, None))
    if USER32.SendInput(1, ctypes.byref(value), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError()


def find_window(pid: int) -> int:
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        process_id = wintypes.DWORD()
        USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == pid and USER32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False
        return True

    USER32.EnumWindows(callback, 0)
    return found[0] if found else 0


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not USER32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError()
    return rect.left, rect.top, rect.right, rect.bottom


def focus(hwnd: int) -> None:
    USER32.ShowWindow(hwnd, SW_RESTORE)
    USER32.SetForegroundWindow(hwnd)
    time.sleep(0.4)


def move(x: int, y: int) -> None:
    USER32.SetCursorPos(x, y)
    time.sleep(0.5)


def click(x: int, y: int, count: int = 1) -> None:
    move(x, y)
    for _ in range(count):
        send_mouse(MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.08)
        send_mouse(MOUSEEVENTF_LEFTUP)
        time.sleep(0.16)


def drag(start: tuple[int, int], end: tuple[int, int], steps: int = 18) -> None:
    move(*start)
    send_mouse(MOUSEEVENTF_LEFTDOWN)
    for index in range(1, steps + 1):
        amount = index / steps
        x = round(start[0] + (end[0] - start[0]) * amount)
        y = round(start[1] + (end[1] - start[1]) * amount)
        USER32.SetCursorPos(x, y)
        time.sleep(0.025)
    send_mouse(MOUSEEVENTF_LEFTUP)
    time.sleep(1.0)


def screenshot(hwnd: int, name: str) -> Path:
    output = ROOT / "physical_mouse_proof"
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    ImageGrab.grab(bbox=window_rect(hwnd), all_screens=True).save(path)
    return path


def read_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def launch() -> None:
    output = ROOT / "physical_mouse_proof"
    isolated = output / "appdata"
    autosave = isolated / "MeteorComposer" / "autosave.json"
    autosave.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(os.environ["APPDATA"]) / "MeteorComposer" / "autosave.json", autosave)
    exe = ROOT / "dist_local_verified_014" / "MeteorStudio" / "MeteorStudio.exe"
    environment = os.environ.copy()
    environment["APPDATA"] = str(isolated)
    process = subprocess.Popen([str(exe)], env=environment, cwd=str(exe.parent))
    hwnd = 0
    for _ in range(80):
        hwnd = find_window(process.pid)
        if hwnd:
            break
        time.sleep(0.1)
    if not hwnd:
        process.kill()
        raise RuntimeError("MeteorStudio window was not created")
    focus(hwnd)
    time.sleep(4)
    state = {"pid": process.pid, "hwnd": hwnd, "rect": window_rect(hwnd), "autosave": str(autosave)}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    screenshot(hwnd, "01_launched.png")
    print(json.dumps(state))


def capture(name: str) -> None:
    state = read_state()
    hwnd = int(state["hwnd"])
    focus(hwnd)
    path = screenshot(hwnd, name)
    print(path)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "launch":
        launch()
    elif command == "capture":
        capture(sys.argv[2])
    elif command == "click":
        state = read_state()
        focus(int(state["hwnd"]))
        click(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif command == "move":
        state = read_state()
        focus(int(state["hwnd"]))
        move(int(sys.argv[2]), int(sys.argv[3]))
    elif command == "drag":
        state = read_state()
        focus(int(state["hwnd"]))
        drag((int(sys.argv[2]), int(sys.argv[3])), (int(sys.argv[4]), int(sys.argv[5])))
    else:
        raise SystemExit(f"Unknown command: {command}")
