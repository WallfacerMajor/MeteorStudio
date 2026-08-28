"""Selectable, copyable error dialog shared by every MeteorStudio workspace."""

from __future__ import annotations

import tkinter as tk
import threading
from collections import deque
from datetime import datetime
from tkinter import ttk


_LOG_ENTRIES: deque[str] = deque(maxlen=500)
_LOG_LOCK = threading.Lock()


def append_runtime_log(message: object, details: object | None = None) -> None:
    """Retain useful terminal-style diagnostics inside the GUI process."""
    entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    detail_text = "" if details is None else str(details).strip()
    if detail_text:
        entry += "\n" + detail_text
    with _LOG_LOCK:
        _LOG_ENTRIES.append(entry)


def runtime_log_text() -> str:
    with _LOG_LOCK:
        values = list(_LOG_ENTRIES)
    return "\n\n".join(values) if values else "当前还没有错误或诊断信息。"


def show_runtime_log(parent: tk.Misc, title: str = "运行日志／错误详情") -> None:
    """Open a non-modal, refreshing replacement for an invisible EXE console."""
    owner = parent.winfo_toplevel()
    existing = getattr(owner, "_runtime_log_window", None)
    try:
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return
    except tk.TclError:
        pass

    window = tk.Toplevel(owner)
    owner._runtime_log_window = window
    window.title(title)
    window.geometry("820x520")
    window.minsize(520, 300)
    frame = ttk.Frame(window, padding=10)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text="打包版没有单独的命令行终端；后台错误和诊断信息会保留在这里。",
    ).pack(anchor="w", pady=(0, 8))
    body = ttk.Frame(frame)
    body.pack(fill="both", expand=True)
    text = tk.Text(body, wrap="none", font=("TkFixedFont", 10), padx=8, pady=8)
    y_scroll = ttk.Scrollbar(body, orient="vertical", command=text.yview)
    x_scroll = ttk.Scrollbar(body, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    text.grid(row=0, column=0, sticky="nsew")
    y_scroll.grid(row=0, column=1, sticky="ns")
    x_scroll.grid(row=1, column=0, sticky="ew")
    body.rowconfigure(0, weight=1)
    body.columnconfigure(0, weight=1)
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", pady=(8, 0))

    last_value = [None]
    def refresh():
        if not window.winfo_exists():
            return
        value = runtime_log_text()
        if value != last_value[0]:
            was_at_end = text.yview()[1] >= 0.98
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", value)
            text.configure(state="disabled")
            if was_at_end:
                text.see("end")
            last_value[0] = value
        window.after(500, refresh)

    def copy_all():
        owner.clipboard_clear()
        owner.clipboard_append(runtime_log_text())
        owner.update_idletasks()
        copy_button.configure(text="已复制")

    def clear_log():
        with _LOG_LOCK:
            _LOG_ENTRIES.clear()
        last_value[0] = None

    ttk.Button(buttons, text="清空日志", command=clear_log).pack(side="left")
    copy_button = ttk.Button(buttons, text="复制全部", command=copy_all)
    copy_button.pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="关闭", command=window.destroy).pack(side="right")
    window.bind("<Control-a>", lambda _event: (text.tag_add("sel", "1.0", "end-1c"), "break")[-1])
    window.bind("<Control-c>", lambda _event: (copy_all(), "break")[-1])
    window.bind("<Escape>", lambda _event: window.destroy())
    refresh()
    window.lift()


def show_copyable_error(
    title: str,
    message: object,
    *,
    parent: tk.Misc | None = None,
    details: object | None = None,
) -> None:
    """Show a modal error whose complete contents can be copied.

    ``tkinter.messagebox`` renders static text, which prevents users from
    selecting a traceback or pasting it into an issue report.  This replacement
    keeps the familiar modal behavior while exposing selectable text, Ctrl+A,
    Ctrl+C, and a one-click copy action.
    """
    message_text = str(message)
    detail_text = "" if details is None else str(details).strip()
    complete_text = message_text
    if detail_text and detail_text not in message_text:
        complete_text += "\n\n详细信息：\n" + detail_text
    append_runtime_log(f"{title}: {message_text}", detail_text)

    owner = parent
    try:
        dialog = tk.Toplevel(owner)
    except (tk.TclError, RuntimeError):
        return
    dialog.title(str(title))
    dialog.geometry("720x430")
    dialog.minsize(480, 260)
    if owner is not None:
        try:
            dialog.transient(owner.winfo_toplevel())
        except tk.TclError:
            pass

    outer = ttk.Frame(dialog, padding=14)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="发生错误", font=("TkDefaultFont", 11, "bold")).pack(
        anchor="w", pady=(0, 8)
    )
    text_frame = ttk.Frame(outer)
    text_frame.pack(fill="both", expand=True)
    text = tk.Text(
        text_frame, wrap="word", undo=False, relief="solid", borderwidth=1,
        padx=10, pady=10, font=("TkFixedFont", 10), cursor="xterm",
    )
    scroll = ttk.Scrollbar(text_frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    text.insert("1.0", complete_text)
    text.configure(state="disabled")

    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(12, 0))
    hint = ttk.Label(buttons, text="可直接选中文字，或按 Ctrl+C 复制")
    hint.pack(side="left")

    def copy_all(_event=None):
        try:
            dialog.clipboard_clear()
            dialog.clipboard_append(complete_text)
            dialog.update_idletasks()
            copy_button.configure(text="已复制")
        except tk.TclError:
            pass
        return "break"

    def select_all(_event=None):
        text.tag_add("sel", "1.0", "end-1c")
        text.mark_set("insert", "1.0")
        text.see("1.0")
        return "break"

    copy_button = ttk.Button(buttons, text="复制错误信息", command=copy_all)
    copy_button.pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="关闭", command=dialog.destroy).pack(side="right")

    text.bind("<Control-a>", select_all)
    text.bind("<Control-A>", select_all)
    dialog.bind("<Control-c>", copy_all)
    dialog.bind("<Control-C>", copy_all)
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    dialog.update_idletasks()
    try:
        if owner is not None:
            root_x = owner.winfo_rootx()
            root_y = owner.winfo_rooty()
            root_w = owner.winfo_width()
            root_h = owner.winfo_height()
            width = dialog.winfo_width()
            height = dialog.winfo_height()
            dialog.geometry(
                f"+{max(0, root_x + (root_w - width) // 2)}"
                f"+{max(0, root_y + (root_h - height) // 2)}"
            )
        dialog.grab_set()
        text.focus_set()
        dialog.wait_window()
    except tk.TclError:
        pass
