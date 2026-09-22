"""Selectable error dialogs using the application-wide runtime log."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from runtime_log import append_runtime_log, runtime_log_text, show_runtime_log


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

    from action_icons import iconize_actions
    iconize_actions(dialog)
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
