"""Shared runtime log storage and overlay panel for all workspaces and companions.

SPDX-License-Identifier: MIT

Copyright (c) 2026 MeteorStudio contributors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from __future__ import annotations

import tkinter as tk
import threading
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import ttk


_LOG_ENTRIES: deque[str] = deque(maxlen=500)
_LOG_LOCK = threading.Lock()


def _runtime_log_path() -> Path:
    if os.environ.get("METEOR_RUNTIME_LOG"):
        return Path(os.environ["METEOR_RUNTIME_LOG"])
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return root / "MeteorComposer" / "runtime.log"


def append_runtime_log(message: object, details: object | None = None) -> None:
    """Retain useful terminal-style diagnostics inside the GUI process."""
    entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    detail_text = "" if details is None else str(details).strip()
    if detail_text:
        entry += "\n" + detail_text
    with _LOG_LOCK:
        _LOG_ENTRIES.append(entry)
        try:
            path = _runtime_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(entry + "\n\n")
            # Never truncate a shared file while the companion may append.
        except OSError:
            pass


def runtime_log_text() -> str:
    with _LOG_LOCK:
        try:
            path = _runtime_log_path()
            if path.is_file():
                with path.open('rb') as handle:
                    handle.seek(max(0,path.stat().st_size-(2 << 20)))
                    value=handle.read().decode('utf-8',errors='replace').strip()
                return value or "当前还没有错误或诊断信息。"
        except OSError:
            pass
        values = list(_LOG_ENTRIES)
        return "\n\n".join(values) if values else "当前还没有错误或诊断信息。"


def _icon_button(parent, kind, hint, command):
    button=ttk.Button(parent,text=hint,width=3,command=command)
    picture=tk.PhotoImage(master=parent,width=22,height=22)
    if kind=='close':
        for x in range(5,17):
            picture.put('#eeeeee',to=(x,x,x+2,x+2));picture.put('#eeeeee',to=(x,21-x,x+2,23-x))
    elif kind=='copy':
        for left,top,right,bottom in ((5,7,14,18),(8,3,18,14)):
            for box in ((left,top,right,top+1),(left,bottom,right,bottom+1),(left,top,left+1,bottom),(right,top,right+1,bottom)):
                picture.put('#eeeeee',to=box)
    else:
        for y in (6,11,16):picture.put('#eeeeee',to=(5,y,18,y+2))
        if kind=='clear':picture.put('#eeeeee',to=(9,3,11,20))
    button.configure(image=picture,compound='none');button._log_icon=picture
    button._action_icon=True;button._action_key='runtime_'+kind
    popup=[None]
    def hide(event=None):
        if popup[0] is not None:
            try:popup[0].destroy()
            except tk.TclError:pass
            popup[0]=None
    def show(event=None):
        hide();popup[0]=tk.Toplevel(button);popup[0].overrideredirect(True)
        ttk.Label(popup[0],text=hint,padding=6).pack()
        popup[0].geometry(f'+{button.winfo_rootx()}+{button.winfo_rooty()+button.winfo_height()+2}')
    button.bind('<Enter>',show);button.bind('<Leave>',hide);button.bind('<Destroy>',hide);button.bind('<Unmap>',hide)
    return button


class LogPanel(ttk.Frame):
    def __init__(self, owner):
        super().__init__(owner, padding=8, relief='solid', borderwidth=1)
        self.owner, self.timer, self.last = owner, None, None
        bar = ttk.Frame(self); bar.pack(fill='x')
        ttk.Label(bar, text='运行日志').pack(side='left')
        self.hide_button = _icon_button(bar,'close','收起日志 · Ctrl+L',self.hide)
        self.hide_button.pack(side='right')
        self.copy_button = _icon_button(bar,'copy','复制日志',self.copy)
        self.copy_button.pack(side='right')
        self.clear_button = _icon_button(bar,'clear','清空日志',self.clear)
        self.clear_button.pack(side='right')
        for button, hint in ((self.hide_button,'收起日志 · Ctrl+L'),(self.copy_button,'复制日志'),(self.clear_button,'清空日志')):
            button.bind('<Enter>', lambda e, h=hint: self.hint.configure(text=h),add=True)
            button.bind('<Leave>', lambda e: self.hint.configure(text=''),add=True)
        self.hint=ttk.Label(bar,text='');self.hint.pack(side='right',padx=8)
        body=ttk.Frame(self);body.pack(fill='both',expand=True,pady=(5,0))
        self.text=tk.Text(body,wrap='word',background='#252525',foreground='#eeeeee',insertbackground='#eeeeee',font='TkFixedFont',height=9)
        scroll=ttk.Scrollbar(body,command=self.text.yview)
        scroll.pack(side='right',fill='y');self.text.pack(fill='both',expand=True)
        self.text.configure(yscrollcommand=scroll.set,state='disabled')
        self.bind('<Destroy>', self.destroyed, add=True)

    def show(self):
        self.place(relx=0, rely=1, relwidth=1, height=min(260,max(130,self.owner.winfo_height()//3)),anchor='sw')
        self.lift();self.refresh()

    def hide(self):
        if self.timer:
            self.after_cancel(self.timer);self.timer=None
        self.place_forget()

    def destroyed(self,event):
        if event.widget is self and self.timer:
            self.after_cancel(self.timer);self.timer=None

    def refresh(self):
        if self.timer:self.after_cancel(self.timer)
        value=runtime_log_text()
        if value!=self.last:
            at_end=self.text.yview()[1]>=.98
            self.text.configure(state='normal');self.text.delete('1.0','end');self.text.insert('1.0',value);self.text.configure(state='disabled')
            if at_end:self.text.see('end')
            self.last=value
        self.timer=self.after(500,self.refresh)

    def copy(self):
        self.clipboard_clear();self.clipboard_append(runtime_log_text())
        self.hint.configure(text='已复制')

    def clear(self):
        with _LOG_LOCK:
            _LOG_ENTRIES.clear()
            try:_runtime_log_path().write_text('',encoding='utf-8')
            except OSError:pass
        self.last=None;self.refresh()


def show_runtime_log(parent, title='运行日志'):
    owner=parent.winfo_toplevel()
    panel=getattr(owner,'_runtime_log_panel',None)
    if panel is None or not panel.winfo_exists():
        panel=owner._runtime_log_panel=LogPanel(owner)
    panel.show()
    return panel


def toggle_runtime_log(parent):
    owner=parent.winfo_toplevel();panel=getattr(owner,'_runtime_log_panel',None)
    if panel is not None and panel.winfo_exists() and panel.winfo_manager():panel.hide()
    else:show_runtime_log(owner)
    return 'break'


def install_log_access(owner, toolbar):
    owner=owner.winfo_toplevel()
    button=_icon_button(toolbar,'log','运行日志 · Ctrl+L',lambda:toggle_runtime_log(owner))
    button.pack(side='right',padx=4)
    # Already an icon; the main application's generic icon pass must preserve it.
    button._action_icon=True
    owner.bind('<Control-l>',lambda e:toggle_runtime_log(owner))
    owner.bind('<Control-L>',lambda e:toggle_runtime_log(owner))
    if sys.platform=='darwin':owner.bind('<Command-l>',lambda e:toggle_runtime_log(owner))
    return button
