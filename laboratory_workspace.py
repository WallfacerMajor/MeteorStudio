"""Single laboratory workspace with cancellable background processing."""
import queue
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from background_tasks import BackgroundTaskScheduler
from error_dialog import show_copyable_error, show_runtime_log
from platform_utils import open_folder
from ui_navigation import cancel_widget_timers
from laboratory import MODES, run_experiment


class LaboratoryWindow(tk.Toplevel):
    def __init__(self, master, mode):
        super().__init__(master)
        self.mode = mode
        self.paths = []
        self.result = None
        self.busy = False
        self.run_id = 0
        self.events = queue.Queue()
        self.scheduler = BackgroundTaskScheduler(max_workers=1)
        self.title(f"实验室 · {MODES[mode][0]}")
        self.geometry("1000x720")
        self.minsize(740, 520)
        self.protocol("WM_DELETE_WINDOW", self._request_close)
        body = ttk.Frame(self, padding=(20, 16))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=MODES[mode][0], style="Hero.TLabel").pack(anchor="w", pady=8)
        inspector = ttk.Frame(body, width=260)
        self.edit_inspector = inspector
        inspector.pack(side="right", fill="y", padx=(12, 0))
        inspector.pack_propagate(False)
        actions = ttk.Frame(inspector)
        actions.pack(fill="x")
        self.add_button = ttk.Button(actions, text="添加照片…", command=self.add_files)
        self.add_button.pack(side="left")
        self.remove_button = ttk.Button(actions, text="移除选中", command=self.remove_selected, state="disabled")
        self.remove_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(actions, text="清空列表", command=self.clear)
        self.clear_button.pack(side="left", padx=8)
        self.status = tk.StringVar(value="")
        self.summary = tk.StringVar()
        ttk.Label(body, textvariable=self.summary, style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
        footer = ttk.Frame(inspector)
        footer.pack(side="bottom", fill="x", pady=(12, 0))
        self.start_button = ttk.Button(footer, text="开始实验", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(footer, text="取消", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(footer, text="打开结果文件夹", command=self.open_result, state="disabled")
        self.open_button.pack(side="right")
        status_label = ttk.Label(body, textvariable=self.status, wraplength=690)
        status_label.pack(side="bottom", fill="x", pady=8)
        body.bind("<Configure>", lambda event: status_label.configure(wraplength=max(300, event.width - 40)))
        self.progress = ttk.Progressbar(body)
        self.progress.pack(side="bottom", fill="x")
        output = ttk.Frame(inspector)
        output.pack(side="bottom", fill="x", pady=14)
        ttk.Label(output, text="输出到").pack(side="left")
        self.destination = tk.StringVar()
        self.destination_entry = ttk.Entry(output, textvariable=self.destination)
        self.destination_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.browse_button = ttk.Button(output, text="选择…", command=self.choose_output)
        self.browse_button.pack(side="right")
        table = ttk.Frame(body)
        table.pack(fill="both", expand=True, pady=(12, 0))
        self.files = ttk.Treeview(table, columns=("name", "format", "path"), show="headings", selectmode="extended")
        for column, title, width in (("name", "照片", 210), ("format", "格式", 75), ("path", "所在文件夹", 520)):
            self.files.heading(column, text=title)
            self.files.column(column, width=width, minwidth=60, stretch=column != "format")
        self.files.bind("<<TreeviewSelect>>", lambda _: self.refresh_controls())
        self.files.bind("<Delete>", self.remove_selected)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.files.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.files.xview)
        self.files.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.files.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        from workspace_layout import scroll_controls
        scroll_controls(inspector, 235)
        self.destination.trace_add("write", lambda *_: self.refresh_controls())
        self.refresh_controls()
        from action_icons import iconize_actions
        iconize_actions(self)
        self.after(80, self.poll)

    def add_files(self):
        if self.busy:
            return
        selected = filedialog.askopenfilenames(parent=self, title="添加实验素材", filetypes=[("图像", "*.tif *.tiff *.png *.jpg *.jpeg")])
        known = set(self.paths)
        added = 0
        for name in selected:
            path = Path(name).resolve()
            if path not in known:
                known.add(path)
                self.paths.append(path)
                self.files.insert("", "end", values=(path.name, path.suffix[1:].upper(), str(path.parent)))
                added += 1
        if selected:
            self.status.set(f"添加 {added} 张 · 忽略 {len(selected) - added} 张重复照片")
        self.refresh_controls()

    def remove_selected(self, event=None):
        if self.busy:
            return "break"
        selected = set(self.files.selection())
        rows = self.files.get_children()
        self.paths = [p for row, p in zip(rows, self.paths) if row not in selected]
        self.files.delete(*selected)
        self.status.set(f"已从列表移除 {len(selected)} 张")
        self.refresh_controls()
        return "break"

    def clear(self):
        if self.busy:
            return
        self.paths.clear()
        self.files.delete(*self.files.get_children())
        self.status.set("列表已清空")
        self.refresh_controls()

    def choose_output(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            self.destination.set(folder)

    def set_busy(self, busy):
        self.busy = busy
        self.refresh_controls()
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def refresh_controls(self):
        minimum = 1 if self.mode == "quality" else 2
        self.summary.set(f"{len(self.paths)} 张照片  ·  {'体检至少 1 张' if minimum == 1 else '叠加至少 2 张'}")
        for widget in (self.add_button, self.destination_entry, self.browse_button):
            widget.configure(state="disabled" if self.busy else "normal")
        self.clear_button.configure(state="normal" if self.paths and not self.busy else "disabled")
        self.remove_button.configure(state="normal" if self.files.selection() and not self.busy else "disabled")
        ready = len(self.paths) >= minimum and not self.busy
        self.start_button.configure(state="normal" if ready else "disabled")
        self.start_button._disabled_reason="正在处理" if self.busy else f"请至少添加 {minimum} 张照片" if len(self.paths)<minimum else ""

    def start(self):
        if self.busy:
            return
        minimum=1 if self.mode=="quality" else 2
        if len(self.paths)<minimum:return
        if not self.destination.get().strip():
            self.choose_output()
            if not self.destination.get().strip():return
        paths, destination = tuple(self.paths), self.destination.get().strip()
        self.set_busy(True)
        # Keep the last completed result available if a later run fails/cancels.
        self.open_button.configure(text="打开上次结果" if self.result else "打开结果文件夹")
        self.progress.configure(value=0, maximum=len(paths))
        self.status.set("准备实验…")
        self.run_id += 1
        events, mode, run_id = self.events, self.mode, self.run_id
        def send(kind, value):
            events.put((run_id, kind, value))
        def work(token):
            completed = False
            try:
                result = run_experiment(paths, destination, mode, token,
                    lambda value, message: send("progress", (value, message)))
                # A committed output wins over a cancellation arriving after
                # the last write; do not lose its path in scheduler filtering.
                completed = True
                send("done", result)
            finally:
                if token.cancelled and not completed:
                    send("cancelled", None)
        self.token = self.scheduler.submit("experiment", work,
            on_error=lambda exc, detail: send("error", (str(exc), detail)))

    def cancel(self):
        if not self.busy:
            return
        # Keep the current token registered so the worker can report cancellation.
        self.token.cancel()
        self.status.set("正在取消；等待当前读写结束…")
        self.cancel_button.configure(state="disabled")

    def poll(self):
        # Bound work per UI tick so a long batch cannot starve pointer events.
        for _ in range(100):
            try:
                run_id, kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if run_id != self.run_id:
                continue
            if kind == "progress":
                self.progress.configure(value=value[0])
                if not self.token.cancelled:
                    self.status.set(value[1])
            elif kind == "done":
                self.set_busy(False)
                self.result = value
                self.open_button.configure(state="normal", text="打开结果文件夹")
                self.status.set(f"已完成 · {value}")
            elif kind == "cancelled":
                self.set_busy(False)
                self.status.set("已取消；未完成的结果已在 experiment.json 标记")
            else:
                self.set_busy(False)
                self.status.set("实验未完成；详情见运行日志")
                show_copyable_error("实验室", str(value[0]), parent=self, details=value[1])
        if self.busy and self.token.cancelled and self.scheduler.active_count() == 0:
            self.set_busy(False)
            self.status.set("已取消")
        if self.winfo_exists():
            self.after(80, self.poll)

    def open_result(self):
        if self.result:
            try:
                open_folder(self.result)
            except OSError as exc:
                show_copyable_error("打开结果", str(exc), parent=self)

    def _request_close(self):
        self.scheduler.shutdown(wait=False)
        cancel_widget_timers(self)
        self.destroy()
