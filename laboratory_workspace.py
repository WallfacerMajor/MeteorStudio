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
        self.events = queue.Queue()
        self.scheduler = BackgroundTaskScheduler(max_workers=1)
        self.title(f"实验室 · {MODES[mode][0]}")
        self.geometry("1000x720")
        self.minsize(740, 520)
        self.protocol("WM_DELETE_WINDOW", self._request_close)
        body = ttk.Frame(self, padding=24)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="N I G H T S C A P E   /   L A B", style="Muted.TLabel").pack(anchor="w")
        ttk.Label(body, text=MODES[mode][0], style="Hero.TLabel").pack(anchor="w", pady=8)
        ttk.Label(body, text=MODES[mode][1], wraplength=690, style="Muted.TLabel").pack(anchor="w", pady=(0, 14))
        actions = ttk.Frame(body)
        actions.pack(fill="x")
        self.add_button = ttk.Button(actions, text="添加照片…", command=self.add_files)
        self.add_button.pack(side="left")
        self.clear_button = ttk.Button(actions, text="清空列表", command=self.clear)
        self.clear_button.pack(side="left", padx=8)
        ttk.Button(actions, text="运行日志", command=lambda: show_runtime_log(self)).pack(side="right")
        self.status = tk.StringVar(value="支持 TIFF / PNG / JPG；原始素材始终只读")
        footer = ttk.Frame(body)
        footer.pack(side="bottom", fill="x", pady=(12, 0))
        self.start_button = ttk.Button(footer, text="开始实验", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(footer, text="取消", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=8)
        self.open_button = ttk.Button(footer, text="打开结果文件夹", command=self.open_result, state="disabled")
        self.open_button.pack(side="right")
        ttk.Label(body, textvariable=self.status, wraplength=690).pack(side="bottom", fill="x", pady=8)
        self.progress = ttk.Progressbar(body)
        self.progress.pack(side="bottom", fill="x")
        output = ttk.Frame(body)
        output.pack(side="bottom", fill="x", pady=14)
        ttk.Label(output, text="输出到").pack(side="left")
        self.destination = tk.StringVar()
        self.destination_entry = ttk.Entry(output, textvariable=self.destination)
        self.destination_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.browse_button = ttk.Button(output, text="选择…", command=self.choose_output)
        self.browse_button.pack(side="right")
        table = ttk.Frame(body)
        table.pack(fill="both", expand=True, pady=(12, 0))
        self.files = ttk.Treeview(table, columns=("path",), show="headings")
        self.files.heading("path", text="素材路径 · 保持添加顺序 · 重复文件自动忽略")
        self.files.column("path", width=900)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.files.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.files.xview)
        self.files.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.files.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.after(80, self.poll)

    def add_files(self):
        selected = filedialog.askopenfilenames(parent=self, title="添加实验素材", filetypes=[("图像", "*.tif *.tiff *.png *.jpg *.jpeg")])
        for name in selected:
            path = Path(name).resolve()
            if path not in self.paths:
                self.paths.append(path)
                self.files.insert("", "end", values=(str(path),))
        self.status.set(f"已添加 {len(self.paths)} 张照片")

    def clear(self):
        self.paths.clear()
        self.files.delete(*self.files.get_children())
        self.status.set("列表已清空")

    def choose_output(self):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            self.destination.set(folder)

    def set_busy(self, busy):
        self.busy = busy
        for widget in (self.start_button, self.add_button, self.clear_button, self.destination_entry, self.browse_button):
            widget.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def start(self):
        if self.busy:
            return
        if not self.paths or not self.destination.get().strip():
            self.status.set("请添加照片并选择独立的输出目录")
            return
        paths, destination = tuple(self.paths), self.destination.get().strip()
        self.set_busy(True)
        self.result = None
        self.open_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=len(paths))
        self.status.set("准备实验…")
        events, mode = self.events, self.mode
        def work(token):
            try:
                return run_experiment(paths, destination, mode, token,
                    lambda value, message: events.put(("progress", (value, message))))
            finally:
                if token.cancelled:
                    events.put(("cancelled", None))
        self.token = self.scheduler.submit("experiment", work,
            on_result=lambda path: events.put(("done", path)),
            on_error=lambda exc, detail: events.put(("error", (exc, detail))))

    def cancel(self):
        # Keep the current token registered so the worker can report cancellation.
        self.token.cancel()
        self.status.set("正在取消；等待当前读写结束…")
        self.cancel_button.configure(state="disabled")

    def poll(self):
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == "progress":
                self.progress.configure(value=value[0])
                self.status.set(value[1])
            elif kind == "done":
                self.set_busy(False)
                self.result = value
                self.open_button.configure(state="normal")
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
