"""White balance workspace with replaceable viewport jobs and 16-bit export."""
import json
import queue
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from background_tasks import BackgroundTaskScheduler
from error_dialog import show_copyable_error, show_runtime_log
from platform_utils import open_folder
from ui_navigation import cancel_widget_timers
from white_balance import read_source, make_pyramid, render_view, sample_neutral, export_image, validate_settings


class WhiteBalanceWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("白平衡 · 星野工具箱")
        self.geometry("1180x780")
        self.minsize(850, 600)
        self.scheduler = BackgroundTaskScheduler(max_workers=2)
        self.events = queue.Queue()
        self.source = None
        self.levels = None
        self.result = None
        self.busy = False
        self.load_id = self.render_id = self.export_id = 0
        self.preview_after = None
        self.neutral = [1, 1, 1]
        self.zoom, self.center, self.fit_mode = 1., (0., 0.), True
        self.drag = None
        self.photo = None
        self.warmth, self.tint = tk.DoubleVar(value=0), tk.DoubleVar(value=0)
        self.original, self.picker = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.destination = tk.StringVar()
        self.status = tk.StringVar(value="打开照片开始 · 支持 sRGB TIFF / PNG / JPG 和 RAW")
        self.info = tk.StringVar(value="原片只读 · 调整可重置")
        self.gain_info = tk.StringVar(value="中性点：未取样")
        body = ttk.Frame(self, padding=(16, 12))
        body.pack(fill="both", expand=True)
        header = ttk.Frame(body)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="白平衡", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  调整冷暖，保留星野细节", style="Muted.TLabel").pack(side="left")
        self.open_button = ttk.Button(header, text="打开照片…", command=self.open_image)
        self.open_button.pack(side="right")
        footer = ttk.Frame(body)
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        self.export_button = ttk.Button(footer, text="导出 16 位 TIFF", style="Accent.TButton", command=self.export)
        self.export_button.pack(side="left")
        self.cancel_button = ttk.Button(footer, text="取消导出", state="disabled", command=self.cancel_export)
        self.cancel_button.pack(side="left", padx=6)
        self.folder_button = ttk.Button(footer, text="打开结果", state="disabled", command=self.open_result)
        self.folder_button.pack(side="right")
        ttk.Button(footer, text="运行日志", command=lambda: show_runtime_log(self)).pack(side="right", padx=6)
        self.progress = ttk.Progressbar(body, maximum=100)
        self.progress.pack(side="bottom", fill="x", pady=(6, 0))
        self.status_label = ttk.Label(body, textvariable=self.status, wraplength=850)
        self.status_label.pack(side="bottom", fill="x", pady=(6, 0))
        body.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(400, e.width-32)))
        output = ttk.Frame(body)
        output.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(output, text="输出到").pack(side="left")
        self.output_entry = ttk.Entry(output, textvariable=self.destination)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.output_button = ttk.Button(output, text="选择…", command=self.choose_output)
        self.output_button.pack(side="right")
        panel = ttk.Frame(body)
        panel.pack(fill="both", expand=True)
        sidebar = ttk.Frame(panel, width=245)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)
        self.control_canvas = tk.Canvas(sidebar, width=225, background="#101824", highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=self.control_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.control_canvas.pack(fill="both", expand=True)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)
        controls = ttk.Frame(self.control_canvas, padding=(0, 8, 8, 0))
        control_item = self.control_canvas.create_window(0, 0, window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        self.control_canvas.bind("<Configure>", lambda e: self.control_canvas.itemconfigure(control_item, width=e.width))
        self.control_canvas.bind("<MouseWheel>", lambda e: self.control_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.control_canvas.bind("<Button-4>", lambda e: self.control_canvas.yview_scroll(-1, "units"))
        self.control_canvas.bind("<Button-5>", lambda e: self.control_canvas.yview_scroll(1, "units"))
        ttk.Label(controls, text="相对调整", style="Title.TLabel").pack(anchor="w", pady=(0, 8))
        self.sliders = []
        for title, var in (("色温偏移  ·  冷 ← → 暖", self.warmth), ("色调  ·  绿 ← → 洋红", self.tint)):
            ttk.Label(controls, text=title).pack(anchor="w", pady=(8, 0))
            value = ttk.Label(controls, text="0", style="Muted.TLabel")
            value.pack(anchor="e")
            scale = ttk.Scale(controls, from_=-100, to=100, variable=var)
            scale.pack(fill="x", pady=(0, 8))
            self.sliders.append(scale)
            var.trace_add("write", lambda *_, v=var, label=value: (label.configure(text=f"{v.get():+.1f}"), self.schedule_render()))
        self.pick_button = ttk.Checkbutton(controls, text="取中性点（点击照片）", variable=self.picker, command=self.pick_mode)
        self.pick_button.pack(anchor="w", pady=(12, 6))
        ttk.Label(controls, text="选择本来应呈灰／白色的区域。\n不要将彩色星云、光污染或\n有色星点当作中性点。", style="Muted.TLabel", wraplength=220).pack(anchor="w")
        ttk.Label(controls, textvariable=self.gain_info, style="Muted.TLabel", wraplength=220).pack(anchor="w", pady=10)
        self.reset_button = ttk.Button(controls, text="重置白平衡", command=self.reset)
        self.reset_button.pack(fill="x", pady=5)
        self.save_button = ttk.Button(controls, text="保存设置…", command=self.save_settings)
        self.save_button.pack(fill="x", pady=5)
        self.load_button = ttk.Button(controls, text="载入设置…", command=self.load_settings)
        self.load_button.pack(fill="x", pady=5)
        ttk.Label(controls, text="零值保留当前解码结果。\n冷暖偏移不是绝对 K 值。\n无 ICC 的图像按 sRGB 处理。", style="Muted.TLabel", wraplength=220).pack(anchor="w", pady=12)
        viewer = ttk.Frame(panel)
        viewer.pack(fill="both", expand=True)
        tools = ttk.Frame(viewer)
        tools.pack(fill="x", pady=(0, 6))
        self.fit_button = ttk.Button(tools, text="适合窗口", command=self.fit)
        self.fit_button.pack(side="left")
        self.actual_button = ttk.Button(tools, text="1:1", command=self.actual)
        self.actual_button.pack(side="left", padx=5)
        self.compare_button = ttk.Checkbutton(tools, text="查看原图", variable=self.original, command=self.schedule_render)
        self.compare_button.pack(side="left", padx=5)
        ttk.Label(viewer, textvariable=self.info, style="Muted.TLabel").pack(side="bottom", anchor="w", pady=(4, 0))
        self.canvas = tk.Canvas(viewer, background="#080f18", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.empty_item = self.canvas.create_text(0, 0, text="打开一张照片\n\n滚轮缩放 · 拖动平移 · 中性点取样", fill="#9aafc5", justify="center")
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-1>", lambda _: setattr(self, "drag", None))
        self.canvas.bind("<MouseWheel>", lambda e: self.wheel(e, 1 if e.delta > 0 else -1))
        self.canvas.bind("<Button-4>", lambda e: self.wheel(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self.wheel(e, -1))
        self.destination.trace_add("write", lambda *_: self.controls())
        self.protocol("WM_DELETE_WINDOW", self._request_close)
        self.controls()
        self.after(60, self.poll)

    def settings(self):
        return validate_settings(dict(warmth=self.warmth.get(), tint=self.tint.get(), neutral=self.neutral))

    def controls(self):
        editable = self.levels is not None and not self.busy
        for widget in (*self.sliders, self.pick_button, self.reset_button, self.save_button, self.load_button):
            widget.configure(state="normal" if editable else "disabled")
        for widget in (self.open_button, self.output_entry, self.output_button):
            widget.configure(state="disabled" if self.busy else "normal")
        self.export_button.configure(state="normal" if editable and self.destination.get().strip() else "disabled")

    def open_image(self):
        path = filedialog.askopenfilename(parent=self, title="选择白平衡素材", filetypes=[("照片", "*.tif *.tiff *.png *.jpg *.jpeg *.arw *.nef *.nrw *.cr2 *.cr3 *.crw *.dng *.raf *.orf *.rw2")])
        if not path or self.busy:
            return
        self.busy = True
        self.controls()
        self.status.set("读取原始像素并准备预览…")
        self.load_id += 1
        identity, events = self.load_id, self.events
        path = Path(path)
        def work(token):
            pixels = read_source(path)
            token.raise_if_cancelled()
            return path, make_pyramid(pixels, token)
        self.scheduler.submit("load", work,
            on_result=lambda result: events.put(("loaded", identity, result)),
            on_error=lambda exc, detail: events.put(("load_error", identity, (str(exc), detail))))

    def apply_settings(self, settings):
        data = validate_settings(settings)
        self.neutral = data["neutral"]
        self.warmth.set(data["warmth"])
        self.tint.set(data["tint"])
        self.gain_info.set("中性点 RGB：" + " / ".join(f"{v:.3f}" for v in self.neutral))
        self.schedule_render()

    def reset(self):
        self.apply_settings({})
        self.picker.set(False)
        self.pick_mode()

    def save_settings(self):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".json", filetypes=[("白平衡设置", "*.json")])
        if path:
            try:
                if Path(path).suffix.lower() != ".json" or (self.source and Path(path).resolve() == self.source.resolve()):
                    raise ValueError("请将设置保存为独立的 JSON 文件")
                Path(path).write_text(json.dumps(self.settings(), ensure_ascii=False, indent=2), encoding="utf-8")
            except (OSError, ValueError) as exc:
                show_copyable_error("保存设置", str(exc), parent=self)

    def load_settings(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("白平衡设置", "*.json")])
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                self.apply_settings(data.get("settings", data) if isinstance(data, dict) else data)
            except (OSError, ValueError) as exc:
                show_copyable_error("载入设置", str(exc), parent=self)

    def pick_mode(self):
        self.canvas.configure(cursor="crosshair" if self.picker.get() else "")

    def fit(self):
        if self.levels:
            h, w = self.levels[0].shape[:2]
            self.zoom = min(max(1, self.canvas.winfo_width()) / w, max(1, self.canvas.winfo_height()) / h)
            self.center, self.fit_mode = (w / 2, h / 2), True
            self.schedule_render()

    def actual(self):
        self.zoom, self.fit_mode = 1., False
        self.clamp()
        self.schedule_render()

    def clamp(self):
        if self.levels:
            h, w = self.levels[0].shape[:2]
            hx, hy = self.canvas.winfo_width() / (2*self.zoom), self.canvas.winfo_height() / (2*self.zoom)
            self.center = (w/2 if hx >= w/2 else max(hx, min(w-hx, self.center[0])),
                           h/2 if hy >= h/2 else max(hy, min(h-hy, self.center[1])))

    def resize(self, event):
        self.canvas.coords(self.empty_item, event.width/2, event.height/2)
        if self.fit_mode:
            self.fit()
        else:
            self.clamp()
            self.schedule_render()

    def image_position(self, x, y):
        return (self.center[0] + (x-self.canvas.winfo_width()/2) / self.zoom,
                self.center[1] + (y-self.canvas.winfo_height()/2) / self.zoom)

    def press(self, event):
        if not self.levels:
            return
        if self.picker.get() and not self.busy:
            x, y = self.image_position(event.x, event.y)
            try:
                if x < 0 or y < 0:
                    raise ValueError("请在照片内取样")
                self.apply_settings({"neutral": sample_neutral(self.levels[0], int(x), int(y))})
                self.original.set(False)
                self.picker.set(False)
                self.pick_mode()
                self.status.set("已取中性点；可继续微调冷暖和色调")
            except ValueError as exc:
                self.status.set(str(exc))
        else:
            self.drag = (event.x, event.y, self.center)

    def pan(self, event):
        if self.drag:
            x, y, center = self.drag
            self.center = (center[0] - (event.x-x)/self.zoom, center[1] - (event.y-y)/self.zoom)
            self.fit_mode = False
            self.clamp()
            self.schedule_render()

    def wheel(self, event, steps):
        if not self.levels:
            return
        anchor = self.image_position(event.x, event.y)
        self.zoom = max(.01, min(8., self.zoom * (1.25 ** steps)))
        self.center = (anchor[0] - (event.x-self.canvas.winfo_width()/2)/self.zoom,
                       anchor[1] - (event.y-self.canvas.winfo_height()/2)/self.zoom)
        self.fit_mode = False
        self.clamp()
        self.schedule_render()

    def schedule_render(self):
        self.render_id += 1
        if self.preview_after:
            self.after_cancel(self.preview_after)
        self.preview_after = self.after(35, self.render)

    def render(self):
        self.preview_after = None
        if not self.levels:
            return
        levels, settings, zoom, center = self.levels, self.settings(), self.zoom, self.center
        size = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        original, identity, events = self.original.get(), self.render_id, self.events
        self.scheduler.submit("preview", lambda token: render_view(levels, settings, zoom, center, size, original),
            on_result=lambda result: events.put(("preview", identity, (result, zoom, original))),
            on_error=lambda exc, detail: events.put(("preview_error", identity, (str(exc), detail))))

    def choose_output(self):
        path = filedialog.askdirectory(parent=self)
        if path:
            self.destination.set(path)

    def export(self):
        if self.busy or self.source is None or not self.destination.get().strip():
            return
        source, destination, settings = self.source, self.destination.get().strip(), self.settings()
        self.busy = True
        self.controls()
        self.cancel_button.configure(state="normal")
        self.progress.configure(value=0)
        self.status.set("开始导出…")
        self.export_id += 1
        events, identity = self.events, self.export_id
        def work(token):
            completed = False
            try:
                folder = export_image(source, destination, settings, token, lambda v, text: events.put(("progress", identity, (v, text))))
                completed = True
                events.put(("exported", identity, folder))
            finally:
                if token.cancelled and not completed:
                    events.put(("cancelled", identity, None))
        self.export_token = self.scheduler.submit("export", work,
            on_error=lambda exc, detail: events.put(("export_error", identity, (str(exc), detail))))

    def cancel_export(self):
        self.export_token.cancel()
        self.cancel_button.configure(state="disabled")
        self.status.set("正在取消，等待当前文件读写结束…")

    def poll(self):
        for _ in range(100):
            try:
                kind, identity, data = self.events.get_nowait()
            except queue.Empty:
                break
            current = self.load_id if kind in ("loaded", "load_error") else self.render_id if kind.startswith("preview") else self.export_id
            if identity != current:
                continue
            if kind == "loaded":
                self.source, self.levels = data
                self.busy = False
                self.original.set(False)
                self.reset()
                self.fit()
                self.controls()
                self.canvas.itemconfigure(self.empty_item, state="hidden")
                h, w = self.levels[0].shape[:2]
                self.status.set(f"{self.source.name} · {w} × {h} · sRGB · 16 位处理")
            elif kind == "preview" and data[0] is not None:
                (pixels, position, clipped), zoom, original = data
                self.photo = ImageTk.PhotoImage(Image.fromarray(pixels), master=self.canvas)
                self.canvas.itemconfigure(self.image_item, image=self.photo)
                self.canvas.coords(self.image_item, *position)
                self.info.set(f"{'原图' if original else '白平衡效果'} · {zoom*100:.1f}% · 可见区通道触顶 {clipped:.2%}")
            elif kind == "progress":
                self.progress.configure(value=data[0])
                if not self.export_token.cancelled:
                    self.status.set(data[1])
            elif kind in ("exported", "cancelled"):
                self.busy = False
                self.cancel_button.configure(state="disabled")
                self.controls()
                if kind == "exported":
                    self.result = data
                    self.folder_button.configure(state="normal")
                    self.status.set(f"导出完成 · {data}")
                else:
                    self.status.set("已取消，原片未修改")
            elif kind.endswith("error"):
                if kind != "preview_error":
                    self.busy = False
                    self.controls()
                    self.cancel_button.configure(state="disabled")
                self.status.set("操作未完成；详情见运行日志")
                show_copyable_error("白平衡", data[0], parent=self, details=data[1])
        if self.busy and hasattr(self, "export_token") and self.export_token.cancelled and self.scheduler.active_count("export") == 0 and self.scheduler.active_count("load") == 0:
            self.busy = False
            self.controls()
            self.status.set("已取消，原片未修改")
        if self.winfo_exists():
            self.after(60, self.poll)

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
