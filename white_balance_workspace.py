"""White balance workspace with replaceable viewport jobs and 16-bit export."""
import json
import queue
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from background_tasks import BackgroundTaskScheduler
from error_dialog import show_copyable_error, show_runtime_log, append_runtime_log
from platform_utils import open_folder
from ui_navigation import cancel_widget_timers
from neutral_candidates import suggest_neutral_points
from white_balance import read_source, make_pyramid, render_view, sample_neutral, export_image, export_batch, validate_settings, RAW_SUFFIXES


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
        self.candidate_id = 0
        self.candidates = []
        self.candidate_preview = None
        self.finding_candidates = False
        self.neutral = [1, 1, 1]
        self.zoom, self.center, self.fit_mode = 1., (0., 0.), True
        self.drag = None
        self.photo = None
        self.warmth, self.tint = tk.DoubleVar(value=0), tk.DoubleVar(value=0)
        self.strength = tk.DoubleVar(value=100)
        self.raw_baseline = tk.StringVar(value="相机白平衡")
        self.loaded_baseline = "camera"
        self.keep_settings = tk.BooleanVar(value=False)
        self.equipment = {key: tk.StringVar() for key in ("camera", "modification", "filter", "reference")}
        self.original, self.picker = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.destination = tk.StringVar()
        self.status = tk.StringVar(value="打开照片开始 · 支持 sRGB TIFF / PNG / JPG 和 RAW")
        self.info = tk.StringVar(value="")
        self.gain_info = tk.StringVar(value="中性点：未取样")
        body = ttk.Frame(self, padding=(16, 12))
        body.pack(fill="both", expand=True)
        header = ttk.Frame(body)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="白平衡", style="Title.TLabel").pack(side="left")
        self.open_button = ttk.Button(header, text="打开照片…", command=self.open_image)
        self.open_button.pack(side="right")
        panel = ttk.Frame(body)
        panel.pack(fill="both", expand=True)
        sidebar = ttk.Frame(panel, width=300)
        sidebar.pack(side="right", fill="y", padx=(12, 0))
        sidebar.pack_propagate(False)
        self.inspector_tabs = ttk.Notebook(sidebar)
        self.inspector_tabs.pack(fill="both", expand=True)
        adjustment_tab = ttk.Frame(self.inspector_tabs)
        calibration_tab = ttk.Frame(self.inspector_tabs)
        self.inspector_tabs.add(adjustment_tab, text="调整")
        self.inspector_tabs.add(calibration_tab, text="设备预设")
        def scroll_panel(parent):
            canvas = tk.Canvas(parent, width=275, background="#292929", highlightthickness=0)
            scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
            scrollbar.pack(side="right", fill="y")
            canvas.pack(fill="both", expand=True)
            canvas.configure(yscrollcommand=scrollbar.set)
            content = ttk.Frame(canvas, padding=(10, 10))
            item = canvas.create_window(0, 0, window=content, anchor="nw")
            content.bind("<Configure>", lambda e: canvas.configure(scrollregion=(0, 0, e.width, max(e.height, canvas.winfo_height()))))
            canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
            canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-3 if e.delta > 0 else 3, "units"))
            canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-3, "units"))
            canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(3, "units"))
            parent._inspector_canvas = canvas
            parent._inspector_scrolled_widgets = (canvas,)
            return canvas, content
        self.control_canvas, controls = scroll_panel(adjustment_tab)
        self.calibration_canvas, calibration_controls = scroll_panel(calibration_tab)
        footer = ttk.Frame(sidebar)
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        self.export_button = ttk.Button(footer, text="导出 16 位 TIFF", style="Primary.TButton", command=self.export)
        self.export_button.pack(side="left")
        self.cancel_button = ttk.Button(footer, text="取消导出", state="disabled", command=self.cancel_export)
        self.cancel_button.pack(side="left", padx=6)
        self.batch_button = ttk.Button(footer, text="批量应用当前设置…", command=self.batch_export)
        self.batch_button.pack(side="left")
        self.folder_button = ttk.Button(footer, text="打开结果", state="disabled", command=self.open_result)
        self.folder_button.pack(side="right")
        self.progress = ttk.Progressbar(body, maximum=100)
        self.progress.pack(side="bottom", fill="x", pady=(6, 0), before=panel)
        self.status_label = ttk.Label(body, textvariable=self.status, wraplength=850)
        self.status_label.pack(side="bottom", fill="x", pady=(6, 0), before=panel)
        body.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(400, e.width-32)))
        output = ttk.Frame(sidebar)
        output.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(output, text="输出到").pack(side="left")
        self.output_entry = ttk.Entry(output, textvariable=self.destination)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.output_button = ttk.Button(output, text="选择…", command=self.choose_output)
        self.output_button.pack(side="right")
        ttk.Label(controls, text="相对调整", style="Section.TLabel").pack(anchor="w", pady=(0, 8))
        self.sliders = []
        for title, var, lower in (("色温偏移  ·  冷 ← → 暖", self.warmth, -100), ("色调  ·  绿 ← → 洋红", self.tint, -100), ("中性校准强度  ·  0–100%", self.strength, 0)):
            from workspace_layout import parameter_slider
            scale = parameter_slider(controls, title, var, lower, 100)
            self.sliders.append(scale)
            var.trace_add("write", lambda *_: self.schedule_render())
        self.pick_button = ttk.Checkbutton(controls, text="取中性点（点击照片）", variable=self.picker, command=self.pick_mode)
        self.pick_button.pack(anchor="w", pady=(12, 6))
        ttk.Label(controls, textvariable=self.gain_info, style="Muted.TLabel", wraplength=220).pack(anchor="w", pady=10)
        self.reset_button = ttk.Button(controls, text="重置白平衡", command=self.reset)
        self.reset_button.pack(fill="x", pady=5)
        self.suggest_button = ttk.Button(controls, text="自动推荐参考点", command=self.suggest_points)
        self.suggest_button.pack(fill="x", pady=5)
        self.confirm_point_button = ttk.Button(controls, text="应用此参考点", command=self.confirm_point)
        self.confirm_point_button.pack(fill="x", pady=5)
        self.cancel_point_button = ttk.Button(controls, text="取消参考点预览", command=self.cancel_point)
        self.cancel_point_button.pack(fill="x")
        controls = calibration_controls
        ttk.Label(controls, text="改机与滤镜校准", style="Section.TLabel").pack(anchor="w")
        self.equipment_widgets = []
        for key, title in (("camera", "机身"), ("modification", "改机方式"), ("filter", "滤镜 / 光学组合"), ("reference", "参考光源 / 拍摄条件")):
            ttk.Label(controls, text=title).pack(anchor="w", pady=(5, 0))
            if key == "modification":
                entry = ttk.Combobox(controls, textvariable=self.equipment[key], values=("未记录", "Hα 增强", "全光谱", "未改机"), state="readonly")
            else:
                entry = ttk.Entry(controls, textvariable=self.equipment[key], validate="key",
                                  validatecommand=(self.register(lambda value: len(value) <= 300), "%P"))
            entry.pack(fill="x")
            self.equipment_widgets.append(entry)
        ttk.Label(controls, text="RAW 解码基准", style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        self.baseline_combo = ttk.Combobox(controls, textvariable=self.raw_baseline, values=("相机白平衡", "固定日光（同组）"), state="readonly")
        self.baseline_combo.pack(fill="x")
        self.baseline_combo.bind("<<ComboboxSelected>>", self.baseline_changed)
        self.keep_button = ttk.Checkbutton(controls, text="下一张沿用当前校准", variable=self.keep_settings)
        self.keep_button.pack(anchor="w", pady=8)
        self.save_button = ttk.Button(controls, text="保存校准预设…", command=self.save_settings)
        self.save_button.pack(fill="x", pady=5)
        self.load_button = ttk.Button(controls, text="载入校准预设…", command=self.load_settings)
        self.load_button.pack(fill="x", pady=5)
        # Keep export and destination alongside adjustments, away from the canvas footer.
        footer.pack_forget()
        output.pack_forget()
        footer.pack(in_=sidebar, side="bottom", fill="x", before=self.inspector_tabs, pady=5)
        output.pack(in_=sidebar, side="bottom", fill="x", before=self.inspector_tabs, pady=5)
        footer.lift()
        output.lift()
        for child in footer.winfo_children():
            child.pack_forget()
        self.export_button.configure(text="导出 TIFF")
        for child, row, column, span in (
            (self.export_button, 0, 0, 1), (self.cancel_button, 0, 1, 1),
            (self.batch_button, 1, 0, 2), (self.folder_button, 2, 0, 1),
        ):
            child.grid(row=row, column=column, columnspan=span, sticky="ew", pady=2)
        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=1)
        ttk.Separator(sidebar).pack(side="bottom", fill="x", before=self.inspector_tabs)
        output.pack_configure(pady=8)
        self.output_entry.pack_configure(padx=5)
        def bind_wheel(widget, canvas):
            if not isinstance(widget, (ttk.Scale, ttk.Combobox, ttk.Entry)):
                for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    widget.bind(event, lambda e, c=canvas: (c.yview_scroll(-3 if getattr(e, 'num', 0)==4 or getattr(e, 'delta', 0)>0 else 3, "units"), "break")[-1])
            for child in widget.winfo_children():
                bind_wheel(child, canvas)
        bind_wheel(adjustment_tab, self.control_canvas)
        bind_wheel(calibration_tab, self.calibration_canvas)
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
        self.canvas = tk.Canvas(viewer, background="#181818", highlightthickness=0)
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
        return validate_settings(dict(warmth=self.warmth.get(), tint=self.tint.get(), neutral=self.neutral,
            neutral_strength=self.strength.get(), raw_baseline="daylight" if self.raw_baseline.get().startswith("固定") else "camera",
            equipment={key: value.get() for key, value in self.equipment.items()}))

    def controls(self):
        editable = self.levels is not None and not self.busy
        pending = self.candidate_preview is not None
        for widget in (*self.sliders, self.pick_button, self.reset_button, self.save_button, self.load_button):
            widget.configure(state="normal" if editable else "disabled")
        for widget in (self.open_button, self.output_entry, self.output_button):
            widget.configure(state="disabled" if self.busy else "normal")
        self.export_button.configure(state="normal" if editable and not pending and self.destination.get().strip() else "disabled")
        self.batch_button.configure(state="normal" if editable and not pending and self.destination.get().strip() else "disabled")
        self.suggest_button.configure(state="normal" if editable and not self.finding_candidates else "disabled")
        for widget in (self.confirm_point_button, self.cancel_point_button):
            widget.configure(state="normal" if editable and pending else "disabled")
        if pending:
            for widget in (*self.sliders, self.pick_button, self.save_button, self.load_button, self.baseline_combo):
                widget.configure(state="disabled")
        self.keep_button.configure(state="disabled" if self.busy else "normal")
        self.baseline_combo.configure(state="disabled" if self.busy else "readonly")
        for widget in self.equipment_widgets:
            widget.configure(state="disabled" if self.busy else "readonly" if isinstance(widget, ttk.Combobox) else "normal")
        if pending:
            self.baseline_combo.configure(state="disabled")

    def open_image(self):
        path = filedialog.askopenfilename(parent=self, title="选择白平衡素材", filetypes=[("照片", "*.tif *.tiff *.png *.jpg *.jpeg *.arw *.nef *.nrw *.cr2 *.cr3 *.crw *.dng *.raf *.orf *.rw2")])
        if not path or self.busy:
            return
        data = self.settings()
        if not self.keep_settings.get():
            data.update(warmth=0, tint=0, neutral=[1, 1, 1], neutral_strength=100)
        self.load_photo(Path(path), data)

    def load_photo(self, path, data, preserve_view=False):
        data = validate_settings(data)
        self.candidate_id += 1
        self.scheduler.cancel("candidates")
        self.candidates, self.candidate_preview, self.finding_candidates = [], None, False
        self.draw_candidates()
        self.schedule_render()
        self.busy = True
        self.controls()
        self.status.set("读取原始像素并准备预览…")
        self.load_id += 1
        identity, events = self.load_id, self.events
        path = Path(path)
        def work(token):
            pixels = read_source(path, data["raw_baseline"])
            token.raise_if_cancelled()
            return path, make_pyramid(pixels, token), data, preserve_view
        self.scheduler.submit("load", work,
            on_result=lambda result: events.put(("loaded", identity, result)),
            on_error=lambda exc, detail: events.put(("load_error", identity, (str(exc), detail))))

    def baseline_changed(self, event=None):
        if self.busy:
            return
        data = self.settings()
        # An old gray-card gain is not valid on a different decode baseline.
        data.update(neutral=[1, 1, 1], neutral_strength=100, warmth=0, tint=0)
        if self.source and self.source.suffix.lower() in RAW_SUFFIXES:
            self.load_photo(self.source, data, preserve_view=True)
        else:
            self.apply_settings(data)
            self.loaded_baseline = data["raw_baseline"]
        self.status.set("RAW 基准已更改；请在相同基准下重新取灰卡或载入匹配预设")

    def apply_settings(self, settings):
        self.candidate_preview = None
        data = validate_settings(settings)
        self.neutral = data["neutral"]
        self.warmth.set(data["warmth"])
        self.tint.set(data["tint"])
        self.strength.set(data["neutral_strength"])
        self.raw_baseline.set("固定日光（同组）" if data["raw_baseline"] == "daylight" else "相机白平衡")
        for key, variable in self.equipment.items():
            variable.set(data["equipment"][key])
        self.gain_info.set("中性点 RGB：" + " / ".join(f"{v:.3f}" for v in self.neutral))
        self.schedule_render()
        self.controls()

    def suggest_points(self):
        if self.busy or not self.levels or self.finding_candidates:
            return
        self.cancel_point()
        self.picker.set(False)
        self.pick_mode()
        self.candidates = []
        self.draw_candidates()
        self.candidate_id += 1
        identity, events, pixels = self.candidate_id, self.events, self.levels[0]
        self.finding_candidates = True
        self.controls()
        self.status.set("正在筛选平滑、未过曝的参考区域…")
        self.scheduler.submit("candidates", lambda token: suggest_neutral_points(pixels, token),
            on_result=lambda result: events.put(("candidates", identity, result)),
            on_error=lambda exc, detail: events.put(("candidates_error", identity, (str(exc), detail))))

    def point_position(self, point):
        return ((point['x']-self.center[0])*self.zoom+self.canvas.winfo_width()/2,
                (point['y']-self.center[1])*self.zoom+self.canvas.winfo_height()/2)

    def draw_candidates(self):
        self.canvas.delete("reference")
        for index, point in enumerate(self.candidates):
            x, y = self.point_position(point)
            color = "#ffcf70" if self.candidate_preview == index else "#70ded2"
            self.canvas.create_oval(x-12, y-12, x+12, y+12, outline=color, width=2, tags="reference")
            self.canvas.create_text(x+18, y-16, text=str(index+1), fill=color, tags="reference")

    def cancel_point(self):
        self.candidate_preview = None
        self.controls()
        self.schedule_render()

    def confirm_point(self):
        if self.candidate_preview is None or self.busy:
            return
        point = self.candidates[self.candidate_preview]
        self.original.set(False)
        self.apply_settings(dict(self.settings(), neutral=point['gains'], warmth=0, tint=0, neutral_strength=100))
        self.status.set("已应用所选参考点；可降低校准强度，保留自然星野色彩")

    def reset(self):
        self.apply_settings(dict(self.settings(), warmth=0, tint=0, neutral=[1, 1, 1], neutral_strength=100))
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
                data = validate_settings(data.get("settings", data) if isinstance(data, dict) else data)
                if self.source and self.source.suffix.lower() in RAW_SUFFIXES and data["raw_baseline"] != self.loaded_baseline:
                    self.load_photo(self.source, data, preserve_view=True)
                else:
                    self.apply_settings(data)
                self.keep_settings.set(True)
            except (OSError, ValueError) as exc:
                show_copyable_error("载入设置", str(exc), parent=self)

    def pick_mode(self):
        self.canvas.configure(cursor="crosshair" if self.picker.get() else "")
        if self.picker.get():
            self.status.set("点击应呈灰／白色的区域；避开星云、星点和光污染。")

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
        if not self.busy and not self.picker.get():
            for index, point in enumerate(self.candidates):
                x, y = self.point_position(point)
                if (event.x-x)**2 + (event.y-y)**2 <= 20**2:
                    self.candidate_preview = index
                    self.original.set(False)
                    self.controls()
                    self.schedule_render()
                    self.status.set(f"参考点 {index+1} 预览 · 尚未应用；此处是否本应中性需人工确认")
                    return
        if self.picker.get() and not self.busy:
            x, y = self.image_position(event.x, event.y)
            try:
                if x < 0 or y < 0:
                    raise ValueError("请在照片内取样")
                self.apply_settings(dict(self.settings(), neutral=sample_neutral(self.levels[0], int(x), int(y)), warmth=0, tint=0, neutral_strength=100))
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
        if self.candidate_preview is not None:
            settings = dict(settings, neutral=self.candidates[self.candidate_preview]['gains'], warmth=0, tint=0, neutral_strength=100)
        size = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        original, identity, events = self.original.get(), self.render_id, self.events
        self.scheduler.submit("preview", lambda token: render_view(levels, settings, zoom, center, size, original),
            on_result=lambda result: events.put(("preview", identity, (result, zoom, original))),
            on_error=lambda exc, detail: events.put(("preview_error", identity, (str(exc), detail))))

    def choose_output(self):
        path = filedialog.askdirectory(parent=self)
        if path:
            self.destination.set(path)

    def batch_export(self):
        if self.busy:
            return
        paths = filedialog.askopenfilenames(parent=self, title="选择整组素材 · 使用当前校准，不逐张自动白平衡",
            filetypes=[("照片", "*.tif *.tiff *.png *.jpg *.jpeg *.arw *.nef *.nrw *.cr2 *.cr3 *.crw *.dng *.raf *.orf *.rw2")])
        if paths:
            self.export(paths)

    def export(self, batch_paths=None):
        if self.busy or self.candidate_preview is not None or self.source is None or not self.destination.get().strip():
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
                operation, inputs = (export_batch, tuple(batch_paths)) if batch_paths else (export_image, source)
                folder = operation(inputs, destination, settings, token, lambda v, text: events.put(("progress", identity, (v, text))))
                completed = True
                summary = "导出完成"
                if batch_paths:
                    report = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
                    passed = sum(item["status"] == "complete" for item in report["items"])
                    for item in report["items"]:
                        if item["status"] != "complete":
                            append_runtime_log(f"批量改机校准失败：{item['source']}", item.get("error", "未知错误"))
                    summary = f"批量结束 · 成功 {passed} / {len(report['sources'])} · 详情见 batch.json"
                events.put(("exported", identity, (folder, summary)))
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
            current = self.candidate_id if kind.startswith("candidates") else self.load_id if kind in ("loaded", "load_error") else self.render_id if kind.startswith("preview") else self.export_id
            if identity != current:
                continue
            if kind == "candidates":
                self.finding_candidates = False
                self.candidates = data
                self.draw_candidates()
                self.controls()
                self.status.set(f"找到 {len(data)} 个平滑参考候选；点击编号预览，确认后应用" if data else "未找到适合推荐的区域；请使用灰卡参考或手动取样")
            elif kind == "loaded":
                self.source, self.levels, parameters, preserve_view = data
                self.loaded_baseline = parameters["raw_baseline"]
                self.busy = False
                self.original.set(False)
                self.apply_settings(parameters)
                self.picker.set(False)
                self.pick_mode()
                if preserve_view:
                    self.clamp()
                    self.schedule_render()
                else:
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
                self.draw_candidates()
                self.info.set(f"{'原图' if original else '参考点预览（未应用）' if self.candidate_preview is not None else '白平衡效果'} · {zoom*100:.1f}% · 可见区通道触顶 {clipped:.2%}")
            elif kind == "progress":
                self.progress.configure(value=data[0])
                if not self.export_token.cancelled:
                    self.status.set(data[1])
            elif kind in ("exported", "cancelled"):
                self.busy = False
                self.cancel_button.configure(state="disabled")
                self.controls()
                if kind == "exported":
                    self.result, summary = data
                    self.folder_button.configure(state="normal")
                    self.status.set(f"{summary} · {self.result}")
                else:
                    self.status.set("已取消，原片未修改")
            elif kind.endswith("error"):
                if kind == "candidates_error":
                    self.finding_candidates = False
                if kind == "load_error" and self.source:
                    self.raw_baseline.set("固定日光（同组）" if self.loaded_baseline == "daylight" else "相机白平衡")
                if kind not in ("preview_error", "candidates_error"):
                    self.busy = False
                    self.controls()
                    self.cancel_button.configure(state="disabled")
                self.controls()
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
