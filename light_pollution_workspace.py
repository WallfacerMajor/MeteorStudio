"""Right-hand light-pollution inspector with shared original-pixel viewport mechanics."""
import queue
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from background_tasks import BackgroundTaskScheduler
from error_dialog import show_copyable_error, show_runtime_log
from platform_utils import open_folder
from white_balance import read_source, make_pyramid, render_view
from white_balance_workspace import WhiteBalanceWindow
from light_pollution import DIRECTIONS, estimate, correct, export_image
from workspace_layout import scroll_controls, parameter_slider


class LightPollutionWindow(WhiteBalanceWindow):
    # Reuse fit/1:1, pan, wheel, coordinate mapping and debouncing. All processing,
    # controls and task handling below belong to this independent workspace.
    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.title('光污染渐变校正 · 星野工具箱')
        self.geometry('1180x780')
        self.minsize(850, 600)
        self.scheduler = BackgroundTaskScheduler(max_workers=2)
        self.events = queue.Queue()
        self.source = self.levels = self.model = self.result = self.photo = None
        self.busy = False
        self.generation = self.render_id = 0
        self.preview_after = None
        self.zoom, self.center, self.fit_mode = 1., (0., 0.), True
        self.drag = self.rectangle_start = None
        self.protected = []
        self.direction = tk.StringVar(value='底部')
        self.start = tk.DoubleVar(value=0)
        self.falloff = tk.DoubleVar(value=1)
        self.strength = tk.DoubleVar(value=60)
        self.original = tk.BooleanVar(value=False)
        self.protect_mode = tk.BooleanVar(value=False)
        self.destination = tk.StringVar()
        self.status = tk.StringVar(value='')
        self.info = tk.StringVar(value='')
        root = ttk.Frame(self, padding=14)
        root.pack(fill='both', expand=True)
        header = ttk.Frame(root)
        header.pack(fill='x', pady=(0, 10))
        ttk.Label(header, text='光污染渐变校正', style='Title.TLabel').pack(side='left')
        self.open_button = ttk.Button(header, text='打开照片…', command=self.open_image)
        self.open_button.pack(side='right')
        self.progress = ttk.Progressbar(root, maximum=100, style='Thin.Horizontal.TProgressbar')
        self.progress.pack(side='bottom', fill='x', pady=5)
        ttk.Label(root, textvariable=self.status, wraplength=780).pack(side='bottom', fill='x')
        inspector = ttk.Frame(root, width=300)
        self.edit_inspector = inspector
        inspector.pack(side='right', fill='y', padx=(12, 0))
        inspector.pack_propagate(False)
        self.editor_widgets = []
        def section(title):
            group = ttk.Frame(inspector, padding=(8, 8))
            group.pack(fill='x')
            ttk.Label(group, text=title, style='Section.TLabel').pack(anchor='w', pady=(0, 8))
            return group
        background = section('背景渐变')
        direction_row = ttk.Frame(background)
        direction_row.pack(fill='x', pady=(0, 4))
        ttk.Label(direction_row, text='光源方向').pack(side='left')
        direction = ttk.Combobox(direction_row, width=9, textvariable=self.direction, values=tuple(DIRECTIONS), state='readonly')
        direction.pack(side='right')
        direction.bind('<<ComboboxSelected>>', lambda e: self.invalidate())
        self.editor_widgets.append(direction)
        self.sliders = []
        for label, variable, lo, hi, refit in (
            ('渐变起点  %', self.start, 0, 80, True),
            ('衰减曲线', self.falloff, .5, 3, True),
            ('去除强度  %', self.strength, 0, 150, False),
        ):
            scale = parameter_slider(background, label, variable, lo, hi,
                (lambda v: self.invalidate()) if refit else (lambda v: self.schedule_render()))
            self.editor_widgets.append(scale)
            self.sliders.append(scale)
        ttk.Separator(inspector).pack(fill='x', pady=4)
        protection = section('保护区域')
        self.protect_button = ttk.Checkbutton(protection, text='在画面拖框保护地景／星云', variable=self.protect_mode)
        self.protect_button.pack(anchor='w')
        actions = ttk.Frame(protection)
        actions.pack(fill='x')
        self.clear_button = ttk.Button(actions, text='清除保护', style='Quiet.TButton', command=self.clear_protection)
        self.clear_button.pack(side='left')
        self.reset_button = ttk.Button(actions, text='重置参数', style='Quiet.TButton', command=self.reset)
        self.reset_button.pack(side='right')
        self.analyze_button = ttk.Button(protection, text='估计光污染背景', style='Primary.TButton', command=self.analyze)
        self.analyze_button.pack(fill='x', pady=(10, 4))
        self.editor_widgets += [self.protect_button, self.clear_button, self.analyze_button, self.reset_button]
        canvas = scroll_controls(inspector, 275, reflow=False)
        # Export remains visible while the adjustment groups scroll independently.
        output = ttk.Frame(inspector, padding=(12, 10))
        output.pack(side='bottom', fill='x', before=canvas)
        ttk.Separator(output).pack(fill='x', pady=(0, 10))
        ttk.Label(output, text='输出', style='Section.TLabel').pack(anchor='w', pady=(0, 8))
        path_row = ttk.Frame(output)
        path_row.pack(fill='x')
        self.output_entry = ttk.Entry(path_row, textvariable=self.destination, width=12)
        self.output_entry.pack(side='left', fill='x', expand=True)
        self.output_button = ttk.Button(path_row, text='浏览…', style='Quiet.TButton', command=self.choose_output)
        self.output_button.pack(side='right', padx=(5, 0))
        self.export_button = ttk.Button(output, text='导出 16 位 TIFF', style='Primary.TButton', command=self.export)
        self.export_button.pack(fill='x', pady=(8, 5))
        utility = ttk.Frame(output)
        utility.pack(fill='x')
        self.cancel_button = ttk.Button(utility, text='取消任务', style='Quiet.TButton', command=self.cancel)
        self.cancel_button.pack(side='left')
        self.folder_button = ttk.Button(utility, text='打开结果', style='Quiet.TButton', command=self.open_result, state='disabled')
        self.folder_button.pack(side='left')
        viewer = ttk.Frame(root)
        viewer.pack(fill='both', expand=True)
        bar = ttk.Frame(viewer)
        bar.pack(fill='x', pady=(0, 6))
        self.fit_button = ttk.Button(bar, text='适合窗口', command=self.fit)
        self.fit_button.pack(side='left')
        self.actual_button = ttk.Button(bar, text='1:1', command=self.actual)
        self.actual_button.pack(side='left', padx=5)
        self.compare_button = ttk.Checkbutton(bar, text='查看原图', variable=self.original, command=self.schedule_render)
        self.compare_button.pack(side='left')
        ttk.Label(viewer, textvariable=self.info, wraplength=650, style='Muted.TLabel').pack(side='bottom', fill='x')
        self.canvas = tk.Canvas(viewer, background='#181818', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor='nw')
        self._build_empty_action()
        self.canvas.bind('<Configure>', self.resize)
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.motion)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        self.canvas.bind('<MouseWheel>', lambda e: self.wheel(e, 1 if e.delta > 0 else -1))
        self.canvas.bind('<Button-4>', lambda e: self.wheel(e, 1))
        self.canvas.bind('<Button-5>', lambda e: self.wheel(e, -1))
        self.destination.trace_add('write', lambda *_: self.controls())
        self.protocol('WM_DELETE_WINDOW', self._request_close)
        self.controls()
        from action_icons import iconize_actions
        iconize_actions(self)
        self.after(60, self.poll)

    def show_help(self):
        messagebox.showinfo('光污染校正说明',
            '先框选保护地景、银河和星云，再估计背景。\n调整方向、起点、曲线或保护区后，请重新估计。\n\n本工具处理单方向平滑渐变，不会自动识别天空。\n均匀星云可能被误当背景；请与原图对比。\n过曝、灯光眩光和复杂局部色块不适用。', parent=self)

    def pollution_settings(self):
        return dict(direction=DIRECTIONS[self.direction.get()], start=self.start.get()/100,
                    falloff=self.falloff.get(), strength=self.strength.get()/100, protected=[r[:] for r in self.protected])

    def controls(self):
        enabled = self.levels is not None and not self.busy
        for widget in self.editor_widgets:
            widget.configure(state=('readonly' if isinstance(widget, ttk.Combobox) else 'normal') if enabled else 'disabled')
        for widget in (self.open_button, self.empty_button, self.output_entry, self.output_button):
            widget.configure(state='disabled' if self.busy else 'normal')
        self.export_button.configure(state='normal' if enabled and self.model is not None and self.destination.get().strip() else 'disabled')
        self.cancel_button.configure(state='normal' if self.busy else 'disabled')

    def open_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self, title='选择光污染校正素材', filetypes=[('照片', '*.tif *.tiff *.png *.jpg *.jpeg *.arw *.nef *.nrw *.cr2 *.cr3 *.crw *.dng *.raf *.orf *.rw2')])
        if path and not self.busy:
            self.submit('loaded', lambda token: (Path(path), make_pyramid(read_source(path), token)))

    def submit(self, kind, work):
        self.drag = self.rectangle_start = None
        self.canvas.delete('draft')
        self.generation += 1
        identity, events = self.generation, self.events
        self.busy = True
        self.controls()
        self.status.set('正在读取照片…' if kind == 'loaded' else '正在估计平滑背景…' if kind == 'modeled' else '正在导出…')
        self.task_token = self.scheduler.submit('operation', work,
            on_result=lambda result: events.put((kind, identity, result)),
            on_error=lambda exc, detail: events.put(('error', identity, (str(exc), detail))))

    def invalidate(self):
        self.model = None
        self.controls()
        self.schedule_render()
        self.status.set('区域或渐变形状已改变，请重新估计背景')

    def reset(self):
        self.start.set(0)
        self.falloff.set(1)
        self.strength.set(60)
        self.direction.set('底部')
        self.invalidate()

    def clear_protection(self):
        self.protected.clear()
        self.invalidate()

    def analyze(self):
        if self.busy or not self.levels:
            return
        pixels, settings = self.levels[0], self.pollution_settings()
        self.submit('modeled', lambda token: estimate(pixels, settings, token))

    def render(self):
        self.preview_after = None
        if not self.levels:
            return
        levels, settings, model = self.levels, self.pollution_settings(), self.model
        zoom, center = self.zoom, self.center
        size = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        original = self.original.get() or model is None
        identity, events = self.render_id, self.events
        def work(token):
            token.raise_if_cancelled()
            return render_view(levels, {}, zoom, center, size, original,
                processor=lambda view, x, y, z: correct(view, settings, model, levels[0].shape[:2], (x, y), 1/z))
        self.scheduler.submit('preview', work,
            on_result=lambda data: events.put(('preview', identity, (data, original))),
            on_error=lambda exc, detail: events.put(('preview_error', identity, (str(exc), detail))))

    def point(self, x, y):
        h, w = self.levels[0].shape[:2]
        return ((x*w-self.center[0])*self.zoom+self.canvas.winfo_width()/2,
                (y*h-self.center[1])*self.zoom+self.canvas.winfo_height()/2)

    def draw_protection(self):
        self.canvas.delete('protection')
        if not self.levels or self.original.get():
            return
        for a,b,c,d in self.protected:
            self.canvas.create_rectangle(*self.point(a,b), *self.point(c,d), outline='#70ded2', dash=(5, 3), width=2, tags='protection')

    def press(self, event):
        if not self.levels or self.busy:
            return
        if self.protect_mode.get():
            self.rectangle_start = self.image_position(event.x, event.y)
        else:
            self.drag = (event.x, event.y, self.center)

    def motion(self, event):
        if self.rectangle_start:
            self.canvas.delete('draft')
            x,y = self.rectangle_start
            h,w = self.levels[0].shape[:2]
            self.canvas.create_rectangle(*self.point(x/w,y/h), event.x,event.y, outline='#ffcf70', tags='draft')
        else:
            self.pan(event)

    def release(self, event):
        if self.rectangle_start:
            h,w = self.levels[0].shape[:2]
            a,b = self.rectangle_start
            c,d = self.image_position(event.x, event.y)
            rect = [max(0,min(1,min(a,c)/w)), max(0,min(1,min(b,d)/h)), max(0,min(1,max(a,c)/w)), max(0,min(1,max(b,d)/h))]
            if rect[2]-rect[0] > .003 and rect[3]-rect[1] > .003 and len(self.protected) < 100:
                self.protected.append(rect)
                self.invalidate()
            self.rectangle_start = None
            self.canvas.delete('draft')
            self.draw_protection()
        self.drag = None

    def export(self):
        if self.busy or self.model is None or not self.destination.get().strip():
            return
        source, destination, settings, model = self.source, self.destination.get(), self.pollution_settings(), dict(self.model)
        events, identity = self.events, self.generation+1
        self.submit('exported', lambda token: export_image(source, destination, settings, model, token,
            lambda value, message: events.put(('progress', identity, (value, message)))))

    def cancel(self):
        if self.busy:
            self.task_token.cancel()
            self.status.set('正在取消，等待当前读写结束…')

    def open_result(self):
        if self.result:
            try:
                open_folder(self.result)
            except OSError as exc:
                show_copyable_error('打开光污染校正结果', str(exc), parent=self)

    def poll(self):
        for _ in range(100):
            try:
                kind, identity, data = self.events.get_nowait()
            except queue.Empty:
                break
            if identity != (self.render_id if kind.startswith('preview') else self.generation):
                continue
            if kind == 'progress':
                self.progress.configure(value=data[0])
                continue
            if kind == 'preview':
                result, original = data
                if result:
                    pixels, position, clipped = result
                    self.photo = ImageTk.PhotoImage(Image.fromarray(pixels), master=self.canvas)
                    self.canvas.itemconfigure(self.image_item, image=self.photo)
                    self.canvas.coords(self.image_item, *position)
                    self.draw_protection()
                    self.info.set(f"{'原图' if original else '渐变校正'} · {self.zoom*100:.1f}% · 保护区域 {len(self.protected)} 个 · 触顶 {clipped:.2%}")
                continue
            if kind == 'preview_error':
                show_copyable_error('光污染预览', data[0], parent=self, details=data[1])
                continue
            self.busy = False
            if kind == 'loaded':
                self.source, self.levels = data
                self.model, self.protected = None, []
                self.original.set(False)
                self.protect_mode.set(False)
                self.canvas.itemconfigure(self.empty_item, state='hidden')
                self.fit()
                self.status.set('照片已加载；请先保护地景／星云，再估计背景')
            elif kind == 'modeled':
                self.model = data
                self.original.set(False)
                self.schedule_render()
                self.status.set(f"已估计背景 · {data['samples']} 个样本 · 请对比原图，避免过度校正")
            elif kind == 'exported':
                self.result = data
                self.folder_button.configure(state='normal')
                self.progress.configure(value=100)
                self.status.set(f'导出完成 · {data}')
            elif kind == 'error':
                self.status.set('操作未完成，详情见运行日志')
                show_copyable_error('光污染校正', data[0], parent=self, details=data[1])
            self.controls()
        if self.busy and self.task_token.cancelled and self.scheduler.active_count('operation') == 0:
            self.busy = False
            self.controls()
            self.status.set('已取消')
        self.after(60, self.poll)
