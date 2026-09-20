from __future__ import annotations

import json
import math
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from ptgui_pipeline import (
    AlignmentResult,
    default_ptgui_path,
    default_siril_path,
    list_images,
    load_confirmed_meteor_tracks,
    read_lens_info,
    run_alignment_pipeline,
)
from platform_utils import open_folder
from error_dialog import show_copyable_error, show_runtime_log
from background_tasks import (
    BackgroundTaskScheduler, CancellationToken, TaskCancelledError,
)


LAB_PROJECTIONS = {
    "透视（保持直线）": "rectilinear",
    "墨卡托（宽幅弧线）": "mercator",
    "等距柱状（天球展开）": "equirectangular",
    "立体投影（强化辐射感）": "stereographic",
}
LAB_CANVASES = {
    "参考图画布 100%": 1.0,
    "扩展公共天空 135%": 1.35,
    "大幅扩展天空 170%": 1.70,
}


class AlignmentWorkspace(tk.Toplevel):
    def __init__(self, master, on_ready: Callable[[AlignmentResult], None], control_points_only: bool = False) -> None:
        super().__init__(master)
        self.title("星野工具箱 — Siril + PTGui")
        self.control_points_only = tk.BooleanVar(value=control_points_only)
        self.geometry("1120x800")
        self.minsize(960, 700)
        self.on_ready = on_ready
        self.base_path = tk.StringVar()
        self.meteor_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.ptgui_path = tk.StringVar(value=str(default_ptgui_path() or ""))
        self.siril_path = tk.StringVar(value=str(default_siril_path() or ""))
        self.reference_focal_length = tk.StringVar(value="")
        self.reference_focal_status = tk.StringVar(value="尚未检查参考图 EXIF")
        self.focal_length = tk.DoubleVar(value=14.0)
        self.sensor_diagonal = tk.DoubleVar(value=43.2666)
        self.laboratory_mode = tk.BooleanVar(value=False)
        self.lab_projection = tk.StringVar(value="墨卡托（宽幅弧线）")
        self.lab_canvas = tk.StringVar(value="扩展公共天空 135%")
        self.status = tk.StringVar(value="选择对齐参考图和流星原图文件夹；输出文件夹会自动创建，也可以手动更改。")
        self.items = []
        self.confirmed_tracks_by_file: dict[str, list[dict]] = {}
        self.worker_queue: queue.Queue = queue.Queue()
        self.background_tasks = BackgroundTaskScheduler(
            max_workers=1, thread_name_prefix="meteor-alignment"
        )
        self.running = False
        self.last_result: AlignmentResult | None = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._request_close)
        self._poll_id = self.after(120, self._poll_queue)

    def destroy(self) -> None:
        if getattr(self, "_poll_id", None):
            self.after_cancel(self._poll_id)
            self._poll_id = None
        scheduler = getattr(self, "background_tasks", None)
        if scheduler is not None:
            scheduler.shutdown(wait=False)
            self.background_tasks = None
        super().destroy()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="Siril 星点 + PTGui 控制点", style="Title.TLabel").pack(side="left")
        inspector = ttk.Frame(root, width=360)
        self.edit_inspector = inspector
        inspector.pack(side="right", fill="y", padx=(12, 0))
        inspector.pack_propagate(False)
        configuration = ttk.Notebook(inspector, height=300)
        configuration.pack(fill="x", pady=(0, 6))
        inputs_tab = ttk.Frame(configuration, padding=6)
        lens_tab = ttk.Frame(configuration, padding=6)
        configuration.add(inputs_tab, text="输入与输出")
        configuration.add(lens_tab, text="镜头与投影")
        paths = ttk.LabelFrame(inputs_tab, text="素材与输出", padding=8)
        self.input_panel = paths
        paths.pack(fill="x")
        self._path_row(paths, 0, "对齐参考图", self.base_path, self._choose_base, "选择文件…")
        self._path_row(paths, 1, "完整流星原图文件夹", self.meteor_dir, self._choose_meteors, "选择文件夹…")
        self._path_row(paths, 2, "输出文件夹（可选）", self.output_dir, self._choose_output, "另选文件夹…")
        paths.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(lens_tab, text="镜头与星空区域", padding=8)
        self.settings_panel = settings
        settings.pack(fill="x", pady=(8, 0))
        ttk.Label(settings, text="参考图焦距(mm)").grid(row=0, column=0, sticky="w")
        self.reference_focal_input = ttk.Spinbox(
            settings, from_=1, to=1000, increment=0.1,
            textvariable=self.reference_focal_length, width=9,
        )
        self.reference_focal_input.grid(row=0, column=1, padx=(5, 18))
        ttk.Label(settings, textvariable=self.reference_focal_status).grid(
            row=0, column=2, columnspan=6, sticky="w",
        )
        ttk.Label(settings, text="素材EXIF缺失兜底焦距(mm)").grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Spinbox(settings, from_=1, to=1000, increment=0.1, textvariable=self.focal_length, width=9).grid(row=1, column=1, padx=(5, 18), pady=(5, 0))
        ttk.Label(settings, text="传感器对角线(mm)").grid(row=1, column=2, sticky="w", pady=(5, 0))
        ttk.Spinbox(settings, from_=1, to=100, increment=0.1, textvariable=self.sensor_diagonal, width=9).grid(row=1, column=3, padx=(5, 18), pady=(5, 0))
        ttk.Label(settings, text="星空区域").grid(row=1, column=4, sticky="w", pady=(5, 0))
        ttk.Label(settings, text="逐张自动识别并生成星点蒙版").grid(row=1, column=5, columnspan=3, sticky="w", padx=5, pady=(5, 0))
        settings.columnconfigure(5, weight=1)

        laboratory = ttk.LabelFrame(lens_tab, text="对齐实验室", padding=8)
        self.laboratory_panel = laboratory
        laboratory.pack(fill="x", pady=(8, 0))
        ttk.Checkbutton(
            laboratory, text="启用实验室模式", variable=self.laboratory_mode,
            command=self._laboratory_changed,
        ).pack(side="left")
        ttk.Label(laboratory, text="输出投影").pack(side="left", padx=(18, 5))
        self.lab_projection_box = ttk.Combobox(
            laboratory, textvariable=self.lab_projection,
            values=list(LAB_PROJECTIONS), state="disabled", width=24,
        )
        self.lab_projection_box.pack(side="left")
        ttk.Label(laboratory, text="公共画布").pack(side="left", padx=(18, 5))
        self.lab_canvas_box = ttk.Combobox(
            laboratory, textvariable=self.lab_canvas,
            values=list(LAB_CANVASES), state="disabled", width=22,
        )
        self.lab_canvas_box.pack(side="left")
        mode_row = ttk.Frame(inspector)
        mode_row.pack(fill="x", pady=(8, 0))
        self.mode_button = ttk.Checkbutton(mode_row, text="仅生成控制点工程（跳过 16 位图层导出）", variable=self.control_points_only, command=self._laboratory_changed)
        self.mode_button.pack(side="left")

        actions = ttk.Frame(inspector)
        actions.pack(fill="x", pady=8)
        self.scan_button = ttk.Button(actions, text="1. 扫描素材", command=self.scan)
        self.scan_button.pack(side="left")
        self.run_button = ttk.Button(actions, text="2. 生成控制点工程" if self.control_points_only.get() else "2. 对齐并导出16位图层", command=self.run, state="disabled", style="Accent.TButton")
        self.run_button.pack(side="left", padx=8)
        self.load_button = ttk.Button(
            actions, text="3. 使用对齐结果并返回流星合成",
            command=self.load_result, state="disabled",
        )
        self.load_button.pack(side="left")
        self.open_output_button = ttk.Button(
            actions, text="打开导出文件夹", command=self._open_output_folder, state="disabled",
        )
        self.open_output_button.pack(side="left", padx=(8, 0))
        result_actions = ttk.Frame(inspector)
        result_actions.pack(fill="x", pady=(0, 8))
        self.creative_button = ttk.Button(result_actions, text="选中待处理项 → 创意放置", command=self.mark_creative, state="disabled")
        self.creative_button.pack(side="right")
        self.discard_button = ttk.Button(result_actions, text="丢弃选中待处理项", command=self.mark_discarded, state="disabled")
        self.discard_button.pack(side="right", padx=6)

        results = ttk.Frame(root)
        results.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(results, columns=("status", "focal", "sky", "cp", "error", "message"), show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="流星原图")
        self.tree.heading("status", text="状态")
        self.tree.heading("focal", text="焦距")
        self.tree.heading("sky", text="星点有效区")
        self.tree.heading("cp", text="控制点")
        self.tree.heading("error", text="匹配误差")
        self.tree.heading("message", text="说明")
        self.tree.column("#0", width=260)
        self.tree.column("status", width=90, anchor="center")
        self.tree.column("focal", width=105, anchor="center")
        self.tree.column("sky", width=90, anchor="center")
        self.tree.column("cp", width=70, anchor="center")
        self.tree.column("error", width=90, anchor="center")
        self.tree.column("message", width=390)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(results, orient="vertical", command=self.tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(results, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)

        footer = ttk.Frame(root)
        footer.pack(side="bottom", fill="x", pady=(8, 0), before=inspector)
        ttk.Label(footer, textvariable=self.status, wraplength=650).pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, length=160)
        self.progress.pack(side="right")

        from workspace_layout import scroll_controls, stack_controls
        scroll_controls(inputs_tab, 310)
        scroll_controls(lens_tab, 310)
        for section in (mode_row, actions, result_actions):
            stack_controls(section, 330)
        scroll_controls(inspector, 335, reflow=False)

    def _path_row(self, parent, row, label, variable, callback, button_text) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=2)
        ttk.Button(parent, text=button_text, command=callback).grid(row=row, column=2, pady=2)

    def _choose_base(self) -> None:
        path = filedialog.askopenfilename(title="选择对齐参考图", filetypes=[("图像", "*.tif *.tiff *.jpg *.jpeg *.png")])
        if path:
            self.base_path.set(path)
            self._refresh_reference_focal_ui(Path(path))

    def _refresh_reference_focal_ui(self, path: Path) -> None:
        try:
            diagonal = float(self.sensor_diagonal.get())
            info = read_lens_info(path, 1.0, diagonal)
            if info.source.startswith("EXIF"):
                self.reference_focal_length.set(f"{info.focal_length:.1f}")
                self.reference_focal_input.configure(state="disabled")
                self.reference_focal_status.set(
                    f"已从参考图EXIF读取：{info.focal_length:.1f}mm"
                )
            else:
                self.reference_focal_length.set("")
                self.reference_focal_input.configure(state="normal")
                self.reference_focal_status.set("参考图没有EXIF，请在左侧填写真实焦距")
        except (OSError, ValueError, TypeError):
            self.reference_focal_length.set("")
            self.reference_focal_input.configure(state="normal")
            self.reference_focal_status.set("无法读取参考图EXIF，请填写真实焦距")

    def _reference_focal(self, base: Path) -> tuple[float, str]:
        diagonal = float(self.sensor_diagonal.get())
        probe = read_lens_info(base, 1.0, diagonal)
        if probe.source.startswith("EXIF"):
            return float(probe.focal_length), probe.source
        text = self.reference_focal_length.get().strip()
        if not text:
            raise ValueError("参考图没有EXIF，请填写“参考图焦距(mm)”后再开始")
        try:
            focal = float(text)
        except ValueError as exc:
            raise ValueError("参考图焦距必须是有效数字") from exc
        if not (math.isfinite(focal) and focal > 0):
            raise ValueError("参考图焦距必须大于0")
        return focal, "用户填写（参考图EXIF缺失）"

    def _laboratory_changed(self) -> None:
        state = "readonly" if self.laboratory_mode.get() else "disabled"
        self.lab_projection_box.configure(state=state)
        self.lab_canvas_box.configure(state=state)
        self.run_button.configure(
            text="2. 生成控制点工程" if self.control_points_only.get() else (
                "2. 实验对齐并导出16位图层" if self.laboratory_mode.get() else "2. 对齐并导出16位图层")
        )
        if self.laboratory_mode.get():
            self.status.set("实验室已启用：将创建独立Lab任务；不同投影可能使直线流星呈弧线")

    def _choose_meteors(self) -> None:
        previous_dir = self.meteor_dir.get().strip()
        previous_default = str(Path(previous_dir).parent / "Nightscape_Output") if previous_dir else ""
        path = filedialog.askdirectory(title="选择完整流星原图文件夹")
        if path:
            current_output = self.output_dir.get().strip()
            self.meteor_dir.set(path)
            if not current_output or current_output == previous_default:
                self.output_dir.set(str(Path(path).parent / "Nightscape_Output"))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出根目录（结果写入独立时间戳子目录）")
        if path:
            self.output_dir.set(path)

    def _choose_ptgui(self) -> None:
        path = filedialog.askopenfilename(title="选择PTGui程序")
        if path:
            self.ptgui_path.set(path)

    def _choose_siril(self) -> None:
        path = filedialog.askopenfilename(title="选择Siril CLI程序")
        if path:
            self.siril_path.set(path)

    def _validated_paths(self) -> tuple[Path, Path, Path, Path, Path]:
        if not self.meteor_dir.get().strip():
            raise ValueError("请选择素材文件夹")
        base, meteor_dir = Path(self.base_path.get()), Path(self.meteor_dir.get())
        ptgui, siril = Path(self.ptgui_path.get()), Path(self.siril_path.get())
        if not base.is_file():
            raise ValueError("请选择有效的对齐参考图")
        if not meteor_dir.is_dir():
            raise ValueError("请选择有效的流星原图文件夹")
        output_text = self.output_dir.get().strip()
        output = Path(output_text) if output_text else meteor_dir.parent / "Nightscape_Output"
        if not output_text:
            self.output_dir.set(str(output))
        if output.exists() and not output.is_dir():
            raise ValueError("输出路径是一个文件，请选择文件夹")
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"无法创建输出文件夹：{exc}") from exc
        return base, meteor_dir, output, ptgui, siril

    def scan(self) -> None:
        if self.running:
            return
        try:
            base, folder, _output, _ptgui, _siril = self._validated_paths()
            reference_focal, reference_source = self._reference_focal(base)
            focal, diagonal = float(self.focal_length.get()), float(self.sensor_diagonal.get())
            if not all(math.isfinite(v) and v > 0 for v in (focal, diagonal)):
                raise ValueError("镜头参数必须是有限的正数")
        except Exception as exc:
            show_copyable_error("星空对齐", str(exc), parent=self)
            return
        self.running = True
        self.last_result = None
        self._set_inputs_running(True)
        for button in (self.run_button, self.load_button, self.open_output_button, self.creative_button, self.discard_button):
            button.configure(state="disabled")
        self.status.set("正在扫描素材与 EXIF…")

        def worker(token):
            sources = list_images(folder)
            if not sources:
                raise ValueError("素材文件夹中没有可用图像")
            tracks = load_confirmed_meteor_tracks(folder)
            rows = []
            for path in sources:
                token.raise_if_cancelled()
                lens = read_lens_info(path, focal, diagonal)
                inherited = tracks.get(path.name.casefold()) or tracks.get(path.stem.casefold()) or []
                note = f"；筛选确认流星 {len(inherited)} 条" if inherited else ""
                rows.append(("等待", f"{lens.focal_length:.1f}mm", "运行时判断", "—", "—", lens.source + note))
            return sources, tracks, rows, reference_focal, reference_source

        self.background_tasks.submit(
            "scan", worker,
            on_result=lambda result: self.worker_queue.put(("scanned", result)),
            on_error=lambda exc, details: self.worker_queue.put(("scan_error", str(exc), details)),
        )

    def run(self) -> None:
        if self.running:
            return
        try:
            base, folder, output, ptgui, siril = self._validated_paths()
            # Re-scan when an editable path changed after the previous scan.
            sources = tuple(self.items)
            if not sources or list(sources) != list_images(folder):
                raise ValueError("请先成功扫描当前素材文件夹")
            tracks = {key: list(value) for key, value in self.confirmed_tracks_by_file.items()}
            export_layers = not self.control_points_only.get()
            reference_focal, _reference_source = self._reference_focal(base)
            focal = float(self.focal_length.get())
            diagonal = float(self.sensor_diagonal.get())
            laboratory = bool(self.laboratory_mode.get())
            projection = LAB_PROJECTIONS.get(self.lab_projection.get(), "rectilinear") if laboratory else "rectilinear"
            canvas_scale = LAB_CANVASES.get(self.lab_canvas.get(), 1.0) if laboratory else 1.0
            if not (math.isfinite(focal) and math.isfinite(diagonal) and focal > 0 and diagonal > 0):
                raise ValueError("EXIF缺失时使用的兜底镜头参数无效")
        except Exception as exc:
            show_copyable_error("星空对齐", str(exc), parent=self)
            return
        from toolbox import require_software
        software = require_software(self, ("ptgui", "siril"))
        if software is None:
            self.status.set("已取消，配置软件后可重新开始。")
            return
        ptgui, siril = software["ptgui"], software["siril"]
        self.ptgui_path.set(str(ptgui))
        self.siril_path.set(str(siril))
        self.running = True
        self._set_inputs_running(True)
        self.last_result = None
        self.run_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.open_output_button.configure(state="disabled")
        self.creative_button.configure(state="disabled")
        self.discard_button.configure(state="disabled")
        self.progress["value"] = 0

        def report(value: float, text: str) -> None:
            self.worker_queue.put(("progress", value, text))

        def worker(token: CancellationToken) -> AlignmentResult:
            return run_alignment_pipeline(
                base, list(sources), output, ptgui, siril, report,
                sky_fraction=None, focal_length=focal, sensor_diagonal=diagonal,
                reference_focal_length=reference_focal,
                export_layers=export_layers, laboratory=laboratory,
                panorama_projection=projection, canvas_scale=canvas_scale,
                cancel_check=token.raise_if_cancelled,
                confirmed_tracks_by_file=tracks,
            )

        def failed(exc: Exception, details: str) -> None:
            if not isinstance(exc, TaskCancelledError):
                self.worker_queue.put(("error", str(exc), details))

        self.background_tasks.submit(
            "alignment", worker,
            on_result=lambda result: self.worker_queue.put(("done", result)),
            on_error=failed, replace=True,
        )

    def _set_inputs_running(self, running: bool) -> None:
        def children(widget):
            for child in widget.winfo_children():
                yield child
                yield from children(child)
        if running:
            self._input_states = []
            for panel in (self.input_panel, self.settings_panel, self.laboratory_panel):
                for widget in children(panel):
                    if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Combobox, ttk.Spinbox)):
                        self._input_states.append((widget, widget.state()))
                        widget.state(["disabled"])
            self.scan_button.state(["disabled"])
            self.mode_button.state(["disabled"])
        else:
            for widget, state in getattr(self, "_input_states", []):
                widget.state(["!disabled"])
                widget.state(state)
            self.scan_button.state(["!disabled"])
            self.mode_button.state(["!disabled"])

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.worker_queue.get_nowait()
                if item[0] == "scanned":
                    self.running = False
                    self._set_inputs_running(False)
                    self.items, self.confirmed_tracks_by_file, rows, focal, source = item[1]
                    self.tree.delete(*self.tree.get_children())
                    for index, (path, values) in enumerate(zip(self.items, rows)):
                        self.tree.insert("", "end", iid=str(index), text=path.name, values=values)
                    self.run_button.configure(state="normal")
                    self.status.set(f"扫描完成：{len(self.items)} 张；参考图 {focal:.1f}mm（{source}）")
                elif item[0] == "progress":
                    _, value, text = item
                    self.progress["value"] = value
                    self.status.set(text)
                elif item[0] == "done":
                    self.running = False
                    self._set_inputs_running(False)
                    self.last_result = item[1]
                    self._show_result(self.last_result)
                    self.run_button.configure(state="normal")
                    has_layers = any(item.output_layer for item in self.last_result.items)
                    self.load_button.configure(state="normal" if has_layers else "disabled")
                    self.open_output_button.configure(state="normal")
                    self.creative_button.configure(state="normal" if has_layers else "disabled")
                    self.discard_button.configure(state="normal" if has_layers else "disabled")
                    mode = (
                        f"实验完成：{self.last_result.projection} · 画布 {self.last_result.canvas_scale:.0%}"
                        if self.last_result.laboratory else ("控制点工程处理完成" if not has_layers else "正式对齐完成")
                    )
                    self.status.set(f"{mode}：{self.last_result.project_dir}")
                    if messagebox.askyesno(
                        "星空对齐与分层导出",
                        f"{mode}。\n\n输出位置：\n{self.last_result.project_dir}\n\n是否打开文件夹？",
                        parent=self,
                    ):
                        self._open_output_folder()
                elif item[0] in {"error", "scan_error"}:
                    self.running = False
                    self._set_inputs_running(False)
                    self.run_button.configure(state="normal")
                    if item[0] == "scan_error":
                        self.items = []
                        self.run_button.configure(state="disabled")
                    self.status.set("扫描失败" if item[0] == "scan_error" else "对齐失败")
                    show_copyable_error(
                        "星空对齐", item[1], parent=self,
                        details=item[2] if len(item) > 2 else None,
                    )
        except queue.Empty:
            pass
        if self.winfo_exists():
            self._poll_id = self.after(120, self._poll_queue)

    def _show_result(self, result: AlignmentResult) -> None:
        by_source = {item.source: item for item in result.items}
        for index, source in enumerate(self.items):
            item = by_source.get(str(source))
            if item is None:
                continue
            error = "—" if item.median_error is None else f"{item.median_error:.2f}px"
            focal = "—" if item.focal_length is None else f"{item.focal_length:.1f}mm"
            sky = "—" if item.sky_coverage is None else f"{item.sky_coverage * 100:.1f}%"
            self.tree.item(str(index), values=(item.status, focal, sky, item.control_points or "—", error, item.message))

    def load_result(self) -> None:
        if self.last_result is None:
            return
        self.on_ready(self.last_result)
        self.destroy()

    def _open_output_folder(self) -> None:
        path = self.last_result.project_dir if self.last_result is not None else self.output_dir.get()
        try:
            open_folder(path)
        except Exception as exc:
            show_copyable_error("打开文件夹", str(exc), parent=self)

    def _request_close(self) -> None:
        if self.running:
            if not messagebox.askyesno(
                "星空对齐正在运行",
                "对齐任务还没有完成，确定要关闭吗？",
                parent=self,
            ):
                return
        elif self.last_result is not None and any(item.output_layer for item in self.last_result.items):
            if messagebox.askyesno(
                "使用已经完成的对齐结果",
                "对齐和分层导出已经完成。\n\n是否使用该结果并返回流星合成功能？",
                parent=self,
            ):
                self.load_result()
                return
        self.background_tasks.shutdown(wait=False)
        self.destroy()

    def _selected_result_items(self):
        if self.last_result is None:
            return []
        by_source = {item.source: item for item in self.last_result.items}
        selected = []
        for tree_id in self.tree.selection():
            index = int(tree_id)
            if 0 <= index < len(self.items):
                item = by_source.get(str(self.items[index]))
                if item is not None and item.status not in {"已导出", "已导出（需复查）"}:
                    selected.append(item)
        return selected

    def _persist_result_choices(self) -> None:
        if self.last_result is None:
            return
        path = Path(self.last_result.project_dir) / "alignment_manifest.json"
        from dataclasses import asdict
        path.write_text(json.dumps(asdict(self.last_result), ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_creative(self) -> None:
        selected = self._selected_result_items()
        for item in selected:
            item.status = "创意放置"
            item.output_layer = item.source
            item.message = "未经过可靠星点对齐；回载后请抠出流星并手动移动、旋转或拉伸"
        if selected:
            self._show_result(self.last_result)
            self._persist_result_choices()
            self.status.set(f"已将{len(selected)}张失败素材标记为创意放置")

    def mark_discarded(self) -> None:
        selected = self._selected_result_items()
        for item in selected:
            item.status = "已丢弃"
            item.output_layer = None
            item.message = "用户选择不使用"
        if selected:
            self._show_result(self.last_result)
            self._persist_result_choices()
            self.status.set(f"已丢弃{len(selected)}张失败素材")


def open_alignment_workspace(master, on_ready: Callable[[AlignmentResult], None], control_points_only: bool = False):
    return AlignmentWorkspace(master, on_ready, control_points_only)
