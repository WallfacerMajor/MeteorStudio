from __future__ import annotations

import json
import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from ptgui_pipeline import (
    AlignmentResult,
    default_ptgui_path,
    default_siril_path,
    list_images,
    read_lens_info,
    run_alignment_pipeline,
)
from platform_utils import open_folder
from error_dialog import show_copyable_error, show_runtime_log


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
    def __init__(self, master, on_ready: Callable[[AlignmentResult], None]) -> None:
        super().__init__(master)
        self.title("MeteorStudio — Siril星点＋PTGui对齐")
        self.geometry("1080x720")
        self.minsize(900, 620)
        self.on_ready = on_ready
        self.base_path = tk.StringVar()
        self.meteor_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.ptgui_path = tk.StringVar(value=str(default_ptgui_path() or ""))
        self.siril_path = tk.StringVar(value=str(default_siril_path() or ""))
        self.focal_length = tk.DoubleVar(value=14.0)
        self.sensor_diagonal = tk.DoubleVar(value=43.2666)
        self.laboratory_mode = tk.BooleanVar(value=False)
        self.lab_projection = tk.StringVar(value="墨卡托（宽幅弧线）")
        self.lab_canvas = tk.StringVar(value="扩展公共天空 135%")
        self.status = tk.StringVar(value="选择对齐参考图和流星原图文件夹；输出文件夹会自动创建，也可以手动更改。")
        self.items = []
        self.worker_queue: queue.Queue = queue.Queue()
        self.running = False
        self.last_result: AlignmentResult | None = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._request_close)
        self.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="星空对齐与分层导出", font=("TkDefaultFont", 15, "bold")).pack(side="left")
        ttk.Button(header, text="运行日志", command=lambda: show_runtime_log(self)).pack(side="right")
        ttk.Label(
            root,
            text="Siril只辅助寻找星点；PTGui以对齐参考图建立星空控制点并原生导出图层。流星在返回主工作区后再抠。",
        ).pack(anchor="w", pady=(2, 10))
        paths = ttk.LabelFrame(root, text="输入、工具与输出（源素材只读）", padding=8)
        paths.pack(fill="x")
        self._path_row(paths, 0, "对齐参考图", self.base_path, self._choose_base, "选择文件…")
        self._path_row(paths, 1, "完整流星原图文件夹", self.meteor_dir, self._choose_meteors, "选择文件夹…")
        self._path_row(paths, 2, "输出文件夹（可选）", self.output_dir, self._choose_output, "另选文件夹…")
        self._path_row(paths, 3, "PTGui程序", self.ptgui_path, self._choose_ptgui, "选择程序…")
        self._path_row(paths, 4, "Siril CLI程序", self.siril_path, self._choose_siril, "选择程序…")
        paths.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(root, text="镜头与星空区域", padding=8)
        settings.pack(fill="x", pady=(8, 0))
        ttk.Label(settings, text="EXIF缺失时兜底焦距(mm)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=1, to=1000, increment=0.1, textvariable=self.focal_length, width=9).grid(row=0, column=1, padx=(5, 18))
        ttk.Label(settings, text="传感器对角线(mm)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(settings, from_=1, to=100, increment=0.1, textvariable=self.sensor_diagonal, width=9).grid(row=0, column=3, padx=(5, 18))
        ttk.Label(settings, text="星空区域").grid(row=0, column=4, sticky="w")
        ttk.Label(settings, text="逐张自动识别并生成星点蒙版").grid(row=0, column=5, columnspan=3, sticky="w", padx=5)
        settings.columnconfigure(5, weight=1)
        ttk.Label(
            settings,
            text="优先读取每张图片的EXIF焦距；焦段变化时使用独立镜头模型。传感器和焦距输入只在EXIF缺失时兜底。",
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(5, 0))

        laboratory = ttk.LabelFrame(root, text="对齐实验室（每次独立输出，不覆盖正式结果）", padding=8)
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
        ttk.Label(
            laboratory, text="参考图和全部流星层会一起重投影",
        ).pack(side="left", padx=(16, 0))

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=8)
        self.scan_button = ttk.Button(actions, text="1. 扫描素材", command=self.scan)
        self.scan_button.pack(side="left")
        self.run_button = ttk.Button(actions, text="2. 对齐并导出16位图层", command=self.run, state="disabled")
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
        self.creative_button = ttk.Button(actions, text="选中失败项→创意放置", command=self.mark_creative, state="disabled")
        self.creative_button.pack(side="right")
        self.discard_button = ttk.Button(actions, text="丢弃选中失败项", command=self.mark_discarded, state="disabled")
        self.discard_button.pack(side="right", padx=6)

        self.tree = ttk.Treeview(root, columns=("status", "focal", "sky", "cp", "error", "message"), show="tree headings", selectmode="extended")
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
        self.tree.pack(fill="both", expand=True)

        footer = ttk.Frame(root)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status).pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, length=260)
        self.progress.pack(side="right")

    def _path_row(self, parent, row, label, variable, callback, button_text) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=2)
        ttk.Button(parent, text=button_text, command=callback).grid(row=row, column=2, pady=2)

    def _choose_base(self) -> None:
        path = filedialog.askopenfilename(title="选择对齐参考图", filetypes=[("图像", "*.tif *.tiff *.jpg *.jpeg *.png")])
        if path:
            self.base_path.set(path)

    def _laboratory_changed(self) -> None:
        state = "readonly" if self.laboratory_mode.get() else "disabled"
        self.lab_projection_box.configure(state=state)
        self.lab_canvas_box.configure(state=state)
        self.run_button.configure(
            text="2. 实验对齐并导出16位图层"
            if self.laboratory_mode.get() else "2. 对齐并导出16位图层"
        )
        if self.laboratory_mode.get():
            self.status.set("实验室已启用：将创建独立Lab任务；不同投影可能使直线流星呈弧线")

    def _choose_meteors(self) -> None:
        previous_dir = self.meteor_dir.get().strip()
        previous_default = str(Path(previous_dir) / "MeteorStudio_Output") if previous_dir else ""
        path = filedialog.askdirectory(title="选择完整流星原图文件夹")
        if path:
            current_output = self.output_dir.get().strip()
            self.meteor_dir.set(path)
            if not current_output or current_output == previous_default:
                self.output_dir.set(str(Path(path) / "MeteorStudio_Output"))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹（可与输入目录相同或位于其中）")
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
        base, meteor_dir = Path(self.base_path.get()), Path(self.meteor_dir.get())
        ptgui, siril = Path(self.ptgui_path.get()), Path(self.siril_path.get())
        if not base.is_file():
            raise ValueError("请选择有效的对齐参考图")
        if not meteor_dir.is_dir():
            raise ValueError("请选择有效的流星原图文件夹")
        output_text = self.output_dir.get().strip()
        output = Path(output_text) if output_text else meteor_dir / "MeteorStudio_Output"
        if not output_text:
            self.output_dir.set(str(output))
        if output.exists() and not output.is_dir():
            raise ValueError("输出路径是一个文件，请选择文件夹")
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ValueError(f"无法创建输出文件夹：{exc}") from exc
        if not ptgui.is_file() or not siril.is_file():
            raise ValueError("找不到PTGui或Siril CLI程序")
        return base, meteor_dir, output, ptgui, siril

    def scan(self) -> None:
        try:
            _base, folder, _output, _ptgui, _siril = self._validated_paths()
            self.items = list_images(folder)
            if not self.items:
                raise ValueError("流星原图文件夹中没有可用图像")
            for item in self.tree.get_children():
                self.tree.delete(item)
            for index, path in enumerate(self.items):
                lens = read_lens_info(path, float(self.focal_length.get()), float(self.sensor_diagonal.get()))
                self.tree.insert(
                    "", "end", iid=str(index), text=path.name,
                    values=("等待", f"{lens.focal_length:.1f}mm", "运行时判断", "—", "—", lens.source),
                )
            self.run_button.configure(state="normal")
            self.status.set(f"只读扫描完成：{len(self.items)}张完整流星原图")
        except Exception as exc:
            show_copyable_error("星空对齐", str(exc), parent=self)

    def run(self) -> None:
        if self.running:
            return
        try:
            base, _folder, output, ptgui, siril = self._validated_paths()
            focal = float(self.focal_length.get())
            diagonal = float(self.sensor_diagonal.get())
            laboratory = bool(self.laboratory_mode.get())
            projection = LAB_PROJECTIONS.get(self.lab_projection.get(), "rectilinear") if laboratory else "rectilinear"
            canvas_scale = LAB_CANVASES.get(self.lab_canvas.get(), 1.0) if laboratory else 1.0
            if not (focal > 0 and diagonal > 0):
                raise ValueError("EXIF缺失时使用的兜底镜头参数无效")
        except Exception as exc:
            show_copyable_error("星空对齐", str(exc), parent=self)
            return
        self.running = True
        self.last_result = None
        self.run_button.configure(state="disabled")
        self.load_button.configure(state="disabled")
        self.progress["value"] = 0

        def report(value: float, text: str) -> None:
            self.worker_queue.put(("progress", value, text))

        def worker() -> None:
            try:
                result = run_alignment_pipeline(
                    base, self.items.copy(), output, ptgui, siril, report,
                    sky_fraction=None, focal_length=focal, sensor_diagonal=diagonal,
                    export_layers=True, laboratory=laboratory,
                    panorama_projection=projection, canvas_scale=canvas_scale,
                )
                self.worker_queue.put(("done", result))
            except Exception as exc:
                self.worker_queue.put(("error", str(exc), traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self.worker_queue.get_nowait()
                if item[0] == "progress":
                    _, value, text = item
                    self.progress["value"] = value
                    self.status.set(text)
                elif item[0] == "done":
                    self.running = False
                    self.last_result = item[1]
                    self._show_result(self.last_result)
                    self.run_button.configure(state="normal")
                    self.load_button.configure(state="normal")
                    self.open_output_button.configure(state="normal")
                    self.creative_button.configure(state="normal")
                    self.discard_button.configure(state="normal")
                    mode = (
                        f"实验完成：{self.last_result.projection} · 画布 {self.last_result.canvas_scale:.0%}"
                        if self.last_result.laboratory else "正式对齐完成"
                    )
                    self.status.set(f"{mode}：{self.last_result.project_dir}")
                    if messagebox.askyesno(
                        "星空对齐与分层导出",
                        f"{mode}。\n\n输出位置：\n{self.last_result.project_dir}\n\n是否打开文件夹？",
                        parent=self,
                    ):
                        self._open_output_folder()
                elif item[0] == "error":
                    self.running = False
                    self.run_button.configure(state="normal")
                    self.status.set("对齐失败")
                    show_copyable_error(
                        "星空对齐", item[1], parent=self,
                        details=item[2] if len(item) > 2 else None,
                    )
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(120, self._poll_queue)

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
        elif self.last_result is not None:
            if messagebox.askyesno(
                "使用已经完成的对齐结果",
                "对齐和分层导出已经完成。\n\n是否使用该结果并返回流星合成功能？",
                parent=self,
            ):
                self.load_result()
                return
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
                if item is not None and item.status != "已导出":
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


def open_alignment_workspace(master, on_ready: Callable[[AlignmentResult], None]):
    return AlignmentWorkspace(master, on_ready)
