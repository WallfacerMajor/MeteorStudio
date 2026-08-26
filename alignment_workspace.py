from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from ptgui_pipeline import (
    AlignmentResult,
    default_ptgui_path,
    default_siril_path,
    list_images,
    run_alignment_pipeline,
)


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
        self.sky_fraction = tk.DoubleVar(value=60.4)
        self.status = tk.StringVar(value="选择底图、流星原图文件夹和独立输出位置。")
        self.items = []
        self.worker_queue: queue.Queue = queue.Queue()
        self.running = False
        self.last_result: AlignmentResult | None = None
        self._build_ui()
        self.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="星空对齐与分层导出", font=("TkDefaultFont", 15, "bold")).pack(side="left")
        ttk.Button(header, text="返回图片合成工作区", command=self.destroy).pack(side="right")
        ttk.Label(
            root,
            text="Siril只辅助寻找星点；PTGui按双底图锁定流程优化并原生导出。流星在返回主工作区后再抠。",
        ).pack(anchor="w", pady=(2, 10))
        paths = ttk.LabelFrame(root, text="输入、工具与输出（源素材只读）", padding=8)
        paths.pack(fill="x")
        self._path_row(paths, 0, "干净底图", self.base_path, self._choose_base, "选择文件…")
        self._path_row(paths, 1, "完整流星原图文件夹", self.meteor_dir, self._choose_meteors, "选择文件夹…")
        self._path_row(paths, 2, "独立输出文件夹", self.output_dir, self._choose_output, "选择文件夹…")
        self._path_row(paths, 3, "PTGui程序", self.ptgui_path, self._choose_ptgui, "选择程序…")
        self._path_row(paths, 4, "Siril CLI程序", self.siril_path, self._choose_siril, "选择程序…")
        paths.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(root, text="镜头与星空区域", padding=8)
        settings.pack(fill="x", pady=(8, 0))
        ttk.Label(settings, text="镜头焦距(mm)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(settings, from_=1, to=1000, increment=0.1, textvariable=self.focal_length, width=9).grid(row=0, column=1, padx=(5, 18))
        ttk.Label(settings, text="传感器对角线(mm)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(settings, from_=1, to=100, increment=0.1, textvariable=self.sensor_diagonal, width=9).grid(row=0, column=3, padx=(5, 18))
        ttk.Label(settings, text="画面顶部作为星空区域").grid(row=0, column=4, sticky="w")
        ttk.Scale(settings, from_=35, to=100, variable=self.sky_fraction, orient="horizontal").grid(row=0, column=5, sticky="ew", padx=5)
        ttk.Label(settings, textvariable=self.sky_fraction, width=5).grid(row=0, column=6)
        ttk.Label(settings, text="%", width=2).grid(row=0, column=7)
        settings.columnconfigure(5, weight=1)
        ttk.Label(
            settings,
            text="全画幅默认对角线43.27mm。星空区域应避开地景；本组素材已验证约60.4%。",
        ).grid(row=1, column=0, columnspan=8, sticky="w", pady=(5, 0))

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=8)
        self.scan_button = ttk.Button(actions, text="1. 扫描素材", command=self.scan)
        self.scan_button.pack(side="left")
        self.run_button = ttk.Button(actions, text="2. 对齐并导出16位图层", command=self.run, state="disabled")
        self.run_button.pack(side="left", padx=8)
        self.load_button = ttk.Button(actions, text="3. 返回主工作区抠流星", command=self.load_result, state="disabled")
        self.load_button.pack(side="left")
        self.creative_button = ttk.Button(actions, text="选中失败项→创意放置", command=self.mark_creative, state="disabled")
        self.creative_button.pack(side="right")
        self.discard_button = ttk.Button(actions, text="丢弃选中失败项", command=self.mark_discarded, state="disabled")
        self.discard_button.pack(side="right", padx=6)

        self.tree = ttk.Treeview(root, columns=("status", "cp", "error", "message"), show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="流星原图")
        self.tree.heading("status", text="状态")
        self.tree.heading("cp", text="控制点")
        self.tree.heading("error", text="匹配误差")
        self.tree.heading("message", text="说明")
        self.tree.column("#0", width=260)
        self.tree.column("status", width=90, anchor="center")
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
        path = filedialog.askopenfilename(title="选择干净底图", filetypes=[("图像", "*.tif *.tiff *.jpg *.jpeg *.png")])
        if path:
            self.base_path.set(path)

    def _choose_meteors(self) -> None:
        path = filedialog.askdirectory(title="选择完整流星原图文件夹")
        if path:
            self.meteor_dir.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择独立输出文件夹")
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
        base, meteor_dir, output = Path(self.base_path.get()), Path(self.meteor_dir.get()), Path(self.output_dir.get())
        ptgui, siril = Path(self.ptgui_path.get()), Path(self.siril_path.get())
        if not base.is_file():
            raise ValueError("请选择有效的干净底图")
        if not meteor_dir.is_dir():
            raise ValueError("请选择有效的流星原图文件夹")
        if not output.is_dir():
            raise ValueError("请选择已经存在的独立输出文件夹")
        if output.resolve() == meteor_dir.resolve() or output.resolve() == base.parent.resolve():
            raise ValueError("输出文件夹不能与任一输入位置相同")
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
                self.tree.insert("", "end", iid=str(index), text=path.name, values=("等待", "—", "—", ""))
            self.run_button.configure(state="normal")
            self.status.set(f"只读扫描完成：{len(self.items)}张完整流星原图")
        except Exception as exc:
            messagebox.showerror("星空对齐", str(exc), parent=self)

    def run(self) -> None:
        if self.running:
            return
        try:
            base, _folder, output, ptgui, siril = self._validated_paths()
            focal = float(self.focal_length.get())
            diagonal = float(self.sensor_diagonal.get())
            sky = float(self.sky_fraction.get()) / 100.0
            if not (focal > 0 and diagonal > 0 and 0.35 <= sky <= 1.0):
                raise ValueError("镜头或星空区域参数无效")
        except Exception as exc:
            messagebox.showerror("星空对齐", str(exc), parent=self)
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
                    sky_fraction=sky, focal_length=focal, sensor_diagonal=diagonal,
                    export_layers=True,
                )
                self.worker_queue.put(("done", result))
            except Exception as exc:
                self.worker_queue.put(("error", str(exc)))

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
                    self.creative_button.configure(state="normal")
                    self.discard_button.configure(state="normal")
                    self.status.set(f"完成：{self.last_result.project_dir}")
                elif item[0] == "error":
                    self.running = False
                    self.run_button.configure(state="normal")
                    self.status.set("对齐失败")
                    messagebox.showerror("星空对齐", item[1], parent=self)
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
            self.tree.item(str(index), values=(item.status, item.control_points or "—", error, item.message))

    def load_result(self) -> None:
        if self.last_result is None:
            return
        self.on_ready(self.last_result)
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
