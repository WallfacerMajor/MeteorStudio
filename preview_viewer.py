"""Standalone export-quality image viewer used by MeteorStudio."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class ExactPreviewViewer(tk.Toplevel):
    """Scrollable full-resolution viewer for the export-equivalent composite."""

    def __init__(
        self, parent: tk.Misc, final_image: np.ndarray, labeled_image: np.ndarray,
        initial_mode: str = "blend",
    ) -> None:
        super().__init__(parent)
        self.title("精准预览")
        self.geometry("1280x820")
        self.minsize(760, 520)
        self.images = {"blend": final_image, "labeled": labeled_image}
        self.mode = tk.StringVar(value="labeled" if initial_mode == "labeled" else "blend")
        self.zoom = 1.0
        self.center_x = final_image.shape[1] / 2.0
        self.center_y = final_image.shape[0] / 2.0
        self.fit_pending = True
        self.fit_mode = False
        self.drag_start: tuple[int, int, float, float] | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.photo_size: tuple[int, int] | None = None
        self.image_item: int | None = None
        self.zoom_label = tk.StringVar()

        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill="x")
        ttk.Radiobutton(
            toolbar, text="最终效果", variable=self.mode, value="blend", command=self._render
        ).pack(side="left")
        ttk.Radiobutton(
            toolbar, text="来源标注", variable=self.mode, value="labeled", command=self._render
        ).pack(side="left", padx=(8, 18))
        ttk.Button(toolbar, text="适合窗口", command=self.fit).pack(side="left")
        ttk.Button(toolbar, text="100%", command=self.actual_size).pack(side="left", padx=4)
        ttk.Button(toolbar, text="−", width=3, command=lambda: self._zoom_by(1 / 1.25)).pack(side="left")
        ttk.Button(toolbar, text="+", width=3, command=lambda: self._zoom_by(1.25)).pack(side="left", padx=(4, 0))
        ttk.Label(toolbar, textvariable=self.zoom_label).pack(side="left", padx=12)

        self.canvas = tk.Canvas(self, background="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda event: self._wheel_steps(event, 1))
        self.canvas.bind("<Button-5>", lambda event: self._wheel_steps(event, -1))
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.canvas.bind("<Double-Button-1>", self._toggle_actual)
        self.bind("<KeyPress-0>", lambda _event: self.fit())
        self.bind("<KeyPress-1>", lambda _event: self.actual_size())
        self.bind("<KeyPress-plus>", lambda _event: self._zoom_by(1.25))
        self.bind("<KeyPress-minus>", lambda _event: self._zoom_by(1 / 1.25))
        self.after_idle(self.actual_size)

    def _image(self) -> np.ndarray:
        return self.images[self.mode.get()]

    def _fit_scale(self) -> float:
        image = self._image()
        height, width = image.shape[:2]
        return min(
            max(1, self.canvas.winfo_width()) / max(1, width),
            max(1, self.canvas.winfo_height()) / max(1, height),
        )

    def fit(self) -> None:
        image = self._image()
        self.center_x = image.shape[1] / 2.0
        self.center_y = image.shape[0] / 2.0
        self.zoom = self._fit_scale()
        self.fit_pending = False
        self.fit_mode = True
        self._render()

    def actual_size(self) -> None:
        self.zoom = 1.0
        self.fit_pending = False
        self.fit_mode = False
        self._clamp_center()
        self._render()

    def _toggle_actual(self, _event=None) -> str:
        if abs(self.zoom - 1.0) < 0.02:
            self.fit()
        else:
            self.actual_size()
        return "break"

    def _view_origin(self) -> tuple[float, float]:
        return (
            self.center_x - self.canvas.winfo_width() / (2.0 * self.zoom),
            self.center_y - self.canvas.winfo_height() / (2.0 * self.zoom),
        )

    def _clamp_center(self) -> None:
        image = self._image()
        height, width = image.shape[:2]
        half_w = self.canvas.winfo_width() / (2.0 * self.zoom)
        half_h = self.canvas.winfo_height() / (2.0 * self.zoom)
        self.center_x = (
            width / 2.0 if half_w >= width / 2.0
            else float(np.clip(self.center_x, half_w, width - half_w))
        )
        self.center_y = (
            height / 2.0 if half_h >= height / 2.0
            else float(np.clip(self.center_y, half_h, height - half_h))
        )

    def _zoom_by(self, factor: float, anchor: tuple[int, int] | None = None) -> None:
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        anchor_x, anchor_y = anchor or (canvas_w // 2, canvas_h // 2)
        origin_x, origin_y = self._view_origin()
        image_x = origin_x + anchor_x / self.zoom
        image_y = origin_y + anchor_y / self.zoom
        fit_scale = self._fit_scale()
        self.zoom = float(np.clip(self.zoom * factor, fit_scale * 0.5, 8.0))
        new_origin_x = image_x - anchor_x / self.zoom
        new_origin_y = image_y - anchor_y / self.zoom
        self.center_x = new_origin_x + canvas_w / (2.0 * self.zoom)
        self.center_y = new_origin_y + canvas_h / (2.0 * self.zoom)
        self.fit_pending = False
        self.fit_mode = False
        self._clamp_center()
        self._render()

    def _wheel(self, event) -> str:
        return self._wheel_steps(event, 1 if event.delta > 0 else -1)

    def _wheel_steps(self, event, steps: int) -> str:
        self._zoom_by(1.25 ** steps, (int(event.x), int(event.y)))
        return "break"

    def _pan_start(self, event) -> str:
        self.drag_start = (event.x, event.y, self.center_x, self.center_y)
        self.fit_mode = False
        self.canvas.configure(cursor="fleur")
        return "break"

    def _pan_move(self, event) -> str:
        if self.drag_start is None:
            return "break"
        start_x, start_y, center_x, center_y = self.drag_start
        self.center_x = center_x - (event.x - start_x) / self.zoom
        self.center_y = center_y - (event.y - start_y) / self.zoom
        self._clamp_center()
        self._render()
        return "break"

    def _pan_end(self, _event=None) -> str:
        self.drag_start = None
        self.canvas.configure(cursor="")
        return "break"

    def _on_configure(self, _event=None) -> None:
        if self.fit_pending or self.fit_mode:
            self.fit()
        else:
            self._clamp_center()
            self._render()

    def _render(self) -> None:
        if not self.winfo_exists():
            return
        image = self._image()
        height, width = image.shape[:2]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self._clamp_center()
        origin_x, origin_y = self._view_origin()
        x0 = max(0, int(np.floor(origin_x)))
        y0 = max(0, int(np.floor(origin_y)))
        x1 = min(width, int(np.ceil(origin_x + canvas_w / self.zoom)))
        y1 = min(height, int(np.ceil(origin_y + canvas_h / self.zoom)))
        if x1 <= x0 or y1 <= y0:
            return
        crop = Image.fromarray(image[y0:y1, x0:x1])
        display_w = max(1, int(round((x1 - x0) * self.zoom)))
        display_h = max(1, int(round((y1 - y0) * self.zoom)))
        if (display_w, display_h) != crop.size:
            resample = Image.Resampling.LANCZOS if self.zoom < 1.0 else Image.Resampling.BICUBIC
            crop = crop.resize((display_w, display_h), resample)
        draw_x = int(round((x0 - origin_x) * self.zoom))
        draw_y = int(round((y0 - origin_y) * self.zoom))
        if self.photo is not None and self.photo_size == (display_w, display_h):
            self.photo.paste(crop)
        else:
            self.photo = ImageTk.PhotoImage(crop)
            self.photo_size = (display_w, display_h)
        if self.image_item is None or not self.canvas.type(self.image_item):
            self.canvas.delete("all")
            self.image_item = self.canvas.create_image(draw_x, draw_y, anchor="nw", image=self.photo)
        else:
            self.canvas.itemconfigure(self.image_item, image=self.photo)
            self.canvas.coords(self.image_item, draw_x, draw_y)
        self.zoom_label.set(f"{self.zoom * 100:.0f}% · {width}×{height}")
