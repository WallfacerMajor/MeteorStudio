"""Product navigation and extensible external-software registry.

Workspace IDs and existing project/storage formats are deliberately independent
of the displayed product name. Adding a tool does not require a new root window.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog

PRODUCT_NAME = "星野工具箱"


@dataclass(frozen=True)
class WorkspaceSpec:
    key: str
    title: str
    description: str
    action: str


WORKSPACES = (
    WorkspaceSpec("screening", "流星批量筛选", "时序分析 · RAW 预览 · 候选确认", "open_screening_workspace"),
    WorkspaceSpec("control_points", "Siril + PTGui 控制点", "星点匹配 · 每张独立工程 · 无需导出图层", "open_control_points_workspace"),
    WorkspaceSpec("alignment", "星空对齐", "独立镜头参数 · 16 位图层 · 对齐结果回载", "open_alignment_workspace"),
    WorkspaceSpec("composite", "流星合成功能", "手绘蒙版 · 逐颗融合 · 原始像素导出", "show_composite_workspace"),
    WorkspaceSpec("video", "视频动态", "流星慢放 · 时间曲线 · 高质量视频", "open_video_workspace"),
)


@dataclass(frozen=True)
class ToolGroup:
    key: str
    title: str
    description: str
    children: tuple[WorkspaceSpec | ToolGroup, ...]


TOOL_MENU = (
    ToolGroup("meteor", "流星工具", "批量筛选、星空对齐、蒙版合成与视频动态", tuple(item for item in WORKSPACES if item.key != "control_points")),
    ToolGroup("control_points", "控制点生成", "连接外部软件，为星野照片自动生成控制点", (WORKSPACES[1],)),
    ToolGroup("color", "色彩工具", "白平衡与中性色校正 · 保留完整像素精度", (
        WorkspaceSpec("white_balance", "白平衡与改机校准", "灰卡取样 · 机身／滤镜预设 · 整组 16 位导出", "open_white_balance_workspace"),
        WorkspaceSpec("light_pollution", "光污染渐变校正", "底部渐变 · 地景／星云保护 · 16 位导出", "open_light_pollution_workspace"),
    )),
    ToolGroup("laboratory", "实验室", "探索星轨、降噪与画质分析 · 独立输出，保留原片", (
        WorkspaceSpec("trails", "星轨叠加", "固定机位 · 取亮叠加 · 16 位 TIFF", "open_lab_trails"),
        WorkspaceSpec("mean", "已对齐降噪", "平均叠加 · 保持原始尺寸 · 16 位 TIFF", "open_lab_mean"),
        WorkspaceSpec("quality", "批量画质体检", "清晰度 · 背景亮度 · 过曝比例 · CSV", "open_lab_quality"),
    )),
)


def menu_level(path=(), nodes=TOOL_MENU):
    """Resolve arbitrary submenu depth from one shared navigation tree."""
    ancestors = []
    for key in path:
        group = next((node for node in nodes if node.key == key and isinstance(node, ToolGroup)), None)
        if group is None:
            raise ValueError(f"未知工具分类：{key}")
        ancestors.append(group)
        nodes = group.children
    return nodes, ancestors


def tool_menu_button(parent, app):
    button = ttk.Menubutton(parent, text="工具菜单 ▾")
    menu = tk.Menu(button, tearoff=False)
    def populate(parent_menu, nodes, path=()):
        for node in nodes:
            if isinstance(node, ToolGroup):
                submenu = tk.Menu(parent_menu, tearoff=False)
                populate(submenu, node.children, (*path, node.key))
                parent_menu.add_cascade(label=node.title, menu=submenu)
            else:
                parent_menu.add_command(label=node.title, command=lambda spec=node, category=path: app.navigate_tool(spec, category))
    populate(menu, TOOL_MENU)
    button.configure(menu=menu)
    return button


@dataclass(frozen=True)
class SoftwareSpec:
    key: str
    title: str
    executables: tuple[str, ...]
    candidates: tuple[str, ...] = ()


SOFTWARE = (
    SoftwareSpec("siril", "Siril CLI", ("siril-cli",), (r"C:\Program Files\Siril\bin\siril-cli.exe", r"C:\Program Files\Siril\siril-cli.exe", "/Applications/Siril.app/Contents/MacOS/siril-cli")),
    SoftwareSpec("ptgui", "PTGui", ("PTGui", "ptgui"), (r"C:\Program Files\PTGui\PTGui.exe", "/Applications/PTGui Pro.app/Contents/MacOS/PTGui Pro")),
    SoftwareSpec("ffmpeg", "FFmpeg", ("ffmpeg",)),
)


def settings_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("APPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "MeteorStudio" / "software.json"


class SoftwareRegistry:
    def __init__(self, path: Path | None = None, specs=SOFTWARE):
        self.path = path or settings_path()
        self.specs = {spec.key: spec for spec in specs}
        self.paths = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.paths = {k: v for k, v in data.items() if isinstance(v, str)}
        except (OSError, ValueError):
            pass

    def resolve(self, key: str) -> Path | None:
        spec = self.specs[key]
        # An explicit missing selection must be repaired, never silently replaced.
        if self.paths.get(key):
            selected = Path(self.paths[key])
            return selected if selected.is_file() else None
        bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
        candidates = [bundled / (name + (".exe" if sys.platform == "win32" else "")) for name in spec.executables]
        candidates += [Path(p) for p in spec.candidates]
        candidates += [Path(p) for name in spec.executables if (p := shutil.which(name))]
        return next((p for p in candidates if p.is_file()), None)

    def save(self, key: str, path: str) -> None:
        if key not in self.specs:
            raise ValueError(f"未知软件：{key}")
        if path and not Path(path).is_file():
            raise ValueError("请选择有效的程序文件")
        updated = dict(self.paths, **{key: path})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)
        self.paths = updated


def build_home(app, menu_path=()) -> ttk.Frame:
    nodes, ancestors = menu_level(menu_path)
    home = ttk.Frame(app, padding=28)
    navigation = ttk.Frame(home)
    navigation.pack(fill="x", pady=(0, 12))
    tool_menu_button(navigation, app).pack(side="right")
    if menu_path:
        ttk.Button(navigation, text="← 返回上级", command=lambda: app.show_toolbox(menu_path[:-1])).pack(side="left", padx=(0, 12))
        ttk.Button(navigation, text=PRODUCT_NAME, command=app.show_toolbox).pack(side="left")
        for index, group in enumerate(ancestors):
            ttk.Label(navigation, text=" / ", style="Muted.TLabel").pack(side="left")
            ttk.Button(navigation, text=group.title, command=lambda path=menu_path[:index + 1]: app.show_toolbox(path)).pack(side="left")
    ttk.Label(home, text="N I G H T S C A P E   /   T O O L B O X", style="Muted.TLabel").pack(anchor="w")
    ttk.Label(home, text=ancestors[-1].title if ancestors else PRODUCT_NAME, style="Hero.TLabel").pack(anchor="w", pady=(8, 4))
    ttk.Label(home, text=ancestors[-1].description if ancestors else "选择工具分类，开始处理你的星野作品。", style="Muted.TLabel").pack(anchor="w", pady=(0, 22))
    cards = ttk.Frame(home)
    cards.pack(fill="both", expand=True)
    home.tool_buttons = {}
    for index, spec in enumerate(nodes):
        card = ttk.LabelFrame(cards, text=f"0{index + 1}   {spec.title}", padding=16)
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=5, pady=5)
        ttk.Label(card, text=spec.description, wraplength=270, style="Muted.TLabel").pack(anchor="w", pady=(0, 18))
        group = isinstance(spec, ToolGroup)
        callback = (lambda path=(*menu_path, spec.key): app.show_toolbox(path)) if group else (lambda item=spec: app.navigate_tool(item, menu_path))
        button = ttk.Button(card, text="打开子菜单  →" if group else "进入工作区  →", style="Accent.TButton", command=callback)
        button.pack(anchor="w", side="bottom")
        home.tool_buttons[spec.key] = button
    for col in range(2):
        cards.columnconfigure(col, weight=1, uniform="tool")
    for row in range((len(nodes) + 1) // 2):
        cards.rowconfigure(row, weight=1)
    if menu_path:
        footer = ttk.Frame(home)
        footer.pack(fill="x", pady=(14, 0))
        ttk.Label(footer, text="源素材只读   /   本地处理   /   独立输出", style="Muted.TLabel").pack(side="left")
        ttk.Button(footer, text="软件连接设置 →", command=app.show_toolbox).pack(side="right")
        return home
    software = ttk.LabelFrame(home, text="外部软件 · 连接与路径", padding=12)
    software.pack(fill="x", pady=(20, 8))
    registry = SoftwareRegistry()
    for row, spec in enumerate(SOFTWARE):
        value = tk.StringVar(value=str(registry.resolve(spec.key) or "未找到程序，请选择路径"))
        ttk.Label(software, text=spec.title, width=12).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Entry(software, textvariable=value, state="readonly").grid(row=row, column=1, sticky="ew", padx=12)
        def choose(key=spec.key, variable=value):
            path = filedialog.askopenfilename(parent=app, title="选择程序文件（macOS 请进入 .app/Contents/MacOS）")
            if path:
                try:
                    registry.save(key, path)
                    variable.set(path)
                except (OSError, ValueError) as exc:
                    from error_dialog import show_copyable_error
                    show_copyable_error("软件路径", str(exc), parent=app)
        ttk.Button(software, text="选择程序…", command=choose).grid(row=row, column=2)
    software.columnconfigure(1, weight=1)
    ttk.Label(home, text="源素材只读   /   本地处理   /   独立输出", style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
    return home
