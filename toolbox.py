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
    ToolGroup("laboratory", "实验室", "星轨叠加、已对齐降噪与画质分析", (
        WorkspaceSpec("star_reduction", "缩星", "本地方案与 Siril · 同步对比 · 16 位 TIFF", "open_star_reduction_workspace"),
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


def iter_tools(nodes, path=()):
    for node in nodes:
        if isinstance(node, ToolGroup):
            yield from iter_tools(node.children, (*path, node.key))
        else:
            yield node, path


def tool_menu_button(parent, app):
    button = ttk.Menubutton(parent, text="切换工具")
    menu = tk.Menu(button, tearoff=False)
    for index, group in enumerate(TOOL_MENU):
        if index:
            menu.add_separator()
        menu.add_command(label=group.title, state="disabled")
        for spec, path in iter_tools(group.children, (group.key,)):
            menu.add_command(label=spec.title, command=lambda item=spec, category=path: app.navigate_tool(item, category))
    button.configure(menu=menu)
    from action_icons import iconize_actions
    iconize_actions(button)
    return button


def settings_menu_button(parent):
    button = ttk.Menubutton(parent, text="设置")
    menu = tk.Menu(button, tearoff=False)
    owner = parent.winfo_toplevel()
    from runtime_log import install_log_access, toggle_runtime_log
    install_log_access(owner, parent)
    menu.add_command(label="外部软件与路径…", command=lambda: show_software_settings(owner))
    from workspace_help import TOPICS, show_help
    help_menu = tk.Menu(menu, tearoff=False)
    for topic in TOPICS:
        help_menu.add_command(label=topic, command=lambda topic=topic: show_help(owner, topic))
    menu.add_cascade(label="使用说明", menu=help_menu)
    menu.add_separator()
    from error_dialog import show_runtime_log
    menu.add_command(label="运行日志 · Ctrl+L", command=lambda: toggle_runtime_log(owner))
    button.configure(menu=menu)
    from action_icons import iconize_actions
    iconize_actions(button)
    return button


def show_software_settings(parent, keys=None, required=False):
    """One modal preferences dialog; closing it never starts a pending operation."""
    registry = SoftwareRegistry()
    keys = tuple(keys or registry.specs)
    dialog = tk.Toplevel(parent)
    dialog.title("设置 · 外部软件")
    dialog.transient(parent)
    dialog.geometry("640x410")
    dialog.minsize(500, 350)
    body = ttk.Frame(dialog, padding=20)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="外部软件", style="Title.TLabel").pack(anchor="w")
    ttk.Label(body, text="当前操作需要配置以下软件。选择程序后点击继续。" if required else "软件会自动查找；仅在未找到或需要更换版本时指定路径。",
              style="Muted.TLabel", wraplength=550).pack(fill="x", pady=(6, 16))
    dialog.path_values, dialog.choose_buttons = {}, {}
    for key in keys:
        spec = registry.specs[key]
        row = ttk.Frame(body)
        row.pack(fill="x", pady=6)
        ttk.Label(row, text=spec.title, width=10).pack(side="left")
        value = tk.StringVar(value=str(registry.resolve(key) or "未找到程序"))
        dialog.path_values[key] = value
        ttk.Entry(row, textvariable=value, state="readonly").pack(side="left", fill="x", expand=True, padx=8)
        def choose(key=key, value=value):
            path = filedialog.askopenfilename(parent=dialog, title="选择程序文件（macOS 请进入 .app/Contents/MacOS）")
            if path:
                try:
                    registry.save(key, path)
                    value.set(path)
                    message.set("已保存，后续操作自动使用此路径。")
                except (OSError, ValueError) as exc:
                    from error_dialog import show_copyable_error
                    show_copyable_error("软件设置", str(exc), parent=dialog)
        choose_button = ttk.Button(row, text="选择…", command=choose)
        choose_button.pack(side="right")
        dialog.choose_buttons[key] = choose_button
    message = tk.StringVar()
    ttk.Label(body, textvariable=message, style="Muted.TLabel", wraplength=550).pack(fill="x", pady=10)
    footer = ttk.Frame(body)
    footer.pack(side="bottom", fill="x")
    accepted = False
    def finish():
        nonlocal accepted
        if required and any(registry.resolve(key) is None for key in keys):
            message.set("请先为以上软件选择有效的程序文件，或取消当前操作。")
            return
        accepted = True
        dialog.destroy()
    dialog.close_button = ttk.Button(footer, text="取消" if required else "关闭", command=dialog.destroy)
    dialog.close_button.pack(side="left")
    dialog.done_button = ttk.Button(footer, text="继续" if required else "完成", style="Primary.TButton", command=finish)
    dialog.done_button.pack(side="right")
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.bind("<Escape>", lambda e: dialog.destroy())
    parent._software_settings_dialog = dialog
    from action_icons import iconize_actions
    iconize_actions(dialog)
    dialog.wait_visibility()
    dialog.grab_set()
    parent.wait_window(dialog)
    parent._software_settings_dialog = None
    # Release widget/variable references on Tk's thread before a background
    # worker can collect the closed dialog's Python reference cycles.
    dialog.choose_buttons.clear()
    dialog.path_values.clear()
    return accepted


def require_software(parent, keys):
    registry = SoftwareRegistry()
    missing = tuple(key for key in keys if registry.resolve(key) is None)
    if missing and not show_software_settings(parent, missing, required=True):
        return None
    registry = SoftwareRegistry()
    paths = {key: registry.resolve(key) for key in keys}
    return paths if all(paths.values()) else None


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
    """Keep categories visible, with one-click access to every workspace."""
    home = ttk.Frame(app, padding=24)
    navigation = ttk.Frame(home)
    navigation.pack(fill="x", pady=(0, 18))
    ttk.Label(navigation, text=PRODUCT_NAME, style="Hero.TLabel").pack(side="left")
    settings_menu_button(navigation).pack(side="right")
    cards = ttk.Frame(home)
    cards.pack(fill="x")
    home.tool_buttons = {}
    for index, group in enumerate(TOOL_MENU):
        card = ttk.LabelFrame(cards, text=group.title, padding=14)
        card.grid(row=index // 2, column=index % 2, sticky="nsew", padx=5, pady=7)
        for tool_index, (spec, path) in enumerate(iter_tools(group.children, (group.key,))):
            button = ttk.Button(card, text=spec.title, command=lambda item=spec, category=path: app.navigate_tool(item, category))
            button.grid(row=0, column=tool_index, padx=6, pady=6)
            home.tool_buttons[spec.key] = button
    for col in range(2):
        cards.columnconfigure(col, weight=1, uniform="tool")
    home.recent_project_buttons = []
    recent = app._recent_project_paths()[:4]
    if recent:
        section = ttk.LabelFrame(home, text="最近项目", padding=12)
        section.pack(fill="x", pady=(16, 0))
        for index, path in enumerate(recent):
            button = ttk.Button(
                section, text=f"{path.name}  ·  {path.parent.name}",
                command=lambda selected=path: app._load_project_path(selected),
            )
            button._keep_text_action = True
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=5, pady=4)
            home.recent_project_buttons.append(button)
        section.columnconfigure(0, weight=1)
        section.columnconfigure(1, weight=1)
    from action_icons import iconize_actions
    iconize_actions(home)
    return home
