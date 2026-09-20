"""Shared night-sky palette for all Tk workspaces, without changing image pixels."""
from tkinter import ttk
from tkinter import font as tkfont
import tkinter as tk
import sys


def apply_theme(root):
    from app_icon import install_icon
    install_icon(root)
    families = set(tkfont.families(root))
    family = next((name for name in ("Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Segoe UI") if name in families), "TkDefaultFont")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name, root=root).configure(family=family, size=9)
    style = ttk.Style(root)
    style.theme_use("clam")
    bg, panel, field = "#292929", "#353535", "#202020"
    text, muted, accent = "#eeeeee", "#b8b8b8", "#81aee8"
    root.configure(background=bg)
    root.option_add("*Background", bg)
    root.option_add("*Foreground", text)
    root.option_add("*selectBackground", "#315574")
    root.option_add("*selectForeground", "#ffffff")
    root.option_add("*Text.background", field)
    root.option_add("*Text.insertBackground", text)
    root.option_add("*Listbox.background", field)
    root.option_add("*TCombobox*Listbox.background", field)
    # Windows draws native popup menus outside ttk's theme. A global light
    # foreground on its system light background makes their entries invisible.
    if sys.platform == 'win32':
        menu_colors = dict(background='SystemMenu', foreground='SystemMenuText',
                           activeBackground='SystemHighlight', activeForeground='SystemHighlightText',
                           disabledForeground='SystemGrayText', selectColor='SystemMenuText')
    else:
        menu_colors = dict(background=panel, foreground=text,
                           activeBackground='#405b7a', activeForeground='#ffffff',
                           disabledForeground=muted, selectColor=text)
    for option, value in menu_colors.items():
        root.option_add(f'*Menu.{option}', value)
    style.configure(".", background=bg, foreground=text, bordercolor="#454545", lightcolor=panel, darkcolor=bg, troughcolor=field, selectbackground="#405b7a", selectforeground="#ffffff")
    style.configure("TButton", background=panel, borderwidth=0, padding=(8, 4), focusthickness=1, focuscolor=accent)
    style.map("TButton", background=[("disabled", bg), ("pressed", "#345977"), ("active", "#263c53")], foreground=[("disabled", "#64778c")])
    style.configure("Accent.TButton", foreground=accent)
    style.configure("Primary.TButton", background="#496c96", foreground="#ffffff", borderwidth=0, padding=(10, 5))
    style.map("Primary.TButton", background=[("disabled", panel), ("pressed", "#3e5c80"), ("active", "#587ead")], foreground=[("disabled", "#858585"), ("!disabled", "#ffffff")])
    style.configure("Quiet.TButton", background=bg, borderwidth=0, padding=(8, 4))
    style.configure("Section.TLabel", foreground=text, font=(family, 10, "bold"))
    style.configure("Value.TLabel", foreground=text, anchor="e")
    style.configure("Editor.Horizontal.TScale", borderwidth=0, troughcolor="#555555", background=bg, sliderlength=14, sliderthickness=12)
    style.configure("Thin.Horizontal.TProgressbar", thickness=3, borderwidth=0)
    style.configure("TMenubutton", background=bg, borderwidth=0, padding=(7, 3), arrowcolor=muted)
    style.configure("Horizontal.TSeparator", background="#444444", borderwidth=0)
    style.configure("Vertical.TSeparator", background="#181818", borderwidth=0)
    style.configure("TEntry", fieldbackground=field, insertcolor=text, padding=4)
    style.configure("TSpinbox", fieldbackground=field, insertcolor=text, arrowsize=14)
    style.configure("TCombobox", fieldbackground=field, arrowcolor=text, padding=3)
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.map(name, fieldbackground=[("readonly", panel), ("disabled", bg)], foreground=[("disabled", "#64778c"), ("readonly", text)])
    for name in ("TCheckbutton", "TRadiobutton"):
        style.map(name, background=[("active", panel)], indicatorbackground=[("selected", accent), ("!selected", field)], foreground=[("disabled", "#64778c")])
    style.configure("TLabelframe", borderwidth=1, relief="solid", bordercolor="#404040")
    style.configure("TLabelframe.Label", foreground=text, font=(family, 9, "bold"))
    style.configure("Muted.TLabel", foreground=muted)
    style.configure("Hero.TLabel", font=(family, 22, "bold"))
    style.configure("Title.TLabel", font=(family, 12, "bold"))
    style.configure("Treeview", background=field, fieldbackground=field, rowheight=27)
    style.configure("Treeview.Heading", background=panel, foreground=muted, padding=5)
    style.map("Treeview", background=[("selected", "#315574")], foreground=[("selected", "#ffffff")])
    style.configure("TNotebook", borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 6), background=field)
    style.map("TNotebook.Tab", background=[("selected", "#414141")], foreground=[("selected", text)])
    style.configure("Horizontal.TProgressbar", background=accent, borderwidth=0)
    for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(name, background=panel, arrowcolor=muted, borderwidth=0)
        style.map(name, background=[("active", "#345977"), ("pressed", "#315574")])
    # A quiet track and distinct thumb replace clam's bulky, ridged slider.
    if 'Editor.Scale.trough' in style.element_names():
        return
    dpi = max(1., float(root.tk.call('tk', 'scaling')) / (96 / 72))
    size = max(12, round(14*dpi))
    track = tk.PhotoImage(master=root, width=3, height=size)
    track.put('#555555', to=(0, size//2-1, 3, size//2+1))
    thumb = tk.PhotoImage(master=root, width=size, height=size)
    for y in range(size):
        for x in range(size):
            if (x-(size-1)/2)**2+(y-(size-1)/2)**2 < (size*.34)**2:
                thumb.put('#d7d7d7', (x, y))
    root._editor_scale_images = (track, thumb)
    for name, image in (('Editor.Scale.trough', track), ('Editor.Scale.slider', thumb)):
        if name not in style.element_names():
            style.element_create(name, 'image', image, border=0, sticky='we')
    layout = [('Editor.Scale.trough', {'sticky': 'we', 'children': [('Editor.Scale.slider', {'side': 'left', 'sticky': ''})]})]
    style.layout('Horizontal.TScale', layout)
    style.layout('Editor.Horizontal.TScale', layout)
    progress_track = tk.PhotoImage(master=root, width=3, height=3)
    progress_track.put(field, to=(0, 0, 3, 3))
    progress_fill = tk.PhotoImage(master=root, width=3, height=3)
    progress_fill.put(accent, to=(0, 0, 3, 3))
    root._editor_progress_images = (progress_track, progress_fill)
    style.element_create('Editor.Progress.trough', 'image', progress_track, sticky='we')
    style.element_create('Editor.Progress.pbar', 'image', progress_fill, sticky='nswe')
    progress_layout = [('Editor.Progress.trough', {'sticky': 'nswe', 'children': [('Editor.Progress.pbar', {'side': 'left', 'sticky': 'ns'})]})]
    style.layout('Horizontal.TProgressbar', progress_layout)
    style.layout('Thin.Horizontal.TProgressbar', progress_layout)
