"""Shared night-sky palette for all Tk workspaces, without changing image pixels."""
from tkinter import ttk
from tkinter import font as tkfont


def apply_theme(root):
    families = set(tkfont.families(root))
    family = next((name for name in ("Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Segoe UI") if name in families), "TkDefaultFont")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        tkfont.nametofont(name, root=root).configure(family=family, size=9)
    style = ttk.Style(root)
    style.theme_use("clam")
    bg, panel, field = "#101824", "#182434", "#0e1621"
    text, muted, accent = "#e4edf6", "#9aafc5", "#76cbd1"
    root.configure(background=bg)
    root.option_add("*Background", bg)
    root.option_add("*Foreground", text)
    root.option_add("*selectBackground", "#315574")
    root.option_add("*selectForeground", "#ffffff")
    root.option_add("*Text.background", field)
    root.option_add("*Text.insertBackground", text)
    root.option_add("*Listbox.background", field)
    root.option_add("*TCombobox*Listbox.background", field)
    style.configure(".", background=bg, foreground=text, bordercolor="#304158", lightcolor=panel, darkcolor=bg, troughcolor=field, selectbackground="#315574", selectforeground="#ffffff")
    style.configure("TButton", background=panel, padding=(9, 5), focusthickness=1, focuscolor=accent)
    style.map("TButton", background=[("disabled", bg), ("pressed", "#345977"), ("active", "#263c53")], foreground=[("disabled", "#64778c")])
    style.configure("Accent.TButton", foreground=accent)
    style.configure("TEntry", fieldbackground=field, insertcolor=text, padding=4)
    style.configure("TSpinbox", fieldbackground=field, insertcolor=text, arrowsize=14)
    style.configure("TCombobox", fieldbackground=field, arrowcolor=text, padding=3)
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.map(name, fieldbackground=[("readonly", panel), ("disabled", bg)], foreground=[("disabled", "#64778c"), ("readonly", text)])
    for name in ("TCheckbutton", "TRadiobutton"):
        style.map(name, background=[("active", panel)], indicatorbackground=[("selected", accent), ("!selected", field)], foreground=[("disabled", "#64778c")])
    style.configure("TLabelframe", borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", foreground=accent)
    style.configure("Muted.TLabel", foreground=muted)
    style.configure("Hero.TLabel", font=(family, 26, "bold"))
    style.configure("Title.TLabel", font=(family, 15, "bold"))
    style.configure("Treeview", background=field, fieldbackground=field, rowheight=27)
    style.configure("Treeview.Heading", background=panel, foreground=muted, padding=5)
    style.map("Treeview", background=[("selected", "#315574")], foreground=[("selected", "#ffffff")])
    style.configure("TNotebook", borderwidth=0)
    style.configure("TNotebook.Tab", padding=(12, 7), background=panel)
    style.map("TNotebook.Tab", background=[("selected", "#29435b")], foreground=[("selected", accent)])
    style.configure("Horizontal.TProgressbar", background=accent, borderwidth=0)
    for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(name, background=panel, arrowcolor=muted, borderwidth=0)
        style.map(name, background=[("active", "#345977"), ("pressed", "#315574")])
