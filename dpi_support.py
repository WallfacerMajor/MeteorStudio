"""Native Windows rendering and logical-pixel layout for Tk 8.6.

Fonts already use points in Tk; only pixel-based layout values are scaled here.
macOS keeps its native Cocoa/Retina coordinate handling.
"""
import re
import sys
import tkinter as tk
from tkinter import ttk


def enable_high_dpi():
    """Call before creating Tk, including when running without the EXE manifest."""
    if sys.platform != 'win32':
        return
    import ctypes
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    try:
        setter = user32.SetProcessDpiAwarenessContext
        setter.argtypes = [ctypes.c_void_p]
        setter.restype = ctypes.c_bool
        if setter(ctypes.c_void_p(-4)) or ctypes.get_last_error() == 5:
            # Access denied means the manifest/host already set awareness.
            return
    except AttributeError:
        pass
    try:
        if ctypes.windll.shcore.SetProcessDpiAwareness(2) == 0:
            return
    except (AttributeError, OSError):
        pass
    user32.SetProcessDPIAware()


def display_scale(widget):
    if sys.platform != 'win32':
        return 1.0
    return max(.75, round(float(widget.tk.call('tk', 'scaling')) / (96 / 72), 2))


def pixels(widget, value):
    return round(value * display_scale(widget))


def _scaled(widget, value, factor):
    """Scale unitless pixel lengths, leaving point/physical-unit lengths alone."""
    values = widget.tk.splitlist(value) if isinstance(value, str) else value
    if not isinstance(values, (tuple, list)):
        values = (values,)
    converted = []
    for item in values:
        try:
            converted.append(round(float(str(item)) * factor))
        except ValueError:
            converted.append(item)
    return tuple(converted) if len(converted) != 1 else converted[0]


def scale_layout(parent):
    """Scale a completed UI once, without changing image pixels or canvas items.

    Navigation can add widgets later. Widget and geometry-manager bookkeeping
    are independent because some buttons are styled before they are packed.
    """
    factor = display_scale(parent)
    for child in parent.winfo_children():
        scale_layout(child)
    if not getattr(parent, '_dpi_options_scaled', False):
        parent._dpi_options_scaled = True
        options = set(parent.keys())
        lengths = options & {'padding', 'padx', 'pady', 'wraplength', 'borderwidth', 'highlightthickness'}
        if isinstance(parent, (tk.Frame, tk.LabelFrame, tk.Canvas, ttk.Frame, ttk.LabelFrame)):
            lengths |= options & {'width', 'height'}
        if isinstance(parent, (ttk.Scale, ttk.Progressbar)):
            lengths |= options & {'length'}
        if factor != 1:
            for option in lengths:
                value = parent.cget(option)
                if str(value):
                    parent.configure(**{option: _scaled(parent, value, factor)})
            if isinstance(parent, ttk.Treeview):
                for column in ('#0', *parent.tk.splitlist(parent.cget('columns'))):
                    for option in ('width', 'minwidth'):
                        parent.column(column, **{option: round(parent.column(column, option) * factor)})
        if isinstance(parent, (tk.Tk, tk.Toplevel)) and not parent.overrideredirect():
            # wm geometry reports 1x1 until the initial request has settled.
            parent.update_idletasks()
            geometry = re.match(r'(\d+)x(\d+)', parent.geometry())
            if geometry and factor != 1:
                width, height = map(int, geometry.groups())
                if width > 1 and height > 1:
                    screen_w, screen_h = parent.winfo_screenwidth(), parent.winfo_screenheight()
                    width = min(round(width * factor), int(screen_w * .94))
                    height = min(round(height * factor), int(screen_h * .88))
                    parent.geometry(f'{width}x{height}+{(screen_w-width)//2}+{max(0, (screen_h-height)//2-20)}')
                min_w, min_h = parent.minsize()
                parent.minsize(min(round(min_w * factor), width), min(round(min_h * factor), height))
    manager = parent.winfo_manager()
    if manager in ('pack', 'grid') and not getattr(parent, '_dpi_manager_scaled', False):
        parent._dpi_manager_scaled = True
        if factor != 1:
            info = parent.pack_info() if manager == 'pack' else parent.grid_info()
            values = {key: _scaled(parent, info[key], factor) for key in ('padx', 'pady', 'ipadx', 'ipady') if key in info}
            if manager == 'pack':
                parent.pack_configure(**values)
            else:
                parent.grid_configure(**values)


def scale_theme(root, style):
    factor = display_scale(root)
    for name in ('.', 'TButton', 'Primary.TButton', 'Quiet.TButton', 'Icon.TButton',
                 'Primary.Icon.TButton', 'Icon.TMenubutton', 'Icon.Toolbutton',
                 'Toolbutton', 'TMenubutton', 'TEntry', 'TSpinbox', 'TCombobox',
                 'Treeview.Heading', 'TNotebook.Tab', 'TLabelframe',
                 'Horizontal.TSeparator', 'Vertical.TSeparator'):
        config = style.configure(name) or {}
        changes = {key: _scaled(root, value, factor) for key, value in config.items()
                   if key in ('padding', 'arrowsize', 'borderwidth', 'focusthickness')}
        if changes:
            style.configure(name, **changes)
    from tkinter import font
    style.configure('Treeview', rowheight=max(pixels(root, 29), font.nametofont('TkDefaultFont', root=root).metrics('linespace') + pixels(root, 8)))
