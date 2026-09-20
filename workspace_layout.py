"""Shared right-side inspector layout; scrolling never resizes the image canvas."""
import tkinter as tk
from tkinter import ttk
from itertools import count

_gradient_ids = count()


def parameter_slider(parent, title, variable, lower, upper, command=None, colors=None):
    """Compact editor row with aligned numeric readout above the slider."""
    row = ttk.Frame(parent)
    row.pack(fill='x', pady=(8, 4))
    heading = ttk.Frame(row)
    heading.pack(fill='x')
    ttk.Label(heading, text=title).pack(side='left')
    value = ttk.Label(heading, text=f'{variable.get():.1f}', style='Value.TLabel', width=6)
    value.pack(side='right')
    variable.trace_add('write', lambda *_: value.configure(text=f'{variable.get():.1f}'))
    scale = ttk.Scale(row, from_=lower, to=upper, variable=variable,
                      command=command, style='Editor.Horizontal.TScale')
    scale.pack(fill='x', pady=(5, 0))
    if colors:
        style = ttk.Style(parent)
        name = f'Gradient{next(_gradient_ids)}'
        track = tk.PhotoImage(master=parent, width=1, height=14)
        scale._gradient_image = track
        style.element_create(name+'.trough', 'image', track, sticky='we')
        style.layout(name+'.Horizontal.TScale', [(name+'.trough', {'sticky': 'we', 'children': [
            ('Editor.Scale.slider', {'side': 'left', 'sticky': ''})]})])
        scale.configure(style=name+'.Horizontal.TScale')
        stops = [tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) for color in colors]
        def paint(event):
            width = max(2, event.width)
            track.configure(width=width)
            track.put('#292929', to=(0, 0, width, 14))
            row = []
            for x in range(width):
                position = x/(width-1)*(len(stops)-1)
                index = min(len(stops)-2, int(position))
                fraction = position-index
                rgb = [round(a+(b-a)*fraction) for a, b in zip(stops[index], stops[index+1])]
                row.append('#%02x%02x%02x' % tuple(rgb))
            track.put('{'+' '.join(row)+'}', to=(0, 4, width, 10))
        scale.bind('<Configure>', paint, add=True)
    return scale


def stack_controls(frame, width=320):
    """Reflow existing controls without replacing widgets or their event bindings."""
    children = list(frame.winfo_children())
    for child in children:
        if child.winfo_manager() == 'grid':
            child.grid_forget()
        elif child.winfo_manager() == 'pack':
            child.pack_forget()
    for column in range(frame.grid_size()[0]):
        frame.columnconfigure(column, weight=0, minsize=0)
    index = 0
    while index < len(children):
        child = children[index]
        if isinstance(child, ttk.Label) and index+1 < len(children) and isinstance(children[index+1], ttk.Scale):
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=5)
            child.configure(width=16, wraplength=100, anchor='w')
            child.pack(in_=row, side='left')
            children[index+1].pack(in_=row, side='left', fill='x', expand=True, padx=4)
            child.lift()
            children[index+1].lift()
            index += 2
            if index < len(children) and isinstance(children[index], ttk.Label) and children[index].cget('textvariable'):
                children[index].configure(width=5)
                children[index].pack(in_=row, side='right')
                children[index].lift()
                index += 1
            continue
        if isinstance(child, (ttk.Frame, ttk.LabelFrame)):
            stack_controls(child, width-16)
        if isinstance(child, ttk.Label):
            child.configure(wraplength=max(180, width-24), width=0, anchor='w')
        child.pack(fill='x', pady=3, anchor='w')
        index += 1


def scroll_controls(frame, width=340, reflow=True):
    if reflow:
        stack_controls(frame, width)
    children = list(frame.pack_slaves())
    for child in children:
        child.pack_forget()
    canvas = tk.Canvas(frame, width=width, highlightthickness=0, background='#292929')
    bar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
    bar.pack(side='right', fill='y')
    canvas.pack(fill='both', expand=True)
    canvas.configure(yscrollcommand=bar.set)
    content = ttk.Frame(canvas, padding=(4, 4, 8, 8))
    item = canvas.create_window(0, 0, window=content, anchor='nw')
    canvas.bind('<Configure>', lambda e: canvas.itemconfigure(item, width=e.width))
    content.bind('<Configure>', lambda e: canvas.configure(scrollregion=(0, 0, e.width, max(e.height, canvas.winfo_height()))))
    frame._inspector_canvas = canvas
    frame._inspector_scrolled_widgets = tuple(children)
    for child in children:
        child.pack(in_=content, fill='x', pady=3)
        child.lift()
    # pack(in_=...) keeps the original Tk parent. Raise its children above
    # the later-created canvas and grouping frames, otherwise they are obscured.
    for child in frame.winfo_children():
        if child not in (canvas, bar) and not isinstance(child, (ttk.Frame, ttk.LabelFrame)):
            child.lift()
    def wheel(event):
        step = -1 if getattr(event, 'num', None) == 4 or getattr(event, 'delta', 0) > 0 else 1
        canvas.yview_scroll(step*3, 'units')
        return 'break'
    def bind(widget):
        if not isinstance(widget, (ttk.Combobox, ttk.Spinbox, ttk.Scale, ttk.Treeview)):
            for event in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
                widget.bind(event, wheel, add=True)
        for child in widget.winfo_children():
            bind(child)
    bind(frame)
    return canvas
