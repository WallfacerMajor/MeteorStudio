"""Shared right-side inspector layout; scrolling never resizes the image canvas."""
import tkinter as tk
from tkinter import ttk
from itertools import count

_gradient_ids = count()


def enable_slider_editing(scale, title=None, value=None, default=None, step=1):
    """Keyboard steps, click-to-reset heading and an editable numeric readout."""
    if getattr(scale,'_numeric_editor',False):return
    scale._numeric_editor=True
    variable=str(scale.cget('variable'))
    default=float(scale.get()) if default is None else default
    def enabled():return not scale.instate(['disabled'])
    def assign(number):
        if not enabled():return
        lower,upper=sorted((float(scale.cget('from')),float(scale.cget('to'))))
        number=round(max(lower,min(upper,number)),6)
        scale.set(number)
    def key(event,direction):
        assign(float(scale.get())+step*direction*(10 if event.state & 1 else 1))
        return 'break'
    for sequence,direction in (('<Left>',-1),('<Down>',-1),('<Right>',1),('<Up>',1)):
        scale.bind(sequence,lambda e,d=direction:key(e,d))
    scale.configure(takefocus=True)
    scale.bind('<ButtonPress-1>',lambda e:scale.focus_set(),add=True)
    if title is not None:
        title.configure(cursor='hand2')
        title.bind('<Button-1>',lambda e:assign(float(default() if callable(default) else default)))
    if value is not None:
        value.configure(cursor='xterm')
        def edit(event=None):
            if not enabled() or getattr(value,'_value_entry',None) is not None:return
            text=tk.StringVar(value=f'{float(scale.get()):g}')
            entry=ttk.Entry(value.master,textvariable=text,width=7,justify='right')
            value._value_entry=entry
            entry.place(in_=value,relwidth=1,relheight=1)
            entry.lift();entry.focus_set();entry.selection_range(0,'end')
            def finish(event=None,commit=True):
                if value._value_entry is not entry:return 'break'
                try:
                    number=float(text.get()) if commit else float(scale.get())
                    import math
                    if commit and not math.isfinite(number):raise ValueError()
                except ValueError:
                    if event is not None and event.type==tk.EventType.FocusOut:commit=False
                    else:entry.bell();return 'break'
                value._value_entry=None
                if commit:assign(number)
                entry.destroy()
                return 'break'
            entry.bind('<Return>',finish)
            entry.bind('<Escape>',lambda e:finish(e,False))
            entry.bind('<FocusOut>',finish)
        value.bind('<Button-1>',edit)
    from action_icons import ActionHint
    for widget,hint_text in ((scale,'方向键微调；Shift + 方向键加快'),(title,'单击恢复默认值'),(value,'单击输入数值；Enter 确认，Esc 取消')):
        if widget is not None:
            hint=ActionHint(widget,hint_text)
            widget.bind('<Enter>',lambda event,h=hint:h.delay(),add=True)
    scale._reset_value=default
    return scale


def parameter_slider(parent, title, variable, lower, upper, command=None, colors=None):
    """Compact editor row with aligned numeric readout above the slider."""
    row = ttk.Frame(parent)
    row.pack(fill='x', pady=(8, 4))
    heading = ttk.Frame(row)
    heading.pack(fill='x')
    label=ttk.Label(heading, text=title);label.pack(side='left')
    value = ttk.Label(heading, text=f'{variable.get():.1f}', style='Value.TLabel', width=6)
    value.pack(side='right')
    variable.trace_add('write', lambda *_: value.configure(text=f'{variable.get():.1f}'))
    scale = ttk.Scale(row, from_=lower, to=upper, variable=variable,
                      command=command, style='Editor.Horizontal.TScale')
    scale.pack(fill='x', pady=(5, 0))
    enable_slider_editing(scale,label,value,step=1 if isinstance(variable,tk.IntVar) else .1)
    if colors:
        from dpi_support import pixels
        style = ttk.Style(parent)
        name = f'Gradient{next(_gradient_ids)}'
        track_height = pixels(parent, 14)
        track = tk.PhotoImage(master=parent, width=1, height=track_height)
        scale._gradient_image = track
        style.element_create(name+'.trough', 'image', track, sticky='we')
        style.layout(name+'.Horizontal.TScale', [(name+'.trough', {'sticky': 'we', 'children': [
            ('Editor.Scale.slider', {'side': 'left', 'sticky': ''})]})])
        scale.configure(style=name+'.Horizontal.TScale')
        stops = [tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) for color in colors]
        def paint(event):
            width = max(2, event.width)
            track.configure(width=width)
            track.put('#292929', to=(0, 0, width, track_height))
            row = []
            for x in range(width):
                position = x/(width-1)*(len(stops)-1)
                index = min(len(stops)-2, int(position))
                fraction = position-index
                rgb = [round(a+(b-a)*fraction) for a, b in zip(stops[index], stops[index+1])]
                row.append('#%02x%02x%02x' % tuple(rgb))
            track.put('{'+' '.join(row)+'}', to=(0, pixels(parent, 4), width, pixels(parent, 10)))
        scale.bind('<Configure>', paint, add=True)
    return scale


def stack_controls(frame, width=320):
    """Reflow existing controls without replacing widgets or their event bindings."""
    from action_icons import is_icon_action
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
        if is_icon_action(child):
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=3)
            for _ in range(max(1, width // 48)):
                if index >= len(children) or not is_icon_action(children[index]):
                    break
                children[index].pack(in_=row, side='left', padx=(0, 5))
                children[index].lift()
                index += 1
            continue
        if isinstance(child, ttk.Label) and index+1 < len(children) and isinstance(children[index+1], ttk.Scale):
            row = ttk.Frame(frame)
            row.pack(fill='x', pady=5)
            child.configure(width=16, wraplength=100, anchor='w')
            child.pack(in_=row, side='left')
            children[index+1].pack(in_=row, side='left', fill='x', expand=True, padx=4)
            child.lift()
            children[index+1].lift()
            slider=children[index+1];numeric=None
            index += 2
            if index < len(children) and isinstance(children[index], ttk.Label) and children[index].cget('textvariable'):
                numeric=children[index]
                children[index].configure(width=5)
                children[index].pack(in_=row, side='right')
                children[index].lift()
                index += 1
            enable_slider_editing(slider,child,numeric)
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
    # Action rows retain their original Tk parent through pack(in_=...).
    frame._inspector_scrolled_widgets = tuple(child for child in frame.winfo_children() if child not in (canvas, bar))
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
