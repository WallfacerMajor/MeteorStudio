"""Presentation-only filtering; hidden materials still participate in export."""
import tkinter as tk
import time
from tkinter import ttk


class MaterialList:
    def __init__(self, parent, app):
        self.app = app
        self.source_filter = tk.StringVar(value='全部图像')
        self.status_filter = tk.StringVar(value='全部状态')
        filters = ttk.Frame(parent)
        filters.pack(fill='x')
        for variable, values in (
            (self.source_filter, ('全部图像', '对齐图', '原始图')),
            (self.status_filter, ('全部状态', '有蒙版', '无蒙版', '混合来源', '需复查')),
        ):
            combo = ttk.Combobox(filters, textvariable=variable, values=values, state='readonly', width=10)
            combo.pack(side='left', fill='x', expand=True, padx=(0, 3))
            combo.bind('<<ComboboxSelected>>', lambda e: self.refresh())
        self.tree = ttk.Treeview(parent, columns=('source', 'status'), show='tree headings', selectmode='browse')
        for column, title, width in (('#0','文件',160),('source','查看图像',72),('status','流星',90)):
            self.tree.heading(column,text=title)
            self.tree.column(column,width=width,minwidth=50,anchor='w' if column=='#0' else 'center')
        self.tree.pack(fill='both',expand=True,pady=4)
        self.tree.bind('<ButtonRelease-1>',self.source_menu,add=True)
        self.tree.bind('<Button-3>',self.source_menu,add=True)
        self.source_editor = None
        self.tree.bind('<MouseWheel>',lambda e:self.close_editor(),add=True)
        self.tree.bind('<Unmap>',lambda e:self.close_editor(),add=True)

    def close_editor(self):
        if self.source_editor is not None:
            self.source_editor.destroy()
            self.source_editor=None

    def source_label(self, key):
        app=self.app
        if key in app.use_original_sources or key not in app.original_sources:
            return '原始图'
        return '对齐图'

    def reset(self):
        self.source_filter.set('全部图像')
        self.status_filter.set('全部状态')
        # Detached rows must be deleted too, before reusing numeric row ids.
        for index in range(len(self.app.files)):
            if self.tree.exists(str(index)):
                self.tree.move(str(index),'','end')
        for iid in getattr(self,'known_rows',()):
            if self.tree.exists(iid):
                self.tree.move(iid,'','end')

    def refresh(self, only_key=None):
        self.close_editor()
        app=self.app
        app.tree_selection_suppress_until = time.monotonic()+.2
        self.known_rows = [str(i) for i in range(len(app.files))]
        attached=set(self.tree.get_children())
        wanted=attached.copy()
        for index,path in enumerate(app.files):
            iid,key=str(index),str(path)
            if only_key is not None and key != only_key:
                continue
            if not self.tree.exists(iid):
                continue
            marks=[s for s in app.strokes.get(key,[]) if not s.erase and s.points]
            modes={s.source_mode for s in marks}
            source=self.source_label(key)
            review='需复查' in app.alignment_statuses.get(key,'')
            detail=str(len(marks)) if marks else '—'
            if marks:
                detail+=' · '+('混合' if len(modes)>1 else ('原始' if 'original' in modes or key not in app.original_sources else '对齐'))
            self.tree.item(iid,text=path.name,values=(source,detail),tags=('review',) if review else ())
            condition=self.status_filter.get()
            visible=(self.source_filter.get() in ('全部图像',source) and
                (condition=='全部状态' or condition=='有蒙版' and bool(marks) or
                 condition=='无蒙版' and not marks or condition=='混合来源' and len(modes)>1 or
                 condition=='需复查' and review))
            if visible:
                wanted.add(iid)
            else:
                if iid in self.tree.selection():
                    self.tree.selection_remove(iid)
                wanted.discard(iid)
        if wanted != attached:
            self.tree.set_children('',*sorted(wanted,key=int))
        self.tree.tag_configure('review',foreground='#e2b46c')

    def source_menu(self,event):
        self.close_editor()
        iid=self.tree.identify_row(event.y)
        if not iid or (event.num != 3 and self.tree.identify_column(event.x) != '#1'):
            return
        self.app._select_tree_item(iid,load=False)
        key=str(self.app.files[int(iid)])
        available=key in self.app.original_sources and self.app.original_sources[key].is_file()
        if not available:
            return 'break'
        x,y,width,height=self.tree.bbox(iid,'source')
        editor=self.source_editor=ttk.Combobox(self.tree,values=('对齐图','原始图'),state='readonly')
        editor.set(self.source_label(key))
        editor.place(x=x,y=y,width=width,height=height)
        editor.bind('<<ComboboxSelected>>',lambda e:self.change_source('original' if editor.get()=='原始图' else 'aligned'))
        editor.bind('<Escape>',lambda e:self.close_editor())
        editor.focus_set()
        editor.event_generate('<ButtonPress-1>',x=width-5,y=height//2)
        editor.event_generate('<ButtonRelease-1>',x=width-5,y=height//2)
        return 'break'

    def change_source(self,mode):
        self.app._set_current_source_state(mode)
        self.refresh()
