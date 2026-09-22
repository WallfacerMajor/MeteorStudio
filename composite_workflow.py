"""Lightweight workflow guidance. Never reads pixels or invalidates previews."""
from tkinter import ttk


class FileActions(ttk.Frame):
    """Common location for workspace document and output actions."""
    def add(self, text, command):
        button = ttk.Button(self, text=text, command=command)
        button.pack(side='left', padx=(0, 5))
        return button


def usable_mark(stroke):
    return not stroke.erase and stroke.opacity > 0 and bool(stroke.points)


class CompositeWorkflow(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=(0, 6))
        self.app = app
        self.signature = None
        self.stage = 0
        steps = ttk.Frame(self)
        steps.pack(fill='x')
        self.steps = []
        for text in ('1 素材', '2 流星', '3 融合', '4 导出'):
            label = ttk.Label(steps, text=text, style='Muted.TLabel')
            label.pack(side='left', expand=True)
            self.steps.append(label)
        # Fixed allocation keeps state changes from resizing the image canvas.
        body = ttk.Frame(self, height=84)
        body.pack(fill='x', pady=(5, 0))
        body.pack_propagate(False)
        self.next_button = ttk.Button(body, text='选择素材', command=self.advance)
        self.next_button.pack(side='right', padx=(6, 0))
        self.message = ttk.Label(body, text='', wraplength=240, justify='left')
        self.message.pack(side='left', fill='both', expand=True)

    def refresh(self):
        app = self.app
        ready = bool(app.files and app.pairs and app.pairing_signature == app._base_selection_signature())
        count = sum(usable_mark(mark) for key, marks in app.strokes.items()
                    if key in app.pairs for mark in marks)
        busy = bool(app.export_running)
        detection = any(button.instate(['disabled']) and not getattr(button, '_workflow_disabled', False)
                        for button in (app.auto_detect_button, app.detect_current_button))
        for button in (app.auto_detect_button, app.detect_current_button):
            if not ready:
                button._workflow_disabled = True
                button._disabled_reason = '请先选择素材并扫描'
                button.configure(state='disabled')
            elif getattr(button, '_workflow_disabled', False):
                button._workflow_disabled = False
                button._disabled_reason = ''
                button.configure(state='normal')
        stage = 0 if not ready else 1 if not count or app.view_mode.get() not in ('blend', 'labeled') else 2
        if busy:
            stage = 3
        signature = (ready, count, busy, detection, stage, bool(app.last_export_path))
        if signature == self.signature:
            return
        self.signature, self.stage = signature, stage
        for index, label in enumerate(self.steps):
            label.configure(style='Section.TLabel' if index == stage else 'Muted.TLabel')
        if not ready:
            message, action = '选择流星素材和底图\n然后扫描素材。', '选择素材'
        elif not count:
            message, action = '还没有流星标记，当前只显示底图。\n检测候选，或用画笔标出流星。', '自动检测全部'
        elif stage == 1:
            message, action = f'已有 {count} 条流星标记\n检查蒙版、删除误选，再预览融合。', '查看最终效果'
        else:
            message, action = f'已加入 {count} 条流星标记', '导出合成结果'
        if busy:
            message = '正在导出，请等待完成。'
        self.message.configure(text=message)
        self.next_button.configure(text=action, state='disabled' if busy or (ready and not count and detection) else 'normal')
        reason = '请先选择素材并扫描' if not ready else '请先检测或画出流星蒙版' if not count else '正在导出' if busy else ''
        app.load_project_button.configure(state='disabled' if busy else 'normal')
        app.load_project_button._disabled_reason = '导出完成后再载入其他项目' if busy else ''
        app.export_button._disabled_reason = reason
        app.export_button.configure(state='disabled' if reason else 'normal')
        app.open_output_button.configure(state='normal' if app.last_export_path else 'disabled')
        app.open_output_button._disabled_reason = '尚未导出结果' if not app.last_export_path else ''

    def advance(self):
        app = self.app
        if self.stage == 0:
            app._set_paths_panel_visible(True)
            if app.source_dir.get().strip() and app.base_dir.get().strip():
                app.scan_inputs()
        elif not any(usable_mark(mark) for key, marks in app.strokes.items() if key in app.pairs for mark in marks):
            app.control_notebook.select(app.mask_tools_tab)
            app._set_view_mode('source')
            app.auto_detect_all()
        elif self.stage == 2:
            app.export()
        else:
            app._set_view_mode('blend')
            app.control_notebook.select(app.blend_tools_tab)
        self.refresh()
