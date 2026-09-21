"""Exercise list filters and source-state menus using real Tk events."""
import tempfile
import time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
from tkinter import ttk

import numpy as np
import tifffile


def choose(app, combo, value, pump):
    combo.event_generate('<ButtonPress-1>',x=combo.winfo_width()-8,y=10)
    combo.event_generate('<ButtonRelease-1>',x=combo.winfo_width()-8,y=10)
    pump(app,.1)
    popup=str(app.tk.call('ttk::combobox::PopdownWindow',str(combo)))+'.f.l'
    index=tuple(combo.cget('values')).index(value)
    bounds=app.tk.call(popup,'bbox',index)
    y=int(bounds[1])+int(bounds[3])//2
    for event in ('<Motion>','<ButtonPress-1>','<ButtonRelease-1>'):
        app.tk.call('event','generate',popup,event,'-x',10,'-y',y)
    pump(app,.2)


def run_smoke(app):
    from toolbox_smoke import click,pump,widgets
    from meteor_composer import Stroke
    app.show_composite_workspace()
    app.geometry('1500x1000+0+0')
    pump(app,.3)
    def wait(predicate):
        deadline=time.monotonic()+25
        while not predicate():
            pump(app,.05)
            assert time.monotonic()<deadline,app.status.get()
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        app.autosave_path=root/'autosave.json'
        inputs=root/'inputs';inputs.mkdir()
        frame=np.full((300,450,3),3500,np.uint16)
        tifffile.imwrite(root/'base.tif',frame)
        tifffile.imwrite(root/'original.tif',frame+500)
        for name in ('a','b','c'):
            tifffile.imwrite(inputs/(name+'.tif'),frame)
        app.source_dir.set(str(inputs));app.base_dir.set(str(root/'base.tif'))
        app.output_dir.set(str(root/'output'));app.output_mode.set('combined')
        click(app,app.composite_workflow.next_button)
        wait(lambda: app.current_path is not None and app.preview_source is not None)
        pump(app,1.6)
        keys=[str(p) for p in app.files]
        app.original_sources={key:root/'original.tif' for key in keys[:2]}
        app.strokes={keys[0]:[Stroke([(.2,.4),(.5,.3)],12,3),Stroke([(.4,.6),(.7,.7)],12,3,source_mode='original')],
                     keys[1]:[Stroke([(.2,.3),(.5,.4)],12,3,source_mode='original')]}
        app.alignment_statuses={keys[1]:'需复查'}
        browser=app.material_list;browser.refresh()
        assert app.blend_preview_label.get()=='显示融合'
        assert app.source_preview_label.get()=='显示来源'
        assert app.tree.item('0','values')==('对齐图','2 · 混合')
        assert app.tree.item('2','values')==('原始图','—')
        combos=[w for w in widgets(browser.tree.master) if isinstance(w,ttk.Combobox)]
        source_combo=next(w for w in combos if str(w.cget('textvariable'))==str(browser.source_filter))
        state_combo=next(w for w in combos if str(w.cget('textvariable'))==str(browser.status_filter))
        def viewport():
            return (app.display_box,app.canvas_center_x,app.canvas_center_y,app.canvas.winfo_width(),app.canvas.winfo_height())
        view=viewport()
        before=app._project_data()
        with ExitStack() as stack:
            monitors=[stack.enter_context(patch.object(app,name,wraps=getattr(app,name))) for name in
                      ('load_selected','_invalidate_global_preview','_global_preview_worker','_exact_preview_worker')]
            for combo,value,rows in ((source_combo,'原始图',('2',)),(source_combo,'对齐图',('0','1')),
                (state_combo,'混合来源',('0',)),(state_combo,'无蒙版',()),(source_combo,'全部图像',('2',)),
                (state_combo,'需复查',('1',)),(state_combo,'有蒙版',('0','1')),(state_combo,'全部状态',('0','1','2'))):
                choose(app,combo,value,pump)
                pump(app,1.65)
                assert app.tree.get_children()==rows,(value,app.tree.get_children())
                assert viewport()==view
                assert not any(m.called for m in monitors),[m.call_count for m in monitors]
        assert app._project_data()==before, 'Filtering changed project/export state'
        choose(app,source_combo,'对齐图',pump)
        app._select_tree_item('0',load=False)
        box=app.tree.bbox('0','source')
        x,y=box[0]+box[2]//2,box[1]+box[3]//2
        app.tree.event_generate('<ButtonPress-1>',x=x,y=y)
        app.tree.event_generate('<ButtonRelease-1>',x=x,y=y)
        pump(app,.2)
        popup=str(app.tk.call('ttk::combobox::PopdownWindow',str(browser.source_editor)))+'.f.l'
        bounds=app.tk.call(popup,'bbox',1)
        y=int(bounds[1])+int(bounds[3])//2
        for event in ('<Motion>','<ButtonPress-1>','<ButtonRelease-1>'):
            app.tk.call('event','generate',popup,event,'-x',10,'-y',y)
        pump(app,1.65)
        assert keys[0] in app.use_original_sources
        assert app.tree.item('0','values')[0]=='原始图'
        assert [s.source_mode for s in app.strokes[keys[0]]]==['aligned','original']
        assert app.tree.get_children()==('1',), 'Source filter did not follow changed state'
        choose(app,source_combo,'原始图',pump)
        app.canvas.focus_force()
        pump(app,.1)
        app.canvas.event_generate('<KeyPress-Right>')
        wait(lambda: app.current_path==app.files[2])
        assert app.tree.selection()==('2',), 'Navigation entered a hidden row'
        # Reloading with detached rows must not retain duplicate numeric ids.
        choose(app,source_combo,'原始图',pump)
        with patch('meteor_composer.messagebox.askyesno',return_value=True):
            app.scan_inputs()
        pump(app,1.65)
        assert app.tree.get_children()==('0','1','2')
    return dict.fromkeys(('view_names','source_column','mixed_meteor_sources','source_and_state_filters',
        'empty_filter','filter_preserves_viewport_and_starts_no_pixel_workers','filter_does_not_change_export',
        'click_row_source_selector','view_switch_preserves_per_meteor_sources','filtered_keyboard_navigation','rescan_detached_rows'),'passed')
