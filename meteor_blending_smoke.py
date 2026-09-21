"""Real Tk blend selections, sliders, delayed ROI checks and 16-bit exports."""
from contextlib import ExitStack
from unittest.mock import patch
from tkinter import ttk

import numpy as np
import tifffile


def check_blending(app, root, click, pump, wait):
    from toolbox_smoke import widgets
    from meteor_composer import compose_meteor_sources
    key = str(app.current_path)
    app.selected_object = (key, 0)
    app._load_selected_object_adjustments()
    tab = next(app.control_notebook.tabs()[i] for i in range(app.control_notebook.index('end'))
               if app.control_notebook.tab(i,'text') == '所选流星')
    app.control_notebook.select(tab)
    pump(app,.3)
    independent = next(w for w in widgets(app) if isinstance(w,ttk.Checkbutton) and w.cget('text')=='单颗独立设置')
    click(app,independent)
    pump(app,1.6)
    combo = next(w for w in widgets(app) if isinstance(w,ttk.Combobox) and str(w.cget('textvariable'))==str(app.selected_blend))
    outputs = []
    for mode in ('滤色','线性减淡（添加）'):
        with ExitStack() as stack:
            monitors = [stack.enter_context(patch.object(app,name,wraps=getattr(app,name)))
                        for name in ('_invalidate_global_preview','_global_preview_worker','_exact_preview_worker')]
            combo.focus_force()
            combo.event_generate('<ButtonPress-1>',x=combo.winfo_width()-10,y=10)
            combo.event_generate('<ButtonRelease-1>',x=combo.winfo_width()-10,y=10)
            pump(app,.1)
            popdown = app.tk.call('ttk::combobox::PopdownWindow',str(combo))
            listbox = str(popdown)+'.f.l'
            index = tuple(combo.cget('values')).index(mode)
            bounds = app.tk.call(listbox,'bbox',index)
            app.tk.call('event','generate',listbox,'<Motion>','-x',10,'-y',int(bounds[1])+int(bounds[3])//2)
            app.tk.call('event','generate',listbox,'<ButtonPress-1>','-x',10,'-y',int(bounds[1])+int(bounds[3])//2)
            app.tk.call('event','generate',listbox,'<ButtonRelease-1>','-x',10,'-y',int(bounds[1])+int(bounds[3])//2)
            pump(app,1.65)
            assert app.strokes[key][0].blend_mode_override == mode
            assert not any(m.called for m in monitors), [(m.call_count) for m in monitors]
        for variable,field in ((app.selected_star_removal,'star_removal'),(app.selected_mask_choke,'mask_choke')):
            scale = next(w for w in widgets(app) if isinstance(w,ttk.Scale) and str(w.cget('variable'))==str(variable))
            with ExitStack() as stack:
                monitors = [stack.enter_context(patch.object(app,name,wraps=getattr(app,name)))
                            for name in ('_invalidate_global_preview','_global_preview_worker','_exact_preview_worker')]
                # Drag the actual thumb; wait beyond both delayed render windows.
                old = getattr(app.strokes[key][0],field)
                start = scale.coords()
                scale.event_generate('<ButtonPress-1>',x=int(start[0]),y=int(start[1]))
                scale.event_generate('<B1-Motion>',x=max(20,scale.winfo_width()//2),y=int(start[1]),state=0x100)
                scale.event_generate('<ButtonRelease-1>',x=max(20,scale.winfo_width()//2),y=int(start[1]))
                pump(app,1.65)
                assert getattr(app.strokes[key][0],field)>0, (field,scale.winfo_width())
                assert not any(m.called for m in monitors), [m.call_count for m in monitors]
                if getattr(app.strokes[key][0],field) != old:
                    edited = getattr(app.strokes[key][0],field)
                    click(app,app.undo_button)
                    pump(app,1.65)
                    assert getattr(app.strokes[key][0],field)==old
                    click(app,app.redo_button)
                    pump(app,1.65)
                    assert getattr(app.strokes[key][0],field)==edited
                    assert not any(m.called for m in monitors), [m.call_count for m in monitors]
        with ExitStack() as stack:
            monitors = [stack.enter_context(patch.object(app,name,wraps=getattr(app,name)))
                        for name in ('_invalidate_global_preview','_global_preview_worker','_exact_preview_worker')]
            geometry = app._object_canvas_geometry(app.selected_object)
            cx,cy = (int(v) for v in geometry['center'])
            prior_x = app.strokes[key][0].offset_x
            app.canvas.event_generate('<ButtonPress-1>',x=cx,y=cy)
            app.canvas.event_generate('<B1-Motion>',x=cx+20,y=cy+10,state=0x100)
            app.canvas.event_generate('<ButtonRelease-1>',x=cx+20,y=cy+10)
            pump(app,1.65)
            assert app.strokes[key][0].offset_x != prior_x
            assert not any(m.called for m in monitors), [m.call_count for m in monitors]
            click(app,app.undo_button)
            pump(app,1.65)
            assert app.strokes[key][0].offset_x==prior_x
        expected,_ = compose_meteor_sources(app.preview_source,app.preview_source,app.preview_base,
            [app._preview_clone(s,app.current_dims[0]) for s in app.strokes[key]],*app._object_composite_settings(key))
        assert np.abs(expected.astype(int)-app.global_preview_rgb.astype(int)).max()<=1, 'ROI differs from complete blend'
        with patch('meteor_composer.messagebox.askyesnocancel',return_value=False), patch('meteor_composer.messagebox.askyesno',return_value=False), patch('meteor_composer.messagebox.showinfo'):
            click(app,app.export_button)
            wait(lambda: not app.export_running and app.last_export_path is not None)
        files = list(app.last_export_path.rglob('*.tif'))+list(app.last_export_path.rglob('*.tiff'))
        rgb = tifffile.imread(files[0])
        assert rgb.dtype==np.uint16 and rgb.max()>3000
        outputs.append(rgb)
    assert np.any(outputs[1]>outputs[0]), 'Screen and Add exports were identical'
