"""Real Tk controls on a full-size photo canvas; isolated autosave and delayed checks."""
import json,time,tempfile
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from meteor_composer import MeteorComposer,Stroke,compose_meteor_objects
from tkinter import ttk


def run():
    with tempfile.TemporaryDirectory() as directory, patch.object(MeteorComposer,'_restore_autosave'),patch.object(MeteorComposer,'_setup_autosave'):
        app=MeteorComposer();app.autosave_path=Path(directory)/'autosave.json'
        app._schedule_autosave=lambda:None
        def pump(seconds=.1):
            end=time.monotonic()+seconds
            while time.monotonic()<end:app.update();time.sleep(.005)
        def click(widget):
            app.lift();app.focus_force();pump(.05)
            x,y=widget.winfo_width()//2,widget.winfo_height()//2
            assert widget.winfo_ismapped() and x>0 and y>0
            t=time.monotonic()
            widget.event_generate('<ButtonPress-1>',x=x,y=y)
            widget.event_generate('<ButtonRelease-1>',x=x,y=y)
            elapsed=time.monotonic()-t
            pump(.02);return elapsed
        def settle():
            end=time.monotonic()+15
            while getattr(app,'_parameter_busy',False) or getattr(app,'_parameter_pending',{}):
                pump(.02);assert time.monotonic()<end
            pump(1.4)
        try:
            app.geometry('1350x920+0+0');app.show_composite_workspace()
            w,h=7952,5304
            base=np.full((h,w,3),20,np.uint8);src=base.copy()
            cv2.line(src,(2900,2500),(3400,2350),(180,200,240),6)
            key=str(Path(directory)/'source.tif');app.current_path=Path(key);app.current_dims=(w,h)
            app.files=[Path(key)];app.pairs={key:Path(directory)/'base.tif'}
            app.preview_base=base;app.preview_source=src;app.preview_aligned_source=src;app.preview_original_source=src
            stroke=Stroke([(2900/(w-1),2500/(h-1)),(3400/(w-1),2350/(h-1))],30,8,locked=True)
            app.strokes={key:[stroke]};app.candidates={};app.output_mode.set('combined');app.blend_mode.set('滤色')
            app.adjustment_defaults.update(auto_optimize=False,match_exposure=False)
            app.global_preview_rgb,_=compose_meteor_objects(src,base,[stroke],*app._object_composite_settings(key))
            app.global_preview_signature=app._global_preview_state_signature()
            app.exact_preview_full_rgb=app.global_preview_rgb;app.exact_preview_signature=app._exact_preview_state_signature()
            app.view_mode.set('blend');app.selected_object=(key,0);app._render_preview();pump(.3);app._canvas_fit()
            app.control_notebook.select(app.selected_tools_tab);app._load_selected_object_adjustments();pump(.2)
            forbidden=[]
            app._invalidate_global_preview=lambda:forbidden.append('invalidate')
            app._request_global_preview=lambda *a:forbidden.append('global')
            app._begin_exact_preview=lambda *a:forbidden.append('exact')
            app._schedule_global_exact_validation=lambda *a:forbidden.append('validation')
            latency=click(app.selected_object_controls[0]);settle()
            assert app.selected_override_enabled.get() and latency<.2,latency
            scale=next(w for w in app.selected_object_controls if isinstance(w,ttk.Scale) and str(w.cget('variable'))==str(app.selected_brightness))
            siblings=list(scale.master.winfo_children());idx=siblings.index(scale);label,value=siblings[idx-1],siblings[idx+1]
            click(scale);settle();scale.focus_force()
            prior=app.selected_brightness.get()
            scale.event_generate('<Right>');pump(.02);scale.event_generate('<Left>');settle()
            assert app.selected_brightness.get()==prior
            click(value);entry=value._value_entry;entry.delete(0,'end');entry.insert(0,'143');entry.event_generate('<Return>');settle()
            assert stroke.brightness_override==143
            click(label);settle();assert stroke.brightness_override==100
            coords=scale.coords();scale.event_generate('<ButtonPress-1>',x=int(coords[0]),y=int(coords[1]))
            t=time.monotonic()
            for fraction in (.2,.4,.6,.8):
                scale.event_generate('<B1-Motion>',x=int(scale.winfo_width()*fraction),y=int(coords[1]),state=0x100)
                pump(.01)
            scale.event_generate('<ButtonRelease-1>',x=int(scale.winfo_width()*.8),y=int(coords[1]));response=time.monotonic()-t
            settle()
            # Cache miss: deliberately slow disk decode must leave Tk responsive.
            app.current_path=Path(directory)/'other.tif'
            app._cached_layer_preview=lambda *a:None
            reads=[]
            def slow_read(path,precision=False):
                reads.append(str(path));time.sleep(.25);return src
            app._cached_full_image=slow_read
            click(value);entry=value._value_entry;entry.delete(0,'end');entry.insert(0,'132')
            t=time.monotonic();entry.event_generate('<Return>');cold_latency=time.monotonic()-t
            assert cold_latency<.2,cold_latency
            heartbeat=[];app.after(20,lambda:heartbeat.append(True));pump(.06)
            assert heartbeat and app._parameter_busy
            settle();assert reads
            expected,_=compose_meteor_objects(src,base,[stroke],*app._object_composite_settings(key))
            np.testing.assert_array_equal(app.global_preview_rgb,expected)
            assert not forbidden,forbidden
            box=app.last_incremental_box;assert (box[2]-box[0])*(box[3]-box[1])<w*h//10
            return {'cache_miss_event_seconds':cold_latency,'override_click_seconds':latency,'four_drag_events_seconds':response,'keyboard_numeric_reset':'passed','latest_pixels_match_export':'passed','no_delayed_full_rebuild':'passed','roi':box}
        finally:app.destroy()

if __name__=='__main__':print(json.dumps(run()))
