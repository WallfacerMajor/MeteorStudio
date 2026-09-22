"""Serialize large-photo parameter previews without blocking Tk or copying a frame."""
from dataclasses import replace
from types import SimpleNamespace, MethodType
import threading


class RegionResult:
    def __init__(self, frame):
        self.frame=frame;self.shape=frame.shape;self.patch=None;self.slices=None
    def __getitem__(self, slices):return self.frame[slices].copy()
    def __setitem__(self, slices, value):self.slices=slices;self.patch=value
    def copy(self):return self


def schedule(app, before):
    if app.preview_base is None or app.global_preview_rgb is None:return False
    current=app._selected_stroke()
    mask_edit=current is not None and (current.star_removal>0 or any(
        getattr(current,name)!=getattr(before,name)
        for name in ('star_removal','mask_choke','feather')))
    if not mask_edit and app.preview_base.shape[0]*app.preview_base.shape[1]<4_000_000:return False
    pending=getattr(app,'_parameter_pending',None)
    if pending is None:app._parameter_pending={};pending=app._parameter_pending
    pending.setdefault(app.selected_object,before)
    app._cancel_deferred_full_preview_work()
    if not getattr(app,'_parameter_busy',False):start(app)
    return True


def start(app):
    pending=app._parameter_pending
    if not pending:return
    reference,before=next(iter(pending.items()))
    base=app.preview_base;frame=app.global_preview_rgb
    if base is None or frame is None:pending.clear();return
    signature=app._global_preview_state_signature()
    exact_signature=app._exact_preview_state_signature()
    generation=app.global_preview_generation
    context=SimpleNamespace()
    for name in ('preview_base','current_path','current_dims','preview_aligned_source',
                 'preview_original_source','preview_source'):
        setattr(context,name,getattr(app,name))
    context.selected_object=reference
    context.strokes={k:[replace(s,points=s.points.copy()) for s in v] for k,v in app.strokes.items()}
    context.pairs=app.pairs.copy();context.original_sources=app.original_sources.copy()
    context.adjustment_defaults=app.adjustment_defaults.copy()
    context.image_adjustments={k:v.copy() for k,v in app.image_adjustments.items()}
    for name in ('base_exposure_tenths','blend_mode','output_mode'):
        value=getattr(app,name).get()
        setattr(context,name,SimpleNamespace(get=lambda v=value:v))
    result=RegionResult(frame)
    context.global_preview_rgb=context.exact_preview_full_rgb=context.preview_rgb=result
    for name in ('_cached_layer_preview','_cached_full_image'):
        setattr(context,name,getattr(app,name))  # these caches already use locks
    for name in ('_uses_shared_base','_object_canvas_width','_preview_clone','_object_composite_settings',
                 '_incremental_parameter_change_image','_incremental_adjusted_object_image',
                 '_incremental_selected_object_image','_incremental_recomposed_object_image'):
        setattr(context,name,MethodType(getattr(type(app),name),context))
    completed=[];app._parameter_busy=True
    def work():
        try:
            value=context._incremental_selected_object_image(before)
            completed.append((value,getattr(context,'last_incremental_delete_error',None)))
        except Exception:
            import traceback
            completed.append((None,traceback.format_exc()))
    threading.Thread(target=work,daemon=True,name='meteor-parameter-preview').start()
    def poll():
        if not completed:app.after(16,poll);return
        app._parameter_busy=False
        value,error=completed[0]
        if app.preview_base is not base:
            pending.clear();return
        if generation!=app.global_preview_generation or app.global_preview_rgb is not frame:
            pending.clear();return
        changed=signature!=app._global_preview_state_signature()
        if changed:
            # Publish completed intermediate frames during a continuous drag,
            # then calculate the latest settings from the exact local scene.
            key,index=reference
            pending[reference]=context.strokes[key][index]
        else:pending.pop(reference,None)
        if value is None:
            pending.pop(reference,None)
            from runtime_log import append_runtime_log
            append_runtime_log('单颗流星预览失败',error or '局部素材不可用')
            app.status.set('局部预览失败，请查看运行日志')
        elif result.patch is not None:
            frame[result.slices]=result.patch
            app.last_incremental_box=context.last_incremental_box
            app._parameter_busy=True
            app._commit_incremental_global_preview(frame,validate=False,realtime=True,dirty_box=context.last_incremental_box)
            app._parameter_busy=False
            if changed:
                app.global_preview_signature=signature;app.exact_preview_signature=exact_signature
        start(app)
    app.after(16,poll)
