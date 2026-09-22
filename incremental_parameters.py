"""Serialize large-photo parameter previews without blocking Tk or copying a frame."""
from dataclasses import replace
from types import SimpleNamespace, MethodType
import threading
import numpy as np


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
    # The first edit cancels older whole-frame work. Doing so again while this
    # local worker runs would invalidate its generation and silently discard
    # the final result of a continuous slider drag.
    if not getattr(app,'_parameter_busy',False):
        app._cancel_deferred_full_preview_work()
    start_fast(app, before)
    if not getattr(app,'_parameter_busy',False):start(app)
    return True


def _fast_patch(context, reference, before, scale):
    """Recompose the affected layers at display resolution, without touching exact pixels."""
    import cv2
    from meteor_composer import (auto_optimized_stroke, transformed_object_crop,
        reset_stroke_geometry, stroke_for_image_crop, compose_meteor_sources,
        adjust_composite_base_exposure)
    base = context.preview_base
    height, width = base.shape[:2]
    key, index = reference
    if key not in context.pairs:
        return None
    auto = bool(context.adjustment_defaults.get('auto_optimize', True))
    def scaled(item, source_key):
        full_width = context.current_dims[0] if context.current_path is not None and source_key == str(context.current_path) else context._object_canvas_width(source_key)
        return auto_optimized_stroke(context._preview_clone(item, full_width), auto)
    old = transformed_object_crop(base, scaled(before, key), bounds_only=True)
    values = context.strokes.get(key, [])
    if old is None or index >= len(values):
        return None
    new = transformed_object_crop(base, scaled(values[index], key), bounds_only=True)
    boxes = [old[3]] + ([new[3]] if new else [])
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    records = []
    shared = context._uses_shared_base()
    for source_key, items in context.strokes.items():
        if source_key not in context.pairs or (not shared and source_key != key):
            continue
        for item in items:
            if not item.points:
                continue
            prepared = scaled(item, source_key)
            destination = transformed_object_crop(base, prepared, bounds_only=True)
            if destination is None:
                continue
            original = replace(prepared, points=prepared.points.copy())
            reset_stroke_geometry(original)
            source = transformed_object_crop(base, original, bounds_only=True)
            records.append((source_key, item, destination[3], source[3] if source else destination[3]))
    changed = True
    while changed:
        changed = False
        for _, _, box, source_box in records:
            if box[0] >= x1 or box[2] <= x0 or box[1] >= y1 or box[3] <= y0:
                continue
            expanded = min(x0, box[0], source_box[0]), min(y0, box[1], source_box[1]), max(x1, box[2], source_box[2]), max(y1, box[3], source_box[3])
            if expanded != (x0, y0, x1, y1):
                x0, y0, x1, y1 = expanded
                changed = True
    if x1 <= x0 or y1 <= y0:
        return None
    crop_w, crop_h = x1-x0, y1-y0
    small_w, small_h = max(1, round(crop_w*scale)), max(1, round(crop_h*scale))
    # Extremely large transformed groups are left to the exact worker.
    if small_w*small_h > 1_500_000:
        return None
    def shrink(image):
        source_h, source_w = image.shape[:2]
        if (source_h,source_w) == (height,width):
            return cv2.resize(image[y0:y1, x0:x1], (small_w, small_h), interpolation=cv2.INTER_AREA)
        # Original camera frames can sit centred inside a larger PTGui canvas.
        # Resample only the small requested rectangle, never allocate that canvas.
        fit = min(1.0,width/max(1,source_w),height/max(1,source_h))
        placed_w,placed_h = max(1,round(source_w*fit)),max(1,round(source_h*fit))
        offset_x,offset_y = (width-placed_w)//2,(height-placed_h)//2
        sx,sy = small_w/crop_w,small_h/crop_h
        matrix = np.asarray([[fit*sx,0,(offset_x-x0)*sx],
                             [0,fit*sy,(offset_y-y0)*sy]],dtype=np.float32)
        return cv2.warpAffine(image,matrix,(small_w,small_h),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    clean = shrink(base)
    result = clean.copy()
    for source_key in context.strokes:
        selected = [item for candidate_key, item, box, _ in records
                    if candidate_key == source_key and box[0] < x1 and box[2] > x0 and box[1] < y1 and box[3] > y0]
        if not any(not item.erase for item in selected):
            continue
        if context.current_path is not None and source_key == str(context.current_path):
            aligned = context.preview_aligned_source if context.preview_aligned_source is not None else context.preview_source
            original = context.preview_original_source if context.preview_original_source is not None else context.preview_source
        else:
            from pathlib import Path
            aligned_path = Path(source_key)
            original_path = context.original_sources.get(source_key, aligned_path)
            aligned = context._cached_layer_preview(aligned_path, width, height)
            original = context._cached_layer_preview(original_path, width, height)
        if aligned is None or original is None:
            return None
        if aligned.shape[:2] != (height,width) and (context.current_path is not None and source_key == str(context.current_path) or aligned is not original):
            return None
        prepared = []
        for item in selected:
            local = stroke_for_image_crop(scaled(item, source_key), width, height, x0, y0, crop_w, crop_h)
            local = replace(local, width=max(1, round(local.width*scale)), feather=max(0, round(local.feather*scale)),
                            offset_x=local.offset_x*scale, offset_y=local.offset_y*scale,
                            auto_feather=None if local.auto_feather is None else max(1, round(local.auto_feather*scale)))
            prepared.append(local)
        result, _ = compose_meteor_sources(shrink(aligned), shrink(original), result, prepared,
                                            *context._object_composite_settings(source_key), background=clean)
    exposure = context.base_exposure_tenths.get()/10.0
    if abs(exposure) > 1e-6:
        result = adjust_composite_base_exposure(result, clean, exposure)
    return result, (x0,y0,x1,y1)


def start_fast(app, before):
    if getattr(app, '_parameter_fast_busy', False):
        return
    if app.preview_base is None or app.preview_photo is None or app.canvas_zoom <= 0:
        return
    reference = app.selected_object
    if reference is None:
        return
    signature = app._global_preview_state_signature()
    generation = app.global_preview_generation
    scale = min(0.35, max(0.2, app.canvas_zoom))
    context = SimpleNamespace()
    for name in ('preview_base','current_path','current_dims','preview_aligned_source',
                 'preview_original_source','preview_source'):
        setattr(context, name, getattr(app,name))
    context.strokes = {k:[replace(s,points=s.points.copy()) for s in v] for k,v in app.strokes.items()}
    context.pairs = app.pairs.copy(); context.original_sources = app.original_sources.copy()
    context.adjustment_defaults = app.adjustment_defaults.copy()
    context.image_adjustments = {k:v.copy() for k,v in app.image_adjustments.items()}
    context.base_exposure_tenths = SimpleNamespace(get=lambda v=app.base_exposure_tenths.get():v)
    for name in ('blend_mode','output_mode'):
        value=getattr(app,name).get();setattr(context,name,SimpleNamespace(get=lambda v=value:v))
    for name in ('_cached_layer_preview','_uses_shared_base','_object_canvas_width','_preview_clone','_object_composite_settings'):
        setattr(context,name,MethodType(getattr(type(app),name),context) if name not in ('_cached_layer_preview',) else getattr(app,name))
    completed=[]
    app._parameter_fast_busy=True
    def work():
        try:completed.append(_fast_patch(context,reference,before,scale))
        except Exception:completed.append(None)
    threading.Thread(target=work,daemon=True,name='meteor-display-preview').start()
    def poll():
        if not completed:
            app.after(8,poll);return
        app._parameter_fast_busy=False
        if (completed[0] is not None and app.preview_base is context.preview_base
            and app.global_preview_generation == generation
            and app._global_preview_state_signature() == signature):
            patch, box = completed[0]
            app._paste_display_preview_patch(patch,box)
        if app.preview_base is context.preview_base and app._global_preview_state_signature()!=signature:
            current=app._selected_stroke()
            if current is not None:
                start_fast(app, before)
    app.after(8,poll)


def start(app):
    pending=app._parameter_pending
    if not pending:return
    reference,before=next(iter(pending.items()))
    base=app.preview_base;frame=app.global_preview_rgb
    if base is None or frame is None:pending.clear();return
    signature=app._global_preview_state_signature()
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
            # Never paint an older exact frame over a newer display preview.
            # Keep the original before-state because the committed full-size
            # buffer still contains it, then calculate only the latest value.
            start(app)
            return
        pending.pop(reference,None)
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
        start(app)
    app.after(16,poll)
