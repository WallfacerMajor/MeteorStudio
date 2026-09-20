"""Read-only real-material regression; paths come from a supplied project."""
import json
import time
from pathlib import Path


def run_case(app, project_path, stem):
    from meteor_composer import detection_preview
    from background_tasks import CancellationToken

    project = json.loads(Path(project_path).read_text(encoding='utf-8'))
    source_path = next(Path(key) for key in project['candidates'] if Path(key).stem == stem)
    base_path = Path(project['base_dir'])
    key = str(source_path)
    app.geometry('1280x820+10000+10000')
    for timer in app.tk.call('after', 'info'):
        app.after_cancel(timer)
    app.autosave_suspended = True
    app._schedule_autosave = lambda: None
    source = app._cached_full_image(source_path, False)
    base = app._cached_full_image(base_path, False)
    app.current_path = source_path
    app.files = [source_path]
    app.pairs = {key: base_path}
    app.strokes = {key: []}
    app.candidates = {key: []}
    app.original_sources = {}
    app.use_original_sources = set()
    app.preview_source, _ = detection_preview(source)
    app.preview_base, _ = detection_preview(base)
    app.preview_rgb = app.preview_source
    app.current_dims = (source.shape[1], source.shape[0])
    app.output_mode.set('separate')
    app.view_mode.set('source')
    app.candidate_threshold.set(55)
    app.candidate_thresholds = {key: 55}
    app._set_paths_panel_visible(False)
    app.control_notebook.select(app.mask_tools_tab)
    app._render_preview()
    app._poll_queue()
    app.update()
    button = app.detect_current_button
    button.event_generate('<Enter>')
    button.event_generate('<ButtonPress-1>', x=button.winfo_width() // 2, y=button.winfo_height() // 2)
    button.event_generate('<ButtonRelease-1>', x=button.winfo_width() // 2, y=button.winfo_height() // 2)
    deadline = time.monotonic() + 45
    while not app.status.get().startswith('单张候选分析完成'):
        if time.monotonic() > deadline:
            raise AssertionError('Real button analysis timed out: ' + app.status.get())
        app.update()
        time.sleep(0.01)
    # DSC06021: independently inspected meteor at x~11%, y~78–86%.
    def is_meteor(stroke):
        x = sum(point[0] for point in stroke.points) / len(stroke.points)
        y = sum(point[1] for point in stroke.points) / len(stroke.points)
        return 0.09 < x < 0.14 and 0.76 < y < 0.89
    candidates = app.candidates[key]
    assert any(is_meteor(stroke) for stroke in candidates), candidates
    assert len(app.strokes[key]) == 1 and is_meteor(app.strokes[key][0]), 'Only the meteor should be selected at 55'
    deadline = time.monotonic() + 1.4
    while time.monotonic() < deadline:
        app.update()
        time.sleep(0.01)
    assert len(app.strokes[key]) == 1 and is_meteor(app.strokes[key][0])
    batch = app._auto_detect_worker(
        CancellationToken('case', 1), [source_path], {key: base_path},
        {key: source_path}, {key: 'aligned'},
    )
    assert len(batch[1][key]) == 1 and is_meteor(batch[1][key][0])
    return {'case': stem, 'real_single_button': 'passed', 'batch_worker': 'passed',
            'candidate_count': len(candidates), 'score': candidates[0].auto_score}


if __name__ == '__main__':
    import sys
    from meteor_composer import MeteorComposer
    app = MeteorComposer()
    try:
        print(json.dumps(run_case(app, sys.argv[1], 'DSC06021')))
    finally:
        app.destroy()
