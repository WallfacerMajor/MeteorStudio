"""Focused video-editor Tk response and frame-cache regression."""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from tkinter import ttk
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_meteor as video


def run():
    with tempfile.TemporaryDirectory() as tmp, patch.object(video.VideoMeteorWindow, '_restore_autosave'):
        root = tk.Tk()
        root.withdraw()
        win = video.VideoMeteorWindow(root)
        win.autosave_file = Path(tmp) / 'video-autosave.json'
        win._schedule_autosave = lambda: None
        def pump(seconds):
            end = time.monotonic()+seconds
            while time.monotonic() < end:
                root.update()
                time.sleep(.005)
        try:
            pump(.1)
            file = Path(tmp) / 'input.avi'
            file.write_bytes(b'video identity for mocked frame decoder')
            win.video_path.set(str(file))
            win.info = video.VideoInfo(96, 64, 24, 8, 8/24)
            win.current_event = video.VideoEvent(3, 90)
            win.preview_frame = np.full((64, 96, 3), 120, np.uint8)
            win.preview_mode.set('effect')
            reads = []
            strengths = []
            def read(_path, frame):
                reads.append((frame, threading.current_thread().name))
                time.sleep(.08)
                return np.full((64,96,3), 90, np.uint8)
            def residual(event, frame, background, settings, source_offset):
                strengths.append(settings.brightness)
                return video.ResidualLayer(event.frame, 20, 20, 40, 40,
                    np.full((20,20,3), settings.brightness*18, np.float32), settings)
            with patch.object(video, 'read_video_frame', read), patch.object(video, 'build_residual_layer', residual):
                beat = []
                root.after(20, lambda: beat.append(True))
                win._render_current()
                pump(.06)
                assert beat and len(reads) == 1, (beat,reads)
                pump(.28)
                assert len(reads) == 2 and win.preview_photo is not None, reads
                image_item = win._preview_image_item
                def descendants(widget):
                    for child in widget.winfo_children():
                        yield child
                        yield from descendants(child)
                scale = next(x for x in descendants(win) if isinstance(x, ttk.Scale)
                             and str(x.cget('variable')) == str(win.brightness))
                assert scale.winfo_ismapped()
                scale.event_generate('<ButtonPress-1>', x=scale.winfo_width()//2, y=scale.winfo_height()//2)
                durations = []
                for fraction in (.55,.7,.85):
                    start = time.monotonic()
                    scale.event_generate('<B1-Motion>', x=int(scale.winfo_width()*fraction),
                                         y=scale.winfo_height()//2, state=0x100)
                    durations.append(time.monotonic()-start)
                    pump(.04)
                scale.event_generate('<ButtonRelease-1>', x=int(scale.winfo_width()*.85), y=scale.winfo_height()//2)
                pump(.5)
                assert max(durations) < .2, durations
                assert len(reads) == 2, reads
                assert win._preview_image_item == image_item
                assert abs(strengths[-1]-win.brightness.get()) < .01, strengths
                assert all(name != threading.main_thread().name for _,name in reads), reads
                win.video_path.set(str(Path(tmp) / 'missing.avi'))
                win._render_current()
                pump(.12)
                assert '预览失败' in win.status.get(), win.status.get()
                return {'first_decode_frames':len(reads), 'drag_seconds':durations,
                        'latest_strength':strengths[-1], 'image_item_stable':True,
                        'missing_video_error_visible':True}
        finally:
            win.destroy()
            root.destroy()


if __name__=='__main__':print(json.dumps(run()))
