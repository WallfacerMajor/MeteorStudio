"""Real photo-open and live adjustment checks for the two color workspaces."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from white_balance_workspace import WhiteBalanceWindow
from light_pollution_workspace import LightPollutionWindow


def run():
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'gradient.png'
        y=np.arange(384,dtype=np.float32)[:,None,None]/383
        pixels=np.clip(np.array([24,28,35],np.float32)[None,None,:]+y*np.array([38,28,16],np.float32)[None,None,:],0,255)
        pixels=np.broadcast_to(pixels,(384,512,3)).astype(np.uint8)
        cv2.imwrite(str(source),pixels)
        root=tk.Tk();root.withdraw()
        def pump(seconds):
            end=time.monotonic()+seconds
            while time.monotonic()<end:root.update();time.sleep(.005)
        def until(test,seconds=6):
            end=time.monotonic()+seconds
            while not test():
                pump(.03)
                assert time.monotonic()<end,'preview timed out'
        def click(widget):
            widget.event_generate('<ButtonPress-1>',x=widget.winfo_width()//2,y=widget.winfo_height()//2)
            widget.event_generate('<ButtonRelease-1>',x=widget.winfo_width()//2,y=widget.winfo_height()//2)
        white=WhiteBalanceWindow(root)
        try:
            pump(.1)
            with patch('white_balance_workspace.filedialog.askopenfilename',return_value=str(source)):
                click(white.open_button)
            until(lambda:white.levels is not None and white.photo is not None)
            assert white.source==source
            first_photo=white.photo
            white.warmth.set(38)
            white.schedule_render()
            until(lambda:white.info.get().startswith('白平衡效果') and white.render_id>1)
            pump(.18)
            assert white.photo is first_photo,'white balance recreated the Tk image'
            white.destination.set('')
            with patch('white_balance_workspace.filedialog.askdirectory',return_value='') as choose:
                click(white.export_button)
                assert choose.call_count==1 and not white.busy
            white.destroy()
            light=LightPollutionWindow(root)
            try:
                pump(.1)
                with patch('light_pollution_workspace.filedialog.askopenfilename',return_value=str(source)):
                    click(light.open_button)
                until(lambda:light.levels is not None and light.model is not None and light.photo is not None,12)
                original_model=light.model
                light.start.set(18)
                light.invalidate()
                assert light.model_pending and str(light.export_button.cget('state'))=='disabled'
                until(lambda:not light.model_pending and light.model is not original_model,12)
                first_photo=light.photo
                light.strength.set(95)
                light.schedule_render()
                pump(.25)
                assert light.photo is first_photo,'pollution preview recreated the Tk image'
                light.destination.set('')
                with patch('light_pollution_workspace.filedialog.askdirectory',return_value='') as choose:
                    click(light.export_button)
                    assert choose.call_count==1 and not light.busy
                return {'white_balance_open_adjust_export':'passed',
                        'pollution_auto_model_and_export':'passed',
                        'preview_images_reused':True}
            finally:light.destroy()
        finally:
            if white.winfo_exists():white.destroy()
            root.destroy()


if __name__=='__main__':print(json.dumps(run()))
