"""Actual Tk pointer events and real Siril execution, using isolated small files."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import numpy as np
import tifffile
from PIL import ImageGrab
from engine import Comparison, read_rgb, write_rgb


def run(app, root):
    def pump(seconds=.2):
        end=time.monotonic()+seconds
        while time.monotonic()<end:app.update();time.sleep(.015)
    def click(widget):
        app.attributes('-topmost',True);app.lift();app.focus_force();pump(.3)
        x,y=widget.winfo_width()//2,widget.winfo_height()//2
        assert app.winfo_containing(widget.winfo_rootx()+x,widget.winfo_rooty()+y)==widget, (str(widget),str(app.winfo_containing(widget.winfo_rootx()+x,widget.winfo_rooty()+y)),widget.winfo_rootx(),widget.winfo_rooty())
        widget.event_generate('<ButtonPress-1>',x=x,y=y)
        widget.event_generate('<ButtonRelease-1>',x=x,y=y);pump()
    def choose(combo,label):
        click(combo)
        pop=app.tk.call('ttk::combobox::PopdownWindow',str(combo));listbox=str(pop)+'.f.l'
        index=list(combo.cget('values')).index(label)
        box=app.tk.call(listbox,'bbox',index)
        for event in ('<Motion>','<ButtonPress-1>','<ButtonRelease-1>'):
            app.tk.call('event','generate',listbox,event,'-x',10,'-y',int(box[1])+int(box[3])//2)
        pump()
    def wait():
        end=time.monotonic()+90
        while app.running:
            pump(.1)
            assert time.monotonic()<end,app.status.get()
        assert not (app.session/'error.log').exists(),(app.session/'error.log').read_text(encoding='utf-8')
    app.geometry('1300x850+0+0');pump(.4)
    with tempfile.TemporaryDirectory(prefix='star-compare-smoke-') as folder:
        folder=Path(folder)
        demo=root/'runs/galaxy-20260922'
        original=tifffile.memmap(demo/'input.tif')[900:1220,2700:3080,:3].copy()
        background=tifffile.memmap(demo/'starless.tif')[900:1220,2700:3080,:3].copy()
        source=folder/'source.tif';tifffile.imwrite(source,original,photometric='rgb')
        with patch('app.filedialog.askopenfilename',return_value=str(source)), patch('app.filedialog.askdirectory',return_value=str(folder)):
            click(app.empty_button if app.toolbox_mode else app.open_button)
        if app.toolbox_mode:
            assert app.session.parent == folder and app.session != source.parent
            assert app.home_button.winfo_ismapped()
        app.session=folder/'output';app.session.mkdir()
        tifffile.imwrite(app.session/'starless.tif',background,photometric='rgb')
        tifffile.imwrite(app.session/'input.tif',original,photometric='rgb')
        engine=Comparison(source,app.session)
        engine.run_siril('fixture','load input.tif\nsave original\nload starless.tif\nsave starless')
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        config_hash=hashlib.sha256(engine.config.read_bytes()).hexdigest()
        # Real slider thumb drag.
        coords=app.scale.coords()
        app.scale.event_generate('<ButtonPress-1>',x=int(coords[0]),y=int(coords[1]))
        app.scale.event_generate('<B1-Motion>',x=app.scale.winfo_width()*2//3,y=int(coords[1]),state=0x100)
        app.scale.event_generate('<ButtonRelease-1>',x=app.scale.winfo_width()*2//3,y=int(coords[1]))
        pump();assert app.amount.get()>55
        click(app.run_button);wait()
        assert set(app.images)=={'原图','本地方案','Siril'}
        local=read_rgb(app.session/'local-result.tif')[0];siril=read_rgb(app.session/'siril-result.tif')[0]
        assert np.any(np.abs(local-siril)>1/65535)
        assert tifffile.imread(app.session/'siril-result.tif').dtype==np.uint16
        old_center=app.center;click(app.actual_button);assert app.zoom==1
        app.canvas.event_generate('<ButtonPress-1>',x=200,y=200)
        app.canvas.event_generate('<B1-Motion>',x=230,y=220,state=0x100)
        app.canvas.event_generate('<ButtonRelease-1>',x=230,y=220)
        pump();assert app.center!=old_center
        before=(app.center,app.zoom)
        choose(app.view_combo,'Siril');assert (app.center,app.zoom)==before
        choose(app.view_combo,'三图对比');assert (app.center,app.zoom)==before
        local_hash=hashlib.sha256((app.session/'local-result.tif').read_bytes()).hexdigest()
        choose(app.mode_combo,'仅 Siril');click(app.run_button);wait()
        assert hashlib.sha256((app.session/'local-result.tif').read_bytes()).hexdigest()==local_hash
        assert 'pm' in (app.session/'siril-reduce.log').read_text(encoding='utf-8')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
        assert hashlib.sha256(engine.config.read_bytes()).hexdigest()==config_hash
        engine.run_siril('identity','load original.fits\npm "~((~mtf(~0.5,$original$)/~mtf(~0.5,$starless$))*~$starless$)"\nsavetif identity')
        identity=tifffile.imread(app.session/'identity.tif')
        assert np.abs(identity.astype(np.int32)-original.astype(np.int32)).max()<=1
    app.load_session(root/'runs/galaxy-20260922');pump(.3)
    click(app.actual_button);app.center=(3000,1150);app.draw();pump(.2)
    ImageGrab.grab((app.winfo_rootx(),app.winfo_rooty(),app.winfo_rootx()+app.winfo_width(),app.winfo_rooty()+app.winfo_height())).save(root/'ui-verified.png')
    if app.toolbox_mode:
        click(app.home_button)
    return dict.fromkeys(('real_file_open','real_slider','real_siril_pixelmath','both_outputs_differ',
        '16bit_outputs','synchronized_zoom_pan','view_switch_preserves_viewport','siril_only_preserves_local',
        'zero_strength_identity','input_read_only','siril_global_config_unchanged'),'passed')
