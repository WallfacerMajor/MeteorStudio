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
from engine import Comparison, read_rgb, write_rgb, reduce_siril_preview


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
    assert app.mode.get()=='仅 Siril'
    with tempfile.TemporaryDirectory(prefix='star-compare-smoke-') as folder:
        folder=Path(folder)
        demo=root/'runs/galaxy-20260922'
        original=tifffile.memmap(demo/'input.tif')[900:1220,2700:3080,:3].copy()
        background=tifffile.memmap(demo/'starless.tif')[900:1220,2700:3080,:3].copy()
        source=folder/'source.tif';tifffile.imwrite(source,original,photometric='rgb')
        with patch('app.filedialog.askopenfilename',return_value=str(source)) as picker, patch('app.filedialog.askdirectory') as output_picker:
            click(app.empty_button if app.toolbox_mode else app.open_button)
            deadline=time.monotonic()+15
            while app.running:
                pump(.05);assert time.monotonic()<deadline
            assert picker.call_count==1 and output_picker.call_count==0
            assert app.source==source and '原图' in app.images
        if app.toolbox_mode:
            assert app.session is None
            assert app.home_button.winfo_ismapped()
        old_image=app.images['原图']
        with patch('app.filedialog.askopenfilename',return_value=''):
            click(app.open_button)
        assert app.images['原图'] is old_image
        invalid=folder/'broken.tif';invalid.write_bytes(b'not a TIFF')
        with patch('app.filedialog.askopenfilename',return_value=str(invalid)):
            click(app.open_button)
            deadline=time.monotonic()+15
            while app.running:
                pump(.05);assert time.monotonic()<deadline
        assert app.images['原图'] is old_image and app.source==source
        panel=app._runtime_log_panel
        assert panel.winfo_ismapped() and panel.text.get('1.0','end').strip()
        before=(app.center,app.zoom,app.canvas.winfo_width(),app.canvas.winfo_height())
        click(panel.hide_button);assert not panel.winfo_ismapped()
        click(app.global_log_button);assert panel.winfo_ismapped()
        app.focus_force();app.event_generate('<Control-l>');pump(.2)
        assert not panel.winfo_ismapped()
        assert before==(app.center,app.zoom,app.canvas.winfo_width(),app.canvas.winfo_height())
        app.error_details=''
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
        choose(app.mode_combo,'同时计算')
        with patch('app.filedialog.askdirectory') as output_picker:
            click(app.run_button);wait();assert output_picker.call_count==0
        assert app.live_pair is not None and not (app.session/'siril-result.tif').exists()
        with patch('app.filedialog.askdirectory',return_value='') as output_picker:
            click(app.export_button)
            assert output_picker.call_count==1 and app.export_directory is None and not app.running
        with patch('app.filedialog.askdirectory',return_value=str(folder)):
            click(app.export_button);wait()
        first_export=app.export_directory
        local=read_rgb(first_export/'local-result.tif')[0];siril=read_rgb(first_export/'siril-result.tif')[0]
        assert local.shape==original.shape and siril.shape==original.shape
        assert json.loads((first_export/'comparison.json').read_text(encoding='utf-8'))['amount']==app.amount.get()/100
        assert np.any(np.abs(local-siril)>1/65535)
        assert tifffile.imread(app.session/'siril-result.tif').dtype==np.uint16
        calculated=reduce_siril_preview(read_rgb(source)[0],read_rgb(app.session/'starless.tif')[0],app.amount.get()/100)
        np.testing.assert_allclose(calculated,siril,atol=4/65535)
        deadline=time.monotonic()+10
        while app.live_key!=app.preview_key():pump(.05);assert time.monotonic()<deadline
        before=(app.center,app.zoom)
        saved_hash=hashlib.sha256((app.session/'siril-result.tif').read_bytes()).hexdigest()
        with patch.object(Comparison,'run_siril',side_effect=AssertionError('slider restarted Siril')):
            t=time.monotonic();old_tile=np.asarray(app.live_tiles['Siril'][0]).copy()
            coords=app.scale.coords()
            app.scale.event_generate('<ButtonPress-1>',x=int(coords[0]),y=int(coords[1]))
            app.scale.event_generate('<B1-Motion>',x=app.scale.winfo_width()*.9,y=int(coords[1]),state=0x100)
            # Keep the pointer held: debounce-until-release is not a live preview.
            deadline=time.monotonic()+5
            while app.live_key!=app.preview_key():pump(.03);assert time.monotonic()<deadline
            assert np.any(old_tile!=np.asarray(app.live_tiles['Siril'][0]))
            app.scale.event_generate('<ButtonRelease-1>',x=app.scale.winfo_width()*.9,y=int(coords[1]))
            pump(1.4)
            assert app.live_key==app.preview_key() and before==(app.center,app.zoom)
        assert hashlib.sha256((app.session/'siril-result.tif').read_bytes()).hexdigest()==saved_hash
        old_center=app.center;click(app.actual_button);assert app.zoom==1
        app.canvas.event_generate('<ButtonPress-1>',x=200,y=200)
        app.canvas.event_generate('<B1-Motion>',x=230,y=220,state=0x100)
        app.canvas.event_generate('<ButtonRelease-1>',x=230,y=220)
        pump();assert app.center!=old_center
        before=(app.center,app.zoom)
        choose(app.view_combo,'Siril');assert (app.center,app.zoom)==before
        choose(app.view_combo,'三图对比');assert (app.center,app.zoom)==before
        local_hash=hashlib.sha256((first_export/'local-result.tif').read_bytes()).hexdigest()
        choose(app.mode_combo,'仅 Siril')
        with patch('app.filedialog.askdirectory',return_value=str(folder)):
            click(app.export_button);wait()
        assert app.export_directory!=first_export
        assert not (app.export_directory/'local-result.tif').exists()
        exported=read_rgb(app.export_directory/'siril-result.tif')[0]
        np.testing.assert_allclose(exported,reduce_siril_preview(read_rgb(source)[0],read_rgb(app.session/'starless.tif')[0],app.amount.get()/100),atol=4/65535)
        assert hashlib.sha256((first_export/'local-result.tif').read_bytes()).hexdigest()==local_hash
        assert 'pm' in (app.session/'siril-reduce.log').read_text(encoding='utf-8')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
        assert hashlib.sha256(engine.config.read_bytes()).hexdigest()==config_hash
        engine.run_siril('identity','load original.fits\npm "~((~mtf(~0.5,$original$)/~mtf(~0.5,$starless$))*~$starless$)"\nsavetif identity')
        identity=tifffile.imread(app.session/'identity.tif')
        assert np.abs(identity.astype(np.int32)-original.astype(np.int32)).max()<=1
    app.load_session(root/'runs/galaxy-20260922');pump(.3)
    deadline=time.monotonic()+45
    while app.live_pair is None or app.live_key!=app.preview_key():
        pump(.05);assert time.monotonic()<deadline
    click(app.fit_button)
    deadline=time.monotonic()+10
    while app.live_key!=app.preview_key():pump(.03);assert time.monotonic()<deadline
    coords=app.scale.coords();t=time.monotonic()
    app.scale.event_generate('<ButtonPress-1>',x=int(coords[0]),y=int(coords[1]))
    app.scale.event_generate('<B1-Motion>',x=app.scale.winfo_width()*.7,y=int(coords[1]),state=0x100)
    deadline=time.monotonic()+5
    while app.live_key!=app.preview_key():pump(.02);assert time.monotonic()<deadline
    from runtime_log import append_runtime_log
    append_runtime_log(f'全幅缩星实时预览耗时 {time.monotonic()-t:.3f}s')
    app.scale.event_generate('<ButtonRelease-1>',x=app.scale.winfo_width()*.7,y=int(coords[1]))
    click(app.actual_button);app.center=(3000,1150);app.draw();pump(.2)
    ImageGrab.grab((app.winfo_rootx(),app.winfo_rooty(),app.winfo_rootx()+app.winfo_width(),app.winfo_rooty()+app.winfo_height())).save(root/'ui-verified.png')
    if app.toolbox_mode:
        click(app.home_button)
    return dict.fromkeys(('explicit_export_current_strength_original_size','prepare_without_output_dialog','full_photo_live_preview','default_siril','live_slider_without_release_or_siril_restart','siril_formula_matches_native','global_log_toggle_preserves_viewport','real_file_open','single_photo_dialog','output_cancel_keeps_photo','decode_error_preserves_photo_and_shows_log','real_slider','real_siril_pixelmath','both_outputs_differ',
        '16bit_outputs','synchronized_zoom_pan','view_switch_preserves_viewport','siril_only_preserves_local',
        'zero_strength_identity','input_read_only','siril_global_config_unchanged'),'passed')
