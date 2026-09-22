"""Real project-open button with a compressed 16-bit TIFF and legacy blend mode."""
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from meteor_composer import MeteorComposer, read_image


def run():
    with tempfile.TemporaryDirectory() as tmp, patch.object(MeteorComposer, '_restore_autosave'), patch.object(MeteorComposer, '_setup_autosave'):
        root = Path(tmp)
        sources, bases = root / 'sources', root / 'bases'
        sources.mkdir()
        bases.mkdir()
        bgr = np.zeros((128, 192, 3), np.uint16)
        bgr[:] = (1000, 2000, 3000)
        source = sources / 'one.tif'
        assert cv2.imwrite(str(source), bgr, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
        assert cv2.imwrite(str(bases / 'one.jpg'), np.full((128, 192, 3), 45, np.uint8))
        pixels = read_image(source)
        assert pixels.dtype == np.uint16 and pixels[0, 0].tolist() == [3000, 2000, 1000]
        app = MeteorComposer()
        app.autosave_path = root / 'autosave.json'
        app._schedule_autosave = lambda: None
        try:
            app.geometry('1300x850+0+0')
            app.show_composite_workspace()
            app.update()
            invalid = root / 'unrelated.json'
            invalid.write_text('{"unrelated": true}', encoding='utf-8')
            button = app.load_project_button
            def click():
                x, y = button.winfo_width() // 2, button.winfo_height() // 2
                button.event_generate('<ButtonPress-1>', x=x, y=y)
                button.event_generate('<ButtonRelease-1>', x=x, y=y)
                app.update()
            with patch('meteor_composer.filedialog.askopenfilename', return_value=str(invalid)), patch('meteor_composer.show_copyable_error') as error:
                click()
                assert error.call_count == 1
                assert app.blend_mode.get() == '线性减淡（添加）'
            data = app._project_data()
            data.update(source_dir=str(sources), base_dir=str(bases / 'one.jpg'),
                        output_dir=str(root / 'out'), output_mode='combined',
                        blend_mode='自然融合')
            project = root / 'old-project.json'
            project.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            with patch('meteor_composer.filedialog.askopenfilename', return_value=str(project)) as chooser, patch('meteor_composer.show_copyable_error') as error:
                click()
                deadline = time.monotonic() + 6
                while app.preview_source is None and time.monotonic() < deadline:
                    app.update()
                    time.sleep(.01)
                app.update()
                assert chooser.call_count == 1 and error.call_count == 0
                assert app.blend_mode.get() == '自然融合'
                assert len(app.pairs) == 1 and app.preview_source is not None
            return {'project_button': 'passed', 'legacy_blend_restored': True,
                    'compressed_16bit_photo_loaded': True, 'invalid_json_kept_session': True}
        finally:
            app.destroy()


if __name__ == '__main__':
    print(json.dumps(run()))
