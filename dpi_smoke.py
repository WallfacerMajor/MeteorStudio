"""Native-pixel and pointer regression; simulated scales never change Windows settings."""
import ctypes
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import ImageGrab


def workspace_layout_checks(app):
    """Exercise DPI-dependent navigation/targets without rerunning batch algorithms."""
    from unittest.mock import patch
    from contextlib import ExitStack
    from toolbox_smoke import click, pump, widgets, capture
    from meteor_screening import MeteorScreeningWindow
    from video_meteor import VideoMeteorWindow
    attributes = {'screening': 'screening_window', 'alignment': 'alignment_window',
                  'control_points': 'alignment_window', 'video': 'video_window',
                  'white_balance': 'white_balance_window', 'light_pollution': 'light_pollution_window',
                  'trails': 'laboratory_window', 'mean': 'laboratory_window', 'quality': 'laboratory_window'}
    with ExitStack() as stack:
        for cls, method in ((MeteorScreeningWindow, '_restore_autosave'), (MeteorScreeningWindow, '_save_autosave'),
                            (VideoMeteorWindow, '_restore_autosave'), (VideoMeteorWindow, '_write_autosave')):
            stack.enter_context(patch.object(cls, method))
        app.show_toolbox()
        pump(app, .3)
        for key in tuple(app.toolbox_home.tool_buttons):
            click(app, app.toolbox_home.tool_buttons[key])
            window = app if key == 'composite' else getattr(app, attributes[key])
            pump(app, .4)
            assert window.winfo_ismapped() and window.state() == 'normal'
            if key in ('trails', 'mean', 'quality'):
                with patch('laboratory_workspace.filedialog.askopenfilenames', return_value=()) as picker:
                    click(app, window.add_button)
                    picker.assert_called_once()
            elif key in ('white_balance', 'light_pollution'):
                module = 'white_balance_workspace' if key == 'white_balance' else 'light_pollution_workspace'
                with patch(module+'.filedialog.askopenfilename', return_value='') as picker:
                    click(app, window.open_button)
                    picker.assert_called_once()
            back = next(w for w in widgets(window) if isinstance(w, ttk.Button) and w.winfo_ismapped()
                        and w.cget('text') in ('← 工具箱', '← 流星工具'))
            capture(window, f'dpi-workspace-{key}.png')
            click(app, back)
            pump(app, .2)
            assert app.toolbox_home.winfo_ismapped()
    return {'all_workspace_dpi_navigation': 'passed', 'scaled_file_picker_targets': 'passed'}


def run_smoke(app):
    from action_icons import iconize_actions
    from dpi_support import display_scale, scale_layout
    from toolbox_smoke import click, pump
    from ui_theme import apply_theme
    from workspace_layout import parameter_slider

    user32 = ctypes.windll.user32
    get_context = user32.GetWindowDpiAwarenessContext
    get_context.argtypes = [ctypes.c_void_p]
    get_context.restype = ctypes.c_void_p
    get_awareness = user32.GetAwarenessFromDpiAwarenessContext
    get_awareness.argtypes = [ctypes.c_void_p]
    get_awareness.restype = ctypes.c_int
    app.update()
    assert get_awareness(get_context(app.winfo_id())) == 2
    assert ImageGrab.grab().width == app.winfo_screenwidth()
    assert app.state() == 'normal'
    assert app.edit_inspector.cget('width') == round(360 * display_scale(app))
    reports = workspace_layout_checks(app)
    app.withdraw()
    folder = Path(os.environ.get('NIGHTSCAPE_SCREENSHOTS', 'review-artifacts/dpi-scales'))
    folder.mkdir(parents=True, exist_ok=True)
    reports.update(native_dpi_awareness='passed', physical_screen_coordinates='passed')
    for factor in (1.0, 1.5, 2.0):
        # Separate Tk interpreters keep fonts and theme images independent.
        root = tk.Tk()
        root.tk.call('tk', 'scaling', factor * 96 / 72)
        root.geometry('720x420+30+30')
        root.minsize(600, 350)
        apply_theme(root)
        try:
            body = ttk.Frame(root, padding=12)
            body.pack(fill='both', expand=True)
            inspector = ttk.Frame(body, width=260)
            inspector.pack(side='right', fill='y')
            inspector.pack_propagate(False)
            ttk.Label(inspector, text='白平衡与改机校准', style='Section.TLabel').pack()
            calls = []
            button = ttk.Button(inspector, text='保存项目', command=lambda: calls.append('save'))
            button.pack(pady=12)
            value = tk.DoubleVar(root, 0)
            scale = parameter_slider(inspector, '色温', value, -100, 100, colors=('#4c88e8', '#c6c6c6', '#edb94f'))
            canvas = tk.Canvas(body, highlightthickness=0, background='#181818')
            canvas.pack(fill='both', expand=True)
            photo = tk.PhotoImage(master=root, width=120, height=80)
            photo.put('{'+' '.join('#ffffff' if x % 2 else '#000000' for x in range(120))+'}', to=(0, 0, 120, 80))
            canvas.create_image(10, 10, image=photo, anchor='nw')
            iconize_actions(root)
            root.attributes('-topmost', True)
            root.lift()
            pump(root, .3)
            assert inspector.winfo_width() == round(260 * factor)
            image_name = root.tk.splitlist(button.cget('image'))[0]
            assert int(root.tk.call('image', 'width', image_name)) == round(22 * factor)
            assert scale._gradient_image.height() == round(14 * factor)
            before = (root.geometry(), inspector.winfo_width(), canvas.winfo_width(), canvas.winfo_height())
            scale_layout(root)
            pump(root, .1)
            assert before == (root.geometry(), inspector.winfo_width(), canvas.winfo_width(), canvas.winfo_height())
            click(root, button)
            assert calls == ['save'] and button._action_hint.popup
            button._action_hint.hide()
            x, y = canvas.winfo_rootx()+10, canvas.winfo_rooty()+10
            # Pixel-exact alternating lines would become grey/interpolated if
            # Windows still stretched the rendered application by 150%.
            pump(root, .1)
            sample = ImageGrab.grab(bbox=(x, y, x+120, y+80))
            assert [sample.getpixel((x, 40))[:3] for x in range(120)] == [(255, 255, 255) if x % 2 else (0, 0, 0) for x in range(120)]
            left, top = root.winfo_rootx(), root.winfo_rooty()
            ImageGrab.grab(bbox=(left, top, left+root.winfo_width(), top+root.winfo_height())).save(folder / f'dpi-{round(factor*100)}.png')
            button.event_generate('<ButtonPress-1>', x=button.winfo_width()//2, y=button.winfo_height()//2)
            button.event_generate('<ButtonRelease-1>', x=button.winfo_width()//2, y=button.winfo_height()//2)
            pump(root, 1.45)
            assert calls == ['save', 'save']
            assert before == (root.geometry(), inspector.winfo_width(), canvas.winfo_width(), canvas.winfo_height())
            reports[f'layout_icons_pointer_pixels_{round(factor*100)}'] = 'passed'
        finally:
            root.destroy()
    return reports


if __name__ == '__main__':
    import json
    from unittest.mock import patch
    from meteor_composer import MeteorComposer
    with patch.object(MeteorComposer, '_restore_autosave'):
        app = MeteorComposer()
    try:
        print(json.dumps(run_smoke(app)))
    finally:
        app.destroy()
