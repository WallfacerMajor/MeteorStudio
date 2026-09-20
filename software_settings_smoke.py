"""Pointer-driven settings checks that never modify personal preferences."""
import tempfile
from pathlib import Path
from unittest.mock import patch
from tkinter import ttk


def exercise_settings(app):
    from toolbox import SoftwareRegistry, SoftwareSpec, SOFTWARE, require_software
    from toolbox_smoke import click, pump, widgets, capture
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        executable = folder / 'test-program'
        executable.touch()
        specs = tuple(SoftwareSpec(s.key, s.title, ()) for s in SOFTWARE)
        registry = lambda: SoftwareRegistry(folder/'software.json', specs)
        with patch('toolbox.SoftwareRegistry', side_effect=registry), patch('toolbox.filedialog.askopenfilename', return_value=str(executable)):
            button = next(w for w in widgets(app.toolbox_home) if isinstance(w, ttk.Menubutton) and w.cget('text') == '设置')
            assert not any(isinstance(w, ttk.LabelFrame) and '外部软件' in w.cget('text') for w in widgets(app.toolbox_home))
            def save_dialog():
                try:
                    dialog = app._software_settings_dialog
                    click(app, dialog.choose_buttons['siril'])
                    assert dialog.path_values['siril'].get() == str(executable)
                    capture(dialog, 'software-settings.png')
                    click(app, dialog.done_button)
                    checks.append('saved')
                except Exception as exc:
                    checks.append(exc)
                    if getattr(app, '_software_settings_dialog', None):
                        app._software_settings_dialog.destroy()
            menu = app.nametowidget(str(button.cget('menu')))
            assert menu.entrycget(0, 'label') == '外部软件与路径…'
            labels = [menu.entrycget(i, 'label') for i in range(menu.index('end')+1) if menu.type(i) != 'separator']
            assert labels == ['外部软件与路径…', '使用说明', '运行日志…'], labels
            help_menu = app.nametowidget(menu.entrycget(1, 'menu'))
            with patch('workspace_help.messagebox.showinfo') as show_help:
                help_menu.invoke(0)
                assert show_help.call_args.args[0] == '白平衡与改机校准'
            app.after(500, save_dialog)
            # Windows native menus run their own loop; invoke the registered
            # menu entry, then exercise the settings with real pointer events.
            menu.invoke(0)
            pump(app, 1.45)
            assert checks == ['saved'], checks
            assert registry().resolve('siril') == executable
            with patch('toolbox.show_software_settings') as dialog:
                assert require_software(app, ('siril',)) == {'siril': executable}
                dialog.assert_not_called()
            app.after(200, lambda: click(app, app._software_settings_dialog.close_button))
            assert require_software(app, ('ptgui',)) is None
            assert registry().resolve('ptgui') is None
            def complete_dialog():
                dialog = app._software_settings_dialog
                click(app, dialog.done_button)
                assert dialog.winfo_exists()
                click(app, dialog.choose_buttons['ptgui'])
                click(app, dialog.done_button)
            app.after(200, complete_dialog)
            assert require_software(app, ('siril', 'ptgui')) == {'siril': executable, 'ptgui': executable}
            pump(app, 1.45)
    return {'settings_menu_save': 'passed', 'settings_missing_cancel_continue': 'passed', 'configured_software_no_prompt': 'passed'}
