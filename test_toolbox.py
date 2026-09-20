import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from toolbox import SoftwareRegistry, SoftwareSpec, WORKSPACES, TOOL_MENU, ToolGroup, menu_level
from background_tasks import BackgroundTaskScheduler
from ptgui_pipeline import run_alignment_pipeline


class ToolboxTests(unittest.TestCase):
    def test_hierarchy_supports_nested_categories(self):
        nodes, ancestors = menu_level(("meteor",))
        self.assertEqual({node.key for node in nodes}, {"screening", "alignment", "composite", "video"})
        self.assertEqual(ancestors[0].title, "流星工具")
        nested = (ToolGroup("a", "A", "", (ToolGroup("b", "B", "", WORKSPACES),)),)
        self.assertEqual(menu_level(("a", "b"), nested)[0], WORKSPACES)
        with self.assertRaises(ValueError):
            menu_level(("missing",))

    def test_destroy_timer_cleanup_preserves_other_workspaces(self):
        import tkinter as tk
        from ui_navigation import cancel_widget_timers
        root = tk.Tk()
        child = tk.Toplevel(root)
        try:
            parent_timer = root.after(10000, lambda: None)
            child_timer = child.after(10000, lambda: None)
            cancel_widget_timers(child)
            timers = root.tk.splitlist(root.tk.call("after", "info"))
            self.assertIn(parent_timer, timers)
            self.assertNotIn(child_timer, timers)
        finally:
            cancel_widget_timers(root)
            root.destroy()

    def test_registry_extensible_and_preserves_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / "another-tool"
            executable.touch()
            specs = (SoftwareSpec("other", "Other", ()),)
            registry = SoftwareRegistry(root / "software.json", specs)
            registry.save("other", str(executable))
            self.assertEqual(SoftwareRegistry(registry.path, specs).resolve("other"), executable)
            executable.unlink()
            self.assertIsNone(SoftwareRegistry(registry.path, specs).resolve("other"))

    def test_registry_corrupt_file_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "software.json"
            path.write_text("[]", encoding="utf-8")
            self.assertEqual(SoftwareRegistry(path).paths, {})

    def test_workspace_actions_are_available(self):
        from meteor_composer import MeteorComposer
        self.assertEqual(len({item.key for item in WORKSPACES}), len(WORKSPACES))
        for item in WORKSPACES:
            self.assertTrue(callable(getattr(MeteorComposer, item.action)))

    def test_shutdown_with_queued_futures_does_not_mutate_iteration(self):
        scheduler = BackgroundTaskScheduler(max_workers=1)
        started, release = threading.Event(), threading.Event()
        def wait(token):
            started.set()
            release.wait(2)
        scheduler.submit("running", wait)
        self.assertTrue(started.wait(1))
        tokens = [scheduler.submit(f"pending:{i}", lambda token: None) for i in range(20)]
        try:
            scheduler.shutdown(wait=False)
            self.assertTrue(all(token.cancelled for token in tokens))
        finally:
            release.set()

    def test_duplicate_stems_fail_before_writing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ("base.tif", "frame.tif", "frame.jpg")]
            for path in paths:
                path.touch()
            with self.assertRaisesRegex(ValueError, "同名素材"):
                run_alignment_pipeline(paths[0], paths[1:], root / "output", root, root)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
