import unittest
import tkinter as tk
from tkinter import ttk

from ui_navigation import keep_tree_row_in_navigation_runway


class FakeTree:
    def __init__(self, y, row_height=20, viewport_height=200):
        self.y = y
        self.row_height = row_height
        self.viewport_height = viewport_height
        self.calls = []

    def see(self, iid):
        self.calls.append(("see", iid))

    def update_idletasks(self):
        self.calls.append(("idle",))

    def bbox(self, _iid):
        return 0, self.y, 100, self.row_height

    def winfo_height(self):
        return self.viewport_height

    def yview_scroll(self, amount, units):
        self.calls.append(("scroll", amount, units))


class TreeNavigationTests(unittest.TestCase):
    def test_bottom_runway_scrolls_before_row_leaves_view(self):
        tree = FakeTree(y=145)
        keep_tree_row_in_navigation_runway(tree, "12")
        self.assertEqual(tree.calls[0], ("see", "12"))
        self.assertIn(("scroll", 3, "units"), tree.calls)

    def test_top_runway_scrolls_back_for_reverse_navigation(self):
        tree = FakeTree(y=30)
        keep_tree_row_in_navigation_runway(tree, "4")
        self.assertIn(("scroll", -3, "units"), tree.calls)

    def test_middle_row_does_not_move_list(self):
        tree = FakeTree(y=90)
        keep_tree_row_in_navigation_runway(tree, "8")
        self.assertFalse(any(call[0] == "scroll" for call in tree.calls))

    def test_real_tk_click_near_bottom_creates_navigation_runway(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        try:
            root.geometry("320x260")
            tree = ttk.Treeview(root, height=8, selectmode="browse")
            tree.pack(fill="both", expand=True)
            for index in range(40):
                tree.insert("", "end", iid=str(index), text=f"frame {index}")
            tree.bind(
                "<<TreeviewSelect>>",
                lambda _event: keep_tree_row_in_navigation_runway(
                    tree, tree.selection()[0]
                ) if tree.selection() else None,
            )
            root.update()
            visible = [iid for iid in tree.get_children() if tree.bbox(iid)]
            target = visible[-2]
            x, y, width, height = tree.bbox(target)
            tree.event_generate("<ButtonPress-1>", x=x + width // 2, y=y + height // 2)
            tree.event_generate("<ButtonRelease-1>", x=x + width // 2, y=y + height // 2)
            root.update()
            self.assertEqual(tree.selection(), (target,))
            self.assertGreater(tree.yview()[0], 0.0)
            self.assertTrue(tree.bbox(target))
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
