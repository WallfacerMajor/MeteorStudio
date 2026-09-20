"""Small Tk navigation helpers shared by the desktop workspaces."""

from __future__ import annotations
import tkinter as tk


def cancel_widget_timers(widget) -> None:
    """Cancel only timers registered by this widget before Tcl deletes them.

    A destroyed Toplevel otherwise leaves queued callbacks in the shared Tk
    interpreter; they fire later against deleted commands while its parent lives.
    """
    commands = set(getattr(widget, "_tclCommands", None) or ())
    for timer in widget.tk.splitlist(widget.tk.call("after", "info")):
        try:
            script, _kind = widget.tk.call("after", "info", timer)
            parts = widget.tk.splitlist(script)
            if parts and parts[0] in commands:
                widget.after_cancel(timer)
        except tk.TclError:
            pass


def keep_tree_row_in_navigation_runway(tree, iid: str, margin_rows: int = 3) -> None:
    """Keep a selected Treeview row away from the viewport edges.

    ``Treeview.see`` waits until a row is completely outside the viewport.  For
    rapid photo review that is too late: the active row hugs the last visible
    line and the next key press appears to lose the selection.  After making
    the row visible, move it a few rows back into the viewport when it enters
    the outer quarter.  This runs before any image decoding or rendering.
    """
    tree.see(iid)
    try:
        tree.update_idletasks()
        _x, y, _width, row_height = tree.bbox(iid)
        viewport_height = int(tree.winfo_height())
    except (AttributeError, TypeError, ValueError):
        return
    if row_height <= 0 or viewport_height <= row_height:
        return

    runway = max(row_height * max(1, int(margin_rows)), viewport_height // 4)
    if y < runway:
        tree.yview_scroll(-max(1, int(margin_rows)), "units")
    elif y + row_height > viewport_height - runway:
        tree.yview_scroll(max(1, int(margin_rows)), "units")
