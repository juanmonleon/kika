"""Draw into memory, never into a window.

These tests build figures and several of them force a `canvas.draw()`. Without
a backend pinned, matplotlib picks an interactive one — `tkagg` on a desktop
Windows box — and the suite then depends on a working Tcl install it has no
business needing. On a machine where that install is broken the failure lands
on whichever test happens to draw first, so it moves between runs and looks
like a flake in the code under test rather than in the harness.

Agg is what CI uses and what every assertion here actually needs.
"""
import matplotlib

matplotlib.use("Agg", force=True)
