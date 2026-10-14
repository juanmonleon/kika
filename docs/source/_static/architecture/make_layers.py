"""Draw ``layers.svg`` and ``layers.pdf``: kika's layers and the direction of imports.

Run from anywhere; it writes both files next to itself. The page cites the
figure as ``layers.*``, so Sphinx takes the SVG for HTML and the PDF for LaTeX,
which cannot include an SVG. Only matplotlib is needed, which kika already
depends on. The content (which package sits in which layer) is the one stated,
and enforced, in ``docs/source/architecture/layers.md``: change the two together.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Top to bottom. Each row: (layer name, rule, packages).
ROWS = [
    ("Doors and consumers", "import any layer below",
     ["kika.read / write", "kika.sampling", "kika.njoy", "kika.tsl"]),
    ("Format layer", "import the layers below",
     ["kika.endf", "kika.gnds", "kika.ace", "kika.g4ndl", "kika.hdf5"]),
    ("Calculation layer", "import no format package",
     ["kika.nuclear_data.model", "kika.processing", "kika.cov"]),
    ("Leaves", "import nothing from kika",
     ["kika.algebra", "kika._records", "kika._legendre_frames"]),
]

FILL = ["#eef2f7", "#e6f0e6", "#f6efe3", "#f0e8f3"]
STROKE = "#4a5568"
TEXT = "#1a202c"

WIDTH, BOX_H, GAP, LABEL_W = 8.2, 0.74, 0.40, 2.2   # inches


def build():
    height = len(ROWS) * BOX_H + (len(ROWS) - 1) * GAP + 0.4
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.axis("off")

    top = height - 0.2
    inner_x = 0.2 + LABEL_W
    inner_w = WIDTH - inner_x - 0.2
    for i, (name, rule, packages) in enumerate(ROWS):
        y = top - BOX_H - i * (BOX_H + GAP)
        ax.add_patch(FancyBboxPatch((0.2, y), WIDTH - 0.4, BOX_H,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=FILL[i], edgecolor=STROKE, linewidth=1.0))
        ax.text(0.32, y + BOX_H - 0.26, name, fontsize=11, fontweight="bold",
                color=TEXT, va="center")
        ax.text(0.32, y + 0.22, rule, fontsize=8.5, color=TEXT, va="center")
        chip_w = (inner_w - 0.1 * (len(packages) + 1)) / len(packages)
        for j, package in enumerate(packages):
            cx = inner_x + 0.1 + j * (chip_w + 0.1)
            ax.add_patch(FancyBboxPatch((cx, y + 0.2), chip_w, BOX_H - 0.4,
                                        boxstyle="round,pad=0,rounding_size=0.04",
                                        facecolor="white", edgecolor=STROKE,
                                        linewidth=0.7))
            ax.text(cx + chip_w / 2, y + BOX_H / 2, package, fontsize=7.8,
                    family="monospace", color=TEXT, ha="center", va="center")
        if i < len(ROWS) - 1:
            ax.add_patch(FancyArrowPatch((WIDTH / 2, y - 0.04),
                                         (WIDTH / 2, y - GAP + 0.04),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color=STROKE, linewidth=1.3))
            ax.text(WIDTH / 2 + 0.1, y - GAP / 2, "imports", fontsize=8.5,
                    color=TEXT, va="center")
    return fig


if __name__ == "__main__":
    here = Path(__file__).parent
    fig = build()
    for suffix in ("svg", "pdf"):
        out = here / f"layers.{suffix}"
        fig.savefig(out, facecolor="white", metadata={"Date": None} if suffix == "svg"
                    else {"CreationDate": None})
        print(out)
