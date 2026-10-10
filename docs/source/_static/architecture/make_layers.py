"""Draw ``layers.svg``: kika's layers and the direction of their imports.

Run from anywhere; it writes the SVG next to itself. No dependency beyond the
standard library, so the figure can be regenerated wherever the docs build.
The content (which package sits in which layer) is the one stated, and
enforced, in ``docs/source/architecture/layers.md``: change the two together.
"""
from pathlib import Path
from xml.sax.saxutils import escape

WIDTH = 760
BOX_H = 74
GAP = 40
LEFT = 20
LABEL_W = 185

# Top to bottom. Each row: (layer name, rule, packages).
ROWS = [
    ("Doors and consumers",
     "import any layer below",
     ["kika.read / kika.write", "kika.sampling", "kika.njoy"]),
    ("Format layer",
     "import the layers below",
     ["kika.endf", "kika.gnds", "kika.ace", "kika.g4ndl"]),
    ("Calculation layer",
     "import no format package",
     ["kika.nuclear_data.model", "kika.processing", "kika.cov"]),
    ("Leaves",
     "import nothing from kika",
     ["kika.algebra", "kika._records", "kika._legendre_frames"]),
]

FILL = ["#eef2f7", "#e6f0e6", "#f6efe3", "#f0e8f3"]
STROKE = "#4a5568"
TEXT = "#1a202c"


def _box(x, y, w, h, fill):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
            f'fill="{fill}" stroke="{STROKE}" stroke-width="1.2"/>')


def _text(x, y, s, size=13, weight="normal", anchor="start", family="sans-serif"):
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{TEXT}">'
            f'{escape(s)}</text>')


def build() -> str:
    height = len(ROWS) * BOX_H + (len(ROWS) - 1) * GAP + 40
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
             f'height="{height}" viewBox="0 0 {WIDTH} {height}">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{STROKE}"/></marker></defs>',
             f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>']
    y = 20
    inner_x = LEFT + LABEL_W
    inner_w = WIDTH - inner_x - LEFT
    for i, (name, rule, packages) in enumerate(ROWS):
        parts.append(_box(LEFT, y, WIDTH - 2 * LEFT, BOX_H, FILL[i]))
        parts.append(_text(LEFT + 12, y + 30, name, size=14, weight="bold"))
        parts.append(_text(LEFT + 12, y + 50, rule, size=11))
        # one chip per package
        chip_w = (inner_w - 10 * (len(packages) + 1)) / len(packages)
        for j, package in enumerate(packages):
            cx = inner_x + 10 + j * (chip_w + 10)
            parts.append(f'<rect x="{cx:.1f}" y="{y + 20}" width="{chip_w:.1f}" '
                         f'height="34" rx="4" fill="#ffffff" stroke="{STROKE}" '
                         f'stroke-width="0.8"/>')
            parts.append(_text(cx + chip_w / 2, y + 42, package, size=12,
                               anchor="middle", family="monospace"))
        if i < len(ROWS) - 1:
            ax = WIDTH / 2
            parts.append(f'<line x1="{ax}" y1="{y + BOX_H + 4}" x2="{ax}" '
                         f'y2="{y + BOX_H + GAP - 4}" stroke="{STROKE}" '
                         f'stroke-width="1.6" marker-end="url(#arrow)"/>')
            parts.append(_text(ax + 10, y + BOX_H + GAP / 2 + 4, "imports",
                               size=11))
        y += BOX_H + GAP
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    out = Path(__file__).with_name("layers.svg")
    out.write_text(build(), encoding="utf-8")
    print(out)
