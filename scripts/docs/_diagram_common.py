"""Shared helpers for the slide diagrams.

Graphviz ships with several ANSYS components (e.g. v252/optiSLang). Rather than
forcing a separate Graphviz install we just prepend whichever bundled copy we
find to PATH before importing ``diagrams``.
"""
from __future__ import annotations

import os
from pathlib import Path


def ensure_graphviz_on_path() -> Path | None:
    """Prepend the first dot.exe we can find to PATH. Returns its folder."""
    if any((Path(p) / "dot.exe").is_file() for p in os.environ.get("PATH", "").split(os.pathsep) if p):
        return None

    # Standalone Graphviz first -- the ANSYS-bundled copies are stripped
    # of the gd / pango image plugins so Custom(image=...) silently drops
    # the icon. Keep them as a last-resort fallback only.
    candidates = [
        Path(r"C:\Program Files\Graphviz\bin"),
        Path(r"C:\Program Files (x86)\Graphviz\bin"),
        Path(r"C:\Program Files\ANSYS Inc\v252\optiSLang\lib\graphviz"),
        Path(r"C:\Program Files\ANSYS Inc\v251\optiSLang\lib\graphviz"),
    ]
    for c in candidates:
        if (c / "dot.exe").is_file():
            os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            return c
    raise RuntimeError(
        "Graphviz dot.exe not found. Install Graphviz (winget install graphviz) "
        "or extend the candidate list in _diagram_common.py."
    )


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "diagrams"
ICONS_DIR = OUTPUT_DIR / "icons"


def icon(name: str) -> str:
    """Return absolute path to a workflow icon. Raises if not yet rendered."""
    p = ICONS_DIR / name
    if not p.is_file():
        raise FileNotFoundError(
            f"icon {name} missing -- run `python scripts/docs/render_icons.py` first"
        )
    return str(p)

# Slide-friendly defaults. We do NOT force size/ratio -- dot picks a natural
# tight layout and you scale the SVG inside PowerPoint to whatever frame fits.
#
# Font choice: Arial -- Helvetica fallback under Windows dot rendering
# produced broken glyph clusters (missing hyphens, kerned-apart letters) at
# small sizes. Arial ships with Windows and renders crisply through libgd.
SLIDE_GRAPH_ATTR = {
    "rankdir": "LR",
    "splines": "spline",
    "nodesep": "0.7",
    "ranksep": "1.3",
    "pad": "0.6",
    "bgcolor": "white",
    "fontname": "Arial",
    "fontsize": "44",
    "labelloc": "t",
    "compound": "true",
}
SLIDE_NODE_ATTR = {"fontname": "Arial", "fontsize": "26"}
SLIDE_EDGE_ATTR = {"fontname": "Arial", "fontsize": "22", "color": "#333333", "penwidth": "2.6"}
