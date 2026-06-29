"""Pipeline overview slide (16:9), tuned for big-screen readability.

Design rules:
  - One concept per slide (the pipeline). Details live on zoom slides.
  - Short labels (<= 3 words) so they read at 30+ feet.
  - Generous whitespace; flow strictly left to right.
  - Real workflow output thumbnails carry the visual weight; intermediate
    stages are quiet text boxes that frame the imagery.
  - One side branch (RVE) routed to its actual consumer (cablestack build)
    rather than long edges that cross the canvas.
"""
from __future__ import annotations

from _diagram_common import (
    OUTPUT_DIR,
    SLIDE_EDGE_ATTR,
    SLIDE_GRAPH_ATTR,
    SLIDE_NODE_ATTR,
    ensure_graphviz_on_path,
    icon,
)

ensure_graphviz_on_path()

from diagrams import Cluster, Diagram, Edge
from diagrams.custom import Custom
from diagrams.generic.blank import Blank


def text(label: str) -> Blank:
    """Quiet rounded-box node for stages without a real workflow image."""
    return Blank(
        label,
        shape="box", style="rounded,filled",
        fillcolor="#eef3f8", color="#6f8ea7",
        fontname="Arial", fontsize="26",
        margin="0.35,0.22", height="1.25", width="2.7",
        imagescale="false",
    )


def img(label: str, name: str, w: float = 3.2, h: float = 2.4) -> Custom:
    """Image-icon node carrying a real workflow artifact.

    Custom() defaults to a ~1.4 inch icon regardless of source size, which
    looks like a postage stamp at slide resolution. We force a generous
    bounding box and scale the source image to fill it; the label sits below.
    """
    return Custom(
        label,
        icon(name),
        imagescale="true",
        imagepos="tc",
        labelloc="b",
        width=str(w), height=str(h),
        fontname="Arial", fontsize="26",
    )


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

graph_attr = {**SLIDE_GRAPH_ATTR, "ranksep": "1.1", "nodesep": "0.55"}

with Diagram(
    "LTS Rutherford pipeline",
    show=False,
    direction="LR",
    outformat=["svg", "png"],
    filename=str(OUTPUT_DIR / "diagram_overview"),
    graph_attr=graph_attr,
    node_attr=SLIDE_NODE_ATTR,
    edge_attr=SLIDE_EDGE_ATTR,
):
    cfg = text("Cable params\n(JSON + Python)")
    step = text("FreeCAD\nSTEP")
    mesh = text("Ansys mesh\n(.k)")
    lsdyna = text("LS-DYNA\nsolve")
    pv = img("Deformed strands\n(ParaView)", "paraview.png")
    conf = img("Conformal mesh\n+ APDL", "conformal.png")

    with Cluster(
        "Cablestack APDL  (Docker / HPC)",
        graph_attr={"margin": "28", "fontname": "Arial", "fontsize": "30"},
    ):
        build = text("build\nbase.db")
        dt = text("disp trans")
        dr = text("disp rad")
        pt = text("pres trans")
        pr = text("pres rad")
        build >> Edge(color="#888888") >> [dt, dr, pt, pr]

    pp = img("Stress-strain\npostprocess", "subplots.png")

    with Cluster(
        "RVE  (parallel sub-element)",
        graph_attr={"margin": "28", "fontname": "Arial", "fontsize": "30"},
    ):
        rve = img("RVE strand", "rve.png")

    # Main horizontal flow
    cfg >> step >> mesh >> lsdyna >> pv >> conf >> build
    [dt, dr, pt, pr] >> pp

    # Parallel RVE branch -- short routing into its real consumer.
    # Subscript via Graphviz HTML-like label so engineering notation renders
    # cleanly (no underscore artifacts from plain-text rendering).
    cfg >> Edge(style="dashed", color="#7a7a7a") >> rve
    rve >> Edge(
        style="dashed", color="#7a7a7a",
        label="<  E<SUB>xx</SUB>, E<SUB>yy</SUB>  >",
    ) >> build

print(f"wrote {OUTPUT_DIR / 'diagram_overview.svg'}")
