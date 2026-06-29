"""Zoom slide: conformal node matching (d3plot -> APDL).

Three-step story, big artwork:
  1. Deformed strand outline   (input from ParaView extraction)
  2. Contact region + alignment   (the algorithm, two real debug plots)
  3. Final conformal mesh   (shared nodes on the contact arc)
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
    return Blank(
        label,
        shape="box", style="rounded,filled",
        fillcolor="#eef3f8", color="#6f8ea7",
        fontname="Arial", fontsize="26",
        margin="0.35,0.22", height="1.25", width="2.7",
        imagescale="false",
    )


def img(label: str, name: str, w: float = 3.2, h: float = 2.4) -> Custom:
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
    "Conformal node matching",
    show=False,
    direction="LR",
    outformat=["svg", "png"],
    filename=str(OUTPUT_DIR / "diagram_nodematching"),
    graph_attr=graph_attr,
    node_attr=SLIDE_NODE_ATTR,
    edge_attr=SLIDE_EDGE_ATTR,
):
    inp = img("Deformed strand\noutline", "paraview.png")

    with Cluster(
        "Per strand",
        graph_attr={"margin": "28", "fontname": "Arial", "fontsize": "30"},
    ):
        spline = text("B-spline fit")
        templ = text("Hex template\nmesh")
        mapper = text("Project to\nB-spline")
        spline >> templ >> mapper

    with Cluster(
        "Per adjacent pair",
        graph_attr={"margin": "28", "fontname": "Arial", "fontsize": "30"},
    ):
        contact = img("Find contact\nregion", "outer_nodes.png")
        align = img("Align nodes\non shared arc", "align_mesh.png")
        contact >> align

    out = img("Conformal mesh\n+ APDL", "conformal.png")

    inp >> spline
    mapper >> Edge(label="  two strand meshes  ") >> contact
    align >> out

print(f"wrote {OUTPUT_DIR / 'diagram_nodematching.svg'}")
