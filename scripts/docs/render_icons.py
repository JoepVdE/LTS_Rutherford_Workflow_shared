"""Build slide-icon PNG thumbnails from real workflow outputs.

Pulls representative artifacts from data/runs/ (ParaView extraction, conformal
mesh, align_debug, RVE hex overlay, postprocess subplots), converts SVGs to
PNG via cairosvg, and downscales to MAX_DIM. Output lands in
data/diagrams/icons/ where the diagram scripts pick them up.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# cairocffi can't find libcairo on a default Windows install. The ANSYS wxPython
# bundle ships one, so we extend PATH before importing cairosvg.
_ANSYS_CAIRO_CANDIDATES = [
    Path(r"C:\Program Files\ANSYS Inc\v252\optiSLang\lib\python3.10\Lib\site-packages\wx"),
    Path(r"C:\Program Files\ANSYS Inc\v251\optiSLang\lib\python3.10\Lib\site-packages\wx"),
    Path(r"C:\Program Files\ANSYS Inc\v251\AnsysEM\commonfiles\CPython\3_10\winx64\Release\python\Lib\site-packages\wx"),
]
for c in _ANSYS_CAIRO_CANDIDATES:
    if (c / "libcairo-2.dll").is_file():
        os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
        break

import cairosvg  # noqa: E402
from PIL import Image  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
ICONS_DIR = REPO / "data" / "diagrams" / "icons"
MAX_DIM = 1200  # px; thumbnails for diagrams.Custom (slide-size readable)

# (target filename, source under REPO/)  --  edit here to swap source artifacts.
SOURCES = [
    ("rve.png",
     r"data\runs\_hex_overlay_check\hex_overlay_R2D2_LF.png"),
    ("paraview.png",
     r"data\runs\20260504_204511_R2D2_LF\APDL\submodel\apdl_runfolder\plots\outer_polygon_and_box_5.png"),
    ("conformal.png",
     r"data\runs\20260504_232855_R2D2_HF_apdl_rerun_5\APDL\submodel\apdl_runfolder\plots\conformal_mesh\conformal_mesh_5_6_1.svg"),
    ("outer_nodes.png",
     r"data\runs\20260504_190849_R2D2_HF_apdl_rerun_2\APDL\submodel\apdl_runfolder\plots\align_debug\stack2_pair_14_15_indice_special_outer.svg"),
    ("align_mesh.png",
     r"data\runs\20260504_190849_R2D2_HF_apdl_rerun_2\APDL\submodel\apdl_runfolder\plots\align_debug\stack2_pair_14_15_indice_special_mesh.svg"),
    ("subplots.png",
     r"data\runs\20260504_232855_R2D2_HF_apdl_rerun_51\APDL\submodel\apdl_runfolder\plots\R2D2_HF_subplots.svg"),
]


def render() -> None:
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    for name, rel in SOURCES:
        src = REPO / rel
        dst = ICONS_DIR / name
        if not src.is_file():
            print(f"[skip] {name}: source missing -> {src}")
            continue
        if src.suffix.lower() == ".svg":
            cairosvg.svg2png(url=str(src), write_to=str(dst), output_width=1800)
        else:
            shutil.copyfile(src, dst)
        img = Image.open(dst).convert("RGBA")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        img.save(dst, optimize=True)
        print(f"[ok]   {name:<16} {img.size}  <-  {rel}")


if __name__ == "__main__":
    render()
