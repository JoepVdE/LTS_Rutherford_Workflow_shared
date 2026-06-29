"""Throwaway sanity check for compute_hex_circumradii_mm + plot_hex_overlay.

Generates one overlay plot per cable in
data/runs/_hex_overlay_check/ so the homogenisation can be eyeballed.
"""
import json
from pathlib import Path

import conformalRutherfordMesh as crm
from strandMeshGenerator import StrandMesh_Hexa


def main():
    cfg = json.loads(Path(r"c:\LTS_Rutherford_Workflow\scripts\main\cable_parameters_user.json").read_text())
    out = Path(r"c:\LTS_Rutherford_Workflow\data\runs\_hex_overlay_check")
    out.mkdir(exist_ok=True, parents=True)
    for name, cable in cfg["cables"].items():
        wire = cable["wire"]
        Ri, Ro = crm.compute_hex_circumradii_mm(wire)
        d_um = cable["D_Strand"] * 1e3
        print(
            f"{name:8s}  N={wire['N_NB3SN']:3d}/{wire['N_TOTAL']:3d}  "
            f"D_core={wire['D_CORE_EQ_UM']:.1f} um  t_Cu={wire['CU_SLEEVE_THICKNESS_UM']:.2f} um  "
            f"-> R_in={Ri*1e3:6.2f} um  R_out={Ro*1e3:6.2f} um  "
            f"(D_strand={d_um:.0f} um, R_strand={d_um/2:.0f} um, "
            f"R_out/R_strand={Ro*1e3/(d_um/2):.3f})"
        )
        m = StrandMesh_Hexa(
            diameter=cable["D_Strand"], radial_divisions=3, angle=0,
            inner_circumradius_mm=Ri, outer_circumradius_mm=Ro,
        )
        crm.plot_hex_overlay(
            m, Ri, Ro, str(out / f"hex_overlay_{name}.svg"),
            title=f"{name}  ({wire['N_NB3SN']}/{wire['N_TOTAL']}, "
                  f"D_core={wire['D_CORE_EQ_UM']} um, "
                  f"t_Cu={wire['CU_SLEEVE_THICKNESS_UM']} um)",
        )
    print("saved to", out)


if __name__ == "__main__":
    main()
