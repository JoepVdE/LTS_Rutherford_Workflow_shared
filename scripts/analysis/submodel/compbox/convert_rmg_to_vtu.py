"""MAG-only VTU conversion for the compression box parent model.

Reads the CompBox magnetic result (.rmg) and writes one "enhanced" VTU per
magnetic loadstep carrying Bx / By / Bz / B_magnitude point data on the
conductor mesh.  create_magnetic_heatmaps.py consumes exactly those four
arrays, so the mechanical .rst (strain / Jc enhancement of the legacy
convert_script.py in paper_clean_version) is intentionally dropped: the
field-table chain never used it, and skipping CompBox_mech halves the box
solve time.

Mesh and node coordinates come straight from the .rmg via ansys-mapdl-reader
(result.grid); no DPF and no .rst are needed.

Env vars (all set by compbox_stage.py; defaults support hand-runs):
    COMPBOX_RMG_PATH     explicit path to the .rmg (else newest match below)
    COMPBOX_BOX_RESULTS  dir holding CompBox_MAG_*.rmg   (default: cwd)
    COMPBOX_VTU_OUT      output dir for the VTUs         (default: COMPBOX_BOX_RESULTS)
"""

import glob
import os

import numpy as np

# NumPy 2.x removed aliases that ansys-mapdl-reader still uses.
if not hasattr(np, "in1d"):
    np.in1d = np.isin
if not hasattr(np, "alltrue"):
    np.alltrue = np.all

import pandas as pd
from scipy.spatial import KDTree
from ansys.mapdl import reader as pymapdl_reader

CONDUCTOR_MATERIAL_ID = 11  # VATT,11,,11 in CompBox_mag.inp


def _resolve_rmg_path() -> str:
    override = os.environ.get("COMPBOX_RMG_PATH")
    if override:
        return override
    results_dir = os.environ.get("COMPBOX_BOX_RESULTS", os.getcwd())
    matches = glob.glob(os.path.join(results_dir, "CompBox_MAG_*.rmg"))
    if not matches:
        raise FileNotFoundError(
            f"No CompBox_MAG_*.rmg in {results_dir} - run the box MAG solve "
            f"first (or set COMPBOX_RMG_PATH)."
        )
    return max(matches, key=os.path.getmtime)


def conductor_element_numbers(result) -> set:
    """Element numbers of every conductor element (material ID 11).

    Membership filtering instead of the legacy positional truncation: in the
    MAG mesh the air volumes are meshed around the conductor, so conductor
    elements are not guaranteed to come first in the element table.
    """
    elem_table = result.mesh.elem
    numbers = {int(e[8]) for e in elem_table if int(e[0]) == CONDUCTOR_MATERIAL_ID}
    if not numbers:
        unique_mats = sorted({int(e[0]) for e in elem_table})
        raise RuntimeError(
            f"No elements with material ID {CONDUCTOR_MATERIAL_ID} in the .rmg "
            f"mesh (materials present: {unique_mats})."
        )
    return numbers


def nodal_field_for_loadstep(result, conductor_set: set, loadstep: int) -> pd.DataFrame:
    """Node-averaged Bx/By/Bz over the conductor elements for one loadstep."""
    enum_mag, edata_mag, enode_mag = result.element_solution_data(
        loadstep, datatype="EFX"
    )

    rows = []
    for idx, elem_number in enumerate(enum_mag):
        if int(elem_number) not in conductor_set:
            continue
        data = edata_mag[idx]
        if data is None or len(data) == 0:
            continue
        b_per_node = np.reshape(np.asarray(data, dtype=float), (-1, 3))
        nodes = np.asarray(enode_mag[idx], dtype=int)[: len(b_per_node)]
        for node, (bx, by, bz) in zip(nodes, b_per_node):
            rows.append((int(node), bx, by, bz))

    df = pd.DataFrame(rows, columns=["Node", "Bx", "By", "Bz"])
    df = df.groupby("Node", as_index=False).mean()
    df["|B|"] = np.sqrt(df["Bx"] ** 2 + df["By"] ** 2 + df["Bz"] ** 2)
    return df


def fill_missing_by_nearest(complete_df: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbour fill (mean of 2 closest valid points) for mesh nodes
    without conductor field data, mirroring the legacy interpolation."""
    nan_mask = complete_df["|B|"].isna()
    if not nan_mask.any():
        return complete_df

    valid_mask = ~nan_mask & np.isfinite(
        complete_df[["x", "y", "z"]].to_numpy()
    ).all(axis=1)
    if valid_mask.sum() < 2:
        print("Not enough valid reference points; filling missing B with 0.")
        for col in ("Bx", "By", "Bz", "|B|"):
            complete_df[col] = complete_df[col].fillna(0.0)
        return complete_df

    valid_coords = complete_df.loc[valid_mask, ["x", "y", "z"]].to_numpy()
    valid_indices = complete_df.index[valid_mask].to_numpy()
    tree = KDTree(valid_coords)

    nan_indices = complete_df.index[nan_mask].to_numpy()
    query = complete_df.loc[nan_indices, ["x", "y", "z"]].to_numpy()
    _, nn = tree.query(query, k=min(2, len(valid_indices)))
    nn = np.atleast_2d(nn)
    for col in ("Bx", "By", "Bz", "|B|"):
        col_values = complete_df[col].to_numpy()
        fills = col_values[valid_indices[nn]].mean(axis=1)
        complete_df.loc[nan_indices, col] = fills
    return complete_df


def main() -> None:
    rmg_path = _resolve_rmg_path()
    out_dir = os.environ.get(
        "COMPBOX_VTU_OUT",
        os.environ.get("COMPBOX_BOX_RESULTS", os.path.dirname(rmg_path) or "."),
    )
    os.makedirs(out_dir, exist_ok=True)

    base_name = (
        os.path.splitext(os.path.basename(rmg_path))[0].lower() + "_final_properties"
    )

    print(f"Reading magnetic result: {rmg_path}")
    result = pymapdl_reader.read_binary(rmg_path)
    num_loadsteps = len(result.time_values)
    print(f"Magnetic loadsteps: {num_loadsteps}")

    conductor_set = conductor_element_numbers(result)
    print(f"Conductor elements (mat {CONDUCTOR_MATERIAL_ID}): {len(conductor_set)}")

    grid = result.grid
    coords = np.asarray(grid.points)
    if "ansys_node_num" in grid.point_data:
        node_ids = np.asarray(grid.point_data["ansys_node_num"], dtype=int)
    else:
        print("WARNING: ansys_node_num missing; assuming sequential node IDs.")
        node_ids = np.arange(1, len(coords) + 1)
    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["Node"] = node_ids

    written = []
    for loadstep in range(num_loadsteps):
        print(f"--- Loadstep {loadstep + 1}/{num_loadsteps} ---")
        field_df = nodal_field_for_loadstep(result, conductor_set, loadstep)
        complete_df = coords_df.merge(field_df, on="Node", how="left")
        complete_df = fill_missing_by_nearest(complete_df)

        mesh_copy = grid.copy()
        mesh_copy.point_data["Bx"] = complete_df["Bx"].to_numpy()
        mesh_copy.point_data["By"] = complete_df["By"].to_numpy()
        mesh_copy.point_data["Bz"] = complete_df["Bz"].to_numpy()
        mesh_copy.point_data["B_magnitude"] = complete_df["|B|"].to_numpy()

        out_path = os.path.join(
            out_dir, f"{base_name}_enhanced_loadstep_{loadstep:02d}.vtu"
        )
        mesh_copy.save(out_path)
        written.append(out_path)
        print(f"Wrote {out_path}")

    print(f"\nDone: {len(written)} enhanced VTUs in {out_dir}")


if __name__ == "__main__":
    main()
