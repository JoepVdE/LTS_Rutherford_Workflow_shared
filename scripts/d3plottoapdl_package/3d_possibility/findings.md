# 3D cablestack meshing — feasibility findings & plan

Status: **scoping complete, no build started**
Date: 2026-05-18
Owner: JoepVdE

## 1. Problem

The 2D cablestack model works because `conformalRutherfordMesh.py` snaps
contacts at 15 µm tolerance and emits a clean, planar conformal mesh that APDL
can mesh slice-by-slice. The same geometry in 3D — extruded strand surfaces
from LS-DYNA — fails standard meshers (SpaceClaim, APDL volume mesher,
GMSH OCC with default tolerance) because LS-DYNA's strand surfaces have
sub-tolerance overlaps and gaps that produce slivers, self-intersections, and
geometry-kernel rejections.

Physical motivation: a clean 3D model would let us load the cable with the
boundary condition that actually matches a given experiment — magnet-coil
(axial periodicity, all strands sharing ε_zz, = GPS) or short Zwick sample
(free axial ends per constituent, = plane stress). The 2× stiffness gap
between the two 2D limits is real and physical, not a modelling defect; see
`memory/project_gps_vs_planestress_findings.md`.

## 2. Corrected mental model of the LS-DYNA geometry

I initially called the contact-pattern z-shifts "the cabling helix rotating
the layers." That was wrong. Reading
`scripts/setup_step/generate_rutherford.FCMacro`:

- The strands are **initialised in their final twisted Rutherford position**.
  Each strand `n` is lofted along the cable perimeter path
  `(x(z), y(z)) = perimeter_xy((n/N + z/Lp) % 1 · P)` where `perimeter_xy`
  parametrises the two long flats + two semicircular ends.
- LS-DYNA does **not** simulate cabling motion. It simulates only the
  compaction: four plates (Top/Bottom at clearance `r + offset_y`,
  Left/Right at `W/2 + offset_x`) are driven inward to squash the cable.
- Therefore the strand-strand contact topology at any z is set by the
  *initial geometry* at that z, and the squashing only decides which
  geometric proximities cross the 15 µm threshold and become active contacts.

This is good news: every topology transition along z is **predictable
analytically** from `Lp`, `N`, and `perimeter_xy` — no need to discover them
per-slice. Each transition = one strand crossing a semicircular end and
swapping layers in the cross-section.

## 3. Materials & contact physics decisions

- **Metal-metal strand contacts exist** in the real cable (confirmed by user
  reference to micrographs). Wax films do not necessarily separate every
  strand. → meshing approaches that wash out contacts (uniform wax fill,
  idealised circular cross-sections) are off the table.
- **Impregnation** = filled wax (E ≈ 1-10 GPa depending on filler) or epoxy.
  Stiff enough to carry load — must be meshed as its own volume.
- **Interface treatment:** bonded shared-nodes for both strand-strand and
  strand-impregnation interfaces. Justification:
  - Matches the 2D code's snap-to-coincident-node pattern → 2D/3D comparison
    is apples-to-apples.
  - Static transverse compression doesn't excite tangential slip, so contact
    vs. bonded gives the same answer for the questions we're asking.
  - 3D contact with mixed Cu/Nb3Sn (~100 GPa) and wax (~5 GPa) penalty
    stiffness produces ill-conditioned tangent matrices and convergence pain
    we don't need.
  - **Keep the option open**: design the GMSH build with separate volumes +
    coincident interface meshes. `NUMMRG,NODE` at APDL assembly gives
    bonded; skipping it and adding CONTA174/TARGE170 gives contact — same
    mesh, no re-mesh required. If we ever care about cyclic hysteresis or
    debond, the switch is one APDL pass.

## 4. Diagnostic: topology invariance along z

Script: `topology_diagnostic.py` (this folder). Per z-slice it builds the
B-spline outline of every strand, applies the same 15 µm + penetration
contact rule as `identify_contact_region`, and reports the contact graph
`G(z)`.

### Run: `data/runs/20260504_204511_R2D2_LF` (34 strands, 15 default slices)

- 14 valid z-stations (stack 15 had empty CSVs because
  `z_location_slice_max = 2*56/21` in `extract_coordinates_stack_sort.py` is
  hardcoded and ran past the deformed strand z-extent — separate bug).
- 49-50 contact pairs per slice; ~34 of those are persistent (intra-layer
  neighbours + end-of-cable inter-layer pairs).
- **5 stable zones, 4 transitions over ~5 mm of z.** Every transition is a
  clean "shift inter-layer pair indices by +1" — exactly what the
  `perimeter_xy` parametrisation predicts.
- Output for this run: `./20260504_204511_R2D2_LF/{report.txt, contact_graphs.json}`.

### Interpretation for the 3D approach

- Lab-frame prismatic sweep is **possible but transition-dense** at default
  15-slice resolution — ~1 transition per mm. A swept-prismatic + tet
  transition layer hybrid would not beat full tet at this density.
- Rotating-frame (sweep follows `perimeter_xy`) keeps topology constant by
  construction — single sweep, no transitions. This is the cleanest
  realisation of the swept-prismatic idea for a Rutherford winding.
- **Open question:** are transitions sharp (single-z) or gradual
  (multi-z with overlap zones)? 15 slices is too coarse to tell. Need
  ≥60 slices to characterise transition width before committing.

## 5. Decisions made

| Decision | Choice | Reason |
|---|---|---|
| Meshing kernel | GMSH OCC (Python bindings) | Better tolerance handling than SpaceClaim/APDL volume mesher; meshio 5.3.5 has read+write for Ansys CDB |
| Geometry source | Per-slice 2D conformal outlines + sweep | Preserves the 2D contact-healing we already trust; voxel discards available geometric fidelity |
| Strand-strand interface | Bonded shared-nodes (in mesh: coincident, in APDL: NUMMRG) | Matches 2D, static compression doesn't need contact, switchable later |
| Strand-impregnation interface | Bonded shared-nodes | Same as above; epoxy bond is physically correct, wax is acceptable for monotonic transverse load |
| Element type | Swept prism within stable zones, tet at transitions, tet for impregnation, swept prism for insulation | Cheapest mesh that respects geometry; CDB export supports SOLID186/187 |
| Z extent | One semi-pitch `L_semi = Lp/N` with periodic BC | Cross-section is geometrically periodic at `L_semi`; 1/N the model size for same physics |
| Default frame | Lab-frame *pending* Phase-1 transition-width result; switch to rotating-frame if transitions are gradual | Tooling is simpler in lab frame |

## 6. Open questions (to resolve before Phase 1 build)

1. **Transition width** — single-z (sharp) or multi-z (gradual)? Decides
   lab-frame vs. rotating-frame. Resolution: re-extract one run at 60 slices
   and re-run the diagnostic.
2. **Cross-cable consistency** — does R2D2_HF and CD1 show the same +1-shift
   transition pattern? Different `Lp/N` should give different transition
   spacing. Resolution: run the diagnostic on one HF and one CD1 folder.
3. **Insulation thickness in `cable_parameters.json`** — currently the 2D
   code uses an alpha-shape buffer; the 3D code needs a single thickness
   value. Confirm one is recorded per cable preset.
4. **Periodic BC node pairing** — required for `L_semi` model to be
   physically meaningful. Need a pre-CDB step that matches each z=0 node
   to its z=L_semi twin after the perimeter rotation. ~1 day of extra work.

## 7. Plan (5 phases, ~2.5 weeks focused)

Full plan with file paths and gotchas is in the conversation transcript;
condensed here:

1. **Phase 1 — dense extraction (~2 days):** add `--n-slices` to
   `extract_coordinates_stack_sort.py`, fix the `z_max = 2*56/21` bug, set
   z_max from `cable_parameters.json`. Re-run diagnostic at 60 slices.
2. **Phase 2 — per-slice geometry prep (~3 days):** new
   `build_3d_geometry.py` in this folder. For each slice: B-spline outline
   per strand (reuse `DeformedStrandInterpolator`), snap contacts to shared
   midpoint curves, compute insulation envelope (port `insulationlayer.py`
   buffer logic).
3. **Phase 3 — GMSH OCC assembly (~4 days, hardest part):** new
   `gmsh_assemble.py`. `addSpline` per outline, `addThruSections` to build
   strand and insulation volumes, boolean-subtract for impregnation. Set
   `Geometry.Tolerance = 1e-7`, normalise outline start angle + direction
   before lofting. Tag physical groups for material assignment.
4. **Phase 4 — mesh (~2 days):** transfinite prism within stable zones, tet
   elsewhere. Expected ~850k elements per `L_semi` for R2D2_LF.
5. **Phase 5 — CDB export + APDL smoke test (~2 days):** `gmsh.write` →
   `meshio.read` → `meshio.write(format='ansys')`. Verify physical groups
   become CDB components; if not, post-process to inject `CMBLOCK`. Load
   into MAPDL; apply 0.1% UY compression as smoke test.

**Fallback if Phase 3 stalls:** mesh-first extrusion. Mesh each 2D slice
with `conformalRutherfordMesh` (existing code), then connect matching nodes
between slices into prismatic elements via `gmsh.model.mesh.extrude` on a
`discrete` entity. Sidesteps OCC volume creation for strands but doesn't
help insulation.

## 8. Artifacts in this folder

- `topology_diagnostic.py` — per-z contact-graph diagnostic; reads
  `<run>/stack/Stack_*_Part*.csv` and writes per-run subfolders here.
- `20260504_204511_R2D2_LF/report.txt` — diagnostic output for the first
  validated LF run.
- `20260504_204511_R2D2_LF/contact_graphs.json` — machine-readable contact
  graph per stack, plus the persistent/intermittent classification (for
  consumption by future Phase 3 code).
- `findings.md` — this file.
