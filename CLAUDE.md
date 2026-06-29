# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Does

End-to-end automated pipeline for Nb3Sn LTS Rutherford cable simulation. Python orchestrates: cable-parameter calculation → FreeCAD STEP geometry → Ansys Mechanical solid meshing (ansys-mechanical-core) → LS-DYNA solve in Docker → ParaView d3plot extraction → APDL submodel generation (conformal mesh on deformed strands) → 4-stage APDL cablestack 2D **generalized-plane-strain** solve (displacement / pressure × transverse / radial). The GPS DOFs let Nb3Sn axial strain develop naturally from transverse loading (Poisson coupling) — feeds Ic prediction.

Single entry point: [scripts/main/main.py](scripts/main/main.py).

## Running the Pipeline

```powershell
$env:PYTHONIOENCODING = "utf-8"
& "C:/Program Files/Python312/python.exe" scripts/main/main.py [OPTIONS]
```

Install Python deps from [pyproject.toml](pyproject.toml): `pip install -e .` (orchestrator deps: `ansys-mechanical-core`, `ansys-dyna-core==0.9.0`, `alphashape`, `matplotlib`, `networkx`, `numpy<3`, `pandas`, `python-pptx`, `scipy`, `shapely`). `pip install -e .[dev]` adds `ruff` + `pytest`. The external tools (ANSYS, Docker, FreeCAD, ParaView) are not pip-installable — see below. A containerized orchestrator is also provided ([Dockerfile](Dockerfile) — Docker-out-of-Docker, spawns sibling mesher/LS-DYNA/MAPDL containers on the host daemon via a mounted socket).

Key flags:

| Flag | Purpose |
|------|---------|
| `-c {R2D2_LF\|R2D2_HF\|CD1}` | Cable preset (default `R2D2_LF`) |
| `--cables <NAME> [NAME …]` | Parallel runs — spawns one subprocess per cable |
| `-t <ms>` | LS-DYNA termination time (default `0.0001`) |
| `--apdl-only` | Re-run d3plot→APDL submodel + cablestack on latest completed run |
| `--quick-run` | Skip geometry + meshing; redo mesh-conversion + LS-DYNA |
| `--no-cablestack` | Generate cablestack `.inp` files but do NOT launch any MAPDL stage |
| `--hpc` | Run cablestack APDL on an SSH-reachable SLURM HPC cluster (upload + sbatch + wait + fetch). Honours `cablestack.stages`. Default target is ETH Euler; override via `HPC_HOST` / `HPC_USER` / `HPC_REMOTE_BASE` env vars. Needs SSH key access to the chosen cluster |
| `--no-cache` | Force a fresh run from STEP geometry onward, bypassing the run cache (see below) |
| `--debug-plots` | Emit per-pair conformal mesh / outer-node SVGs (slow) |
| `--compbox` | Run the compression box simulation (step 9) after cablestack, even if `compression_box.enabled` is false |
| `--compbox-only` | Run only the compression box simulation on the latest run folder for the selected cable (needs step-7 geometry); add `--hpc` to solve on the cluster |

`--cables` passes `ACTIVE_CABLE=<name>` as env var to avoid concurrent-write corruption of `cable_parameters_user.json`.

**Cablestack stage selection is JSON-only:** the four-stage matrix is controlled by `cablestack.stages` in [scripts/main/cable_parameters_user.json](scripts/main/cable_parameters_user.json) — no per-stage CLI flag. `--no-cablestack` overrides to `[]`.

**Run cache (opt-out via `--no-cache`).** [scripts/main/cache.py](scripts/main/cache.py) is a content-addressed cache with two levels: `lsdyna` (covers params→geometry→mesh→LS-DYNA→ParaView extraction — a hit means the d3plot + per-stack CSVs are reusable, APDL submodel + cablestack still re-run on top) and `cablestack` (covers the above PLUS cablestack config/templates — a hit returns the existing run folder as the final answer). Index lives at `data/cache/index.json` (atomic write, registrations serialized via a `.idx.lock` cross-process file lock so parallel `--cables` runs cannot drop each other's entries); entries point at `data/runs/` folders and self-evict when the folder is deleted or its `metadata.json` `workflow_steps` aren't `completed`. Caching is on by default; `--no-cache` / `--apdl-only` / `--quick-run` bypass it.

## Software Requirements

- **Python 3.12** at `C:/Program Files/Python312/python.exe`
- **ANSYS 2025 R2 (v252)** — Mechanical APDL, LS-DYNA, valid license
- **Docker Desktop** — Ansys Mechanical mesher, LS-DYNA, and MAPDL solver containers. Image registry is selected by the `REGISTRY_PREFIX` env var (default `gitea.psi.ch/vanden_j`); CERN users set `REGISTRY_PREFIX=registry.cern.ch/chart-magnum`. Images: `<prefix>/mechanical:25.2`, `<prefix>/lsdyna:25.2`
- **FreeCAD 1.0.2** bundled in [tools/freecad/](tools/freecad), runs headless
- **ParaView 6.0.1** bundled in [tools/paraview/](tools/paraview), invoked via `pvpython.exe`

LS-DYNA license auto-detected: ETH `1801@lic-ansys-research.ethz.ch`, PSI `1055@winlic03.psi.ch`.

## Repository Layout

| Path | Role |
|------|------|
| [scripts/main/main.py](scripts/main/main.py) | Sole CLI entry point; orchestrates the whole pipeline (`WorkflowRunner`). Delegates solving/licensing/caching/stage-registry to the modules below |
| [scripts/main/cablestack_stages.py](scripts/main/cablestack_stages.py) | **Single source** for `CABLESTACK_STAGES`, `STAGE_USECASE_SUFFIX`, `resolve_cablestack_stage_order`, `write_cablestack_jobslurm`. Imported by both main.py and analyse_pressure.py |
| [scripts/main/solver.py](scripts/main/solver.py) | `CablestackSolver` port + `LocalMAPDL` (one `docker compose up` per stage; **independent load stages run concurrently** up to `cablestack.max_parallel_stages`, each with its own initial jobname `stage_<name>`) and `HPCMAPDL` (upload + one SLURM job for all stages on any SSH-reachable cluster; default ETH Euler) adapters; both expose `run_stages(...)` with a per-stage `on_stage_complete` callback |
| [scripts/main/hpc_submit.py](scripts/main/hpc_submit.py) | Upload + sbatch + wait + fetch a runfolder on any SSH-reachable SLURM HPC cluster (wrapped by `solver.HPCMAPDL`); generic `submit_runfolder`/`fetch_outputs` used by the compbox stage; env overrides `HPC_{USER,HOST,REMOTE_BASE}` (defaults target ETH Euler) |
| [scripts/main/compbox_stage.py](scripts/main/compbox_stage.py) | Compression box simulation (step 9): staging + deck patching + local-Docker/HPC solves + field-coupling and analysis sub-steps |
| [scripts/apdl/parentmodel/compbox/](scripts/apdl/parentmodel/compbox) | Parent 3D box MAG deck (`CompBox_mag.inp`, MAG-only), `BOX9.txt` measurement (single source for the generated loading schedule), `box9_loading.inp`/`box6_loading.inp` reference schedules |
| [scripts/apdl/submodel/compbox/](scripts/apdl/submodel/compbox) | One-turn submodel decks (`0-start.inp` driver + `2-geo`/`3-mesh`/`4-cont`/`5-BC-submodel`/`6-PP-submodel`) + `cable_materials_LN2.inp` (Nb3Sn fixed at the 70 GPa standard) |
| [scripts/analysis/submodel/compbox/](scripts/analysis/submodel/compbox) | `convert_rmg_to_vtu.py` (MAG-only `.rmg`→VTU), `create_magnetic_heatmaps.py` (field tables), `complete_analysis.py` (strain/Ic correlation), `ic_calculator.py` + `nb3sn_law.py` |
| [scripts/main/license_detector.py](scripts/main/license_detector.py) | `LicenseDetector` port + `NetworkProbeLicenseDetector` (TCP-probes ETH/PSI/CERN candidates); override with `ANSYS_LICENSE_SERVER` env var |
| [scripts/main/cache.py](scripts/main/cache.py) | Content-addressed run cache (`lsdyna` + `cablestack` levels); index at `data/cache/index.json` |
| [scripts/main/build_plane_stress_run.py](scripts/main/build_plane_stress_run.py) | Clones a GPS `apdl_runfolder/` → `apdl_runfolder_ps/` sibling, patches `formulation = 0`, writes a non-fatal `jobslurm.sh`. CLI: `python build_plane_stress_run.py [CABLE ...]` — auto-picks each cable's most recent run folder |
| [scripts/main/calc_cable_params_sim.py](scripts/main/calc_cable_params_sim.py) | Computes pitch/twist/velocity geometry from user JSON |
| [scripts/main/cable_parameters_user.json](scripts/main/cable_parameters_user.json) | Cable presets (`R2D2_LF`, `R2D2_HF`, `CD1`), wire material, `cablestack.{impreg, bc_type, boundary_type, formulation, stages, max_parallel_stages, mesh_size_um, strand_mesh_size_um, pressure}` |
| [scripts/setup_step/generate_step.py](scripts/setup_step/generate_step.py) | Drives FreeCAD headless macro to write `.step` cable geometry |
| [scripts/lsdyna/script/](scripts/lsdyna/script) | Ansys Mechanical meshing (`mesh_to_lsdyna.py`); `primemesh.py` and `lsdyna_setup.py` are legacy/unused |
| [scripts/meshconverter/](scripts/meshconverter) | Templated `.k` blocks + `inputfile_generator.py` building `processed_input.k` |
| [scripts/paraview/extract_coordinates_stack_sort.py](scripts/paraview/extract_coordinates_stack_sort.py) | Extracts deformed strand cross-sections from d3plot into per-stack CSVs |
| [scripts/d3plottoapdl_package/](scripts/d3plottoapdl_package) | `conformalRutherfordMesh.py`, `hexagon.py`, `insulationlayer.py` — builds APDL submodel from strand outlines |
| [scripts/apdl/submodel/cablestack/](scripts/apdl/submodel/cablestack) | APDL templates for all 4 cablestack stages; `jobslurm.sh` for HPC |
| [scripts/apdl/submodel/RVE/](scripts/apdl/submodel/RVE) | Placeholder folder. RVE sub-element pipeline is in development; not yet released. See its README for status. |
| [scripts/analysis/submodel/cablestack/analyse_pressure.py](scripts/analysis/submodel/cablestack/analyse_pressure.py) | Per-stage cablestack postprocessing (4 `postprocess_*` functions + `analyse`) |
| [scripts/analysis/submodel/cablestack/analysis_utils.py](scripts/analysis/submodel/cablestack/analysis_utils.py) | Shared helpers for the analysis scripts: apdl_runfolder resolution, latest-run discovery, cable-label extraction, float-table / stress-strain-txt parsers. Import target for analyse_pressure, plot_fd_good and the compare scripts |
| [scripts/analysis/submodel/cablestack/plot_fd_good.py](scripts/analysis/submodel/cablestack/plot_fd_good.py) | Standalone fd_good plotter (CLI use; older sibling of analyse_pressure) |
| [scripts/analysis/submodel/cablestack/](scripts/analysis/submodel/cablestack) | Ad-hoc comparison/diagnostic plotters: `compare_cables.py`, `compare_gps_vs_planestrain.py`, `diagnostic_envelope_slope.py`, `presentation_plots.py`, `probe_step_boundary.py`, etc. — standalone CLI scripts, not part of the pipeline |
| [scripts/apdl/test/gps_minimal/](scripts/apdl/test/gps_minimal) | Minimal standalone GPS validation deck (`gps_minimal.inp`) — confirms `KEYOPT(3,5)` + `GSGDATA`/`GSBDATA` + mixed u-P compatibility |
| [scripts/docs/](scripts/docs) | Presentation/diagram generators (`slide_*.py`, `diagram_*.py`, `render_icons.py`) — not part of the pipeline |
| [Dockerfile](Dockerfile) / [pyproject.toml](pyproject.toml) | Containerized orchestrator (Docker-out-of-Docker) + Python package metadata / dependency manifest |
| [data/runs/](data/runs) | Output: one folder per run, named `YYYYMMDD_HHMMSS_<CABLE>[_apdl_rerun[_N]]` |
| [data/cache/index.json](data/cache) | Run-cache index (see Run cache above) |

## Pipeline Stage Summary

1. **Cable params** — `calc_cable_params_sim.export_parameters_to_json` → `cable_parameters.json`
2. **Geometry** — FreeCAD headless → `<cable>.step`
3. **Meshing** — Ansys Mechanical in Docker (`USE_DOCKER=True`, image `mechanical:25.2`) → `mesh.k`. Element size: `π × D_strand / 20` mm unless `--min-mesh-size` is given
4. **Mesh conversion** — `inputfile_generator` stitches templated `.k` blocks → `processed_input.k`
5. **LS-DYNA** — `docker compose up -p lsdyna_<run>`; tailed log prints live `[<CABLE>] LS-DYNA: 45.2%`
6. **ParaView extraction** — `pvpython extract_coordinates_stack_sort.py <run_folder>` → per-stack CSVs
7. **APDL submodel** — `conformalRutherfordMesh.run(...)` → conformal mesh + APDL `.inp` fragments
8. **Cablestack copy/patch** — templates copied into `apdl_runfolder/` and variable-patched; `loading_cycle.json` (schema v2) recorded with per-stage usecase + output map
9. **Cablestack solve** — `run_cablestack_stages` resolves `cablestack.stages` into dependency levels; `build` runs alone, then the independent load stages run **concurrently** (one MAPDL container each, capped by `cablestack.max_parallel_stages`, default 4 — each container holds a license seat); per-stage postprocess fires on success (callbacks serialized)
11. **Compression box** *(opt-in; `9_compression_box` in metadata)* — parent MAG box solve → `.rmg`→VTU → field tables → one-turn submodel solve → strain/Ic analysis. See the **Compression Box Simulation** section below.

`metadata.json` tracks `workflow_steps` status dict (`5_lsdyna_simulation`, `6_paraview_extraction`, `7_apdl_submodel`, `8_cablestack`, plus `9_compression_box` when the opt-in compbox stage runs). `--apdl-only` uses these keys to find the latest qualifying run.

## Configuration (`cable_parameters_user.json`)

- `active_cable`: written by `setup_cable_config`; parallel subprocesses use `ACTIVE_CABLE` env var
- `wire_material.sigy_MPa`, `wire_material.etan_MPa`: bilinear plasticity for LS-DYNA
- `cablestack.impreg`: `1`=epoxy RT, `2`=wax RT, `3`=epoxy LN2, `4`=wax LN2
- `cablestack.bc_type`: `'cyclic'` or `'linear'` → selects which `5-BC-*.inp` is copied as `5-BC.inp` (used only by `displacement_transverse`)
- `cablestack.stages`: ordered list of stage names to run (auto-includes dependencies). Default = all four. Empty list = generate `.inp` files only. `--no-cablestack` overrides to `[]`
- `cablestack.max_parallel_stages`: max concurrent MAPDL containers for independent load stages on local Docker runs (default 4; set 1 for fully sequential, e.g. when license seats are scarce). HPC runs are unaffected (one SLURM job sequences the stages)
- `cablestack.formulation`: `1` = GPS + mixed u-P (default), `0` = plane stress. Patched into `0-start.inp` by `copy_cablestack_files` — see the element formulation toggle below
- `cablestack.mesh_size_um` / `strand_mesh_size_um`: reference element sizes at `D_Strand = 0.85 mm` → `s_ae` / `s_ae_str`. Applied size = `value * (D_Strand / 0.85)` µm — the JSON value scales linearly with wire diameter. Default reference is 50 µm when the key is omitted
- `cablestack.pressure`: `gauge_length_mm`, `peak_force_N`, `min_force_N`, `ramp_pressures_MPa` — drives the **same** pressure-cycle block into both `5-BC-pressure.inp` and `5-BC-radial.inp`
- **Element formulation toggle** — `cablestack.formulation` in the JSON, patched into the `formulation` parameter of `0-start.inp` at copy time:
  - `formulation = 1` (default) — generalized plane strain + mixed u-P (`KEYOPT(1,3,5)` + `KEYOPT(1,6,1)` + `GSGDATA` + `GSBDATA` in every BC). Physically correct for a long magnet cable; lets ε_zz develop on Nb3Sn for Ic prediction. Required for radial stage convergence on heterogeneous near-incompressible composite.
  - `formulation = 0` — plane stress (`KEYOPT(1,3,0)`, no u-P, no GSGDATA/GSBDATA). Matches Zwick uniaxial-compression test BC where the sample is free at both axial ends. Radial stage may not converge; use the non-fatal `run_stage_continue` jobslurm pattern (see `scripts/main/build_plane_stress_run.py`). For composite-cable models the in-plane response under plane stress is 2–3× softer than under GPS-with-F=0 because each material can independently Poisson-contract in z, instead of being forced to share one uniform ε_zz.
  - `2-geo.inp` wraps the KEYOPT/GSGDATA block in `*IF formulation EQ 1 THEN … *ELSE keyopt,1,3,0 *ENDIF`. The `GSBDATA` call (same `*IF`) lives once in the shared `5-BC-solver-settings.inp` include.
  - To build a plane-stress sibling of an existing GPS run folder, use `python scripts/main/build_plane_stress_run.py` — clones `apdl_runfolder/` → `apdl_runfolder_ps/` (in a `<run>_ps` sibling), overwrites formulation-sensitive templates, patches `formulation = 0`, and writes a non-fatal `jobslurm.sh` (each stage runs even if a prior one fails to converge).

## Cablestack Stage Architecture

Registry `CABLESTACK_STAGES` lives in [scripts/main/cablestack_stages.py](scripts/main/cablestack_stages.py) (main.py re-imports it; analyse_pressure.py imports the derived `STAGE_USECASE_SUFFIX`):

| Stage name | `input_file` | `depends_on` | `usecase_suffix` | Outputs |
|---|---|---|---|---|
| `build`                   | `0-start.inp` | — | `""` *(build itself produces no `pp/` output; just `base.db`)* | `base.db` |
| `displacement_transverse` | `00-restart-transverse.inp` | `build` | `""` | `fd_good_<cable>.txt` |
| `displacement_radial`     | `00-restart-radial.inp` | `build` | `_disp_radial` | `fd_good_<cable>_disp_radial.txt` |
| `pressure_transverse`     | `00-restart-pressure.inp` | `build` | `_pressure` | `fd_pressure_<cable>_pressure.txt`, `uy_top_<cable>_pressure.txt` |
| `pressure_radial`         | `00-restart-pressure-radial.inp` | `build` | `_radial` | `fd_radial_<cable>_radial.txt`, `ux_left_<cable>_radial.txt` |
| `thermal_cooldown` *(SKELETON, not implemented)* | `0-start-thermal.inp` | — | `_thermal` | *(none — `/EXIT,NOSAVE` stub; do not add to `cablestack.stages` until physics filled in)* |

**Strict start / restart split.** `0-start.inp` is the only **build** deck — it sets parameters, runs `2-geo` + `3-mesh` + `4-cont`, then `SAVE,base,db` and exits. It performs **no** BC application and **no** SOLVE. The four load stages are pure **restart** decks: each begins with `FINISH / /CLEAR,START / RESUME,base,db`, overrides only the `usecase`, then calls the appropriate `5-BC-*` file (canonical or `-free` per `cablestack.boundary_type`) and a `7-PP` / `8-PP-*` postprocessor. All four load stages start from an **undeformed** cable — no inherited displacement, plastic strain, or stress.

Stage-naming convention: `<bc_mode>_<direction>` where `bc_mode ∈ {displacement, pressure}` and `direction ∈ {transverse (Y, top wall), radial (X, left wall)}`. **"radial" is what an earlier version called "lateral"** — all lateral→radial renames are complete (decks, BC files, fd_*.txt prefixes, usecase suffixes, postprocess function names). Do not reintroduce "lateral" in cablestack code.

**Free vs. constrained boundary type.** `cablestack.boundary_type` in the JSON config (default `constrained`) selects which BC file is active for every load stage:
- `constrained` (default): sidewalls perpendicular to the load are pinned in that direction (cable inside a rigid die). Uses the canonical `5-BC-*.inp` files.
- `free`: sidewalls drop their perpendicular constraints; the anchor moves to the loaded-against wall (full `UX=UY=0`) so the cable can bulge under Poisson effect (matches an unconfined Zwick test). At `copy_cablestack_files` time, the patcher overwrites the canonical `5-BC-{cyclic,displacement-radial,pressure,radial}.inp` files with their `-free` siblings. `linear` + `free` is intentionally not implemented; the patcher raises a clear error.

Key invariants:
- The usecase-suffix map is **single-sourced**: analyse_pressure.py imports `STAGE_USECASE_SUFFIX` from `cablestack_stages.py` — never re-declare it locally
- All four load stages need `base.db` produced by the `build` stage. `LocalMAPDL._run_one` pre-flights this with `(dst_dir / "base.db").is_file()` and skips with a warning if absent
- Each load stage exits with its own jobname → distinct `.db` files (`…_disp_radial.db`, `…_pressure.db`, `…_radial.db`, `<cable>.db`) sit alongside `base.db`; restart decks always `RESUME,base,db` by name (not the per-stage `.db`)
- Docker project names are `mapdl_<run_folder>_<stage_name>` (lowercase); per-stage logs go to `<apdl_runfolder>/mapdl_<stage>.log`. Each local container starts MAPDL with initial jobname `stage_<stage_name>` (via `MAPDL_JOBNAME`) so concurrent stages in the same folder cannot collide on `file.lock` before their decks' `/filname` takes over
- The four load stages are mutually independent and run in parallel locally (dependency level 1 after `build`); on HPC one SLURM job runs them sequentially but **non-fatally** — a failed stage logs `<stage>_FAILED`, sets the job exit code, and the remaining stages still run. rc=0 stages drop a `<stage>.success` sentinel (fetched back by hpc_submit)
- `resolve_cablestack_stage_order` is dependency-order-aware: requesting only `pressure_radial` automatically prepends `build` (every load stage depends only on `build`)

## Cablestack Postprocessing

[scripts/analysis/submodel/cablestack/analyse_pressure.py](scripts/analysis/submodel/cablestack/analyse_pressure.py) exposes one public function per stage, plus `analyse(apdl_runfolder)` which runs them all:

| Stage | Function |
|---|---|
| `displacement_transverse` | `postprocess_displacement_transverse(apdl_runfolder)` |
| `displacement_radial`     | `postprocess_displacement_radial(apdl_runfolder)` |
| `pressure_transverse`     | `postprocess_pressure_transverse(apdl_runfolder)` |
| `pressure_radial`         | `postprocess_pressure_radial(apdl_runfolder)` |
| `thermal_cooldown` *(SKELETON)* | `postprocess_thermal_cooldown(apdl_runfolder)` — returns `False` until implemented |

All read `loading_cycle.json` for geometry/schedule and `0-start.inp` for `y_cab` (→ `total_height = n_stacks × 2 × y_cab`). Outputs: `<usecase>_subplots.svg` + `<usecase>_stress_strain.svg` in `<apdl_runfolder>/plots/`, plus `<usecase>_stress_strain.txt` in the apdl_runfolder root. Missing inputs return False without raising — never break the pipeline because of postprocessing.

`WorkflowRunner.run_cablestack_postprocess(dst_dir, stage_name=None)` is the main.py-side dispatcher. With a `stage_name` it calls only that stage's function; without one, it runs every postprocessor (each silently skipping itself if its inputs aren't present). `run_cablestack_stages` calls the dispatcher per-stage immediately after each stage's MAPDL container succeeds.

## Compression Box Simulation (step 9, opt-in)

Ported from the `paper_clean_version` box+submodel pipeline. Orchestrated by
[scripts/main/compbox_stage.py](scripts/main/compbox_stage.py)
(`run_compression_box`), invoked by `WorkflowRunner.run_compression_box` after
the cablestack stage when `compression_box.enabled` is true or `--compbox` is
passed; `--compbox-only` runs it standalone on the latest run folder.

Five sub-steps (per-substep status in `<run>/APDL/compbox/compbox_summary.json`):

| Substep | What it does |
|---|---|
| `parent_mag` | Stage + solve the 3D box magnetic model (`CompBox_mag.inp` → `CompBox_MAG_<cable>_submodel.rmg`). **MAG-only** — the legacy `CompBox_mech` `.rst` was visualization-only and is dropped from the field path |
| `vtu_export` | `convert_rmg_to_vtu.py`: `.rmg` → per-loadstep enhanced VTUs (Bx/By/Bz/B_magnitude on the conductor mesh, mat ID 11). Reads mesh straight from the `.rmg`; no DPF, no `.rst` |
| `field_tables` | `create_magnetic_heatmaps.py`: VTUs + `keypoints_nodes_<i>.txt` (from step 7) → `nb3sn_combined_data_case_<i>_<t>.inp` field tables + heatmap PNGs. Driven entirely by `COMPBOX_*` env vars |
| `submodel` | Stage + solve the one-turn 2D submodel: one case per stack cross-section (`cases = n_stacks`), 28-step BOX9 schedule, writes `strains_out_strand_*_set_*_case_*.out` |
| `analysis` | `complete_analysis.py` (strain + Ic correlation CSVs/SVGs) and `ic_calculator.py` |

Run-folder layout: `<run>/APDL/compbox/{parent_runfolder, vtu, field_tables,
submodel_runfolder, results, compbox_summary.json}`. Both runfolders are staged
**flat** (decks `/INPUT` geometry, field tables and materials by bare filename)
— same proven convention as the paper_clean_version orchestrator and the HPC
uploader, which only transfers top-level files.

Key invariants:
- **Nb3Sn = 70 GPa standard, fixed.** The compbox submodel does NOT take the
  RVE-homogenised modulus override that cablestack uses — the box Mat 11
  homogenised conductor (70 GPa × 1.2 amplification) is calibrated against the
  70 GPa value, and the RVE amplification factor is unresolved. Stamped in
  `compbox_summary.json` under `nb3sn_modulus`.
- **Deck patching is sentinel/regex based** (`compbox_stage.stage_*_runfolder`):
  the parent deck's `<<<COMPBOX_PARENT_PARAMS_START/END>>>` block carries
  `cable_type` (always `<cable>_submodel`) and `wf_cable_width/height`
  (2·x_cab / 2·y_cab in metres — override the cable_type dimension branch so
  any preset runs without a new `*elseif`); the submodel `0-start.inp` gets
  `cases` (= `n_stacks`), `n_strands`, `usecase`, `x_cab`, `y_cab` patched, and
  `1-material_properties.inp` gets `impreg`. x_cab/y_cab use the same formula
  as the cablestack patch (`cable_width/2 + x_cab_margin_mm`,
  `stack_height_mm/2`).
- **Loading schedule is single-sourced from the measurement table.**
  `compression_box.measurement_file` (BOX9.txt-style: pressure [MPa], Ic under
  load [A], Ic after unload [A/NaN] per row; row 1 = baseline/virgin) is the
  ONLY place to define a campaign's loading. At staging time
  `compbox_stage.build_loading_steps` expands it into interleaved
  load/unload pairs and writes the identical `nsteps`/`totalCurr`/`pres_pusher`
  block into the parent runfolder (`loading_schedule.inp`, `/INPUT`'d by
  `CompBox_mag.inp`) and into the staged `5-BC-submodel.inp` (between the
  `<<<COMPBOX_LOADING_BLOCK_START/END>>>` sentinels) — the two decks cannot
  drift. Audit: `<run>/APDL/compbox/loading_schedule.json`. Optional
  `compression_box.nsteps` caps the step count (keep it even to preserve
  pairs). The `box9_loading.inp`/`box6_loading.inp` templates and the
  hardcoded block in the `5-BC-submodel.inp` template are reference-only for
  hand-runs — editing them has no effect on staged runs.
- **Solvers:** `compression_box.solver` = `local` (the same docker-compose
  MAPDL service as cablestack; `MAPDL_NCPU`/`MAPDL_MEMORY` from
  `local_ncpu`/`local_memory_mb`) or `hpc` (one SLURM job per solve via
  `hpc_submit.submit_runfolder`; jobscripts are generated by
  `compbox_stage._write_jobscript` with the `rm -f ./*.lock` stale-lock guard
  and `<tag>_END rc=N` markers parsed by `parse_stage_results`).
- The Python analysis scripts under `scripts/analysis/submodel/compbox/` are
  path-parameterised via `COMPBOX_*` env vars set by compbox_stage — do not
  hardcode run paths in them.

## ⚠ APDL Authoring Rules (READ BEFORE TOUCHING ANY `.inp`) ⚠

**These rules are non-negotiable.** Every line of APDL written or edited in this repo must satisfy them. Violations have repeatedly caused silent solver failures, malformed restart `.db` files, and hours-long debug sessions. When in doubt, stop and verify — do not guess at syntax.

### Hard rules

1. **Verify EVERY command against ANSYS v18.2 docs** before writing it: https://www.mm.bme.hu/~gyebro/files/ans_help_v182/ans_cmd/Hlp_C_CmdTOC.html
   - Do not invent commands, inline functions, KEYOPTs, or argument orderings from memory or from web search results targeting other ANSYS versions.
   - If a command's signature isn't in v18.2, it doesn't exist for this codebase — find a different approach.

2. **`.inp` files must be pure ASCII.** No µ, →, em-dashes, smart quotes, or any non-ASCII glyph — MAPDL silently mis-tokenises them and the failure mode is often a downstream `*GET` returning 0.
   - Python generators that emit APDL **must transcode to ASCII** even when their source is UTF-8 (use `.encode('ascii', 'replace')` or hand-mapped substitutions).
   - When editing existing `.inp` files, do not paste from Word / browser / chat clients without checking — they routinely insert U+2013, U+2014, U+00B5.

3. **All lengths are SI metres at the APDL level.** mm and µm exist only in JSON config and Python visualisation. A factor of 1000 in the wrong place corrupts the entire stack solve and is not caught by APDL.

4. **Entity-numbering offsets** are documented in [memories/repo/apdl-offset-scheme.md](memories/repo/apdl-offset-scheme.md). Capacity: 15 stacks × 50 strands. Reusing numbers across stacks breaks NUMMRG and contact assignment — always check the scheme before introducing new keypoint/line/area numbers.

### Concrete gotchas (do not re-discover these)

- **`NMAX(a,b)` does not exist** — use `*IF,…,GT,…,THEN`. APDL has no scalar max/min function.
- **`LESIZE,…,KYNDIV=1`** lets SmartSize override your divisions; use `KYNDIV=0` to hard-fix NDIV.
- **`SUM` is not a valid `*VOPER` operation** — read the v18.2 `*VOPER` page for the actual operators.
- **Avoid leading-minus arithmetic**: write `-0.015*(ISTEP/10)`, NOT `(ISTEP/10)*-0.015`. The latter is parsed as `(ISTEP/10) * -0.015` differently across MAPDL contexts and has produced silent sign errors.
- **`*VGET,…,NLIST` and `*VGET,…,LOC,X` are misaligned** when a selection is active. Use scalar-cursor `*GET,…,NUM,MIN` + `NXTH` traversal instead (see [8-PP-pressure.inp](scripts/apdl/submodel/cablestack/8-PP-pressure.inp) for the canonical pattern).
- **Inside `*DO` loops with a loop variable** (`*DO,K,1,N`), `array(K)` subscript expansion in `*GET` and `*VWRITE` is unreliable — copy to a scalar first or use the `NXTH` cursor pattern.
- **`RESUME,<jobname>,db`** restores the **entire parameter table** from the `.db`. Any scalar you set **before** `RESUME` is silently overwritten. Re-assert every parameter you need to change **after** the `RESUME` line — see any of the `00-restart-*.inp` decks for the pattern (they only re-assert `usecase`).
- **After a `RESUME`**, BC constraints from the prior run persist. Restart BC files must `DDELE` the prior face's constraint **before** applying a new one (e.g. delete the cyclic UY on `nset_top` before applying SFL pressure there).
- **`btol`** default is 1e-5 m, which exceeds the minimum keypoint distance (~4.5 µm) on deformed strands. [2-geo.inp](scripts/apdl/submodel/cablestack/2-geo.inp) sets `btol,1e-6` at the top before all `ASBA` calls — do not remove this.
- **No global `NUMMRG`** in the cablestack deck; only the auto-generated per-interface `stack_interface_nummrg.inp`. A blanket `NUMMRG,NODE` would merge coincident contact-zone nodes from different strands and break contact pairs.
- **Stale `file.lock` after an abnormal MAPDL exit blocks every subsequent run with rc=100.** When a SLURM job is `scancel`led, or MAPDL crashes/OOMs, the per-jobname `<jobname>.lock` (and the default `file.lock`) is left behind. Any re-launch — including a restart deck that uses a different `/filname` — will hit `*** ERROR *** Another ANSYS job with the same job name (file) is already running ... Do you wish to override this lock and continue (y or n)?` and immediately Abort(100). The `[y/n]` prompt never gets answered in batch. Fix: `rm -f *.lock` (or specifically `file.lock` and `<jobname>.lock`) at the top of any restart slurm script, before module load. Setting `export ANSYS_LOCK=OFF` works too but masks legitimate "already running" cases — prefer explicit `rm`. **Symptom:** every stage in a restart job returns rc=100 within seconds and pp/ stays empty. The generated `jobslurm.sh` (see `cablestack_stages.write_cablestack_jobslurm`) already does `rm -f *.lock` at job start and after any failed stage — keep that behaviour when editing the generator.
- **Generalized plane strain uses `GSGDATA` + `GSBDATA`, NOT `SECTYPE,,GENS`.** `GENS` in `SECTYPE` is "preintegrated general shell section" (a completely different feature) and silently makes `amesh` produce zero elements when combined with `KEYOPT(3)=5`. The real GPS workflow is: `KEYOPT(1,3,5)` → `GSGDATA, LFIBER, XREF, YREF, ROTX0, ROTY0` (coordinates, not node IDs) **before** any `amesh` → `GSBDATA, LabZ, VALUEZ, LabX, VALUEX, LabY, VALUEY` (defaults `F=0, MX=0, MY=0` give free axial / no bending) inside the BC deck. MAPDL auto-creates 2 internal DOF nodes — do not create control nodes yourself. Read results via `GSLIST,RESULTS` (fiber length change = ε_zz × L_fiber). Confirmed compatible with `KEYOPT(6,1)` (mixed u-P) via `scripts/apdl/test/gps_minimal/`.

### When you must change APDL

- **Sanity-check after every edit**: a 30-second `docker compose up` to the first `SOLVE` is cheaper than a 6-hour failed run. Tail the `mapdl_<stage>.log` for `*** ERROR ***` and any line starting with `***` you don't recognise.
- **Preserve sentinel blocks**: `! <<<PRESSURE_CYCLE_BLOCK_START>>>` / `! <<<PRESSURE_CYCLE_BLOCK_END>>>` in `5-BC-pressure.inp` and `5-BC-radial.inp` are patched at runtime by `copy_cablestack_files` — do not delete the comment markers and do not put load-step code inside them by hand.
- **Stay aligned with the four-stage architecture**: if you add a new BC or PP file, register the stage (incl. `usecase_suffix`) in `CABLESTACK_STAGES` in `cablestack_stages.py` — analyse_pressure.py picks the suffix up automatically — and write a `postprocess_<name>` function plus a dispatch entry in `run_cablestack_postprocess`. Do not bolt a new run mode onto an existing stage's `.inp` — it will break the postprocess dispatcher's filename assumptions.
- **Solver settings live in one place**: `/solu` options (`nropt`, `neqit`, `nlgeom`, `AUTOTS`, `DELTIM`, `CUTCONTROL`), the OUTRES policy and the `GSBDATA` call are in [5-BC-solver-settings.inp](scripts/apdl/submodel/cablestack/5-BC-solver-settings.inp), `/INPUT`-included by every `5-BC-*` deck. Tune there, not in the individual decks. The default OUTRES is **selective** (`nsol`+`rsol` every substep, `esol` only at load-step ends — all the PP decks need); set `full_output = 1` in that file to restore `outres,all,all` for field-plot debugging (~10x larger `.rst`).

## d3plottoapdl_package — Critical Invariants

### StrandMesh_Hexa (`strandMeshGenerator.py`)

`generate_mesh()` **must** build inner HexagonMesh first (91 nodes, 72 quad elements), then hex ring layers, then circular transition layers. `core_nodes=91`, `square_size=72`, `circumferential_divisions=36` are constants assumed by `meshMapping.py` and APDL emitters — do not change them. `_template_diameter` uses no scaling factor (a 0.75 factor was removed because it placed outer-ring nodes at 4/3× the physical radius).

### MeshMapping (`meshMapping.py`)

Both `translate_mesh_to_barycenter` and `map_circumferential_layer_to_bspline` use `_polygon_centroid` (shoelace on 500 B-spline samples) — **not** `np.mean(bspline_nodes)` (arc-length-biased). `find_bspline_intersection` uses `bounds=(0, 2)` (outward only); `bounds=(-0.1, 2)` caused cave-in on deformed strands. A pre-scaling step guards against this: if `max_outer_ring_r > min_bspline_r`, scale `mapped_nodes` toward `barycenter_bspline` by `0.99 * min_bspline_r / max_outer_r` before mapping.

### ConformalRutherfordMesh (`conformalRutherfordMesh.py`)

Key invariants for `align_nodes`:
- `outer_orig1, outer_orig2` must be **snapshots** (`.copy()`) — live `mapped_nodes` views are aliased and mutate mid-iteration
- The `while True:` march is **bidirectional** (projects from both strands), not one-sided
- Contact-zone threshold in `identify_contact_region` is `0.015` mm (15 µm)
- The block setting `change1/change2/change11/change22 = True` must appear **before** the `while True:` loop
- 1-vs-2 contact-node special case: move the single node to the closest of the two opposite nodes, then `return` (not `break`)
- Narrow-arc collapse threshold is `0.5 * ref_spacing` (raised from 0.25; the tighter value caused B-spline penetration on 25 µm arcs with 100 µm spacing)
- `identify_contact_region` uses **union** of distance test (≤15 µm via cKDTree) and penetration test (`matplotlib.path.Path.contains_points`) — the penetration test is required; removing it causes visibly short contact arcs on heavily deformed strands

### Cablestack APDL Geometry Notes

- `2-geo.inp` sets `btol,1e-6` at the top (before all ASBA) — default 1e-5 m is above the minimum keypoint distance (~4.5 µm) on deformed strands
- Insulation boolean uses **no `sepo`**; inner-impreg boolean **keeps `sepo`**
- There is no global `NUMMRG` in the cablestack deck — only the auto-generated per-interface `stack_interface_nummrg.inp`
- `4-cont.inp` bonded-contact loop and "bond insulation to impregnation outside" block are commented out — do not re-enable; both conflict with geometry already sharing nodes
- `rotate_outer_layer_nodes()` moves contact-zone nodes to coincident midpoint positions; no concavity smoothing is applied (smoothing would break NUMMRG)

### Element Formulation (PLANE183, single `ET`)

All material areas share `ET,1,PLANE183` defined at the top of `2-geo.inp`. Active KEYOPTs:
- `KEYOPT(1,3,5)` — **generalized plane strain.** Cross-section sees a single uniform out-of-plane strain ε_zz plus two bending DOFs, all carried by 2 auto-allocated internal nodes. Picks up the long-cable z-invariance physically and exposes Nb3Sn ε_zz for Ic prediction.
- `KEYOPT(1,6,1)` — **mixed u-P formulation.** Separate hydrostatic-pressure DOF per element. Required to avoid volumetric locking under copper J2 plasticity (Voce, near-incompressible flow). Confirmed compatible with GPS.

Set up immediately after the KEYOPTs (still in `2-geo.inp`, before any `amesh`):
```
GSGDATA, 1.0, 0, (n_stacks-1)*y_cab, 0, 0
```
LFIBER=1 m means `GSLIST,RESULTS` reports fiber length change numerically equal to ε_zz. Reference point sits at the cable centroid (X=0 by symmetry, Y at the geometric centre of the n_stacks vertical run).

Every BC deck reaches bare `GSBDATA` (no args) via the shared `5-BC-solver-settings.inp` include before its `*do` solve loop — defaults `F=0, MX=0, MY=0` give zero net axial force + no bending = "free Poisson contraction in z" boundary condition matching an unloaded gauge section. To apply Lorentz axial tension later, swap that to `GSBDATA, F, <force_N>, MX, 0, MY, 0`. To prescribe ε_zz directly, use `GSBDATA, LFIBER, <eps_zz>, ROTX, 0, ROTY, 0`.

### Per-Stage BC / Restart Notes

**`5-BC-cyclic.inp` / `5-BC-linear.inp`** — chosen by `bc_type`, used only by `displacement_transverse`. Apply UY ramp on `nset_top` from `myArray` percentages of `2*n_stacks*y_cab`. Bottom/left/right walls fully X- or Y-constrained.

**`5-BC-displacement-radial.inp`** — UX ramp on `nset_left` (mirror of cyclic but in X). After RESUME it must `DDELE, nset_top, all` then re-`D, nset_top, Uy, 0` (locks cable at compacted Y) and `DDELE, nset_left, UX` before applying the cyclic UX. Same myArray ramp shape, scaled by `2*x_cab`. Positive UX on left wall (at `x=-x_cab`) compresses radially.

**`5-BC-pressure.inp`** and **`5-BC-radial.inp`** — share the same patched pressure cycle block delimited by `! <<<PRESSURE_CYCLE_BLOCK_START>>>` / `! <<<PRESSURE_CYCLE_BLOCK_END>>>` sentinels. Pressure stage applies `SFL,all,PRES,…` on `lset_top`; radial stage applies it on `lset_left`. Both must `DDELE` the carried-over displacement constraints on the loaded face before SFL.

**`8-PP-pressure.inp` / `8-PP-radial.inp`** — both use safe `NXTH` cursor traversal for nodal output (avoids `array(loop_var)` subscript expansion issues inside `*DO`). Pressure stage writes `fd_pressure_<usecase>.txt` + `uy_top_<usecase>.txt`; radial writes `fd_radial_<usecase>.txt` + `ux_left_<usecase>.txt` (the `ux_left_*` name is geometric — it's the wall component name, not a load-direction tag).

**`7-PP.inp`** — shared between both displacement stages. Writes the 6-column `fd_good_<usecase>.txt` (Set, Time, UY, FY_total, UX, FX_total) and the `area_summary.txt` material-area breakdown.

**`loading_cycle.json`** (schema_version 2) — written by `copy_cablestack_files`. Contains a `stages` block ({input_file, usecase, depends_on, post_tag} per stage), a per-stage `outputs` map, the `steps` array (nominal pressure schedule used by both pressure stages), the active `formulation`, and an `nb3sn_modulus` audit block (`value_Pa`, `source: 'fallback'`). The same `nb3sn_modulus` block is stamped into the run's `metadata.json`; the fallback is 70 GPa. The v1 `usecases.{pressure, lateral}` short names were replaced by `usecases.{pressure, radial}` plus a richer `stages` block.

## Conventions

- Run folders are immutable history; reruns create new folders or `*_apdl_rerun[_N]/` siblings
- Tool versions are auto-detected from directory names under `tools/` and recorded in `metadata.json`
- Docker project names are lowercase: `lsdyna_<run_folder>`, `mapdl_<run_folder>_<stage_name>` for cablestack stages, and `mapdl_<run_folder>_<strand_dir>` for d3plot→APDL helpers
- Conformal mesh / outer-node SVGs are only written when `--debug-plots` is passed; align-debug SVGs go to `apdl_runfolder/plots/align_debug/` when a diagnostic reason is appended to `_align_debug_reasons`
- Stage / postprocess naming is part of the API contract: when adding a new cablestack stage, register it (incl. `usecase_suffix`) in `CABLESTACK_STAGES` in `cablestack_stages.py`, add a `postprocess_<name>` function, and extend the dispatch table in `run_cablestack_postprocess` (analyse_pressure.py imports the suffix map — no mirroring needed)
