# LTS Rutherford Cable Workflow

End-to-end automated pipeline for **Nb3Sn LTS Rutherford-cable** simulation.
One Python entry point chains:

1. **Cable parameter calculation** (twist pitch, strand layout, geometry).
2. **FreeCAD** headless macro -> STEP geometry.
3. **Ansys Mechanical** (Docker) sweep-hex meshing -> `mesh.k`.
4. **LS-DYNA** (Docker) compaction solve.
5. **ParaView** (`pvpython`) extraction of deformed strand cross-sections.
6. **APDL submodel** -- conformal mesh fitted to deformed strand outlines.
7. **Cablestack solve in MAPDL** -- 2D generalized-plane-strain, four
   independently selectable stages (displacement / pressure x transverse / radial).
8. **Per-stage postprocess** -> stress-strain SVGs + tabular dumps.
9. **Compression box simulation** *(opt-in)* -- 3D parent magnetic box solve ->
   field tables interpolated onto the deformed strand positions -> one-turn 2D
   submodel under the measured BOX9 load/current schedule -> strain + Ic
   degradation analysis.

PLANE183 + `KEYOPT(3)=5` (generalized plane strain) + `KEYOPT(6)=1` (mixed u-P)
means the Nb3Sn axial strain develops naturally from Poisson coupling under
transverse load -- directly feedable into an Ic prediction.

---

## TL;DR

**Containerised run (recommended -- works the same on Linux, macOS, Windows):**

```bash
docker build -t lts-cable .
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD/data:/app/data" \
    -e ANSYS_LICENSE_SERVER=1055@lxlicen01.cern.ch \
    lts-cable --list-cables
```

**Native (Linux/macOS):**

```bash
pip install -e .
export ANSYS_LICENSE_SERVER=1055@lxlicen01.cern.ch   # see "License servers" below
lts-cable --list-cables
```

**Native (Windows / current development environment):**

```powershell
& "C:/Program Files/Python312/python.exe" -m pip install -e .
$env:PYTHONIOENCODING = "utf-8"
& "C:/Program Files/Python312/python.exe" scripts/main/main.py --list-cables
```

`--list-cables` prints the available presets (`R2D2_LF`, `R2D2_HF`, `CD1`) --
those *are* the sample inputs: each is a real Rutherford-cable spec you can run
end-to-end with no further configuration.

---

## Requirements

### Software (host machine)
| Component | Version | Source |
|---|---|---|
| Python | 3.12 | python.org / Microsoft Store / `apt`/`brew` |
| **ANSYS** | 2025 R2 (v252) -- Mechanical APDL, LS-DYNA, Mechanical | Host install + reachable license server |
| **Docker** | Engine 24+ / Desktop 4+ | docker.com |
| FreeCAD | 1.0.2 | Bundled in `tools/freecad/` (Windows) or apt/brew `freecad` (Linux/macOS) |
| ParaView | 6.0.1 | Bundled in `tools/paraview/` (Windows) or apt/brew `paraview` (Linux/macOS) |

The supplied **`Dockerfile`** bundles Python 3.12 + FreeCAD + ParaView + the
Python dependencies, so the only thing you provide is Docker itself, a host
ANSYS license, and (when running cablestack stages) a host ANSYS installation
the MAPDL container can mount.

### Python dependencies (managed by `pyproject.toml`)
`ansys-mechanical-core`, `ansys-dyna-core==0.9.0`, `alphashape`, `matplotlib`,
`networkx`, `numpy`, `pandas`, `python-pptx`, `scipy`, `shapely`.

Install everything in one go: `pip install -e .`

### Docker images used at runtime
- **Meshing / MAPDL:** `<prefix>/mechanical:25.2` (Ansys 2025 R2).
- **LS-DYNA:** `<prefix>/lsdyna:25.2` — see `scripts/lsdyna/docker/docker-compose.yaml`.
- **Registry:** `<prefix>` is set by the `REGISTRY_PREFIX` env var. Default `gitea.psi.ch/vanden_j` (PSI). CERN users set `REGISTRY_PREFIX=registry.cern.ch/chart-magnum`.

---

## License servers

The pipeline auto-detects which ANSYS license server is reachable from the
machine it's running on. Built-in candidates:

| Institute | FlexLM string |
|---|---|
| CERN | `1055@lxlicen01.cern.ch` |
| ETH | `1801@lic-ansys-research.ethz.ch` |
| PSI | `1055@winlic03.psi.ch` |

**Set `ANSYS_LICENSE_SERVER` to skip probing and use a specific server:**

```bash
# Linux / macOS
export ANSYS_LICENSE_SERVER=1055@lxlicen01.cern.ch

# Windows PowerShell
$env:ANSYS_LICENSE_SERVER = "1055@lxlicen01.cern.ch"
```

The pipeline honours this env var verbatim and propagates it into the
LS-DYNA, RVE, and MAPDL container environments (`ANSYSLI_SERVERS` and
`ANSYSLMD_LICENSE_FILE` both get set to it).

**Combine several servers** with `:` for FlexLM failover:

```bash
export ANSYS_LICENSE_SERVER="1055@lxlicen01.cern.ch:1801@lic-ansys-research.ethz.ch"
```

**If you're at a different institute** (not CERN / ETH / PSI), the no-server-
reachable warning at the start of the run tells you exactly what to set. Ask
your IT helpdesk for your site's ANSYS FlexLM host:port. The simplest
permanent fix is to add the line above to your `.bashrc` / `.zshrc` / Windows
user environment variables.

---

## Configuration

Edit [scripts/main/cable_parameters_user.json](scripts/main/cable_parameters_user.json)
to pick a cable preset and tune the cablestack solve. Three sample cables
ship in this file as ready-to-run sample inputs:

| Preset | Configuration |
|---|---|
| `R2D2_LF` | Low-field R2D2 cable |
| `R2D2_HF` | High-field R2D2 cable |
| `CD1` | CD1 cable |

Each preset includes wire material (Nb3Sn / Cu bilinear plasticity), strand
count, pitch, diameters, and cablestack solve settings -- pick one with the
`active_cable` key or pass `-c <NAME>` on the CLI.

Key fields (see the file for the full set with inline `_comment_*` docs):

```jsonc
{
  "active_cable": "CD1",
  "cablestack": {
    "impreg":   4,            // 1=epoxy RT, 2=wax RT, 3=epoxy LN2, 4=wax LN2
    "bc_type":  "cyclic",     // 'cyclic' or 'linear' (displacement_transverse only)

    // Which cablestack stages MAPDL will run after the .inp templates are written.
    // Each entry is a stage name; dependencies are auto-included.
    // Available stages: displacement_transverse, displacement_radial,
    //                   pressure_transverse,     pressure_radial
    // Empty list = generate files only, do not launch MAPDL.
    // --no-cablestack on the CLI overrides this to [].
    "stages": [
      "displacement_transverse",
      "displacement_radial",
      "pressure_transverse",
      "pressure_radial"
    ],

    "mesh_size_um":        50,    // reference impreg+insulation size at D_Strand=0.85mm (scales linearly)
    "strand_mesh_size_um": 50,    // reference strand size at D_Strand=0.85mm (scales linearly)

    "pressure": {                 // applied to BOTH pressure stages
      "gauge_length_mm": 15.0,
      "peak_force_N":    45000,
      "min_force_N":     200,
      "ramp_pressures_MPa": [50.0, 100.0, 150.0]
    }
  }
}
```

---

## Running the pipeline

The Python entry point is `scripts/main/main.py` (also exposed as the
`lts-cable` console script when installed via `pip install -e .`).

```bash
# Linux/macOS (after `pip install -e .`):
lts-cable [OPTIONS]

# Windows:
& "C:/Program Files/Python312/python.exe" scripts/main/main.py [OPTIONS]

# Containerised:
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD/data:/app/data" \
    -e ANSYS_LICENSE_SERVER=$ANSYS_LICENSE_SERVER \
    lts-cable [OPTIONS]
```

Key options:

| Flag | Purpose |
|------|---------|
| `-c {R2D2_LF\|R2D2_HF\|CD1}` | Select cable preset (default `R2D2_LF`) |
| `--cables <NAME> [NAME ...]` | Run multiple cables in parallel (one subprocess per cable) |
| `-t <ms>` | LS-DYNA termination time (default `0.0001`) |
| `--apdl-only` | Copy latest run -> new `*_apdl_rerun/` folder; re-run d3plot->APDL + cablestack |
| `--quick-run` | Skip geometry + meshing; redo mesh-conversion + LS-DYNA |
| `--no-cablestack` | Generate cablestack `.inp` files but **do not launch any MAPDL stages** |
| `--hpc` | Run cablestack on an SSH-reachable SLURM HPC cluster (upload + sbatch + wait + fetch). Default target is ETH Euler; override via `HPC_HOST` / `HPC_USER` / `HPC_REMOTE_BASE` env vars. Requires SSH key access to the chosen cluster. |
| `--debug-plots` | Emit per-pair conformal-mesh / outer-node SVGs (slow) |
| `--compbox` | Run the compression box simulation (step 9) after cablestack, even if `compression_box.enabled` is false |
| `--compbox-only` | Run only the compression box simulation on the latest run folder for the selected cable; add `--hpc` to solve on the cluster |
| `--list-cables` | Print available presets and exit |

**Stage selection is JSON-only:** the `cablestack.stages` array controls
which cablestack stages run. The CLI deliberately has no per-stage flag --
to skip a stage, remove it from the array. To skip them all, use
`--no-cablestack`.

---

## Pipeline at a glance

1. **Parameter calculation** -- strand pitch, twist rate, geometry from the selected preset.
2. **Metadata generation** -- timestamps, tool versions, workflow-step status.
3. **STEP geometry** -- headless FreeCAD writes `<cable>.step`.
4. **Meshing** -- Ansys Mechanical (Docker) sweep-hex meshes the STEP -> `mesh.k`.
5. **Mesh conversion** -- `inputfile_generator` stitches templated `.k` blocks + parsed mesh into `processed_input.k`.
6. **LS-DYNA solve** -- `docker compose up` runs the compaction; log is tailed for live `[<CABLE>] LS-DYNA: 45.2%` progress.
7. **ParaView extraction** -- `pvpython` extracts deformed strand cross-sections from `d3plot` into per-stack CSVs.
8. **APDL submodel build** -- `conformalRutherfordMesh.run(...)` produces the conformal mesh and APDL `.inp` fragments.
9. **Cablestack copy/patch** -- templates copied into `apdl_runfolder/`, all geometry / usecase / pressure variables patched in, `loading_cycle.json` recorded.
10. **Cablestack solve** -- runs every stage from `cablestack.stages` in dependency order (one MAPDL container per stage), with per-stage Python postprocessing on success.

The cablestack uses the **70 GPa Nb3Sn standard** for the strand modulus (stamped in `loading_cycle.json` + `metadata.json` under `nb3sn_modulus`). An RVE-based homogenisation sub-pipeline is in development; see `scripts/apdl/submodel/RVE/README.md`.

---

## Cablestack stage architecture

The cablestack solve is a 2x2 matrix of independent MAPDL stages (plus a
skeleton thermal-cooldown stage that's wired but not implemented):

| Stage name | BC type | Load axis | Driver `.inp` | Output prefix | Usecase suffix |
|---|---|---|---|---|---|
| `build`                   | geometry + mesh + contacts only (no SOLVE) | -- | [0-start.inp](scripts/apdl/submodel/cablestack/0-start.inp) -> SAVE `base.db` | *(none -- produces `base.db` only)* | *(empty)* |
| `displacement_transverse` | UY ramp on top wall (fresh, undeformed) | Y (vertical) | [00-restart-transverse.inp](scripts/apdl/submodel/cablestack/00-restart-transverse.inp) (RESUMEs `base.db`) -> 5-BC.inp -> 7-PP | `fd_good_<cable>.txt` | *(empty)* |
| `displacement_radial`     | UX ramp on left wall (fresh, undeformed) | X (radial)   | [00-restart-radial.inp](scripts/apdl/submodel/cablestack/00-restart-radial.inp) (RESUMEs `base.db`) -> 5-BC-displacement-radial -> 7-PP | `fd_good_<cable>_disp_radial.txt` | `_disp_radial` |
| `pressure_transverse`     | SFL pressure on top wall (fresh, undeformed) | Y (vertical) | [00-restart-pressure.inp](scripts/apdl/submodel/cablestack/00-restart-pressure.inp) (RESUMEs `base.db`) -> 5-BC-pressure -> 8-PP-pressure | `fd_pressure_<cable>_pressure.txt` + `uy_top_...` | `_pressure` |
| `pressure_radial`         | SFL pressure on left wall (fresh, undeformed) | X (radial)   | [00-restart-pressure-radial.inp](scripts/apdl/submodel/cablestack/00-restart-pressure-radial.inp) (RESUMEs `base.db`) -> 5-BC-radial -> 8-PP-radial | `fd_radial_<cable>_radial.txt` + `ux_left_...` | `_radial` |
| `thermal_cooldown` *(SKELETON)* | TUNIF cooldown 293->4.2 K *(TODO)* | -- | [0-start-thermal.inp](scripts/apdl/submodel/cablestack/0-start-thermal.inp) -> 5-BC-thermal -> 8-PP-thermal *(all `/EXIT,NOSAVE` stubs)* | *(none)* | `_thermal` |

> **Strict start/restart split.** `0-start.inp` is the only **build** deck
> (geometry, mesh, contacts, then `SAVE,base,db` -- no BC, no SOLVE). The four
> load stages are pure restart decks that `RESUME,base,db` and apply their own
> BCs from an undeformed configuration. Adding a new load case = adding one
> `00-restart-<name>.inp`, one `5-BC-<name>.inp`, and one entry in
> `CABLESTACK_STAGES`.

> **Free vs. constrained.** `cablestack.boundary_type` in the JSON config
> picks the BC mode for all four load stages. `constrained` (default) pins
> the sidewalls perpendicular to the load (cable inside a rigid die). `free`
> drops those constraints and anchors via the loaded-against wall (full
> `UX=UY=0`), so the cable can bulge under Poisson effect (unconfined Zwick
> test). The patcher overwrites the canonical `5-BC-*.inp` files with their
> `-free` siblings when needed. `bc_type='linear' + boundary_type='free'` is
> not implemented (no `5-BC-linear-free.inp`).

> **Note on the GPS upgrade:** all stages now run PLANE183 with
> `KEYOPT(3)=5` (generalized plane strain) + `KEYOPT(6)=1` (mixed u-P).
> Setup is one `GSGDATA` call in [2-geo.inp](scripts/apdl/submodel/cablestack/2-geo.inp) before meshing,
> and one bare `GSBDATA` call (defaults: free axial, no bending) in each BC
> deck. Results from prior plane-stress runs are **not directly comparable**
> -- the new formulation is stiffer transversely (no z-escape) but reports
> physically correct Nb3Sn epsilon_zz from Poisson coupling.

> **`thermal_cooldown` is a skeleton:** the stage is registered in
> `CABLESTACK_STAGES`, the template files exist, and
> `postprocess_thermal_cooldown` is wired into the dispatcher -- but the
> cooldown physics (CTE per material in `1-material_properties.inp`, `TUNIF`
> load step, Nb3Sn epsilon_zz dump) is *not* implemented. The `.inp` stubs
> `/EXIT,NOSAVE` immediately. Do not add this stage to `cablestack.stages`
> in the JSON until the physics is filled in.

**Architecture (Clean-Architecture style adapters).** Two external-system
boundaries are factored as ports + adapters:

| Port (Protocol) | Adapters | Implementation file |
|---|---|---|
| `CablestackSolver.run_stages(...)` | `LocalMAPDL` (per-stage `docker compose up`), `HPCMAPDL` (one sbatch for all stages on any SSH-reachable SLURM cluster; default ETH Euler; parses rc per-stage from `mapdl_run.log`) | [scripts/main/solver.py](scripts/main/solver.py) |
| `LicenseDetector.detect()` | `NetworkProbeLicenseDetector` (probes CERN/ETH/PSI with 3s TCP timeout, respects `ANSYS_LICENSE_SERVER` env var), `StaticLicenseDetector` (fixed string for tests/CI) | [scripts/main/license_detector.py](scripts/main/license_detector.py) |

If you want to add a new site / cluster / solver, write an adapter that
implements the relevant Protocol and inject it into the `WorkflowRunner`.
You don't need to touch `main.py`.

**Dependencies:** every restart stage depends on the `build` stage because
it `RESUME`s from `base.db`. Missing dependencies are auto-included in the
run order (see `resolve_cablestack_stage_order` in [scripts/main/main.py](scripts/main/main.py)).
If a stage's required `.db` is missing on disk, it's skipped with a warning
rather than launching a failing container.

**One container per stage:** Docker projects are named
`mapdl_<run_folder>_<stage_name>` (lowercase). Per-stage logs go to
`<apdl_runfolder>/mapdl_<stage>.log`.

**Postprocessing fires per stage:** after each stage's MAPDL container exits,
`run_cablestack_postprocess(dst_dir, stage_name=name)` dispatches to the
matching function in [scripts/analysis/submodel/cablestack/analyse_pressure.py](scripts/analysis/submodel/cablestack/analyse_pressure.py):

| Stage | Postprocess function | Outputs |
|---|---|---|
| `displacement_transverse` | `postprocess_displacement_transverse` | `<usecase>_subplots.svg`, `<usecase>_stress_strain.svg`, `<usecase>_stress_strain.txt` |
| `displacement_radial`     | `postprocess_displacement_radial`     | same triple, with `<usecase> = <cable>_disp_radial` |
| `pressure_transverse`     | `postprocess_pressure_transverse`     | same triple, plus uses `loading_cycle.json` nominal-pressure schedule |
| `pressure_radial`         | `postprocess_pressure_radial`         | same triple, with `<usecase> = <cable>_radial` |

All SVGs land in `<apdl_runfolder>/plots/` and the `.txt` exports sit next
to the APDL dumps. The CLI form
`python scripts/analysis/submodel/cablestack/analyse_pressure.py [run_folder]`
runs every stage's postprocess against an existing folder.

---

## Compression box simulation (step 9, opt-in)

Couples the workflow's deformed-strand geometry to the measured BOX9
load/current schedule: a 3D homogenized "compression box" magnetic model
provides the field environment, which is interpolated onto the strand
positions of every stack cross-section; a one-turn 2D submodel then resolves
per-strand strains over the 28-step loading history, and the analysis chain
correlates them against the measured Ic degradation.

Enable with `compression_box.enabled: true` in
[scripts/main/cable_parameters_user.json](scripts/main/cable_parameters_user.json)
(or force per-run with `--compbox`); run standalone on the latest completed
run with `--compbox-only`. `compression_box.solver` picks the backend:
`local` (Docker MAPDL, default) or `hpc` (SLURM on any SSH-reachable cluster; also forced by `--hpc`).

Sub-steps (status in `<run>/APDL/compbox/compbox_summary.json`):
`parent_mag` (box MAG solve -> `.rmg`) -> `vtu_export` -> `field_tables`
-> `submodel` (one case per stack cross-section -> `strains_out_*.out`)
-> `analysis` (strain/Ic CSVs + SVGs).

**Starting a new campaign = dropping in a new measurement table.** The
load/current schedule is generated at staging time from
`compression_box.measurement_file` (a BOX9.txt-style table: peak pressure
[MPa], Ic under load [A], Ic after unload [A or NaN] per row; row 1 is the
baseline) and written identically into the box deck and the submodel BC deck,
so the two can never drift. `compression_box.nsteps` optionally caps the
schedule (even values keep load/unload pairs intact); the expanded schedule is
audited in `<run>/APDL/compbox/loading_schedule.json`.

Note: the submodel's Nb3Sn modulus is the fixed **70 GPa standard** (the value
the box's homogenized-conductor amplification factor is calibrated against).

---

## Output structure

```
data/runs/<timestamp>_<cable>/
|-- cable_parameters.json                     # Calculated parameters
|-- metadata.json                             # Run metadata + workflow_steps status
|-- <cable>.step                              # FreeCAD geometry
|-- LSDYNA/
|   |-- mesh.k                                # Ansys Mechanical output
|   |-- processed_input.k                     # Final LS-DYNA deck
|   |-- d3plot, ...                           # Solver output
|   `-- lsdyna_container.log
|-- stack/                                    # Per-stack deformed-strand CSVs (ParaView)
`-- APDL/
    |-- submodel/
    |   `-- apdl_runfolder/
    |       |-- 0-start.inp, 1-material_properties.inp, ...  # Patched cablestack deck
    |       |-- loading_cycle.json            # Nominal load schedule + per-stage usecase map (schema v2)
    |       |-- base.db, submodel_cable_<n>_<cable>.db, ...
    |       |-- fd_good_<cable>.txt           # displacement_transverse output
    |       |-- fd_good_<cable>_disp_radial.txt
    |       |-- fd_pressure_<cable>_pressure.txt + uy_top_...
    |       |-- fd_radial_<cable>_radial.txt   + ux_left_...
    |       |-- <usecase>_stress_strain.txt   # Python postprocess export (one per stage)
    |       |-- mapdl_<stage>.log             # one per stage launched
    |       `-- plots/<usecase>_subplots.svg, <usecase>_stress_strain.svg
    `-- compbox/                              # step 9 (opt-in) compression box simulation
        |-- parent_runfolder/                 # box MAG deck + CompBox_MAG_*.rmg
        |-- vtu/                              # per-loadstep enhanced VTUs (Bx/By/Bz/|B|)
        |-- field_tables/                     # nb3sn_combined_data_case_<i>_<t>.inp
        |-- submodel_runfolder/               # one-turn decks + geometry + strains_out_*.out
        |-- results/                          # heatmaps/ + strain_analysis/ CSVs + SVGs
        `-- compbox_summary.json              # per-substep status + Nb3Sn modulus audit
```

`workflow_steps` in `metadata.json` tracks `5_lsdyna_simulation`,
`6_paraview_extraction`, `7_apdl_submodel`, `8_cablestack`, and
`9_compression_box` when the opt-in compbox stage runs.
`--apdl-only` uses these keys to find the latest qualifying run.

---

## Troubleshooting

**No ANSYS license server reachable**
- Set `ANSYS_LICENSE_SERVER` to your institute's FlexLM string before re-running
  (see [License servers](#license-servers) above).
- Verify network connectivity / VPN to the server.

**`Docker daemon not running` / Docker errors**
- Ensure Docker Desktop (Windows / macOS) or `dockerd` (Linux) is running.
- Verify `<prefix>/mechanical:25.2` is reachable, where `<prefix>` is `REGISTRY_PREFIX` (default `gitea.psi.ch/vanden_j`; CERN `registry.cern.ch/chart-magnum`).
- Inside the Dockerfile environment, the orchestrator talks to the *host*
  Docker daemon via the mounted socket (`-v /var/run/docker.sock:/var/run/docker.sock`).
  On Windows, replace with `-v //var/run/docker.sock:/var/run/docker.sock`.

**`pvpython not found`**
- Install ParaView (`apt install paraview`, `brew install --cask paraview`,
  or the Windows installer), **or** unzip a portable ParaView under
  `tools/paraview/ParaView*/` so the bundled-lookup finds it, **or** set
  `PVPYTHON_EXE` to point at the binary.

**`FreeCAD executable not found`**
- Install FreeCAD (`apt install freecad`, `brew install --cask freecad`,
  or the Windows installer), **or** unzip a portable FreeCAD under
  `tools/freecad/FreeCAD*/` so the bundled-lookup finds it, **or** set
  `FREECAD_EXE` to point at the binary.

**Cablestack stage skipped with "base.db not found"**
- A restart stage tried to launch before the `build` stage wrote `base.db`.
- Re-run with `build` included in `cablestack.stages` (it's auto-included
  by `resolve_cablestack_stage_order`, so the most common cause is that
  the build stage exited non-zero -- check `mapdl_build.log`).

**`fd_good_*.txt` missing after a successful MAPDL run**
- Open `mapdl_<stage>.log` in the apdl_runfolder -- look for the `7-PP.inp`
  block; non-zero rc on a stage means the stress-strain dump was not written.
- The postprocess function will print `fd_good_... not found; skipping.`
  and return False; the pipeline does not abort.

**Stale `file.lock` after an abnormal exit (HPC / MAPDL crash / scancel)**
- `<jobname>.lock` and `file.lock` linger and block subsequent runs with
  rc=100 within seconds. The HPC jobslurm.sh handles this with
  `rm -f *.lock` at the top; for local re-runs, manually delete them in
  `apdl_runfolder/` before relaunching.

---

## Contact / contributing

Run-by-run state lives under `data/runs/<run>/` -- start there when
debugging. For questions about the pipeline architecture or adding a new
solver / license / site adapter, see the **Ports + adapters** table in the
cablestack section above.
