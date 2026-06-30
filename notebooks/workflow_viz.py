"""Helper module for ``workflow_explorer.ipynb``.

Reads run-folder artefacts produced by ``scripts/main/main.py`` and renders
matplotlib + HTML views.  All heavy solver logic is delegated to
``WorkflowRunner`` — this file only parses/plots/displays, so the notebook
stays in sync with main.py changes automatically.

Importing the module makes ``scripts/main`` and the cablestack analysis
package importable from the notebook.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "data" / "runs"
SCRIPTS_MAIN = REPO_ROOT / "scripts" / "main"
SCRIPTS_ANALYSIS = REPO_ROOT / "scripts" / "analysis" / "submodel" / "cablestack"
CABLE_PARAMS_USER = SCRIPTS_MAIN / "cable_parameters_user.json"

for _p in (SCRIPTS_MAIN, SCRIPTS_ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


STAGE_LABELS = {
    "1_cable_parameters": "Cable parameters",
    "2_freecad_geometry": "FreeCAD STEP geometry",
    "3_mesh_conversion": "Mesh conversion (.k)",
    "5_lsdyna_simulation": "LS-DYNA solve",
    "6_paraview_extraction": "ParaView strand extraction",
    "7_apdl_submodel": "APDL conformal submodel",
    "8_cablestack": "Cablestack 4-stage solve",
    "9_compression_box": "Compression box (opt-in)",
}
STAGE_ORDER = list(STAGE_LABELS.keys())

_STATUS_ICON = {
    "completed": "OK",
    "pending": "...",
    "failed": "X",
    "mesh_completed": "MESH",
}
_STATUS_COLOR = {
    "completed": "#2e7d32",
    "pending": "#888888",
    "failed": "#c62828",
    "mesh_completed": "#ed6c02",
}


def list_cables() -> List[str]:
    with CABLE_PARAMS_USER.open() as f:
        return list(json.load(f).get("cables", {}).keys())


def active_cable_from_user_json() -> Optional[str]:
    try:
        with CABLE_PARAMS_USER.open() as f:
            return json.load(f).get("active_cable")
    except Exception:
        return None


def list_run_folders(cable: str) -> List[Path]:
    """Run folders for ``cable`` (newest first), including ``_apdl_rerun*`` and
    ``_ps`` siblings."""
    if not RUNS_ROOT.exists():
        return []
    pattern = re.compile(rf"_{re.escape(cable)}(_apdl_rerun(_\d+)?)?(_ps)?$")
    matches = [p for p in RUNS_ROOT.iterdir() if p.is_dir() and pattern.search(p.name)]
    return sorted(matches, key=lambda p: p.name, reverse=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_metadata(run_folder: Path) -> dict:
    return read_json(run_folder / "metadata.json")


def read_cable_params(run_folder: Path) -> dict:
    return read_json(run_folder / "cable_parameters.json")


def apdl_runfolder(run_folder: Path) -> Optional[Path]:
    p = run_folder / "APDL" / "submodel" / "apdl_runfolder"
    return p if p.is_dir() else None


def read_loading_cycle(run_folder: Path) -> dict:
    arf = apdl_runfolder(run_folder)
    if arf is None:
        return {}
    return read_json(arf / "loading_cycle.json")


# ─ Stage tracker ─────────────────────────────────────────────────────────────


def stage_status_html(run_folder: Path) -> str:
    """Pretty HTML table of ``workflow_steps`` from ``metadata.json``."""
    meta = read_metadata(run_folder)
    steps = meta.get("workflow_steps", {}) or {}
    rows = []
    for key in STAGE_ORDER:
        if key in ("3_mesh_conversion", "9_compression_box") and key not in steps:
            continue
        status = steps.get(key, "pending")
        label = STAGE_LABELS.get(key, key)
        icon = _STATUS_ICON.get(status, "?")
        color = _STATUS_COLOR.get(status, "#444")
        rows.append(
            "<tr>"
            f"<td style='padding:2px 8px;font-family:monospace;color:{color}'>[{icon}]</td>"
            f"<td style='padding:2px 8px;'>{label}</td>"
            f"<td style='padding:2px 8px;font-family:monospace;color:{color}'>{status}</td>"
            "</tr>"
        )
    cable = meta.get("cable_name", "?")
    timestamp = meta.get("timestamp", "?")
    return (
        f"<div style='font-family:sans-serif'>"
        f"<div><b>Run:</b> <code>{run_folder.name}</code>"
        f" &nbsp; <b>Cable:</b> {cable}"
        f" &nbsp; <b>Started:</b> {timestamp}</div>"
        f"<table style='border-collapse:collapse;margin-top:6px'>"
        + "".join(rows)
        + "</table></div>"
    )


def display_stage_tracker(run_folder: Path) -> None:
    from IPython.display import HTML, display
    display(HTML(stage_status_html(run_folder)))


# ─ Geometry visualisations ──────────────────────────────────────────────────


def plot_cable_schematic(cable_params: dict, ax=None) -> Tuple[plt.Figure, plt.Axes]:
    """Bird's-eye target cross-section: Rutherford rectangle + n strand circles
    laid across the width in two staggered rows.  Derived purely from
    ``cable_parameters.json`` — no solver outputs required."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 2.6))
    else:
        fig = ax.figure
    width = float(cable_params.get("cable_width", 0))
    height = float(cable_params.get("cable_height", 0))
    n = int(cable_params.get("N_Strands", 0))
    d = float(cable_params.get("D_Strand_base", cable_params.get("D_Strand", 0)))
    name = cable_params.get("cable_name", "")

    ax.add_patch(mpatches.Rectangle(
        (-width / 2, -height / 2), width, height,
        linewidth=2, edgecolor="black", facecolor="#f5f5f5"))

    if n > 0 and width > 0 and d > 0:
        n_top = (n + 1) // 2
        n_bot = n - n_top
        if n_top:
            xs_top = np.linspace(-width / 2 + d / 2, width / 2 - d / 2, n_top)
            for x in xs_top:
                ax.add_patch(mpatches.Circle(
                    (x, +height / 4), d / 2,
                    edgecolor="#c0392b", facecolor="#e74c3c",
                    alpha=0.65, linewidth=0.6))
        if n_bot:
            xs_bot = np.linspace(-width / 2 + d / 2, width / 2 - d / 2, n_bot)
            for x in xs_bot:
                ax.add_patch(mpatches.Circle(
                    (x, -height / 4), d / 2,
                    edgecolor="#2c3e50", facecolor="#34495e",
                    alpha=0.65, linewidth=0.6))

    pad_x = max(0.5, 0.05 * width)
    pad_y = max(0.3, 0.30 * height)
    ax.set_aspect("equal")
    ax.set_xlim(-width / 2 - pad_x, width / 2 + pad_x)
    ax.set_ylim(-height / 2 - pad_y, height / 2 + pad_y)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(
        f"{name}: {n} strands, D_strand={d:.3f} mm, "
        f"{width:.2f} x {height:.2f} mm cable (target geometry)")
    ax.grid(True, linestyle=":", alpha=0.4)
    return fig, ax


_STACK_CSV_RE = re.compile(r"Stack_(\d+)_Part(\d+)\.csv$")


def plot_deformed_strands(run_folder: Path, ax=None,
                          stack: Optional[int] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay ``Stack_<S>_Part<P>.csv`` (deflected XY coordinates from
    ParaView) onto the cable rectangle.  Columns are
    ``Deflected Coordinates:0/1/2`` then ``Points:0/1/2``."""
    stack_dir = run_folder / "stack"
    if not stack_dir.exists():
        raise FileNotFoundError(
            f"No 'stack/' directory under {run_folder}. "
            "Run the ParaView extraction step first.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    cp = read_cable_params(run_folder)
    width = float(cp.get("cable_width", 0))
    height = float(cp.get("cable_height", 0))
    if width and height:
        ax.add_patch(mpatches.Rectangle(
            (-width / 2, -height / 2), width, height,
            linewidth=1.5, edgecolor="black",
            facecolor="none", linestyle="--", label="cable outline"))

    csvs = sorted(stack_dir.glob("Stack_*_Part*.csv"))
    cmap = plt.get_cmap("tab20")
    n_plotted = 0
    for csv in csvs:
        m = _STACK_CSV_RE.search(csv.name)
        if not m:
            continue
        s = int(m.group(1))
        p = int(m.group(2))
        if stack is not None and s != stack:
            continue
        try:
            arr = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=(0, 1))
        except Exception:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        ax.plot(arr[:, 0], arr[:, 1], "o", ms=0.7,
                color=cmap(p % 20), alpha=0.75)
        n_plotted += 1

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    title = f"Deformed strands (post-LS-DYNA, z=10 um slice) - {run_folder.name}"
    if stack is not None:
        title += f"  |  stack {stack}"
    ax.set_title(f"{title}\n{n_plotted} strand outlines from {len(csvs)} CSV files")
    ax.grid(True, linestyle=":", alpha=0.4)
    return fig, ax


# ─ Cablestack outputs ────────────────────────────────────────────────────────


_CONFORMAL_MESH_SVGS = {"final_geometry.svg", "hex_overlay.svg"}


def find_cablestack_plots(run_folder: Path) -> Dict[str, List[Path]]:
    """Group SVGs under ``apdl_runfolder/plots/`` by category.

    * ``conformal_mesh`` — outputs from step 7 (``final_geometry.svg``,
      ``hex_overlay.svg``, ``overview_*.svg``)
    * ``postprocess`` — outputs from step 8 ``analyse_pressure`` (everything
      else: ``<cable>[_<suffix>]_stress_strain.svg``,
      ``<cable>[_<suffix>]_subplots.svg``, ``<cable>_displacement_combined.svg``,
      ``<cable>_all_combined.svg``, ...)
    """
    out: Dict[str, List[Path]] = {"conformal_mesh": [], "postprocess": []}
    arf = apdl_runfolder(run_folder)
    if arf is None:
        return out
    plots = arf / "plots"
    if not plots.is_dir():
        return out
    for svg in sorted(plots.glob("*.svg")):
        name = svg.name
        if name in _CONFORMAL_MESH_SVGS or name.startswith("overview_"):
            out["conformal_mesh"].append(svg)
        else:
            out["postprocess"].append(svg)
    return out


def cablestack_stress_strain_txt(run_folder: Path) -> List[Path]:
    """``<usecase>_stress_strain.txt`` exports written by ``analyse_pressure``."""
    arf = apdl_runfolder(run_folder)
    if arf is None:
        return []
    return sorted(arf.glob("*_stress_strain.txt"))


def show_svgs(paths: List[Path], header: str = "") -> None:
    """Render SVGs inline.  Falls back to a 'not yet run' notice if empty."""
    from IPython.display import HTML, SVG, display
    if header:
        display(HTML(f"<h4 style='margin-bottom:4px'>{header}</h4>"))
    if not paths:
        display(HTML(
            "<i>(no plots in this category yet - run the corresponding "
            "stage first)</i>"))
        return
    for p in paths:
        display(HTML(
            f"<div style='font-family:sans-serif;margin-top:6px'>"
            f"<b>{p.name}</b> &nbsp; "
            f"<a href='file:///{p.as_posix()}'>open file</a></div>"))
        try:
            display(SVG(filename=str(p)))
        except Exception as e:
            display(HTML(f"<i>Failed to render {p.name}: {e}</i>"))


def plot_stress_strain(run_folder: Path, ax=None) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay every ``<usecase>_stress_strain.txt`` for a quick numeric view.
    Useful when the SVGs aren't present yet (e.g. notebook running on a
    machine without the analyse_pressure outputs)."""
    txts = cablestack_stress_strain_txt(run_folder)
    if not txts:
        raise FileNotFoundError(
            "No *_stress_strain.txt files under apdl_runfolder. "
            "Either the cablestack stage hasn't finished, or "
            "analyse_pressure hasn't run yet.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    for txt in txts:
        try:
            data = np.loadtxt(txt, comments="#")
        except Exception:
            continue
        if data.ndim != 2 or data.shape[1] < 2:
            continue
        ax.plot(data[:, 0] * 100.0, data[:, 1] / 1e6, "-",
                label=txt.stem.replace("_stress_strain", ""))
    ax.set_xlabel("Strain [%]")
    ax.set_ylabel("Stress [MPa]")
    ax.set_title(f"Cablestack stress-strain - {run_folder.name}")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    return fig, ax


# ─ Light steps (delegate to WorkflowRunner) ─────────────────────────────────


def make_runner():
    """Return a configured ``WorkflowRunner`` rooted at the repo."""
    from main import WorkflowRunner  # noqa: WPS433 (local import is intentional)
    return WorkflowRunner(REPO_ROOT)


def run_cable_parameters(run_folder: Path, termination_time: float = 1e-4) -> Path:
    """Recompute ``cable_parameters.json`` for the active cable.  Cheap."""
    return make_runner().run_cable_parameters(run_folder, termination_time)


def run_postprocess(run_folder: Path, stage_name: Optional[str] = None) -> bool:
    """Re-run ``analyse_pressure`` on an existing apdl_runfolder.  Cheap."""
    arf = apdl_runfolder(run_folder)
    if arf is None:
        print(f"No apdl_runfolder under {run_folder}")
        return False
    return make_runner().run_cablestack_postprocess(arf, stage_name=stage_name)


# ─ Picker widget ─────────────────────────────────────────────────────────────


class RunPicker:
    """Tiny stateful holder around the cable + folder dropdowns.

    Use ``rp.display()`` once at the top of the notebook, then call
    ``rp.run_folder`` / ``rp.cable`` anywhere downstream to read the *current*
    selection.  Re-execute a cell after changing the dropdown to refresh that
    cell's view — no observer callbacks needed.
    """

    def __init__(self, default_cable: Optional[str] = None):
        import ipywidgets as W  # type: ignore

        cables = list_cables()
        if default_cable is None or default_cable not in cables:
            default_cable = active_cable_from_user_json() or (cables[0] if cables else "")

        self._W = W
        self.cable_dd = W.Dropdown(
            options=cables, value=default_cable, description="Cable:",
            layout=W.Layout(width="220px"))
        self.folder_dd = W.Dropdown(
            options=[], description="Run folder:",
            layout=W.Layout(width="55%"))
        self.refresh_btn = W.Button(
            description="Refresh", icon="refresh",
            layout=W.Layout(width="120px"))

        self.cable_dd.observe(lambda _c: self._refresh_folders(), names="value")
        self.refresh_btn.on_click(lambda _b: self._refresh_folders())
        self._refresh_folders()

    def _refresh_folders(self):
        folders = list_run_folders(self.cable_dd.value)
        opts = [(f.name, str(f)) for f in folders]
        self.folder_dd.options = opts
        if opts:
            self.folder_dd.value = opts[0][1]

    @property
    def cable(self) -> str:
        return self.cable_dd.value

    @property
    def run_folder(self) -> Optional[Path]:
        v = self.folder_dd.value
        return Path(v) if v else None

    def display(self):
        from IPython.display import display
        display(self._W.HBox(
            [self.cable_dd, self.folder_dd, self.refresh_btn]))


# ─ Dependency / external-tool checks ─────────────────────────────────────────


# Python packages the notebook + pipeline need at runtime.  Keep aligned with
# pyproject.toml [project.dependencies] + [project.optional-dependencies.notebook].
_REQUIRED_PIP_PKGS = [
    "numpy", "scipy", "pandas", "matplotlib", "shapely", "networkx",
    "alphashape", "pptx",  # python-pptx exposes as 'pptx'
    "ansys.mechanical.core", "ansys.dyna.core",
    "ipywidgets", "IPython",
]


def _module_available(modname: str) -> Tuple[bool, str]:
    """Fast availability probe using importlib.metadata + find_spec.

    Does NOT execute the module, so heavy native-extension packages
    (ansys.mechanical.core, ansys.dyna.core, etc.) don't pay the multi-second
    import-time cost just to be detected.  Returns (ok, version_or_error).
    """
    import importlib.metadata
    import importlib.util
    try:
        spec = importlib.util.find_spec(modname)
    except (ValueError, ModuleNotFoundError) as e:
        return False, f"{type(e).__name__}: {e}"
    if spec is None:
        return False, "not installed"
    # Try to get the version without importing.  importlib.metadata uses the
    # distribution name (e.g. "ansys-mechanical-core" not "ansys.mechanical.core")
    # so we map the most common ones.
    dist_name = _PIP_DIST_NAME.get(modname, modname.replace(".", "-"))
    try:
        ver = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        ver = "installed"
    return True, ver


# Module-name -> pip distribution name for the cases where they differ.
_PIP_DIST_NAME = {
    "pptx": "python-pptx",
    "ansys.mechanical.core": "ansys-mechanical-core",
    "ansys.dyna.core": "ansys-dyna-core",
    "IPython": "ipython",
}


def check_pip_deps() -> Dict[str, Dict[str, str]]:
    """Return ``{pkg: {'ok': bool, 'info': str}}`` for every entry in
    ``_REQUIRED_PIP_PKGS``.  Uses ``find_spec`` for sub-second runtime even
    when the kernel hasn't imported the heavy ANSYS packages yet."""
    out: Dict[str, Dict[str, str]] = {}
    for pkg in _REQUIRED_PIP_PKGS:
        ok, info = _module_available(pkg)
        out[pkg] = {"ok": ok, "info": info}
    return out


def _find_bundled_tool(root_name: str, exe_glob: str) -> Optional[Path]:
    """Search ``REPO_ROOT/tools/<root_name>/**/<exe_glob>`` for the first match."""
    tools_dir = REPO_ROOT / "tools" / root_name
    if not tools_dir.exists():
        return None
    matches = list(tools_dir.rglob(exe_glob))
    return matches[0] if matches else None


def check_external_tools() -> Dict[str, Dict[str, str]]:
    """Probe Docker daemon + bundled FreeCAD/ParaView.  ANSYS license is
    detected on-demand by ``WorkflowRunner`` so we don't probe it here."""
    import shutil
    import subprocess

    status: Dict[str, Dict[str, str]] = {}

    # Docker daemon
    docker_exe = shutil.which("docker")
    if docker_exe is None:
        status["docker"] = {"ok": False, "info": "docker CLI not on PATH"}
    else:
        try:
            r = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                status["docker"] = {"ok": True, "info": f"server {r.stdout.strip()}"}
            else:
                status["docker"] = {"ok": False, "info": "daemon not reachable (start Docker Desktop?)"}
        except Exception as e:
            status["docker"] = {"ok": False, "info": str(e)}

    # FreeCAD (bundled)
    fc = _find_bundled_tool("freecad", "FreeCADCmd*") or _find_bundled_tool("freecad", "freecadcmd*")
    if fc is None:
        # Try system-installed
        fc_sys = shutil.which("FreeCADCmd") or shutil.which("freecadcmd")
        if fc_sys:
            status["freecad"] = {"ok": True, "info": f"system: {fc_sys}"}
        else:
            status["freecad"] = {"ok": False,
                                 "info": f"no FreeCAD under tools/freecad/ nor on PATH"}
    else:
        status["freecad"] = {"ok": True, "info": str(fc)}

    # ParaView (bundled) - look for pvpython
    pv = _find_bundled_tool("paraview", "pvpython*")
    if pv is None:
        pv_sys = shutil.which("pvpython")
        if pv_sys:
            status["paraview"] = {"ok": True, "info": f"system: {pv_sys}"}
        else:
            status["paraview"] = {"ok": False,
                                  "info": "no pvpython under tools/paraview/ nor on PATH"}
    else:
        status["paraview"] = {"ok": True, "info": str(pv)}

    return status


def check_docker_images(registry_prefix: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """Verify the two pipeline Docker images are present locally.  ``None``
    registry_prefix falls through to whatever the docker-compose default is."""
    import subprocess
    prefix = registry_prefix or "gitea.psi.ch/vanden_j"
    images = [f"{prefix}/mechanical:25.2", f"{prefix}/lsdyna:25.2"]
    out: Dict[str, Dict[str, str]] = {}
    for img in images:
        try:
            r = subprocess.run(["docker", "image", "inspect", img],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                # image ID is in JSON; just report 'present'
                out[img] = {"ok": True, "info": "present"}
            else:
                out[img] = {"ok": False, "info": "not pulled"}
        except Exception as e:
            out[img] = {"ok": False, "info": str(e)}
    return out


def _status_row(name: str, ok: bool, info: str) -> str:
    color = "#2e7d32" if ok else "#c62828"
    icon = "OK" if ok else "MISSING"
    return (
        "<tr>"
        f"<td style='padding:2px 8px;font-family:monospace;color:{color}'>[{icon}]</td>"
        f"<td style='padding:2px 8px;font-family:monospace'>{name}</td>"
        f"<td style='padding:2px 8px;font-family:monospace;color:#444'>{info}</td>"
        "</tr>"
    )


def render_status_table(title: str, items: Dict[str, Dict[str, str]]) -> str:
    rows = [_status_row(k, bool(v["ok"]), v["info"]) for k, v in items.items()]
    return (
        f"<div style='font-family:sans-serif;margin-top:6px'>"
        f"<b>{title}</b>"
        f"<table style='border-collapse:collapse;margin-top:4px'>"
        + "".join(rows) + "</table></div>"
    )


def install_missing_pip(extras: str = "notebook", upgrade: bool = False) -> bool:
    """Install/repair the project's pip deps if any are missing in the current
    kernel.

    Returns True iff every required package imports successfully after the
    install attempt (or no install was needed).  Output streams to the caller's
    stdout so it shows up in the notebook cell's Output widget.

    No-op when all packages already import cleanly, so calling this every
    notebook startup is cheap.
    """
    import subprocess
    import sys
    from IPython.display import HTML, display

    before = check_pip_deps()
    missing = [k for k, val in before.items() if not val["ok"]]
    if not missing:
        display(HTML(
            "<div style='font-family:sans-serif;color:#2e7d32'>"
            "All required Python packages already import - nothing to install.</div>"))
        return True

    display(HTML(
        f"<div style='font-family:sans-serif'>"
        f"Missing: <code>{', '.join(missing)}</code> - "
        f"running <code>pip install -e .[{extras}]</code> from <code>{REPO_ROOT}</code></div>"))

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd += ["-e", f"{REPO_ROOT}[{extras}]"]
    rc = subprocess.run(cmd)
    print(f"\npip exit code: {rc.returncode}")

    # Force re-import of every missing module so the next check_pip_deps()
    # call sees the installed packages.
    for modname in missing:
        for cached in list(sys.modules):
            if cached == modname or cached.startswith(modname + "."):
                del sys.modules[cached]

    after = check_pip_deps()
    still_missing = [k for k, val in after.items() if not val["ok"]]
    if still_missing:
        display(HTML(
            f"<div style='font-family:sans-serif;color:#c62828'>"
            f"Still missing after install: <code>{', '.join(still_missing)}</code>. "
            f"Restart the kernel and run the dep check again.</div>"))
        return False

    display(HTML(
        "<div style='font-family:sans-serif;color:#2e7d32'>"
        "All packages now import. Re-running the dep check below to confirm.</div>"))
    return True


def display_environment_check(include_docker_images: bool = False,
                              registry_prefix: Optional[str] = None) -> Dict[str, bool]:
    """Render pip + external-tool status tables.  Returns ``{section: all_ok}``."""
    from IPython.display import HTML, display

    pip_status = check_pip_deps()
    tool_status = check_external_tools()
    display(HTML(render_status_table("Python packages", pip_status)))
    display(HTML(render_status_table("External tools", tool_status)))

    summary = {
        "pip": all(v["ok"] for v in pip_status.values()),
        "tools": all(v["ok"] for v in tool_status.values()),
    }

    if include_docker_images:
        img_status = check_docker_images(registry_prefix)
        display(HTML(render_status_table(
            f"Docker images (prefix={registry_prefix or 'gitea.psi.ch/vanden_j'})",
            img_status)))
        summary["docker_images"] = all(v["ok"] for v in img_status.values())

    return summary


# ─ Run configuration widget (mirrors main.py CLI) ────────────────────────────


class RunConfig:
    """ipywidgets panel mirroring ``main.py``'s CLI flags.

    Each property reads the *current* widget value at call time, so the same
    config object can drive multiple stage cells.

    The HPC backend is generic: any cluster reachable via SSH works.  The
    overrides are the ``HPC_{USER,HOST,REMOTE_BASE}`` env vars (set by
    ``apply_hpc_env`` before launching).
    """

    def __init__(self):
        import ipywidgets as W  # type: ignore
        cables = list_cables()
        default_cable = active_cable_from_user_json() or (cables[0] if cables else "")

        self._W = W

        # — Core run flags —
        self.cable_dd = W.Dropdown(
            options=cables, value=default_cable, description="Cable:",
            layout=W.Layout(width="240px"))
        self.time_ms = W.FloatText(
            value=1e-4, step=1e-5, description="LS-DYNA time [ms]:",
            layout=W.Layout(width="260px"),
            style={"description_width": "140px"})
        self.min_mesh = W.FloatText(
            value=0.0, step=0.01,
            description="Min mesh [mm] (0=auto):",
            layout=W.Layout(width="320px"),
            style={"description_width": "180px"})

        # — Boolean toggles —
        self.no_cablestack = W.Checkbox(value=False, description="--no-cablestack (generate .inp only)")
        self.no_cache = W.Checkbox(value=False, description="--no-cache")
        self.quick_run = W.Checkbox(value=False, description="--quick-run (skip geometry + meshing)")
        self.apdl_only = W.Checkbox(value=False, description="--apdl-only (rerun APDL+cablestack on latest)")
        self.debug_plots = W.Checkbox(value=False, description="--debug-plots")
        self.run_compbox = W.Checkbox(value=False, description="--compbox (compression box stage)")

        # — Docker registry —
        self.registry_prefix = W.Text(
            value="gitea.psi.ch/vanden_j",
            description="REGISTRY_PREFIX:",
            layout=W.Layout(width="420px"),
            style={"description_width": "140px"})

        # — Generic HPC backend —
        self.use_hpc = W.Checkbox(value=False, description="Solve cablestack/compbox on HPC (SSH)")
        self.hpc_host = W.Text(
            value="euler.ethz.ch",
            description="Host:",
            layout=W.Layout(width="320px"))
        self.hpc_user = W.Text(
            value="jvanden",
            description="User:",
            layout=W.Layout(width="240px"))
        self.hpc_remote_base = W.Text(
            value="/cluster/scratch/jvanden/cablestack_runs",
            description="Remote base:",
            layout=W.Layout(width="520px"),
            style={"description_width": "120px"})

        # Layout
        flag_box = W.VBox([
            self.no_cablestack, self.no_cache,
            self.quick_run, self.apdl_only, self.debug_plots, self.run_compbox])
        hpc_box = W.VBox([
            self.use_hpc, self.hpc_host, self.hpc_user, self.hpc_remote_base])
        self._panel = W.VBox([
            W.HBox([self.cable_dd, self.time_ms, self.min_mesh]),
            W.HTML("<b>Pipeline flags</b>"),
            flag_box,
            W.HTML("<b>Docker registry</b>"),
            self.registry_prefix,
            W.HTML("<b>HPC backend (any cluster reachable via SSH)</b>"),
            hpc_box,
        ])

    @property
    def cable(self) -> str:
        return self.cable_dd.value

    @property
    def termination_time(self) -> float:
        return float(self.time_ms.value)

    @property
    def min_mesh_size(self) -> Optional[float]:
        return float(self.min_mesh.value) if self.min_mesh.value else None

    @property
    def kwargs_for_run_workflow(self) -> dict:
        return dict(
            selected_cable=self.cable,
            termination_time=self.termination_time,
            min_mesh_size=self.min_mesh_size,
            quick_run=bool(self.quick_run.value),
            debug_plots=bool(self.debug_plots.value),
            run_cablestack=not bool(self.no_cablestack.value),
            use_hpc=bool(self.use_hpc.value),
            run_compbox=bool(self.run_compbox.value),
        )

    def apply_hpc_env(self) -> Dict[str, str]:
        """Set ``HPC_{USER,HOST,REMOTE_BASE}`` env vars from the widget.
        Returns the env-var snapshot that was applied (for audit)."""
        import os
        applied: Dict[str, str] = {}
        if self.use_hpc.value:
            applied = {
                "HPC_HOST": self.hpc_host.value,
                "HPC_USER": self.hpc_user.value,
                "HPC_REMOTE_BASE": self.hpc_remote_base.value,
            }
            for k, val in applied.items():
                os.environ[k] = val
        if self.registry_prefix.value:
            os.environ["REGISTRY_PREFIX"] = self.registry_prefix.value
            applied["REGISTRY_PREFIX"] = self.registry_prefix.value
        return applied

    def display(self):
        from IPython.display import display
        display(self._panel)


# ─ Container abort (LS-DYNA + cablestack MAPDL containers) ───────────────────


def list_run_containers(run_folder: Path) -> Dict[str, List[str]]:
    """Return docker compose project names belonging to a run folder.

    Naming conventions:
    * LS-DYNA: ``lsdyna_<run_folder_name>`` (lowercase)
    * Cablestack MAPDL: ``mapdl_<run_folder_name>_<stage_name>`` (lowercase)
    * Compbox MAPDL: ``mapdl_<run_folder_name>_compbox_<tag>``
    * Meshing: ``ansys_mechanical_<run_folder_name>``

    Returned dict keys: ``lsdyna``, ``mapdl``, ``mesher``. Values are project
    names actually running on the local Docker daemon (so the caller can
    decide what to abort).  Empty dict if the docker CLI is not available.
    """
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return {}

    folder_lower = run_folder.name.lower()
    try:
        r = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}|{{.Labels}}"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return {}
    if r.returncode != 0:
        return {}

    projects: Dict[str, List[str]] = {"lsdyna": [], "mapdl": [], "mesher": []}
    seen: set = set()
    # `docker ps --format {{.Labels}}` exposes the compose project name in the
    # `com.docker.compose.project` label.  Parse that to dedupe per-project.
    for line in r.stdout.splitlines():
        if "|" not in line:
            continue
        _name, labels = line.split("|", 1)
        project_label = None
        for kv in labels.split(","):
            if kv.startswith("com.docker.compose.project="):
                project_label = kv.split("=", 1)[1].strip()
                break
        if not project_label or project_label in seen:
            continue
        seen.add(project_label)
        if project_label.startswith(f"lsdyna_{folder_lower}"):
            projects["lsdyna"].append(project_label)
        elif project_label.startswith(f"mapdl_{folder_lower}"):
            projects["mapdl"].append(project_label)
        elif project_label.startswith(f"ansys_mechanical_{folder_lower}"):
            projects["mesher"].append(project_label)
    return projects


def stop_run_containers(run_folder: Path) -> Dict[str, int]:
    """``docker compose -p <project> down`` for every container project tied
    to this run folder.  Returns ``{project: exit_code}``.  Safe to call when
    nothing is running (returns an empty dict).
    """
    import subprocess

    projects = list_run_containers(run_folder)
    results: Dict[str, int] = {}
    for category, project_names in projects.items():
        for proj in project_names:
            print(f"docker compose -p {proj} down  ({category})")
            try:
                r = subprocess.run(
                    ["docker", "compose", "-p", proj, "down", "--remove-orphans"],
                    capture_output=True, text=True, timeout=60)
                results[proj] = r.returncode
                if r.stdout.strip():
                    print(r.stdout)
                if r.returncode != 0 and r.stderr.strip():
                    print(r.stderr)
            except Exception as e:
                print(f"  failed: {e}")
                results[proj] = -1
    if not results:
        print(f"No running containers found for run folder '{run_folder.name}'.")
    return results


def container_abort_commands(run_folder: Optional[Path] = None) -> str:
    """Human-readable shell commands for manually stopping each container
    category.  Useful for the docs section + when the helper button can't
    reach the Docker daemon."""
    fname = run_folder.name.lower() if run_folder else "<run_folder_name>"
    return (
        f"# LS-DYNA solve container (detached after launch)\n"
        f"docker compose -p lsdyna_{fname} down\n"
        f"\n"
        f"# Cablestack MAPDL stage containers (one per parallel stage)\n"
        f"docker compose -p mapdl_{fname}_build down\n"
        f"docker compose -p mapdl_{fname}_displacement_transverse down\n"
        f"docker compose -p mapdl_{fname}_displacement_radial down\n"
        f"docker compose -p mapdl_{fname}_pressure_transverse down\n"
        f"docker compose -p mapdl_{fname}_pressure_radial down\n"
        f"\n"
        f"# Ansys Mechanical mesher\n"
        f"docker compose -p ansys_mechanical_{fname} down\n"
        f"\n"
        f"# Compression box (opt-in)\n"
        f"docker compose -p mapdl_{fname}_compbox_parent_mag down\n"
        f"docker compose -p mapdl_{fname}_compbox_submodel down\n"
        f"\n"
        f"# To find every container for this run:\n"
        f"docker ps --filter label=com.docker.compose.project --format \\\n"
        f"  '{{{{.Names}}}}  {{{{.Labels}}}}' | grep '{fname}'\n"
    )


# ─ Interactive cable designer ────────────────────────────────────────────────


def rutherford_natural_width_mm(n_strands: int, d_strand_mm: float) -> float:
    """Natural (uncompressed) Rutherford width for ``n_strands`` round wires
    of diameter ``d_strand_mm``.  Matches
    ``calc_cable_params_sim._Weff + D_Strand_base``: minimum-touching strand
    pitch times the per-row count, plus one strand diameter for the rounded
    ends.  Used as the lower bound on ``cable_width`` in the designer.
    """
    if n_strands < 2 or d_strand_mm <= 0:
        return 0.0
    import math
    pitch_min = math.pi * d_strand_mm * (n_strands - 2) / (2 * n_strands)
    return (n_strands * pitch_min / 2.0) + d_strand_mm


def centered_hex_numbers(max_value: int = 1000) -> List[int]:
    """Centered hexagonal numbers H(k) = 3k^2 - 3k + 1 up to ``max_value``:
    1, 7, 19, 37, 61, 91, 127, 169, 217, 271, ...

    These are the only subelement counts that tile a *complete* hexagonal
    billet stack (a central rod plus k-1 full rings of 6(k-1) rods each), so
    both N_TOTAL and N_NB3SN must be one of them.  With N_NB3SN <= N_TOTAL and
    both centered-hex, the difference H(m) - H(j) is automatically the sum of
    whole outer rings j+1..m -- i.e. the Cu stabiliser always forms complete
    rings.
    """
    out: List[int] = []
    k = 1
    while True:
        h = 3 * k * k - 3 * k + 1
        if h > max_value:
            return out
        out.append(h)
        k += 1


def centered_hex_rings(count: int) -> Optional[int]:
    """Ring count k such that H(k) == ``count`` (k=1 is the lone central rod),
    or ``None`` if ``count`` is not a centered hexagonal number."""
    k = 1
    while True:
        h = 3 * k * k - 3 * k + 1
        if h == count:
            return k
        if h > count:
            return None
        k += 1


def snap_to_centered_hex(value: int, options: Optional[List[int]] = None) -> int:
    """Nearest centered hexagonal number to ``value`` (ties round up).

    When ``options`` is omitted the candidate list is generated up to
    ``2*value + 7`` so the number *above* ``value`` is always included --
    otherwise the snap could only ever round down.
    """
    opts = options if options is not None \
        else centered_hex_numbers(2 * max(value, 1) + 7)
    return min(opts, key=lambda h: (abs(h - value), -h))


class CableDesigner:
    """Live cable-design widget panel.

    Inputs (top-level cable + wire/RVE sub-block) come from ipywidgets sliders
    and number boxes.  Min/max are derived from physical constraints and
    re-clipped whenever a related field changes:

    * ``cable_width >= W_rutherford(n_strands, D_strand)`` -- you can't squash
      narrower than where the round strands just touch in their natural row.
    * ``D_strand < cable_height < 2 * D_strand`` -- Rutherford keystone range:
      below 1 strand diameter is impossible, above 2 means you'd fit a third
      row instead of compressing two.

    A red warning banner above the plot lights up when cross-field combos go
    outside the recommended range; the user can still see what they typed.

    The plot shows two overlapping representations on the same axes:

    * Pre-compaction (LS-DYNA initial state): two rows of ``n_strands/2``
      circles each, at the natural Rutherford spacing.
    * Post-compaction (final): a dashed rectangle ``cable_width x cable_height``
      that the LS-DYNA solve compresses the strands into.

    Save behaviour: writes ``cables.<NAME>`` plus the wire sub-block back into
    ``scripts/main/cable_parameters_user.json``, sets ``active_cable``, and
    backs the existing JSON up to ``*.bak``.
    """

    # Sensible bounds for slider min/max -- enough headroom for unusual
    # designs without being silly.
    BOUNDS = {
        "n_strands":    (4, 60, 1),         # min, max, step (existing presets have N=21 odd)
        "d_strand":     (0.3, 1.5, 0.01),
        "t_pitch":      (30.0, 200.0, 1.0),
        "n_stacks":     (1, 20, 1),
        "n_nb3sn":      (10, 300, 1),
        "cu_noncu":     (0.3, 3.0, 0.05),
        "d_core_eq":    (20.0, 100.0, 0.5),  # um
        "nb_thick":     (0.5, 5.0, 0.1),     # um
        "bronze_frac":  (0.30, 0.70, 0.01),
        "cu_sleeve":    (0.5, 15.0, 0.1),    # um
        "n_strands_default": 21,
        "d_strand_default":  0.85,
        "n_total_max":       1000,    # largest centered-hex billet offered (H<=1000 -> 919)
    }

    def __init__(self, name: str = "NEW_CABLE",
                 initial_from: Optional[str] = None):
        import ipywidgets as W  # type: ignore
        import matplotlib.pyplot as plt
        self._W = W
        self._plt = plt

        seed_cable, seed_wire = self._seed_values(initial_from)

        # ---- Top-level cable widgets ----
        self.name_w = W.Text(
            value=name, description="Preset name:",
            layout=W.Layout(width="320px"),
            style={"description_width": "120px"})
        self.n_strands_w = W.IntSlider(
            value=seed_cable.get("N_Strands", self.BOUNDS["n_strands_default"]),
            min=self.BOUNDS["n_strands"][0], max=self.BOUNDS["n_strands"][1],
            step=self.BOUNDS["n_strands"][2],
            description="n_strands:", continuous_update=False,
            layout=W.Layout(width="420px"),
            style={"description_width": "120px"})
        self.d_strand_w = W.FloatSlider(
            value=seed_cable.get("D_Strand", self.BOUNDS["d_strand_default"]),
            min=self.BOUNDS["d_strand"][0], max=self.BOUNDS["d_strand"][1],
            step=self.BOUNDS["d_strand"][2],
            description="D_Strand [mm]:", continuous_update=False,
            readout_format=".3f",
            layout=W.Layout(width="420px"),
            style={"description_width": "120px"})
        # cable_width and cable_height min/max get re-set live by _refresh_bounds.
        # Initial values aim for packing factor ~ 0.88 (typical production
        # Rutherford) so the user starts in a physically realistic spot.
        import math as _math
        _n_init = int(seed_cable.get("N_Strands", self.BOUNDS["n_strands_default"]))
        _d_init = float(seed_cable.get("D_Strand", self.BOUNDS["d_strand_default"]))
        _h_init = float(seed_cable.get("cable_height", round(1.55 * _d_init, 3)))
        _a_strands = _n_init * _math.pi * (_d_init ** 2) / 4.0
        _w_init = float(seed_cable.get(
            "cable_width",
            round(_a_strands / (0.88 * _h_init), 3)))
        self.cable_width_w = W.FloatSlider(
            value=_w_init, min=1.0, max=30.0, step=0.01,
            description="cable_width [mm]:", continuous_update=False,
            readout_format=".3f",
            layout=W.Layout(width="420px"),
            style={"description_width": "120px"})
        self.cable_height_w = W.FloatSlider(
            value=_h_init, min=0.1, max=3.0, step=0.005,
            description="cable_height [mm]:", continuous_update=False,
            readout_format=".3f",
            layout=W.Layout(width="420px"),
            style={"description_width": "120px"})
        self.t_pitch_w = W.FloatSlider(
            value=seed_cable.get("T_pitch", 79.0),
            min=self.BOUNDS["t_pitch"][0], max=self.BOUNDS["t_pitch"][1],
            step=self.BOUNDS["t_pitch"][2],
            description="T_pitch [mm]:", continuous_update=False,
            readout_format=".1f",
            layout=W.Layout(width="420px"),
            style={"description_width": "120px"})
        self.n_stacks_w = W.IntSlider(
            value=seed_cable.get("n_stacks", 6),
            min=self.BOUNDS["n_stacks"][0], max=self.BOUNDS["n_stacks"][1],
            step=self.BOUNDS["n_stacks"][2],
            description="n_stacks:", continuous_update=False,
            layout=W.Layout(width="420px"),
            style={"description_width": "120px"})

        # ---- Wire / RVE sub-block ----
        # The subelement billet must tile a complete hexagon, so N_TOTAL and
        # N_NB3SN are restricted to centered hexagonal numbers (1, 7, 19, 37,
        # 61, 91, 127, ...).  Snap any imported seed to the nearest valid stack
        # so a preset can never start in a non-stackable state.  N_NB3SN's
        # options are capped at N_TOTAL (see _refresh_bounds), which keeps the
        # Cu remainder = whole outer rings by construction.
        self._hex_opts = centered_hex_numbers(self.BOUNDS["n_total_max"])
        _seed_total = snap_to_centered_hex(
            int(seed_wire.get("N_TOTAL", 127)), self._hex_opts)
        _seed_nb3sn = snap_to_centered_hex(
            int(seed_wire.get("N_NB3SN", 91)), self._hex_opts)
        if _seed_nb3sn > _seed_total:
            _seed_nb3sn = _seed_total

        self.n_nb3sn_w = W.Dropdown(
            options=[(self._hex_label(h), h)
                     for h in self._hex_opts if h <= _seed_total],
            value=_seed_nb3sn,
            description="N_NB3SN:",
            layout=W.Layout(width="420px"),
            style={"description_width": "150px"})
        self.cu_noncu_w = W.FloatSlider(
            value=seed_wire.get("CU_NONCU", 1.2),
            min=self.BOUNDS["cu_noncu"][0], max=self.BOUNDS["cu_noncu"][1],
            step=self.BOUNDS["cu_noncu"][2],
            description="CU_NONCU:", continuous_update=False,
            readout_format=".2f",
            layout=W.Layout(width="420px"),
            style={"description_width": "150px"})
        self.d_core_eq_w = W.FloatSlider(
            value=seed_wire.get("D_CORE_EQ_UM", 55.0),
            min=self.BOUNDS["d_core_eq"][0], max=self.BOUNDS["d_core_eq"][1],
            step=self.BOUNDS["d_core_eq"][2],
            description="D_CORE_EQ [um]:", continuous_update=False,
            readout_format=".1f",
            layout=W.Layout(width="420px"),
            style={"description_width": "150px"})
        self.nb_thick_w = W.FloatSlider(
            value=seed_wire.get("NB_THICKNESS_UM", 2.0),
            min=self.BOUNDS["nb_thick"][0], max=self.BOUNDS["nb_thick"][1],
            step=self.BOUNDS["nb_thick"][2],
            description="NB_THICKNESS [um]:", continuous_update=False,
            readout_format=".1f",
            layout=W.Layout(width="420px"),
            style={"description_width": "150px"})
        self.bronze_frac_w = W.FloatSlider(
            value=seed_wire.get("BRONZE_FRAC", 0.55),
            min=self.BOUNDS["bronze_frac"][0], max=self.BOUNDS["bronze_frac"][1],
            step=self.BOUNDS["bronze_frac"][2],
            description="BRONZE_FRAC:", continuous_update=False,
            readout_format=".2f",
            layout=W.Layout(width="420px"),
            style={"description_width": "150px"})
        self.cu_sleeve_w = W.FloatSlider(
            value=seed_wire.get("CU_SLEEVE_THICKNESS_UM", 5.94),
            min=self.BOUNDS["cu_sleeve"][0], max=self.BOUNDS["cu_sleeve"][1],
            step=self.BOUNDS["cu_sleeve"][2],
            description="CU_SLEEVE [um]:", continuous_update=False,
            readout_format=".2f",
            layout=W.Layout(width="420px"),
            style={"description_width": "150px"})
        self.n_total_w = W.Dropdown(
            options=[(self._hex_label(h), h) for h in self._hex_opts],
            value=_seed_total,
            description="N_TOTAL:",
            layout=W.Layout(width="240px"),
            style={"description_width": "80px"})

        # ---- Status / output / save ----
        self.warn_box = W.HTML(value="")
        self.plot_out = W.Output(layout=W.Layout(border="1px solid #ccc"))
        self.json_out = W.HTML(value="")
        self.save_btn = W.Button(
            description="Save preset to JSON",
            button_style="success",
            layout=W.Layout(width="240px"))
        self.save_status = W.HTML(value="<i>not saved yet</i>")
        self.save_btn.on_click(self._on_save)

        # ---- Observers (live update) ----
        for w in self._all_widgets():
            w.observe(self._on_change, names="value")

        # Initial bounds + render.
        self._refresh_bounds(initial=True)
        self._render()

        # ---- Final layout ----
        cable_box = W.VBox([
            W.HTML("<b>Top-level cable parameters</b>"),
            self.name_w,
            self.n_strands_w, self.d_strand_w,
            self.cable_width_w, self.cable_height_w,
            self.t_pitch_w, self.n_stacks_w,
        ])
        wire_box = W.VBox([
            W.HTML("<b>Wire / RVE sub-block</b>"),
            self.n_nb3sn_w, self.n_total_w,
            self.cu_noncu_w, self.d_core_eq_w,
            self.nb_thick_w, self.bronze_frac_w,
            self.cu_sleeve_w,
        ])
        self._panel = W.VBox([
            W.HBox([cable_box, wire_box]),
            self.warn_box,
            self.plot_out,
            W.HTML("<b>cable_parameters_user.json preview</b>"),
            self.json_out,
            W.HBox([self.save_btn, self.save_status]),
        ])

    # ---- Helpers ----

    def _all_widgets(self):
        return [self.name_w, self.n_strands_w, self.d_strand_w,
                self.cable_width_w, self.cable_height_w,
                self.t_pitch_w, self.n_stacks_w,
                self.n_nb3sn_w, self.n_total_w,
                self.cu_noncu_w, self.d_core_eq_w,
                self.nb_thick_w, self.bronze_frac_w, self.cu_sleeve_w]

    def _seed_values(self, name: Optional[str]):
        """Pre-fill widget values from an existing cable preset if provided."""
        if name is None:
            return {}, {}
        try:
            with CABLE_PARAMS_USER.open() as f:
                cfg = json.load(f)
        except Exception:
            return {}, {}
        c = cfg.get("cables", {}).get(name)
        if not c:
            return {}, {}
        return c, c.get("wire", {})

    def _on_change(self, change):
        # Re-clip bounds (e.g. when n_strands or D_strand changed, the
        # natural Rutherford width changes, so cable_width's min has to
        # follow), then re-render plot + JSON + warnings.
        self._refresh_bounds(initial=False)
        self._render()

    def _refresh_bounds(self, initial: bool):
        """Re-clip cable_width and cable_height widget ranges to physically
        reasonable values given the current n_strands + D_strand.

        Constraints (post-compaction rectangle):
        * cable_height in (D_strand, 2*D_strand) - Rutherford keystone range:
          below 1*D is physically impossible (strand can't be flatter than
          its own diameter), above 2*D means a third row would fit.
        * cable_width.min derived from area conservation: at packing factor
          0.98 (very tight) the rectangle is just large enough to hold the
          total strand area at the chosen height.
        * cable_width.max: at packing factor 0.50 (very loose), so the user
          has room to explore designs with non-trivial void space.

        We update min/max but only clip the current widget value when it
        falls outside the new band -- the "soft clip" UX.
        """
        import math
        n = self.n_strands_w.value
        d = self.d_strand_w.value
        a_strands = n * math.pi * (d ** 2) / 4.0

        # Height bounds first, then width bounds depend on the new height.
        h_lo = max(0.05, 1.0 * d)
        h_hi = max(h_lo + 0.01, 2.0 * d)
        if h_lo > self.cable_height_w.max:
            self.cable_height_w.max = h_hi
            self.cable_height_w.min = h_lo
        else:
            self.cable_height_w.min = h_lo
            self.cable_height_w.max = h_hi
        if self.cable_height_w.value < h_lo or self.cable_height_w.value > h_hi:
            self.cable_height_w.value = round((h_lo + h_hi) / 2.0, 3)

        h_now = self.cable_height_w.value
        # Width lower bound: packing factor <= 0.98 at this height.
        w_lo = max(0.5, a_strands / 0.98 / max(h_now, 1e-3))
        # Width upper bound: packing factor >= 0.50 (loose).
        w_hi = max(w_lo + 0.5, a_strands / 0.50 / max(h_now, 1e-3))
        if w_lo > self.cable_width_w.max:
            self.cable_width_w.max = w_hi
            self.cable_width_w.min = w_lo
        else:
            self.cable_width_w.min = w_lo
            self.cable_width_w.max = w_hi
        if self.cable_width_w.value < w_lo:
            self.cable_width_w.value = round(w_lo * 1.02, 3)
        elif self.cable_width_w.value > w_hi:
            self.cable_width_w.value = round(w_hi * 0.98, 3)

        # Hex subelement stack: N_NB3SN can only be a centered-hex count <=
        # N_TOTAL.  Re-clip its options whenever N_TOTAL changes, then clamp
        # the current value down to N_TOTAL if it now exceeds it.  (N_TOTAL is
        # itself a centered-hex number, so the largest allowed N_NB3SN == it.)
        total = int(self.n_total_w.value)
        allowed = [h for h in self._hex_opts if h <= total]
        if getattr(self, "_nb3sn_allowed", None) != allowed:
            self._nb3sn_allowed = list(allowed)
            keep = int(self.n_nb3sn_w.value)
            self.n_nb3sn_w.options = [(self._hex_label(h), h) for h in allowed]
            self.n_nb3sn_w.value = keep if keep <= total else total

    def _packing_factor(self) -> float:
        """Strand-area fill of the squashed cross-section.

        ``packing = N * pi * D^2 / 4 / (cable_width * cable_height)``.
        Typical Rutherford: 0.85 - 0.95.  > 1 means strands cannot fit
        without overlap; < 0.70 means a lot of void space (loose squash).
        """
        import math
        a_strands = self.n_strands_w.value * math.pi * \
            (self.d_strand_w.value ** 2) / 4.0
        a_box = self.cable_width_w.value * self.cable_height_w.value
        return float("inf") if a_box <= 0 else a_strands / a_box

    @staticmethod
    def _hex_label(h: int) -> str:
        """Dropdown label for a centered-hex count, e.g. '127  (7 rings)'."""
        k = centered_hex_rings(h)
        return f"{h}  ({k} rings)" if k else str(h)

    def _hex_stack_summary(self) -> str:
        """One-line decomposition of the subelement billet into Nb3Sn core +
        whole Cu outer rings, e.g.
        '127-rod billet (7 rings) = 91 Nb3Sn (6-ring core) + 36 Cu (ring 7).'"""
        total = int(self.n_total_w.value)
        nb = int(self.n_nb3sn_w.value)
        kt = centered_hex_rings(total)
        kn = centered_hex_rings(nb)
        if kt is None or kn is None:
            return ""
        cu = total - nb
        if cu == 0:
            return f"{total}-rod billet ({kt} rings), all Nb3Sn, no Cu ring."
        lo, hi = kn + 1, kt
        rings = f"ring {lo}" if lo == hi else f"rings {lo}-{hi}"
        return (f"{total}-rod billet ({kt} rings) = {nb} Nb3Sn ({kn}-ring core) "
                f"+ {cu} Cu ({rings}).")

    def _violations(self) -> List[str]:
        """Hard physical violations.  Soft warnings are emitted by
        ``_warnings`` separately."""
        out: List[str] = []
        n = self.n_strands_w.value
        d = self.d_strand_w.value
        h = self.cable_height_w.value
        pf = self._packing_factor()

        # The cable cross-section must hold all the strand area.  Real
        # Rutherford cables have some plastic deformation void compaction,
        # but pf > 1 means the squashed rectangle is physically too small.
        if pf > 1.0:
            out.append(
                f"Strand cross-section area > cable rectangle "
                f"(packing factor = {pf:.3f} > 1.00). "
                f"N*pi*D^2/4 must fit inside cable_width * cable_height.")
        if h <= d:
            out.append(
                f"cable_height={h:.3f} mm <= D_strand={d:.3f} mm - "
                f"physically impossible to compress below one strand diameter.")
        if h >= 2.0 * d:
            out.append(
                f"cable_height={h:.3f} mm >= 2*D_strand={2*d:.3f} mm - "
                f"the cable is taller than two stacked strands; you would have room "
                f"for a third row instead of squashing the two.")
        return out

    def _warnings(self) -> List[str]:
        out: List[str] = []
        n = self.n_strands_w.value
        n_nb3sn = self.n_nb3sn_w.value
        n_total = self.n_total_w.value
        pf = self._packing_factor()

        if pf > 0.95:
            out.append(
                f"Tight squash: packing factor = {pf:.3f} (> 0.95). "
                f"Production Rutherford typically sits at 0.85-0.92.")
        elif pf < 0.70:
            out.append(
                f"Loose squash: packing factor = {pf:.3f} (< 0.70). "
                f"Lots of void space - strands won't pack densely.")
        if n_nb3sn > n_total:
            out.append(
                f"N_NB3SN={n_nb3sn} > N_TOTAL={n_total} - bronze + Nb3Sn cannot exceed total hex count.")
        if n_total > 0 and n_nb3sn < 0.5 * n_total:
            out.append(
                f"N_NB3SN/N_TOTAL = {n_nb3sn/n_total:.2f} is < 0.5 - unusually low Nb3Sn fill.")
        return out

    def _render(self):
        # Clear the existing plot and redraw.
        self.plot_out.clear_output(wait=True)
        with self.plot_out:
            self._draw_plot()
        self._render_warnings()
        self._render_json()

    def _draw_plot(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        n = int(self.n_strands_w.value)
        d = float(self.d_strand_w.value)
        w_target = float(self.cable_width_w.value)
        h_target = float(self.cable_height_w.value)
        w_nat = rutherford_natural_width_mm(n, d)

        fig, ax = plt.subplots(figsize=(11, 3.2))

        # ---- Pre-compaction (LS-DYNA initial natural Rutherford layout) ----
        # Two rows of n/2 strands.  Top row centres at y = +d/2, bottom at -d/2.
        # X positions are evenly spaced across w_nat so the outermost strands sit
        # half a diameter in from the edges.
        n_top = (n + 1) // 2
        n_bot = n - n_top
        if n_top:
            xs_top = np.linspace(-w_nat/2 + d/2, w_nat/2 - d/2, n_top)
            for x in xs_top:
                ax.add_patch(mpatches.Circle(
                    (x, +d/2), d/2,
                    facecolor="#bdd7ee", edgecolor="#1f4e79", linewidth=1.0))
        if n_bot:
            xs_bot = np.linspace(-w_nat/2 + d/2, w_nat/2 - d/2, n_bot)
            for x in xs_bot:
                ax.add_patch(mpatches.Circle(
                    (x, -d/2), d/2,
                    facecolor="#bdd7ee", edgecolor="#1f4e79", linewidth=1.0))

        # Annotate natural-width bounding box (light blue dashed).
        ax.add_patch(mpatches.Rectangle(
            (-w_nat/2, -d), w_nat, 2*d,
            linewidth=1.0, edgecolor="#1f4e79", facecolor="none",
            linestyle=":", label=f"natural Rutherford (w={w_nat:.2f}, h={2*d:.2f})"))

        # ---- Post-compaction (squashed boundary) ----
        ax.add_patch(mpatches.Rectangle(
            (-w_target/2, -h_target/2), w_target, h_target,
            linewidth=2.2, edgecolor="#c62828", facecolor="none",
            linestyle="--",
            label=f"squashed target (w={w_target:.2f}, h={h_target:.2f})"))

        # Strand-diameter reference (1*D and 2*D horizontal guides).
        ax.axhline(+d/2, color="#666", linestyle=":", linewidth=0.5)
        ax.axhline(-d/2, color="#666", linestyle=":", linewidth=0.5)
        ax.axhline(+d, color="#999", linestyle=":", linewidth=0.5)
        ax.axhline(-d, color="#999", linestyle=":", linewidth=0.5)

        ax.set_aspect("equal")
        pad = max(0.8, 0.06 * max(w_nat, w_target))
        ax.set_xlim(-max(w_nat, w_target)/2 - pad,
                    +max(w_nat, w_target)/2 + pad)
        ax.set_ylim(-max(d, h_target/2) - 0.4, +max(d, h_target/2) + 0.4)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        pf = self._packing_factor()
        ax.set_title(
            f"Pre-squash (blue, natural Rutherford) vs post-squash (red, target)\n"
            f"{n} strands of D={d:.3f} mm   |   packing factor = {pf:.3f}   |   "
            f"compaction strain = "
            f"{(w_nat - w_target)/w_nat*100:.1f}% width, "
            f"{(2*d - h_target)/(2*d)*100:.1f}% height")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="lower center", fontsize=8, ncol=2,
                  bbox_to_anchor=(0.5, -0.45))
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def _render_warnings(self):
        v = self._violations()
        w = self._warnings()
        stack = self._hex_stack_summary()
        stack_html = (
            f"<div style='color:#444;padding:2px 0;font-size:90%'>"
            f"Subelement stack: {stack}</div>") if stack else ""
        if not v and not w:
            self.warn_box.value = (
                "<div style='font-family:sans-serif;color:#2e7d32;"
                "padding:4px 8px;background:#e8f5e9;border-radius:4px'>"
                "Design within physical bounds." + stack_html + "</div>")
            return
        rows = []
        for msg in v:
            rows.append(
                f"<div style='color:#c62828;padding:2px 0'>"
                f"<b>ERROR</b> &nbsp; {msg}</div>")
        for msg in w:
            rows.append(
                f"<div style='color:#ed6c02;padding:2px 0'>"
                f"<b>WARN</b> &nbsp; {msg}</div>")
        self.warn_box.value = (
            "<div style='font-family:sans-serif;padding:6px 10px;"
            "background:#fff3e0;border-radius:4px'>"
            + "".join(rows) + stack_html + "</div>")

    def to_preset_dict(self) -> dict:
        """Build the cables.<NAME> dict that would be written to JSON."""
        return {
            "cable_name":     self.name_w.value,
            "cable_width":    round(float(self.cable_width_w.value), 4),
            "cable_height":   round(float(self.cable_height_w.value), 4),
            "cable_length":   round(float(self.t_pitch_w.value) / 2.0, 4),
            "T_pitch":        round(float(self.t_pitch_w.value), 4),
            "N_Strands":      int(self.n_strands_w.value),
            "D_Strand":       round(float(self.d_strand_w.value), 4),
            "n_stacks":       int(self.n_stacks_w.value),
            "stack_height_mm": round(float(self.cable_height_w.value) + 0.3, 4),
            "wire": {
                "_comment": (
                    f"Designed via CableDesigner. CU_NONCU is informational - "
                    f"the geometric Cu/non-Cu is set by D_CORE_EQ_UM, "
                    f"N_NB3SN and CU_SLEEVE_THICKNESS_UM."),
                "N_NB3SN":               int(self.n_nb3sn_w.value),
                "N_TOTAL":               int(self.n_total_w.value),
                "D_CORE_EQ_UM":          round(float(self.d_core_eq_w.value), 2),
                "CU_SLEEVE_THICKNESS_UM": round(float(self.cu_sleeve_w.value), 2),
                "CU_NONCU":              round(float(self.cu_noncu_w.value), 2),
                "NB_THICKNESS_UM":       round(float(self.nb_thick_w.value), 2),
                "BRONZE_FRAC":           round(float(self.bronze_frac_w.value), 2),
                "REFINE_MESH":           False,
            },
        }

    def _render_json(self):
        preset = self.to_preset_dict()
        wrapped = {"cables": {self.name_w.value: preset}}
        text = json.dumps(wrapped, indent=4)
        self.json_out.value = (
            f"<pre style='font-family:Consolas,monospace;font-size:11px;"
            f"max-height:280px;overflow:auto;background:#f7f7f7;"
            f"padding:6px 10px;border:1px solid #ddd;border-radius:4px'>"
            f"{text}</pre>")

    def _on_save(self, _btn):
        import shutil
        from datetime import datetime
        if self._violations():
            self.save_status.value = (
                "<span style='color:#c62828;font-family:monospace'>"
                "Refusing to save: fix the ERROR(s) above first.</span>")
            return
        if not self.name_w.value or not self.name_w.value.strip():
            self.save_status.value = (
                "<span style='color:#c62828;font-family:monospace'>"
                "Empty preset name.</span>")
            return

        # Backup, then merge into the existing JSON.
        bak = CABLE_PARAMS_USER.with_suffix(
            CABLE_PARAMS_USER.suffix + ".bak")
        try:
            shutil.copy2(CABLE_PARAMS_USER, bak)
        except Exception as e:
            self.save_status.value = (
                f"<span style='color:#c62828;font-family:monospace'>"
                f"Could not back up JSON: {e}</span>")
            return

        try:
            with CABLE_PARAMS_USER.open() as f:
                cfg = json.load(f)
        except Exception as e:
            self.save_status.value = (
                f"<span style='color:#c62828;font-family:monospace'>"
                f"Could not read existing JSON: {e}</span>")
            return

        name = self.name_w.value.strip()
        cfg.setdefault("cables", {})[name] = self.to_preset_dict()
        cfg["active_cable"] = name

        try:
            with CABLE_PARAMS_USER.open("w") as f:
                json.dump(cfg, f, indent=4)
        except Exception as e:
            self.save_status.value = (
                f"<span style='color:#c62828;font-family:monospace'>"
                f"Write failed: {e}</span>")
            return

        ts = datetime.now().strftime("%H:%M:%S")
        self.save_status.value = (
            f"<span style='color:#2e7d32;font-family:monospace'>"
            f"Saved '{name}' (active_cable updated) at {ts}. "
            f"Backup: {bak.name}</span>")

    def display(self):
        from IPython.display import display
        display(self._panel)


# ─ Per-stage runner (background thread + log tail) ───────────────────────────


class StageRunner:
    """Per-stage button + status + Output for log capture.

    The work function ``fn`` is run in a daemon thread with stdout/stderr
    redirected to the Output widget (logging output goes there too because the
    pipeline's stream handler writes to stderr).  Optionally tails a log file
    in parallel.

    Use ``add_postaction`` to chain a viz function that fires once the stage
    completes successfully.
    """

    def __init__(self, label: str, fn, log_file_fn=None,
                 needs_run_folder: bool = True,
                 stop_run_folder_fn=None):
        import ipywidgets as W  # type: ignore
        self._W = W
        self.label = label
        self.fn = fn
        self.log_file_fn = log_file_fn
        self.needs_run_folder = needs_run_folder
        # If provided, a callable returning the run folder whose containers
        # the Stop button should `docker compose down`.  When None the stage
        # has no heavy container to abort (cheap stage) -- no Stop button.
        self.stop_run_folder_fn = stop_run_folder_fn
        self._post_actions: List = []
        self._thread = None
        self._tail_stop = False

        self.button = W.Button(
            description=f"Run: {label}",
            button_style="primary",
            layout=W.Layout(width="320px"))
        self.status = W.HTML(value="<i>idle</i>")
        self.output = W.Output(layout=W.Layout(
            border="1px solid #ccc", max_height="320px",
            overflow="auto", padding="4px"))
        self.button.on_click(self._on_click)

        if stop_run_folder_fn is not None:
            self.stop_button = W.Button(
                description="Stop containers",
                button_style="danger",
                tooltip="docker compose down for every container project "
                        "tied to this run folder",
                layout=W.Layout(width="180px"))
            self.stop_button.on_click(self._on_stop_click)
        else:
            self.stop_button = None

    def add_postaction(self, fn):
        """Register a no-arg callable to fire after a successful run."""
        self._post_actions.append(fn)
        return self

    def _set_status(self, color: str, text: str):
        self.status.value = (
            f"<span style='font-family:monospace;color:{color}'>{text}</span>")

    def _on_click(self, _btn):
        import threading
        if self._thread is not None and self._thread.is_alive():
            self._set_status("#ed6c02", "already running - wait for completion or click Stop")
            return
        self.output.clear_output()
        self.button.disabled = True
        self._set_status("#1565c0", f"running: {self.label} ...")
        self._tail_stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _on_stop_click(self, _btn):
        """Aborts running Docker containers for the current run folder.

        Does NOT kill the in-process daemon thread (the worker is still
        blocked on its subprocess; once docker compose down kills the
        container the worker's subprocess returns rc=130 and the thread
        exits).  Restart the kernel if you also want to free the worker
        thread immediately.
        """
        with self.output:
            try:
                rf = self.stop_run_folder_fn() if self.stop_run_folder_fn else None
            except Exception as e:
                print(f"Cannot resolve run folder for Stop: {e}")
                return
            if rf is None:
                print("No run folder selected - nothing to stop.")
                return
            print(f"\n[Stop] docker compose down for run folder '{rf.name}'")
            results = stop_run_containers(Path(rf))
            print(f"[Stop] results: {results}")
        self._set_status("#ed6c02", "stop requested - worker thread will exit when subprocess returns")

    def _tail_log(self, log_path: Path):
        """Append new bytes from ``log_path`` to the Output widget until
        ``self._tail_stop`` is set."""
        import time
        try:
            # Wait briefly for the log file to appear.
            for _ in range(30):
                if log_path.exists():
                    break
                if self._tail_stop:
                    return
                time.sleep(0.5)
            if not log_path.exists():
                return
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)  # tail from end of any existing content
                while not self._tail_stop:
                    line = f.readline()
                    if line:
                        with self.output:
                            print(line, end="")
                    else:
                        time.sleep(1.0)
        except Exception as e:
            with self.output:
                print(f"[log tail error] {e}")

    def _worker(self):
        import threading
        import traceback
        log_thread = None
        try:
            # Spawn tail thread if a log path is provided.
            if self.log_file_fn is not None:
                try:
                    log_path = self.log_file_fn()
                except Exception:
                    log_path = None
                if log_path is not None:
                    log_thread = threading.Thread(
                        target=self._tail_log, args=(Path(log_path),), daemon=True)
                    log_thread.start()

            with self.output:
                try:
                    result = self.fn()
                except Exception as e:
                    traceback.print_exc()
                    self._set_status("#c62828", f"FAILED: {e}")
                    return

            ok = result is not False  # treat None/True/Path/anything-not-False as success
            self._set_status(
                "#2e7d32" if ok else "#c62828",
                f"done: {self.label}  (result={result!r})")

            for post in list(self._post_actions):
                try:
                    post()
                except Exception as e:
                    with self.output:
                        print(f"[postaction error] {e}")
        finally:
            self._tail_stop = True
            self.button.disabled = False

    def display(self):
        from IPython.display import display
        if self.stop_button is not None:
            top_row = self._W.HBox([self.button, self.stop_button, self.status])
        else:
            top_row = self._W.HBox([self.button, self.status])
        display(self._W.VBox([top_row, self.output]))


# ─ Stage closures (each binds picker + config + runner) ─────────────────────


def _ensure_run_folder(picker: "RunPicker", config: "RunConfig",
                       create_if_missing: bool = False) -> Path:
    """Return the picker's selected run folder, or create a new one named with
    the current cable + timestamp.  Used by stage closures that need a folder
    on disk before they can do anything."""
    if picker.run_folder is not None and picker.run_folder.exists():
        return picker.run_folder
    if not create_if_missing:
        raise RuntimeError("No run folder selected. Use 'Create new run folder' first.")
    runner = make_runner()
    cable = config.cable
    runner.setup_cable_config(cable)
    run_id = runner.generate_run_id(cable)
    folder = runner.setup_directories(run_id)
    return folder


def make_stage_runners(picker: "RunPicker", config: "RunConfig") -> Dict[str, "StageRunner"]:
    """Build the per-stage StageRunner objects bound to the live picker+config.

    Returns a dict keyed by stage label so the notebook can lay them out in
    whatever order it wants.
    """

    def _create_run_folder():
        runner = make_runner()
        cable = config.cable
        runner.setup_cable_config(cable)
        run_id = runner.generate_run_id(cable)
        folder = runner.setup_directories(run_id)
        # Refresh picker so downstream cells see the new folder.
        picker._refresh_folders()
        try:
            picker.folder_dd.value = str(folder)
        except Exception:
            pass
        print(f"Created and selected run folder: {folder}")
        return folder

    def _cable_params():
        rf = _ensure_run_folder(picker, config, create_if_missing=True)
        return make_runner().run_cable_parameters(rf, config.termination_time)

    def _metadata():
        rf = _ensure_run_folder(picker, config)
        runner = make_runner()
        runner.setup_cable_config(config.cable)
        cable_params_file = rf / "cable_parameters.json"
        if not cable_params_file.exists():
            cable_params_file = runner.run_cable_parameters(rf, config.termination_time)
        return runner.generate_metadata(rf.name, rf, cable_params_file)

    def _step():
        rf = _ensure_run_folder(picker, config)
        runner = make_runner()
        # generate_metadata is required upstream of run_geometry_generation.
        metadata_file = rf / "metadata.json"
        if not metadata_file.exists():
            cable_params_file = rf / "cable_parameters.json"
            if not cable_params_file.exists():
                cable_params_file = runner.run_cable_parameters(rf, config.termination_time)
            metadata_file = runner.generate_metadata(rf.name, rf, cable_params_file)
        return runner.run_geometry_generation(rf, metadata_file)

    def _lsdyna_setup():
        rf = _ensure_run_folder(picker, config)
        config.apply_hpc_env()
        return make_runner().run_lsdyna_setup(
            rf, config.termination_time, config.min_mesh_size)

    def _mesh_conv():
        rf = _ensure_run_folder(picker, config)
        return make_runner().run_mesh_conversion(rf, config.termination_time)

    def _lsdyna_solve():
        rf = _ensure_run_folder(picker, config)
        config.apply_hpc_env()
        runner = make_runner()
        sim_started = runner.run_lsdyna_simulation(rf)
        if not sim_started:
            return False
        log_file = rf / "LSDYNA" / "lsdyna_container.log"
        return runner._wait_for_lsdyna_completion(log_file, config.termination_time)

    def _lsdyna_log():
        rf = picker.run_folder
        return (rf / "LSDYNA" / "lsdyna_container.log") if rf else None

    def _paraview():
        rf = _ensure_run_folder(picker, config)
        return make_runner().run_paraview_extraction(rf)

    def _apdl_submodel():
        rf = _ensure_run_folder(picker, config)
        return make_runner().run_apdl_submodel(rf, debug_plots=bool(config.debug_plots.value))

    def _cablestack():
        rf = _ensure_run_folder(picker, config)
        config.apply_hpc_env()
        runner = make_runner()
        return runner.copy_cablestack_files(
            rf,
            launch_apdl=not bool(config.no_cablestack.value),
            use_hpc=bool(config.use_hpc.value),
        )

    def _compbox():
        rf = _ensure_run_folder(picker, config)
        config.apply_hpc_env()
        return make_runner().run_compression_box(rf, use_hpc=bool(config.use_hpc.value))

    # Returns the picker's current run folder; bound late so the Stop button
    # sees whatever the user has selected at click time.
    def _current_run_folder():
        return picker.run_folder

    return {
        "create_folder":   StageRunner("Create new run folder", _create_run_folder,
                                       needs_run_folder=False),
        "cable_params":    StageRunner("1. Cable parameters", _cable_params),
        "metadata":        StageRunner("1b. Metadata", _metadata),
        "step":            StageRunner("2. FreeCAD STEP geometry", _step),
        "lsdyna_setup":    StageRunner("3a. LS-DYNA mesh (Ansys Mechanical)", _lsdyna_setup,
                                       stop_run_folder_fn=_current_run_folder),
        "mesh_conv":       StageRunner("3b. Mesh conversion (.k)", _mesh_conv),
        "lsdyna_solve":    StageRunner("5. LS-DYNA solve (heavy)", _lsdyna_solve,
                                       log_file_fn=_lsdyna_log,
                                       stop_run_folder_fn=_current_run_folder),
        "paraview":        StageRunner("6. ParaView strand extraction", _paraview),
        "apdl_submodel":   StageRunner("7. APDL conformal submodel", _apdl_submodel),
        "cablestack":      StageRunner("8. Cablestack (heavy; HPC if toggled)", _cablestack,
                                       stop_run_folder_fn=_current_run_folder),
        "compbox":         StageRunner("9. Compression box (heavy)", _compbox,
                                       stop_run_folder_fn=_current_run_folder),
    }
