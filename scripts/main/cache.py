"""Content-addressed run cache.

Two cache levels:

- ``lsdyna``     -- covers everything up through LS-DYNA solve + ParaView
                    extraction.  A hit means the d3plot and per-stack CSVs in
                    the referenced run folder are reusable; the APDL submodel +
                    cablestack stages still need to be (re)run on top.

- ``cablestack`` -- covers everything above PLUS the cablestack config and
                    templates.  A hit means the full pipeline's outputs are
                    already in the referenced run folder and we can return it
                    as the answer.

Index lives at ``data/cache/index.json`` (atomic write).  Entries point at
run folders under ``data/runs/``; on lookup we verify the referenced folder
still exists and ``metadata.json`` shows the relevant ``workflow_steps`` as
``completed``, so manually-deleted runs self-evict.

The cache is opt-out by default.  Callers pass ``--no-cache`` (or
``no_cache=True``) to force a fresh run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


INDEX_VERSION = 1

# Mirrors scripts/lsdyna/script/mesh_to_lsdyna.py.  Hashing the tag (not the
# image digest) is a deliberate trade-off: catching the realistic "we bumped
# to 25.3" case without requiring Docker to be running for fingerprinting.
DEFAULT_DOCKER_IMAGE_TAG = "gitea.psi.ch/vanden_j/mechanical:25.2"

REQUIRED_STEPS = {
    "lsdyna":     ("5_lsdyna_simulation", "6_paraview_extraction"),
    "cablestack": ("5_lsdyna_simulation", "6_paraview_extraction",
                   "7_apdl_submodel", "8_cablestack"),
}


@dataclass
class CacheHit:
    level: str            # 'lsdyna' or 'cablestack'
    folder: Path          # absolute path to the cached run folder
    fingerprint: str


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_dir(root: Path, patterns: tuple) -> str:
    """SHA256 over sorted (relpath, sha256(content)) for matching files."""
    if not root.is_dir():
        return "MISSING"
    items = []
    seen = set()
    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file() and p not in seen:
                seen.add(p)
                items.append((p.relative_to(root).as_posix(), _hash_file(p)))
    items.sort()
    payload = json.dumps(items, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash_parts(parts: dict) -> str:
    return hashlib.sha256(_canonical(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------

def compute_lsdyna_fingerprint(
    workspace_root: Path,
    cable_block: dict,
    termination_ms: float,
    min_mesh_size_mm: Optional[float],
    max_mesh_size_mm: Optional[float],
    docker_image_tag: str = DEFAULT_DOCKER_IMAGE_TAG,
) -> str:
    """Fingerprint covering everything up through LS-DYNA + ParaView extraction.

    Inputs that determine the d3plot: cable params, termination time, mesh
    sizes, the Docker image tag, and the code/templates that produce them
    (FreeCAD macro + LS-DYNA setup + meshconverter).
    """
    parts = {
        "v": INDEX_VERSION,
        "cable_block": cable_block,
        "termination_ms": termination_ms,
        "min_mesh_size_mm": min_mesh_size_mm,
        "max_mesh_size_mm": max_mesh_size_mm,
        "docker_image_tag": docker_image_tag,
        "freecad_macro":   _hash_dir(workspace_root / "scripts" / "setup_step",          ("*.py", "*.FCMacro")),
        "lsdyna_setup":    _hash_dir(workspace_root / "scripts" / "lsdyna" / "script",   ("*.py",)),
        "meshconverter":   _hash_dir(workspace_root / "scripts" / "meshconverter",       ("*.py", "*.k")),
        "paraview_script": _hash_dir(workspace_root / "scripts" / "paraview",            ("*.py",)),
    }
    return _hash_parts(parts)


def compute_cablestack_fingerprint(
    workspace_root: Path,
    lsdyna_fp: str,
    cablestack_block: dict,
) -> str:
    """Fingerprint covering the full pipeline.

    Builds on the LS-DYNA fingerprint and adds the cablestack config plus the
    cablestack and d3plottoapdl template/code hashes.
    """
    parts = {
        "v": INDEX_VERSION,
        "lsdyna_fp": lsdyna_fp,
        "cablestack_block": cablestack_block,
        "cablestack_templates": _hash_dir(
            workspace_root / "scripts" / "apdl" / "submodel" / "cablestack",
            ("*.inp", "*.sh"),
        ),
        "d3plottoapdl": _hash_dir(
            workspace_root / "scripts" / "d3plottoapdl_package",
            ("*.py",),
        ),
        "analysis":     _hash_dir(
            workspace_root / "scripts" / "analysis" / "submodel" / "cablestack",
            ("*.py",),
        ),
    }
    return _hash_parts(parts)


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def _index_path(workspace_root: Path) -> Path:
    return workspace_root / "data" / "cache" / "index.json"


def _read_index(workspace_root: Path) -> dict:
    ip = _index_path(workspace_root)
    if not ip.is_file():
        return {"version": INDEX_VERSION, "lsdyna": {}, "cablestack": {}}
    try:
        data = json.loads(ip.read_text(encoding="utf-8"))
        data.setdefault("lsdyna", {})
        data.setdefault("cablestack", {})
        return data
    except Exception as e:
        logger.warning(f"Cache index unreadable ({e}); ignoring.")
        return {"version": INDEX_VERSION, "lsdyna": {}, "cablestack": {}}


def _write_index_atomic(workspace_root: Path, data: dict) -> None:
    ip = _index_path(workspace_root)
    ip.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=ip.parent, prefix=".idx.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, ip)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


@contextmanager
def _index_lock(workspace_root: Path):
    """Exclusive cross-process lock around the index read-modify-write.

    Without it, two parallel `--cables` subprocesses finishing close together
    can interleave read/replace and silently drop one registration.  Uses
    msvcrt byte-range locking on Windows and flock elsewhere; both block until
    the lock is free and release automatically if the process dies.
    """
    lock_path = _index_path(workspace_root).parent / ".idx.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    # LK_LOCK gives up after ~10 s of contention; keep waiting.
                    continue
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


# ---------------------------------------------------------------------------
# Lookup / register
# ---------------------------------------------------------------------------

def lookup(workspace_root: Path, level: str, fingerprint: str) -> Optional[Path]:
    """Return the cached run folder for (level, fingerprint), or None.

    Self-evicts entries whose run folder was deleted or whose
    metadata.json no longer shows the relevant workflow_steps as completed.
    """
    if level not in REQUIRED_STEPS:
        return None
    idx = _read_index(workspace_root)
    entry = idx.get(level, {}).get(fingerprint)
    if not entry:
        return None
    folder = workspace_root / "data" / "runs" / entry["run_folder"]
    if not folder.is_dir():
        return None
    meta_path = folder / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    steps = meta.get("workflow_steps", {})
    for key in REQUIRED_STEPS[level]:
        if steps.get(key) != "completed":
            return None
    return folder


def register(workspace_root: Path, level: str, fingerprint: str,
             run_folder: Path, active_cable: str) -> None:
    if level not in REQUIRED_STEPS:
        return
    with _index_lock(workspace_root):
        idx = _read_index(workspace_root)
        idx.setdefault(level, {})[fingerprint] = {
            "run_folder": run_folder.name,
            "active_cable": active_cable,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_index_atomic(workspace_root, idx)
    logger.info(f"Cache registered ({level}): {fingerprint[:12]}... -> {run_folder.name}")


# ---------------------------------------------------------------------------
# Convenience: load cable config + look up both levels in one call.
# ---------------------------------------------------------------------------

def load_cache_inputs(workspace_root: Path, cable_name: str) -> tuple:
    """Read cable_parameters_user.json and return (cable_block, cablestack_block).

    cable_block:     the per-cable spec under cables[cable_name].
    cablestack_block: the top-level cablestack settings (impreg, bc_type,
                     stages, mesh sizes, pressure schedule, ...).
    """
    cfg_path = workspace_root / "scripts" / "main" / "cable_parameters_user.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cable_block = cfg.get("cables", {}).get(cable_name)
    if cable_block is None:
        raise KeyError(f"cable '{cable_name}' not in {cfg_path.name}")
    cablestack_block = cfg.get("cablestack", {})
    return cable_block, cablestack_block


def check(
    workspace_root: Path,
    cable_name: str,
    termination_ms: float,
    min_mesh_size_mm: Optional[float],
    max_mesh_size_mm: Optional[float],
    docker_image_tag: str = DEFAULT_DOCKER_IMAGE_TAG,
) -> tuple:
    """Compute both fingerprints and look them up.

    Returns (lsdyna_fp, cablestack_fp, hit_or_None).  Cablestack hit takes
    precedence over LS-DYNA hit -- if we have the whole pipeline cached,
    we return it directly.
    """
    cable_block, cablestack_block = load_cache_inputs(workspace_root, cable_name)

    lsdyna_fp = compute_lsdyna_fingerprint(
        workspace_root=workspace_root,
        cable_block=cable_block,
        termination_ms=termination_ms,
        min_mesh_size_mm=min_mesh_size_mm,
        max_mesh_size_mm=max_mesh_size_mm,
        docker_image_tag=docker_image_tag,
    )
    cablestack_fp = compute_cablestack_fingerprint(
        workspace_root=workspace_root,
        lsdyna_fp=lsdyna_fp,
        cablestack_block=cablestack_block,
    )

    cs_hit = lookup(workspace_root, "cablestack", cablestack_fp)
    if cs_hit is not None:
        return lsdyna_fp, cablestack_fp, CacheHit("cablestack", cs_hit, cablestack_fp)

    ld_hit = lookup(workspace_root, "lsdyna", lsdyna_fp)
    if ld_hit is not None:
        return lsdyna_fp, cablestack_fp, CacheHit("lsdyna", ld_hit, lsdyna_fp)

    return lsdyna_fp, cablestack_fp, None
