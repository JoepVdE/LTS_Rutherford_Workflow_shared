# tools/

This directory hosts the **bit-identical** binary distribution of the two
external tools the pipeline depends on but cannot install via pip:

| Subdir | Tool | Version |
|---|---|---|
| `freecad/` | FreeCAD (headless `freecadcmd.exe`) | 1.0.2 |
| `paraview/` | ParaView (`pvpython.exe`) | 6.0.1 |

The binaries are **not committed to git** (5.6 GB combined). They live
as assets on a GitHub Release (tagged `v0.1-tools`) and are fetched on
demand by either `fetch_tools.ps1` (Windows) or `fetch_tools.sh`
(Linux/macOS).

## First-time setup

From the repository root:

```powershell
# Windows
pwsh tools/fetch_tools.ps1
```

```bash
# Linux / macOS
bash tools/fetch_tools.sh
```

Both scripts:
1. Read `MANIFEST.sha256` (committed) for expected hashes.
2. Download each archive from the GitHub Release.
3. **Verify SHA256** -- abort if the download doesn't match.
4. Extract into `tools/freecad/` and `tools/paraview/`.

If the script aborts on hash mismatch, the release assets have either
been corrupted in transit or replaced. Report it; do not blindly retry
with `-Force`.

## Why this and not Git LFS?

Same bit-identical guarantee, simpler distribution: SHA256 manifest in
git, large binaries on release assets, free, no LFS quota. The
verification step actually *enforces* the hash, while LFS only checks
internally.

## Repackaging the archives (maintainers only)

The release assets were built with 7-Zip on the development machine and
re-saved as standard ZIP volumes (under 2 GiB each, so they fit in a
single GitHub Release asset). The exact bytes hashed in
`MANIFEST.sha256` are what gets shipped.

If you replace either tool, regenerate the manifest:

```powershell
$h = Get-FileHash -Algorithm SHA256 tools_release_staging\freecad.zip
"$($h.Hash.ToLower())  freecad.zip  $((Get-Item tools_release_staging\freecad.zip).Length)  freecad"
```

then edit `tools/MANIFEST.sha256` and upload the new asset to a fresh
release tag.
