# Download and verify FreeCAD + ParaView portable binaries.
#
# Reads tools/MANIFEST.sha256 (committed in this repo) and fetches each
# entry from the GitHub Release tagged $ReleaseTag. Verifies SHA256 before
# extracting. Exits non-zero on any verification or download failure --
# bit-identical reproducibility is enforced, not just hoped for.
#
# Usage (from the repo root):
#   pwsh tools/fetch_tools.ps1
#   pwsh tools/fetch_tools.ps1 -ReleaseTag v0.1-tools
#   pwsh tools/fetch_tools.ps1 -Force          # re-download even if already extracted
#
# The archives total ~2 GB. First-run download time depends on your link.

[CmdletBinding()]
param(
    [string]$ReleaseTag = 'v0.1-tools',
    [string]$RepoSlug   = 'JoepVdE/LTS_Rutherford_Workflow_shared',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$ToolsDir   = Split-Path -Parent $PSCommandPath
$Manifest   = Join-Path $ToolsDir 'MANIFEST.sha256'

if (-not (Test-Path $Manifest)) {
    throw "Manifest not found: $Manifest"
}

# Parse manifest. Each non-comment line: <sha256>  <filename>  <bytes>  <unpack_dir>
$entries = @()
Get-Content -LiteralPath $Manifest | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq '' -or $line.StartsWith('#')) { return }
    $parts = $line -split '\s+'
    if ($parts.Count -lt 4) {
        Write-Warning "Skipping malformed manifest line: $line"
        return
    }
    $entries += [pscustomobject]@{
        Sha256     = $parts[0].ToLower()
        Filename   = $parts[1]
        Bytes      = [int64]$parts[2]
        UnpackDir  = $parts[3]
    }
}

if ($entries.Count -eq 0) {
    throw "Manifest contained no entries: $Manifest"
}

Write-Host "Fetching tools from release '$ReleaseTag' of $RepoSlug" -ForegroundColor Cyan

foreach ($e in $entries) {
    $extractedPath = Join-Path $ToolsDir $e.UnpackDir
    if ((Test-Path $extractedPath) -and (-not $Force)) {
        # Heuristic: a non-empty directory with at least one subdirectory is "good enough"
        # to skip re-download. -Force overrides.
        $children = Get-ChildItem -LiteralPath $extractedPath -Force -ErrorAction SilentlyContinue
        if ($children -and $children.Count -gt 0) {
            Write-Host "[$($e.UnpackDir)] already populated, skipping (pass -Force to redo)." -ForegroundColor Yellow
            continue
        }
    }

    $url     = "https://github.com/$RepoSlug/releases/download/$ReleaseTag/$($e.Filename)"
    $tmpZip  = Join-Path $env:TEMP "lts_rutherford_$($e.Filename)"

    Write-Host "[$($e.UnpackDir)] downloading $($e.Filename) ($([math]::Round($e.Bytes/1MB,1)) MB)..." -ForegroundColor Cyan
    Write-Host "  URL: $url" -ForegroundColor DarkGray
    if (Test-Path $tmpZip) { Remove-Item -LiteralPath $tmpZip -Force }
    # Invoke-WebRequest is slow for large files due to progress overhead.
    # Use a BITS-free approach via .NET WebClient for speed.
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $tmpZip)
    } finally {
        if ($wc) { $wc.Dispose() }
    }

    Write-Host "[$($e.UnpackDir)] verifying SHA256..." -ForegroundColor Cyan
    $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $tmpZip).Hash.ToLower()
    if ($got -ne $e.Sha256) {
        Remove-Item -LiteralPath $tmpZip -Force
        throw "SHA256 mismatch for $($e.Filename): expected $($e.Sha256), got $got. Download corrupted or release asset replaced."
    }

    Write-Host "[$($e.UnpackDir)] extracting..." -ForegroundColor Cyan
    if (Test-Path $extractedPath) {
        Remove-Item -LiteralPath $extractedPath -Recurse -Force
    }
    # Expand-Archive is built into PowerShell 5+ and handles standard ZIP files.
    Expand-Archive -LiteralPath $tmpZip -DestinationPath $ToolsDir -Force
    Remove-Item -LiteralPath $tmpZip -Force

    if (-not (Test-Path $extractedPath)) {
        throw "Extraction completed but expected directory missing: $extractedPath"
    }
    Write-Host "[$($e.UnpackDir)] OK" -ForegroundColor Green
}

Write-Host "All tools ready under: $ToolsDir" -ForegroundColor Green
