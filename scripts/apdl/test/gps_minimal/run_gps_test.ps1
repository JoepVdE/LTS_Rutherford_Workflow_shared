# Run the minimal GPS test deck against native MAPDL v252.
# Usage:  pwsh ./run_gps_test.ps1   (from this directory or any other)

$ErrorActionPreference = 'Stop'
$here  = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $here

$ansys = "C:/Program Files/ANSYS Inc/v252/ansys/bin/winx64/ANSYS252.exe"
if (-not (Test-Path $ansys)) {
    throw "ANSYS not found at $ansys"
}

# Clean prior outputs (db/results/log) so we never read stale data
$wipe = @('gps_test.out','gps_test.err','gps_test.db','gps_test.rst','gps_test.esav','gps_test.full','gps_test.mntr','gps_test.stat','gps_test.PAGE','file.*','*.rst','*.db','*.esav','*.full','*.mntr','*.stat','*.PAGE')
foreach ($pat in $wipe) {
    Get-ChildItem -Path . -Filter $pat -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

Write-Host "Running: $ansys -b -i gps_minimal.inp -o gps_test.out -j gps_test"
& $ansys -b -i gps_minimal.inp -o gps_test.out -j gps_test
$exit = $LASTEXITCODE
Write-Host "MAPDL exit code: $exit"

if (Test-Path gps_test.out) {
    Write-Host "`n--- key log lines ---"
    Get-Content gps_test.out |
        Select-String -Pattern 'MESH_RESULT|GPS_RESULT|n_elem|n_node|\*\*\* ERROR|\*\*\* WARNING|GENERALIZED PLANE STRAIN|GSGDATA|GSBDATA'
} else {
    Write-Host "No gps_test.out produced -- MAPDL did not run."
}
