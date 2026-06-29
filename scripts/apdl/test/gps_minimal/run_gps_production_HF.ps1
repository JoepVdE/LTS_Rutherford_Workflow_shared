# Run the production cablestack deck (R2D2_HF, displacement_transverse stage only)
# against native MAPDL v252. Reads from the cloned apdl_runfolder_gps_test/ so
# the original run folder is untouched.

$ErrorActionPreference = 'Stop'

$dst = "C:/LTS_Rutherford_Workflow/data/runs/20260504_232855_R2D2_HF_apdl_rerun_48/APDL/submodel/apdl_runfolder_gps_test"
$ansys = "C:/Program Files/ANSYS Inc/v252/ansys/bin/winx64/ANSYS252.exe"

if (-not (Test-Path $dst)) { throw "Run folder not found: $dst" }
if (-not (Test-Path $ansys)) { throw "MAPDL not found: $ansys" }

Set-Location $dst

# Clean prior outputs from any earlier attempt in this clone
$wipe = @('gps_test.out','gps_test.err','gps_test.db','gps_test.rst','gps_test.esav','gps_test.full','gps_test.mntr','gps_test.stat','gps_test.PAGE','*.lock','file.*')
foreach ($pat in $wipe) {
    Get-ChildItem -Path . -Filter $pat -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}

Write-Host "Working dir: $dst"
Write-Host "Running: $ansys -b -i 0-start.inp -o gps_test.out -j gps_test"
$t0 = Get-Date
& $ansys -b -i 0-start.inp -o gps_test.out -j gps_test
$exit = $LASTEXITCODE
$elapsed = (Get-Date) - $t0
Write-Host "MAPDL exit code: $exit  (elapsed: $($elapsed.TotalMinutes.ToString('F1')) min)"

if (Test-Path gps_test.out) {
    Write-Host "`n--- key log lines ---"
    Get-Content gps_test.out |
        Select-String -Pattern 'GSGDATA|GSBDATA|GENERALIZED PLANE STRAIN|FIBER LENGTH|n_elem|n_node|\*\*\* ERROR|There are no elements|ESURF command is ignored|CONVERGED|Finished solve'
}
