#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name=CABLE_LABEL
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --tasks-per-node=16
#SBATCH --constraint=ethernet
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joep@ethz.ch

# Static reference jobslurm.sh -- the per-run version is generated dynamically
# by cablestack_stages.write_cablestack_jobslurm based on cablestack.stages.
# All four restart stages RESUME base.db (written by the build stage).
# A failure in any restart must NOT prevent the others from running, but the
# first non-zero rc becomes the job's exit code so SLURM reports the failure.
set -uo pipefail

DB_FILE=base.db

# Clean up stale MAPDL lock files and stage sentinels from any prior abnormal
# exit in this dir.
rm -f *.lock file.lock *.success

module purge
module load ansys/24.2_research stack/2024-06 mesa-glu/9.0.2 motif/2.3.8-ehjpbwx

COMBINED_LOG="mapdl_run.log"
GLOBAL_RC=0

echo "[$(date '+%F %T')] JOB_START jobid=${SLURM_JOB_ID:-unknown} host=$(hostname)" | tee -a "${COMBINED_LOG}"

# Fatal stage: failure aborts the whole job (used for the build stage only).
run_stage_fatal () {
  local tag="$1"
  local inp="$2"

  echo "[$(date '+%F %T')] ${tag}_START input=${inp}" | tee -a "${COMBINED_LOG}"
  mapdl -np 16 -mpp -i "${inp}" >> "${COMBINED_LOG}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] ${tag}_END rc=${rc}" | tee -a "${COMBINED_LOG}"

  if [ "${rc}" -ne 0 ]; then
    echo "[$(date '+%F %T')] ${tag}_FAILED rc=${rc} -- aborting job" | tee -a "${COMBINED_LOG}"
    exit "${rc}"
  fi
  echo "rc=0 $(date '+%F %T')" > "${tag}.success"
}

# Non-fatal stage: failure is logged, GLOBAL_RC is set, other stages continue.
run_stage () {
  local tag="$1"
  local inp="$2"

  echo "[$(date '+%F %T')] ${tag}_START input=${inp}" | tee -a "${COMBINED_LOG}"
  mapdl -np 16 -mpp -i "${inp}" >> "${COMBINED_LOG}" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] ${tag}_END rc=${rc}" | tee -a "${COMBINED_LOG}"

  if [ "${rc}" -ne 0 ]; then
    echo "[$(date '+%F %T')] ${tag}_FAILED rc=${rc} -- continuing with next stage" | tee -a "${COMBINED_LOG}"
    GLOBAL_RC="${rc}"
    rm -f *.lock file.lock
  else
    echo "rc=0 $(date '+%F %T')" > "${tag}.success"
  fi
}

run_stage_fatal build ./0-start.inp

# Guard: verify base.db exists before launching the restart stages.
if [ ! -f "${DB_FILE}" ]; then
  echo "[$(date '+%F %T')] restart_SKIPPED db='${DB_FILE}' not found -- build did not complete cleanly" | tee -a "${COMBINED_LOG}"
  exit 1
fi

# Four mutually-independent restart stages, each RESUMEing base.db.
run_stage displacement_transverse ./00-restart-transverse.inp
run_stage displacement_radial     ./00-restart-radial.inp
run_stage pressure_transverse     ./00-restart-pressure.inp
run_stage pressure_radial         ./00-restart-pressure-radial.inp

echo "[$(date '+%F %T')] JOB_END rc=${GLOBAL_RC}" | tee -a "${COMBINED_LOG}"
exit "${GLOBAL_RC}"
