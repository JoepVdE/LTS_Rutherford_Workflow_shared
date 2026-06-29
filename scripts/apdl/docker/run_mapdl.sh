#!/bin/bash
set -e

echo "=========================================="
echo "ANSYS MAPDL Docker Runner"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Input file:        ${MAPDL_INPUT}"
echo "Job name:          ${MAPDL_JOBNAME}"
echo "CPUs:              ${MAPDL_NCPU}"
echo "Memory per rank:   ${MAPDL_MEMORY} MB"
echo "License server:    ${ANSYSLI_SERVERS}"
echo "=========================================="

# Verify input file exists
if [ ! -f "${MAPDL_INPUT}" ]; then
    echo "ERROR: Input file not found: ${MAPDL_INPUT}"
    echo "Contents of /run:"
    ls -lah /run
    exit 1
fi

# Source Intel MPI if needed (adjust path based on your MAPDL container)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh
fi

# ANSYS installation path (mechanical:25.2 ships under /install/ansys_inc)
if [ -d "/install/ansys_inc/v${ANSYS_VERSION}" ]; then
    ANSYS_ROOT="/install/ansys_inc/v${ANSYS_VERSION}"
else
    ANSYS_ROOT="/opt/ansys/v${ANSYS_VERSION}"
fi

# MAPDL executable (distributed memory parallel)
MAPDL_EXE="${ANSYS_ROOT}/ansys/bin/ansys${ANSYS_VERSION}"

# Build MAPDL command
# -b: batch mode (no GUI)
# -np: number of processors
# -i: input file
# -o: output file
# -j: job name
# -m: memory per rank (MB)
# -dis: distributed memory parallel
MAPDL_CMD="${MAPDL_EXE} \
  -b -dis \
  -np ${MAPDL_NCPU} \
  -m ${MAPDL_MEMORY} \
  -i ${MAPDL_INPUT} \
  -o ${MAPDL_JOBNAME}.out \
  -j ${MAPDL_JOBNAME} \
  ${MAPDL_EXTRA_ARGS}"

echo ""
echo "Running MAPDL command:"
echo "${MAPDL_CMD}"
echo ""

# Execute MAPDL
${MAPDL_CMD}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "MAPDL completed successfully"
else
    echo "MAPDL exited with code ${EXIT_CODE}"
fi
echo "=========================================="

# Keep output files readable on host
chmod -R a+rw /run/* 2>/dev/null || true

exit ${EXIT_CODE}
