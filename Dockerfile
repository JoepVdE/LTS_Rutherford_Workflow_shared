# LTS Rutherford Workflow orchestrator container.
#
# Bundles: Python 3.12 + FreeCAD (headless) + ParaView (pvpython) + Docker CLI
#          + all Python deps from pyproject.toml + the repo itself.
#
# The orchestrator inside this container spawns sibling Docker containers
# (mesher / LS-DYNA / MAPDL) on the *host* Docker daemon via a mounted socket.
# It does NOT use Docker-in-Docker -- it uses Docker-out-of-Docker.
#
# Build:
#     docker build -t lts-cable .
#
# Run (Linux/macOS):
#     docker run --rm \
#         -v /var/run/docker.sock:/var/run/docker.sock \
#         -v "$PWD/data:/app/data" \
#         -e ANSYS_LICENSE_SERVER=1055@lxlicen01.cern.ch \
#         lts-cable --list-cables
#
# Run (Windows PowerShell):
#     docker run --rm `
#         -v //var/run/docker.sock:/var/run/docker.sock `
#         -v "${PWD}/data:/app/data" `
#         -e ANSYS_LICENSE_SERVER=$env:ANSYS_LICENSE_SERVER `
#         lts-cable --list-cables
#
# Required at runtime:
#   - A reachable ANSYS license server (set ANSYS_LICENSE_SERVER, or be on a
#     network where CERN/ETH/PSI auto-probe succeeds). CERN default is
#     1055@licenansys.
#   - Docker on the host (the orchestrator spawns sibling mesher / LS-DYNA /
#     MAPDL containers via the mounted socket -- Docker-out-of-Docker).
#   - Pull access to the two ANSYS images (each ship ANSYS internally; NO
#     host ANSYS install is required and nothing is bind-mounted from a host
#     ANSYS tree):
#       - <prefix>/mechanical:25.2 (used for both meshing AND MAPDL cablestack/compbox)
#       - <prefix>/lsdyna:25.2
#     <prefix> comes from the REGISTRY_PREFIX env var (default
#     `gitea.psi.ch/vanden_j`; CERN users set `registry.cern.ch/chart-magnum`).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# --- system deps -------------------------------------------------------------
# python3.12 is the default on Ubuntu 24.04.
# freecad provides both the GUI binary and `freecadcmd` (headless).
# paraview provides `pvpython`.
# docker.io supplies the docker CLI (it's used to spawn sibling containers on
# the host -- the daemon itself is the host's, mounted via /var/run/docker.sock).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        freecad \
        freecad-common \
        paraview \
        docker.io \
        ca-certificates \
        curl \
        git \
        openssh-client \
        libgl1 \
        libglu1-mesa \
        libxrender1 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        libxi6 \
        libxrandr2 \
        libxt6 \
    && rm -rf /var/lib/apt/lists/*

# --- python orchestrator -----------------------------------------------------
WORKDIR /app

# Install Python deps first (better layer caching: pyproject.toml changes
# rarely, source code changes constantly).
COPY pyproject.toml /app/pyproject.toml

# We need `scripts/main/` available for setuptools to discover the package
# during `pip install -e .` -- copy that ahead of the rest so the editable
# install resolves.
COPY scripts/main /app/scripts/main

RUN python3.12 -m pip install --break-system-packages -e .

# --- repo --------------------------------------------------------------------
COPY . /app

# Verify both external tools resolve at build time (fail fast if the apt
# packages didn't ship the binaries we expect).
RUN which freecadcmd && which pvpython && which docker \
    && freecadcmd --version 2>&1 | head -1 \
    && pvpython --version 2>&1 | head -1

# data/ is the only volume mount point that needs to persist between runs.
VOLUME ["/app/data"]

# `lts-cable` is the console script declared in pyproject.toml -> [project.scripts].
ENTRYPOINT ["lts-cable"]
CMD ["--list-cables"]
