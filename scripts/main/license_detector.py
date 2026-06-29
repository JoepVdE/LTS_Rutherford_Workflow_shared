"""LicenseDetector port + adapters.

The ANSYS license server reachable from this machine depends on which network
we're on (ETH vs PSI vs CERN vs VPN'd from elsewhere).  Probing each candidate
with a short TCP timeout and returning only the reachable one prevents FlexLM
from hanging on an unreachable server during checkout.

Override mechanism: set the ANSYS_LICENSE_SERVER environment variable to a
verbatim FlexLM string (e.g. "1055@lxlicen01.cern.ch") to bypass probing
entirely.  Useful for sites we haven't added to the default candidate list, or
when probing isn't desirable (CI, behind a NAT, etc.).

Consumed by LS-DYNA launch, RVE launch, and the LocalMAPDL cablestack adapter.
"""
from __future__ import annotations

import logging
import os
import socket
from typing import Protocol

logger = logging.getLogger(__name__)


class LicenseDetector(Protocol):
    def detect(self) -> str: ...


class NetworkProbeLicenseDetector:
    """Probes ETH, PSI, and CERN license servers with a short TCP timeout.

    If the ANSYS_LICENSE_SERVER env var is set, it short-circuits probing and
    returns the value verbatim -- this is the recommended way for a new
    institute (or a CI machine) to point the pipeline at its own license.
    """

    DEFAULT_CANDIDATES = (
        # (FlexLM string,                              hostname,                       port)
        ("1801@lic-ansys-research.ethz.ch",            "lic-ansys-research.ethz.ch",   1801),
        ("1055@winlic03.psi.ch",                       "winlic03.psi.ch",              1055),
        # CERN ANSYS license -- update the host:port to match your site's FlexLM server.
        # Override via ANSYS_LICENSE_SERVER env var if your CERN account uses a
        # different server (e.g. a department-specific one).
        ("1055@lxlicen01.cern.ch",                     "lxlicen01.cern.ch",            1055),
    )
    FALLBACK = "1801@lic-ansys-research.ethz.ch:1055@winlic03.psi.ch:1055@lxlicen01.cern.ch"
    ENV_VAR = "ANSYS_LICENSE_SERVER"

    def __init__(self, candidates=None, timeout_s: float = 3.0):
        self.candidates = candidates if candidates is not None else self.DEFAULT_CANDIDATES
        self.timeout_s = timeout_s

    def detect(self) -> str:
        override = os.environ.get(self.ENV_VAR)
        if override:
            logger.info(f"{self.ENV_VAR} set; using {override!r} (skipping probe).")
            return override

        reachable = []
        for license_str, host, port in self.candidates:
            try:
                with socket.create_connection((host, port), timeout=self.timeout_s):
                    reachable.append(license_str)
                    logger.info(f"License server reachable: {license_str}")
            except OSError:
                logger.info(f"License server unreachable (skipping): {license_str}")

        if not reachable:
            self._log_no_server_help()
            return self.FALLBACK
        result = ":".join(reachable)
        logger.info(f"Using license server(s): {result}")
        return result

    def _log_no_server_help(self) -> None:
        logger.warning(
            "No ANSYS license server reachable.  Point the pipeline at your "
            "institute's server by setting the ANSYS_LICENSE_SERVER environment "
            "variable to its FlexLM string before re-running.  Examples:\n"
            "  CERN (lxplus / cmf):  export ANSYS_LICENSE_SERVER=1055@lxlicen01.cern.ch\n"
            "  ETH:                  export ANSYS_LICENSE_SERVER=1801@lic-ansys-research.ethz.ch\n"
            "  PSI:                  export ANSYS_LICENSE_SERVER=1055@winlic03.psi.ch\n"
            "Combine several servers with `:` (FlexLM tries each in order)."
            "  If you do not know your institute's server, ask your IT helpdesk for "
            "the ANSYS FlexLM host:port.  As a last resort, this run will fall "
            "back to trying all three above which will hang during checkout."
        )


class StaticLicenseDetector:
    """Returns a fixed license-server string; useful for tests and CI."""

    def __init__(self, license_server: str):
        self.license_server = license_server

    def detect(self) -> str:
        return self.license_server
