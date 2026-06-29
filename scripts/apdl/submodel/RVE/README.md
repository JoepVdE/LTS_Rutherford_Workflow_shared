# RVE Sub-Element Pipeline (in development)

The RVE (representative volume element) sub-pipeline computes a
homogenised Nb3Sn modulus from the strand's internal subelement geometry
(Nb3Sn / Cu / Nb-barrier / bronze) and feeds that modulus into the
cablestack solve.

This pipeline is **in active development** and has not been published
yet. The code is intentionally not included in this release.

Until it lands, the cablestack uses the **70 GPa Nb3Sn standard** — the
value the compression-box current-amplification factor (1.2) is
calibrated against. The choice is stamped in every run's
`loading_cycle.json` and `metadata.json` under `nb3sn_modulus.source =
"fallback"`, so it is always auditable.

When the RVE pipeline is released, this folder will contain:

```
RVE/
├── prep.py            # builds APDL deck for the strand pair under study
├── run_pipeline.py    # spawn MAPDL in Docker, drive the solve
├── postprocess.py     # extract E_x, E_y and emit summary_<rrp>.json
└── rrp_<N>_<M>/       # per-strand-pair APDL deck + results
```

For now, all references to `--enable-rve` / `--rve-only` have been
removed from `scripts/main/main.py` and from the notebook UI. The
70 GPa fallback path is the only path.
