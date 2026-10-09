"""Compare the run's surface with the GLORYS12 reanalysis (notebook 03 approach)."""

from __future__ import annotations

import os

from ._common import err, prepare_out_dir, protected_for, resolve_hist, run_worker


def compare_glorys(hist_dir: str | None = None, casename: str | None = None, dates: list[str] | None = None,
                   variables: list[str] | None = None, glorys_root: str = "/gdex/data/d010049",
                   out_dir: str | None = None, max_dates: int = 3, plot: bool = False,
                   calendar: str | None = None, caseroot: str | None = None) -> dict:
    """Area-weighted model-vs-GLORYS12 statistics at the surface for a few days.

    `dates`: "YYYY-MM-DD" days inside the run (only the first `max_dates`, default
    3, are used: each GLORYS daily file is a ~1.3 GB global file, of which only
    the surface over the model box is read).  `variables`: model names, default
    tos, sos, zos (SSH alias resolved), compared with GLORYS thetao, so, zos.
    GLORYS is sampled onto model h-points with pointwise interpolation
    (mom6_tools to_model_grid), both fields restricted to common valid wet cells
    (common_mask), then mom6_tools area_stats gives n_cells, model_mean,
    glorys_mean, bias, RMSE and pattern correlation per variable per date.  SSH
    reference levels differ between MOM6 and GLORYS, so judge zos by pattern_r
    and de-biased error.  plot=True writes model-minus-GLORYS difference maps
    to `out_dir`.

    `glorys_root` must be readable (/gdex is mounted on Casper/Derecho login and
    compute nodes, not elsewhere); a clear error is returned otherwise.
    """
    if not dates:
        return err("give dates=['YYYY-MM-DD', ...] inside the run")
    if not os.path.isdir(glorys_root) or not os.access(glorys_root, os.R_OK | os.X_OK):
        return err(f"GLORYS root {glorys_root} is not readable from this host",
                   hint="needs /gdex mounted (NCAR Casper/Derecho); or pass glorys_root to a local copy laid out as <root>/<YYYY>/*_<YYYYMMDD>_*.nc")
    try:
        hist_dir, casename, info = resolve_hist(hist_dir, casename, caseroot)
        out_dir = prepare_out_dir(out_dir, protected_for(hist_dir, info)) if plot else None
    except (ValueError, PermissionError, FileNotFoundError) as e:
        return err(str(e))
    capped = list(dates)[: max(1, int(max_dates))]
    r = run_worker("compare_glorys", {"hist_dir": hist_dir, "casename": casename, "dates": capped,
                                      "variables": variables, "glorys_root": glorys_root, "out_dir": out_dir,
                                      "max_dates": max_dates, "plot": plot,
                                      "calendar": calendar or (info or {}).get("mom_calendar")}, timeout=1800)
    if len(dates) > len(capped) and isinstance(r, dict):
        r.setdefault("notes", []).append(f"{len(dates) - len(capped)} date(s) dropped by max_dates={max_dates}")
    return r
