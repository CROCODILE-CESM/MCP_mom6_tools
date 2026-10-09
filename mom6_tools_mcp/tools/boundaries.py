"""Open-boundary tools: inspect OBC segment files and compare them with the run.

Both use mom6_tools.regional_m6toolbox (open_obc, to_t_points, on_boundary,
mask_below_floor) in the mom6-tools interpreter -- the same logic as the
02_boundary_conditions notebook.
"""

from __future__ import annotations

from ._common import case_info, err, obc_dir_of, prepare_out_dir, protected_for, resolve_hist, run_worker


def _input_dir(input_dir, caseroot):
    if input_dir:
        return input_dir
    if caseroot:
        return case_info(caseroot).get("input_dir")
    return None


def obc_summary(input_dir: str | None = None, plot: bool = False, out_dir: str | None = None,
                max_times: int = 120, caseroot: str | None = None) -> dict:
    """Sanity-check the OBC segment files a run is forced with.

    `input_dir`: the CrocoDash input dir (INPUTDIR in user_nl_mom; the
    forcing_obc_segment_*.nc may be there or in its ocn/, ocean/ or ocnice/), or
    pass `caseroot` to read it from user_nl_mom.

    Per segment: orientation (zonal = constant lat / meridional = constant
    lon), the constant coordinate, tracer-point count, levels, number of times,
    first/last time and time step, lon/lat extent, the normal-velocity
    component; vs the model topography (ocean_topog*.nc beside the OBC files,
    sliced with on_boundary): floor max/mean, wet/land points, and a problem
    when the OBC levels stop above the sea floor.  Per variable (temp, salt, u,
    v, eta) after blanking levels below the floor and land (mask_below_floor --
    OBC files extrapolate every level): min/max/mean, NaN/fill counts where a
    value is expected, and physical-range flags.  Stats use at most `max_times`
    evenly spaced records.  plot=True writes a segment map over bathymetry and a
    per-segment temp/salt/normal-velocity/eta section PNG to `out_dir`.
    """
    try:
        input_dir = _input_dir(input_dir, caseroot)
        if not input_dir:
            return err("give input_dir (or caseroot with INPUTDIR in user_nl_mom)")
        if not obc_dir_of(input_dir):
            return err(f"no forcing_obc_segment_*.nc under {input_dir} (or ocn/, ocean/, ocnice/)")
        info = case_info(caseroot) if caseroot else None
        out_dir = prepare_out_dir(out_dir, protected_for(None, info, input_dir)) if plot else None
    except (ValueError, PermissionError, FileNotFoundError) as e:
        return err(str(e))
    return run_worker("obc_summary", {"input_dir": input_dir, "plot": plot, "out_dir": out_dir,
                                      "max_times": max_times}, timeout=1200)


def obc_vs_model(hist_dir: str | None = None, casename: str | None = None, input_dir: str | None = None,
                 var: str = "temp", out_dir: str | None = None, calendar: str | None = None,
                 caseroot: str | None = None) -> dict:
    """Does the interior follow the boundary?  Model surface along each open
    boundary vs the OBC file's surface level, over the times they share.

    `var`: "temp" (vs tos), "salt" (vs sos), "eta" (vs zos/SSH), "u", "v" or
    "normal" (the boundary-normal velocity; SSU/SSV are interpolated to tracer
    points with xgcm).  The model line is the outermost row/column of h-points
    (on_boundary against ocean_topog), the OBC is taken every other supergrid
    point so both share the same `n` -- no spatial interpolation.  OBC times are
    interpolated onto the (daily-mean) model times; noleap runs get the OBC
    converted to noleap.  Per segment: bias and RMSE of (model - OBC), max |diff|
    with where/when, an RMSE time series (<= 200 points), and a 3-panel
    Hovmoller PNG (OBC | model | difference) in `out_dir`.

    A large, growing RMSE near one boundary usually points at OBC problems
    (wrong sign/orientation, bad nudging/radiation settings, mismatched SSH
    reference) -- pair with ocean-debugger check_obc_forcing_status and
    read_mom6_params(filter="OBC").
    """
    try:
        hist_dir, casename, info = resolve_hist(hist_dir, casename, caseroot)
        input_dir = input_dir or (info or {}).get("input_dir")
        if not input_dir:
            return err("give input_dir (or caseroot with INPUTDIR in user_nl_mom)")
        out_dir = prepare_out_dir(out_dir, protected_for(hist_dir, info, input_dir))
    except (ValueError, PermissionError, FileNotFoundError) as e:
        return err(str(e))
    return run_worker("obc_vs_model", {"hist_dir": hist_dir, "casename": casename, "input_dir": input_dir, "var": var,
                                       "out_dir": out_dir, "calendar": calendar or (info or {}).get("mom_calendar")},
                      timeout=1200)
