"""History-file tools: variables, grid, domain statistics, maps and sections.

Each tool resolves paths in-process, then runs the work in the mom6-tools
interpreter ($MOM6_TOOLS_PYTHON) through the worker.  Replies are small JSON;
figures are written as PNG/GIF files whose paths are returned (view them).
"""

from __future__ import annotations

from ._common import err, prepare_out_dir, protected_for, resolve_hist, run_worker


def _resolve(hist_dir, casename, caseroot=None):
    try:
        return resolve_hist(hist_dir, casename, caseroot)
    except Exception as e:
        raise ValueError(str(e))


def _cal(calendar, info):
    return calendar or (info or {}).get("mom_calendar")


def list_history_variables(hist_dir: str | None = None, casename: str | None = None, stream: str = "sfc",
                           caseroot: str | None = None) -> dict:
    """What is in one MOM6 history stream.

    `hist_dir` is the directory holding `<casename>.mom6.h.<stream>.*.nc` (the
    archive ocn/hist, or RUNDIR before archiving); `casename` is inferred if only
    one case is there.  Or pass `caseroot` instead.  `stream` is e.g. "sfc"
    (daily surface), "native" (monthly, model layers), "z" (monthly, z-levels),
    "static" -- see find_mom6_output for what exists.

    Returns per variable: name, dims, shape (without time), C-grid position
    (h/u/v/q), vertical dim, units, long_name; plus n_files, total n_time, first
    and last decoded time, the time-axis calendar attribute and the calendar
    actually used for decoding (MOM6 labels proleptic-Gregorian axes "gregorian";
    noleap stays noleap), and which SSH name the stream uses (zos vs SSH).
    """
    try:
        hist_dir, casename, _ = _resolve(hist_dir, casename, caseroot)
    except ValueError as e:
        return err(str(e))
    return run_worker("list_vars", {"hist_dir": hist_dir, "casename": casename, "stream": stream}, timeout=300)


def describe_grid(hist_dir: str | None = None, casename: str | None = None, plot: bool = False,
                  out_dir: str | None = None, caseroot: str | None = None) -> dict:
    """Describe the model grid from the static + ocean_geometry history files.

    Returns dims; lon/lat ranges and longitude convention (-180..180 / 0..360);
    whether the domain straddles the dateline (then use 0..360 when comparing);
    wet fraction; depth min/max/mean/median over wet cells; total wet area (km2);
    dx/dy (km) mean/min/max and the smallest wet-cell spacing; a rough
    barotropic gravity-wave time-step bound min(dx / sqrt(g H)) (s) to sanity-check
    DT_BT / DT; nominal degree spacing.  With plot=True a bathymetry map PNG is
    written to `out_dir` (default: a new temp dir) and its path returned.
    """
    try:
        hist_dir, casename, info = _resolve(hist_dir, casename, caseroot)
        out_dir = prepare_out_dir(out_dir, protected_for(hist_dir, info)) if plot else None
    except (ValueError, PermissionError) as e:
        return err(str(e))
    return run_worker("describe_grid", {"hist_dir": hist_dir, "casename": casename, "plot": plot, "out_dir": out_dir},
                      timeout=300)


def surface_stats(hist_dir: str | None = None, casename: str | None = None, variables: list[str] | None = None,
                  start: str | None = None, end: str | None = None, calendar: str | None = None,
                  stream: str = "sfc", caseroot: str | None = None) -> dict:
    """Domain-wide health check of surface fields over time -- the fastest way to
    see *when* a run started going wrong.

    For each variable (default tos, sos, zos/SSH, speed; SSH aliases are
    resolved automatically; 3-D variables use their top level) and each record
    in [start, end] (dates "YYYY-MM-DD", inclusive): area-weighted wet-cell mean,
    min, max, and the fraction of wet cells that are NaN/fill.  Returns per
    variable: overall min/max and when, first/last mean and change, physical
    range flags with the first time they trip (T < -2.5 or > 40 degC, S < 0 or
    > 45, |u|,|v| or speed > 3 m/s, |SSH| > 5 m), first time NaN/fill appears in
    wet cells, and a series downsampled to <= 200 points.  `problems` lists
    every flag in plain words; `ok` is True when there are none.

    `calendar`: "noleap" or "proleptic_gregorian"; default from the case
    (caseroot) or the file's time attribute.  Works on any stream with a time
    dim (e.g. stream="native" for monthly means -- partial months show up as
    all-fill, reported as such).
    """
    try:
        hist_dir, casename, info = _resolve(hist_dir, casename, caseroot)
    except ValueError as e:
        return err(str(e))
    return run_worker("surface_stats", {"hist_dir": hist_dir, "casename": casename, "variables": variables,
                                        "start": start, "end": end, "calendar": _cal(calendar, info), "stream": stream},
                      timeout=1200)


def plot_field(hist_dir: str | None = None, casename: str | None = None, var: str | None = None, stream: str = "sfc",
               time: str = "last", level: float | int | None = None, out_dir: str | None = None,
               derived: str | None = None, animate: bool = False, cmap: str | None = None,
               calendar: str | None = None, caseroot: str | None = None) -> dict:
    """Map one field to a PNG (or an animated GIF) and return its path + range.

    `var`: any 2-D/3-D variable of `stream` (e.g. tos, sos, zos/SSH, mlotst,
    SSU, thetao); u/v/q-point fields are drawn on their own geolonu/geolatv/
    geolonb coordinates.  `derived` instead of var: "speed" (SSU/SSV or uo/vo
    moved to tracer points with xgcm via mom6_tools' make_grid), "vorticity"
    (relative vorticity on corner points, mom6_tools relative_vorticity; also
    returns max |zeta/f|), "sigma0" (Wright EOS via mom6_tools sigma0).
    `time`: "last" (default), "first", a record index, or "YYYY-MM-DD" (nearest).
    `level`: for 3-D fields, an int = level index, a float = nearest depth in m
    on the z stream (default: top level).  Land is masked with the static wet
    masks; fill values (1e20 / 9.97e36) are blanked.  Colour range is the 2-98%
    percentile (symmetric for velocities/vorticity).
    `animate=True`: a GIF over all records (subsampled to <= 60 frames) using
    mom6_tools save_field_gif (cartopy) or save_gif.

    Coastlines are drawn only if a Natural Earth shapefile is already on disk
    (compute nodes are offline); otherwise the map is drawn without them -- it
    never fails on that.  Dateline domains are plotted on 0..360.  Output goes to
    `out_dir` (default: a new temp dir), never into case/archive/input dirs.
    """
    try:
        hist_dir, casename, info = _resolve(hist_dir, casename, caseroot)
        out_dir = prepare_out_dir(out_dir, protected_for(hist_dir, info))
    except (ValueError, PermissionError) as e:
        return err(str(e))
    return run_worker("plot_field", {"hist_dir": hist_dir, "casename": casename, "var": var, "stream": stream,
                                     "time": time, "level": level, "out_dir": out_dir, "derived": derived,
                                     "animate": animate, "cmap": cmap, "calendar": _cal(calendar, info)},
                      timeout=1500 if animate else 600)


def plot_section(hist_dir: str | None = None, casename: str | None = None, var: str = "thetao", stream: str = "z",
                 lon: float | None = None, lat: float | None = None, time: str = "last", out_dir: str | None = None,
                 calendar: str | None = None, caseroot: str | None = None) -> dict:
    """Vertical section of a 3-D field along the model row/column nearest a
    latitude (zonal section: give `lat`) or a longitude (meridional: give `lon`).

    On the "z" stream depths are the z_l levels; on the "native" stream layer
    depths come from interface heights `e` (mom6_tools layer_depths), so tracer
    sections work there too (velocities on native need stream="z").  The local
    sea floor (deptho) is drawn on top.  Returns PNG path, which row/column was
    used, min/max/mean and number of valid cells.  Monthly native/z files from a
    run that stopped mid-month are all fill -- reported in `notes`.
    """
    try:
        hist_dir, casename, info = _resolve(hist_dir, casename, caseroot)
        out_dir = prepare_out_dir(out_dir, protected_for(hist_dir, info))
    except (ValueError, PermissionError) as e:
        return err(str(e))
    return run_worker("plot_section", {"hist_dir": hist_dir, "casename": casename, "var": var, "stream": stream,
                                       "lon": lon, "lat": lat, "time": time, "out_dir": out_dir,
                                       "calendar": _cal(calendar, info)}, timeout=600)
