"""Light, in-process tools: locate MOM6 output and summarise ocean.stats.

Stdlib only (glob, XML, text parsing) -- these answer in milliseconds and work
even when the mom6-tools environment is unavailable.
"""

from __future__ import annotations

import datetime as _dt
import glob
import math
import os
import re
from pathlib import Path

from ._common import (
    STREAM_RE,
    case_info,
    caseroot_from_rundir,
    err,
    is_history_dir,
    mom6_files,
    mom_calendar,
    obc_dir_of,
    run_worker,
)

MAX_SERIES = 200


def _scan(d: str, casename: str | None) -> dict:
    """Group *.mom6.*.nc files in one directory by stream."""
    streams: dict[str, dict] = {}
    restarts, ics = [], []
    for f in mom6_files(d, casename):
        b = os.path.basename(f)
        m = STREAM_RE.match(b)
        if not m:
            continue
        s, date = m["stream"], m["date"]
        if s == "r" or s.startswith("r."):
            restarts.append(b)
            continue
        if s == "ic" or s.startswith("ic."):
            ics.append(b)
            continue
        s = s[2:] if s.startswith("h.") else s
        e = streams.setdefault(s, {"n_files": 0, "dates": [], "bytes": 0})
        e["n_files"] += 1
        e["bytes"] += os.path.getsize(f)
        if date:
            e["dates"].append(date)
    out = {}
    for s, e in sorted(streams.items()):
        r = {"n_files": e["n_files"], "size_MB": round(e["bytes"] / 1e6, 1)}
        if e["dates"]:
            ds = sorted(e["dates"])
            r["first"], r["last"] = ds[0], ds[-1]
        out[s] = r
    return {"streams": out, "restart_files": restarts[:10], "n_restart_files": len(restarts), "ic_files": ics}


def find_mom6_output(caseroot: str | None = None, rundir: str | None = None, archive_dir: str | None = None,
                     casename: str | None = None, input_dir: str | None = None) -> dict:
    """Locate everything a MOM6 regional run has produced, and what it was forced with.

    Give a CESM `caseroot` (best -- RUNDIR, DOUT_S_ROOT, CASE, CALENDAR and the
    CrocoDash INPUTDIR from user_nl_mom are all read from it), or any of `rundir`,
    `archive_dir` (the case's short-term-archive root, e.g.
    /glade/derecho/scratch/$USER/archive/<case>, or its ocn/hist directly) and
    `input_dir`.  Read-only; stdlib only, so it is instant.

    Returns, per location (RUNDIR and archive ocn/hist):
      * streams: {stream: {n_files, size_MB, first, last}} -- e.g. "sfc" (daily
        surface), "native" / "z" (monthly 3-D), "static", "ocean_geometry";
        date ranges come from file names (YYYY-MM for monthly files).
      * ic_files / restart files present.
    Plus: whether static + ocean_geometry exist (needed by every map/grid tool),
    ocean.stats / ocean.stats.nc in RUNDIR (for summarize_run), and in the input
    dir the OBC segment files, ocean_topog / hgrid / vgrid and init files.

    `hist_dir_for_tools` in the reply is the directory to pass as `hist_dir` to
    the other tools.  Typical next steps: summarize_run (did it blow up?),
    describe_grid, surface_stats, render_notebook.
    """
    info = None
    notes = []
    try:
        if caseroot:
            info = case_info(caseroot)
            casename = casename or info.get("CASE")
            rundir = rundir or info.get("RUNDIR")
            if not archive_dir and info.get("DOUT_S_ROOT"):
                archive_dir = info["DOUT_S_ROOT"]
            input_dir = input_dir or info.get("input_dir")
        elif rundir and not casename:
            cr = caseroot_from_rundir(rundir)
            if cr:
                info = case_info(cr)
                casename = info.get("CASE")
                input_dir = input_dir or info.get("input_dir")
                archive_dir = archive_dir or info.get("DOUT_S_ROOT")
                notes.append(f"caseroot found via RUNDIR/CASEROOT: {cr}")
    except Exception as e:
        return err(f"could not read case: {e}")

    out: dict = {"casename": casename, "locations": {}}
    if info:
        out["case"] = {k: info.get(k) for k in ("CASEROOT", "RUNDIR", "DOUT_S", "DOUT_S_ROOT", "CALENDAR",
                                                "mom_calendar", "RUN_STARTDATE", "STOP_OPTION", "STOP_N", "COMPSET")}
    hist_candidates = []
    if archive_dir:
        a = Path(archive_dir).expanduser()
        hist = a if a.name == "hist" else a / "ocn" / "hist"
        hist_candidates.append(("archive", str(hist)))
    if rundir:
        hist_candidates.append(("rundir", str(Path(rundir).expanduser())))
    best = None
    for label, d in hist_candidates:
        if not os.path.isdir(d):
            out["locations"][label] = {"path": d, "exists": False}
            continue
        sc = _scan(d, casename)
        if casename is None and sc["streams"]:
            names = {os.path.basename(f).split(".mom6.")[0] for f in mom6_files(d)}
            if len(names) == 1:
                casename = names.pop()
                out["casename"] = casename
        sc["path"] = d
        sc["exists"] = True
        sc["has_static"] = "static" in sc["streams"]
        sc["has_ocean_geometry"] = "ocean_geometry" in sc["streams"] or os.path.exists(os.path.join(d, "ocean_geometry.nc"))
        out["locations"][label] = sc
        if best is None and is_history_dir(sc["streams"]):
            best = d
    out["hist_dir_for_tools"] = best
    if best is None:
        notes.append("no directory with both static and history streams found -- has the run written output / been archived?")

    if rundir and os.path.isdir(os.path.expanduser(rundir)):
        rd = Path(rundir).expanduser()
        stats = {n: (rd / n).exists() for n in ("ocean.stats", "ocean.stats.nc")}
        out["ocean_stats"] = {"rundir": str(rd), **stats}
        if not any(stats.values()):
            notes.append("RUNDIR has no ocean.stats (cleaned or never ran); summarize_run needs it")
        logs = sorted(glob.glob(str(rd / "ocn.log.*")))[-3:]
        if logs:
            out["recent_ocn_logs"] = [os.path.basename(x) for x in logs]

    if input_dir:
        od = obc_dir_of(input_dir)
        base = Path(od or input_dir).expanduser()
        if base.is_dir():
            def names(p):
                return sorted(os.path.basename(x) for x in glob.glob(str(base / p)))
            out["input"] = {
                "input_dir": str(Path(input_dir).expanduser()),
                "obc_dir": od,
                "obc_segments": names("forcing_obc_segment_*.nc"),
                "topog": names("ocean_topog*.nc"),
                "hgrid": names("ocean_hgrid*.nc"),
                "vgrid": names("ocean_vgrid*.nc") + names("vcoord*.nc"),
                "init": names("init_*.nc"),
                "tides": names("tu_*.nc") + names("tz_*.nc"),
            }
            if not od:
                notes.append(f"no forcing_obc_segment_*.nc under {input_dir} (or ocn/, ocean/, ocnice/)")
        else:
            out["input"] = {"input_dir": input_dir, "exists": False}
    out["notes"] = notes
    return out


# --------------------------------------------------------------------------
# ocean.stats
# --------------------------------------------------------------------------

_TAGS = {"En": "energy", "CFL": "max_cfl", "SL": "mean_sea_level", "M": "total_mass", "S": "mean_salt",
         "T": "mean_temp", "Me": "frac_mass_err", "Se": "salt_err", "Te": "temp_err"}


def _num(s: str) -> float:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        if s.lower() in ("nan", "-nan", "+nan"):
            return float("nan")
        if s.lower() in ("inf", "+inf", "infinity"):
            return float("inf")
        if s.lower() in ("-inf", "-infinity"):
            return float("-inf")
        # Fortran overflow "*******" or exponent without E ("1.0-100")
        if set(s) <= {"*"}:
            return float("nan")
        m = re.fullmatch(r"([-+]?\d*\.?\d+)([-+]\d+)", s)
        if m:
            return float(m[1]) * 10 ** int(m[2])
        raise


def parse_ocean_stats(path: str) -> list[dict]:
    """Rows of a MOM6 text ocean.stats file.  Header lines (repeated after each
    restart) are skipped; columns are recognised by their tags (En, CFL, ...),
    so configurations with fewer/more columns parse too."""
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                step, day, trunc = int(parts[0]), _num(parts[1]), _num(parts[2])
            except ValueError:
                continue
            r = {"step": step, "day": day, "truncs": trunc}
            for p in parts[3:]:
                bits = p.split(None, 1)
                if len(bits) == 2 and bits[0] in _TAGS:
                    try:
                        r[_TAGS[bits[0]]] = _num(bits[1])
                    except ValueError:
                        r[_TAGS[bits[0]]] = float("nan")
            rows.append(r)
    return rows


def day_to_date(day: float, calendar: str | None) -> str | None:
    """MOM6 ocean.stats 'Day' (days since 0001-01-01) -> ISO date in the run's calendar."""
    if day is None or not math.isfinite(day):
        return None
    cal = (calendar or "proleptic_gregorian").lower()
    whole = int(math.floor(day))
    frac = day - whole
    hh = int(round(frac * 24))
    if cal == "noleap":
        cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
        y, doy = divmod(whole, 365)
        mo = max(i for i in range(12) if cum[i] <= doy)
        return f"{y + 1:04d}-{mo + 1:02d}-{doy - cum[mo] + 1:02d}" + (f" {hh:02d}h" if hh else "")
    if cal == "360_day":
        y, r = divmod(whole, 360)
        mo, d = divmod(r, 30)
        return f"{y + 1:04d}-{mo + 1:02d}-{d + 1:02d}"
    try:
        dt = _dt.datetime(1, 1, 1) + _dt.timedelta(days=day)
        return dt.strftime("%Y-%m-%d") + (dt.strftime(" %Hh") if dt.hour else "")
    except OverflowError:
        return None


def _ds(n: int) -> list[int]:
    if n <= MAX_SERIES:
        return list(range(n))
    return sorted({round(i * (n - 1) / (MAX_SERIES - 1)) for i in range(MAX_SERIES)})


def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def summarize_run(rundir_or_caseroot: str, calendar: str | None = None) -> dict:
    """Did the run blow up?  Summarise MOM6's ocean.stats for a run.

    `rundir_or_caseroot` may be a CESM CASEROOT (RUNDIR is read from
    env_run.xml), a RUNDIR, or the path of an ocean.stats / ocean.stats.nc file.
    The calendar for turning MOM6's "Day" column (days since 0001-01-01) into
    dates comes from `calendar` ("noleap", "proleptic_gregorian", "360_day"),
    else from the case's CALENDAR (CASEROOT, or RUNDIR/CASEROOT), else gregorian.

    Reported (text ocean.stats; ocean.stats.nc is used only when the text file
    is missing, via the worker):
      * n_records, first/last step, first/last day -> date, `last_date_reached`;
      * energy (En, m2 s-2): first, last, max (+ when), last/first ratio;
      * max CFL: max and when, n records > 0.5 (MOM6's default CFL_TRUNCATE)
        and > 1.0;
      * truncations: total, first record with any (MOM6 truncates velocities when
        CFL exceeds CFL_TRUNCATE -- a precursor to blow-ups);
      * drift: total mass relative change, mean T / S change, max |Me|/|Se|/|Te|;
      * flags: NaN anywhere, energy growth > 100x, CFL > 0.5, truncations, and a
        `verdict` string;
      * downsampled (<=200 points) series of day/date/energy/CFL/truncs.

    Pair with: ocean-debugger find_errors / classify_error (log text),
    check_cfl (DT vs resolution); here max CFL tells you how close the run is.
    """
    p = Path(rundir_or_caseroot).expanduser()
    notes = []
    caseroot = None
    if p.is_file():
        stats_txt = p if p.suffix != ".nc" else p.with_suffix("")
        stats_nc = p if p.suffix == ".nc" else Path(str(p) + ".nc")
        rundir = p.parent
    else:
        if (p / "env_run.xml").exists():
            caseroot = str(p)
            try:
                rundir = Path(case_info(p)["RUNDIR"])
            except Exception as e:
                return err(f"could not read RUNDIR from {p}: {e}")
        else:
            rundir = p
        stats_txt, stats_nc = rundir / "ocean.stats", rundir / "ocean.stats.nc"
    if caseroot is None:
        caseroot = caseroot_from_rundir(rundir)
    cal_src = "argument"
    if not calendar and caseroot:
        try:
            calendar = case_info(caseroot).get("mom_calendar")
            cal_src = f"CALENDAR in {caseroot}/env_run.xml"
        except Exception:
            pass
    if not calendar:
        calendar, cal_src = "proleptic_gregorian", "default (no case found)"
    calendar = mom_calendar(calendar)

    rows = []
    source = None
    if stats_txt.exists():
        rows = parse_ocean_stats(str(stats_txt))
        source = str(stats_txt)
    elif stats_nc.exists():
        r = run_worker("ocean_stats_nc", {"path": str(stats_nc)}, timeout=300)
        if "error" in r:
            return r
        cols = r["columns"]
        n = len(cols.get("Day", []))
        for i in range(n):
            row = {"step": None, "day": cols["Day"][i], "truncs": (cols.get("Ntrunc") or [None] * n)[i],
                   "energy": (cols.get("En") or [None] * n)[i],
                   "max_cfl": max([x for x in ((cols.get("max_CFL_trans") or [None] * n)[i], (cols.get("max_CFL_lin") or [None] * n)[i]) if x is not None], default=None),
                   "total_mass": (cols.get("Mass") or [None] * n)[i]}
            rows.append(row)
        source = str(stats_nc)
    else:
        return err(f"no ocean.stats or ocean.stats.nc in {rundir}",
                   hint="RUNDIR may have been cleaned; ocean.stats is not short-term archived. Check ocn.log in the archive logs/ with the ocean-debugger server.")
    if not rows:
        return err(f"{source} has no data rows (run died before the first energy report?)")

    def col(k):
        return [r.get(k) for r in rows]

    out: dict = {"source": source, "calendar": calendar, "calendar_source": cal_src, "n_records": len(rows)}
    day = col("day")
    out["first_step"], out["last_step"] = rows[0]["step"], rows[-1]["step"]
    out["first_day"], out["last_day"] = day[0], day[-1]
    out["first_date"], out["last_date_reached"] = day_to_date(day[0], calendar), day_to_date(day[-1], calendar)
    if len(day) > 1 and all(_fin(x) for x in day):
        out["simulated_days"] = round(day[-1] - day[0], 4)
        gaps = [b - a for a, b in zip(day, day[1:])]
        if any(g <= 0 for g in gaps):
            notes.append("Day goes backwards/repeats: the file spans more than one run segment (restart or rerun appended)")
    flags = []
    has_nan = [i for i, r in enumerate(rows) if any(isinstance(v, float) and math.isnan(v) for k, v in r.items() if k not in ("step",) and v is not None)]
    if has_nan:
        flags.append(f"NaN first appears at record {has_nan[0]} (day {day[has_nan[0]]} = {day_to_date(day[has_nan[0]], calendar)})")
    en = col("energy")
    fe = [(i, x) for i, x in enumerate(en) if _fin(x)]
    if fe:
        imax = max(fe, key=lambda t: t[1])[0]
        out["energy_units"] = "J (total, from ocean.stats.nc En)" if str(source).endswith(".nc") else "m2 s-2 (energy per unit mass)"
        out["energy"] = {"first": fe[0][1], "last": fe[-1][1], "max": en[imax], "max_at": day_to_date(day[imax], calendar),
                         "last_over_first": (fe[-1][1] / fe[0][1]) if fe[0][1] else None}
        if fe[0][1] and en[imax] / fe[0][1] > 100:
            flags.append(f"energy grew {en[imax] / fe[0][1]:.3g}x from the first record (max at {out['energy']['max_at']})")
    cfl = col("max_cfl")
    fc = [(i, x) for i, x in enumerate(cfl) if _fin(x)]
    if fc:
        imax = max(fc, key=lambda t: t[1])[0]
        out["cfl"] = {"max": cfl[imax], "max_at": day_to_date(day[imax], calendar), "last": fc[-1][1],
                      "n_records_gt_0p5": sum(1 for _, x in fc if x > 0.5), "n_records_gt_1": sum(1 for _, x in fc if x > 1.0)}
        if cfl[imax] > 0.5:
            flags.append(f"max CFL {cfl[imax]:.3g} at {out['cfl']['max_at']} exceeds 0.5 (default CFL_TRUNCATE) -- reduce DT or check bathymetry/OBC")
    tr = col("truncs")
    ft = [(i, x) for i, x in enumerate(tr) if _fin(x) and x > 0]
    out["truncations"] = {"total": sum(x for x in tr if _fin(x)), "records_with_truncations": len(ft)}
    if ft:
        out["truncations"]["first_at"] = day_to_date(day[ft[0][0]], calendar)
        flags.append(f"{len(ft)} record(s) with velocity truncations, first at {out['truncations']['first_at']}")
    drift = {}
    m = [x for x in col("total_mass") if _fin(x)]
    if len(m) > 1 and m[0]:
        drift["mass_rel_change"] = (m[-1] - m[0]) / m[0]
    for k, lab in (("mean_temp", "mean_temp_change_degC"), ("mean_salt", "mean_salt_change_psu"), ("mean_sea_level", "mean_sea_level_change_m")):
        v = [x for x in col(k) if _fin(x)]
        if len(v) > 1:
            drift[lab] = v[-1] - v[0]
            drift[k.replace("mean_", "") + "_first_last"] = [v[0], v[-1]]
    for k in ("frac_mass_err", "salt_err", "temp_err"):
        v = [abs(x) for x in col(k) if _fin(x)]
        if v:
            drift[f"max_abs_{k}"] = max(v)
    out["drift"] = drift
    idx = _ds(len(rows))
    out["series"] = {"date": [day_to_date(day[i], calendar) for i in idx], "day": [day[i] for i in idx],
                     "energy": [en[i] if _fin(en[i]) else None for i in idx],
                     "max_cfl": [cfl[i] if _fin(cfl[i]) else None for i in idx],
                     "truncs": [tr[i] if _fin(tr[i]) else None for i in idx]}
    if len(idx) < len(rows):
        out["series_downsampled_from"] = len(rows)
    out["flags"] = flags
    if has_nan:
        out["verdict"] = f"BLEW UP: NaNs appear by {day_to_date(day[has_nan[0]], calendar)}"
    elif flags:
        out["verdict"] = "RAN, with warnings: " + "; ".join(flags[:2])
    else:
        out["verdict"] = f"healthy: {len(rows)} records to {out['last_date_reached']}, max CFL {out.get('cfl', {}).get('max')}, no truncations"
    out["notes"] = notes
    return out
