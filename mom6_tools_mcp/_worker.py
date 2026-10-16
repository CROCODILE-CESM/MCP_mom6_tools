"""Heavy-lifting worker for the mom6-tools MCP server.

Run as a separate process, in the interpreter named by ``MOM6_TOOLS_PYTHON``
(an environment with xarray, xgcm, dask, cartopy, cmocean, matplotlib, netCDF4,
pillow and mom6_tools installed):

    python _worker.py <op> '<json args>'

Exactly one JSON document is written to stdout, as the last line.  Everything
else (library chatter, warnings, progress) goes to stderr.  On failure the JSON
is ``{"error": ..., "traceback": ...}`` and the exit code is 1.

Rules this file follows:
  * matplotlib is switched to Agg before anything else imports it;
  * history files are opened lazily with ``chunks={"time": 1}`` and reduced
    lazily -- arrays are never returned, only scalars and short series;
  * any series is capped at MAX_SERIES points;
  * files are only ever *read* from case / archive / input directories --
    PNGs, GIFs and notebooks are written only to the ``out_dir`` passed in.
"""

import json
import os
import sys

# The package directory sits at sys.path[0] when this file is run by path; keep
# its modules (server.py, tools/) from shadowing anything in the worker env.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import glob  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

MAX_SERIES = 200
FILL_ABS = 1e19  # MOM6 writes 1e20 / netCDF default 9.97e36 as fill; anything this big is fill
GIF_MAX_FRAMES = 60

# Physical-range flags: (kind) -> (low, high, description).  `None` = no bound.
RANGES = {
    "temperature": (-2.5, 40.0, "T < -2.5 degC or T > 40 degC"),
    "salinity": (0.0, 45.0, "S < 0 or S > 45 psu"),
    "velocity": (-3.0, 3.0, "|velocity| > 3 m/s"),
    "speed": (None, 3.0, "speed > 3 m/s"),
    "ssh": (-5.0, 5.0, "|SSH| > 5 m"),
}

SSH_CANDIDATES = ("zos", "SSH", "ssh")
AREA_CANDIDATES = ("areacello", "area_t", "Ah")
DEPTH_CANDIDATES = ("deptho", "depth_ocean", "D")
VERT_DIMS = ("zl", "z_l", "zi", "z_i")


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------


def _np():
    import numpy as np

    return np


def _xr():
    import xarray as xr

    return xr


def _tb():
    """The mom6_tools regional toolbox (imported lazily; pulls in xgcm/cmocean)."""
    from mom6_tools import regional_m6toolbox

    return regional_m6toolbox


def _clean(obj):
    """Make an object JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    np = _np()
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, (str, int, bool, type(None))):
        return obj
    return str(obj)  # datetimes, cftime objects, paths, anything else


def _f(x, nd=None):
    """float() that maps NaN to None and optionally rounds."""
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def _tstr(t):
    s = str(t)
    return s[:19].replace("T", " ")


def _downsample_idx(n, cap=MAX_SERIES):
    np = _np()
    if n <= cap:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, cap).round().astype(int))


def _out_path(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def norm_calendar(cal):
    """Map CESM/CF calendar names to the calendar decode_mom6_time should use.

    MOM6/FMS labels proleptic-Gregorian time axes as "gregorian" (CF: mixed
    Julian/Gregorian), so gregorian/standard/julian all map to proleptic_gregorian.
    """
    if not cal:
        return "proleptic_gregorian"
    c = str(cal).strip().lower().replace("-", "_")
    if c in ("noleap", "no_leap", "365_day", "365day"):
        return "noleap"
    if c in ("360_day", "360day"):
        return "360_day"
    if c in ("all_leap", "366_day"):
        return "all_leap"
    return "proleptic_gregorian"


# --------------------------------------------------------------------------
# finding and opening files
# --------------------------------------------------------------------------


def stream_files(hist_dir, casename, stream):
    """Files of one history stream, trying the CESM naming variants in turn."""
    esc = glob.escape(hist_dir)
    pats = [
        f"{casename}.mom6.h.{stream}.*.nc",
        f"{casename}.mom6.h.{stream}.nc",
        f"{casename}.mom6.h.{stream}_*.nc",
        f"{casename}.mom6.{stream}.*.nc",
        f"{casename}.mom6.{stream}.nc",
    ]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(esc, p)))
        if hits:
            return hits
    if stream == "ocean_geometry":
        p = os.path.join(hist_dir, "ocean_geometry.nc")
        if os.path.exists(p):
            return [p]
    return []


def _n_time(path):
    """Length of the time dim of a file (None if unknown / no time dim)."""
    try:
        import netCDF4

        with netCDF4.Dataset(path) as nc:
            return len(nc.dimensions["time"]) if "time" in nc.dimensions else None
    except Exception:
        return None


def available_streams(hist_dir, casename):
    names = set()
    for f in glob.glob(os.path.join(glob.escape(hist_dir), f"{casename}.mom6.*.nc")):
        b = os.path.basename(f)[len(casename) + len(".mom6.") : -3]
        b = re.sub(r"^h\.", "", b)
        b = re.sub(r"[._]\d{4}(-\d{2}){0,2}(-\d{5})?$", "", b)
        names.add(b)
    return sorted(names)


def _no_files(hist_dir, casename, stream):
    return FileNotFoundError(
        f"no files for stream {stream!r} of case {casename!r} in {hist_dir}; "
        f"streams present: {available_streams(hist_dir, casename)}"
    )


def open_grid(hist_dir, casename):
    """(static, geom) Datasets on history-file dim names (yh, xh, yq, xq).

    `geom` comes from ocean_geometry (renamed lath/lonh/latq/lonq -> yh/xh/yq/xq);
    if that file is missing a minimal stand-in is assembled from static.
    """
    xr = _xr()
    sf = stream_files(hist_dir, casename, "static")
    if not sf:
        raise _no_files(hist_dir, casename, "static")
    static = xr.open_dataset(sf[0], decode_times=False).squeeze(drop=True)
    gf = stream_files(hist_dir, casename, "ocean_geometry")
    if gf:
        geom = xr.open_dataset(gf[0], decode_times=False)
        ren = {k: v for k, v in {"lath": "yh", "lonh": "xh", "latq": "yq", "lonq": "xq"}.items() if k in geom.dims}
        geom = geom.rename_dims(ren).drop_vars([k for k in ren if k in geom.variables])
    else:
        m = {
            "geolon": "geolon", "geolat": "geolat",
            "geolonb": "geolon_c", "geolatb": "geolat_c",
            "geolonu": "geolon_u", "geolatu": "geolat_u",
            "geolonv": "geolon_v", "geolatv": "geolat_v",
        }
        geom = xr.Dataset({k: static[v] for k, v in m.items() if v in static})
    geom.attrs["_has_geometry_file"] = bool(gf)
    return static, geom


def pos_of(dims):
    """C-grid position of a field from its horizontal dims."""
    d = set(dims)
    if {"yh", "xh"} <= d:
        return "h"
    if {"yh", "xq"} <= d:
        return "u"
    if {"yq", "xh"} <= d:
        return "v"
    if {"yq", "xq"} <= d:
        return "q"
    return None


POS_COORDS = {
    "h": ("geolon", "geolat"),
    "u": ("geolonu", "geolatu"),
    "v": ("geolonv", "geolatv"),
    "q": ("geolonb", "geolatb"),
}
POS_WET = {"h": "wet", "u": "wet_u", "v": "wet_v", "q": "wet_c"}
POS_AREA = {"h": ("areacello", "area_t"), "u": ("areacello_cu",), "v": ("areacello_cv",), "q": ("areacello_bu",)}


def lonlat(geom, static, pos):
    lo, la = POS_COORDS[pos]
    if lo in geom and la in geom:
        return geom[lo], geom[la]
    alt = {"h": ("geolon", "geolat"), "u": ("geolon_u", "geolat_u"), "v": ("geolon_v", "geolat_v"), "q": ("geolon_c", "geolat_c")}[pos]
    if alt[0] in static:
        return static[alt[0]], static[alt[1]]
    raise KeyError(f"no 2-D lon/lat for {pos}-points in static/ocean_geometry")


def wet_of(static, geom, pos):
    name = POS_WET[pos]
    if name in static:
        return static[name]
    if pos == "h" and "wet" in geom:
        return geom["wet"]
    return None


def area_of(static, geom, pos):
    for name in POS_AREA[pos]:
        if name in static:
            return static[name]
    if pos == "h" and "Ah" in geom:
        return geom["Ah"]
    if pos == "q" and "Aq" in geom:
        return geom["Aq"]
    return None


def depth_of(static, geom):
    for name in DEPTH_CANDIDATES:
        if name in static:
            return static[name], name
        if name in geom:
            return geom[name], name
    return None, None


def infer_calendar(hist_dir, casename, stream="sfc"):
    """The calendar attribute of a stream's time axis, normalised."""
    xr = _xr()
    files = stream_files(hist_dir, casename, stream)
    if not files:
        return "proleptic_gregorian", None
    with xr.open_dataset(files[0], decode_times=False) as ds:
        raw = ds["time"].attrs.get("calendar") if "time" in ds else None
    return norm_calendar(raw), raw


def open_stream(hist_dir, casename, stream, calendar=None, with_coords=True):
    """Open a history stream lazily (chunks time=1), decode its calendar.

    Returns (ds, static, geom, calendar_used).
    """
    xr = _xr()
    files = stream_files(hist_dir, casename, stream)
    if not files:
        raise _no_files(hist_dir, casename, stream)
    nonempty = [f for f in files if _n_time(f) != 0]
    if not nonempty:
        raise ValueError(
            f"stream {stream!r}: all {len(files)} file(s) have 0 time records ({os.path.basename(files[0])} ...). "
            "MOM6 wrote the file but never a record: the averaging period did not complete before the run "
            "ended/crashed, or the file was not closed cleanly -- check the run end and the diag_table frequency.")
    files = nonempty
    if len(files) == 1:
        ds = xr.open_dataset(files[0], decode_times=False, chunks={"time": 1})
    else:
        ds = xr.open_mfdataset(
            files, decode_times=False, chunks={"time": 1}, combine="nested", concat_dim="time",
            data_vars="minimal", coords="minimal", compat="override", parallel=False,
        )
    raw_cal = ds["time"].attrs.get("calendar") if "time" in ds else None
    cal = norm_calendar(calendar) if calendar else norm_calendar(raw_cal)
    if "time" in ds.dims:
        ds = _tb().decode_mom6_time(ds, calendar=cal)
    static, geom = open_grid(hist_dir, casename)
    if with_coords:
        coords = {}
        for pos in ("h", "u", "v", "q"):
            try:
                lo, la = lonlat(geom, static, pos)
            except KeyError:
                continue
            if set(lo.dims) <= set(ds.dims):
                coords[POS_COORDS[pos][0]] = lo.reset_coords(drop=True)
                coords[POS_COORDS[pos][1]] = la.reset_coords(drop=True)
        ds = ds.assign_coords(coords)
    return ds, static, geom, cal


def mask_fill(da):
    """Blank fill values (|x| >= 1e19) that were not caught by _FillValue."""
    return da.where(abs(da) < FILL_ABS)


def pick_time(da, time):
    """Select one record: "last", "first", an int index, or a date string."""
    if "time" not in da.dims:
        return da, None
    n = da.sizes["time"]
    if time is None or time == "last":
        i = n - 1
    elif time == "first":
        i = 0
    elif isinstance(time, int) or (isinstance(time, str) and re.fullmatch(r"-?\d+", time)):
        i = int(time)
    else:
        idx = da.indexes["time"]
        ts = str(time).strip()
        if re.fullmatch(r"\d{1,4}-\d{1,2}-\d{1,2}", ts):
            ts += " 12:00"  # a bare date means that day: daily means are stamped at noon
        key = _time_like(idx, ts)
        i = int(abs(_as_days(idx) - _as_days_scalar(key, idx)).argmin())
    out = da.isel(time=i)
    return out, _tstr(out["time"].values)


def _time_like(idx, s):
    """Parse 'YYYY-MM-DD[ HH:MM]' into the same kind of object as the index."""
    np = _np()
    m = re.match(r"(\d{1,4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?", s)
    if not m:
        raise ValueError(f"cannot parse time {s!r}; use YYYY-MM-DD, 'first', 'last' or an index")
    y, mo, d = int(m[1]), int(m[2]), int(m[3])
    hh, mm = int(m[4] or 0), int(m[5] or 0)
    if hasattr(idx, "calendar"):  # CFTimeIndex
        import cftime

        return cftime.datetime(y, mo, d, hh, mm, calendar=idx.calendar)
    return np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{hh:02d}:{mm:02d}")


def _as_days(idx):
    np = _np()
    if hasattr(idx, "calendar"):
        import cftime

        return np.asarray(cftime.date2num(list(idx), "days since 0001-01-01", calendar=idx.calendar))
    return np.asarray(idx.values.astype("datetime64[s]").astype("int64")) / 86400.0


def _as_days_scalar(t, idx):
    np = _np()
    if hasattr(idx, "calendar"):
        import cftime

        return float(cftime.date2num(t, "days since 0001-01-01", calendar=idx.calendar))
    return float(np.datetime64(t, "s").astype("int64")) / 86400.0


def time_slice(ds, start, end):
    if "time" not in ds.dims or (not start and not end):
        return ds
    idx = ds.indexes["time"]
    days = _as_days(idx)
    keep = _np().ones(len(days), bool)
    if start:
        keep &= days >= _as_days_scalar(_time_like(idx, start), idx) - 1e-6
    if end:
        keep &= days <= _as_days_scalar(_time_like(idx, end), idx) + 0.999
    return ds.isel(time=_np().nonzero(keep)[0])


def var_kind(name, units=""):
    n = name.lower()
    u = (units or "").strip()
    if n in ("tos", "thetao", "sst", "temp", "tob", "thetaoga", "tosga") or u in ("degC", "deg_C", "Celsius", "C"):
        return "temperature"
    if n in ("sos", "so", "sss", "salt", "sob", "soga", "sosga") or u in ("psu", "1e-3", "ppt"):
        return "salinity"
    if n == "speed":
        return "speed"
    if n in ("ssu", "ssv", "uo", "vo", "u", "v") or u == "m s-1":
        return "velocity"
    if n in SSH_CANDIDATES or n == "eta":
        return "ssh"
    return None


def resolve_var(ds, var):
    """Resolve a requested name, mapping SSH aliases (zos/SSH/ssh) onto what exists."""
    if var in ds:
        return var
    if var in SSH_CANDIDATES:
        for c in SSH_CANDIDATES:
            if c in ds:
                return c
    low = {v.lower(): v for v in ds.data_vars}
    if var.lower() in low:
        return low[var.lower()]
    raise KeyError(f"variable {var!r} not in stream; available: {sorted(ds.data_vars)[:80]}")


def vert_dim(da):
    for d in VERT_DIMS:
        if d in da.dims:
            return d
    return None


def pick_level(da, level):
    """Choose a vertical level: int -> index; float -> nearest depth (z_l/z_i) or index."""
    vd = vert_dim(da)
    if vd is None:
        return da, None
    if level is None:
        level = 0
    if isinstance(level, float) and vd in ("z_l", "z_i") and vd in da.coords:
        out = da.sel({vd: float(level)}, method="nearest")
        return out, f"{vd}={float(out[vd]):g} m"
    out = da.isel({vd: int(level)})
    lab = f"{vd}[{int(level)}]"
    if vd in out.coords:
        lab += f"={float(out[vd]):g}"
    return out, lab


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def _cmap(name, fallback="viridis"):
    import matplotlib.pyplot as plt

    try:
        import cmocean.cm  # noqa: F401
    except Exception:
        pass
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap(fallback)


def default_cmap(name, kind):
    n = name.lower()
    if kind == "temperature":
        return "cmo.thermal"
    if kind == "salinity":
        return "cmo.haline"
    if kind == "speed":
        return "cmo.speed"
    if kind in ("velocity", "ssh") or n in ("zeta", "vorticity", "diff"):
        return "cmo.balance"
    if n in ("deptho", "depth", "d", "bathymetry", "mlotst", "oml"):
        return "cmo.deep"
    if n == "sigma0":
        return "cmo.dense"
    return "viridis"


def coastline_scale():
    """A Natural Earth coastline scale available *locally*, or (None, reason).

    cartopy downloads shapefiles on first use; compute nodes are offline, so a
    missing shapefile would hang/fail at savefig time.  Only scales already on
    disk are used, unless MOM6_TOOLS_MCP_ALLOW_DOWNLOAD=1.
    """
    try:
        import cartopy
    except Exception as e:
        return None, f"cartopy unavailable ({e})"
    if os.environ.get("MOM6_TOOLS_MCP_ALLOW_DOWNLOAD") == "1":
        return "50m", None
    roots = []
    for key in ("pre_existing_data_dir", "data_dir", "repo_data_dir"):
        p = cartopy.config.get(key)
        if p:
            roots.append(str(p))
    for scale in ("50m", "110m", "10m"):
        for r in roots:
            if os.path.exists(os.path.join(r, "shapefiles", "natural_earth", "physical", f"ne_{scale}_coastline.shp")):
                return scale, None
    return None, "no Natural Earth coastline shapefile on disk (offline) -- drawn without coastlines"


def unwrap_lon(lon):
    """Put a dateline-straddling longitude array onto a continuous 0..360 range."""
    np = _np()
    lon = np.asarray(lon, dtype=float)
    span = np.nanmax(lon) - np.nanmin(lon)
    if span > 180:
        alt = np.mod(lon, 360.0)
        if np.nanmax(alt) - np.nanmin(alt) < span:
            return alt
    return lon


def color_limits(data, robust=True, center=None, vmin=None, vmax=None):
    np = _np()
    a = np.asarray(data, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo, hi = (np.percentile(a, [2, 98]) if robust else (a.min(), a.max()))
    if vmin is not None:
        lo = vmin
    if vmax is not None:
        hi = vmax
    if center is not None:
        lim = max(abs(lo - center), abs(hi - center)) or 1e-12
        lo, hi = center - lim, center + lim
    if lo == hi:
        hi = lo + 1e-12
    return float(lo), float(hi)


def map_png(data, lon, lat, path, title, cmap="viridis", label="", vmin=None, vmax=None,
            center=None, robust=True, lines=None, notes=None):
    """Map a 2-D field on its 2-D lon/lat.  Cartopy if available, else plain axes.

    Never fails because of cartopy: coastlines only when a shapefile is on disk.
    `lines` = list of (lon_array, lat_array, label) drawn on top.
    """
    import matplotlib.pyplot as plt

    np = _np()
    d = np.ma.masked_invalid(np.asarray(data, dtype=float))
    lo = unwrap_lon(lon)
    la = np.asarray(lat, dtype=float)
    vmin, vmax = color_limits(d.compressed(), robust=robust, center=center, vmin=vmin, vmax=vmax)
    lat_span = max(np.nanmax(la) - np.nanmin(la), 1e-3)
    lon_span = max(np.nanmax(lo) - np.nanmin(lo), 1e-3)  # PlateCarree keeps degrees square
    aspect = lat_span / lon_span
    w = 11.0
    h = float(min(max(w * 0.82 * aspect + 1.0, 3.2), 12.0))
    cm = _cmap(cmap)
    used_cartopy = False
    try:
        import cartopy.crs as ccrs

        clon = float(np.nanmean(lo))
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=clon))
        pcm = ax.pcolormesh(lo, la, d, transform=ccrs.PlateCarree(), cmap=cm, vmin=vmin, vmax=vmax, shading="auto")
        scale, why = coastline_scale()
        if scale:
            ax.coastlines(scale, linewidth=0.6)
        elif why and notes is not None:
            notes.append(why)
        try:
            gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.5", alpha=0.5)
            gl.top_labels = gl.right_labels = False
        except Exception:
            pass
        if lines:
            for llo, lla, lab in lines:
                ax.plot(unwrap_lon(llo), lla, lw=2.2, transform=ccrs.PlateCarree(), label=lab)
            ax.legend(fontsize=8, loc="lower left")
        ax.set_extent([np.nanmin(lo), np.nanmax(lo), np.nanmin(la), np.nanmax(la)], crs=ccrs.PlateCarree())
        used_cartopy = True
    except Exception as e:  # fall back to a plain lon/lat plot
        plt.close("all")
        if notes is not None:
            notes.append(f"cartopy map failed ({type(e).__name__}: {e}); plain lon/lat axes used")
        fig, ax = plt.subplots(figsize=(w, h))
        pcm = ax.pcolormesh(lo, la, d, cmap=cm, vmin=vmin, vmax=vmax, shading="auto")
        if lines:
            for llo, lla, lab in lines:
                ax.plot(unwrap_lon(llo), lla, lw=2.2, label=lab)
            ax.legend(fontsize=8, loc="lower left")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(label)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"path": path, "vmin": vmin, "vmax": vmax, "cartopy": used_cartopy}


# --------------------------------------------------------------------------
# ops
# --------------------------------------------------------------------------


def op_locate_templates(a):
    import mom6_tools

    root = os.path.dirname(os.path.abspath(mom6_tools.__file__))
    d = os.path.join(root, "nb_templates", "regional_notebooks")
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.ipynb")))
    have = {}
    for mod in ("nbclient", "nbformat", "ipykernel", "jupyter_client"):
        try:
            __import__(mod)
            have[mod] = True
        except Exception:
            have[mod] = False
    return {
        "templates_dir": d,
        "templates": files,
        "mom6_tools_dir": root,
        "python": sys.executable,
        "execution_available": all(have.values()),
        "execution_modules": have,
    }


def op_list_vars(a):
    xr = _xr()
    hist_dir, casename, stream = a["hist_dir"], a["casename"], a.get("stream", "sfc")
    files = stream_files(hist_dir, casename, stream)
    if not files:
        raise _no_files(hist_dir, casename, stream)
    n_time = 0
    for f in files:
        try:
            import netCDF4

            with netCDF4.Dataset(f) as nc:
                n_time += len(nc.dimensions["time"]) if "time" in nc.dimensions else 0
        except Exception:
            with xr.open_dataset(f, decode_times=False) as d:
                n_time += d.sizes.get("time", 0)
    empty = [os.path.basename(f) for f in files if _n_time(f) == 0]
    out = {"stream": stream, "n_files": len(files), "files_first": os.path.basename(files[0]),
           "files_last": os.path.basename(files[-1]), "n_time": n_time}
    if empty:
        out["empty_files"] = empty
        out["warning"] = (f"{len(empty)} file(s) contain 0 time records (averaging period never completed or file "
                          "not closed cleanly)")
    ds0 = xr.open_dataset(files[0], decode_times=False)
    if "time" in ds0 and ds0.sizes.get("time", 0) > 0:
        import cftime

        units = ds0["time"].attrs.get("units")
        rawcal = ds0["time"].attrs.get("calendar")
        cal = norm_calendar(rawcal)
        out["time_units"] = units
        out["calendar_attr"] = rawcal
        out["calendar_used"] = cal
        try:
            first = float(ds0["time"].values[0])
            with xr.open_dataset(files[-1], decode_times=False) as dl:
                last = float(dl["time"].values[-1]) if dl.sizes.get("time", 0) else first
            out["time_first"] = _tstr(cftime.num2date(first, units, calendar=cal))
            out["time_last"] = _tstr(cftime.num2date(last, units, calendar=cal))
        except Exception as e:
            out["time_decode_error"] = str(e)
    skip = {"average_T1", "average_T2", "average_DT", "time_bounds", "time_bnds"}
    vars_ = []
    for name, v in ds0.data_vars.items():
        if name in skip:
            continue
        vars_.append({
            "name": name,
            "dims": list(v.dims),
            "shape": [int(s) for d, s in zip(v.dims, v.shape) if d != "time"],
            "position": pos_of(v.dims),
            "vertical": vert_dim(v),
            "units": v.attrs.get("units"),
            "long_name": v.attrs.get("long_name"),
        })
    out["variables"] = vars_
    out["n_variables"] = len(vars_)
    ssh = [c for c in SSH_CANDIDATES if c in ds0]
    out["ssh_variable"] = ssh[0] if ssh else None
    return out


def op_describe_grid(a):
    np = _np()
    hist_dir, casename = a["hist_dir"], a["casename"]
    static, geom = open_grid(hist_dir, casename)
    lon, lat = lonlat(geom, static, "h")
    lonv, latv = np.asarray(lon, float), np.asarray(lat, float)
    out = {"dims": {k: int(v) for k, v in static.sizes.items()},
           "has_ocean_geometry": bool(geom.attrs.get("_has_geometry_file"))}
    lo_min, lo_max = float(np.nanmin(lonv)), float(np.nanmax(lonv))
    out["lon_range"] = [lo_min, lo_max]
    out["lat_range"] = [float(np.nanmin(latv)), float(np.nanmax(latv))]
    conv = "0..360" if lo_max > 180 else ("-180..180" if lo_min < 0 else "ambiguous (all within 0..180)")
    out["lon_convention"] = conv
    straddle = False
    alt = np.mod(lonv, 360)
    if lo_max - lo_min > 180 and (np.nanmax(alt) - np.nanmin(alt)) < (lo_max - lo_min):
        straddle = True  # -180..180 grid that wraps across the dateline
    elif lo_min < 180 < lo_max:
        straddle = True  # 0..360 grid that spans 180E
    out["straddles_dateline"] = straddle
    if straddle:
        u = unwrap_lon(lonv)
        out["lon_range_continuous"] = [float(np.nanmin(u)), float(np.nanmax(u))]
        out["dateline_note"] = "domain crosses 180 deg: compare/plot with 0..360 longitudes"
    wet = wet_of(static, geom, "h")
    if wet is not None:
        w = np.asarray(wet) > 0
        out["wet_fraction"] = _f(w.mean(), 4)
        out["n_wet"] = int(w.sum())
    else:
        w = np.ones_like(lonv, bool)
    dep, dname = depth_of(static, geom)
    if dep is not None:
        dv = np.asarray(dep, float)
        dv = np.where(w & np.isfinite(dv) & (np.abs(dv) < FILL_ABS), dv, np.nan)
        out["depth"] = {"variable": dname, "min_wet": _f(np.nanmin(dv), 2), "max": _f(np.nanmax(dv), 2),
                        "mean_wet": _f(np.nanmean(dv), 2), "median_wet": _f(np.nanmedian(dv), 2)}
    area = area_of(static, geom, "h")
    if area is not None:
        av = np.asarray(area, float)
        out["area_km2"] = {"total_wet": _f(np.nansum(np.where(w, av, 0)) / 1e6, 1),
                           "total_all": _f(np.nansum(av) / 1e6, 1)}
    if "dxT" in geom and "dyT" in geom:
        dx, dy = np.asarray(geom["dxT"], float) / 1e3, np.asarray(geom["dyT"], float) / 1e3
        out["dx_km"] = {"mean": _f(np.nanmean(dx), 3), "min": _f(np.nanmin(dx), 3), "max": _f(np.nanmax(dx), 3)}
        out["dy_km"] = {"mean": _f(np.nanmean(dy), 3), "min": _f(np.nanmin(dy), 3), "max": _f(np.nanmax(dy), 3)}
        mind = np.where(w, np.minimum(dx, dy), np.nan)
        out["min_wet_spacing_km"] = _f(np.nanmin(mind), 3)
        if dep is not None:
            # barotropic gravity-wave CFL hint: dt_bt < dx / sqrt(g H)
            c = np.sqrt(9.81 * np.where(np.isfinite(dv), dv, np.nan))
            dtbt = np.nanmin(mind * 1e3 / c)
            out["max_stable_barotropic_dt_s_estimate"] = _f(dtbt, 1)
    elif area is not None:
        out["dx_km_estimate_sqrt_area"] = _f(np.nanmean(np.sqrt(av)) / 1e3, 3)
    if "xh" in static.coords and static.sizes.get("xh", 0) > 1:
        out["nominal_dlon_deg"] = _f(np.nanmean(np.diff(np.asarray(static["xh"], float))), 5)
    if "yh" in static.coords and static.sizes.get("yh", 0) > 1:
        out["nominal_dlat_deg"] = _f(np.nanmean(np.diff(np.asarray(static["yh"], float))), 5)
    out["static_variables"] = sorted(static.data_vars)
    if a.get("plot") and dep is not None:
        notes = []
        p = _out_path(a["out_dir"], f"{_safe_name(casename)}_bathymetry.png")
        r = map_png(np.where(w, dv, np.nan), lonv, latv, p, f"{casename}: bathymetry ({dname})", cmap="cmo.deep",
                    label="depth [m]", robust=False, notes=notes)
        out["png"] = r["path"]
        if notes:
            out["plot_notes"] = sorted(set(notes))
    return out


def _wet_area_for(da, static, geom):
    pos = pos_of(da.dims)
    if pos is None:
        return None, None, pos
    return wet_of(static, geom, pos), area_of(static, geom, pos), pos


def op_surface_stats(a):
    import dask

    np = _np()
    hist_dir, casename = a["hist_dir"], a["casename"]
    stream = a.get("stream") or "sfc"
    ds, static, geom, cal = open_stream(hist_dir, casename, stream, a.get("calendar"), with_coords=False)
    ds = time_slice(ds, a.get("start"), a.get("end"))
    if "time" not in ds.dims or ds.sizes["time"] == 0:
        raise ValueError("no time records in the requested window")
    req = a.get("variables") or ["tos", "sos", "zos", "speed"]
    names, notes = [], []
    for v in req:
        try:
            names.append(resolve_var(ds, v))
        except KeyError:
            notes.append(f"{v!r} not in stream {stream!r}; skipped")
    names = list(dict.fromkeys(names))
    times = [_tstr(t) for t in ds["time"].values]
    nt = len(times)
    lazy, meta = {}, {}
    for name in names:
        da = ds[name]
        vd = vert_dim(da)
        lev = None
        if vd:
            da = da.isel({vd: 0})
            lev = f"{vd}[0] (top level)"
        hdims = [d for d in da.dims if d != "time"]
        wet, area, pos = _wet_area_for(da, static, geom)
        finite = np.isfinite(da) & (abs(da) < FILL_ABS)
        valid = da.where(finite)
        if wet is not None:
            wetm = wet > 0
            nwet = float(wetm.sum())
            bad_in_wet = ((~finite) & wetm).sum(hdims) / max(nwet, 1.0)
            valid = valid.where(wetm)
        else:
            bad_in_wet = (~finite).mean(hdims)
        fill_frac = (abs(da) >= FILL_ABS).mean(hdims)
        if area is not None:
            mean = valid.weighted(area.fillna(0.0)).mean(hdims)
        else:
            mean = valid.mean(hdims)
        lazy[name] = (mean, valid.min(hdims), valid.max(hdims), bad_in_wet, fill_frac)
        meta[name] = {"units": ds[name].attrs.get("units"), "long_name": ds[name].attrs.get("long_name"),
                      "position": pos, "level": lev, "area_weighted": area is not None}
    (computed,) = dask.compute(lazy)
    results, problems = {}, []
    for name, (mean, mn, mx, bad, fill) in computed.items():
        mean, mn, mx, bad, fill = (np.asarray(x, float) for x in (mean, mn, mx, bad, fill))
        kind = var_kind(name, meta[name]["units"])
        r = dict(meta[name])
        r["kind"] = kind
        r["n_time"] = nt
        if np.all(~np.isfinite(mean)):
            r["all_missing"] = True
            problems.append(f"{name}: every record is NaN/fill (partial-month file or not written)")
        r["min"] = _f(np.nanmin(mn)) if np.isfinite(mn).any() else None
        r["max"] = _f(np.nanmax(mx)) if np.isfinite(mx).any() else None
        if np.isfinite(mx).any():
            r["time_of_max"] = times[int(np.nanargmax(mx))]
            r["time_of_min"] = times[int(np.nanargmin(mn))]
        fm = np.where(np.isfinite(mean))[0]
        if fm.size:
            r["mean_first"] = _f(mean[fm[0]])
            r["mean_last"] = _f(mean[fm[-1]])
            r["mean_change"] = _f(mean[fm[-1]] - mean[fm[0]])
        r["max_nan_or_fill_frac_in_wet"] = _f(np.nanmax(bad), 6)
        r["max_fill_frac"] = _f(np.nanmax(fill), 6)
        if np.nanmax(bad) > 0:
            first_bad = int(np.argmax(bad > 0))
            r["first_time_nan_in_wet"] = times[first_bad]
        if np.nanmax(bad) > 0 and not r.get("all_missing"):
            problems.append(f"{name}: NaN/fill inside wet cells from {times[first_bad]} "
                            f"(max {100 * np.nanmax(bad):.2f}% of wet cells)")
        flags = []
        if kind in RANGES:
            lo, hi, desc = RANGES[kind]
            trip = np.zeros(nt, bool)
            if lo is not None:
                trip |= mn < lo
            if hi is not None:
                trip |= mx > hi
            if trip.any():
                i0 = int(np.argmax(trip))
                flags.append({"flag": desc, "first_time": times[i0], "n_records": int(trip.sum()),
                              "value_at_first": _f(mx[i0] if (hi is not None and mx[i0] > hi) else mn[i0])})
                problems.append(f"{name}: {desc} first at {times[i0]} ({int(trip.sum())} of {nt} records)")
        r["flags"] = flags
        idx = _downsample_idx(nt)
        r["series"] = {"time": [times[i] for i in idx], "mean": [_f(mean[i], 6) for i in idx],
                       "min": [_f(mn[i], 6) for i in idx], "max": [_f(mx[i], 6) for i in idx]}
        if len(idx) < nt:
            r["series_downsampled_from"] = nt
        results[name] = r
    return {"stream": stream, "calendar_used": cal, "n_time": nt, "time_first": times[0], "time_last": times[-1],
            "variables": results, "problems": problems, "ok": not problems, "notes": notes}


def _derived(ds, static, geom, which, level):
    """Compute speed / vorticity / sigma0 lazily.  Returns (DataArray, pos, label)."""
    tb = _tb()
    np = _np()
    if which == "sigma0":
        for t, s in (("tos", "sos"), ("thetao", "so")):
            if t in ds and s in ds:
                T, S = ds[t], ds[s]
                T, lab = pick_level(T, level)
                S, _ = pick_level(S, level)
                out = tb.sigma0(mask_fill(T), mask_fill(S))
                return out, "h", f"sigma0 from {t}/{s}" + (f" {lab}" if lab else "")
        raise KeyError("sigma0 needs tos+sos (sfc) or thetao+so in the stream")
    for u, v in (("SSU", "SSV"), ("uo", "vo")):
        if u in ds and v in ds:
            break
    else:
        if which == "speed" and "speed" in ds:
            return ds["speed"], "h", "speed (diagnosed)"
        raise KeyError(f"{which} needs SSU/SSV or uo/vo in the stream")
    U, lab = pick_level(ds[u], level)
    V, _ = pick_level(ds[v], level)
    sub = ds.drop_vars(list(ds.data_vars)).assign({u: mask_fill(U), v: mask_fill(V)})
    sub = sub.drop_vars([c for c in sub.coords if c.startswith("geo")], errors="ignore")
    if which == "speed":
        wu, wv = wet_of(static, geom, "u"), wet_of(static, geom, "v")
        grid, sm = tb.make_grid(sub, geom, with_metrics=False)
        uu = sm[u] if wu is None else sm[u].where(wu > 0)
        vv = sm[v] if wv is None else sm[v].where(wv > 0)
        ut = grid.interp(uu.fillna(0.0), "X")
        vt = grid.interp(vv.fillna(0.0), "Y")
        sp = np.sqrt(ut ** 2 + vt ** 2)
        sp.name = "speed"
        sp.attrs = {"units": "m s-1", "long_name": f"speed from {u}/{v} interpolated to h-points"}
        return sp, "h", f"speed from {u}/{v}" + (f" {lab}" if lab else "")
    if which == "vorticity":
        if not geom.attrs.get("_has_geometry_file"):
            raise FileNotFoundError("vorticity needs the ocean_geometry file (dxCu, dyCv, Aq metrics)")
        grid, sm = tb.make_grid(sub, geom, with_metrics=True)
        sm = sm.assign({u: sm[u].fillna(0.0), v: sm[v].fillna(0.0)})
        z = tb.relative_vorticity(grid, sm, u=u, v=v)
        return z, "q", f"relative vorticity from {u}/{v}" + (f" {lab}" if lab else "")
    raise ValueError(f"unknown derived quantity {which!r}; use speed, vorticity or sigma0")


def op_plot_field(a):
    np = _np()
    hist_dir, casename = a["hist_dir"], a["casename"]
    stream = a.get("stream") or "sfc"
    derived = a.get("derived")
    ds, static, geom, cal = open_stream(hist_dir, casename, stream, a.get("calendar"), with_coords=False)
    notes = []
    if derived:
        da, pos, label = _derived(ds, static, geom, derived, a.get("level"))
        name = derived
    else:
        if not a.get("var"):
            raise ValueError("give var (or derived=speed|vorticity|sigma0)")
        name = resolve_var(ds, a["var"])
        da = mask_fill(ds[name])
        da, lab = pick_level(da, a.get("level"))
        pos = pos_of(da.dims)
        label = name + (f" {lab}" if lab else "")
        if pos is None:
            raise ValueError(f"{name} dims {da.dims} are not a horizontal map")
    wet = wet_of(static, geom, pos)
    if wet is not None:
        da = da.where(wet > 0)
    lon, lat = lonlat(geom, static, pos)
    units = da.attrs.get("units") or (ds[name].attrs.get("units") if name in ds else "")
    kind = var_kind(name, units)
    cmap = a.get("cmap") or default_cmap(name if name != "vorticity" else "zeta", kind)
    center = 0.0 if (kind in ("velocity",) or derived == "vorticity" or a.get("center_zero")) else None
    out_dir = a["out_dir"]
    out = {"stream": stream, "variable": name, "label": label, "position": pos, "units": units, "calendar_used": cal}

    if a.get("animate"):
        if "time" not in da.dims:
            raise ValueError("animate needs a time dimension")
        nt = da.sizes["time"]
        idx = _downsample_idx(nt, GIF_MAX_FRAMES)
        field = da.isel(time=idx).load()
        field = field.assign_coords(geolon=(lon.dims, unwrap_lon(lon.values)), geolat=lat.reset_coords(drop=True))
        if not np.isfinite(np.asarray(field, float)).any():
            notes.append(
                "field entirely NaN/fill in every record (partial-month native/z file "
                "averages are written as fill until the month completes, or a blow-up)"
                " -- no GIF written"
            )
            out.update({"gif": None, "frames": 0, "records_available": int(nt)})
            out["notes"] = sorted(set(notes))
            return out
        gif = _out_path(out_dir, f"{_safe_name(casename)}_{_safe_name(name)}_{stream}.gif")
        method = _make_gif(field, gif, label, cmap, center, notes)
        out.update({"gif": gif, "frames": int(len(idx)), "records_available": int(nt), "gif_method": method,
                    "min": _f(field.min()), "max": _f(field.max())})
        if nt > GIF_MAX_FRAMES:
            notes.append(f"{nt} records subsampled to {len(idx)} frames")
        out["notes"] = sorted(set(notes))
        return out

    da2, tlab = pick_time(da, a.get("time", "last"))
    vals = np.asarray(da2.load(), float)
    out["time"] = tlab
    fin = np.isfinite(vals)
    out["min"] = _f(np.nanmin(vals)) if fin.any() else None
    out["max"] = _f(np.nanmax(vals)) if fin.any() else None
    out["mean_unweighted"] = _f(np.nanmean(vals)) if fin.any() else None
    out["n_valid"] = int(fin.sum())
    if not fin.any():
        notes.append(
            "field entirely NaN/fill at this time (partial-month native/z file "
            "averages are written as fill until the month completes, or a blow-up)"
            " -- no PNG written"
        )
        out["png"] = None
        out["notes"] = sorted(set(notes))
        return out
    if derived == "vorticity" and "Coriolis" in static and fin.any():
        ro = np.asarray(abs(da2 / static["Coriolis"].where(abs(static["Coriolis"]) > 0)), float)
        out["max_abs_zeta_over_f"] = _f(np.nanmax(ro))
        out["p99_abs_zeta_over_f"] = _f(np.nanpercentile(ro, 99))
    title = f"{casename}: {label}" + (f"   {tlab}" if tlab else "")
    png = _out_path(out_dir, f"{_safe_name(casename)}_{_safe_name(name)}_{stream}_{_safe_name(tlab or 'static')}.png")
    r = map_png(vals, lon.values, lat.values, png, title, cmap=cmap, label=f"{name} [{units}]",
                center=center, robust=bool(a.get("robust", True)), notes=notes)
    out.update({"png": r["path"], "color_range": [r["vmin"], r["vmax"]]})
    out["notes"] = sorted(set(notes))
    return out


def _make_gif(field, gif, label, cmap, center, notes):
    """GIF via mom6_tools' save_field_gif when cartopy + local coastlines exist,
    otherwise via our own frames + mom6_tools' save_gif."""
    import matplotlib.pyplot as plt

    tb = _tb()
    np = _np()
    scale, why = coastline_scale()
    vmin, vmax = color_limits(np.asarray(field), robust=True, center=center)
    try:
        import cartopy.crs as ccrs

        if scale is None:
            raise RuntimeError(why)
        clon = float(np.nanmean(field["geolon"]))
        aspect = float(field.geolat.max() - field.geolat.min()) / max(float(field.geolon.max() - field.geolon.min()), 1e-3)
        figsize = (10.0, float(min(max(10 * aspect + 1, 3), 10)))
        field = field.clip(vmin, vmax)
        tb.save_field_gif(field, gif, {"projection": ccrs.PlateCarree(central_longitude=clon)},
                          {"x": "geolon", "y": "geolat", "transform": ccrs.PlateCarree()}, figsize,
                          cmap=_cmap(cmap), coastlines=scale, title=label, clean=True)
        return "mom6_tools.regional_m6toolbox.save_field_gif"
    except Exception as e:
        notes.append(f"save_field_gif not used ({e}); frames drawn without cartopy")
    frames = []
    base = os.path.splitext(gif)[0]
    for i in range(field.sizes["time"]):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        pcm = ax.pcolormesh(field["geolon"], field["geolat"], field.isel(time=i), cmap=_cmap(cmap),
                            vmin=vmin, vmax=vmax, shading="auto")
        fig.colorbar(pcm, ax=ax)
        ax.set_title(f"{label}   {_tstr(field.time.values[i])[:10]}")
        p = f"{base}_frame_{i:03d}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        frames.append(p)
    tb.save_gif(frames, gif, duration=150, clean=True)
    return "mom6_tools.regional_m6toolbox.save_gif"


def _nearest_ij(lon2d, lat2d, lon, lat):
    np = _np()
    lo = unwrap_lon(lon2d)
    if lon is not None and np.nanmax(lo) > 180:
        lon = lon % 360
    dlon = (lo - lon) * np.cos(np.radians(lat2d))
    d2 = dlon ** 2 + (lat2d - lat) ** 2
    j, i = np.unravel_index(np.nanargmin(d2), d2.shape)
    return int(j), int(i)


def op_plot_section(a):
    import matplotlib.pyplot as plt

    np = _np()
    hist_dir, casename = a["hist_dir"], a["casename"]
    stream = a.get("stream") or "z"
    lon_req, lat_req = a.get("lon"), a.get("lat")
    if (lon_req is None) == (lat_req is None):
        raise ValueError("give exactly one of lat= (zonal section along a model row) or lon= (meridional section along a column)")
    ds, static, geom, cal = open_stream(hist_dir, casename, stream, a.get("calendar"), with_coords=False)
    name = resolve_var(ds, a["var"])
    da = mask_fill(ds[name])
    vd = vert_dim(da)
    if vd is None:
        raise ValueError(f"{name} has no vertical dim in stream {stream!r}; use the z or native stream")
    pos = pos_of(da.dims)
    lon2d, lat2d = (np.asarray(x, float) for x in lonlat(geom, static, pos))
    ulon = unwrap_lon(lon2d)
    if lat_req is not None and not (np.nanmin(lat2d) <= float(lat_req) <= np.nanmax(lat2d)):
        raise ValueError(f"lat={lat_req} is outside the domain ({np.nanmin(lat2d):.3f} .. {np.nanmax(lat2d):.3f})")
    if lon_req is not None:
        lr = float(lon_req) % 360 if np.nanmax(ulon) > 180 else float(lon_req)
        if not (np.nanmin(ulon) <= lr <= np.nanmax(ulon)):
            raise ValueError(f"lon={lon_req} is outside the domain ({np.nanmin(ulon):.3f} .. {np.nanmax(ulon):.3f})")
    if lat_req is not None:
        j, i = _nearest_ij(lon2d, lat2d, float(np.nanmean(unwrap_lon(lon2d))), float(lat_req))
        ydim = [d for d in da.dims if d in ("yh", "yq")][0]
        sec = da.isel({ydim: j})
        xcoord, xname = unwrap_lon(lon2d[j, :]), "longitude"
        where = f"row {ydim}={j} (lat ~ {np.nanmean(lat2d[j, :]):.3f})"
    else:
        j, i = _nearest_ij(lon2d, lat2d, float(lon_req), float(np.nanmean(lat2d)))
        xdim = [d for d in da.dims if d in ("xh", "xq")][0]
        sec = da.isel({xdim: i})
        xcoord, xname = lat2d[:, i], "latitude"
        where = f"column {xdim}={i} (lon ~ {np.nanmean(unwrap_lon(lon2d[:, i])):.3f})"
    sec, tlab = pick_time(sec, a.get("time", "last"))
    sec = sec.load()
    notes = []
    if vd in ("z_l", "z_i"):
        z = np.asarray(sec[vd], float)
        Z = np.broadcast_to(z[:, None], sec.shape)
        zsrc = f"{vd} coordinate (fixed depths)"
    else:  # native layers: depths from interface heights e
        if pos != "h":
            raise ValueError("native-stream sections are only supported for h-point fields; use stream='z' for velocities")
        if "e" not in ds:
            raise ValueError("native stream has no interface height 'e' to place layers in depth; use stream='z'")
        e = ds["e"]
        e = e.isel({"yh": j}) if lat_req is not None else e.isel({"xh": i})
        e, _ = pick_time(e, a.get("time", "last"))
        zc = _tb().layer_depths(mask_fill(e).load())
        Z = np.asarray(zc.transpose(*sec.dims), float)
        zsrc = "layer centres from interface heights e (layer_depths)"
    X = np.broadcast_to(np.asarray(xcoord, float)[None, :], sec.shape)
    V = np.asarray(sec.transpose(vd, ...), float)
    fin = np.isfinite(V)
    Z = np.array(Z, float)
    if not np.isfinite(Z).all():  # land columns / fill in e: give pcolormesh finite depths
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lvl = np.nanmean(Z, axis=1)
        for k in range(len(lvl)):
            if not np.isfinite(lvl[k]):
                lvl[k] = (lvl[k - 1] + 1.0) if k else 0.0
        Z = np.where(np.isfinite(Z), Z, lvl[:, None])
    units = ds[name].attrs.get("units", "")
    kind = var_kind(name, units)
    out = {"stream": stream, "variable": name, "units": units, "section": where, "time": tlab,
           "vertical": zsrc, "n_levels": int(V.shape[0]), "n_points": int(V.shape[1]), "calendar_used": cal,
           "min": _f(np.nanmin(V)) if fin.any() else None, "max": _f(np.nanmax(V)) if fin.any() else None,
           "mean_unweighted": _f(np.nanmean(V)) if fin.any() else None, "n_valid": int(fin.sum())}
    if not fin.any():
        notes.append("section entirely NaN/fill (partial-month native/z file averages are written as fill "
                     "until the month completes, or a blow-up) -- no PNG written")
        out["png"] = None
        out["notes"] = notes
        return out
    fig, ax = plt.subplots(figsize=(11, 4.8))
    center = 0.0 if kind == "velocity" else None
    vmin, vmax = color_limits(V[fin] if fin.any() else [0, 1], robust=True, center=center)
    pcm = ax.pcolormesh(X, Z, np.ma.masked_invalid(V), cmap=_cmap(a.get("cmap") or default_cmap(name, kind)),
                        vmin=vmin, vmax=vmax, shading="nearest")
    dep, dname = depth_of(static, geom)
    if dep is not None and pos == "h":
        floor = np.asarray(dep, float)[j, :] if lat_req is not None else np.asarray(dep, float)[:, i]
        floor = np.where(np.abs(floor) < FILL_ABS, floor, np.nan)
        ax.plot(xcoord, floor, "k-", lw=1.2, label=f"sea floor ({dname})")
        ax.legend(fontsize=8, loc="lower right")
        zmax = np.nanmax(floor) if np.isfinite(floor).any() else np.nanmax(Z)
        ax.set_ylim(1.05 * max(zmax, 1.0), 0)
    else:
        ax.invert_yaxis()
    fig.colorbar(pcm, ax=ax, label=f"{name} [{units}]")
    ax.set_xlabel(xname)
    ax.set_ylabel("depth [m]")
    ax.set_title(f"{casename}: {name} section, {where}" + (f"  {tlab}" if tlab else ""), fontsize=10)
    png = _out_path(a["out_dir"], f"{_safe_name(casename)}_{_safe_name(name)}_{stream}_section_{'lat' if lat_req is not None else 'lon'}{round(float(lat_req if lat_req is not None else lon_req), 3):g}_{_safe_name(tlab or '')}.png")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    out["png"] = png
    out["notes"] = notes
    return out


# --------------------------------------------------------------------------
# OBC
# --------------------------------------------------------------------------


def obc_dir(input_dir):
    for d in (input_dir, os.path.join(input_dir, "ocn"), os.path.join(input_dir, "ocean"), os.path.join(input_dir, "ocnice")):
        if glob.glob(os.path.join(glob.escape(d), "forcing_obc_segment_*.nc")):
            return d
    raise FileNotFoundError(f"no forcing_obc_segment_*.nc under {input_dir} (or its ocn/, ocean/, ocnice/)")


def open_topog(d):
    xr = _xr()
    hits = sorted(glob.glob(os.path.join(glob.escape(d), "ocean_topog*.nc")))
    if not hits:
        return None, None
    t = xr.open_dataset(hits[0])
    ren = {k: v for k, v in {"ny": "yh", "nx": "xh"}.items() if k in t.dims}
    return t.rename(ren), hits[0]


FLOOR_CHECKS = "floor-depth check, below-floor masking, NaN-above-floor count"


def _failed_segment(path, stage, error, skipped=FLOOR_CHECKS, sid=None):
    """One entry of obc_summary's failed_segments."""
    if sid is None:
        m = re.search(r"segment_(\d+)\.nc$", path)
        sid = m.group(1) if m else os.path.basename(path)
    return {
        "segment": sid,
        "file": os.path.basename(path),
        "stage": stage,
        "error": error,
        "skipped": skipped,
    }


def _segments(input_dir, notes, failed=None):
    tb = _tb()
    d = obc_dir(input_dir)
    segs = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(d), "forcing_obc_segment_*.nc"))):
        try:
            seg = tb.to_t_points(tb.open_obc(path))
            segs[seg.attrs["segment"]] = (seg, path)
        except Exception as e:
            notes.append(f"{os.path.basename(path)}: could not open with open_obc ({type(e).__name__}: {e})")
            if failed is not None:
                err = f"{type(e).__name__}: {e}"
                failed.append(_failed_segment(path, "open_obc", err, "all checks"))
    return d, segs


def op_obc_summary(a):
    import matplotlib.pyplot as plt

    np = _np()
    tb = _tb()
    notes, problems, failed = [], [], []
    d, segs = _segments(a["input_dir"], notes, failed)
    topog, tpath = open_topog(d)
    if topog is None:
        notes.append("no ocean_topog*.nc beside the OBC files: depth-vs-floor checks skipped")
    max_times = int(a.get("max_times") or 120)
    out = {"obc_dir": d, "topog_file": tpath, "n_segments": len(segs), "segments": {}}
    UNITS = {"temp": "degC", "salt": "psu", "u": "m s-1", "v": "m s-1", "eta": "m"}
    KIND = {"temp": "temperature", "salt": "salinity", "u": "velocity", "v": "velocity", "eta": "ssh"}
    floors, masked = {}, {}
    for sid, (seg, path) in segs.items():
        nt = seg.sizes["time"]
        tidx = _downsample_idx(nt, max_times)
        s = seg.isel(time=tidx)
        info = {"file": os.path.basename(path), "orientation": seg.attrs["orientation"],
                "const_coord": _f(seg.attrs["const_coord"], 4), "label": tb.obc_label(seg),
                "normal_velocity": tb.normal_var(seg),
                "n_points_t": int(seg.sizes["n"]), "n_levels": int(seg.sizes["depth"]), "n_times": int(nt),
                "time_first": _tstr(seg.time.values[0]), "time_last": _tstr(seg.time.values[-1]),
                "obc_max_level_depth": _f(seg["depth"].max(), 1),
                "lon_range": [_f(seg.lon.min(), 3), _f(seg.lon.max(), 3)],
                "lat_range": [_f(seg.lat.min(), 3), _f(seg.lat.max(), 3)]}
        if len(tidx) < nt:
            info["stats_from_n_times"] = int(len(tidx))
        if nt > 1:
            dt = np.diff(np.asarray(_as_days(seg.indexes["time"])))
            info["time_step_days"] = sorted({round(float(x), 4) for x in dt})[:5]
        floor = None
        if topog is not None:
            try:
                floor = tb.on_boundary(topog, seg, topog)
                floors[sid] = floor
                fd = np.asarray(floor["depth"], float)
                wetb = np.asarray(floor["mask"]) > 0
                info.update({"floor_max": _f(np.nanmax(np.where(wetb, fd, np.nan)), 1),
                             "floor_mean_wet": _f(np.nanmean(np.where(wetb, fd, np.nan)), 1),
                             "n_wet": int(wetb.sum()), "n_land": int((~wetb).sum()),
                             "boundary_line": floor.attrs.get("boundary_line")})
                if info["floor_max"] is not None and info["obc_max_level_depth"] < info["floor_max"]:
                    problems.append(f"segment {sid}: OBC levels stop at {info['obc_max_level_depth']} m but the "
                                    f"sea floor reaches {info['floor_max']} m -- bottom cells get extrapolated values")
            except Exception as e:
                notes.append(f"segment {sid}: on_boundary failed ({e}); depth-vs-floor skipped")
                err = f"{type(e).__name__}: {e}"
                failed.append(_failed_segment(path, "on_boundary", err, sid=sid))
        else:
            err = "no ocean_topog*.nc beside the OBC files"
            failed.append(_failed_segment(path, "topog", err, sid=sid))
        if floor is not None:
            m = tb.mask_below_floor(s, floor)
            info["kept_fraction_after_floor_mask"] = _f(m["temp"].notnull().mean(), 4)
        else:
            m = s
        masked[sid] = m
        stats = {}
        for v in ("temp", "salt", "u", "v", "eta"):
            if v not in m:
                continue
            raw = s[v]
            fin = np.isfinite(raw) & (abs(raw) < FILL_ABS)
            dm = m[v].where(fin)
            vals = {"min": dm.min(), "max": dm.max(), "mean": dm.mean()}
            if floor is not None:
                expected = (floor["mask"] > 0)
                if "depth" in raw.dims:
                    expected = expected & (seg["depth"] < floor["depth"])
                vals["nan_where_expected_valid"] = ((~fin) & expected).sum()
            else:
                vals["nan_total"] = (~fin).sum()
            vals = {k: float(x) for k, x in vals.items()}
            r = {"units": UNITS[v], "min": _f(vals["min"], 5), "max": _f(vals["max"], 5), "mean": _f(vals["mean"], 5)}
            if "nan_where_expected_valid" in vals:
                r["nan_where_expected_valid"] = int(vals["nan_where_expected_valid"])
                if r["nan_where_expected_valid"]:
                    problems.append(f"segment {sid} {v}: {r['nan_where_expected_valid']} NaN/fill values in wet cells above the floor")
            else:
                r["nan_total"] = int(vals["nan_total"])
            lo, hi, desc = RANGES[KIND[v]]
            if (lo is not None and r["min"] is not None and r["min"] < lo) or (hi is not None and r["max"] is not None and r["max"] > hi):
                r["flag"] = desc
                problems.append(f"segment {sid} {v}: {desc} (min {r['min']}, max {r['max']})")
            stats[v] = r
        info["variables"] = stats
        out["segments"][sid] = info
    if a.get("plot") and segs:
        pngs = []
        od = a["out_dir"]
        if topog is not None:
            dep = np.where(np.asarray(topog["mask"]) > 0, np.asarray(topog["depth"], float), np.nan)
            lines = [(np.asarray(sg.lon), np.asarray(sg.lat), f"seg {sid}") for sid, (sg, _) in segs.items()]
            p = _out_path(od, "obc_segment_map.png")
            map_png(dep, np.asarray(topog["x"]), np.asarray(topog["y"]), p, "bathymetry + OBC segments",
                    cmap="cmo.deep", label="depth [m]", robust=False, lines=lines, notes=notes)
            pngs.append(p)
        for sid, m in masked.items():
            seg = segs[sid][0]
            xn = tb.along_name(seg)
            nv = tb.normal_var(seg)
            fig, axs = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
            x = np.asarray(seg[xn], float)
            if xn == "lon":
                x = unwrap_lon(x)
            for ax, v in zip(axs.ravel()[:3], ("temp", "salt", nv)):
                fld = np.asarray(m[v].isel(time=0).transpose("depth", "n"), float)
                kind = KIND[v]
                vmin, vmax = color_limits(fld, robust=True, center=0.0 if kind == "velocity" else None)
                pc = ax.pcolormesh(x, np.asarray(seg["depth"], float), np.ma.masked_invalid(fld), shading="nearest",
                                   cmap=_cmap(default_cmap(v, kind)), vmin=vmin, vmax=vmax)
                fig.colorbar(pc, ax=ax, label=f"{v} [{UNITS[v]}]")
                if sid in floors:
                    fl = np.asarray(floors[sid]["depth"], float)
                    ax.plot(x, fl, "k-", lw=1.1)
                    ax.set_ylim(1.05 * max(np.nanmax(fl), 1.0), 0)
                else:
                    ax.invert_yaxis()
                ax.set_title(v + ("  (normal)" if v == nv else ""), fontsize=10)
                ax.set_ylabel("depth [m]")
            ax = axs.ravel()[3]
            eta = m["eta"]
            ax.fill_between(x, np.asarray(eta.min("time"), float), np.asarray(eta.max("time"), float), color="0.8",
                            label=f"range over {eta.sizes['time']} records")
            ax.plot(x, np.asarray(eta.isel(time=0), float), lw=1, label="eta (first record)")
            ax.axhline(0, color="0.5", lw=0.6)
            ax.set_title("eta", fontsize=10)
            ax.legend(fontsize=8)
            for axx in axs[-1]:
                axx.set_xlabel(f"{xn} [deg]")
            fig.suptitle(f"{tb.obc_label(seg)} -- {_tstr(seg.time.values[0])[:10]}"
                         + ("  (levels below the floor masked)" if sid in floors else ""))
            fig.tight_layout()
            p = _out_path(od, f"obc_segment_{sid}_sections.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            pngs.append(p)
        out["pngs"] = pngs
    out["problems"] = problems
    out["failed_segments"] = failed
    # ok = every segment opened and fully checked against the topography, and no
    # problem found; a partial failure is ok=False even when problems is empty.
    out["ok"] = not problems and not failed
    out["notes"] = sorted(set(notes))
    return out


def op_obc_vs_model(a):
    import matplotlib.pyplot as plt

    np = _np()
    tb = _tb()
    xr = _xr()
    notes = []
    hist_dir, casename = a["hist_dir"], a["casename"]
    var = a.get("var") or "temp"
    d, segs = _segments(a["input_dir"], notes)
    topog, tpath = open_topog(d)
    if topog is None:
        raise FileNotFoundError(f"obc_vs_model needs ocean_topog*.nc beside the OBC files in {d}")
    sfc, static, geom, cal = open_stream(hist_dir, casename, a.get("stream") or "sfc", a.get("calendar"), with_coords=False)
    ssh = next((c for c in SSH_CANDIDATES if c in sfc), None)
    fields = {"temp": "tos", "salt": "sos", "eta": ssh}
    wet = wet_of(static, geom, "h")
    need_uv = var in ("u", "v", "normal")
    model = {}
    for k, src in fields.items():
        if src and src in sfc:
            model[k] = mask_fill(sfc[src])
    if need_uv:
        sub = sfc[["SSU", "SSV"]].drop_vars([c for c in sfc.coords if c.startswith("geo")], errors="ignore")
        grid, sm = tb.make_grid(sub, geom, with_metrics=False)
        wu, wv = wet_of(static, geom, "u"), wet_of(static, geom, "v")
        model["u"] = grid.interp(mask_fill(sm["SSU"]).where(wu > 0) if wu is not None else mask_fill(sm["SSU"]), "X").reset_coords(drop=True)
        model["v"] = grid.interp(mask_fill(sm["SSV"]).where(wv > 0) if wv is not None else mask_fill(sm["SSV"]), "Y").reset_coords(drop=True)
    if wet is not None:
        model = {k: m.where(wet > 0) for k, m in model.items()}
    model_is_cf = hasattr(sfc.indexes["time"], "calendar")
    out = {"variable": var, "convention": "diff = model - OBC (surface level of the OBC file vs the outermost interior row/column of h-points)",
           "model_ssh_variable": ssh, "calendar_used": cal, "obc_dir": d, "segments": {}, "pngs": []}
    for sid, (seg, path) in segs.items():
        v = tb.normal_var(seg) if var == "normal" else var
        if v not in model:
            notes.append(f"segment {sid}: model has no surface field for {v!r}")
            continue
        try:
            ob = seg[v].isel(depth=0) if "depth" in seg[v].dims else seg[v]
            ob = mask_fill(ob)
            if model_is_cf:
                ob = ob.convert_calendar(sfc.indexes["time"].calendar, use_cftime=True, align_on="date")
            line = tb.on_boundary(model[v], seg, topog)
            mdays, odays = _as_days(line.indexes["time"]), _as_days(ob.indexes["time"])
            t0, t1 = max(mdays[0], odays[0]), min(mdays[-1], odays[-1])
            if t1 < t0:
                notes.append(f"segment {sid}: no overlap between OBC times and model times")
                continue
            line = line.isel(time=np.nonzero((mdays >= t0) & (mdays <= t1))[0]).load()
            obi = ob.load().interp(time=line["time"])
            diff = line - obi
            dv = np.asarray(diff.transpose("time", "n"), float)
            fin = np.isfinite(dv)
            rmse_t = np.sqrt(np.nanmean(dv ** 2, axis=1))
            r = {"label": tb.obc_label(seg), "compared_field": v, "n_times": int(line.sizes["time"]),
                 "time_first": _tstr(line.time.values[0]), "time_last": _tstr(line.time.values[-1]),
                 "boundary_line": line.attrs.get("boundary_line"), "n_valid": int(fin.sum()),
                 "bias": _f(np.nanmean(dv), 6), "rmse": _f(np.sqrt(np.nanmean(dv ** 2)), 6),
                 "max_abs_diff": _f(np.nanmax(np.abs(dv)), 6) if fin.any() else None,
                 "obc_mean": _f(np.nanmean(np.where(fin, np.asarray(obi.transpose("time", "n"), float), np.nan)), 6) if fin.any() else None,
                 "model_mean": _f(np.nanmean(np.where(fin, np.asarray(line.transpose("time", "n"), float), np.nan)), 6) if fin.any() else None}
            if fin.any():
                ti, ni = np.unravel_index(np.nanargmax(np.abs(dv)), dv.shape)
                r["max_abs_diff_at"] = {"time": _tstr(line.time.values[ti]), "n": int(ni),
                                        "lon": _f(seg.lon.values[ni], 3), "lat": _f(seg.lat.values[ni], 3)}
            idx = _downsample_idx(len(rmse_t))
            r["rmse_series"] = {"time": [_tstr(line.time.values[i]) for i in idx], "rmse": [_f(rmse_t[i], 6) for i in idx]}
            out["segments"][sid] = r
            # Hovmoller: OBC | model | diff
            xn = tb.along_name(seg)
            xv = np.asarray(seg[xn], float)
            if xn == "lon":
                xv = unwrap_lon(xv)
            tt = np.arange(line.sizes["time"])
            A = np.asarray(obi.transpose("n", "time"), float)
            B = np.asarray(line.transpose("n", "time"), float)
            kind = var_kind(v)
            lo, hi = color_limits(np.concatenate([A[np.isfinite(A)], B[np.isfinite(B)]]) if (np.isfinite(A).any() or np.isfinite(B).any()) else [0, 1],
                                  robust=True, center=0.0 if kind in ("velocity", "ssh") else None)
            fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
            cm = _cmap(default_cmap(v, kind))
            for ax, M, ttl, kw in ((axs[0], A, f"OBC {v} (surface)", dict(cmap=cm, vmin=lo, vmax=hi)),
                                   (axs[1], B, f"model {v} on {r['boundary_line']}", dict(cmap=cm, vmin=lo, vmax=hi)),
                                   (axs[2], B - A, "model - OBC", None)):
                if kw is None:
                    dl, dh = color_limits(M, robust=True, center=0.0)
                    kw = dict(cmap=_cmap("cmo.balance"), vmin=dl, vmax=dh)
                pc = ax.pcolormesh(tt, xv, np.ma.masked_invalid(M), shading="nearest", **kw)
                fig.colorbar(pc, ax=ax)
                ax.set_title(ttl, fontsize=10)
                ticks = np.linspace(0, len(tt) - 1, min(6, len(tt))).round().astype(int)
                ax.set_xticks(ticks)
                ax.set_xticklabels([_tstr(line.time.values[k])[:10] for k in ticks], rotation=30, fontsize=8)
            axs[0].set_ylabel(f"{xn} [deg]")
            fig.suptitle(f"{casename}: {tb.obc_label(seg)} -- bias {r['bias']:.4g}, RMSE {r['rmse']:.4g}" if r["bias"] is not None else f"{casename}: {tb.obc_label(seg)}")
            fig.tight_layout()
            p = _out_path(a["out_dir"], f"{_safe_name(casename)}_obc_vs_model_{v}_seg{sid}.png")
            fig.savefig(p, dpi=100)
            plt.close(fig)
            out["pngs"].append(p)
        except Exception as e:
            notes.append(f"segment {sid}: {type(e).__name__}: {e}")
    out["notes"] = sorted(set(notes))
    return out


# --------------------------------------------------------------------------
# GLORYS
# --------------------------------------------------------------------------

GLORYS_MAP = {"tos": "thetao", "sos": "so", "zos": "zos", "SSH": "zos", "ssh": "zos"}


def op_compare_glorys(a):
    np = _np()
    tb = _tb()
    hist_dir, casename = a["hist_dir"], a["casename"]
    root = a.get("glorys_root") or "/gdex/data/d010049"
    if not os.path.isdir(root) or not os.access(root, os.R_OK | os.X_OK):
        raise PermissionError(f"GLORYS root {root} is not readable from this host (needs /gdex mounted, e.g. Casper/Derecho)")
    dates = list(a["dates"])[: int(a.get("max_dates") or 3)]
    files = tb.glorys_files(dates, root)  # raises FileNotFoundError per missing date
    sfc, static, geom, cal = open_stream(hist_dir, casename, a.get("stream") or "sfc", a.get("calendar"), with_coords=False)
    if "geolon" not in geom:
        lon, lat = lonlat(geom, static, "h")
        geom = geom.assign(geolon=lon, geolat=lat)
    gl = tb.open_glorys(dates, geom, root)
    area = area_of(static, geom, "h")
    wet = wet_of(static, geom, "h")
    notes = []
    out = {"glorys_root": root, "glorys_files": [os.path.basename(f) for f in files], "dates": dates,
           "calendar_used": cal, "results": {}, "pngs": []}
    tidx = sfc.indexes["time"]
    mdays = _as_days(tidx)
    for d in dates:
        key = _time_like(tidx, d + " 12:00")
        k = int(np.argmin(np.abs(mdays - _as_days_scalar(key, tidx))))
        off = abs(mdays[k] - _as_days_scalar(key, tidx))
        if off > 1.0:
            notes.append(f"{d}: nearest model record is {off:.1f} days away; skipped")
            continue
        msel = sfc.isel(time=k)
        gsel = gl.sel(time=np.datetime64(d + "T12:00"), method="nearest").load()
        obs = tb.to_model_grid(gsel, geom)
        res = {"model_time": _tstr(msel.time.values), "glorys_time": _tstr(gsel.time.values)}
        for v in a.get("variables") or ["tos", "sos", "zos"]:
            try:
                mv = resolve_var(sfc, v)
            except KeyError:
                res[v] = {"error": f"{v} not in model stream"}
                continue
            gv = GLORYS_MAP.get(v, GLORYS_MAP.get(mv))
            if gv is None or gv not in obs:
                res[v] = {"error": f"no GLORYS counterpart for {v}"}
                continue
            mod = mask_fill(msel[mv]).load()
            if wet is not None:
                mod = mod.where(wet > 0)
            mod, ob = tb.common_mask(mod.reset_coords(drop=True), obs[gv].reset_coords(drop=True))
            st = tb.area_stats(mod, ob, area if area is not None else xr_ones_like(mod))
            st = {k2: _f(x, 6) if not isinstance(x, int) else x for k2, x in st.items()}
            st["model_var"], st["glorys_var"] = mv, gv
            if gv == "zos":
                st["note"] = "SSH reference levels differ between MOM6 and GLORYS: read pattern_r/RMSE after removing bias"
            res[v] = st
            if a.get("plot"):
                lon, lat = lonlat(geom, static, "h")
                p = _out_path(a["out_dir"], f"{_safe_name(casename)}_glorys_diff_{mv}_{d}.png")
                diffv = np.asarray(mod - ob, float)
                if gv == "zos":
                    diffv = diffv - np.nanmean(diffv)
                map_png(diffv, lon.values, lat.values, p,
                        f"{casename}: {mv} - GLORYS {gv}  {d}" + ("  (mean removed)" if gv == "zos" else ""),
                        cmap="cmo.balance", label=f"{mv} - {gv}", center=0.0, notes=notes)
                out["pngs"].append(p)
        out["results"][d] = res
    out["notes"] = sorted(set(notes))
    return out


def xr_ones_like(da):
    return _xr().ones_like(da)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------


def op_ocean_stats_nc(a):
    """Read RUNDIR/ocean.stats.nc: per-record series of the scalar diagnostics."""
    np = _np()
    xr = _xr()
    ds = xr.open_dataset(a["path"], decode_times=False)
    tname = "Time" if "Time" in ds.dims else ("time" if "time" in ds.dims else None)
    out = {"n_records": int(ds.sizes.get(tname, 0)), "variables": sorted(ds.data_vars)}
    cols = {}
    if tname:
        cols["Day"] = np.asarray(ds[tname], float)
    for v in ("Ntrunc", "En", "max_CFL_trans", "max_CFL_lin", "Mass", "Mass_chg", "Mass_anom",
              "Salt", "Salt_chg", "Salt_anom", "Heat", "Heat_chg", "Heat_anom"):
        if v in ds and ds[v].dims == (tname,):
            cols[v] = np.asarray(ds[v], float)
    out["columns"] = {k: [_f(x) for x in v] for k, v in cols.items()}
    return out


def op_execute_notebook(a):
    """Execute a notebook with nbclient in *this* interpreter's kernel.

    A throwaway kernelspec pointing at sys.executable is used so that a user
    kernelspec called "python3" (which may point at a different env) is not
    picked up.  Cells keep running after an error (allow_errors) so that every
    failing cell is reported.
    """
    import tempfile

    import nbformat

    kdir = tempfile.mkdtemp(prefix="mom6_tools_mcp_kernel_")
    spec_dir = os.path.join(kdir, "kernels", "mom6tools_mcp")
    os.makedirs(spec_dir)
    with open(os.path.join(spec_dir, "kernel.json"), "w") as fh:
        json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                   "display_name": "mom6-tools (mcp)", "language": "python"}, fh)
    os.environ["JUPYTER_PATH"] = kdir + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError, DeadKernelError

    nb = nbformat.read(a["path"], as_version=4)
    client = NotebookClient(nb, timeout=int(a.get("cell_timeout") or 1200), kernel_name="mom6tools_mcp",
                            allow_errors=True, resources={"metadata": {"path": a["cwd"]}})
    fatal = None
    try:
        client.execute()
    except (CellTimeoutError, DeadKernelError) as e:
        fatal = f"{type(e).__name__}: {str(e)[:500]}"
    except Exception as e:  # kernel start failure etc.
        fatal = f"{type(e).__name__}: {str(e)[:500]}"
    nbformat.write(nb, a["out_path"])
    errors, n_exec = [], 0
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        if c.get("execution_count"):
            n_exec += 1
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                src = c.source.strip().splitlines()
                errors.append({"cell_index": i, "first_line": src[0][:120] if src else "", "ename": o.get("ename"),
                               "evalue": str(o.get("evalue"))[:400]})
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    return {"executed_path": a["out_path"], "code_cells": n_code, "cells_executed": n_exec, "errors": errors,
            "fatal": fatal, "ok": not errors and not fatal}


OPS = {
    "locate_templates": op_locate_templates,
    "list_vars": op_list_vars,
    "describe_grid": op_describe_grid,
    "surface_stats": op_surface_stats,
    "plot_field": op_plot_field,
    "plot_section": op_plot_section,
    "obc_summary": op_obc_summary,
    "obc_vs_model": op_obc_vs_model,
    "compare_glorys": op_compare_glorys,
    "ocean_stats_nc": op_ocean_stats_nc,
    "execute_notebook": op_execute_notebook,
}


def main(argv):
    # Keep fd 1 clean for the JSON reply: route all other stdout output to stderr.
    real_out = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    warnings.simplefilter("ignore")
    code = 0
    try:
        if len(argv) < 2 or argv[1] not in OPS:
            raise ValueError(f"usage: _worker.py <op> '<json>'; ops: {sorted(OPS)}")
        args = json.loads(argv[2]) if len(argv) > 2 else {}
        result = OPS[argv[1]](args)
    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
        code = 1
    real_out.write(json.dumps(_clean(result)) + "\n")
    real_out.flush()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
