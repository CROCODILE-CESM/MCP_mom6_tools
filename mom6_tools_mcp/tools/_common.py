"""Shared stdlib-only helpers: running the worker, reading CESM case XML,
resolving history / input directories, and guarding output locations.

Nothing here imports xarray or mom6_tools -- the server process stays light.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

WORKER = Path(__file__).resolve().parent.parent / "_worker.py"

# Never write outputs under these, whatever the caller asks for.
PROTECTED_PREFIXES = ("/gdex", "/glade/campaign", "/glade/collections")


# --------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------


def worker_python() -> str:
    """Interpreter for heavy work: $MOM6_TOOLS_PYTHON, else this interpreter."""
    return os.environ.get("MOM6_TOOLS_PYTHON") or sys.executable


def run_worker(op: str, args: dict, timeout: int = 900) -> dict:
    """Run one worker op in the mom6-tools interpreter; return its JSON reply.

    On any failure a dict with an "error" key is returned (never raises), with
    the tail of the worker's stderr when that helps.
    """
    py = worker_python()
    cmd = [py, str(WORKER), op, json.dumps(args)]
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except FileNotFoundError:
        return {"error": f"worker interpreter not found: {py} (set MOM6_TOOLS_PYTHON to the mom6-tools env python)"}
    except subprocess.TimeoutExpired:
        return {"error": f"worker op {op!r} timed out after {timeout}s"}
    lines = [ln for ln in (p.stdout or "").strip().splitlines() if ln.strip()]
    try:
        result = json.loads(lines[-1])
    except Exception:
        return {
            "error": f"worker op {op!r} produced no JSON (exit {p.returncode})",
            "stderr_tail": (p.stderr or "")[-2500:],
            "python": py,
        }
    if isinstance(result, dict) and "error" in result:
        tail = (p.stderr or "").strip()[-1500:]
        if tail:
            result["stderr_tail"] = tail
        result["python"] = py
    return result


# --------------------------------------------------------------------------
# CESM case XML
# --------------------------------------------------------------------------

_VAR = re.compile(r"\$(?:\{(\w+)\}|(\w+))")
_ENV = re.compile(r"\$ENV\{(\w+)\}")


def read_case_vars(caseroot: str | os.PathLike) -> dict:
    """All <entry id=... value=...> pairs from CASEROOT/env_*.xml, $VARs expanded."""
    root = Path(caseroot).expanduser()
    raw: dict[str, str] = {}
    for f in sorted(root.glob("env_*.xml")):
        try:
            tree = ET.parse(f)
        except ET.ParseError:
            continue
        for e in tree.iter("entry"):
            k, v = e.get("id"), e.get("value")
            if k and v is not None and k not in raw:
                raw[k] = v
    raw.setdefault("CASEROOT", str(root))

    def expand(v: str, depth: int = 0) -> str:
        v = _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
        if depth > 8:
            return v
        new = _VAR.sub(lambda m: raw.get(m.group(1) or m.group(2), m.group(0)), v)
        return expand(new, depth + 1) if new != v else new

    return {k: expand(v) for k, v in raw.items()}


def mom_calendar(cesm_calendar: str | None) -> str | None:
    """CESM CALENDAR (NO_LEAP / GREGORIAN) -> calendar for decode_mom6_time."""
    if not cesm_calendar:
        return None
    c = cesm_calendar.strip().upper()
    if c in ("NO_LEAP", "NOLEAP", "365_DAY", "365DAY"):
        return "noleap"
    if c in ("GREGORIAN", "PROLEPTIC_GREGORIAN", "STANDARD", "JULIAN"):
        return "proleptic_gregorian"
    if c in ("360_DAY", "360DAY"):
        return "360_day"
    return cesm_calendar.lower()


def user_nl_mom_inputdir(caseroot: str | os.PathLike) -> str | None:
    """INPUTDIR from CASEROOT/user_nl_mom (the CrocoDash-generated ocean input dir)."""
    f = Path(caseroot).expanduser() / "user_nl_mom"
    if not f.exists():
        return None
    for line in f.read_text(errors="replace").splitlines():
        line = line.split("!")[0].strip()
        m = re.match(r"INPUTDIR\s*=\s*(.+)$", line, re.I)
        if m:
            return os.path.expanduser(m.group(1).strip().strip("'\""))
    return None


def case_info(caseroot: str | os.PathLike) -> dict:
    """The case facts the tools need, from a CASEROOT (read-only)."""
    root = Path(caseroot).expanduser()
    if not (root / "env_run.xml").exists():
        raise FileNotFoundError(f"{root} has no env_run.xml -- not a CESM CASEROOT")
    v = read_case_vars(root)
    info = {k: v.get(k) for k in ("CASE", "RUNDIR", "DOUT_S", "DOUT_S_ROOT", "CALENDAR", "RUN_STARTDATE",
                                  "STOP_OPTION", "STOP_N", "CONTINUE_RUN", "COMPSET", "OCN_GRID")}
    info["CASEROOT"] = str(root)
    info["mom_calendar"] = mom_calendar(v.get("CALENDAR"))
    info["archive_hist_dir"] = os.path.join(v["DOUT_S_ROOT"], "ocn", "hist") if v.get("DOUT_S_ROOT") else None
    info["input_dir"] = user_nl_mom_inputdir(root)
    return info


def caseroot_from_rundir(rundir: str | os.PathLike) -> str | None:
    """CESM writes RUNDIR/CASEROOT containing the case path."""
    f = Path(rundir).expanduser() / "CASEROOT"
    if f.is_file():
        p = f.read_text(errors="replace").strip().splitlines()
        if p and Path(p[0].strip()).is_dir():
            return p[0].strip()
    return None


# --------------------------------------------------------------------------
# history files
# --------------------------------------------------------------------------


def mom6_files(d: str | os.PathLike, casename: str | None = None) -> list[str]:
    d = str(Path(d).expanduser())
    pat = f"{glob.escape(casename)}.mom6.*.nc" if casename else "*.mom6.*.nc"
    return sorted(glob.glob(os.path.join(glob.escape(d), pat)))


STREAM_RE = re.compile(
    r"^(?P<case>.+?)\.mom6\.(?P<stream>.+?)"
    r"(?:[._](?P<date>\d{4}(?:-\d{2}){0,2}(?:-\d{5})?))?\.nc$"
)
GRID_STREAMS = ("static", "ocean_geometry")


def history_streams(d: str | os.PathLike, casename: str | None = None) -> set[str]:
    """Stream names in d ("h." prefix stripped; restart and ic files left out)."""
    out = set()
    for f in mom6_files(d, casename):
        m = STREAM_RE.match(os.path.basename(f))
        if m and not re.match(r"(r|ic)(\.|$)", m["stream"]):
            out.add(m["stream"].removeprefix("h."))
    return out


def is_history_dir(streams) -> bool:
    """The static file plus at least one stream that is not a grid file."""
    return "static" in streams and any(s not in GRID_STREAMS for s in streams)


def infer_casename(hist_dir: str | os.PathLike) -> str:
    """The case name, when exactly one case's *.mom6.* files are in the directory."""
    names = {os.path.basename(f).split(".mom6.")[0] for f in mom6_files(hist_dir)}
    if len(names) == 1:
        return names.pop()
    if not names:
        raise FileNotFoundError(f"no *.mom6.*.nc files in {hist_dir}")
    raise ValueError(f"several cases in {hist_dir}: {sorted(names)} -- pass casename")


def resolve_hist(hist_dir: str | None = None, casename: str | None = None, caseroot: str | None = None) -> tuple[str, str, dict | None]:
    """(hist_dir, casename, case_info|None) from whatever the caller gave.

    With a caseroot: the short-term archive ocn/hist if it holds MOM6 history,
    else RUNDIR. A directory counts only if it has the static file plus at
    least one other history stream (ocean_geometry/static alone do not), the
    same rule as find_mom6_output's hist_dir_for_tools.
    """
    info = None
    if caseroot:
        info = case_info(caseroot)
        casename = casename or info.get("CASE")
        if not hist_dir:
            seen = []
            for label, cand in (
                ("archive", info.get("archive_hist_dir")),
                ("RUNDIR", info.get("RUNDIR")),
            ):
                if not cand:
                    continue
                streams = history_streams(cand, casename)
                if is_history_dir(streams):
                    hist_dir = cand
                    break
                if not os.path.isdir(cand):
                    has = "does not exist"
                else:
                    has = ", ".join(sorted(streams)) or "no MOM6 history files"
                seen.append(f"{label} {cand}: {has}")
            else:
                raise FileNotFoundError(
                    f"no MOM6 history for {casename} ({'; '.join(seen)}). A history "
                    "dir needs <case>.mom6.h.static.nc plus at least one history "
                    "stream (e.g. sfc); has the run written output yet?"
                )
    if not hist_dir:
        raise ValueError("give hist_dir (or caseroot)")
    hist_dir = str(Path(hist_dir).expanduser())
    if not os.path.isdir(hist_dir):
        raise FileNotFoundError(f"hist_dir {hist_dir} does not exist")
    casename = casename or infer_casename(hist_dir)
    return hist_dir, casename, info


def obc_dir_of(input_dir: str) -> str | None:
    """Directory holding forcing_obc_segment_*.nc: input_dir or its ocn/ ocean/ ocnice/."""
    base = Path(input_dir).expanduser()
    for d in (base, base / "ocn", base / "ocean", base / "ocnice"):
        if glob.glob(os.path.join(glob.escape(str(d)), "forcing_obc_segment_*.nc")):
            return str(d)
    return None


# --------------------------------------------------------------------------
# output locations
# --------------------------------------------------------------------------


def _within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_out_dir(out_dir: str | None, protected: list[str | None] = ()) -> str:
    """Create/validate the output directory.

    None -> a fresh temp dir (honours $TMPDIR).  Refuses any location inside a
    protected data directory (case dirs, history/archive dirs, inputdirs, GLADE
    data collections) so the tools can never write into model data.
    """
    if not out_dir:
        return tempfile.mkdtemp(prefix="mom6_tools_mcp_")
    out = Path(out_dir).expanduser().resolve()
    check_writable_location(out, protected)
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def check_writable_location(out: Path, protected: list[str | None] = ()) -> None:
    out = Path(out).expanduser().resolve()
    for pref in PROTECTED_PREFIXES:
        if _within(out, Path(pref)):
            raise PermissionError(f"refusing to write under {pref} (data collection): {out}")
    for p in protected:
        if not p:
            continue
        pp = Path(p).expanduser().resolve()
        if _within(out, pp):
            raise PermissionError(f"refusing to write inside a model data directory ({pp}): {out}")


def protected_for(hist_dir: str | None = None, info: dict | None = None, input_dir: str | None = None) -> list[str]:
    """Directories outputs must never land in, for a given case."""
    out = []
    if hist_dir:
        h = Path(hist_dir).expanduser()
        out.append(str(h))
        if h.name == "hist" and h.parent.name == "ocn":  # .../archive/<case>/ocn/hist
            out.append(str(h.parent.parent))
    if info:
        for k in ("CASEROOT", "RUNDIR", "DOUT_S_ROOT", "input_dir"):
            if info.get(k):
                out.append(info[k])
    if input_dir:
        out.append(input_dir)
        p = Path(input_dir).expanduser()
        if p.name in ("ocn", "ocean", "ocnice"):
            out.append(str(p.parent))
    return out


def err(msg: str, **extra) -> dict:
    d = {"error": msg}
    d.update(extra)
    return d
