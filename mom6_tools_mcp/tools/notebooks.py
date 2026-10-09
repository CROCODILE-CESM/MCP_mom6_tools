"""Pillar A: render mom6-tools' regional notebooks pre-filled for a case.

The templates are read at runtime from the installed mom6_tools package
(``mom6_tools/nb_templates/regional_notebooks``) -- nothing is vendored here.
Rendering is plain JSON editing (stdlib); execution runs nbclient in the
mom6-tools interpreter through the worker.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path

from ._common import (
    case_info,
    check_writable_location,
    err,
    protected_for,
    resolve_hist,
    run_worker,
)

# Keys of the template's `CASE = dict(...)` and the module-level settings in the
# same cell that render_notebook fills.
CASE_KEYS = ("casename", "hist", "input_dir", "date_range", "length")
DASK_KEYS = {"dask": "DASK", "n_workers": "N_WORKERS", "account": "ACCOUNT", "queue": "QUEUE",
             "interface": "INTERFACE", "memory": "MEMORY", "cores": "CORES", "processes": "PROCESSES",
             "walltime": "WALLTIME"}
GLORYS_NB_MAX_DAYS = 5


def _templates() -> dict:
    env_dir = os.environ.get("MOM6_TOOLS_TEMPLATES_DIR")
    if env_dir:
        files = sorted(p.name for p in Path(env_dir).glob("*.ipynb"))
        return {"templates_dir": env_dir, "templates": files, "execution_available": None}
    return run_worker("locate_templates", {}, timeout=180)


def _src(cell) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def _set_src(cell, text: str):
    lines = text.split("\n")
    cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _lit(v) -> str:
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    return repr(v)


def _knobs(nb) -> dict:
    """Top-level settings with literal values (numbers/strings/lists) that `overrides` can set.

    Dask settings and the CASE-cell values are left out -- they have their own keys.
    """
    import ast

    skip = set(DASK_KEYS.values()) | {"MOM6_CALENDAR", "GLORYS_ROOT"}
    out = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        for ln in _src(c).splitlines():
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*(#.*)?$", ln)
            if not m or m[1] in out or m[1] in skip:
                continue
            try:
                if ast.literal_eval(m[2]) in ({}, [], ()):
                    continue  # accumulators, not settings
            except Exception:
                if m[1] != "MAX_FRAMES":
                    continue
            out[m[1]] = m[2][:80]
    return out


def _describe(name: str, nb) -> dict:
    title, blurb = None, ""
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            s = _src(c).strip()
            m = re.match(r"#\s+(.+)", s)
            if m:
                title = m[1].strip()
                blurb = " ".join(s.splitlines()[1:]).strip()
                break
    sections = [re.sub(r"^#+\s*", "", ln).strip() for c in nb["cells"] if c["cell_type"] == "markdown"
                for ln in _src(c).splitlines() if re.match(r"^##\s", ln)]
    case_keys = []
    for c in nb["cells"]:
        s = _src(c)
        if "CASE = dict(" in s:
            body = s.split("CASE = dict(", 1)[1].split("\n    )", 1)[0]
            case_keys = re.findall(r"^\s*(\w+)\s*=", body, re.M)
    needs = ["hist_dir with <case>.mom6.h.static.nc, .h.ocean_geometry.nc, .h.sfc.*.nc"]
    full = json.dumps(nb)
    if "forcing_obc_segment" in full:
        needs.append("input_dir with forcing_obc_segment_*.nc and ocean_topog_*.nc")
    if "GLORYS" in full and "open_glorys" in full:
        needs.append("GLORYS12 daily files under GLORYS_ROOT (/gdex/data/d010049 on NCAR systems)")
    return {"notebook": name, "title": title, "summary": blurb, "sections": sections,
            "case_parameters": case_keys, "other_knobs": _knobs(nb), "needs": needs,
            "n_cells": len(nb["cells"])}


def list_notebooks() -> dict:
    """The regional analysis notebooks shipped with the installed mom6_tools.

    Read from `<mom6_tools>/nb_templates/regional_notebooks/*.ipynb` in the
    $MOM6_TOOLS_PYTHON environment (or $MOM6_TOOLS_TEMPLATES_DIR).  For each:
    file name, title (first markdown heading), summary, section headings (what
    it shows), the CASE parameters render_notebook fills (casename, hist,
    input_dir, date_range, ...), other top-level knobs you can set through
    `overrides` (e.g. NTIME, HOV_VAR, THIN, MAX_FRAMES) with their template
    defaults, and the data each needs.  Also whether notebook execution
    (nbclient + ipykernel) is available in that environment.

    Currently: 01_output_basics_xgcm (grid, xgcm C-grid interpolation, maps,
    vorticity, sigma0, GIF), 02_boundary_conditions (OBC files vs floor, sections,
    time series, OBC vs interior Hovmollers), 03_glorys_comparison (surface maps,
    bias/RMSE/pattern-r, T-S diagram vs GLORYS12).
    """
    t = _templates()
    if "error" in t:
        return t
    d = t["templates_dir"]
    out = []
    for name in t["templates"]:
        try:
            nb = json.loads(Path(d, name).read_text())
            out.append(_describe(name, nb))
        except Exception as e:
            out.append({"notebook": name, "error": str(e)})
    return {"templates_dir": d, "execution_available": t.get("execution_available"),
            "worker_python": t.get("python"), "notebooks": out}


def _pick_template(notebook: str, t: dict) -> Path:
    p = Path(notebook).expanduser()
    if p.suffix == ".ipynb" and p.is_file():
        return p
    names = t["templates"]
    key = notebook.strip().lower().removesuffix(".ipynb")
    hits = [n for n in names if n.lower().removesuffix(".ipynb") == key] or \
           [n for n in names if n.lower().startswith(key)] or [n for n in names if key in n.lower()]
    if len(hits) != 1:
        raise ValueError(f"notebook {notebook!r} matches {hits or 'nothing'}; available: {names}")
    return Path(t["templates_dir"], hits[0])


def _date_range_from_case(info: dict) -> list[str] | None:
    try:
        start = _dt.date.fromisoformat(info["RUN_STARTDATE"])
        n, opt = int(info.get("STOP_N") or 0), (info.get("STOP_OPTION") or "").lower()
        days = {"ndays": n, "nday": n, "nmonths": 30 * n, "nmonth": 30 * n, "nyears": 365 * n, "nyear": 365 * n}.get(opt)
        if not days:
            return None
        return [start.isoformat(), (start + _dt.timedelta(days=days)).isoformat()]
    except Exception:
        return None


def render_notebook(notebook: str, out_path: str, caseroot: str | None = None, hist_dir: str | None = None,
                    casename: str | None = None, input_dir: str | None = None, overrides: dict | None = None,
                    execute: bool = False, overwrite: bool = False, execute_timeout: int = 1200,
                    fix_known_issues: bool = True) -> dict:
    """Write a ready-to-run copy of a mom6-tools regional notebook for one case.

    `notebook`: "01", "02", "03", a template file name (see list_notebooks), or
    a path to any .ipynb using the same layout.  `out_path`: where to write the
    .ipynb (its directory is created; figures the notebook saves land in a
    `diagnostic_figures/` folder beside it).  Refuses to overwrite unless
    overwrite=True, and refuses to write inside case/archive/input/data dirs.

    Filled automatically (each can be forced through `overrides`):
      * casename, hist (history dir), input_dir -- from `caseroot`
        (env_case/env_run XML: CASE, RUNDIR, DOUT_S_ROOT; user_nl_mom INPUTDIR)
        or from the explicit arguments;
      * date_range -- the first..last record of the sfc stream (or
        RUN_STARTDATE + STOP_N); for 03_glorys_comparison capped to 5 days because
        every day reads a 1.3 GB GLORYS file;
      * MOM6_CALENDAR -- from CALENDAR (NO_LEAP -> "noleap") or the file's axis;
      * GLORYS_ROOT (override key "glorys_root").
    The dask cell is rewritten so no cluster starts by default (DASK = None runs
    in-kernel, enough for regional runs); the hard-coded account is replaced by
    $PROJECT / $PBS_ACCOUNT.  Override keys for it: dask ("LocalCluster" or
    "PBSCluster"), n_workers, account, queue, interface, memory, cores,
    processes, walltime.  Any other top-level variable in the notebook can be
    overridden by name (e.g. {"NTIME": 10, "HOV_VAR": "salt", "THIN": 4,
    "MAX_FRAMES": 20, "figdir": "figs"}).

    `fix_known_issues=True` patches known template bugs (reported in
    `patches`), e.g. 01's notebook-local save_field_gif signature mismatch, by
    importing the mom6_tools copy of the function in that cell.

    execute=True runs the rendered notebook with nbclient in the mom6-tools env
    (cwd = the notebook's folder, per-cell timeout `execute_timeout` s, keeps
    going after errors) and writes `<name>_executed.ipynb`, returning per-cell
    errors.  Returns filled parameters, `needs_attention` (anything unresolved or
    still hard-coded) and the paths.
    """
    overrides = dict(overrides or {})
    t = _templates()
    if "error" in t:
        return t
    try:
        src_path = _pick_template(notebook, t)
    except ValueError as e:
        return err(str(e))
    out = Path(out_path).expanduser()
    if out.suffix != ".ipynb":
        out = out.with_suffix(".ipynb")
    attention, filled, patches = [], {}, []

    info = None
    try:
        if caseroot:
            info = case_info(caseroot)
    except Exception as e:
        return err(f"could not read caseroot: {e}")
    try:
        hist_dir, casename, info2 = resolve_hist(hist_dir, casename, caseroot)
        info = info or info2
    except Exception as e:
        attention.append(f"history dir not resolved ({e}); set CASE['hist'] by hand")
        casename = casename or (info or {}).get("CASE")
    input_dir = input_dir or (info or {}).get("input_dir")
    try:
        check_writable_location(out.parent, protected_for(hist_dir, info, input_dir))
    except PermissionError as e:
        return err(str(e))
    if out.exists() and not overwrite:
        return err(f"{out} exists; pass overwrite=True to replace it")

    calendar = overrides.pop("calendar", None) or (info or {}).get("mom_calendar")
    date_range = overrides.pop("date_range", None)
    span = None
    if hist_dir and casename and (not date_range or not calendar):
        lv = run_worker("list_vars", {"hist_dir": hist_dir, "casename": casename, "stream": "sfc"}, timeout=300)
        if "error" not in lv:
            span = [lv.get("time_first", "")[:10], lv.get("time_last", "")[:10]]
            calendar = calendar or lv.get("calendar_used")
        else:
            attention.append(f"could not read the sfc stream: {lv['error']}")
    if not date_range:
        if span and all(span):
            date_range = span
            filled["date_range_source"] = "first..last record of the sfc stream"
        elif info and _date_range_from_case(info):
            date_range = _date_range_from_case(info)
            filled["date_range_source"] = "RUN_STARTDATE + STOP_N"
        else:
            attention.append("date_range unknown; template default kept -- set it")
    is_glorys_nb = "glorys" in src_path.name.lower()
    if date_range and is_glorys_nb and "date_range" not in (overrides or {}):
        d0, d1 = _dt.date.fromisoformat(date_range[0][:10]), _dt.date.fromisoformat(date_range[1][:10])
        if (d1 - d0).days + 1 > GLORYS_NB_MAX_DAYS:
            date_range = [d0.isoformat(), (d0 + _dt.timedelta(days=GLORYS_NB_MAX_DAYS - 1)).isoformat()]
            attention.append(f"03: date_range capped to {GLORYS_NB_MAX_DAYS} days (each day reads a 1.3 GB GLORYS file); "
                             "override date_range to widen")
    if not calendar:
        calendar = "proleptic_gregorian"
        attention.append("calendar not found; MOM6_CALENDAR left at proleptic_gregorian (use 'noleap' for NO_LEAP cases)")
    glorys_root = overrides.pop("glorys_root", None)
    if is_glorys_nb:
        gr = glorys_root or "/gdex/data/d010049"
        if not os.access(gr, os.R_OK | os.X_OK):
            attention.append(f"GLORYS root {gr} not readable from this host (03 needs it)")
    if input_dir is None and "boundary" in src_path.name.lower():
        attention.append("input_dir unknown: 02 needs the dir with forcing_obc_segment_*.nc and ocean_topog_*.nc")

    nb = json.loads(src_path.read_text())
    case_vals = {"casename": casename, "hist": hist_dir, "input_dir": input_dir, "date_range": date_range,
                 "length": overrides.pop("length", None)}
    for k in CASE_KEYS:
        if k in overrides:
            case_vals[k] = overrides.pop(k)
    dask_vals = {DASK_KEYS[k]: overrides.pop(k) for k in list(overrides) if k in DASK_KEYS}
    figdir = overrides.pop("figdir", None)

    found_case = found_dask = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = _src(c)
        if "CASE = dict(" in s:
            found_case = True
            for k, v in case_vals.items():
                if v is None:
                    continue
                pat = {"date_range": rf"(\b{k}\s*=\s*)\[[^\]]*\]"}.get(k, rf"(\b{k}\s*=\s*)(\"[^\"]*\"|'[^']*')")
                s2, n = re.subn(pat, lambda m: m.group(1) + _lit(v), s, count=1)
                if n:
                    s = s2
                    filled[k] = v
                else:
                    attention.append(f"CASE key {k!r} not found in the template; not filled")
            for var, val in (("MOM6_CALENDAR", calendar), ("GLORYS_ROOT", glorys_root)):
                if val is None:
                    continue
                s2, n = re.subn(rf"^({var}\s*=\s*).*$", lambda m: m.group(1) + _lit(val), s, count=1, flags=re.M)
                if n:
                    s = s2
                    filled[var] = val
            _set_src(c, s)
        elif "get_cluster(" in s and "import" not in s.split("get_cluster(")[0].splitlines()[-1]:
            found_dask = True
            vals = {"DASK": None, "N_WORKERS": 4, "QUEUE": "casper", "INTERFACE": "ext", "MEMORY": "20GB",
                    "CORES": 4, "PROCESSES": 1, "WALLTIME": "03:00:00"}
            for var in vals:  # keep the template's own defaults when present
                m = re.search(rf"^{var}\s*=\s*(.+?)\s*(#.*)?$", s, re.M)
                if m and var != "DASK":
                    vals[var] = m.group(1)
                else:
                    vals[var] = _lit(vals[var])
            for var, v in dask_vals.items():
                vals[var] = _lit(v)
            acct = _lit(dask_vals["ACCOUNT"]) if "ACCOUNT" in dask_vals else \
                'os.environ.get("PROJECT") or os.environ.get("PBS_ACCOUNT")'
            new = (
                "# --- Dask cluster (rewritten by the mom6-tools MCP render_notebook) ---\n"
                "# DASK = None runs everything in this kernel (threads) -- plenty for most regional runs.\n"
                "# Set DASK = \"LocalCluster\" (inside a PBS job with spare cores) or \"PBSCluster\"\n"
                "# (submits one PBS job per worker) to scale out.\n"
                f"DASK = {vals['DASK']}\n\n"
                f"N_WORKERS = {vals['N_WORKERS']}\n"
                f"ACCOUNT = {acct}  # charge PBS worker jobs here\n"
                f"QUEUE = {vals['QUEUE']}  # or \"derecho\"\n"
                f"INTERFACE = {vals['INTERFACE']}  # or \"ib0\" on derecho\n\n"
                "# Resources for each worker (total = N_WORKERS * each)\n"
                f"MEMORY = {vals['MEMORY']}\nCORES = {vals['CORES']}\nPROCESSES = {vals['PROCESSES']}\n"
                f"WALLTIME = {vals['WALLTIME']}\n\n"
                "client = cluster = None\n"
                "if DASK:\n"
                "    success, client, cluster = get_cluster(nw=N_WORKERS, cluster_class=DASK, account=ACCOUNT,\n"
                "                                           queue=QUEUE, walltime=WALLTIME, memory=MEMORY,\n"
                "                                           processes=PROCESSES, cores=CORES, interface=INTERFACE,\n"
                "                                           log_directory=\"logs\")"
            )
            _set_src(c, new)
            filled["dask"] = {"DASK": vals["DASK"], "ACCOUNT": acct}
    if not found_case:
        attention.append("no `CASE = dict(...)` cell found in this template; case parameters NOT filled")
    if not found_dask:
        filled["dask"] = "no get_cluster cell in template"

    # generic top-level overrides
    if figdir:
        overrides["FIGDIR"] = ("__path__", figdir)
    for key, val in overrides.items():
        hit = False
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = _src(c)
            lit = f"Path({_lit(val[1])})" if isinstance(val, tuple) and val and val[0] == "__path__" else _lit(val)
            s2, n = re.subn(rf"^({re.escape(key)}\s*=\s*)(.+?)(\s*#.*)?$", lambda m: m.group(1) + lit + (m.group(3) or ""),
                            s, count=1, flags=re.M)
            if n:
                _set_src(c, s2)
                hit = True
                filled[key] = val[1] if isinstance(val, tuple) else val
                break
        if not hit:
            attention.append(f"override {key!r}: no top-level `{key} = ...` in the template; ignored")

    # known template bugs
    if fix_known_issues:
        local_def = any(re.search(r"def save_field_gif\(field, frames_dir,", _src(c)) for c in nb["cells"])
        if local_def:
            for c in nb["cells"]:
                s = _src(c)
                if c["cell_type"] == "code" and re.search(r"^\s*gif = save_field_gif\(", s, re.M) and \
                        "# from mom6_tools.regional_m6toolbox import save_field_gif" in s:
                    _set_src(c, s.replace("# from mom6_tools.regional_m6toolbox import save_field_gif",
                                          "from mom6_tools.regional_m6toolbox import save_field_gif  "
                                          "# MCP patch: notebook-local copy takes an extra frames_dir arg the call omits"))
                    patches.append("01: GIF cell imports mom6_tools.regional_m6toolbox.save_field_gif -- the notebook-local "
                                   "definition has an extra `frames_dir` positional parameter the call does not pass "
                                   "(TypeError) and builds frame paths from out_path.stem")

    # leftovers that still point at someone else's data / account
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = _src(c)
        if "P93300012" in s:
            attention.append(f"cell {i}: hard-coded account P93300012 remains")
        for m in re.findall(r"[\"'](/glade/[^\"']+)[\"']", s):
            if m not in (hist_dir, input_dir):
                attention.append(f"cell {i}: hard-coded path {m}")
        c["outputs"] = []
        c["execution_count"] = None

    banner = {
        "cell_type": "markdown", "metadata": {},
        "source": [
            f"> **Rendered by the mom6-tools MCP server** on {_dt.date.today().isoformat()} from "
            f"`{src_path.name}` for case **{casename}**.\n",
            ">\n",
            f"> history: `{hist_dir}`  \n",
            f"> input_dir: `{input_dir}`  \n",
            f"> date_range: `{date_range}`, calendar: `{calendar}`  \n",
            "> Dask is off by default (`DASK = None`); edit the cluster cell to scale out. Run all cells.\n",
        ],
    }
    nb["cells"].insert(1, banner)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    res = {"notebook_path": str(out), "template": str(src_path), "filled": filled, "patches": patches,
           "needs_attention": attention, "figures_dir": str(out.parent / "diagnostic_figures")}
    if execute:
        if t.get("execution_available") is False:
            res["executed"] = {"error": "nbclient/ipykernel not available in the mom6-tools env",
                               "modules": t.get("execution_modules")}
            return res
        exe = out.with_name(out.stem + "_executed.ipynb")
        if exe.exists() and not overwrite:
            res["executed"] = {"error": f"{exe} exists; pass overwrite=True"}
            return res
        res["executed"] = run_worker("execute_notebook", {"path": str(out), "out_path": str(exe), "cwd": str(out.parent),
                                                          "cell_timeout": int(execute_timeout)},
                                     timeout=int(execute_timeout) * 3 + 300)
    return res
