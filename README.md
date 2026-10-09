# mom6-tools-mcp

MCP server for **visualising and debugging regional MOM6 output** (CESM / CrocoDash cases) with
[mom6-tools](https://github.com/NCAR/mom6-tools).

Two pillars:

- **Notebooks.** Render mom6-tools' regional notebooks (`01_output_basics_xgcm`,
  `02_boundary_conditions`, `03_glorys_comparison`) pre-filled for a case: paths, dates, calendar.
  The dask cluster is off by default and known template bugs are patched, so the user gets a
  notebook they can run straight away. Optionally execute it and get back the failing cells.
- **Quick debugging.** Tools that answer in the conversation with small JSON plus PNG/GIF paths
  (an LLM client can open the PNGs): did it blow up, when, where, are the boundaries sane, how
  does it compare with GLORYS12.

The server process is light (stdlib + fastmcp). Anything that needs xarray, xgcm or mom6_tools runs
in a subprocess under the mom6-tools environment (`mom6_tools_mcp/_worker.py`). The worker returns
one JSON document and never returns arrays. Series are capped at 200 points.

## Environment

| Variable | Meaning |
|---|---|
| `MOM6_TOOLS_PYTHON` | Python of an env with mom6_tools, xarray, xgcm, dask, cartopy, cmocean, matplotlib, netCDF4, pillow (plus nbclient/ipykernel for notebook execution). Default: the server's own interpreter, e.g. `/path/to/conda-envs/mom6-tools/bin/python` |
| `MOM6_TOOLS_TEMPLATES_DIR` | Optional: read notebook templates from here instead of the installed `mom6_tools/nb_templates/regional_notebooks` |
| `MOM6_TOOLS_MCP_ALLOW_DOWNLOAD=1` | Let cartopy download Natural Earth coastlines. By default only shapefiles already on disk are used, because compute nodes are offline. Maps are drawn without coastlines rather than failing. |

## Setup and running

```bash
pip install -e .                       # into any env with fastmcp >= 2
export MOM6_TOOLS_PYTHON=/path/to/conda-envs/mom6-tools/bin/python
mom6-tools-mcp                         # or: python server.py  /  python -m mom6_tools_mcp
```

`.mcp.json`:

```json
{
  "mcpServers": {
    "mom6-tools": {
      "command": "/path/to/env-with-fastmcp/bin/python",
      "args": ["/path/to/MCP_mom6_tools/server.py"],
      "env": {"MOM6_TOOLS_PYTHON": "/path/to/conda-envs/mom6-tools/bin/python"}
    }
  }
}
```

## Tools

Every tool accepts `caseroot=` in place of explicit paths. RUNDIR, DOUT_S_ROOT, CASE and CALENDAR
come from `env_*.xml`, and the CrocoDash INPUTDIR comes from `user_nl_mom`. `casename` is inferred
when only one case is present.

| Tool | Purpose |
|---|---|
| `list_notebooks()` | Regional notebook templates in the installed mom6_tools: title, sections, CASE parameters, other knobs, data needed, whether execution is available |
| `render_notebook(notebook, out_path, caseroot=..., overrides=..., execute=False)` | Ready-to-run copy with casename/hist/input_dir/date_range/calendar filled, dask cluster optional, hard-coded account removed, known bugs patched; `execute=True` runs it with nbclient and reports cell errors |
| `find_mom6_output(caseroot \| rundir, archive_dir, input_dir)` | Streams with file counts and date ranges (archive and RUNDIR), static/geometry presence, ocean.stats, ic/restart files, OBC/topog/grid/init files in the inputdir |
| `list_history_variables(hist_dir, casename, stream)` | Variables, dims, C-grid position, units, long_name, n_time, decoded time span, calendar, SSH name, empty files |
| `describe_grid(hist_dir, plot=False)` | Lon/lat range and convention, dateline straddle, wet fraction, depth stats, area, dx/dy, smallest wet spacing, barotropic dt bound; optional bathymetry PNG |
| `summarize_run(rundir_or_caseroot)` | ocean.stats: last date reached (any calendar), energy growth, max CFL, truncations, NaNs, mass/heat/salt drift, verdict |
| `surface_stats(hist_dir, variables, start, end)` | Area-weighted mean/min/max per record, NaN/fill fraction in wet cells, physical-range flags with the first time each one trips |
| `plot_field(hist_dir, var \| derived, time, level, animate)` | Map PNG (or GIF of up to 60 frames); `derived` = speed, vorticity (+ max \|zeta/f\|), sigma0 |
| `plot_section(hist_dir, var, stream="z", lat \| lon)` | Vertical section PNG along the nearest row/column, with the sea floor; native stream uses `layer_depths(e)` |
| `obc_summary(input_dir, plot=False)` | Per segment: orientation, sizes, dates, OBC levels vs sea floor, per-variable ranges/NaNs after masking below the floor; optional map and sections |
| `obc_vs_model(hist_dir, input_dir, var)` | Model surface along each boundary vs the OBC: bias/RMSE/max diff per segment and a Hovmöller PNG |
| `compare_glorys(hist_dir, dates, variables)` | Area-weighted bias/RMSE/pattern-r vs GLORYS12 (`/gdex/data/d010049`) for up to `max_dates` days; optional difference maps |

Outputs go only to `out_dir` (default: a fresh temp dir, honouring `$TMPDIR`). The tools refuse to
write inside case directories, RUNDIR, the archive, inputdirs, `/gdex`, `/glade/campaign` and
`/glade/collections`. All data access is read-only.

## Walkthrough: "my run blew up / looks wrong"

Uses this server together with **regional-ocean-debugger** (logs, MOM_input, CFL advice) and
**cesm-runner** (case state).

1. **Did it finish?** Use cesm-runner `get_case_status` / `tail_log`, or this server's
   `find_mom6_output(caseroot)` to see which streams exist, up to which month, and whether
   ocean.stats is still in RUNDIR.
2. **How did it die?** Run `summarize_run(caseroot)`. The verdict tells you the last date
   reached, when energy started growing, the max CFL and when it occurred, and when truncations
   began. In parallel, run the debugger's `find_errors` and then `classify_error` on the
   ocn.log text.
3. **When did fields go bad?** Run `surface_stats(caseroot=..., end=<last date>)` for the first
   time T, S, SSH or speed left a physical range or went NaN.
4. **Where?** Call `plot_field(caseroot=..., var="tos", time=<that date>)`. For grid-scale noise
   use `derived="vorticity"` (look at `max_abs_zeta_over_f`); for jets use `derived="speed"`.
   `animate=True` shows it developing over time. Noise or extremes hugging an edge point to the
   boundaries.
5. **Boundaries.** `obc_summary(caseroot=...)` checks the forcing files themselves (NaNs, absurd
   values, OBC levels that stop above the sea floor). `obc_vs_model(caseroot=..., var="eta")` or
   `var="normal"` checks whether the interior follows the boundary. Then use the debugger's
   `read_mom6_params(filter="OBC")` and `query_domain_knowledge("OBC sponge")`.
6. **Grid and timestep.** `describe_grid` gives the smallest wet dx and a barotropic dt bound. Feed
   that to the debugger's `check_cfl` / `suggest_timestep`. If DT is too large, cesm-runner
   `xmlchange` plus `case_submit` reruns it.
7. **Realism, once it runs.** Use `compare_glorys(caseroot=..., dates=[...])` and
   `plot_section(var="thetao", lat=...)`. For a deeper look, call
   `render_notebook("02", out_path, caseroot=...)` or `"03"` and give the user the notebook.

## Pitfalls handled

- **Calendar.** MOM6 labels proleptic-Gregorian axes `gregorian`. NO_LEAP cases are decoded as
  noleap. The ocean.stats `Day` column is converted with the case calendar.
- **SSH naming.** SSH may be `zos` or `SSH`. Area may be `areacello` or `area_t`, and depth may be
  `deptho`, `depth_ocean` or geometry `D`. Each tool probes for whichever name is present.
- **Incomplete monthly files.**
  - native/z files from a partial month hold fill values (9.97e36). These are reported as such
    rather than plotted.
  - Monthly files with 0 records give a clear error.
- **OBC levels below the floor.** OBC levels extend below the sea floor, so the tools mask them
  with `mask_below_floor`. OBC times are converted to noleap for noleap runs.
- **C-grid placement.** SSU/SSV are interpolated onto tracer points with xgcm (`make_grid`).
  Vorticity is computed on corner points.
- **Dateline.** Domains that cross the dateline are plotted on continuous 0..360 longitudes.
- **Coastlines.** Cartopy coastlines are used only when the shapefiles are cached locally.

## Not covered

- Budgets.
- mom6-tools' CLI drivers. These need `diag_config.yml` and assume global basins.
- BGC tracers in OBC files. `open_obc` expects exactly temp, salt, u, v and eta.
