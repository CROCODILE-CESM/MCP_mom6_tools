"""
mom6-tools MCP -- visualise and debug regional MOM6 output with mom6-tools.

Two pillars:
  A. ready-to-run mom6-tools regional notebooks, pre-filled for a case;
  B. quick in-conversation diagnostics returning small JSON + PNG paths.

The server process stays light (stdlib + fastmcp).  Anything touching
xarray / xgcm / mom6_tools runs in a subprocess under $MOM6_TOOLS_PYTHON
(see _worker.py).
"""

from fastmcp import FastMCP

from mom6_tools_mcp.tools.boundaries import obc_summary, obc_vs_model
from mom6_tools_mcp.tools.discovery import find_mom6_output, summarize_run
from mom6_tools_mcp.tools.fields import (
    describe_grid,
    list_history_variables,
    plot_field,
    plot_section,
    surface_stats,
)
from mom6_tools_mcp.tools.glorys import compare_glorys
from mom6_tools_mcp.tools.notebooks import list_notebooks, render_notebook

INSTRUCTIONS = """\
mom6-tools MCP -- visualisation and debugging of regional MOM6 (CESM/CrocoDash) output,
built on mom6_tools.regional_m6toolbox and the mom6-tools regional notebooks.

WHAT IT GIVES YOU
  Notebooks (for the user to open and explore):
    list_notebooks -> render_notebook(notebook="01"|"02"|"03", out_path, caseroot=...)
      01 grid/xgcm/maps/vorticity/sigma0/GIF, 02 open boundaries, 03 GLORYS12 comparison.
      The notebook comes back with case paths, dates and calendar filled and the dask
      cluster disabled by default; execute=True runs it and reports failing cells.
  Quick diagnostics (for you, in the conversation): small JSON + PNG/GIF paths.
      View returned PNGs to judge them -- they are written to out_dir (default a temp dir).

TYPICAL DEBUGGING FLOW ("my run blew up / looks wrong")
  1. find_mom6_output(caseroot)       where history/ocean.stats/OBC files are; gives hist_dir.
  2. summarize_run(caseroot)          ocean.stats: last date reached, energy growth, max CFL,
                                      truncations, NaNs, drift -> verdict.
  3. surface_stats(hist_dir)          WHEN fields left physical ranges or went NaN, per variable.
  4. plot_field(..., time=<that date>) WHERE: map of tos/sos/zos/speed, derived="vorticity"
                                      (grid-scale noise, |zeta/f|) or "speed"; animate=True for a GIF.
  5. Boundary suspects: obc_summary(input_dir) (OBC files: ranges, NaNs, levels vs floor),
     obc_vs_model(hist_dir, input_dir, var="temp"|"eta"|"normal") (interior vs boundary).
  6. Grid suspects: describe_grid (min dx, max depth, barotropic dt bound, dateline straddle).
  7. Realism: compare_glorys(hist_dir, dates=[...]) bias/RMSE/pattern-r vs GLORYS12;
     plot_section(var="thetao", lat=...) for vertical structure (z stream).
  8. Hand the user render_notebook(...) outputs for deeper interactive work.

PAIRING WITH THE SIBLING SERVERS
  regional-ocean-debugger: find_errors / classify_error / read_run_log (log text, FATAL
    messages), read_mom6_params(filter="DT"|"OBC") and check_cfl / suggest_timestep
    (timestep vs resolution), check_obc_forcing_status, query_domain_knowledge (expert
    advice). Use it for "what does the log say"; use this server for "what do the fields
    look like". summarize_run's max CFL + describe_grid's min dx feed suggest_timestep.
  cesm-runner: xmlquery (RUNDIR, DOUT_S_ROOT, CALENDAR, STOP_N), get_case_status /
    get_job_status / tail_log to know whether the run finished and was archived;
    xmlchange + case_submit to rerun after a fix.
  crocodash: the case's inputdir (OBC segments, topog, init files) and forcing config;
    regenerate forcings there if obc_summary flags bad boundary data.

CONVENTIONS / PITFALLS HANDLED
  * hist_dir = archive <DOUT_S_ROOT>/ocn/hist after short-term archiving, else RUNDIR; any
    tool accepts caseroot= instead and resolves it. casename is inferred when unambiguous.
  * Calendar: MOM6 labels proleptic-Gregorian axes "gregorian"; NO_LEAP cases decode as
    noleap. Taken from the case CALENDAR when caseroot is given; override with calendar=.
  * SSH is zos in some cases, SSH in others -- aliases resolve automatically.
  * Monthly native/z files from a run that stopped mid-month are all fill -> reported, not
    plotted as garbage. Fill values 1e20 / 9.97e36 are blanked everywhere.
  * OBC levels extend below the sea floor by design; stats mask them (mask_below_floor).
  * Coastlines only from locally cached Natural Earth files (compute nodes are offline);
    otherwise maps are drawn without them. Dateline domains are handled on 0..360.
  * Read-only on case, archive, inputdir and GLADE/GDEX data; outputs only go to out_dir.
  * Heavy calls (surface_stats on long runs, animate, compare_glorys, execute) can take
    minutes; tell the user before starting them.
"""

mcp = FastMCP("mom6-tools", instructions=INSTRUCTIONS)

# Pillar A -- notebooks
mcp.add_tool(list_notebooks)
mcp.add_tool(render_notebook)
# Pillar B -- quick diagnostics
mcp.add_tool(find_mom6_output)
mcp.add_tool(list_history_variables)
mcp.add_tool(describe_grid)
mcp.add_tool(summarize_run)
mcp.add_tool(surface_stats)
mcp.add_tool(plot_field)
mcp.add_tool(plot_section)
mcp.add_tool(obc_summary)
mcp.add_tool(obc_vs_model)
mcp.add_tool(compare_glorys)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
