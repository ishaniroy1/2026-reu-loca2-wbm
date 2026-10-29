"""
figures_sp_ratio_overview.py
=============================
Summary figures for the Northeast snowmelt/runoff trends project, ensemble-
mean S/P ratio (averaged across all available GCMs -- see
climatology.multi_model_climatology, which keeps every model's climatology
individually and only averages across models as the final step), for three
seasonal windows: water year, cold season (Nov-Apr), and winter (Dec-Feb).

For each window:
  1. Historical baseline S/P ratio map (single panel), 1980-2013.
  2. A 2 (mid-century / late-century) x 3 (SSP2-4.5 / SSP3-7.0 / SSP5-8.5)
     grid of S/P ratio maps, sharing one color scale across all six panels.

Color scale: vmin is fixed at 0 (a meaningful physical floor -- "no snow").
vmax is NOT a flat 1.0 -- it's computed from the historical ensemble's own
98th percentile FOR THAT WINDOW, then reused as a FIXED value across the
historical, mid-century, and late-century figures for that window. This
both uses the colorbar's dynamic range properly (actual data rarely
approaches 1.0) and keeps the three separately-saved images visually
comparable to each other, since the whole point of the three figures is
to show the ratio declining over time -- an independently auto-scaled
vmax per figure would silently defeat that.

Each ensemble S/P ratio grid is also saved as NetCDF (not just PNG) to
OUTPUT_DIR/sp_ratio_grids, alongside whatever sp_ratio_future_ensemble_maps.py
already writes there, for the "identify largest S/P shifts" step.

Reuses the CDO-cached yearly sums (cdo_reduce.py via climatology.py), so
this is fast if you've already run run_climatology_pipeline.py or
run_pipeline.py -- otherwise it will trigger the same reduction/caching
those scripts do.

NOTE on the "winter" window: this script passes window="winter" straight
through to climatology.multi_model_climatology() and cdo_reduce's caching
layer, the same way it already passes "water_year"/"cold_season". Whether
that actually works depends on whether those two modules already have a
Dec-Feb case wired in -- if "winter" isn't recognized there yet, this
script will need climatology.py/cdo_reduce.py updated too (pending seeing
those files).

Run with: python figures_sp_ratio_overview.py
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd

import config
import io_utils
import climatology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOWS = ["water_year", "cold_season", "winter"]
WINDOW_LABELS = {
    "water_year": "water year (Oct-Sep)",
    "cold_season": "cold season (Nov-Apr)",
    "winter": "winter (Dec-Feb)",
}

GRIDS_DIR = config.OUTPUT_DIR / "sp_ratio_grids"
GRIDS_DIR.mkdir(parents=True, exist_ok=True)


def _load_ne_state_outlines():
    """Individual (non-dissolved) Northeast state boundaries, for drawing
    state outlines on top of the S/P ratio maps."""
    states = gpd.read_file(config.STATES_SHAPEFILE)
    ne_states = states[states[config.STATE_ID_FIELD].isin(config.NE_STATES)].to_crs("EPSG:4326")
    return ne_states


# Purple gradient: dark purple = low S/P ratio (more liquid precip),
# light purple = high S/P ratio (more snow). "Purples" is a built-in
# sequential colormap that runs light(low)->dark(high); reversing it
# ("Purples_r") gives exactly dark(low)->light(high).
SP_RATIO_CMAP = "Purples_r"


def _draw_sp_ratio(ax, da, ne_borders, vmin=0, vmax=1, cmap=SP_RATIO_CMAP, title=""):
    mesh = da.plot.pcolormesh(ax=ax, x="lon", y="lat", cmap=cmap, vmin=vmin, vmax=vmax,
                                add_colorbar=False)
    try:
        ne_borders.boundary.plot(ax=ax, linewidth=0.4, color="black", alpha=0.6)
    except Exception:
        pass
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    return mesh


def _save_netcdf(da, out_path):
    da = da.copy()
    da.name = "sp_ratio"
    da.to_netcdf(out_path)
    log.info(f"  Saved grid: {out_path}")


def build_historical_figure(ne_borders, window):
    log.info(f"Building historical S/P ratio figure ({window})...")
    gcm_runs = io_utils.discover_model_runs(config.LOCA2WBM_HIST_DIR)
    log.info(f"  {len(gcm_runs)} historical model run(s): {sorted(gcm_runs)}")

    per_model, ensemble = climatology.multi_model_climatology(
        gcm_runs, config.CLIMATOLOGY_PERIODS["historical"], window=window
    )
    log.info(f"  Ensemble built from {len(per_model)} model(s)")

    sp_da = ensemble["sp_ratio_clim"]
    _save_netcdf(sp_da, GRIDS_DIR / f"sp_ratio_historical_{window}.nc")

    # data-driven vmax, fixed from here on for every figure in this window
    # (see module docstring for why this isn't auto-scaled per figure)
    valid_vals = sp_da.values[~np.isnan(sp_da.values)]
    vmax = float(np.nanpercentile(valid_vals, 98)) if valid_vals.size else 1.0
    log.info(f"  Color scale for {window}: 0.00 - {vmax:.2f} (historical 98th percentile)")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = _draw_sp_ratio(ax, sp_da, ne_borders, vmin=0, vmax=vmax,
                           title="Historical (1980-2013)")
    fig.suptitle(
        f"Northeast US Snow-to-Precipitation Ratio \u2014 Historical Baseline\n"
        f"({WINDOW_LABELS[window]}, ensemble mean of {len(per_model)} models)",
        fontsize=13, fontweight="bold")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, label="S/P ratio")
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    out_path = config.PLOTS_DIR / f"sp_ratio_historical_{window}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info(f"  Saved {out_path}")

    return vmax


def build_future_period_figure(period_name, period_label, ne_borders, window, vmax):
    """One figure per future period (mid-century or late-century), one row
    of panels -- one per SSP scenario -- sharing the vmax passed in from
    the historical figure for this window, so all three figures (historical
    + both future periods) use the identical color scale."""
    log.info(f"Building future S/P ratio figure: {period_name} / {window} ...")
    scenarios = config.SCENARIOS  # ["ssp245", "ssp370", "ssp585"]

    panels = {}
    for scenario in scenarios:
        log.info(f"  {period_name} / {scenario} / {window} ...")
        gcm_runs = io_utils.discover_model_runs(config.LOCA2WBM_FUT_DIR, scenario=scenario)
        if not gcm_runs:
            log.warning(f"    No model runs found for {scenario}; leaving panel blank")
            panels[scenario] = None
            continue
        per_model, ensemble = climatology.multi_model_climatology(
            gcm_runs, config.CLIMATOLOGY_PERIODS[period_name], window=window
        )
        sp_da = ensemble["sp_ratio_clim"]
        _save_netcdf(sp_da, GRIDS_DIR / f"sp_ratio_{scenario}_{period_name}_{window}.nc")
        panels[scenario] = (sp_da, len(per_model))

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.5 * len(scenarios), 5),
                              constrained_layout=True)
    axes = np.atleast_1d(axes)

    mesh = None
    for col, scenario in enumerate(scenarios):
        ax = axes[col]
        entry = panels[scenario]
        scen_label = config.SCENARIO_LABELS.get(scenario, scenario)

        if entry is None:
            ax.axis("off")
            ax.set_title(f"{scen_label}\n(no data)", fontsize=11)
            continue

        da, n_models = entry
        mesh = _draw_sp_ratio(ax, da, ne_borders, vmin=0, vmax=vmax, title=scen_label)

    fig.suptitle(
        f"Northeast US Snow-to-Precipitation Ratio \u2014 {period_label}\n"
        f"({WINDOW_LABELS[window]}, ensemble mean per scenario)",
        fontsize=14, fontweight="bold")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.8, label="S/P ratio")
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    out_path = config.PLOTS_DIR / f"sp_ratio_{period_name}_{window}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info(f"Saved {out_path}")


def main():
    ne_borders = _load_ne_state_outlines()
    for window in WINDOWS:
        log.info(f"\n{'=' * 60}\nWINDOW: {WINDOW_LABELS[window]}\n{'=' * 60}")
        vmax = build_historical_figure(ne_borders, window)
        build_future_period_figure("mid_century", "Mid-Century (2040-2070)", ne_borders, window, vmax)
        build_future_period_figure("late_century", "Late-Century (2070-2100)", ne_borders, window, vmax)
    log.info(f"Done. Figures written to {config.PLOTS_DIR}, grids written to {GRIDS_DIR}")


if __name__ == "__main__":
    main()
