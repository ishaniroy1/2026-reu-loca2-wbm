"""
figures_sp_ratio_overview.py
=============================
Two summary figures for the Northeast snowmelt/runoff trends project,
both ensemble-mean S/P ratio (averaged across all available GCMs --
see climatology.multi_model_climatology, which keeps every model's
climatology individually and only averages across models as the final
step):

  1. Historical baseline S/P ratio map (single panel), water-year window.
  2. A 2 (mid-century / late-century) x 3 (SSP2-4.5 / SSP3-7.0 / SSP5-8.5)
     grid of S/P ratio maps, sharing one color scale across all six
     panels so the scenarios/periods are directly comparable.

Reuses the CDO-cached yearly sums (cdo_reduce.py via climatology.py), so
this is fast if you've already run run_climatology_pipeline.py or
run_pipeline.py -- otherwise it will trigger the same reduction/caching
those scripts do.

Run with: python figures_sp_ratio_overview.py
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

import config
import io_utils
import climatology

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOW = "cold_season"

def _load_ne_state_outlines():
    """Individual (non-dissolved) Northeast state boundaries, for drawing
    state outlines on top of the S/P ratio maps."""
    states = gpd.read_file(config.STATES_SHAPEFILE)
    ne_states = states[states[config.STATE_ID_FIELD].isin(config.NE_STATES)].to_crs("EPSG:4326")
    return ne_states


def _draw_sp_ratio(ax, da, ne_borders, vmin=0, vmax=1, cmap="coolwarm", title=""):
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


def build_historical_figure(ne_borders):
    log.info("Building historical S/P ratio figure...")
    gcm_runs = io_utils.discover_model_runs(config.LOCA2WBM_HIST_DIR)
    log.info(f"  {len(gcm_runs)} historical model run(s): {sorted(gcm_runs)}")

    per_model, ensemble = climatology.multi_model_climatology(
        gcm_runs, config.CLIMATOLOGY_PERIODS["historical"], window=WINDOW
    )
    log.info(f"  Ensemble built from {len(per_model)} model(s)")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = _draw_sp_ratio(ax, ensemble["sp_ratio_clim"], ne_borders,
                           title="Historical (1980-2013)")
    fig.suptitle(
        f"Northeast US Snow-to-Precipitation Ratio \u2014 Historical Baseline\n"
        f"({WINDOW.replace('_', ' ')}, ensemble mean of {len(per_model)} models)",
        fontsize=13, fontweight="bold")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, label="S/P ratio")

    out_path = config.PLOTS_DIR / f"sp_ratio_historical_{WINDOW}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info(f"  Saved {out_path}")


def build_future_grid_figure(ne_borders):
    log.info("Building future S/P ratio grid figure (mid/late-century x SSP)...")

    periods = [("mid_century", "Mid-century\n(2040-2070)"),
               ("late_century", "Late-century\n(2070-2100)")]
    scenarios = config.SCENARIOS  # ["ssp245", "ssp370", "ssp585"]

    # compute every panel first so the color scale can be shared and
    # consistent across all six, and so a slow step doesn't happen mid-plot
    panels = {}
    for period_name, _ in periods:
        for scenario in scenarios:
            log.info(f"  {period_name} / {scenario} ...")
            gcm_runs = io_utils.discover_model_runs(config.LOCA2WBM_FUT_DIR, scenario=scenario)
            if not gcm_runs:
                log.warning(f"    No model runs found for {scenario}; leaving panel blank")
                panels[(period_name, scenario)] = None
                continue
            per_model, ensemble = climatology.multi_model_climatology(
                gcm_runs, config.CLIMATOLOGY_PERIODS[period_name], window=WINDOW
            )
            panels[(period_name, scenario)] = (ensemble["sp_ratio_clim"], len(per_model))

    fig, axes = plt.subplots(len(periods), len(scenarios),
                              figsize=(5.5 * len(scenarios), 5 * len(periods)),
                              constrained_layout=True)
    axes = np.atleast_2d(axes)

    mesh = None
    for row, (period_name, period_label) in enumerate(periods):
        for col, scenario in enumerate(scenarios):
            ax = axes[row, col]
            entry = panels[(period_name, scenario)]
            scen_label = config.SCENARIO_LABELS.get(scenario, scenario)

            if entry is None:
                ax.axis("off")
                ax.set_title(f"{scen_label}\n(no data)", fontsize=11)
                continue

            da, n_models = entry
            title = f"{scen_label}" if row == 0 else ""
            mesh = _draw_sp_ratio(ax, da, ne_borders, title=title)

            if col == 0:
                ax.set_ylabel(period_label, fontsize=11, fontweight="bold")
                ax.yaxis.set_label_coords(-0.08, 0.5)

    fig.suptitle(
        f"Northeast US Snow-to-Precipitation Ratio \u2014 Future Projections\n"
        f"({WINDOW.replace('_', ' ')}, ensemble mean per scenario/period)",
        fontsize=14, fontweight="bold")

    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.7, label="S/P ratio")

    out_path = config.PLOTS_DIR / f"sp_ratio_future_grid_{WINDOW}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    log.info(f"Saved {out_path}")


def main():
    ne_borders = _load_ne_state_outlines()
    build_historical_figure(ne_borders)
    build_future_grid_figure(ne_borders)
    log.info(f"Done. Figures written to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
