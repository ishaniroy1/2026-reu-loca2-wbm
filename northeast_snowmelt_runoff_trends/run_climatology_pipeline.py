"""
run_climatology_pipeline.py

Builds the S/P-ratio climatology maps

  * historical baseline (config.CLIMATOLOGY_PERIODS["historical"])
  * mid-century (2040-2070) and late-century (2070-2100), per SSP scenario
  * both water-year and cold-season (Oct-Apr) windows
  * total snowfall, total precipitation, and S/P ratio, each as a map
  * every individual model plotted, plus the ensemble mean (models are
    only averaged together at the very last step)
"""

import logging

import config
import io_utils
import climatology
import plotting

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WINDOWS = ["water_year", "cold_season"]


def run_for_period(period_label, period_years, base_dir, scenario=None):
    gcm_runs = io_utils.discover_model_runs(base_dir, scenario=scenario)
    log.info(f"{period_label}: found {len(gcm_runs)} model run(s): {sorted(gcm_runs)}")

    for window in WINDOWS:
        log.info(f"  Building {window} climatology ({period_label}) ...")
        try:
            per_model, ensemble = climatology.multi_model_climatology(
                gcm_runs, period_years, window=window
            )
        except RuntimeError as e:
            log.warning(f"  Skipping {period_label}/{window}: {e}")
            continue

        plotting.plot_climatology_maps(ensemble, "Ensemble", window, period_label)
        for gcm, ds in per_model.items():
            plotting.plot_climatology_maps(ds, gcm, window, period_label)
        plotting.plot_all_three_model_grids(per_model, ensemble, window, period_label)

        log.info(f"  Done: {len(per_model)} model maps + ensemble + comparison grids "
                  f"({period_label}, {window})")


def main():
    # Historical baseline (no scenario -- LOCA2-WBM historical runs)
    run_for_period("historical", config.CLIMATOLOGY_PERIODS["historical"], config.LOCA2WBM_HIST_DIR)

    # Future periods, per SSP scenario
    for scenario in config.SCENARIOS:
        label = config.SCENARIO_LABELS.get(scenario, scenario)
        for period_name in ("mid_century", "late_century"):
            run_for_period(
                f"{period_name}_{scenario}",
                config.CLIMATOLOGY_PERIODS[period_name],
                config.LOCA2WBM_FUT_DIR,
                scenario=scenario,
            )

    log.info(f"Done. Plots written to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
