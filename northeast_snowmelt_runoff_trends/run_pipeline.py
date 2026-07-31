"""
run_pipeline.py
================
End-to-end driver for the Northeast US rain/snow-partitioning and spring
snowmelt-runoff analysis, built on LOCA2-WBM daily model output.

The WBM model already partitions precipitation into snow/rain and tracks
snowmelt itself (config.VAR_SNOW / VAR_RAIN / VAR_SNOWMELT), for BOTH the
historical and future runs. Water-year S/P ratio sums are computed via
CDO (see cdo_reduce.py) and cached to small netCDFs, rather than reduced
in-Python with dask -- this was the original performance bottleneck (the
lazy dask reduction over the full daily 1980-2100 record wasn't actually
running until xagg forced it, at "aggregating sp_ratio..."). Melt for
the spring-runoff metrics is loaded directly since daily resolution is
needed, but restricted to the Mar-Jul window before aggregation.
(See phase_partition.py / snowpack_model.py if you need a from-scratch
recompute for QA/comparison against the model's own partitioning.)

Pipeline stages:
  1. Discover GCM runs under LOCA2WBM_HIST_DIR (historical) and
     LOCA2WBM_FUT_DIR (future, per SSP scenario).
  2. Compute water-year S/P ratio from CDO-cached yearly snow/precip sums
     (fast, cached -- see cdo_reduce.py), per grid cell.
  3. Aggregate S/P ratio and spring-window daily snowmelt flux to HUC8
     watersheds (area-weighted), then take the multi-GCM ensemble mean.
  4. Mann-Kendall trend test on HUC8 S/P ratio, historical and future;
     flag watersheds with significant decline in both periods, and any
     that approach a snow-free (S/P -> 0) state.
  5. Compute spring runoff timing (center-of-timing) and magnitude
     metrics per HUC8/water year, and summarize the historical ->
     future shift.

Run with:  python run_pipeline.py
Outputs land in config.OUTPUT_DIR as CSVs (and a HUC8 GeoJSON with trend
results joined on, for quick mapping in GIS/QGIS/geopandas).
"""

import logging
import pandas as pd
import xarray as xr

import config
import io_utils
import huc8_zonal
import cdo_reduce
import trends
import runoff_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def process_period_to_huc8(gcm_run_dir, weightmap=None, gdf=None):
    """
    Aggregate one GCM run to HUC8:
      - S/P ratio: from the CDO-cached water-year snow/precip sums (see
        cdo_reduce.py) -- a few dozen timesteps, not tens of thousands of
        daily records, so this is fast and doesn't force a full re-scan
        of the daily archive on every pipeline run.
      - spring melt: loaded from the daily WBM output but restricted to
        the Mar-Jul spring window (config.SPRING_MONTHS) before HUC8
        aggregation, since that's the only part of the year
        runoff_metrics.py actually uses -- cuts the aggregated volume by
        roughly 5/12 for free.
    """
    snow_path = cdo_reduce.get_or_build(gcm_run_dir, config.VAR_SNOW, window="water_year")
    precip_path = cdo_reduce.get_or_build(gcm_run_dir, config.VAR_PRECIP_TOTAL, window="water_year")

    snow = xr.open_dataset(snow_path)[config.VAR_SNOW]
    precip = xr.open_dataset(precip_path)[config.VAR_PRECIP_TOTAL]
    sp_ratio = (snow / precip.where(precip > 0)).rename("sp_ratio")
    sp_ratio_ds = io_utils.clip_to_region(sp_ratio.to_dataset())

    # CDO's yearsum output is timestamped (post-shift); relabel that time
    # coordinate as a plain integer water year, matching trends.py's
    # expected column.
    sp_ratio_ds = sp_ratio_ds.rename({"time": "water_year"})
    sp_ratio_ds["water_year"] = sp_ratio_ds["water_year"].dt.year

    if gdf is None:
        gdf = huc8_zonal.load_huc8()
    if weightmap is None:
        weightmap = huc8_zonal.build_overlap_weights(sp_ratio_ds, gdf)

    sp_df = huc8_zonal.aggregate_to_huc8(sp_ratio_ds, weightmap, gdf, ["sp_ratio"])

    melt_ds = io_utils.load_wbm_daily(gcm_run_dir, variables=[config.VAR_SNOWMELT])
    melt_ds = melt_ds.sel(time=melt_ds["time"].dt.month.isin(config.SPRING_MONTHS))
    melt_df = huc8_zonal.aggregate_to_huc8(
        melt_ds[[config.VAR_SNOWMELT]], weightmap, gdf, [config.VAR_SNOWMELT]
    ).rename(columns={config.VAR_SNOWMELT: "melt"})

    return sp_df, melt_df, weightmap, gdf


def _run_period(base_dir, period, scenario=None, weightmap=None, gdf=None):
    """
    Shared driver for both historical and future: discover every GCM run
    under base_dir, aggregate each to HUC8, and return the multi-GCM
    ensemble mean S/P ratio and melt flux.
    """
    gcm_runs = io_utils.discover_model_runs(base_dir, scenario=scenario)
    log.info(f"Found {len(gcm_runs)} {period} GCM run(s)"
              + (f" for {scenario}" if scenario else "") + f": {sorted(gcm_runs)}")

    all_sp, all_melt = [], []
    for gcm, run_dir in gcm_runs.items():
        log.info(f"Processing LOCA2-WBM {period} output for {gcm}"
                  + (f"/{scenario}" if scenario else "") + " ...")
        try:
            sp_df, melt_df, weightmap, gdf = process_period_to_huc8(
                run_dir, weightmap=weightmap, gdf=gdf
            )
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            log.warning(f"Skipping {gcm} ({period}{'/' + scenario if scenario else ''}): {e}")
            continue

        sp_df["gcm"] = gcm
        melt_df["gcm"] = gcm
        all_sp.append(sp_df)
        all_melt.append(melt_df)

    if not all_sp:
        log.warning(f"No usable GCM runs for {period}"
                     + (f"/{scenario}" if scenario else "") + "; skipping.")
        return None

    sp_df = pd.concat(all_sp, ignore_index=True)
    melt_df = pd.concat(all_melt, ignore_index=True)

    # Multi-GCM ensemble mean per HUC8/timestep before trend testing, to
    # characterize the multi-model-consensus signal rather than any one
    # model's internal variability.
    sp_ensemble = sp_df.groupby(["huc8", "name", "water_year"], as_index=False)["sp_ratio"].mean()
    melt_ensemble = melt_df.groupby(["huc8", "name", "time"], as_index=False)["melt"].mean()

    return dict(sp_df=sp_ensemble, melt_df=melt_ensemble, weightmap=weightmap, gdf=gdf)


def run_historical():
    result = _run_period(config.LOCA2WBM_HIST_DIR, "historical")
    if result is None:
        raise RuntimeError("No historical GCM runs produced valid output; check config.py paths.")

    result["sp_df"].to_csv(config.OUTPUT_DIR / "huc8_sp_ratio_hist.csv", index=False)
    result["melt_df"].to_csv(config.OUTPUT_DIR / "huc8_daily_melt_hist.csv", index=False)

    result["trend"] = trends.huc8_sp_trends(result["sp_df"])
    result["trend"].to_csv(config.OUTPUT_DIR / "huc8_sp_trend_hist.csv", index=False)

    result["runoff"] = runoff_metrics.compute_spring_runoff_metrics(result["melt_df"])
    result["runoff"].to_csv(config.OUTPUT_DIR / "huc8_spring_runoff_metrics_hist.csv", index=False)

    return result


def run_future(scenario, weightmap, gdf):
    result = _run_period(config.LOCA2WBM_FUT_DIR, "future", scenario=scenario,
                          weightmap=weightmap, gdf=gdf)
    if result is None:
        return None

    result["sp_df"].to_csv(config.OUTPUT_DIR / f"huc8_sp_ratio_{scenario}.csv", index=False)
    result["melt_df"].to_csv(config.OUTPUT_DIR / f"huc8_daily_melt_{scenario}.csv", index=False)

    result["trend"] = trends.huc8_sp_trends(result["sp_df"])
    result["trend"].to_csv(config.OUTPUT_DIR / f"huc8_sp_trend_{scenario}.csv", index=False)

    result["runoff"] = runoff_metrics.compute_spring_runoff_metrics(result["melt_df"])
    result["runoff"].to_csv(config.OUTPUT_DIR / f"huc8_spring_runoff_metrics_{scenario}.csv", index=False)

    return result


def main():
    hist = run_historical()

    combined_trends = []

    for scenario in config.SCENARIOS:
        fut = run_future(scenario, hist["weightmap"], hist["gdf"])
        if fut is None:
            continue

        combined = trends.combine_hist_future_trends(hist["trend"], fut["trend"], scenario)
        combined.to_csv(config.OUTPUT_DIR / f"huc8_sp_trend_combined_{scenario}.csv", index=False)
        combined_trends.append(combined)

        # Historical baseline: last 20 yrs of record (1994-2013 water years)
        # Future endpoint: end-of-century (2081-2100 water years)
        shift = runoff_metrics.summarize_shift(
            pd.concat([hist["runoff"], fut["runoff"]], ignore_index=True),
            period_a_years=(1994, 2013),
            period_b_years=(2081, 2100),
        )
        shift["scenario"] = scenario
        shift.to_csv(config.OUTPUT_DIR / f"huc8_runoff_shift_{scenario}.csv", index=False)

    if combined_trends:
        all_trends = pd.concat(combined_trends, ignore_index=True)
        declining = all_trends[all_trends["declining_both_periods"]]
        declining.to_csv(config.OUTPUT_DIR / "huc8_declining_sp_both_periods.csv", index=False)
        log.info(f"{declining['huc8'].nunique()} HUC8 watersheds show a significant "
                  f"S/P decline in BOTH the historical record and at least one SSP scenario.")

        near_zero_cols = [c for c in all_trends.columns if c.startswith("reaches_near_zero_")]
        near_zero = all_trends[all_trends[near_zero_cols].any(axis=1)]
        near_zero.to_csv(config.OUTPUT_DIR / "huc8_near_zero_sp_ratio.csv", index=False)

        # Join trend/decline flags onto HUC8 geometries for mapping. Each
        # scenario gets its own set of columns so a single HUC8 polygon
        # carries results for all SSPs at once.
        gdf_out = hist["gdf"].copy()
        for scenario in config.SCENARIOS:
            scen_trends = all_trends[all_trends["scenario"] == scenario][
                ["huc8", "slope_per_decade_hist", f"slope_per_decade_{scenario}",
                 "declining_both_periods"]
            ].rename(columns={
                "slope_per_decade_hist": "sp_slope_per_decade_hist",
                f"slope_per_decade_{scenario}": f"sp_slope_per_decade_{scenario}",
                "declining_both_periods": f"declining_both_{scenario}",
            })
            gdf_out = gdf_out.merge(scen_trends, on="huc8", how="left")
        gdf_out.to_file(config.OUTPUT_DIR / "huc8_trend_results.geojson", driver="GeoJSON")

    log.info(f"Done. Outputs written to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
