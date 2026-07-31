# Northeast US Rain/Snow Partitioning & Spring Runoff Pipeline

Analyzes how the balance between snowfall and rainfall is shifting across
the Northeast US, historically and under CMIP6 SSP scenarios (LOCA2
downscaled), and what that means for the timing and magnitude of
snowmelt-driven spring runoff at HUC8 watershed scale.

## Two pipelines in this project

1. **`run_pipeline.py`** — HUC8-scale Mann-Kendall trend screening (which
   watersheds show a significant S/P decline, historically and in the
   future; spring runoff timing/magnitude shift).
2. **`run_climatology_pipeline.py`** — grid-cell climatology maps (total
   snowfall, total precipitation, S/P ratio) for the historical baseline
   and future mid-/late-century periods, per SSP, per model and ensemble.

Both share the same CDO-cached yearly reduction layer (`cdo_reduce.py`),
so running one warms the cache for the other.

## Why CDO caching (`cdo_reduce.py`)

The original approach summed daily precip/snowfall into water-year
totals inside xarray/dask. That computation is *lazy* — nothing actually
runs until something downstream forces it, which turned out to be
xagg's aggregation step. So every pipeline run was silently re-reading
and re-summing the entire multi-decade daily record right at
`aggregating sp_ratio...`, with no progress feedback, which is why it
looked stuck.

`cdo_reduce.py` shells out to the `cdo` command-line tool (must be on
`PATH` — e.g. `conda install -c conda-forge cdo` or `apt install cdo`)
to do the same daily→annual reduction in compiled C, and caches the
result as small netCDFs (one value per grid cell per year) under
`config.CDO_CACHE_DIR`. Every reduction function checks file
modification times and skips recomputation if the cache is already up
to date — so after the first full run, subsequent runs only reduce
GCM runs that are new or have changed on disk. Delete `CDO_CACHE_DIR`
to force a full rebuild.

`run_pipeline.py`'s melt aggregation (for spring runoff timing, which
needs daily resolution and can't be reduced via CDO) is also cut down
to just the Mar–Jul spring window before HUC8 aggregation, since that's
the only part of the year `runoff_metrics.py` uses.

## Climatology pipeline (`climatology.py`, `plotting.py`, `run_climatology_pipeline.py`)

Implements the analysis-meeting requests:

- **Cold season**: Oct–Apr by default (`config.COLD_SEASON_MONTHS`);
  change to Nov–Apr to match Burakowski et al. exactly.
- **Periods**: historical baseline (1980–2013), mid-century (2040–2070),
  late-century (2070–2100) — all in `config.CLIMATOLOGY_PERIODS`.
- **Order of operations**: sum snow/precip per water-year or cold-season
  first (CDO, daily resolution) → average those yearly sums within a
  period to get the climatology → ratio = mean(snow sum) / mean(precip
  sum). This is *not* the same as averaging yearly ratios, per the
  meeting notes.
- **Model averaging happens last**: `climatology.multi_model_climatology()`
  computes and keeps every model's climatology individually, and only
  produces the ensemble mean as a final step. Both are plotted.
- **Three map types per period/window**: total snowfall, total
  precipitation, S/P ratio — for the ensemble and every individual
  model, plus a multi-model comparison grid for each variable.

Run with `python run_climatology_pipeline.py`; PNGs land in
`config.PLOTS_DIR` (`outputs/plots/`).

## Install

```bash
pip install -r requirements.txt
```

## Before running: edit `config.py`

You must point the paths in `config.py` at your actual data:

| Variable | What it should contain |
|---|---|
| `LOCA2_DIR` | LOCA2-downscaled CMIP6 daily tasmax/tasmin/pr, organized `<GCM>/<scenario>/<var>.*.nc` |
| `LIVNEH_DIR` | Livneh daily tmax/tmin/pr NetCDFs (historical baseline, 1980-2013) |
| `SNODAS_DIR` | SNODAS daily SWE + snowMelt NetCDFs (historical, used for QA / validation) |
| `HUC8_SHAPEFILE` | NHD/WBD HUC8 polygons, clipped or filtered to the Northeast |
| `GCM_LIST` | Which LOCA2 GCMs you have on disk; loop runs over whatever's present |

Also confirm/adjust `REGION_BBOX`, `SCENARIOS`, and the phase-partition
thresholds `T_ALL_SNOW`/`T_ALL_RAIN` (default -1°C / +1°C linear ramp, as
specified in the analysis plan).

## Run

```bash
python run_pipeline.py
```

## What it does, stage by stage

1. **`io_utils.py`** — loads and standardizes Livneh (historical) and
   LOCA2 (future, per GCM/SSP) daily temperature + precipitation, and
   SNODAS SWE/melt, all clipped to the NE bounding box and unit-normalized
   (temp → °C, precip → mm/day).

2. **`phase_partition.py`** — implements the -1°C/+1°C linear-ramp
   rain/snow split exactly as specified: 100% snow ≤ -1°C, 100% rain ≥
   +1°C, linear interpolation between. Produces daily `snow`/`rain` grids
   and the water-year (Oct–Sep) **S/P ratio** (snow-water-equivalent
   precip ÷ total precip) per grid cell.

3. **`snowpack_model.py`** — a simple degree-day (temperature-index)
   snowpack model that accumulates the partitioned snowfall and melts it
   based on daily mean temperature. This is what the future-period melt
   flux is built from, since LOCA2/CMIP6 doesn't provide a physical
   snowmelt output variable — only downscaled T/P, per the analysis plan.
   For the historical period, the *observed* SNODAS `snowMelt` variable
   is used as the primary source, with the degree-day model run in
   parallel as a QA cross-check (`huc8_snodas_observed_melt_hist.csv` vs
   `huc8_daily_melt_sim_hist.csv`).

4. **`huc8_zonal.py`** — area-weighted aggregation of gridded output to
   HUC8 polygons using `xagg`, which precomputes pixel/polygon overlap
   fractions once and reuses them across every day/year/variable (much
   faster than per-timestep zonal stats over 30–120 years of daily data).

5. **`trends.py`** — Mann-Kendall trend test + Theil-Sen slope on each
   HUC8's water-year S/P ratio series, run separately on the historical
   record and on each future SSP. Flags:
   - `significant_decline`: MK-significant (α=0.05) negative trend
   - `declining_both_periods`: significant decline in *both* the
     historical record and the given future scenario — this is the
     primary watershed-screening output requested in the analysis plan
   - `reaches_near_zero`: minimum S/P ratio in the series falls below
     `NEAR_ZERO_SP_THRESHOLD` (default 0.05), i.e. watersheds trending
     toward an effectively snow-free precipitation regime

6. **`runoff_metrics.py`** — per HUC8/water-year:
   - **center of timing (CT)**: day-of-water-year by which 50% of the
     Mar–Jul cumulative snowmelt has occurred (earlier CT = earlier
     spring pulse, the classic "earlier peak flow" signal)
   - **spring melt total**: Mar–Jul cumulative melt (mm), a magnitude
     proxy for the spring runoff pulse
   - **peak melt rate**: maximum daily melt rate and the day it occurs
   - `summarize_shift()` compares a historical baseline window
     (default: last 20 years of record) against an end-of-century window
     (default: 2081–2100) per HUC8/scenario

## Outputs (in `config.OUTPUT_DIR`)

- `huc8_sp_ratio_{hist,ssp245,ssp370,ssp585}.csv` — annual HUC8 S/P ratio
- `huc8_sp_trend_{hist,ssp245,...}.csv` — per-HUC8 MK trend stats
- `huc8_sp_trend_combined_{scenario}.csv` — historical+future trend joined
- `huc8_declining_sp_both_periods.csv` — **watersheds with significant
  S/P decline historically AND in a future scenario** (the main screening
  output requested)
- `huc8_near_zero_sp_ratio.csv` — watersheds approaching/reaching S/P ≈ 0
- `huc8_spring_runoff_metrics_{hist,scenario}.csv` — CT/magnitude by year
- `huc8_runoff_shift_{scenario}.csv` — historical vs. end-of-century shift
  in timing (days earlier/later) and magnitude (% change)
- `huc8_trend_results.geojson` — HUC8 polygons with trend/decline flags
  joined on, ready to map in QGIS/geopandas/kepler.gl

## Key assumptions to revisit

- Degree-day factor (3.0 mm/°C/day) and melt-onset threshold (0°C) are
  reasonable NE-US defaults but not calibrated to any specific basin —
  calibrate against SNODAS historically per-HUC8 if better accuracy is
  needed than this screening-level analysis provides.
- Rain-on-snow contributions to melt are not modeled explicitly; melt is
  driven by temperature alone in the degree-day model.
- Multi-GCM ensemble mean is used for future trend/runoff results;
  inspect per-GCM spread (`gcm` column retained pre-aggregation) before
  treating the ensemble mean as anything more than a central estimate —
  spread across LOCA2 GCMs is often as large as the SSP-to-SSP spread for
  this region.
