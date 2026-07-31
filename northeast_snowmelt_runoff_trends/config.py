"""
config.py
=========
Central configuration for the Northeast US rain/snow-partitioning and
spring-runoff analysis pipeline.

EDIT THE PATHS BELOW to point at your actual data holdings. Nothing in this
file does I/O itself -- it's just shared constants imported by every module.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project directory -- this file lives at the project root
# (.../northeast_snowmelt_runoff_trends/config.py), so PROJECT_ROOT resolves
# correctly regardless of where the project folder itself is checked out.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Study region: actual Northeast state boundaries (not a bounding box).
# Dissolved from a US states shapefile down to just the NE states below,
# same approach/state list used in the Taylor-diagram validation script.
# ---------------------------------------------------------------------------
STATES_SHAPEFILE = Path("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp").expanduser()

# postal codes of the states included in "Northeast US" for this analysis
NE_STATES = ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "WV"]

# field in the shapefile holding the 2-letter state postal abbreviation
STATE_ID_FIELD = "STUSPS"

# ---------------------------------------------------------------------------
# Time periods
# ---------------------------------------------------------------------------
HIST_START, HIST_END = "1980-01-01", "2013-12-31"
FUT_START, FUT_END = "2015-01-01", "2100-12-31"

# Water year runs Oct 1 -> Sep 30; "spring" window used for runoff metrics
SPRING_MONTHS = (3, 4, 5, 6, 7)  # Mar-Jul, captures snowmelt-driven runoff

# ---------------------------------------------------------------------------
# Scenarios
# Confirmed folder naming, flat under each period directory (no separate
# per-scenario subdirectory) -- the scenario/period is the SECOND
# underscore-separated field in the GCM-run folder name:
#   {GCM}_{period_or_scenario}_{years}_newprcp
# e.g. ACCESS-CM2_historical_1980-2014_newprcp
#      NorESM2-MM_ssp585_2015-2100_newprcp
# All GCM x SSP combinations for the future period live flat together
# under LOCA2WBM_FUT_DIR; io_utils.discover_model_runs filters on that
# second field when a scenario is requested.
# ---------------------------------------------------------------------------
SCENARIOS = ["ssp245", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

# GCM runs are auto-discovered from LOCA2WBM_HIST_DIR / LOCA2WBM_FUT_DIR
# via MODEL_FOLDER_GLOB (see io_utils.discover_model_runs). Leave GCM_LIST
# as None to use every GCM found on disk, or set it to a list of GCM
# names (matching the prefix before the first "_" in the folder name,
# e.g. "ACCESS-CM2") to restrict the analysis to a subset.
GCM_LIST = None

# ---------------------------------------------------------------------------
# Rain/snow phase-partitioning thresholds (deg C), linear ramp between them
# ---------------------------------------------------------------------------
T_ALL_SNOW = -1.0   # at/below this, precip is 100% snow
T_ALL_RAIN = 1.0    # at/above this, precip is 100% rain

# ---------------------------------------------------------------------------
# Trend-test / significance settings
# ---------------------------------------------------------------------------
MK_ALPHA = 0.05          # significance threshold for Mann-Kendall test
NEAR_ZERO_SP_THRESHOLD = 0.05   # S/P ratio below this counted as "~snow-free"

# ---------------------------------------------------------------------------
# Data paths -- EDIT THESE (SNODAS is optional -- see note below)
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/data/climate")  # placeholder root, only used for SNODAS_DIR below

# LOCA2-WBM model output (this is the real analysis workhorse -- both the
# "historical" and "future" branches of the pipeline are WBM model runs,
# not raw LOCA2 tasmax/tasmin/pr). Layout, per period directory:
#   {period_dir}/{GCM}_{period}_..._newprcp/daily/wbm_{year}.nc
# Each wbm_{year}.nc is one calendar year of DAILY data and already
# contains all variables bundled together (temps, precip, and -- per the
# WBM's own phase partitioning -- snow/rain/snowmelt/SWE), so loading is
# "glob one file per year" rather than "one file-glob per variable".
LOCA2WBM_ROOT = Path("/net/nfs/echo/ankaa/LOCA2-WBM_output")
LOCA2WBM_HIST_DIR = LOCA2WBM_ROOT / "LOCA2-WBM_historical"
LOCA2WBM_FUT_DIR = LOCA2WBM_ROOT / "LOCA2-WBM_future"

# GCM-run subfolder naming pattern under each period directory, e.g.
# "ACCESS-CM2_historical_1980-2014_newprcp". Matched with glob, same
# approach as the Taylor-diagram validation script, so exact date ranges
# in the folder name don't need to be hardcoded here.
MODEL_FOLDER_GLOB = "*_newprcp"
DAILY_SUBDIR = "daily"
DAILY_FILE_GLOB = "wbm_*.nc"   # one file per year, e.g. wbm_1980.nc ... wbm_2013.nc

# Variable names as they appear INSIDE each wbm_{year}.nc file, confirmed
# from the file's actual header (ncdump -h):
VAR_TAS = "airT"              # mean air temp, degC -- output directly, no need to average max/min
VAR_TASMAX = "airTmax"        # degC
VAR_TASMIN = "airTmin"        # degC
VAR_PRECIP_TOTAL = "precip"   # mm/day
VAR_SNOW = "snowFall"         # mm/day -- this is the model's snowfall flux
VAR_SWE = "snowPack"          # mm -- snow pack water equivalent (state variable)
VAR_SWE_CHANGE = "snowPackChg"  # mm/day -- day-over-day change in snowPack
VAR_SNOWMELT = "snowMelt"     # mm/day
# There is no separate "rain" variable in the WBM output. Rain is derived
# as precip - snowFall (both mm/day) -- see io_utils.load_wbm_daily.

# Livneh observational data is only available here at MONTHLY resolution,
# organized one subfolder per variable (matching the Taylor-diagram
# script): {LIVNEH_MONTHLY_DIR}/{var}/wbm_*.nc. There's no confirmed daily
# Livneh source, so this is used for MODEL VALIDATION (Taylor diagrams)
# only -- it is NOT the source for this pipeline's historical S/P-ratio
# or snowmelt analysis, which instead runs on LOCA2WBM_HIST_DIR above.
LIVNEH_MONTHLY_DIR = Path(
    "/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly"
)

HUC8_SHAPEFILE = Path("~/LOCA2-WBM_code/data/WBD/HUC8_northeast.shp").expanduser()

# SNODAS is NOT required by the current pipeline -- the LOCA2-WBM historical
# run already provides snowFall/precip/snowMelt directly, so there's no gap
# to fill. This path only matters if you want an independent-observation
# QA check on the WBM's own snow simulation (optional; see chat for specifics
# on what to pull if you want to add this).
SNODAS_DIR = DATA_ROOT / "SNODAS"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HUC8 attribute field name in the shapefile holding the 8-digit code
HUC8_ID_FIELD = "huc8"
HUC8_NAME_FIELD = "name"

# ---------------------------------------------------------------------------
# Climatology periods and cold-season definition (from analysis meeting)
# ---------------------------------------------------------------------------
# Cold season: Oct-Apr (7 months), kept aligned with the water-year Oct-Sep
# start month. If you want to match Burakowski et al. exactly (Nov-Apr,
# 6 months), change this to (11, 12, 1, 2, 3, 4).
COLD_SEASON_MONTHS = (10, 11, 12, 1, 2, 3, 4)

# (start_year, end_year), inclusive. historical = full observational/model
# baseline; mid_century and late_century are the two future climatology
# windows requested.
CLIMATOLOGY_PERIODS = {
    "historical": (1980, 2013),
    "mid_century": (2040, 2070),
    "late_century": (2070, 2100),
}

# ---------------------------------------------------------------------------
# CDO cache: pre-reduced yearly (water-year / cold-season) sum files.
# The expensive daily -> annual reduction runs once per GCM run / variable
# / window and is cached here, so re-running the pipeline doesn't start
# from zero every time. Safe to delete this folder to force a full rebuild.
# ---------------------------------------------------------------------------
CDO_CACHE_DIR = PROJECT_ROOT / "cdo_cache"
CDO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
