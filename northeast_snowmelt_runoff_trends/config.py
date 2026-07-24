"""
Top 5 models for [snowMelt] by lowest CRMSD:
  1. NorESM2-MM      | CRMSE: 2.453 | Corr: 0.701
  2. ACCESS-CM2      | CRMSE: 2.494 | Corr: 0.663
  3. MPI-ESM1-2-HR   | CRMSE: 2.578 | Corr: 0.679
  4. BCC-CSM2-MR     | CRMSE: 2.590 | Corr: 0.625
  5. IPSL-CM6A-LR    | CRMSE: 2.670 | Corr: 0.640
"""

"""
Central configuration for the Northeast US rain/snow-partitioning and spring runoff analysis pipeline.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
STATES_SHAPEFILE = PROJECT_ROOT / shapefiles / "cb_2025_us_state_5m.shp"
NE_STATES = ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "WV"]
STATE_ID_FIELD = "STUSPS"

# time periods
HIST_START, HIST_END = "1980-01-01", "2013-12-31"
FUT_START, FUT_END = "2015-01-01", "2100-12-31"

# water year runs Oct 1 to Sep 30
# using Mar - Jul to understand snowmelt-driven runoff
SPRING_MONTHS = (3, 4, 5, 6, 7)

# scenarios
SCENARIOS = ["ssp245", "ssp370", "ssp585"]
SCENARIO_LABELS = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

# using 5 best gcms (determined by Taylor diagram of snowMelt variable)
GCM_LIST = [
    "ACCESS-CM2",
    "NorESM2-MM",
    "MPI-ESM1-2-HR",
    "BCC-CSM2-MR",
    "IPSL-CM6A-LR"
]

T_ALL_SNOW = -1.0   # at/below this, precip is 100% snow
T_ALL_RAIN = 1.0    # at/above this, precip is 100% rain

MK_ALPHA = 0.05          # significance threshold for Mann-Kendall test
NEAR_ZERO_SP_THRESHOLD = 0.05   # S/P ratio below this counted as "~snow-free"

DATA_ROOT = PROJECT_ROOT/ "data"

LOCA2_WBM_ROOT = Path("/net/nfs/echo/ankaa/LOCA2-WBM_output")
LOCA2_WBM_HIST_DIR = LOCA2_WBM_ROOT / "LOCA2-WBM_historical"
LOCA2_WBM_FUT_DIR = LOCA2_WBM_ROOT / "LOCA2-WBM_future"

MODEL_FOLDER_GLOB = "*_newprcp"
DAILY_SUBDIR = "daily"
DAILY_FILE_GLOB = "wbm_*.nc"

VAR_TAS = "airT"
VAR_TASMAX = "airTmax"
VAR_TASMIN = "airTmin"
VAR_PRECIP_TOTAL = "precip"
VAR_SNOW = "snowFall"
VAR_SWE = "snowPack"
VAR_SWE_CHANGE = "snowPackChg"
VAR_SNOWMELT = "snowMelt"

LIVNEH_MONTHLY_DIR = Path("/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly")

# working on this next
"""
SNODAS_DIR = DATA_ROOT / "SNODAS"
HUC8_SHAPEFILE = DATA_ROOT / "WBD" / "HUC8_northeast.shp"
"""

OUTPUT_DIR = PROJECT_ROOT / "northeast_snowmelt_runoff_trends" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

"""
HUC8_ID_FIELD = "huc8"
HUC8_NAME_FIELD = "name"
"""
