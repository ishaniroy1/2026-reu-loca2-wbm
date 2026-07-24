import os
import sys

proj_path = "/net/home/wsag/ir1101/miniconda3/share/proj"
os.environ["PROJ_DATA"] = proj_path
os.environ["PROJ_LIB"] = proj_path

import pyproj
pyproj.datadir.set_data_dir(proj_path)

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skill_metrics as sm
import geopandas as gpd
import rioxarray

# filepaths
LIVNEH_BASE_DIR = "/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly"
MODEL_BASE_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

# output location for taylor diagrams
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/taylor_diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# state boundary shapefile used to clip datasets to CONUS
SHAPEFILE_PATH = os.path.expanduser("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

# northeast states
nca_ne_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'WV']

# variable of interest: snowMelt is a monthly-accumulated flux (mm/month)
# on disk, so annual aggregation is a sum, same convention as precip.
VAR = "snowMelt"
OPERATION = "sum"

# how many top-performing models (by lowest CRMSD) to report at the end
N_BEST_MODELS = 5

model_folders = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split('_')[0] for f in model_folders]

# colormap and symbols for plotting
cmap = plt.get_cmap('tab20')
symbols = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', 'p', '8', 'd', 'H']

MODEL_STYLE = {
    m: {
        'faceColor': 'w',
        'edgeColor': cmap(i % 20),
        'symbol': symbols[i % len(symbols)],
        'size': 9,
        'labelColor': 'black',
    }
    for i, m in enumerate(model_names)
}


def load_northeast_boundary(shapefile_path=SHAPEFILE_PATH):
    """Load the state shapefile, keep only the northeast states, and
    dissolve them into a single boundary in EPSG:4326 (plain lat/lon)."""
    states = gpd.read_file(shapefile_path)
    print("Columns in shapefile:", states.columns)
    print("Sample rows:\n", states[['STUSPS', 'NAME', 'STATEFP']].head())

    northeast_states = states[states['STUSPS'].isin(nca_ne_states)]
    northeast_states = northeast_states.to_crs("EPSG:4326")

    print(f"Number of features matched: {len(northeast_states)}")
    print(f"Shapefile bounds: {northeast_states.total_bounds}")
    return northeast_states.dissolve().geometry


def clip_to_northeast(da_or_ds, northeast_geom, lat_name="lat", lon_name="lon", all_touched=True):
    """Clip an xarray DataArray/Dataset to the Northeast boundary. Assumes lon is
    already in -180..180 convention and lat/lon are 1-D coordinates."""
    obj = da_or_ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    return obj.rio.clip(northeast_geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def aggregate_yearly(ds, operation):
    """Slice to the 1980-2014 period and resample to annual resolution.
    snowMelt is already a monthly accumulation (mm/month) on disk,
    so a simple annual sum is mathematically correct."""
    ds_sliced = ds.sel(time=slice('1980-01-01', '2014-12-31'))

    if operation == "mean":
        return ds_sliced.resample(time='YS').mean(dim='time')
    elif operation == "sum":
        return ds_sliced.resample(time='YS').sum(dim='time')
    else:
        raise ValueError(f"Unknown aggregation operation: {operation}")


print(f"Loading Northeast boundary from {SHAPEFILE_PATH}")
northeast_geom = load_northeast_boundary()

# check how many rows were actually matched
print(f"Number of features in northeast_geom: {len(northeast_geom)}")
print(northeast_geom.head())

print(f"\n--- Processing variable: {VAR} (yearly {OPERATION}) ---")

livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, VAR)
livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
livneh_files = glob.glob(livneh_pattern)

if not livneh_files:
    sys.exit(f"No Livneh files found for variable: {VAR} at {livneh_var_dir}")

print(f"Loading Livneh monthly files for [{VAR}]...")
with xr.open_mfdataset(livneh_pattern, combine='by_coords', data_vars='all') as livneh_ds:
    livneh_yearly = aggregate_yearly(livneh_ds, OPERATION)

    # explicitly set coordinate reference system to WGS84 (EPSG:4326)
    livneh_yearly = livneh_yearly.rio.write_crs("EPSG:4326", inplace=True)
    livneh_yearly = livneh_yearly.rio.set_spatial_dims(x_dim='lon', y_dim='lat')

    # TEMPORARY: check CRS alignment
    print(f"Raster CRS: {livneh_yearly.rio.crs}")
    print(f"Shapefile CRS: {northeast_geom.crs}")

    print(f"Raster Lon Range: {livneh_yearly.lon.min().values} to {livneh_yearly.lon.max().values}")
    print(f"Shapefile bounds (xmin, ymin, xmax, ymax): {northeast_geom.total_bounds}")

    print(f"Clipping Livneh [{VAR}] to Northeast")
    livneh_yearly = clip_to_northeast(livneh_yearly, northeast_geom)
    # pull into memory now so the data survives past the file closing
    obs_da = livneh_yearly[VAR].load()

print(f"Yearly Livneh [{VAR}] dims: {dict(obs_da.sizes)}")

raw_obs_data = obs_da.values.flatten()
obs_mask = ~np.isnan(raw_obs_data) & (raw_obs_data < 1e10) & (raw_obs_data != -9999.0)
obs_clean = raw_obs_data[obs_mask]

if obs_clean.size == 0:
    sys.exit(f"No valid Livneh data for {VAR} after Northeast clip; cannot build Taylor diagram.")

# initialize SkillMetrics tracking lists w/ baseline's perfect score
sdev_list = [np.std(obs_clean)]
crmsd_list = [0.0]
cc_list = [1.0]

active_models = []
# (name, corr, crmsd) for every model that produced valid results, used
# to rank and report the best-performing models at the end
model_scores = []

best_model = None
lowest_error = float('inf')

for folder, name in zip(model_folders, model_names):
    model_pattern = os.path.join(folder, "monthly", VAR, "wbm_*.nc")
    if not glob.glob(model_pattern):
        print(f"  Skipping {name} [{VAR}]: no files found at {model_pattern}")
        continue

    try:
        with xr.open_mfdataset(model_pattern, combine='by_coords', data_vars='all') as model_ds:
            model_yearly = aggregate_yearly(model_ds, OPERATION)
            # clip the model grid to Northeast before regridding
            model_yearly = clip_to_northeast(model_yearly, northeast_geom)
            model_da = model_yearly[VAR].load()

        obs_regrid = obs_da.interp_like(model_da, method='nearest')

        model_data = model_da.values.flatten()
        obs_regrid_data = obs_regrid.values.flatten()

        valid_mask = (~np.isnan(model_data) & (model_data != -9999.0) &
                      ~np.isnan(obs_regrid_data) & (obs_regrid_data < 1e10))

        m_clean = model_data[valid_mask]
        o_clean = obs_regrid_data[valid_mask]

        # size/dimension checking
        total_cells = len(model_data)
        valid_cells = len(m_clean)
        pct_valid = (valid_cells / total_cells) * 100 if total_cells > 0 else 0
        print(f"    [size check] {name:<15} | Raw 1D: {total_cells:,} | Masked 1D: {valid_cells:,} ({pct_valid:.1f}% valid)")

        if len(m_clean) == 0 or len(o_clean) == 0:
            print(f"  Skipping {name} [{VAR}]: no overlapping valid data")
            continue

        # numpy math
        sdev = float(np.std(m_clean))
        o_std = float(np.std(o_clean))

        # extract correlation scalar
        corr = float(np.corrcoef(m_clean, o_clean)[0, 1])

        # standard CRMSD formula
        crmsd = float(np.sqrt(sdev**2 + o_std**2 - 2 * sdev * o_std * corr))

        sdev_list.append(round(sdev, 3))
        crmsd_list.append(round(crmsd, 3))
        cc_list.append(round(corr, 3))
        active_models.append(name)
        model_scores.append((name, corr, crmsd))

        print(f"  Model: {name:<15} | Corr: {corr:.3f} | CRMSE: {crmsd:.3f}")

        if crmsd < lowest_error:
            lowest_error = crmsd
            best_model = name

    except Exception as e:
        print(f"  Skipping model {name} [{VAR}] due to calculation mismatch: {e}")
        continue

# generating taylor diagram
if active_models:
    sdevs = np.array(sdev_list)
    crmsds = np.array(crmsd_list)
    ccs = np.array(cc_list)

    fig = plt.figure(figsize=(12, 10))

    markers_dict = {m: MODEL_STYLE[m] for m in active_models}

    sm.taylor_diagram(sdevs, crmsds, ccs,
        markerLegend='on',
        markers=markers_dict,
        markerObs='o',
        colObs='red',
        titleOBS='Reference',
        axisMax=float(np.max(sdevs) * 1.2),
        colCOR='black',
        colRMS='RoyalBlue',
        colSTD='SlateGray',
        styleRMS=':',
        styleSTD='--')

    ax = plt.gca()
    leg = ax.get_legend()
    handles = leg.legend_handles
    labels = [t.get_text() for t in leg.get_texts()]
    leg.remove()
    fig.subplots_adjust(right=0.9)
    ax.legend(handles, labels, loc='upper right', fontsize=7,
              ncol=2, framealpha=0.9, title='Models', title_fontsize=8)

    plt.title(f'LOCA2 Northeast Yearly Historical Validation (1980-2014): {VAR}',
              y=1.08, fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_plot = os.path.join(OUTPUT_DIR, f"northeast_{VAR}_yearly_taylor.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"Plot saved to {output_plot}")
    print(f"{best_model} runs closest to observed data")

    # rank models by lowest CRMSD (best agreement with Livneh observations)
    # and report the top N_BEST_MODELS
    ranked_models = sorted(model_scores, key=lambda x: x[2])
    top_models = ranked_models[:N_BEST_MODELS]

    print(f"\nTop {len(top_models)} models for [{VAR}] by lowest CRMSD:")
    for rank, (name, corr, crmsd) in enumerate(top_models, start=1):
        print(f"  {rank}. {name:<15} | CRMSE: {crmsd:.3f} | Corr: {corr:.3f}")
else:
    print(f"No models produced valid results for {VAR}; skipping Taylor diagram.")
