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

# states/territories in the shapefile that are not part of CONUS
NON_CONUS = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
}

# map variables to aggregation operations
variables_config = {
    "airTmax": "mean",
    "airTmin": "mean",
    "precip": "sum"
}

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


def load_conus_boundary(shapefile_path=SHAPEFILE_PATH):
    """Load the state shapefile, keep only the 48 contiguous states + DC, and
    dissolve them into a single boundary in EPSG:4326 (plain lat/lon)."""
    states = gpd.read_file(shapefile_path)
    name_col = "NAME" if "NAME" in states.columns else "STUSPS"
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
    conus_states = conus_states.to_crs("EPSG:4326")
    return conus_states.dissolve().geometry


def clip_to_conus(da_or_ds, conus_geom, lat_name="lat", lon_name="lon", all_touched=True):
    """Clip an xarray DataArray/Dataset to the CONUS boundary. Assumes lon is
    already in -180..180 convention and lat/lon are 1-D coordinates."""
    obj = da_or_ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    return obj.rio.clip(conus_geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def aggregate_yearly(ds, operation):
    """Slice to the 1980-2014 period and resample to annual resolution.
    Precipitation is already a monthly accumulation (mm/month) on disk,
    so a simple annual sum is mathematically correct."""
    ds_sliced = ds.sel(time=slice('1980-01-01', '2014-12-31'))
    
    if operation == "mean":
        return ds_sliced.resample(time='YS').mean(dim='time')
    elif operation == "sum":
        return ds_sliced.resample(time='YS').sum(dim='time')
    else:
        raise ValueError(f"Unknown aggregation operation: {operation}")


print(f"Loading CONUS boundary from {SHAPEFILE_PATH}")
conus_geom = load_conus_boundary()

for var, operation in variables_config.items():
    print(f"\n--- Processing variable: {var} (yearly {operation}) ---")

    livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, var)
    livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
    livneh_files = glob.glob(livneh_pattern)

    if not livneh_files:
        print(f"No Livneh files found for variable: {var} at {livneh_var_dir}")
        continue

    print(f"Loading Livneh monthly files for [{var}]...")
    with xr.open_mfdataset(livneh_pattern, combine='by_coords', data_vars='all') as livneh_ds:
        livneh_yearly = aggregate_yearly(livneh_ds, operation)
        
        # TEMPORARY: Checking CRS alignment
        print(f"Raster CRS: {livneh_yearly.rio.crs}")
        print(f"Shapefile CRS: {conus_geom.crs}")

        print(f"Clipping Livneh [{var}] to CONUS")
        livneh_yearly = clip_to_conus(livneh_yearly, conus_geom)
        # pull into memory now so the data survives past the file closing
        obs_da = livneh_yearly[var].load()

    print(f"Yearly Livneh [{var}] dims: {dict(obs_da.sizes)}")

    raw_obs_data = obs_da.values.flatten()
    obs_mask = ~np.isnan(raw_obs_data) & (raw_obs_data < 1e10) & (raw_obs_data != -9999.0)
    obs_clean = raw_obs_data[obs_mask]

    if obs_clean.size == 0:
        print(f"No valid Livneh data for {var} after CONUS clip; skipping Taylor diagram.")
        continue

    # initialize SkillMetrics tracking lists w/ baseline's perfect score
    sdev_list = [np.std(obs_clean)]
    crmsd_list = [0.0]
    cc_list = [1.0]

    active_models = []
    best_model = None
    lowest_error = float('inf')

    for folder, name in zip(model_folders, model_names):
        model_pattern = os.path.join(folder, "monthly", var, "wbm_*.nc")
        if not glob.glob(model_pattern):
            print(f"  Skipping {name} [{var}]: no files found at {model_pattern}")
            continue

        try:
            with xr.open_mfdataset(model_pattern, combine='by_coords', data_vars='all') as model_ds:
                model_yearly = aggregate_yearly(model_ds, operation)
                # clip the model grid to CONUS before regridding
                model_yearly = clip_to_conus(model_yearly, conus_geom)
                model_da = model_yearly[var].load()

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
                print(f"  Skipping {name} [{var}]: no overlapping valid data")
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

            print(f"  Model: {name:<15} | Corr: {corr:.3f} | CRMSE: {crmsd:.3f}")

            if crmsd < lowest_error:
                lowest_error = crmsd
                best_model = name

        except Exception as e:
            print(f"  Skipping model {name} [{var}] due to calculation mismatch: {e}")
            continue

    # generating taylor diagrams
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

        plt.title(f'LOCA2 CONUS Yearly Historical Validation (1980-2014): {var}',
                  y=1.08, fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_plot = os.path.join(OUTPUT_DIR, f"conus_{var}_yearly_taylor.png")
        plt.savefig(output_plot, dpi=300)
        plt.close()

        print(f"Plot saved to {output_plot}")
        print(f"{best_model} runs closest to observed data")
    else:
        print(f"No models produced valid results for {var}; skipping Taylor diagram.")
