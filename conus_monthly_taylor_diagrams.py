import os
os.environ["PROJ_LIB"] = "/net/home/wsag/ir1101/miniconda3/share/proj"  # <-- Replace with the exact folder path found by the 'find' command
os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"]

import pyproj
proj_dir = os.path.join(os.path.dirname(pyproj.__file__), "proj_dir", "share", "proj")
if os.path.exists(proj_dir):
    os.environ["PROJ_DATA"] = proj_dir
    pyproj.datadir.set_data_dir(proj_dir)

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skill_metrics as sm
import seaborn as sns
import geopandas as gpd
import rioxarray
import shapefile
import shapely
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.vectorized import contains

# path to CDO-aggregated 1980-2013 baseline reference
LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980-2013.nc")

# directory with raw LOCA2-WBM downscaled model folders
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

# output location for taylor diagrams
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/taylor_diagrams/monthly")
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

# mapping project variables to CDO processed Livneh keys
var_mapping = {
    'airTmax': 'Tmax',
    'airTmin': 'Tmin',
    'precip': 'Prec'
}

folders = sorted(glob.glob(os.path.join(MODEL_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split('_')[0] for f in folders]

# colormap and symbols for plotting (later)
cmap = plt.get_cmap('tab20')
symbols = ['o','s','^','D','v','P','*','X','h','<','>','p','8','d','H']

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

    # Exclude non-CONUS states/territories by STATEFP
    # 02=AK, 15=HI, 60=AS, 66=GU, 69=MP, 72=PR, 78=VI
    excluded_statefp = {'02', '15', '60', '66', '69', '72', '78'}

    sf = shapefile.Reader(shapefile_path)
    geoms = []
    for shape_rec in sf.shapeRecords():
        record_dict = shape_rec.record.as_dict()
        if record_dict.get('STATEFP') not in excluded_statefp:
            geoms.append(shape(shape_rec.shape.__geo_interface__))

    return unary_union(geoms)


def clip_to_conus(ds, conus_geom):
    """Clips an xarray Dataset to CONUS geometry using Shapely vectorized operations.
    Completely avoids rioxarray/pyproj/GDAL.
    """
    # Identify lat/lon coordinate names
    lat_name = 'lat' if 'lat' in ds.coords else ('latitude' if 'latitude' in ds.coords else None)
    lon_name = 'lon' if 'lon' in ds.coords else ('longitude' if 'longitude' in ds.coords else None)

    if not lat_name or not lon_name:
        raise KeyError(f"Could not find lat/lon coordinates in dataset. Found: {list(ds.coords.keys())}")

    # Ensure longitudes are in [-180, 180] format to match shapefiles
    lons = ds[lon_name].values
    if np.any(lons > 180):
        ds = ds.assign_coords({lon_name: np.where(ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name])})
        ds = ds.sortby(lon_name)

    # Bounding box slice to speed up computation
    bounds = conus_geom.bounds  # (minx, miny, maxx, maxy)

    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(bounds[3] + 0.1, bounds[1] - 0.1)
    else:
        lat_slice = slice(bounds[1] - 0.1, bounds[3] + 0.1)

    ds_sub = ds.sel({
        lat_name: lat_slice,
        lon_name: slice(bounds[0] - 0.1, bounds[2] + 0.1)
    })

    # Build 2D grid of coordinates
    lon_2d, lat_2d = np.meshgrid(ds_sub[lon_name].values, ds_sub[lat_name].values)

    # Vectorized point-in-polygon check (using GEOS, no pyproj required)
    mask_2d = shapely.contains_xy(conus_geom, lon_2d, lat_2d)

    # Apply boolean mask to xarray object
    mask_da = xr.DataArray(mask_2d, coords={lat_name: ds_sub[lat_name], lon_name: ds_sub[lon_name]}, dims=[lat_name, lon_name])
    return ds_sub.where(mask_da)

print(f"Loading CONUS boundary from {SHAPEFILE_PATH}")
conus_geom = load_conus_boundary()

# run historical validation loop per variable
with xr.open_dataset(LIVNEH_REF) as obs_ds:

    # clip the full Livneh dataset to CONUS once so every variable pulled from it is already restricted to the CONUS region
    print("Clipping Livneh reference to CONUS...")
    obs_ds = clip_to_conus(obs_ds, conus_geom)

    for model_var, livneh_var in var_mapping.items():
        print(f"Generating Taylor Plot for {model_var}")

        obs_da = obs_ds[livneh_var].sel(time=slice('1980-01-01', '2013-12-31'))

        # If Livneh is in mm/month, convert to mm/day using exact calendar days
        if model_var == 'precip':
            # Check if values are large (~50-200), indicating mm/month
            if float(obs_da.mean(skipna=True)) > 10.0:
                print("Converting Livneh precip from mm/month -> mm/day")
                obs_da = obs_da / obs_da.time.dt.days_in_month

        obs_flat_ref = obs_da.values.flatten()
        ref_valid_mask = ~np.isnan(obs_flat_ref) & (obs_flat_ref != -9999.0)
        if model_var == 'precip':
            ref_valid_mask &= (obs_flat_ref >= 0)
        obs_clean_ref = obs_flat_ref[ref_valid_mask]

        sdev_list = [round(float(np.std(obs_clean_ref)), 3)]
        crmsd_list = [0.0]
        cc_list = [1.0]
        active_models = []
        best_model = None
        lowest_error = float('inf')

        # parse and compare each gcm folder
        for folder, name in zip(folders, model_names):
            model_pattern = os.path.join(folder, "monthly", model_var, "wbm_*.nc")
            if not glob.glob(model_pattern):
                continue
            try:
                with xr.open_mfdataset(model_pattern, combine='by_coords', data_vars='all') as model_ds:
                    model_da = model_ds[model_var].sel(time=slice('1980-01-01', '2013-12-31'))

                    precip_units = model_da.attrs.get('units') if model_var == 'precip' else None

                    # clip the model grid to CONUS before regridding so the extra domain is not looked at
                    model_da = clip_to_conus(model_da, conus_geom)

                    # ensure precipitation units match (mm/day)
                    if model_var == 'precip':
                        # if LOCA2/WBM is in kg/m^2/s, convert to mm/day
                        if precip_units == 'kg m-2 s-1':
                            model_da = model_da * 86400
                        # if model is mm/month, convert to mm/day
                        elif float(model_da.mean(skipna=True)) > 10.0:
                            model_da = model_da / model_da.time.dt.days_in_month

                    # regrid observations to model grid in xarray spatial space
                    # standardize time coordinates
                    obs_aligned = obs_da.assign_coords(time=model_da.time)
                    obs_regrid = obs_aligned.interp_like(model_da, method='nearest')

                    # flatten and mask
                    m_vals = model_da.values.flatten()
                    o_vals = obs_regrid.values.flatten()

                    if model_var == 'precip':
                        valid_mask = (
                            ~np.isnan(m_vals) & (m_vals != -9999.0) & (m_vals >= 0) &
                            ~np.isnan(o_vals) & (o_vals < 1e10) & (o_vals >= 0)
                        )
                    else:
                        valid_mask = (
                            ~np.isnan(m_vals) & (m_vals != -9999.0) &
                            ~np.isnan(o_vals) & (o_vals < 1e10)
                        )

                    m_clean = m_vals[valid_mask]
                    o_clean = o_vals[valid_mask]

                if len(m_clean) == 0 or len(o_clean) == 0:
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

                print(f"Model: {name:<15} | Corr: {corr:.3f} | CRMSE: {crmsd:.3f}")

                if crmsd < lowest_error:
                    lowest_error = crmsd
                    best_model = name

            except Exception as e:
                print(f"Skipping model {name} due to calculation mismatch: {e}")
                continue

        if active_models:
            sdevs = np.array(sdev_list)
            crmsds = np.array(crmsd_list)
            ccs = np.array(cc_list)

            fig = plt.figure(figsize=(12,10))

            markers_dict = {m: MODEL_STYLE[m] for m in active_models}

            sm.taylor_diagram(sdevs, crmsds, ccs,
                markerLegend='on',
                markers=markers_dict,
                markerObs='o',
                colObs='red',
                titleOBS='Reference',
                axisMax=float(np.max(sdevs)*1.2),
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

            plt.title(f'LOCA2 CONUS Historical Validation (1980-2013): {model_var}', y=1.08, fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_plot = os.path.join(OUTPUT_DIR, f"conus_{model_var}_monthly_taylor.png")
            plt.savefig(output_plot, dpi=300)
            plt.close()

            print(f"Plot saved to {output_plot}")
            print(f"{best_model} runs closest to observed data")
