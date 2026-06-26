import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask

# paths
LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980-2014.nc")
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_future"
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots")
SHAPEFILE_PATH = os.path.expanduser(
    "~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# region setup
nca_ne_states = ['ME', 'NH', 'VT', 'MA', 'RI',
                 'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'WV']
states_gdf = gpd.read_file(SHAPEFILE_PATH)
northeast_gdf = states_gdf[states_gdf['STUSPS'].isin(nca_ne_states)]

# set up 3 scenarios
scenarios = {
    'ssp245': {'label': 'SSP2-4.5 (Middle of the Road)', 'color': 'darkorange'},
    'ssp370': {'label': 'SSP3-7.0 (Regional Rivalry)', 'color': 'purple'},
    'ssp585': {'label': 'SSP5-8.5 (Fossil-fueled Development)', 'color': 'crimson'}
}

# mapping project variables to CDO processed Livneh keys
var_mapping = {
    'airTmax': 'Tmax',
    'airTmin': 'Tmin',
    'precip': 'Prec',
}

for model_var, livneh_var in var_mapping.items():
    print(f"\nProcessing Future Timeseries for {model_var}")

    fig, ax = plt.subplots(figsize=(14, 6.5))

    # process livneh observations
    with xr.open_dataset(LIVNEH_REF) as obs_ds:
        obs_mask = regionmask.mask_3D_geopandas(
            northeast_gdf, obs_ds['lon'], obs_ds['lat']).any(dim='region')
        obs_ts = obs_ds[livneh_var].where(obs_mask).mean(dim=['lat', 'lon'])

        # resample to annual mean to smooth out monthly noise
        obs_annual = obs_ts.resample(time='1YS').mean()

        ax.plot(obs_annual['time'].dt.year, obs_annual.values, color='black',
                linewidth=2.5, label='Livneh Observations (1980-2014)')

        # use last obs point to bridge year gap
        obs_bridge_year = int(obs_annual['time'].dt.year.values[-1])
        obs_bridge_value = float(obs_annual.values[-1])

    # process future ensembles by scenario
    for ssp_key, ssp_info in scenarios.items():

        ssp_folders = sorted(
            glob.glob(os.path.join(MODEL_DIR, f"*{ssp_key}*_newprcp")))

        if not ssp_folders:
            continue

        all_model_series = []

        for folder in ssp_folders:
            model_name = os.path.basename(folder).split('_')[0]
            model_pattern = os.path.join(
                folder, "monthly", model_var, "wbm_*.nc")

            if not glob.glob(model_pattern):
                continue

            try:
                with xr.open_mfdataset(model_pattern, combine='by_coords') as model_ds:
                    # slice time
                    model_trimmed = model_ds[model_var].sel(
                        time=slice('2015-01-01', '2100-12-31'))

                    # generate spatial mask for model's resolution layout
                    m_mask = regionmask.mask_3D_geopandas(
                        northeast_gdf, model_trimmed['lon'], model_trimmed['lat']).any(dim='region')

                    spatial_mean = model_trimmed.where(
                        m_mask).mean(dim=['lat', 'lon'])

                    if model_var == 'precip':
                        spatial_mean = spatial_mean * 30.5  # convert mm/day rate to monthly total

                    # resample to annual timeline
                    annual_mean = spatial_mean.resample(time='1YS').mean()
                    all_model_series.append(annual_mean.load())

            except Exception as e:
                print(
                    f"Skipping future processing for {model_name} ({ssp_key}): {e}")
                continue

        if all_model_series:
            # concatenate individual model arrays
            ensemble_ds = xr.concat(all_model_series, dim='model')
            ensemble_mean = ensemble_ds.mean(dim='model')
            years = ensemble_mean['time'].dt.year

            # prepend obs bridge point to close the gap
            years_connected = np.concatenate([[obs_bridge_year], years])
            mean_connected = np.concatenate(
                [[obs_bridge_value], ensemble_mean.values])

            # plot scenario multi-model average trendline
            ax.plot(years_connected, mean_connected,
                    color=ssp_info['color'], linewidth=2, label=ssp_info['label'])

            # shade the spread representing cross-model variation bounds
            ensemble_annual = ensemble_ds.resample(time='1YS').mean()
            ensemble_min = ensemble_annual.min(dim='model').values.flatten()
            ensemble_max = ensemble_annual.max(
                dim='model').values.flatten()

            # prepend obs value for spread too (zero spread at the bridge point)
        min_connected = np.concatenate([[obs_bridge_value], ensemble_min])
        max_connected = np.concatenate([[obs_bridge_value], ensemble_max])

        ax.fill_between(years_connected, min_connected,
                        max_connected, color=ssp_info['color'], alpha=0.10)

    # era shading
    ax.axvspan(2015, 2040, color='royalblue', alpha=0.03)
    ax.axvspan(2041, 2070, color='darkorange', alpha=0.03)
    ax.axvspan(2071, 2100, color='crimson', alpha=0.03)

    # indicators
    ax.autoscale_view()
    y_lo, y_hi = ax.get_ylim()
    y_text_pos = y_lo + (y_hi - y_lo) * 0.92

    ax.text(2027, y_text_pos, 'Early-Century\n(2015-2040)', ha='center',
            va='top', fontsize=9, style='italic', color='navy')
    ax.text(2055, y_text_pos, 'Mid-Century\n(2041-2070)', ha='center',
            va='top', fontsize=9, style='italic', color='darkgoldenrod')
    ax.text(2085, y_text_pos, 'Late-Century\n(2071-2100)', ha='center',
            va='top', fontsize=9, style='italic', color='darkred')

    # figure labels
    unit_label = "Precipitation (mm/month)" if model_var == 'precip' else "Temperature (°C)"
    ax.set_ylabel(unit_label, fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title(
        f"Northeast Region Future Projections (2015-2100): {model_var}", fontsize=14, fontweight='bold')

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    plt.tight_layout()

    output_plot = os.path.join(
        OUTPUT_DIR, f"{model_var}_future_timeseries.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"Completed plot written to {output_plot}")
