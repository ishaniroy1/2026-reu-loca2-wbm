import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skill_metrics as sm
import seaborn as sns
import geopandas as gpd
import regionmask

# paths
LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980-2014.nc")
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
symbols = ['o','s','^','D','v','P','*','X','h','<','>','p','8','d','H']
# path to US states shapefile
SHAPEFILE_PATH = os.path.expanduser("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

# load shapefile and filter to northeast region
nca_ne_states = ['ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'WV']
states_gdf = gpd.read_file(SHAPEFILE_PATH)
northeast_gdf = states_gdf[states_gdf['STUSPS'].isin(nca_ne_states)]

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

with xr.open_dataset(LIVNEH_REF) as obs_ds:
    for model_var, livneh_var in var_mapping.items():
        print(f"\nGenerating Northeast Shapefile Masked Taylor Plot for {model_var}")

        # creating boolean spatial mask over the observation grid coordinates
        obs_mask_region = regionmask.mask_3D_geopandas(
            northeast_gdf,
            obs_ds['lon'],
            obs_ds['lat']
        ).squeeze()

        # apply shapefile mask to reference var
        obs_ne = obs_ds[livneh_var].where(obs_mask_region)

        raw_obs_data = obs_ne.values.flatten()
        obs_mask = ~np.isnan(raw_obs_data) & (raw_obs_data < 1e10) & (raw_obs_data != -9999.0)
        obs_clean = raw_obs_data[obs_mask]

        sdev_list = [np.std(obs_clean)]
        crmsd_list = [0.0]
        cc_list = [1.0]

        active_models = []
        best_model = None
        lowest_error = float('inf')

        for folder, name in zip(folders, model_names):
            model_pattern = os.path.join(folder, "monthly", model_var, "wbm_*.nc")
            if not glob.glob(model_pattern):
                continue
            try:
                with xr.open_mfdataset(model_pattern, combine='by_coords') as model_ds:
                    model_trimmed = model_ds[model_var].sel(time=slice('1980-01-01', '2014-12-31'))

                    # creating unique shapefile mask corresponding to model's spatial resolution
                    model_mask_region = regionmask.mask_3D_geopandas(
                        northeast_gdf,
                        model_trimmed['lon'],
                        model_trimmed['lat']
                    ).squeeze()

                    # applying shapefile mask to target model
                    model_masked = model_trimmed.where(model_mask_region)

                    # match masked observations to model layout
                    obs_regrid = obs_ne.interp_like(model_masked, method='nearest')

                    obs_regrid_data = obs_regrid.values.flatten()
                    model_data = model_masked.values.flatten()

                    if model_var == 'precip':
                        model_data = model_data * 30.5 # unit scaling

                valid_mask = (~np.isnan(model_data) & (model_data != -9999.0) & ~np.isnan(obs_regrid_data) & (obs_regrid_data < 1e10))

                m_clean = model_data[valid_mask]
                o_clean = obs_regrid_data[valid_mask]

                if len(m_clean) == 0 or len(o_clean) == 0:
                    continue

                sdev = float(np.std(m_clean))
                o_std = float(np.std(o_clean))
                corr = float(np.corrcoef(m_clean, o_clean)[0, 1])
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
                print(f"Skipping model {name} due to error: {e}")
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
            handles = leg.legendHandles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            fig.subplots_adjust(right=0.9)
            ax.legend(handles, labels, loc='upper right', fontsize=7, ncol=2, framealpha=0.9, title='Models', title_fontsize=8)

            plt.title(f"LOCA2 Northeast Historical Validation (1980-2014): {model_var}", y=1.08, fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_plot = os.path.join(OUTPUT_DIR, f"northeast_{model_var}_monthly_taylor.png")
            plt.savefig(output_plot, dpi=300)
            plt.close()

            print(f"Shapefile-masked plot saved to {output_plot}")
            print(f"{best_model} runs closest to observed data")
