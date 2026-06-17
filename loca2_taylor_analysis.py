import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skill_metrics as sm
import seaborn as sns

# path to CDO-aggregated 1980-2010 baseline reference
LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980_2010.nc")

# directory with raw LOCA2-WBM downscaled model folders
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

# output location for taylor diagrams
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# run historical validation loop per variable
with xr.open_dataset(LIVNEH_REF) as obs_ds:
    for model_var, livneh_var in var_mapping.items():
        print(f"Generating Taylor Plot for {model_var}")

        raw_obs_data = obs_ds[livneh_var].values.flatten()
        obs_mask = ~np.isnan(raw_obs_data) & (raw_obs_data < 1e10) & (raw_obs_data != -9999.0)
        obs_clean = raw_obs_data[obs_mask]

        # initialize SkillMetrics tracking lists w/ baseline's perfect score
        sdev_list = [np.std(obs_clean)]
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
                with xr.open_mfdataset(model_pattern, combine='by_coords') as model_ds:
                    model_trimmed = model_ds[model_var].sel(time=slice('1980-01-01', '2010-12-31'))
                    obs_regrid = obs_ds[livneh_var].interp_like(model_trimmed, method='nearest')

                    obs_regrid_data = obs_regrid.values.flatten()
                    model_data = model_trimmed.values.flatten()
                    # unit scaling for precip
                    if model_var == 'precip':
                        model_data = model_data*30.5 # converting mm/day rate to monthly accumulation total

                valid_mask = (~np.isnan(model_data) & (model_data != -9999.0) & ~np.isnan(obs_regrid_data) & (obs_regrid_data < 1e10))

                m_clean = model_data[valid_mask]
                o_clean = obs_regrid_data[valid_mask]

                if len(m_clean) == 0 or len(o_clean) == 0:
                    continue

                # numpy math
                sdev = float(np.std(m_clean))
                o_std = float(np.std(o_clean))
                
                # extract correlation scalar
                corr = float(np.corrcoef(m_clean, o_clean)[0, 1])
                
                # standard CRMSD formula
                crmsd = float(np.sqrt(sdev**2 + o_std**2 - 2 * sdev * o_std * corr))
                
                sdev_list.append(sdev)
                crmsd_list.append(crmsd)
                cc_list.append(corr)
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
            handles = leg.legendHandles
            labels = [t.get_text() for t in leg.get_texts()]
            leg.remove()
            fig.subplots_adjust(right=0.9)
            ax.legend(handles, labels, loc='upper right', fontsize=7,
                    ncol=2, framealpha=0.9, title='Models', title_fontsize=8)
                    
            plt.title(f'LOCA2 Historical Validation (1980-2014): {model_var}', y=1.08, fontsize=14, fontweight='bold')
            plt.tight_layout()
            output_plot = os.path.join(OUTPUT_DIR, f"{model_var}_taylor_diagram.png")
            plt.savefig(output_plot, dpi=300)
            plt.close()

            print(f"Plot saved to {output_plot}")
            print(f"{best_model} runs closest to observed data")
