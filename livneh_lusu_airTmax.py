import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skill_metrics as sm
import seaborn as sns

# paths
LIVNEH_REF_DIR = os.path.expanduser("/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly/airTmax")
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

# target variables
model_var = 'airTmax'
livneh_var = 'airTmax'

folders = sorted(glob.glob(os.path.join(MODEL_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split('_')[0] for f in folders]

# capture multiple files for baseline references
livneh_pattern = os.path.join(LIVNEH_REF_DIR, "wbm_*.nc")
print(f"Loading reference baseline files from: {livneh_pattern}")

with xr.open_mfdataset(livneh_pattern, combine='by_coords', data_vars='all') as obs_ds:
    # slice observation reference to 1980-2014
    obs_sliced = obs_ds[livneh_var].sel(time=slice('1980-01-01', '2014-12-31'))

    # get reference standard deviation
    obs_ref_raw = obs_sliced.values.flatten()
    ref_mask = ~np.isnan(obs_ref_raw) & (obs_ref_raw < 1e10) & (obs_ref_raw != -9999.0)
    clean_obs_ref = obs_ref_raw[ref_mask]

    if len(clean_obs_ref) == 0:
        raise ValueError("The sliced observation reference dataset contains no valid points.")

    # Initialize tracking lists with the reference values
    obs_std = float(np.std(clean_obs_ref))

    sdev_list = [obs_std]
    crmsd_list = [0.0]
    cc_list = [1.0]
    active_models = ["Observation"]

    best_model = None
    lowest_error = float('inf')

    # Parse and compare each GCM folder
    for folder, name in zip(folders, model_names):
        model_pattern = os.path.join(folder, "monthly", model_var, "wbm_*.nc")
        if not glob.glob(model_pattern):
            continue

        try:
            # Lazy open matching LOCA2 runs
            with xr.open_mfdataset(model_pattern, combine='by_coords', data_vars='all') as model_ds:
                # slice target period
                model_trimmed = model_ds[model_var].sel(time=slice('1980-01-01', '2014-12-31'))

                # regrid baseline reference to match GCM spatial domain
                obs_regrid = obs_sliced.interp_like(model_trimmed, method='nearest')
                
                obs_regrid_data = obs_regrid.values.flatten()
                model_data = model_trimmed.values.flatten()

                # robust double-sided masking
                valid_mask = (
                    ~np.isnan(model_data) &
                    (model_data != -9999.0) &
                    ~np.isnan(obs_regrid_data) &
                    (obs_regrid_data < 1e10) &
                    (obs_regrid_data != -9999.0)
                )

                m_clean = model_data[valid_mask]
                o_clean = obs_regrid_data[valid_mask]

                if len(m_clean) == 0 or len(o_clean) == 0:
                    print(f"Skipping {name}: No intersecting spatial cells found")
                    continue

                # compute std dev of model and observations
                sdev = float(np.std(m_clean))
                o_std = float(np.std(o_clean))

                # compute corr coeff
                corr = float(np.corrcoef(m_clean, o_clean)[0, 1])
                # CRMSD using updated formula
                crmsd = np.sqrt(np.mean(((m_clean - np.mean(m_clean)) - (o_clean - np.mean(o_clean)))**2))

                sdev_list.append(round(sdev, 3))
                crmsd_list.append(round(crmsd, 3))
                cc_list.append(round(corr, 3))
                active_models.append(name)

                print(f"Model: {name:<15} | Corr: {corr:.3f} | CRMSE: {crmsd:.3f}")

                if crmsd < lowest_error:
                    lowest_error = crmsd
                    best_model = name

        except Exception as e:
            print(f"Skipping model {name} due to calculation mismatch.")
            continue
