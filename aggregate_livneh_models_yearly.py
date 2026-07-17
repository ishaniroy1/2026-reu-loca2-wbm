import os
import glob
import xarray as xr

# filepaths
LIVNEH_BASE_DIR = "/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly"
MODEL_BASE_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

# map variables to aggregation operations
variables_config = {
    "airTmax": "mean",
    "airTmin": "mean",
    "precip": "sum"
}

# processing livneh reference data
print("--- Processing Livneh Reference Datasets ---")
for var, operation in variables_config.items():
    livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, var)
    livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
    
    files = glob.glob(livneh_pattern)
    if not files:
        print(f"No Livneh files found for variable: {var} at {livneh_var_dir}")
        continue
        
    print(f"\nLoading Livneh monthly files for [{var}]...")
    with xr.open_mfdataset(livneh_pattern, combine='by_coords', data_vars='all') as ds:
        # slice target period
        ds_sliced = ds.sel(time=slice('1980-01-01', '2014-12-31'))
        
        # resampling
        if operation == "mean":
            ds_yearly = ds_sliced.resample(time='YS').mean(dim='time')
        elif operation == "sum":
            ds_yearly = ds_sliced.resample(time='YS').sum(dim='time')
            
        print(f"Successfully aggregated Livneh [{var}] (Operation: {operation})")
        print(f"Original dimensions: {dict(ds_sliced.sizes)}")
        print(f"Yearly dimensions:   {dict(ds_yearly.sizes)}")
        
        # save aggregated dataset to netCDF
        # output_path = os.path.join(livneh_var_dir, f"yearly_{var}_1980_2014.nc")
        # ds_yearly.to_netcdf(output_path)


# processing model data
print("\n--- Processing Model Datasets ---")
model_folders = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "*_newprcp")))

for folder in model_folders:
    model_name = os.path.basename(folder).split('_')[0]
    print(f"\n--- Model: {model_name} ---")
    
    for var, operation in variables_config.items():
        model_pattern = os.path.join(folder, "monthly", var, "wbm_*.nc")
        files = glob.glob(model_pattern)
        
        if not files:
            print(f"  Skipping {var}: No files found at {model_pattern}")
            continue
            
        try:
            with xr.open_mfdataset(model_pattern, combine='by_coords', data_vars='all') as ds:
                ds_sliced = ds.sel(time=slice('1980-01-01', '2014-12-31'))
                
                # resampling
                if operation == "mean":
                    ds_yearly = ds_sliced.resample(time='YS').mean(dim='time')
                elif operation == "sum":
                    ds_yearly = ds_sliced.resample(time='YS').sum(dim='time')
                
                print(f"  [{var}] Aggregated ({operation}) -> Yearly Dims: {dict(ds_yearly.sizes)}")
                
                # save to a new yearly directory within the GCM folder
                # out_dir = os.path.join(folder, "yearly", var)
                # os.makedirs(out_dir, exist_ok=True)
                # ds_yearly.to_netcdf(os.path.join(out_dir, f"yearly_wbm_{var}.nc"))
                
        except Exception as e:
            print(f"  Failed processing {var} for {model_name}: {e}")
