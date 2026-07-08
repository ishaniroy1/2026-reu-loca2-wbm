"""
"""

from pathlib import Path
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# dynamic path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# resolve references relative to repo root
LIVNEH_REF = REPO_ROOT / "livneh_monthly_1980-2013.nc"
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"
OUTPUT_DIR = REPO_ROOT / "plots" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAR = "airTmax"
LIVNEH_VAR = "Tmax"

# first model folder with airTmax data
folders = sorted(glob.glob(os.path.join(MODEL_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split("_")[0] for f in folders]

first_folder, first_model, model_pattern = None, None, None
for folder, name in zip(folders, model_names):
    pattern = os.path.join(folder, "monthly", VAR, "wbm_*.nc")
    if glob.glob(pattern):
        first_folder, first_model, model_pattern = folder, name, pattern
        break

if first_folder is None:
    raise RuntimeError("No model folder in {MODEL_DIR} has data for {VAR}")

print(f"Using {first_model}:\n")

with xr.open_dataset(LIVNEH_REF) as obs_ds, \
        xr.open_mfdataset(model_pattern, combine="by_coords", data_vars="all") as model_ds:

            obs_da = obs_ds[LIVNEH_VAR]
            model_da = model_ds[VAR].sel(time=slice("1980-01-01", "2013-12-31"))

            # diagnostic 1: check coordinate names and dimension structure
            print("\nDiagnostic 1: Checking coordinate structures...")
            
            print(f"Livneh Dimensions: {obs_da.dims}")
            print(f"Model dimensions: {model_da.dims}")

            print("\nLivneh Coordinates:")
            for coord in obs_da.coords:
                print(f"    {coord}: min={obs_da[coord].values.min()}, max={obs_da[coord].values.max()}, shape={obs_da[coord].shape}")

            print("\nModel Coordinates:")
            for coord in model_da.coords:
                print(f"    {coord}: min={model_da[coord].values.min()}, max={model_da[coord].values.max()}, shape={model_da[coord].shape}")

            # diagnostic 2: look at straight-edged northern crown values
            print("\nDiagnostic 2: Sampling the northern edge block to identify hidden fill values...")
            
            model_t0 = model_da.isel(time=0)

            # slice from very top rows of the matrix with the blue region
            northern_slice = model_t0.isel(lat=slice(-20, None))
            northern_values = northern_slice.values.flatten()

            # exclude standard Nans to see what raw numbers are hiding there

            valid_north = northern_values[~np.isnan(northern_values)]

            if len(valid_north) > 0:
                unique_vals, counts = np.unique(valid_north, return_counts=True)
                # sort by frequency to find common background fill numbers
                top_indices = np.argsort(counts)[::-1]
                print("Most common raw values in the northern slice:")
                for idx in top_indices[:5]:
                    print(f"    Value: {unique_vals[idx]:.4f}   (Occurrences: {counts[idx]})")
            else:
                print(" No non-NaN values found in the northern slice.")

            # diagnostic 3: regenerate clean maps with proper masking
            print("\nDiagnostic 3: Generating side-by-side verification plot...")
            
            model_t0_masked = model_t0.where((model_t0 > -29.9) & (model_t0 != -9999.0))

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # using robust=True trims extreme 2nd and 98th percentiles to bypass hidden spikes
            obs_da.isel(time=0).plot(ax=axes[0], cmap="RdBu_r", robust=True)
            axes[0].set_title(f"Livneh Reference {LIVNEH_VAR} (time=0)")

            model_t0_masked.plot(ax=axes[1], cmap="RdBu_r", robust=True)
            axes[1].set_title(f"{first_model} {VAR} (Proper Masking and Robust Scales)")

            plt.tight_layout()
            static_map_path = os.path.join(OUTPUT_DIR, f"new_static_map_{VAR}.png")
            plt.savefig(static_map_path, dpi=200)
            plt.close(fig)
            print(f"Saved updated static map to {static_map_path}")
