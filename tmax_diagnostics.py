"""
Diagnostics for high std. dev. values showing up in the Taylor Diagrams. Focuses on airTmax and the ACCESS-CM2 model.
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980-2013.nc")
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

            # raw value comparison to check units
            print("Raw value comparison (checking units):")

            obs_t0 = obs_da.isel(time=0).values
            obs_valid = obs_t0[~np.isnan(obs_t0) & (obs_t0 < 1e10) & (obs_t0 != -9999.0)]
            print(f"Livneh {LIVNEH_VAR}, first time step, first 20 valid values:")
            print(np.round(obs_valid[:20], 3))
            print(f"    min={np.min(obs_valid):.3f} mean={np.mean(obs_valid):.3f}   max={np.max(obs_valid):.3f}\n")

            model_t0 = model_da.isel(time=0).values
            model_valid = model_t0[~np.isnan(model_t0) & (model_t0 != -9999.0)]
            print(f"{first_model} {VAR}, first time step, first 20 valid values:")
            print(np.round(model_valid[:20], 3))
            print(f"    min={np.min(model_valid):.3f}, mean={np.mean(model_valid)}, max={np.max(model_valid):.3f}\n")

            offset = np.mean(model_valid) - np.mean(obs_valid)
            print(f"Difference in means: {offset:.3f}")
            print("An offset near +273 or +32 is a sign of a units mismatch between the livneh and model datasets\n")

            # ncview and static stand-in map
            print("ncview and static stand-in map:")
            model_file = sorted(glob.glob(model_pattern))[0]
            print("Run the following ncview commands in a terminal window:")
            print(f"    ncview {LIVNEH_REF}")
            print(f"    ncview {model_file}")

            """
            Check:
            1. Does the color bar range match the expected units
            2. Does the spatial pattern look like a real temperature field (smooth and coherent w/ latitude/elevation) rather than noisy or misaligned (which would suggest a grid/regridding problem)
            """

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            obs_da.isel(time=0).plot(ax=axes[0], cmap="RdBu_r")
            axes[0].set_title(f"Livneh {LIVNEH_VAR} (time=0)")
            model_da.isel(time=0).plot(ax=axes[1], cmap="RdBu_r")
            axes[1].set_title(f"{first_model} {VAR} (time=0")
            plt.tight_layout()
            static_map_path = os.path.join(OUTPUT_DIR, f"static_map_{VAR}.png")
            plt.savefig(static_map_path, dpi=200)
            plt.close(fig)
            print(f"Static map saved to {static_map_path}\n")

            # year-by-year difference (Livneh - Model)
            print("Year-by-year difference (Livneh - Model)")
            obs_regrid = obs_da.interp_like(model_da, method="nearest")

            obs_annual = obs_regrid.groupby("time.year").mean("time")
            model_annual = model_da.groupby("time.year").mean("time")
            diff_annual = obs_annual - model_annual # dims: year, lat, lon

            spatial_dims = [d for d in diff_annual.dims if d != "year"]
            diff_by_year = diff_annual.mean(dim=spatial_dims)

            years = diff_by_year["year"].values
            diffs = diff_by_year.values

            print("Domain-mean difference per year:")
            for y, d, in zip(years, diffs):
                print(f"    {y}: {d:.3f}")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(years, diffs, color="steelblue")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Year")
            ax.set_ylabel(f"Livneh - {first_model} ({VAR}) mean diff")
            ax.set_title(f"Annual mean difference: Livneh vs {first_model} ({VAR})")
            plt.tight_layout()
            diff_plot_path = os.path.join(OUTPUT_DIR, f"annual_diff_{VAR}_{first_model}.png")
            plt.savefig(diff_plot_path, dpi=200)
            plt.close(fig)

            # full period spatial mean-difference map (find location of error)
            diff_mean_map = diff_annual.mean(dim="year")
            fig, ax = plt.subplots(figsize=(8, 6))
            diff_mean_map.plot(ax=ax, cmap="RdBu_r", center=0)
            ax.set_title(f"Mean 1980-2013 difference: Livneh - {first_model} ({VAR})")
            plt.tight_layout()
            diff_map_path = os.path.join(OUTPUT_DIR, f"mean_diff_map_{VAR}_{first_model}.png")
            plt.savefig(diff_map_path, dpi=200)
            plt.close(fig)

            print(f"\nYearly difference plot saved to {diff_plot_path}")
            print(f"\nSpatial mean-difference map saved to {diff_map_path}")
            print(f"Overall mean diff: {np.nanmean(diffs):.3f} \nmin: {np.nanmin(diffs):.3f} \nmax: {np.nanmax(diffs):.3f}")
