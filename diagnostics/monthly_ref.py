from pathlib import Path
import xarray as xr
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "livneh_monthly_reference.nc"
ds = xr.open_dataset(data_path)

print("TIME DIFF (median):")
print(ds.time.diff("time").median().values)

print("\nTIME RANGE:")
print(ds.time.values[0], "→", ds.time.values[-1])

print("\nSTATS (Tmax):")
print("mean SD over time:", ds["Tmax"].std("time").mean().values)
print("max SD over time:", ds["Tmax"].std("time").max().values)


print(ds["Tmax"].shape)

print("NaN fraction:",
      np.isnan(ds["Tmax"]).sum().item() / ds["Tmax"].size)

print("Min:", float(ds["Tmax"].min()))
print("Max:", float(ds["Tmax"].max()))

cell = ds["Tmax"].isel(lat=300, lon=400)

print(cell.values[:12])
print(cell.std().values)
