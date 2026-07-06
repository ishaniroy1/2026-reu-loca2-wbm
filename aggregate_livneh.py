import xarray as xr
import glob
import os

LIVNEH_DIR = "/net/nfs/squam/raid/data/LIVNEH/data.nodc.noaa.gov/nodc/archive/data/0129374/daily"
OUT_FILE = "livneh_monthly_reference.nc"

all_years = []

for year in range(1980, 2014):  # 1980–2013 (matches your data)
    files = sorted(glob.glob(f"{LIVNEH_DIR}/livneh_NAmerExt_15Oct2014.{year}*.nc"))
    
    if not files:
        print(f"WARNING: missing year {year}")
        continue

    print("Processing", year)

    ds = xr.open_mfdataset(files, combine="by_coords")

    # Monthly aggregation
    monthly = xr.Dataset()

    monthly["Tmax"] = ds["Tmax"].resample(time="MS").mean()
    monthly["Tmin"] = ds["Tmin"].resample(time="MS").mean()
    monthly["Prec"] = ds["Prec"].resample(time="MS").sum()

    all_years.append(monthly)

final = xr.concat(all_years, dim="time")
final.to_netcdf(OUT_FILE)

print("Done:", OUT_FILE)
