import argparse
from pathlib import Path
 
import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor on xarray objects)
 
SHAPEFILE_PATH = Path("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp").expanduser()
 
# states/territories in the Census file that are NOT part of the contiguous US
NON_CONUS = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
}
 
 
def load_conus_boundary(shapefile_path=SHAPEFILE_PATH):
    """Load the state shapefile, keep only the 48 contiguous states + DC, and
    dissolve them into a single boundary (so internal state lines don't
    fragment the mask). Returns geometry in EPSG:4326 (plain lat/lon)."""
    states = gpd.read_file(shapefile_path)
 
    name_col = "NAME" if "NAME" in states.columns else "STUSPS"
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
 
    conus_states = conus_states.to_crs("EPSG:4326")
    conus_geom = conus_states.dissolve().geometry  # single (multi)polygon
 
    return conus_geom, conus_states
 
 
def clip_to_conus(da_or_ds, conus_geom, lat_name="lat", lon_name="lon", all_touched=True):
    """Clip an xarray DataArray/Dataset to the CONUS boundary.
 
    Assumes lon is already in -180..180 convention (matches this project's
    files) and that lat/lon are 1-D coordinates. all_touched=True keeps any
    grid cell that the boundary polygon touches at all, which avoids losing
    thin coastal/border cells at your grid resolution.
    """
    obj = da_or_ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
 
    clipped = obj.rio.clip(conus_geom, crs="EPSG:4326", drop=True, all_touched=all_touched)
    return clipped
 
 
def main():
    parser = argparse.ArgumentParser(description="Clip a netCDF file to CONUS using a state shapefile.")
    parser.add_argument("--input", required=True, help="Path to input .nc file")
    parser.add_argument("--var", required=True, help="Variable name to clip (e.g. airTmax, Tmax)")
    parser.add_argument("--output", required=True, help="Path to write the clipped .nc file")
    parser.add_argument("--lat-name", default="lat")
    parser.add_argument("--lon-name", default="lon")
    parser.add_argument("--shapefile", default=str(SHAPEFILE_PATH))
    args = parser.parse_args()
 
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
 
    print(f"Loading CONUS boundary from {args.shapefile}")
    conus_geom, conus_states = load_conus_boundary(args.shapefile)
    print(f"Kept {len(conus_states)} states/districts for CONUS")
 
    print(f"Opening {input_path}")
    with xr.open_dataset(input_path) as ds:
        da = ds[args.var]
 
        print("Clipping to CONUS...")
        clipped = clip_to_conus(da, conus_geom, lat_name=args.lat_name, lon_name=args.lon_name)
 
        print(f"Original shape: {da.shape} -> Clipped shape: {clipped.shape}")
 
        out_ds = clipped.to_dataset(name=args.var)
        out_ds.attrs["history"] = f"Clipped to CONUS using {args.shapefile}"
 
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}")
        out_ds.to_netcdf(output_path)
 
    print("Done.")
 
 
if __name__ == "__main__":
    main()
