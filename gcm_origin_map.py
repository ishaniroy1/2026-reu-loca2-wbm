import os
import sys

proj_path = "/net/home/wsag/ir1101/miniconda3/share/proj"
os.environ["PROJ_DATA"] = proj_path
os.environ["PROJ_LIB"] = proj_path

import pyproj
pyproj.datadir.set_data_dir(proj_path)

# lets GDAL/OGR regenerate a missing .shx sidecar file instead of failing
# outright -- guards against incomplete shapefile downloads/extractions
os.environ["SHAPE_RESTORE_SHX"] = "YES"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import geopandas as gpd

OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/gcm_origin_map")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Natural Earth 110m Admin-0 Countries shapefile -- download once from
# https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
WORLD_SHAPEFILE_PATH = os.path.expanduser(
    "~/LOCA2-WBM_code/shapefiles/world/ne_110m_admin_0_countries.shp"
)

# EDIT THIS to your actual final GCM list (trim/adjust from the table below)
GCM_COUNTRY = {
    "ACCESS-CM2": "AUS",
    "ACCESS-ESM1-5": "AUS",
    "BCC-CSM2-MR": "CHN",
    "CanESM5": "CAN",
    "EC-Earth3": "NLD",
    "EC-Earth3-Veg": "NLD",
    "FGOALS-g3": "CHN",
    "GFDL-CM4": "USA",
    "GFDL-ESM4": "USA",
    "INM-CM4-8": "RUS",
    "INM-CM5-0": "RUS",
    "IPSL-CM6A-LR": "FRA",
    "KACE-1-0-G": "KOR",
    "MIROC6": "JPN",
    "MPI-ESM1-2-HR": "DEU",
    "MPI-ESM1-2-LR": "DEU",
    "MRI-ESM2-0": "JPN",
    "NorESM2-LM": "NOR",
    "NorESM2-MM": "NOR",
}

COUNTRY_LABELS = {
    "AUS": "Australia", "CHN": "China", "CAN": "Canada", "NLD": "Netherlands",
    "USA": "USA", "RUS": "Russia", "FRA": "France", "KOR": "South Korea",
    "JPN": "Japan", "DEU": "Germany", "NOR": "Norway",
}

print(f"Loading world boundary from {WORLD_SHAPEFILE_PATH}")
world = gpd.read_file(WORLD_SHAPEFILE_PATH)

iso_col = "ISO_A3" if "ISO_A3" in world.columns else "ADM0_A3"
world = world.to_crs("EPSG:4326")

country_counts = {}
country_models = {}
for gcm, iso3 in GCM_COUNTRY.items():
    country_counts[iso3] = country_counts.get(iso3, 0) + 1
    country_models.setdefault(iso3, []).append(gcm)

world["gcm_count"] = world[iso_col].map(country_counts).fillna(0)

n_models = len(GCM_COUNTRY)
n_countries = len(country_counts)
print(f"{n_models} GCMs across {n_countries} countries")
for iso3, count in sorted(country_counts.items(), key=lambda x: -x[1]):
    label = COUNTRY_LABELS.get(iso3, iso3)
    print(f"  {label} ({iso3}): {count} model(s) -- {', '.join(country_models[iso3])}")

fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

world.boundary.plot(ax=ax, linewidth=0.3, color="black")

max_count = int(world["gcm_count"].max())
bounds = np.arange(-0.5, max_count + 1.5, 1)
cmap = plt.get_cmap("YlOrRd", max_count + 1)
norm = BoundaryNorm(bounds, cmap.N)

world.plot(column="gcm_count", ax=ax, cmap=cmap, norm=norm,
           edgecolor="black", linewidth=0.3)

for iso3, count in country_counts.items():
    match = world[world[iso_col] == iso3]
    if match.empty:
        print(f"  WARNING: no shapefile match for ISO3 '{iso3}' -- check {iso_col} values")
        continue
    centroid = match.geometry.iloc[0].representative_point()
    ax.annotate(str(count), (centroid.x, centroid.y), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")

ax.set_xticks([])
ax.set_yticks([])
ax.set_title(
    f"Home Institutions of the {n_models}-Model LOCA2 GCM Ensemble\n"
    f"(number of models per country)",
    fontsize=14, fontweight="bold")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, ticks=range(0, max_count + 1))
cbar.set_label("Number of GCMs")

out_path = os.path.join(OUTPUT_DIR, "gcm_origin_world_map.png")
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"\nSaved {out_path}")
