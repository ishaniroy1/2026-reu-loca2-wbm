"""
huc8_zonal.py
=============
Aggregate gridded (lat/lon) daily or annual fields to HUC8 watershed
polygons using area-weighted overlap fractions. Uses `xagg`, which
precomputes the pixel-to-polygon overlap weights once and then reuses
them for every timestep/variable -- much faster than per-timestep
rasterstats zonal stats when you have 30-120 years of daily data.

Install: pip install xagg geopandas
"""

import geopandas as gpd
import pandas as pd
import xarray as xr
import xagg as xa

import config


def load_huc8(shapefile=None):
    """Load HUC8 polygons, reproject to EPSG:4326 to match climate grids."""
    shapefile = shapefile or config.HUC8_SHAPEFILE
    gdf = gpd.read_file(shapefile)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[[config.HUC8_ID_FIELD, config.HUC8_NAME_FIELD, "geometry"]].rename(
        columns={config.HUC8_ID_FIELD: "huc8", config.HUC8_NAME_FIELD: "name"}
    )
    return gdf


def build_overlap_weights(ds, gdf):
    """
    Precompute grid-cell -> HUC8 area-weight overlaps. `ds` just needs to
    carry the correct lat/lon coordinates/grid; any variable will do as
    the weightmap target.
    """
    weightmap = xa.pixel_overlaps(ds, gdf)
    return weightmap


def aggregate_to_huc8(ds, weightmap, gdf, variables):
    """
    Apply precomputed overlap weights to produce a per-HUC8, per-timestep
    (or static) area-weighted mean for each requested variable.

    Returns a tidy pandas DataFrame with columns:
        huc8, name, (time if present), <variable columns...>
    """
    aggregated = xa.aggregate(ds[list(variables)], weightmap)
    df = aggregated.to_dataframe()

    # xagg's output index/columns naming varies by version; normalize here
    if "poly_idx" in df.columns:
        df = df.rename(columns={"poly_idx": "_idx"})
        id_lookup = gdf.reset_index().rename(columns={"index": "_idx"})[
            ["_idx", "huc8", "name"]
        ]
        df = df.merge(id_lookup, on="_idx", how="left").drop(columns="_idx")
    return df


def huc8_timeseries(ds, variables, shapefile=None):
    """
    High-level convenience function: load HUC8 polygons, build overlap
    weights against `ds`'s grid, and return an area-weighted daily/annual
    HUC8-level timeseries dataframe for the given variables.
    """
    gdf = load_huc8(shapefile)
    weightmap = build_overlap_weights(ds, gdf)
    return aggregate_to_huc8(ds, weightmap, gdf, variables)
