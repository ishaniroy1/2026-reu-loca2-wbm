"""
snowpack_model.py
==================
A simple temperature-index (degree-day) snowpack model, used to derive a
daily snowmelt flux from downscaled CMIP6 temperature/precipitation for
periods and scenarios where no direct snowpack observation (SNODAS) exists.

For the *historical* period, prefer the observed SNODAS `snowMelt`
variable directly (see io_utils.load_snodas) -- this model exists so the
same accounting can be extended consistently into the future, and so it
can be run on the historical period too as a sanity check against SNODAS.

Model (per grid cell, per day):

    SWE(t) = SWE(t-1) + snow(t) - melt(t)
    melt(t) = min(SWE(t-1) + snow(t),
                   DDF * max(tas(t) - T_melt, 0))     [mm/day]

where DDF is the degree-day melt factor (mm / degC / day) and T_melt is
the melt-onset temperature threshold (degC), both configurable below.
Rain falling on an existing snowpack is assumed to pass through and
contribute to melt-day runoff but is not tracked separately here (a
common simplification for this kind of screening-level analysis).
"""

import numpy as np
import xarray as xr

# Degree-day factor: literature range for the NE US mixed forest/mixed
# open terrain is roughly 2-4 mm/degC/day; 3.0 is a reasonable default.
DEGREE_DAY_FACTOR = 3.0    # mm / degC / day
T_MELT = 0.0                # degC, melt-onset threshold


def simulate_snowpack(ds, tas_var="tas", snow_var="snow", ddf=DEGREE_DAY_FACTOR,
                       t_melt=T_MELT):
    """
    Run the degree-day snowpack model over a (time, lat, lon) dataset that
    already has daily mean temperature and partitioned snowfall (from
    phase_partition.partition). Must be time-sorted and gap-free within
    each water year (small gaps are treated as zero accumulation/melt).

    Returns the input dataset with two new variables added:
      swe          -- simulated snow water equivalent (mm)
      snowmelt_sim -- simulated daily melt flux (mm/day)
    """
    tas = ds[tas_var].values
    snow = ds[snow_var].values
    n_time = tas.shape[0]

    swe = np.zeros_like(tas)
    melt = np.zeros_like(tas)

    swe_prev = np.zeros(tas.shape[1:], dtype=np.float64)

    potential_melt_coeff = ddf
    for t in range(n_time):
        available = swe_prev + snow[t]
        potential_melt = potential_melt_coeff * np.clip(tas[t] - t_melt, 0, None)
        m = np.minimum(available, potential_melt)
        new_swe = available - m

        swe[t] = new_swe
        melt[t] = m
        swe_prev = new_swe

    out = ds.copy()
    out["swe"] = (ds[tas_var].dims, swe)
    out["snowmelt_sim"] = (ds[tas_var].dims, melt)
    out["swe"].attrs["units"] = "mm"
    out["snowmelt_sim"].attrs["units"] = "mm/day"
    out["snowmelt_sim"].attrs["description"] = (
        f"Degree-day model melt flux, DDF={ddf} mm/degC/day, "
        f"T_melt={t_melt} degC"
    )
    return out
