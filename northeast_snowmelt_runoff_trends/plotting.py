"""
plotting.py
===========
Map plots for the S/P-ratio climatology, per the analysis-meeting
request: three map types (total snow, total precip, S/P ratio) for
every climatology period/window, plotted for the ensemble mean AND
every individual model.
"""

import numpy as np
import matplotlib.pyplot as plt

import config


def _clean_label(*parts):
    return "_".join(str(p) for p in parts).replace(" ", "_").replace("/", "-")


def _plot_map(ax, da, title, cmap, vmin=None, vmax=None, cbar_label=""):
    im = da.plot.pcolormesh(
        ax=ax, x="lon", y="lat", cmap=cmap, vmin=vmin, vmax=vmax,
        add_colorbar=True, cbar_kwargs={"label": cbar_label, "shrink": 0.8},
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    return im


def plot_climatology_maps(clim_ds, label, window, period_label, out_dir=None):
    """
    One figure, three panels, for a single model (or 'Ensemble') / period
    / window combination: total snowfall, total precipitation, S/P ratio.
    """
    out_dir = out_dir or config.PLOTS_DIR
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    _plot_map(axes[0], clim_ds["snow_clim"], "Total snowfall (mm)", "Blues", cbar_label="mm")
    _plot_map(axes[1], clim_ds["precip_clim"], "Total precipitation (mm)", "YlGnBu", cbar_label="mm")
    _plot_map(axes[2], clim_ds["sp_ratio_clim"], "Snow / Precip ratio", "coolwarm",
               vmin=0, vmax=1, cbar_label="ratio")

    fig.suptitle(f"{label} \u2014 {window.replace('_', ' ')} climatology, {period_label}", fontsize=13)

    out_path = out_dir / f"{_clean_label(label, window, period_label)}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all_models_grid(per_model, ensemble, window, period_label, var="sp_ratio_clim",
                          out_dir=None, cmap="coolwarm", vmin=0, vmax=1,
                          var_label="Snow/Precip ratio"):
    """
    One figure, one panel per model plus the ensemble mean, for a single
    variable (default S/P ratio) -- lets you eyeball inter-model spread
    at a glance. Call once per variable if you want all three (snow,
    precip, ratio) as separate multi-model grids.
    """
    out_dir = out_dir or config.PLOTS_DIR
    models = list(per_model) + ["Ensemble"]
    n = len(models)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(models):
        ds = ensemble if name == "Ensemble" else per_model[name]
        _plot_map(axes[i], ds[var], name, cmap, vmin=vmin, vmax=vmax)
    for j in range(len(models), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{var_label} by model \u2014 {window.replace('_', ' ')}, {period_label}", fontsize=13)
    out_path = out_dir / f"all_models_{_clean_label(var, window, period_label)}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all_three_model_grids(per_model, ensemble, window, period_label, out_dir=None):
    """Convenience: the three requested multi-model comparison grids at once."""
    paths = []
    paths.append(plot_all_models_grid(per_model, ensemble, window, period_label, var="snow_clim",
                                       out_dir=out_dir, cmap="Blues", vmin=None, vmax=None,
                                       var_label="Total snowfall (mm)"))
    paths.append(plot_all_models_grid(per_model, ensemble, window, period_label, var="precip_clim",
                                       out_dir=out_dir, cmap="YlGnBu", vmin=None, vmax=None,
                                       var_label="Total precipitation (mm)"))
    paths.append(plot_all_models_grid(per_model, ensemble, window, period_label, var="sp_ratio_clim",
                                       out_dir=out_dir, cmap="coolwarm", vmin=0, vmax=1,
                                       var_label="Snow/Precip ratio"))
    return paths
