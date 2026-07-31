"""
cdo_reduce.py
=============
CDO-based preprocessing that reduces the full daily LOCA2-WBM output
(tens of thousands of daily timesteps across 1980-2100, times ~5 GCMs x
3 SSPs) down to small yearly water-year and cold-season SUM files,
cached to disk.

WHY THIS EXISTS: the original approach summed daily precip/snowfall into
water-year totals inside xarray/dask (phase_partition.water_year_sp_ratio).
That computation is *lazy* -- nothing actually runs until something
downstream forces it, which turned out to be xagg's aggregation step.
So every pipeline run was silently re-reading and re-summing the entire
multi-decade daily record right at "aggregating sp_ratio...", with no
progress feedback, which is why it looked stuck.

CDO does the same reduction in compiled C -- an order of magnitude
faster -- and every output here is a tiny netCDF (one value per grid
cell per year) that's instant to reload. Each function is idempotent: if
the cached output already exists and is newer than its inputs, it's
returned immediately without recomputing anything. Delete config.CDO_CACHE_DIR
to force a full rebuild.

Requires the `cdo` command-line tool on PATH (not the Python `cdo`
bindings package -- this module shells out directly via subprocess).
"""

import logging
import shutil
import subprocess
from pathlib import Path

import config

log = logging.getLogger(__name__)


def _cdo_available():
    return shutil.which("cdo") is not None


def _run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"CDO command failed (exit code {result.returncode}):\n"
            f"  {' '.join(cmd)}\n"
            f"--- output ---\n{result.stdout}"
        )


def _run_atomic(cmd, out_file):
    """
    Run a CDO command whose LAST argument is the desired final output
    path, but actually write to a hidden temp file and atomically rename
    it into place only after the command succeeds. This guarantees a
    cache entry at `out_file` is never left corrupt/truncated if the
    process is killed (Ctrl+C) or the command fails partway through --
    a half-written file at the real cache path would otherwise pass the
    mtime-based _is_stale() check on the next run and be silently reused
    as if it were valid.
    """
    out_file = Path(out_file)
    tmp_out = out_file.with_name(f".{out_file.name}.partial")
    cmd = list(cmd[:-1]) + [str(tmp_out)]
    try:
        _run(cmd)
        tmp_out.replace(out_file)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()


def _is_stale(output_path, input_paths):
    """True if output doesn't exist or is older than any input file."""
    output_path = Path(output_path)
    if not output_path.exists():
        return True
    out_mtime = output_path.stat().st_mtime
    return any(Path(p).stat().st_mtime > out_mtime for p in input_paths)


def _has_data(path):
    """
    Cheap sanity check that a cached netCDF actually contains timesteps
    (catches corrupt/empty files -- e.g. leftovers from before atomic
    writes were added here, or any other partial-write edge case that
    predates a given cache entry). A file can pass the mtime-based
    _is_stale() check and still be garbage, so cache hits are gated on
    both checks together (see _cache_ok).
    """
    result = subprocess.run(["cdo", "-s", "-ntime", str(path)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        return False
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return False
    try:
        return int(lines[-1]) > 0
    except ValueError:
        return False


def _cache_ok(output_path, input_paths):
    """True only if output exists, is newer than every input, AND
    actually contains data -- the combined check every cache-hit
    decision in this module should use."""
    if _is_stale(output_path, input_paths):
        return False
    if not _has_data(output_path):
        log.warning(f"    [stale cache] {Path(output_path).name} exists but is empty/corrupt "
                     f"-- rebuilding")
        return False
    return True


def _require_cdo():
    if not _cdo_available():
        raise RuntimeError(
            "cdo not found on PATH. Install it (e.g. `conda install -c conda-forge cdo` "
            "or `apt install cdo`) to use the CDO-cached reduction pipeline."
        )


def _daily_files(gcm_run_dir):
    gcm_run_dir = Path(gcm_run_dir)
    files = sorted((gcm_run_dir / config.DAILY_SUBDIR).glob(config.DAILY_FILE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No daily files found at {gcm_run_dir / config.DAILY_SUBDIR / config.DAILY_FILE_GLOB}"
        )
    return files


def get_merged(gcm_run_dir, variables, out_dir=None):
    """
    Merge every wbm_{year}.nc file for one GCM run into a single time
    series, keeping only `variables` (much smaller than the full file
    set, and shared across every downstream reduction for this run so
    the expensive mergetime only happens once). Cached.

    IMPORTANT: each yearly file is reduced (variable selection) INDIVIDUALLY
    before merging. Chaining `-selname -mergetime` in a single command
    would apply selname to mergetime's *output* instead of to each input
    file, which forces CDO to merge the full multi-variable, full-grid
    record first -- for an all-variable, multi-decade daily archive
    that's an enormous amount of unnecessary I/O and is almost certainly
    what makes this step look stuck rather than just slow.

    NOTE: an earlier version also tried a `sellonlatbox` spatial
    pre-filter here, but the WBM output uses a custom/projected grid
    definition (grid_mapping="crs") that CDO's sellonlatbox can't
    interpret ("Unsupported grid type: projection"), so that's dropped.
    Variable selection alone (17 vars -> 2) is already the dominant
    volume reduction; the precise Northeast clip still happens in
    io_utils.clip_to_region after loading.
    """
    _require_cdo()
    gcm_run_dir = Path(gcm_run_dir)
    out_dir = Path(out_dir or config.CDO_CACHE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_files = _daily_files(gcm_run_dir)
    tag = gcm_run_dir.name
    varkey = "-".join(variables)
    out_file = out_dir / f"{tag}__{varkey}__merged.nc"

    if _cache_ok(out_file, daily_files):
        log.info(f"    [cache hit] {out_file.name}")
        return out_file

    varlist = ",".join(variables)

    tmp_files = []
    try:
        for i, f in enumerate(daily_files, 1):
            tmp = out_dir / f"_tmp_{tag}_{varkey}_{Path(f).stem}.nc"
            if not _cache_ok(tmp, [f]):
                log.info(f"    reducing {Path(f).name} ({i}/{len(daily_files)}) ...")
                _run_atomic(["cdo", "-O", f"-selname,{varlist}", str(f), str(tmp)], tmp)
            tmp_files.append(tmp)

        log.info(f"    merging {len(tmp_files)} reduced yearly files -> {out_file.name} ...")
        _run_atomic(["cdo", "-O", "mergetime"] + [str(t) for t in tmp_files] + [str(out_file)],
                    out_file)
    finally:
        for t in tmp_files:
            if t.exists():
                t.unlink()

    return out_file


def water_year_sums(gcm_run_dir, varname, out_dir=None):
    """
    Water-year (Oct-Sep) sum of `varname`, one value per grid cell per
    water year, labeled by the ending calendar year (matches the
    io_utils/trends.py water_year() convention elsewhere in the project).
    Cached; skipped if already up to date.
    """
    _require_cdo()
    gcm_run_dir = Path(gcm_run_dir)
    out_dir = Path(out_dir or config.CDO_CACHE_DIR)
    merged = get_merged(gcm_run_dir, [config.VAR_PRECIP_TOTAL, config.VAR_SNOW], out_dir)

    tag = gcm_run_dir.name
    out_file = out_dir / f"{tag}__{varname}__wateryear_sum.nc"
    if _cache_ok(out_file, [merged]):
        log.info(f"    [cache hit] {out_file.name}")
        return out_file

    # shifttime,3mon moves Oct(Y-1)..Sep(Y) into Jan(Y)..Dec(Y), so a plain
    # yearsum afterward groups exactly one water year per calendar-year bin
    _run_atomic(["cdo", "-O", "yearsum", "-shifttime,3mon", f"-selname,{varname}",
                 str(merged), str(out_file)], out_file)
    return out_file


def cold_season_sums(gcm_run_dir, varname, out_dir=None):
    """
    Cold-season (config.COLD_SEASON_MONTHS, default Oct-Apr) sum of
    `varname`, one value per grid cell per season, labeled by the ending
    calendar year (Oct(Y-1)-Apr(Y) -> labeled Y). Cached.
    """
    _require_cdo()
    gcm_run_dir = Path(gcm_run_dir)
    out_dir = Path(out_dir or config.CDO_CACHE_DIR)
    merged = get_merged(gcm_run_dir, [config.VAR_PRECIP_TOTAL, config.VAR_SNOW], out_dir)

    tag = gcm_run_dir.name
    months_str = ",".join(str(m) for m in config.COLD_SEASON_MONTHS)
    out_file = out_dir / f"{tag}__{varname}__coldseason_sum.nc"
    if _cache_ok(out_file, [merged]):
        log.info(f"    [cache hit] {out_file.name}")
        return out_file

    # selmon picks the cold-season months first (leaving May-Sep out
    # entirely), then the same +3mon shift trick lands every cold season
    # inside a single calendar year before yearsum groups them
    _run_atomic(["cdo", "-O", "yearsum", "-shifttime,3mon", f"-selmon,{months_str}",
                 f"-selname,{varname}", str(merged), str(out_file)], out_file)
    return out_file


def get_or_build(gcm_run_dir, varname, window="water_year", out_dir=None):
    """Convenience dispatcher; returns the cached netCDF path for either window."""
    if window == "water_year":
        return water_year_sums(gcm_run_dir, varname, out_dir=out_dir)
    elif window == "cold_season":
        return cold_season_sums(gcm_run_dir, varname, out_dir=out_dir)
    else:
        raise ValueError(f"Unknown window '{window}'; expected 'water_year' or 'cold_season'")
