# Diffusion (ALR) algorithm adapted from:
#   BigBendNP/nightskyquality (https://github.com/BigBendNP/nightskyquality)
#   Written by Katy Abbott, Geoscientists-in-the-Parks intern at Big Bend NP, 2019-2020.
#   Licensed under the repository's LICENSE.txt (see above repo).
#
# Adaptation uses scipy/numpy in place of astropy/gdal and operates on
# xarray DataArrays rather than raw GDAl datasets.

import numpy as np
from scipy.signal import fftconvolve
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401 – registers .rio accessor


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def circular_annulus_footprint(radius_inner: float, radius_outer: float) -> np.ndarray:
    """Return a 2-D boolean kernel representing a filled annulus.

    Parameters
    ----------
    radius_inner:
        Inner radius in pixels.  Use a small epsilon (e.g. 0.001) for the
        innermost ring so the central pixel is excluded.
    radius_outer:
        Outer radius in pixels.

    Returns
    -------
    numpy.ndarray
        Integer (0/1) array of shape ``(2*radius_outer+1, 2*radius_outer+1)``.
    """
    if radius_inner > radius_outer:
        raise ValueError("radius_outer must be >= radius_inner")
    size = int(radius_outer) * 2 + 1
    cy, cx = radius_outer, radius_outer
    y, x = np.ogrid[:size, :size]
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    kernel = (dist2 >= radius_inner ** 2) & (dist2 <= radius_outer ** 2)
    return kernel.astype(np.int32)


# ---------------------------------------------------------------------------
# ALR calculation
# ---------------------------------------------------------------------------

def calculate_alr(
    radiance_array: np.ndarray,
    pixel_size_km: float,
    n_rings: int = 38,
    max_radius_km: float = 300.0,
    calibration_constant: float = 562.72,
    min_radiance: float = 0.5,
) -> np.ndarray:
    """Calculate the All-sky Light-pollution Ratio (ALR) for every pixel.

    Algorithm originally described in Duriscoe et al. (2018):
    "A simplified model of all-sky artificial sky glow derived from VIIRS
    Day/Night Band data."  Implementation follows the annular-convolution
    approach in BigBendNP/nightskyquality by Katy Abbott.

    Parameters
    ----------
    radiance_array:
        2-D float array of upward radiance values (nW·cm⁻²·sr⁻¹).
        NaN where no data.
    pixel_size_km:
        Side length of a single pixel in **kilometres**.
    n_rings:
        Number of annular rings used for the summation.  More rings → higher
        accuracy but longer runtime.  Default 38 (matches original script).
    max_radius_km:
        Maximum radius to consider, in km.  Default 300 km per Duriscoe.
    calibration_constant:
        Divisor to relate model output to observed ALR.
        Default 562.72 from Duriscoe et al. (2018).
    min_radiance:
        Radiance values below this threshold are set to zero before
        calculation (per Duriscoe et al.).  Default 0.5.

    Returns
    -------
    numpy.ndarray
        ALR array of same shape as *radiance_array*.
    """
    arr = radiance_array.copy().astype(float)

    # Zero-out sub-threshold values (keep NaN)
    valid = ~np.isnan(arr)
    arr[valid & (arr < min_radiance)] = 0.0

    # Replace NaN with 0 for convolution (preserve_nan handled below)
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, 0.0, arr)

    # --- distance conversion helpers ---
    def cell2dist(cells):
        return cells * pixel_size_km

    def dist2cell(km):
        return km / pixel_size_km

    def alpha(d_km):
        """Duriscoe distance-weighting exponent."""
        return 2.3 * (d_km / 350.0) ** 0.28

    # Annulus ring boundaries in cell units
    annulus_cells = np.linspace(0, dist2cell(max_radius_km), n_rings + 1)

    # Average radius (km) for each ring – used for distance weighting
    ring_radii_km = 0.5 * (cell2dist(annulus_cells[:-1]) + cell2dist(annulus_cells[1:]))
    # Avoid d=0 for the innermost ring
    ring_radii_km[ring_radii_km == 0] = pixel_size_km / 2.0

    alr = np.zeros_like(arr_filled)
    eps = 0.001  # exclude central pixel from innermost ring

    for i in range(n_rings):
        r_inner = annulus_cells[i] + eps if i == 0 else annulus_cells[i]
        r_outer = annulus_cells[i + 1]
        kernel = circular_annulus_footprint(r_inner, r_outer)
        if kernel.sum() == 0:
            continue
        d = ring_radii_km[i]
        weight = d ** (-alpha(d))
        convolved = fftconvolve(arr_filled, kernel, mode="same")
        alr += convolved * weight

    alr /= calibration_constant
    alr[nan_mask] = np.nan
    return alr


# ---------------------------------------------------------------------------
# High-level pipeline helper
# ---------------------------------------------------------------------------

def calculate_alr_for_region(
    viirs_da: xr.DataArray,
    region_gdf: gpd.GeoDataFrame,
    albers_epsg: int,
    target_res_m: float = 450.0,
    n_rings: int = 38,
) -> xr.DataArray:
    """End-to-end ALR pipeline for a buffered region of interest.

    Steps
    -----
    1. Reproject *region_gdf* to *albers_epsg* and create a 300 km buffer.
    2. Clip *viirs_da* to the buffer extent and reproject to equal-area Albers.
    3. Run :func:`calculate_alr`.
    4. Clip result back to the original (un-buffered) region boundary.

    Parameters
    ----------
    viirs_da:
        VIIRS DNB radiance DataArray (already opened with rioxarray).
    region_gdf:
        GeoDataFrame of the region of interest (un-buffered).
    albers_epsg:
        EPSG code for an equal-area Albers projection appropriate for the
        region (units **must** be metres).
    target_res_m:
        Target pixel resolution in metres after reprojection.  Default 450 m.
    n_rings:
        Number of annular rings for ALR convolution.

    Returns
    -------
    xarray.DataArray
        ALR raster clipped to *region_gdf*, in the Albers CRS.
    """
    buffer_m = 300_000  # 300 km in metres

    # --- 1. Build buffered ROI in equal-area projection ---
    region_albers = region_gdf.to_crs(epsg=albers_epsg)
    buffer_albers = gpd.GeoDataFrame(
        geometry=[region_albers.union_all().buffer(buffer_m)],
        crs=region_albers.crs,
    )

    # --- 2. Clip VIIRS to buffer bbox, reproject to Albers ---
    viirs_src_crs = viirs_da.rio.crs
    buffer_src_crs = buffer_albers.to_crs(viirs_src_crs)

    # Clip to bounding box first (fast), then reproject
    minx, miny, maxx, maxy = buffer_src_crs.total_bounds
    viirs_clipped = viirs_da.rio.clip_box(minx, miny, maxx, maxy)
    viirs_albers = viirs_clipped.rio.reproject(
        f"EPSG:{albers_epsg}",
        resolution=target_res_m,
        nodata=np.nan,
    )

    # Clip to actual buffer polygon
    viirs_buffered = viirs_albers.rio.clip(
        buffer_albers.geometry, all_touched=True, drop=True
    )

    # --- 3. Run ALR calculation ---
    pixel_size_km = target_res_m / 1000.0
    arr = viirs_buffered.values.squeeze().astype(float)
    arr[arr < 0] = np.nan  # remove negative fill values

    alr_arr = calculate_alr(arr, pixel_size_km=pixel_size_km, n_rings=n_rings)

    # Re-wrap as DataArray with same spatial metadata
    alr_da = viirs_buffered.copy(data=alr_arr[np.newaxis])
    alr_da.attrs["long_name"] = "All-sky Light Pollution Ratio (ALR)"
    alr_da.attrs["source"] = (
        "Duriscoe et al. (2018); algorithm ported from "
        "https://github.com/BigBendNP/nightskyquality"
    )

    # --- 4. Clip to original (un-buffered) ROI ---
    alr_clipped = alr_da.rio.clip(
        region_albers.geometry, all_touched=True, drop=True
    )
    return alr_clipped
