import rioxarray

# from rasterio.errors import NoDataInBounds
import numpy as np
from rasterio.warp import reproject, Resampling
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd


import numpy as np
import xarray as xr


def aggregate_imp_matrix(data_smaller, data_larger, thresholds=(10, 30, 50, 75), mean_only=False):
    """
    Aggregate fine-resolution imperviousness (0–100) to match coarser VIIRS grid,
    using Rasterio/GDAL resampling with built-in mean, max, and rms.
    """
    if mean_only:
        mean_agg = reproject_resample(
            data_smaller, data_larger, resampling=Resampling.average
        )
        ds = xr.Dataset({"mean": mean_agg})
        ds = ds.rio.write_crs(data_larger.rio.crs)
        ds = ds.rio.write_transform(data_larger.rio.transform(), inplace=False) 
        return ds

    else:
        # Step 2: Use rasterio/GDAL resampling for each desired aggregation
        mean_agg = data_smaller.rio.reproject_match(
            data_larger, resampling=Resampling.average
        )

        # RMS (if GDAL >= 3.3)
        rms_agg = data_smaller.rio.reproject_match(
            data_larger, resampling=Resampling.rms
        )

        # Max
        max_agg = data_smaller.rio.reproject_match(
            data_larger, resampling=Resampling.max
        )
        sum_agg = data_smaller.rio.reproject_match(
            data_larger, resampling=Resampling.sum
        )
        cubic_agg = data_smaller.rio.reproject_match(
            data_larger, resampling=Resampling.cubic
        )
        lanczos_agg = data_smaller.rio.reproject_match(
            data_larger, resampling=Resampling.lanczos
        )

        ds = xr.Dataset({
            "mean": mean_agg,
            "rms": rms_agg,
            "max": max_agg,
            "sum": sum_agg,
            "cubic": cubic_agg,
            "lanczos": lanczos_agg
        })

        # Fractions above thresholds
        for t in thresholds:
            mask = (data_smaller > t).astype("float32")
            frac = mask.rio.reproject_match(
                data_larger, resampling=Resampling.average
            )
            ds[f"frac_gt_{t}"] = frac

        # Ensure metadata consistent with data_larger
        ds = ds.rio.write_crs(data_larger.rio.crs)
        ds = ds.rio.write_transform(data_larger.rio.transform(), inplace=False)

        return ds


def reproject_resample(data_smaller, data_larger, resampling=Resampling.average):
    """
    Reproject/resample data_smaller onto the grid of data_larger
    using GDAL's built-in resampling methods.

    Parameters
    ----------
    data_smaller : xarray.DataArray
        Higher-resolution input raster with .rio.crs and .rio.transform().
    data_larger : xarray.DataArray
        Target raster whose CRS/grid to match.
    resampling : rasterio.warp.Resampling
        Resampling method (default=average).

    Returns
    -------
    xarray.DataArray
        Reprojected and resampled version of data_smaller
        aligned to data_larger.
    """
    dst_height, dst_width = data_larger.shape
    out = np.empty((dst_height, dst_width), dtype=data_smaller.dtype)

    reproject(
        source=data_smaller.values,
        destination=out,
        src_transform=data_smaller.rio.transform(),
        src_crs=data_smaller.rio.crs,
        dst_transform=data_larger.rio.transform(),
        dst_crs=data_larger.rio.crs,
        dst_width=dst_width,
        dst_height=dst_height,
        resampling=resampling,
    )

    return data_larger.copy(data=out)


def summarize_by_impervious_bins(df, state_name, variable="mean", n_bins=20):
    """
    Summarize VIIRS brightness (response) by imperviousness bins (predictor).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'viirs' and the imperviousness variable (e.g. 'mean').
    state_name : str
        State name for labeling output.
    variable : str, default 'mean'
        Imperviousness variable name to bin by.
    n_bins : int, default 20
        Number of bins (equal-width bins from 0–100, not percentiles).
    """
    # Drop missing or invalid data
    df = df[["viirs", variable]].dropna()

    # Define bin edges and labels (e.g. 0–5, 5–10, …)
    bin_edges = np.linspace(0, 100, n_bins + 1)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(n_bins)]

    # Clip out-of-range imperviousness (some NLCD pixels can be >100 due to rounding)
    df = df[(df[variable] >= 0) & (df[variable] <= 100)]

    # Bin by imperviousness
    df["imp_bin"] = pd.cut(df[variable], bins=bin_edges, labels=bin_labels, include_lowest=True)

    # Aggregate VIIRS brightness by bin
    summary = (
        df.groupby("imp_bin")["viirs"]
          .agg(["count", "mean", "std", "min", "max"])
          .reset_index()
    )

    # Add metadata
    summary["state"] = state_name
    summary["predictor"] = variable

    return summary


def custom_aggregate(data_smaller, data_larger, threshold=10):
    """
    Aggregate data_smaller onto the grid of data_larger with custom logic.
    Example: count number of pixels above a threshold.

    Parameters
    ----------
    data_smaller : xarray.DataArray
        Higher-resolution raster.
    data_larger : xarray.DataArray
        Target raster whose grid/CRS to match.
    threshold : float
        Threshold for counting.

    Returns
    -------
    xarray.DataArray
        Aggregated raster aligned to data_larger.
    """
    mask = (data_smaller > threshold).astype("int16")

    scale_y = int(data_smaller.sizes["y"] / data_larger.sizes["y"])
    scale_x = int(data_smaller.sizes["x"] / data_larger.sizes["x"])

    counts = mask.coarsen(y=scale_y, x=scale_x, boundary="trim").sum()

    if data_smaller.rio.crs != data_larger.rio.crs:
        counts = counts.rio.reproject_match(data_larger)
    else:
        # Align coords if grid spacing differs but CRS is the same
        counts = counts.rio.reproject_match(data_larger)

    return counts


def clip_raster_safely(filename, reprojected_roi_gdf):
    """Clips a large raster using a pre-reprojected shapefile."""
    # this function is safer because it handles NoDataInBounds exception
    with rioxarray.open_rasterio(filename, chunks="auto", masked=True) as src:
        if "_FillValue" in src.attrs:
            del src.attrs["_FillValue"]
        # The function now assumes the CRS of the roi_gdf matches the src.
        try:
            clipped_to_box = src.rio.clip_box(*reprojected_roi_gdf.total_bounds)
            final_clip = clipped_to_box.rio.clip(
                reprojected_roi_gdf.geometry, drop=True
            )
            return final_clip
        except Exception as e:
            # Broad except because rasterio's NoDataInBounds may not be imported here.
            print(f"Warning: could not clip {filename}. Exception: {e}")
            return None


def process_state_relationship(state_name,
                               viirs_raster_path,
                               nlcd_imp_raster_path,
                               nlcd_desc_raster_path,
                               tiger_state_shapefile,
                               tiger_county_shapefile=None,
                               county_name=None,
                               use_county_mask=False,
                               impervious_threshold=50,
                               thresholds=None):
    """
    Process a state's rasters and return a pandas DataFrame with one row per VIIRS pixel.

    Parameters
    ----------
    state_name : str
        Name of the state to process (must match shapefile 'NAME' field).
    viirs_raster_path : str
        Path to the VIIRS raster used as the target grid.
    nlcd_imp_raster_path : str
        Path to NLCD imperviousness raster.
    nlcd_desc_raster_path : str
        Path to NLCD descriptor raster (for masking roads, etc.).
    tiger_state_shapefile : str
        Path to US states shapefile.
    tiger_county_shapefile : str or None
        Optional counties shapefile used when use_county_mask is True.
    county_name : str or None
        Name of the county to restrict to when use_county_mask is True.
    use_county_mask : bool
        Whether to mask to a specific county within the state.
    impervious_threshold : float
        Threshold for "highly impervious" when building binary masks (unused if thresholds provided).
    thresholds : tuple or None
        If provided (non-empty), pass through to aggregation function. If None or empty, aggregate_imp_matrix
        will return only the mean variable as requested.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for aggregated predictors and the VIIRS radiance value for every pixel in the ROI.
    """
    # --- Load shapes and build ROI ---
    states_gdf = gpd.read_file(tiger_state_shapefile)
    viirs_temp = rioxarray.open_rasterio(viirs_raster_path, chunks="auto")
    viirs_crs = viirs_temp.rio.crs

    state_roi = states_gdf[states_gdf["NAME"] == state_name].to_crs(viirs_crs)
    if state_roi.empty:
        raise ValueError(f"State '{state_name}' not found in {tiger_state_shapefile}")

    if use_county_mask and tiger_county_shapefile is not None and county_name is not None:
        county_gdf = gpd.read_file(tiger_county_shapefile)
        roi_counties = county_gdf[county_gdf["STATEFP"] == state_roi.iloc[0]["STATEFP"]]
        county_roi = roi_counties[roi_counties["NAME"] == county_name].to_crs(viirs_crs)
        roi = county_roi
    else:
        roi = state_roi

    # --- Reproject ROI to NLCD CRS once for clipping high-res rasters ---
    with rioxarray.open_rasterio(nlcd_imp_raster_path, masked=True) as src:
        if "_FillValue" in src.attrs:
            del src.attrs["_FillValue"]
        nlcd_crs = src.rio.crs

    roi_reproj_nlcd = roi.to_crs(nlcd_crs)

    # --- Clip NLCD rasters ---
    impervious_ds = clip_raster_safely(nlcd_imp_raster_path, roi_reproj_nlcd)
    descriptor_ds = clip_raster_safely(nlcd_desc_raster_path, roi_reproj_nlcd)
    if impervious_ds is None or descriptor_ds is None:
        raise ValueError("NLCD clipping returned no data; check inputs and ROI.")

    # Mask descriptor values of 250 to NaN and remove roads (descriptor == 1)
    impervious_ds = impervious_ds.where(impervious_ds != 250)
    descriptor_ds = descriptor_ds.where(descriptor_ds != 250)
    impervious_no_roads = impervious_ds.where(descriptor_ds != 1)

    # --- Clip VIIRS to ROI in VIIRS CRS ---
    viirs_ds = clip_raster_safely(viirs_raster_path, roi.to_crs(viirs_crs))
    if viirs_ds is None:
        raise ValueError("VIIRS clipping returned no data; check inputs and ROI.")

    # --- Aggregate NLCD to VIIRS grid ---
    # If thresholds is None or empty, aggregate_imp_matrix will return only mean
    aggregated = aggregate_imp_matrix(impervious_ds, viirs_ds, thresholds=thresholds)
    aggregated_no_roads = aggregate_imp_matrix(impervious_no_roads, viirs_ds, thresholds=thresholds)

    # --- Build pandas DataFrame ---
    df = pd.DataFrame()
    for var in aggregated.data_vars:
        df[var] = aggregated[var].values.ravel()

    df["viirs"] = viirs_ds.values.ravel()

    # Drop rows with NaNs
    df = df.dropna()

    # Add state label
    df["state"] = state_name

    return df
