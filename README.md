# nightlights

Exploratory analysis of night-time light pollution using VIIRS DNB remote-sensing data.

## Notebooks

| Notebook | Description |
|---|---|
| `00_setup.ipynb` | Environment setup and data paths |
| `01_exporation.ipynb` | Initial data exploration (note: filename preserved as-is) |
| `02_brightness_inference_testing.ipynb` | Brightness inference testing |
| `03_inference_spatial_test.ipynb` | Spatial inference testing |
| `04_diffusion_testing.ipynb` | Diffusion algorithm scratch space |
| `04_radiance_comparison.ipynb` | Radiance comparison across years |
| `05_diffusion_testing.ipynb` | **ALR diffusion demo on North Carolina** |

## Diffusion Algorithm (ALR)

`diffusion.py` implements the **All-sky Light-pollution Ratio (ALR)** algorithm
described in:

> Duriscoe, D. M., Anderson, C. B., Duriscoe, E. M., & Eck, K. M. (2018).
> *A simplified model of all-sky artificial sky glow derived from VIIRS Day/Night Band data.*
> Journal of Quantitative Spectroscopy and Radiative Transfer, 212, 1–8.

The implementation is ported from
**[BigBendNP/nightskyquality](https://github.com/BigBendNP/nightskyquality)**,
originally written by **Katy Abbott** as part of the Geoscientists-in-the-Parks
programme at Big Bend National Park (2019–2020), and is used here under the
terms of that repository's [LICENSE](https://github.com/BigBendNP/nightskyquality/blob/master/LICENSE.txt).

Key changes from the original:
- Replaced `astropy.convolution.convolve_fft` with `scipy.signal.fftconvolve`
- Replaced `gdal` / `osr` with `rioxarray` / `geopandas`
- Exposed a high-level `calculate_alr_for_region()` function that handles
  reprojection, buffering, and clipping automatically

## Data

Data files are **not** included in this repository.  Paths are configured in `config.py`:

| Variable | Description |
|---|---|
| `RADIANCE_OLDER` | VIIRS DNB annual composite (2021) – `config.RADIANCE_OLDER` |
| `RADIANCE_NEWER` | VIIRS DNB annual composite (2019) – `config.RADIANCE_NEWER` |
| `STATES_SHP` | US TIGER state boundaries shapefile |
| `COUNTIES_SHP` | US TIGER county boundaries shapefile |
