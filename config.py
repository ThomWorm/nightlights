

# config.py
from pathlib import Path

DATA_DIR = Path("data")
STATES_SHP = DATA_DIR / "shapefiles/states/tl_2024_us_state.shp"
COUNTIES_SHP = DATA_DIR / "shapefiles/counties/tl_2024_us_county.shp"
RADIANCE_OLDER = DATA_DIR / "VIIRS/2016/VNL_v21_npp_2016_global_vcmslcfg_c202205302300.average_masked.dat.tif"
RADIANCE_NEWER = DATA_DIR / "VIIRS/2021/VNL_v21_npp_2021_global_vcmslcfg_c202205302300.average_masked.dat.tif"
IMPERV_NEWER = DATA_DIR / "NLCD/Annual_NLCD_FctImp_2021_CU_C1V0.tif"
IMPERV_OLDER = DATA_DIR / "NLCD/Annual_NLCD_FctImp_2016_CU_C1V1.tif"

OUTPUT_DIR = DATA_DIR / "processed_states"