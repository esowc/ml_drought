import sys
from pathlib import Path
import numpy as np
from typing import Optional, Union, List

sys.path.append("..")
from src.preprocess import (
    VHIPreprocessor,
    CHIRPSPreprocessor,
    PlanetOSPreprocessor,
    GLEAMPreprocessor,
    S5Preprocessor,
    ESACCIPreprocessor,
    SRTMPreprocessor,
    ERA5MonthlyMeanPreprocessor,
    ERA5HourlyPreprocessor,
    BokuNDVIPreprocessor,
    KenyaASALMask,
    ERA5LandPreprocessor,
    ERA5LandMonthlyMeansPreprocessor,
)

from src.preprocess.admin_boundaries import KenyaAdminPreprocessor

from scripts.utils import get_data_path


def process_vci(subset_str: str = "kenya"):
    data_path = get_data_path()
    processor = VHIPreprocessor(get_data_path(), "VCI")
    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor.preprocess(
        subset_str=subset_str, resample_time="M", upsampling=False, regrid=regrid_path
    )


def process_precip_2018(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = CHIRPSPreprocessor(data_path)

    processor.preprocess(subset_str=subset_str, regrid=regrid_path, parallel=False)


def process_era5POS_2018(subset_str: str = "kenya"):
    data_path = get_data_path()
    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = PlanetOSPreprocessor(data_path)

    processor.preprocess(
        subset_str=subset_str,
        regrid=regrid_path,
        parallel=False,
        resample_time="M",
        upsampling=False,
    )


def process_era5_land(
    variables: Optional[Union[List, str]] = None,
    subset_str: str = "kenya",
    monmean: bool = True,
):
    data_path = get_data_path()

    # Check all the provided variables exist
    if variables is None:
        variables = [d.name for d in (data_path / "raw/reanalysis-era5-land").iterdir()]
        assert (
            variables != []
        ), f"Expecting to find some variables in: {(data_path / 'raw/reanalysis-era5-land')}"
    else:
        if isinstance(variables, str):
            variables = [variables]
            assert variables in [
                d.name for d in (data_path / "raw/reanalysis-era5-land").iterdir()
            ], f"Expect to find {variables} in {(data_path / 'raw/reanalysis-era5-land')}"
        else:
            assert all(
                np.isin(
                    variables,
                    [
                        d.name
                        for d in (data_path / "raw/reanalysis-era5-land").iterdir()
                    ],
                )
            ), f"Expected to find {variables}"

    # regrid_path = data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    # assert regrid_path.exists(), f"{regrid_path} not available"
    regrid_path = None

    if monmean:
        processor = ERA5LandMonthlyMeansPreprocessor(data_path)
    else:
        processor = ERA5LandPreprocessor(data_path)

    for variable in variables:
        processor.preprocess(
            subset_str=subset_str,
            regrid=None,
            resample_time="M",
            upsampling=False,
            variable=variable,
        )


def process_gleam(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"


def process_gleam():
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")
    regrid_path = (
        data_path
        / "interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = GLEAMPreprocessor(data_path)

    processor.preprocess(
        subset_str=subset_str, regrid=regrid_path, resample_time="M", upsampling=False
    )


def process_seas5():
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")
    regrid_path = (
        data_path
        / "interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    datasets = [d.name for d in (data_path / "raw").iterdir() if "seasonal" in d.name]
    for dataset in datasets:
        variables = [v.name for v in (data_path / "raw" / dataset).glob("*")]

        for variable in variables:
            if variable == "total_precipitation":
                processor = S5Preprocessor(data_path)
                processor.preprocess(
                    subset_str="kenya",
                    regrid=regrid_path,
                    resample_time=None,
                    upsampling=False,
                    variable=variable,
                )


def process_esa_cci_landcover():
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")
    regrid_path = (
        data_path
        / "interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"
    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(subset_str=subset_str, regrid=regrid_path)


def preprocess_srtm(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    print(
        "Warning: regridding with CDO using the VCI preprocessor data fails because"
        "CDO reads the grid type as generic instead of latlon. This can be fixed "
        "just by changing the grid type to latlon in the grid definition file."
    )

    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str=subset_str, regrid=regrid_path)


def preprocess_kenya_boundaries(selection: str = "level_1"):
    assert selection in [
        f"level_{i}" for i in range(1, 6)
    ], f'selection must be one of {[f"level_{i}" for i in range(1,6)]}'

    data_path = get_data_path()

    regrid_path = data_path / "interim/chirps_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = KenyaAdminPreprocessor(data_path)
    processor.preprocess(reference_nc_filepath=regrid_path, selection=selection)


def preprocess_asal_mask():
    data_path = get_data_path()

    regrid_path = data_path / "interim/chirps_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = KenyaASALMask(data_path)
    processor.preprocess(reference_nc_filepath=regrid_path)


def preprocess_era5(subset_str: str = "kenya"):
    data_path = get_data_path()

    # regrid_path = data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    # assert regrid_path.exists(), f"{regrid_path} not available"
    regrid_path = None

    processor = ERA5MonthlyMeanPreprocessor(data_path)
    processor.preprocess(subset_str=subset_str, regrid=regrid_path)


def preprocess_era5_hourly(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = ERA5HourlyPreprocessor(data_path)

    # W-MON is weekly each monday (the same as the NDVI data from Atzberger)
    processor.preprocess(subset_str=subset_str, resample_time="W-MON")
    # processor.merge_files(subset_str='W-MON')


def preprocess_boku_ndvi(subset_str: str = "kenya"):
    data_path = get_data_path()
    processor = BokuNDVIPreprocessor(data_path)

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor.preprocess(
        subset_str=subset_str, resample_time="W-MON", regrid=regrid_path
    )


def preprocess_era5_hourly():
    data_path = get_data_path()

    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = ERA5HourlyPreprocessor(data_path)

    # W-MON is weekly each monday (the same as the NDVI data from Atzberger)
    # processor.preprocess(subset_str="kenya", resample_time="W-MON")
    processor.merge_files(subset_str="W-MON")

    
def preprocess_s5_ouce():
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")
    variable = "total_precipitation"
    daily_s5_dir = Path("/soge-home/data/model/seas5/1.0x1.0/daily")
    s = S5Preprocessor(data_path, ouce_server=True)
    s.preprocess(
        variable=variable,
        regrid=None,
        resample_time=None,
        **{"ouce_dir": daily_s5_dir, "infer": True},
    )


if __name__ == "__main__":
    subset_str = "india"
    # preprocess_era5(subset_str=subset_str)
    process_era5_land(
        subset_str=subset_str,
        variables=[
            "volumetric_soil_water_layer_1",
            "potential_evaporation",
        ],  #  total_precipitation 2m_temperature evapotranspiration
        monmean=False,
    )
    # process_vci(subset_str=subset_str)
    # process_precip_2018(subset_str=subset_str)
    # process_era5POS_2018(subset_str=subset_str)
    # process_gleam(subset_str=subset_str)
    # process_esa_cci_landcover(subset_str=subset_str)
    # preprocess_srtm(subset_str=subset_str)
    # preprocess_kenya_boundaries(selection="level_1")
    # preprocess_kenya_boundaries(selection="level_2")
    # preprocess_kenya_boundaries(selection="level_3")
    # preprocess_era5_hourly(subset_str=subset_str)
    # preprocess_boku_ndvi(subset_str=subset_str)
    # preprocess_asal_mask(subset_str=subset_str)
