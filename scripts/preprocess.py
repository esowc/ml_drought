from pathlib import Path

import sys
sys.path.append('..')
from src.preprocess import (VHIPreprocessor, CHIRPSPreprocesser,
                            PlanetOSPreprocessor, GLEAMPreprocessor,
                            S5Preprocessor,
                            ESACCIPreprocessor, SRTMPreprocessor,
                            ERA5MonthlyMeanPreprocessor)

from src.preprocess.admin_boundaries import KenyaAdminPreprocessor


def process_vci_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    processor = VHIPreprocessor(data_path, 'VCI')

    processor.preprocess(subset_str='kenya',
                         resample_time='M', upsampling=False)


def process_precip_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    regrid_path = data_path / 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = CHIRPSPreprocesser(data_path)

    processor.preprocess(subset_str='kenya',
                         regrid=regrid_path,
                         parallel=False)


def process_era5POS_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = PlanetOSPreprocessor(data_path)

    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         parallel=False, resample_time='M', upsampling=False)


def process_gleam():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = GLEAMPreprocessor(data_path)

    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         resample_time='M', upsampling=False)

def process_seas5():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / \
        'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    datasets = [
        d.name
        for d in (data_path / 'raw').iterdir()
        if 'seasonal' in d.name
    ]
    for dataset in datasets:
        variables = [
            v.name for v in (data_path / 'raw' / dataset).glob('*')
        ]

        for variable in variables:
            if variable == 'total_precipitation':
                processor = S5Preprocessor(data_path)
                processor.preprocess(subset_str='kenya', regrid=regrid_path,
                                    resample_time=None, upsampling=False,
                                    variable=variable)


def process_esa_cci_landcover():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


def preprocess_srtm():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / \
        'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    print('Warning: regridding with CDO using the VCI preprocessor data fails because'
          'CDO reads the grid type as generic instead of latlon. This can be fixed '
          'just by changing the grid type to latlon in the grid definition file.')

    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


def preprocess_kenya_boundaries(selection: str = 'level_1'):
    assert selection in [f'level_{i}' for i in range(1,6)], \
        f'selection must be one of {[f"level_{i}" for i in range(1,6)]}'

    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / \
        'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = KenyaAdminPreprocessor(data_path)
    processor.preprocess(
        reference_nc_filepath=regrid_path, selection=selection
    )


def preprocess_era5():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = ERA5MonthlyMeanPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


def preprocess_s5_ouce():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    variable = 'total_precipitation'
    daily_s5_dir = Path('/soge-home/data/model/seas5/1.0x1.0/daily')
    s = S5Preprocessor(data_path, ouce_server=True)
    s.preprocess(
        variable=variable, regrid=None, resample_time=None,
        **{'ouce_dir': daily_s5_dir, 'infer': True},
    )


if __name__ == '__main__':
    # process_vci_2018()
    # process_precip_2018()
    # process_era5POS_2018()
    # process_gleam()
    # process_esa_cci_landcover()
    # preprocess_srtm()
    # preprocess_era5()
    # preprocess_kenya_boundaries(selection='level_1')
    # preprocess_kenya_boundaries(selection='level_2')
    # preprocess_kenya_boundaries(selection='level_3')
    # process_seas5()
    preprocess_s5_ouce()
