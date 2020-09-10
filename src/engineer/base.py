import numpy as np
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
import pickle
import xarray as xr
import warnings
from collections.abc import Iterable

from typing import cast, DefaultDict, Dict, List, Optional, Union, Tuple


class _EngineerBase:
    name: str

    def __init__(
        self, data_folder: Path = Path("data"), process_static: bool = False
    ) -> None:

        self.data_folder = data_folder
        self.process_static = process_static

        self.interim_folder = data_folder / "interim"
        assert (
            self.interim_folder.exists()
        ), f'{data_folder / "interim"} does not exist. Has the preprocesser been run?'

        try:
            # specific folder for that
            self.output_folder = data_folder / "features" / self.name
            if not self.output_folder.exists():
                self.output_folder.mkdir(parents=True)
        except AttributeError:
            print("Name not defined! No experiment folder set up")

        if self.process_static:
            self.static_output_folder = data_folder / "features/static"
            if not self.static_output_folder.exists():
                self.static_output_folder.mkdir(parents=True)

    def engineer(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
        global_means: bool = True,
        pixel_means: bool = True,
    ) -> None:

        self._process_dynamic(test_year, target_variable, pred_months, expected_length)
        if self.process_static:
            self._process_static(
                test_year=test_year, global_means=global_means, pixel_means=pixel_means
            )

    def calculate_static_means_ds(
        self,
        static_ds: xr.Dataset,
        test_year: Union[int, List[int]],
        global_means_bool: bool = False,
        pixel_means_bool: bool = True,
    ) -> xr.Dataset:
        dynamic_ds = self._make_dataset(static=False, overwrite_dims=False)

        # ignore test years in calculation of means
        min_year = dynamic_ds.time.min()
        test_year = [test_year] if not isinstance(test_year, Iterable) else test_year
        dynamic_ds = dynamic_ds.sel(
            time=slice(min_year, str(min(test_year) - 1))  # type: ignore
        )
        assert all(
            int(yr) not in np.unique(dynamic_ds["time.year"].values) for yr in test_year
        )

        ones = xr.ones_like(static_ds)
        ones_da = ones[[v for v in ones.data_vars][0]]

        if global_means_bool:
            # 1. create global means ds
            global_means = dynamic_ds.mean(dim=["lat", "lon", "time"])
            global_means = ones_da * global_means
            # rename variables
            rename_map = {v: f"{v}_global_mean" for v in global_means.data_vars}
            global_means = global_means.rename(rename_map)

        if pixel_means_bool:
            # 2. create pixel means ds
            pixel_means = dynamic_ds.mean(dim=["time"])
            pixel_means = ones_da * pixel_means
            # rename variables
            rename_map = {v: f"{v}_pixel_mean" for v in pixel_means.data_vars}
            pixel_means = pixel_means.rename(rename_map)

        # TODO: this can be cleaned
        if global_means_bool & pixel_means_bool:
            static_mean_ds = xr.auto_combine([global_means, pixel_means])
        elif global_means_bool & ~pixel_means_bool:
            static_mean_ds = xr.auto_combine([global_means])
        elif ~global_means_bool & pixel_means_bool:
            static_mean_ds = xr.auto_combine([pixel_means])
        else:
            # return an empty dataset
            static_mean_ds = xr.Dataset()

        return static_mean_ds

    def _process_static(
        self,
        test_year: Union[int, List[int]],
        global_means: bool = True,
        pixel_means: bool = True,
    ) -> None:
        """
        Note:
        requires `test_year` so that can ignore test years in the calculation
        of spatial means and global means of dynamic variables.
        """
        # this function assumes the static data has only two dimensions,
        # lat and lon

        output_file = self.static_output_folder / "data.nc"
        if output_file.exists():
            warnings.warn("A static data file already exists!")
            return None

        # here, we overwrite the dims because topography (a static variable)
        # uses CDO for regridding, which yields very slightly different
        # coordinates (it seems from rounding)
        try:
            static_ds = self._make_dataset(static=True, overwrite_dims=True)
        except ValueError:
            print("No static data features included! Creating static ds")
            static_ds = None
        # create dynamic_variable means for input to static data
        static_mean_ds = self.calculate_static_means_ds(
            static_ds=static_ds,
            test_year=test_year,
            global_means_bool=global_means,
            pixel_means_bool=pixel_means,
        )

        if static_ds is None:
            static_ds = static_mean_ds
        else:
            static_ds = xr.auto_combine([static_ds, static_mean_ds])

        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        for var in static_ds.data_vars:
            if var.endswith("one_hot"):
                mean = 0.0
                std = 1.0
            else:
                mean = float(
                    static_ds[var].mean(dim=["lat", "lon"], skipna=True).values
                )
                std = float(static_ds[var].std(dim=["lat", "lon"], skipna=True).values)

            normalization_values[var]["mean"] = mean
            normalization_values[var]["std"] = std

        static_ds.to_netcdf(self.static_output_folder / "data.nc")
        savepath = self.static_output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    def _process_dynamic(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
    ) -> None:
        if expected_length is None:
            warnings.warn(
                "** `expected_length` is None. This means that \
            missing data will not be skipped. Are you sure? **"
            )

        # read in all the data from interim/{var}_preprocessed
        data = self._make_dataset(static=False)  # .sortby('lat')

        # ensure test_year is List[int]
        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        # save test data (x, y) and return the train_ds (subset of `data`)
        train_ds = self._train_test_split(
            ds=data,
            test_years=cast(List, test_year),
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )
        assert train_ds.time.shape[0] > 0, (
            "Expect the train_ds to have" f"`time` dimension. \n{train_ds}"
        )

        normalization_values = self._calculate_normalization_values(train_ds)

        # split train_ds into x, y for each year-month before `test_year` & save
        self._stratify_training_data(
            train_ds=train_ds,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        savepath = self.output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    def _get_preprocessed_files(self, static: bool) -> List[Path]:
        processed_files = []
        if static:
            interim_folder = self.interim_folder / "static"
        else:
            interim_folder = self.interim_folder
        for subfolder in interim_folder.iterdir():
            if str(subfolder).endswith("_preprocessed") and subfolder.is_dir():
                processed_files.extend(list(subfolder.glob("*.nc")))
        return processed_files

    def _make_dataset(self, static: bool, overwrite_dims: bool = False) -> xr.Dataset:

        datasets = []
        dims = ["lon", "lat"]
        coords = {}
        for idx, file in enumerate(self._get_preprocessed_files(static)):
            print(f"Processing {file}")
            datasets.append(xr.open_dataset(file))

            if idx == 0:
                for dim in dims:
                    coords[dim] = datasets[idx][dim].values
            else:
                for dim in dims:
                    array_equal = np.array_equal(datasets[idx][dim].values, coords[dim])
                    if (not overwrite_dims) and (not array_equal):
                        # SORT the values first (xarray clever enough to figure out joining)
                        assert np.array_equal(
                            np.sort(datasets[idx][dim].values), np.sort(coords[dim])
                        ), f"{dim} is different! Was this run using the preprocessor?"
                    elif overwrite_dims and (not array_equal):
                        assert len(datasets[idx][dim].values) == len(coords[dim])
                        datasets[idx][dim] = coords[dim]

        # join all preprocessed datasets
        main_dataset = datasets[0]
        for dataset in datasets[1:]:
            # ensure equal timesteps ('inner' join)
            main_dataset = main_dataset.merge(dataset, join="inner")

        return main_dataset

    def _stratify_training_data(
        self,
        train_ds: xr.Dataset,
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> None:
        """split `train_ds` into x, y and save the outputs to
        self.output_folder (data/features) """

        min_date = self._get_datetime(train_ds.time.values.min())
        max_date = self._get_datetime(train_ds.time.values.max())

        cur_pred_year, cur_pred_month = max_date.year, max_date.month

        # for every month-year create & save the x, y datasets for training
        cur_min_date = max_date
        while cur_min_date >= min_date:
            # each iteration count down one month (02 -> 01 -> 12 ...)
            arrays, cur_min_date = self._stratify_xy(
                ds=train_ds,
                year=cur_pred_year,
                target_variable=target_variable,
                target_month=cur_pred_month,
                pred_months=pred_months,
                expected_length=expected_length,
            )
            if arrays is not None:
                self._save(
                    arrays,
                    year=cur_pred_year,
                    month=cur_pred_month,
                    dataset_type="train",
                )
            cur_pred_year, cur_pred_month = cur_min_date.year, cur_min_date.month

    def _train_test_split(
        self,
        ds: xr.Dataset,
        test_years: List[int],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
    ) -> xr.Dataset:
        """save the test data and return the training dataset"""

        test_years.sort()

        # for the first `year` Jan calculate the xy_test dictionary and min date
        xy_test, min_test_date = self._stratify_xy(
            ds=ds,
            year=test_years[0],
            target_variable=target_variable,
            target_month=1,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        # the train_ds MUST BE from before minimum test date
        train_dates = ds.time.values <= np.datetime64(str(min_test_date))
        train_ds = ds.isel(time=train_dates)

        # save the xy_test dictionary
        if xy_test is not None:
            self._save(xy_test, year=test_years[0], month=1, dataset_type="test")

        # each month in test_year produce an x,y pair for testing
        for year in test_years:
            for month in range(1, 13):
                if year > test_years[0] or month > 1:
                    # prevents the initial test set from being recalculated
                    xy_test, _ = self._stratify_xy(
                        ds=ds,
                        year=year,
                        target_variable=target_variable,
                        target_month=month,
                        pred_months=pred_months,
                        expected_length=expected_length,
                    )
                    if xy_test is not None:
                        self._save(xy_test, year=year, month=month, dataset_type="test")
        return train_ds

    def _stratify_xy(
        self,
        ds: xr.Dataset,
        year: int,
        target_variable: str,
        target_month: int,
        pred_months: int,
        expected_length: Optional[int],
    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:
        raise NotImplementedError

    @staticmethod
    def _get_datetime(time: np.datetime64) -> date:
        return datetime.strptime(time.astype(str)[:10], "%Y-%m-%d").date()

    def _save(
        self, ds_dict: Dict[str, xr.Dataset], year: int, month: int, dataset_type: str
    ) -> None:

        save_folder = self.output_folder / dataset_type
        save_folder.mkdir(exist_ok=True)

        output_location = save_folder / f"{year}_{month}"
        output_location.mkdir(exist_ok=True)

        for x_or_y, output_ds in ds_dict.items():
            print(f"Saving data to {output_location.as_posix()}/{x_or_y}.nc")
            output_ds.to_netcdf(output_location / f"{x_or_y}.nc")

    def _calculate_normalization_values(
        self, x_data: xr.Dataset
    ) -> DefaultDict[str, Dict[str, float]]:
        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        for var in x_data.data_vars:
            mean = float(
                x_data[var].mean(dim=["lat", "lon", "time"], skipna=True).values
            )
            std = float(x_data[var].std(dim=["lat", "lon", "time"], skipna=True).values)
            normalization_values[var]["mean"] = mean
            normalization_values[var]["std"] = std

        return normalization_values

    @staticmethod
    def _make_fill_value_dataset(
        ds: Union[xr.Dataset, xr.DataArray], fill_value: Union[int, float] = -9999.0
    ) -> Union[xr.Dataset, xr.DataArray]:
        nan_ds = xr.full_like(ds, fill_value)
        return nan_ds
