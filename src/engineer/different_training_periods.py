import pandas as pd
from datetime import date
import xarray as xr
import warnings
import pickle
import pandas as pd

from typing import cast, Dict, Optional, Tuple, List, Union

from .one_month_forecast import _OneMonthForecastEngineer


class _DifferentTrainingPeriodsEngineer(_OneMonthForecastEngineer):
    name = "one_month_forecast"

    @staticmethod
    def check_data_leakage(
        train_ds: xr.Dataset, xy_test: Dict[str, xr.Dataset]
    ) -> None:
        # CHECK DATA LEAKAGE
        train_dts = [pd.to_datetime(t) for t in train_ds.time.values]
        test_dt = pd.to_datetime(xy_test["y"].time.values)
        assert test_dt not in train_dts, (
            "Data Leakage!" f"{test_dt} found in train_ds datetimes: \n {train_ds}"
        )

    def engineer(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
        train_years: Optional[List[int]] = None,
        train_timesteps: Optional[List[pd.Timestamp]] = None,
        test_timesteps: Optional[List[pd.Timestamp]] = None,
    ) -> None:

        self._process_dynamic(
            test_year=test_year,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
            train_years=train_years,
            train_timesteps=train_timesteps,
            test_timesteps=test_timesteps,
        )
        if self.process_static:
            self._process_static()

    def _process_dynamic(
        self,
        target_variable: str = "VHI",
        pred_months: int = 3,
        expected_length: Optional[int] = 3,
        test_year: Optional[Union[int, List[int]]] = None,
        train_years: Optional[List[int]] = None,
        test_timesteps: Optional[Union[pd.Timestamp, List[pd.Timestamp]]] = None,
        train_timesteps: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        if expected_length is None:
            warnings.warn(
                "** `expected_length` is None. This means that \
            missing data will not be skipped. Are you sure? **"
            )

        # read in all the data from interim/{var}_preprocessed
        data = self._make_dataset(static=False)

        # ensure test_year is List[int]
        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        # save test data (x, y) and return the train_ds (subset of `data`)
        if train_years is not None:
            train_ds, test_dts = self.year_train_test_split(
                ds=data,
                target_variable=target_variable,
                test_years=cast(List, test_year),
                pred_months=pred_months,
                expected_length=expected_length,
                train_years=train_years,
            )
        elif train_timesteps is not None:
            train_ds, test_dts = self.timestep_train_test_split(
                ds=data,
                target_variable=target_variable,
                pred_months=pred_months,
                expected_length=expected_length,
                test_timesteps=test_timesteps,
                train_timesteps=train_timesteps,
            )
        else:
            assert False, "Must provide either `train_years` or `train_timesteps`"

        assert train_ds.time.shape[0] > 0, (
            "Expect the train_ds to have" f"`time` dimension. \n{train_ds}"
        )

        normalization_values = self._calculate_normalization_values(train_ds)

        # split train_ds into x, y for each year-month before `test_year` & save
        if train_years is not None:
            # Define experiments that use whole years
            self._stratify_training_data(
                train_ds=train_ds,
                test_dts=test_dts,
                target_variable=target_variable,
                pred_months=pred_months,
                expected_length=expected_length,
                train_years=train_years,
            )

        elif train_timesteps is not None:
            # Define experiments that use individual TIMESTEPS not whole years
            self._stratify_training_data(
                train_ds=train_ds,
                test_dts=test_dts,
                target_variable=target_variable,
                pred_months=pred_months,
                expected_length=expected_length,
                train_timesteps=train_timesteps,
            )
        else:
            assert False, "Must provide either `train_years` or `train_timesteps`"

        savepath = self.output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    def timestep_train_test_split(
        self,
        ds: xr.Dataset,
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
        test_timesteps: Optional[Union[pd.Timestamp, List[pd.Timestamp]]],
        train_timesteps: Optional[List[pd.Timestamp]] = None,
    ) -> Tuple[xr.Dataset, List[date]]:
        """save the test data and return the training dataset
        1) SAVE [x.nc, y.nc] for each test timestep
        2) Return the dataset
        """
        test_dts = []

        # each month in test_year produce an x,y pair for testing
        for timestep in test_timesteps:
            # CALCULATE X, y for data for TEST timesteps
            year = int(timestep.year)
            month = int(timestep.month)
            xy_test, _ = self._stratify_xy(
                ds=ds,
                year=year,
                target_variable=target_variable,
                target_month=month,
                pred_months=pred_months,
                expected_length=expected_length,
            )

            if xy_test is not None:
                # check for data leakage
                # self.check_data_leakage(train_ds, xy_test)
                test_dts.append(self._get_datetime(xy_test["y"].time.values[0]))
                self._save(xy_test, year=year, month=month, dataset_type="test")

        train_ds = ds

        return train_ds, test_dts

    def year_train_test_split(
        self,
        ds: xr.Dataset,
        test_years: Union[int, List[int]],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
        train_years: Optional[List[int]] = None,
    ) -> Tuple[xr.Dataset, List[date]]:
        """save the test data and return the training dataset"""

        test_years.sort()
        test_dts = []

        months = ds["time.month"].values

        if ds["time.year"].min() == test_years[0]:
            # if we have the FIRST year in our dataset as a test year
            # then minimum target month should be `pred_months + 1`
            # e.g. x = 1984 Jan/Feb/Mar, y = 1984 April
            init_target_month = pred_months + months[0]
            print(f"\n* init_target_month = {init_target_month} *\n")
        else:
            init_target_month = 1

        # NOTE: the test year is
        # for the first `year` calculate the xy_test dictionary and min date
        xy_test, min_test_date = self._stratify_xy(
            ds=ds,
            year=test_years[0],
            target_variable=target_variable,
            target_month=init_target_month,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        train_ds = ds

        # save the xy_test dictionary
        if xy_test is not None:
            test_dts.append(self._get_datetime(xy_test["y"].time.values[0]))
            self._save(
                xy_test,
                year=test_years[0],
                month=init_target_month,
                dataset_type="test",
            )

            # check for data leakage
            # self.check_data_leakage(train_ds, xy_test)

        # each month in test_year produce an x,y pair for testing
        for year in test_years:
            for month in range(init_target_month, 13):
                if (month > init_target_month) | (year != test_years[0]):
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
                        # check for data leakage
                        # self.check_data_leakage(train_ds, xy_test)
                        test_dts.append(self._get_datetime(xy_test["y"].time.values[0]))
                        self._save(xy_test, year=year, month=month, dataset_type="test")
        return train_ds, test_dts

    def _stratify_training_data(  # type: ignore
        self,
        train_ds: xr.Dataset,
        test_dts: List[date],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
        train_years: Optional[List[int]] = None,
        train_timesteps: Optional[List[pd.Timestamp]] = None,
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
                print_status=False,
            )

            # Preventing data leakage:
            if train_years is not None:
                # only save if that year is in train_years
                # and that month is not in test_dts
                if arrays is not None:
                    if (cur_pred_year in train_years) and (  # type: ignore
                        self._get_datetime(arrays["y"].time.values[0]) not in test_dts
                    ):
                        self._save(
                            arrays,
                            year=cur_pred_year,
                            month=cur_pred_month,
                            dataset_type="train",
                        )

            # ALREADY PREVENTED here ...
            if train_timesteps is not None:
                # only save if that ts is in train_timesteps
                # pd.to_datetime(f'{cur_pred_year}-{int(cur_pred_month):02}')
                if arrays is not None:
                    if (
                        pd.to_datetime(self._get_datetime(arrays["y"].time.values[0]))
                        in train_timesteps
                    ):
                        self._save(
                            arrays,
                            year=cur_pred_year,
                            month=cur_pred_month,
                            dataset_type="train",
                        )

            cur_pred_year, cur_pred_month = cur_min_date.year, cur_min_date.month