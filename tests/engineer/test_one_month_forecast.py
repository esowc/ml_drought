import pytest
import pickle
import numpy as np
import xarray as xr
import datetime as dt

from src.engineer import _OneMonthForecastEngineer as OneMonthForecastEngineer
from ..utils import _make_dataset
from .test_base import _setup


class TestOneMonthForecastEngineer:
    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            OneMonthForecastEngineer(tmp_path)
            assert "does not exist. Has the preprocesser been run?" in str(e)

        (tmp_path / "interim").mkdir()

        OneMonthForecastEngineer(tmp_path)

        assert (tmp_path / "features").exists(), "Features directory not made!"
        assert (
            tmp_path / "features" / "one_month_forecast"
        ).exists(), "\
        one_month_forecast directory not made!"

    def test_static(self, tmp_path):
        _ = _setup(tmp_path, add_times=True, static=False)
        _, expected_vars = _setup(tmp_path, add_times=False, static=True)
        engineer = OneMonthForecastEngineer(tmp_path, process_static=True)

        assert (
            tmp_path / "features/static"
        ).exists(), "Static output folder does not exist!"

        engineer._process_static(test_year=2001)

        output_file = tmp_path / "features/static/data.nc"
        assert output_file.exists(), "Static output folder does not exist!"
        static_data = xr.open_dataset(output_file)

        for var in expected_vars:
            assert var in static_data.data_vars

    def test_yearsplit(self, tmp_path):

        _setup(tmp_path)

        dataset, _, _ = _make_dataset(size=(2, 2))

        engineer = OneMonthForecastEngineer(tmp_path)
        train = engineer._train_test_split(
            dataset,
            test_years=[2001],
            target_variable="VHI",
            pred_months=11,
            expected_length=11,
        )

        assert (
            train.time.values < np.datetime64("2001-01-01")
        ).all(), "Got years greater than the test year in the training set!"

    @staticmethod
    def _check_folder(folder_path, expected_length):
        y = xr.open_dataset(folder_path / "y.nc")
        assert "b" not in set(y.variables), "Got unexpected variables in test set"

        x = xr.open_dataset(folder_path / "x.nc")
        for expected_var in {"a", "b"}:
            assert expected_var in set(
                x.variables
            ), "Missing variables in testing input dataset"
        assert (
            len(x.time.values) == expected_length
        ), "Wrong number of months in the test x dataset"
        assert len(y.time.values) == 1, "Wrong number of months in test y dataset"

    def test_engineer(self, tmp_path):

        _setup(tmp_path)

        pred_months = expected_length = 11

        engineer = OneMonthForecastEngineer(tmp_path)
        engineer.engineer(
            test_year=2001,
            target_variable="a",
            pred_months=pred_months,
            expected_length=expected_length,
        )

        # _check_folder(tmp_path / 'features/one_month_forecast/train/1999_12')
        for month in range(1, 13):
            self._check_folder(
                tmp_path / f"features/one_month_forecast/test/2001_{month}",
                expected_length=expected_length,
            )
            self._check_folder(
                tmp_path / f"features/one_month_forecast/train/2000_{month}",
                expected_length=expected_length,
            )

        assert (
            len(list((tmp_path / "features/one_month_forecast/train").glob("2001_*")))
            == 0
        ), "Test data in the training data!"

        assert (
            tmp_path / "features/one_month_forecast/normalizing_dict.pkl"
        ).exists(), f"Normalizing dict not saved!"
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "rb"
        ) as f:
            norm_dict = pickle.load(f)

        for key, val in norm_dict.items():
            assert key in {"a", "b"}, f"Unexpected key!"
            assert norm_dict[key]["mean"] == 1, f"Mean incorrectly calculated!"
            assert norm_dict[key]["std"] == 0, f"Std incorrectly calculated!"

    def test_stratify(self, tmp_path):
        _setup(tmp_path)
        engineer = OneMonthForecastEngineer(tmp_path)
        ds_target, _, _ = _make_dataset(
            size=(20, 20), start_date="1999-01-01", end_date="2001-12-31"
        )
        ds_predictor, _, _ = _make_dataset(
            size=(20, 20), start_date="1999-01-01", end_date="2001-12-31"
        )
        ds_predictor = ds_predictor.rename({"VHI": "predictor"})
        ds = ds_predictor.merge(ds_target)

        xy_dict, max_train_date = engineer._stratify_xy(
            ds=ds,
            year=2001,
            target_variable="VHI",
            target_month=1,
            pred_months=4,
            expected_length=4,
        )

        assert (
            xy_dict["x"].time.size == 4
        ), f'OneMonthForecast experiment `x`\
        should have 4 times Got: {xy_dict["x"].time.size}'

        assert (
            max_train_date == dt.datetime(2000, 12, 31).date()
        ), f"\
        the max_train_date should be one month before the `target_month`,\
        `year`"
