import pytest
import pickle
import numpy as np
import xarray as xr
import datetime as dt

from src.engineer import _NowcastEngineer as NowcastEngineer

from ..utils import _make_dataset
from .test_base import _setup


class TestNowcastEngineer:
    def test_init(self, tmp_path):

        with pytest.raises(AssertionError) as e:
            NowcastEngineer(tmp_path)
            assert "does not exist. Has the preprocesser been run?" in str(e)

        (tmp_path / "interim").mkdir()

        NowcastEngineer(tmp_path)

        assert (tmp_path / "features").exists(), "Features directory not made!"
        assert (
            tmp_path / "features" / "nowcast"
        ).exists(), "\
        nowcast directory not made!"

    def test_static(self, tmp_path):
        _ = _setup(tmp_path, add_times=True, static=False)
        _, expected_vars = _setup(tmp_path, add_times=False, static=True)
        engineer = NowcastEngineer(tmp_path, process_static=True)

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

        engineer = NowcastEngineer(tmp_path)
        train = engineer._train_test_split(
            dataset,
            test_years=[2001],
            target_variable="VHI",
            expected_length=11,
            pred_months=11,
        )

        assert (
            train.time.values < np.datetime64("2001-01-01")
        ).all(), "Got years greater than the test year in the training set!"

    def test_engineer(self, tmp_path):

        _setup(tmp_path)

        engineer = NowcastEngineer(tmp_path)
        engineer.engineer(
            test_year=2001, target_variable="a", pred_months=11, expected_length=11
        )

        def check_folder(folder_path):
            y = xr.open_dataset(folder_path / "y.nc")
            assert "b" not in set(y.variables), "Got unexpected variables in test set"

            x = xr.open_dataset(folder_path / "x.nc")
            for expected_var in {"a", "b"}:
                assert expected_var in set(
                    x.variables
                ), "Missing variables in testing input dataset"
            # NB different number of months in the `nowcast`
            assert (
                len(x.time.values) == 12
            ), "Wrong number of months in the test x dataset"
            assert len(y.time.values) == 1, "Wrong number of months in test y dataset"

        check_folder(tmp_path / "features/nowcast/train/1999_12")
        for month in range(1, 13):
            check_folder(tmp_path / f"features/nowcast/test/2001_{month}")
            check_folder(tmp_path / f"features/nowcast/train/2000_{month}")

        assert (
            len(list((tmp_path / "features/nowcast/train").glob("2001_*"))) == 0
        ), "Test data in the training data!"

        assert (
            tmp_path / "features/nowcast/normalizing_dict.pkl"
        ).exists(), f"Normalizing dict not saved!"
        with (tmp_path / "features/nowcast/normalizing_dict.pkl").open("rb") as f:
            norm_dict = pickle.load(f)

        for key, val in norm_dict.items():
            assert key in {"a", "b"}, f"Unexpected key!"
            # TODO: fix how to test for the final (12th) value
            assert norm_dict[key]["mean"] == 1, f"Mean incorrectly calculated!"
            assert norm_dict[key]["std"] == 0, f"Std incorrectly calculated!"

    def test_stratify(self, tmp_path):
        _setup(tmp_path)
        engineer = NowcastEngineer(tmp_path)
        ds_target, _, _ = _make_dataset(size=(20, 20))
        ds_predictor, _, _ = _make_dataset(size=(20, 20))
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
            xy_dict["x"].time.size == 5
        ), f'Nowcast experiment `x`\
        should have 5 times (the final time is all -9999 for `target`)\
        Got: {xy_dict["x"].time.size}'

        assert (
            (xy_dict["x"].VHI.isel(time=-1) == -9999).all().values
        ), f"\
        the final VHI timestep should ALL be -9999 (to avoid model leakage)"

        assert (
            max_train_date == dt.datetime(2000, 12, 31).date()
        ), f"\
        the max_train_date should be one month before the `target_month`,\
        `year`"

    def test_stratify_catches_not_equal_expected_length(self, tmp_path):
        _setup(tmp_path)
        engineer = NowcastEngineer(tmp_path)

        ds_target, _, _ = _make_dataset(size=(20, 20))
        ds_predictor, _, _ = _make_dataset(size=(20, 20))
        ds_predictor = ds_predictor.rename({"VHI": "predictor"})
        ds = ds_predictor.merge(ds_target)

        xy_dict, max_train_date = engineer._stratify_xy(
            ds=ds,
            year=2001,
            target_variable="VHI",
            target_month=1,
            pred_months=4,
            expected_length=5,
        )

        assert (
            xy_dict is None
        ), f"xy_dict should be None because the number of\
        expected timesteps is different from `expected_length`"
