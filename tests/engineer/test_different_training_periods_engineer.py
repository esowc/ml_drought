import xarray as xr
from src.engineer import Engineer
from .test_base import _setup


class TestDifferentTrainingPeriodsEngineer:
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

    def test_stratify_with_non_sequential_test_train_years(self, tmp_path):
        _setup(tmp_path)
        engineer = Engineer(
            tmp_path,
            experiment="one_month_forecast",
            process_static=True,
            different_training_periods=True,
        )

        expected_length = 3

        # -------------------------------
        # test a first test/train option
        # -------------------------------
        engineer.engineer_class._process_dynamic(
            test_year=[1999],
            target_variable="a",
            pred_months=3,
            expected_length=3,
            train_years=[2000, 2001],
        )

        for month in range(4, 13):
            # TEST DATA
            # because our first year is 1999 (test_year) we can only test from
            # the 4th month because we don't have any data from before that date
            # so should X dates = {1999-01, 1999-02, 1999-03}, y dates = 1999-04
            self._check_folder(
                tmp_path / f"features/one_month_forecast/test/1999_{month}",
                expected_length=expected_length,
            )

            # TRAIN DATA
            # because of data leakage we should not have 1999-10, 1999-11, 1999-12 in
            # train data. Therefore, to predict with 3 month lead times we need
            # to train from y = 2000-04, X = {2000-01, 2000-02, 2000-03}
            self._check_folder(
                tmp_path / f"features/one_month_forecast/train/2000_{month}",
                expected_length=expected_length,
            )

        for month in range(1, 13):
            # TRAIN DATA
            # for 2001 we should have all 12 of the months
            self._check_folder(
                tmp_path / f"features/one_month_forecast/train/2001_{month}",
                expected_length=expected_length,
            )

        # -------------------------------
        # test a second test/train option
        # -------------------------------
        engineer.engineer_class._process_dynamic(
            test_year=[2000],
            target_variable="a",
            pred_months=3,
            expected_length=3,
            train_years=[1999, 2001],
        )

        for month in range(4, 13):
            # TEST DATA
            # because our first year is 2000 (test_year) we can only test from
            # the 4th month because we don't have any data from before that date
            # so should X dates = {2000-01, 2000-02, 2000-03}, y dates = 2000-04
            self._check_folder(
                tmp_path / f"features/one_month_forecast/test/2000_{month}",
                expected_length=expected_length,
            )

            # TRAIN DATA
            # because of data leakage we should not have 2000-10, 2000-11, 2000-12 in
            # train data. Therefore, to predict with 3 month lead times we need
            # to train from y = 2001-04, X = {2001-01, 2001-02, 2001-03}
            self._check_folder(
                tmp_path / f"features/one_month_forecast/train/2001_{month}",
                expected_length=expected_length,
            )
            self._check_folder(
                tmp_path / f"features/one_month_forecast/train/2000_{month}",
                expected_length=expected_length,
            )
