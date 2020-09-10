import numpy as np
import calendar
from datetime import date
import xarray as xr

from typing import cast, Dict, Optional, Tuple

from ..utils import minus_months
from .base import _EngineerBase


class _ForecastEngineer(_EngineerBase):
    r"""Engineer the preprocessed `.nc` files into `/train`, `/test` `{x, y}.nc`
    for the `forecast` experiment.

    This takes the forecast variables for the target timestep and the
    target variables for all previous timesteps (not the target timestep).

    This produces a dataset:
        t-3  t-2  t-1  t=0   |  y (t=0)
        --------------------------------
        X Y  X Y  X Y  fcast |  Y
                             |  Y

    Where X is the `non_target_variable` and Y is the `target_variable`.
    fcast is the `forecast_variable`
    t=0 is the `target_timestep`.
    """
    name = "forecast"

    def _stratify_xy(
        self,
        ds: xr.Dataset,
        year: int,
        target_variable: str,
        forecast_variables: List[str],
        target_month: int,
        pred_months: int,
        expected_length: Optional[int] = 11,
    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:
        """
        The forecast experiment has both `target_variables` and
        `forecast_variables` at the `target_time`.

        e.g. If we have `tprate_mean_1` and `tprate_std_1` as
        input variables

        `x_target_variable` = 11 timesteps
        `x_non_target_variable` = 12 timesteps (`pred_months + 1`).

        We overcome this by creating an extra timestep with all nan values
        in the `x_dataset`. This way the `x_dataset` contains the `y_dataset`
        timestep but the `target_variable` is an array of all `np.nan` for that
        target timestep. This prevents model leakage.
        """
        print(
            f"Generating data for year: {year}, target month: {target_month}")

        # get the test datetime
        max_date = date(year, target_month,
                        calendar.monthrange(year, target_month)[-1])
        mx_year, mx_month, max_train_date = minus_months(
            year, target_month, diff_months=1
        )
        _, _, min_date = minus_months(
            mx_year, mx_month, diff_months=pred_months)

        # convert to numpy datetime
        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))
        max_train_date_np = np.datetime64(str(max_train_date))

        print(
            f"Max date: {str(max_date)}, max input date: {str(max_train_date)}, "
            f"min input date: {str(min_date)}"
        )

        # boolean array indexing the TARGET VARIABLE timestamps to filter `ds`
        x_target = (ds.time.values > min_date_np) & (
            ds.time.values <= max_train_date_np
        )
        y_target = (ds.time.values > max_train_date_np) & (
            ds.time.values <= max_date_np
        )

        # boolean array indexing the other variables
        x_non_target = (ds.time.values > min_date_np) & (
            ds.time.values <= max_date_np)

        # only expect ONE y timestamp
        if sum(y_target) != 1:
            print(
                f"Wrong number of y values! Expected 1, got {sum(y_target)};\
            returning None"
            )
            return None, cast(date, max_train_date)

        # create the target dataset `y_dataset` & the `x_non_target_dataset`
        y_dataset = ds.isel(time=y_target)[target_variable].to_dataset(
            name=target_variable
        )
        x_non_target_dataset = ds.drop(target_variable).sel(time=x_non_target)

        # create the x_target_dataset with all -9999.0 at target time
        nan_target_variable = self._make_fill_value_dataset(y_dataset)
        x_target_dataset = (
            ds[target_variable].isel(
                time=x_target).to_dataset(name=target_variable)
        )

        if expected_length is not None:
            # filter for missing values in timesteps!
            if sum(x_target) != expected_length:
                print(
                    f"Wrong number of x values! Got {sum(x_target)} \
                Returning None"
                )

                return None, cast(date, max_train_date)

            if sum(x_non_target) != expected_length + 1:
                print(
                    f"Wrong number of x values! Got {sum(x_target)}\
                Returning None"
                )

                return None, cast(date, max_train_date)

        x_target_dataset = x_target_dataset.merge(nan_target_variable)

        # merge the x_non_target_dataset + x_target_dataset -> x_dataset
        x_dataset = x_non_target_dataset.merge(x_target_dataset)

        if x_dataset.time.size != cast(int, expected_length) + 1:
            # catch the errors as we get closer to the MINIMUM year
            print(
                "For the `nowcast` experiment we expect the\
                  number of timesteps to be: {pred_months + 1}.\
                  Currently: {x_dataset.time.size}"
            )
            return None, cast(date, max_train_date)

        return {"x": x_dataset, "y": y_dataset}, cast(date, max_train_date)
