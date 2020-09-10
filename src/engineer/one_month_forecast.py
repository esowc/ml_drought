import numpy as np
import calendar
from datetime import date
import xarray as xr

from typing import cast, Dict, Optional, Tuple

from ..utils import minus_months
from .base import _EngineerBase


class _OneMonthForecastEngineer(_EngineerBase):
    name = "one_month_forecast"

    @staticmethod
    def _stratify_xy(
        ds: xr.Dataset,
        year: int,
        target_variable: str,
        target_month: int,
        pred_months: int = 11,
        expected_length: Optional[int] = 11,
        print_status: bool = False,
    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:
        """
        Note: expected_length should be the same as pred_months when the timesteps
        are monthly, but should be more if the timesteps are at shorter resolution
        than monthly.
        """
        if print_status:
            print(f"Generating data for year: {year}, target month: {target_month}")

        max_date = date(year, target_month, calendar.monthrange(year, target_month)[-1])
        mx_year, mx_month, max_train_date = minus_months(
            year, target_month, diff_months=1
        )
        _, _, min_date = minus_months(mx_year, mx_month, diff_months=pred_months)

        # `max_date` is the date to be predicted;
        # `max_train_date` is one timestep before;
        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))
        max_train_date_np = np.datetime64(str(max_train_date))

        if print_status:
            print(
                f"Max date: {str(max_date)}, max input date: {str(max_train_date)}, "
                f"min input date: {str(min_date)}"
            )

        # boolean array indexing the timestamps to filter `ds`
        x = (ds.time.values > min_date_np) & (ds.time.values <= max_train_date_np)
        y = (ds.time.values > max_train_date_np) & (ds.time.values <= max_date_np)

        # only expect ONE y timestamp
        if sum(y) != 1:
            print(f"Wrong number of y values! Expected 1, got {sum(y)}; returning None")
            return None, cast(date, max_train_date)

        if expected_length is not None:
            if sum(x) != expected_length:
                print(f"Wrong number of x values! Got {sum(x)} Returning None")

                return None, cast(date, max_train_date)

        # filter the dataset
        x_dataset = ds.isel(time=x)
        y_dataset = ds.isel(time=y)[target_variable].to_dataset(name=target_variable)

        if x_dataset.time.size != expected_length:
            # catch the errors as we get closer to the MINIMUM year
            print(
                "For the `nowcast` experiment we expect the "
                f"number of timesteps to be: {pred_months}. "
                f"Currently: {x_dataset.time.size} "
                f"You provided the argument expected_length: {expected_length}"
            )
            return None, cast(date, max_train_date)

        return {"x": x_dataset, "y": y_dataset}, cast(date, max_train_date)
