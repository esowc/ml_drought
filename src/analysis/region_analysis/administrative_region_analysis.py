from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Optional, List

from .base import RegionAnalysis

# from .region_geo_plotter import RegionGeoPlotter


class AdministrativeRegionAnalysis(RegionAnalysis):
    admin_boundaries = True

    def __init__(
        self,
        data_dir: Path = Path("data"),
        experiment: str = "one_month_forecast",
        true_data_experiment: str = "one_month_forecast",
        models_experiment_dir: str = "one_month_forecast",
    ):
        super().__init__(
            data_dir=data_dir,
            experiment=experiment,
            true_data_experiment=true_data_experiment,
            admin_boundaries=self.admin_boundaries,
            models_experiment_dir=models_experiment_dir,
        )

    @staticmethod
    def load_region_data(region_data_path: Path) -> Tuple[xr.DataArray, Dict, str]:
        """Load the preprocessed `region_data` from the
        `data/analysis/boundaries_preprocessed` directory. This will
        not only return the categorical DataArray but also the
        associated lookup data (stored in the `attrs`).

        Returns:
        -------
        :xr.DataArray
            the categorical xr.DataArray with the region data (same shape)

        :Dict
            the lookup dictionary with the keys referring to the
            values in the xr.DataArray and the values the names
            of the regions (for joining to the shapefile data)

        """
        # LOAD in region lookup DataArray
        assert "analysis" in region_data_path.parts, (
            "Only preprocessed"
            "region files (as netcdf) should be used"
            "`data/analysis`"
        )
        region_group_name: str = region_data_path.name
        region_ds: xr.Dataset = xr.open_dataset(region_data_path)
        region_da: xr.DataArray = region_ds[[v for v in region_ds.data_vars][0]]
        region_lookup: Dict = dict(
            zip(
                [int(k.strip()) for k in region_ds.attrs["keys"].split(",")],
                [v.strip() for v in region_ds.attrs["values"].split(",")],
            )
        )

        return region_da, region_lookup, region_group_name

    def compute_mean_statistics(
        self,
        region_da: xr.DataArray,
        region_lookup: Dict,
        pred_da: xr.DataArray,
        true_da: xr.DataArray,
        datetime: datetime,
    ) -> Tuple[List, List, List, List]:
        """compute the mean values in the DataArray for each region
        Returns:
        --------
        datetimes: List
            datetime of the true/predicted values
        region_names: List
            the name of the region
        predicted_mean_value: List
            the mean predicted value for ROI
        true_mean_value: List
            the mean true value for ROI
        """
        # For each region calculate mean `target_variable` in true / pred
        valid_region_ids: List = [k for k in region_lookup.keys()]
        region_names: List = []
        predicted_mean_value: List = []
        true_mean_value: List = []
        datetimes: List = []

        # check the shapes match
        pred_latlon_shape = (pred_da.lat.shape[0], pred_da.lon.shape[0])
        true_latlon_shape = (true_da.lat.shape[0], true_da.lon.shape[0])
        assert true_latlon_shape == true_latlon_shape == region_da.shape, (
            "Expect the lat/lon shapes to match in all input DataArrays"
            f"{pred_latlon_shape} == {true_latlon_shape} == {region_da.shape}. "
            "are you sure these have all been run with the same experiment"
            " and the same reference_nc_file to regrid onto same reference_grid"
        )

        for valid_region_id in valid_region_ids:
            region_names.append(region_lookup[valid_region_id])
            predicted_mean_value.append(
                pred_da.where(region_da == valid_region_id).mean().values
            )
            true_mean_value.append(
                true_da.where(region_da == valid_region_id).mean().values
            )
            # assert true_da.time == pred_da.time, 'time must be matching!'
            datetimes.append(datetime)

        assert len(region_names) == len(predicted_mean_value) == len(datetimes)
        return datetimes, region_names, predicted_mean_value, true_mean_value

    def _analyze_single(self, region_data_path: Path) -> Optional[pd.DataFrame]:
        """For a single shapefile (with multiple regions) calculate
        the mean predicted and mean true values in that region. Returns
        a DataFrame of the values in each region for each timestep that
        can then be used to calculate performance metrics in that region
        or at that administrative unit.

        example output:
          admin_level_name   model   datetime region_name predicted_mean_value true_mean_value
        0               l1  ealstm 2018-01-31    region_0                45.89           54.00
        1               l1  ealstm 2018-01-31    region_1                58.00           60.67
        2               l1  ealstm 2018-01-31    region_2                50.92           28.00
        3               l1  ealstm 2018-02-28    region_0                50.89           48.22
        4               l1  ealstm 2018-02-28    region_1                34.33           77.67
        """
        admin_level_name = region_data_path.name.replace(".nc", "")
        # for ONE REGION compute the each model statistics (pred & true)
        region_da, region_lookup, region_group_name = self.load_region_data(
            region_data_path
        )
        return self._base_analyze_single(
            admin_level_name=admin_level_name,
            region_da=region_da,
            region_lookup=region_lookup,
            region_group_name=region_group_name,
        )

    def analyze(
        self,
        compute_global_errors: bool = True,
        compute_regional_errors: bool = True,
        save_all_df: bool = True,
    ) -> None:
        """For all preprocessed regions calculate the mean True value and
        mean predicted values. Also have the option to calculate global
        errors (across all regions (in an admin level) / landcover classes).
        """
        all_regions_dfs = []

        for region_data_path in self.region_data_paths:
            df = self._analyze_single(region_data_path)
            all_regions_dfs.append(df)

        # clean up the DataFrame
        all_regions_dfs = [df for df in all_regions_dfs if df is not None]
        all_regions_df = pd.concat(all_regions_dfs).reset_index()

        if "index" in all_regions_df.columns:
            all_regions_df = all_regions_df.drop(columns="index")
        if "level_0" in all_regions_df.columns:
            all_regions_df = all_regions_df.drop(columns="level_0")

        self.df = all_regions_df.astype(
            {"predicted_mean_value": "float64", "true_mean_value": "float64"}
        )
        print("* Assigned all region dfs to `self.df` *")

        # compute error metrics for each model globally
        self.compute_metrics(compute_global_errors, compute_regional_errors)

        if save_all_df:
            self.df.to_csv(self.out_dir / f"{self.experiment}_all_admin_regions.csv")
