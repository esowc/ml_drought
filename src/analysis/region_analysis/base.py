from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, List, Union, Optional
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools

from .region_geo_plotter import RegionGeoPlotter


class RegionAnalysis:
    """Create summary statistics for all Regions
    (defined as xr.Dataset objects) comparing the model
    predictions against the true values for both train and
    test datasets.

    This class produces two different formats for the same
    data. Firstly, it writes a csv file for each
    model-region object. Secondly, it saves all of the
    data to a class attribute (`self.df`) in long format.
    This allows the class to then produce summary
    statistics.

    Attributes:
    -----------
    :region_ds: xr.DataArray
    :region_lookup: Dict
    :self.pred_variable: str
    :self.true_variable: str
    :self.models_dir: Path
    :self.features_dir: Path
    :self.models: List[str]
    :self.admin_boundaries: bool

    TODO:
    # train or test `true` data ?
    self.mode = 'test' if test_mode else 'train'
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        experiment: str = "one_month_forecast",
        true_data_experiment: str = "one_month_forecast",
        models: Union[List[str], None] = None,
        admin_boundaries: bool = True,
        models_experiment_dir: Optional[str] = None,
    ):
        """Base RegionAnalysis class.

        Arguments:
        :data_dir: Path = Path('data')
            path to data directory

        :experiment: str = 'one_month_forecast'
            the name of the model dir that want to analyze

        true_data_experiment: str = "one_month_forecast",
            the name of the features/ dir used for Train/Test X/y split

        :models: Union[List[str], None] = None
            a list of models (as strings)

        :admin_boundaries: bool = True
        """
        self.pred_variable: Optional[str] = None
        self.true_variable: Optional[str] = None

        self.data_dir: Path = data_dir
        self.models_dir: Path = data_dir / "models" / experiment
        self.features_dir: Path = data_dir / "features" / true_data_experiment / "test"
        assert self.models_dir.exists(), (
            "Require {data_path}/models to have been" "created by the pipeline."
        )
        assert self.features_dir.exists(), (
            "Require {data_path}/features to have been" "created by the pipeline."
        )

        self.models: List[str] = [
            m.name
            for m in self.models_dir.iterdir()
            if m.name[0] != "."  # hidden files
        ]
        if models is not None:
            self.models = self.models[np.isin(self.models, models)]
            assert self.models is not [], (
                "None of the `models` are here in "
                f"try one of: {[d.name for d in self.models_dir.iterdir()]}"
            )

        if not (("one_month_forecast" in experiment) or ("nowcast" in experiment)):
            print(
                "WARNING: could not find one_month_forecast or nowcast in experiment name. Are you sure this has been run through the pipeline?"
            )
        self.experiment: str = experiment

        # NOTE: this shouldn't be specific for the boundaries it should
        # also be able to work with landcover
        self.admin_boundaries: bool = admin_boundaries

        if self.admin_boundaries:
            self.shape_data_dir = data_dir / "analysis" / "boundaries_preprocessed"
        else:
            static_dir = (
                data_dir / "interim" / "static" / "esa_cci_landcover_preprocessed"
            )
            self.shape_data_dir = static_dir

        self.region_data_paths: List[Path] = [
            f for f in self.shape_data_dir.glob("*.nc")
        ]

        self.out_dir: Path = data_dir / "analysis" / "region_analysis"
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self.df: Optional[pd.DataFrame] = None

        print(f"Initialised the Region Analysis for experiment: {self.experiment}")
        print(f"Models: {self.models}")
        print(f"Regions: {[r.name for r in self.region_data_paths]}")
        region_type = (
            "administrative_boundaries" if self.admin_boundaries else "landcover"
        )
        print(f"Region Type: {region_type}")
        # print(f'Test timesteps: {}')

    def load_prediction_data(self, preds_data_path: Path) -> xr.DataArray:
        assert "models" in preds_data_path.parts, (
            "Only modelled"
            "from the pipeline should be used using this class"
            "`data/models`"
        )

        pred_ds = xr.open_dataset(preds_data_path)
        # get the variable = Dataset -> DataArray
        if self.pred_variable is None:
            # Check that variables are only length 1
            pred_variables = [v for v in pred_ds.data_vars]
            assert len(pred_variables) == 1, "Only expect one variable in pred_ds"
            self.pred_variable = pred_variables[0]

        return pred_ds[self.pred_variable]

    def load_true_data(self, true_data_path: Path) -> xr.DataArray:
        assert "features" in true_data_path.parts, (
            "Only engineered data"
            "from the pipeline should be used using this class"
            "`data/features`"
        )
        true_ds = xr.open_dataset(true_data_path)
        # Dataset -> DataArray
        if self.true_variable is None:
            # Check that variables are only length 1
            true_variables = [v for v in true_ds.data_vars]
            assert len(true_variables) == 1, "Only expect one variable in true_ds"
            self.true_variable = true_variables[0]

        return true_ds[self.true_variable]

    def get_pred_data_on_timestep(self, datetime: datetime, model: str) -> Path:
        # TODO: fix this method to be more flexible to higher time-resolution data
        warnings.warn("This functionality only works with MONTHLY predictions")
        month = int(datetime.month)
        year = int(datetime.year)

        preds_data_path = self.models_dir / model / f"preds_{year}_{month}.nc"
        return preds_data_path

    @staticmethod
    def read_xr_datetime(xr_obj: Union[xr.Dataset, xr.DataArray]) -> datetime:
        dt = pd.to_datetime(xr_obj.time.values)
        assert len(dt) == 1, "only meant to have ONE datetime in this example"

        return dt.to_pydatetime()[0]

    @staticmethod
    def compute_error_metrics(group_model_performance) -> Tuple[float, float, float]:
        # drop nans
        group_model_performance = group_model_performance.dropna(how="any")
        if group_model_performance.empty:
            return np.nan, np.nan, np.nan

        rmse = np.sqrt(
            mean_squared_error(
                group_model_performance.true_mean_value,
                group_model_performance.predicted_mean_value,
            )
        )
        mae = mean_absolute_error(
            group_model_performance.true_mean_value,
            group_model_performance.predicted_mean_value,
        )
        r2 = r2_score(
            group_model_performance.true_mean_value,
            group_model_performance.predicted_mean_value,
        )
        return rmse, mae, r2

    def compute_global_error_metrics(self) -> pd.DataFrame:
        models = []
        admin_regions = []
        rmses = []
        maes = []
        r2s = []

        assert self.df is not None, (
            "This method requires `self.analyze`"
            "to have been run. Have you run the `analyze()` method?"
        )

        # TODO: pandas groupby functionality?
        groups = [
            p
            for p in itertools.product(
                self.df.admin_level_name.unique(), self.df.model.unique()
            )
        ]
        for admin_name, model in groups:
            group_model_performance = (
                self.df.loc[self.df.admin_level_name == admin_name]
                .loc[
                    self.df.model == model, ["predicted_mean_value", "true_mean_value"]
                ]
                .astype("float")
            )
            rmse, mae, r2 = self.compute_error_metrics(group_model_performance)

            admin_regions.append(admin_name)
            models.append(model)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

        return pd.DataFrame(
            {
                "model": models,
                "admin_level_name": admin_regions,
                "rmse": rmses,
                "mae": maes,
                "r2": r2s,
            }
        )

    def compute_regional_error_metrics(self) -> pd.DataFrame:
        """calculate the mean error in each Region over time
        (12 test months by default).
        """
        assert self.df is not None, (
            "require `RegionAnalysis.df`. Has" " `RegionAnalysis.analyze()` been run?"
        )
        models = self.df.model.unique()
        region_names = self.df.region_name.unique()
        admin_level_names = self.df.admin_level_name.unique()

        models_array = []
        admin_level_names_array = []
        region_names_array = []
        rmses = []
        maes = []
        r2s = []

        # calculate TEMPORAL average
        products = itertools.product(models, region_names, admin_level_names)
        groups = [g for g in products]  # 0: model, 1: region, 2: admin_level

        for model, region, admin_level in groups:
            df_group = self.df.loc[
                # rows
                (self.df.model == model)
                & (self.df.region_name == region)
                & (self.df.admin_level_name == admin_level),
                # columns
                ["predicted_mean_value", "true_mean_value"],
            ].astype("float")

            rmse, mae, r2 = self.compute_error_metrics(df_group)

            # create the arrays
            models_array.append(model)
            admin_level_names_array.append(admin_level)
            region_names_array.append(region)
            rmses.append(rmse)
            maes.append(mae)
            r2s.append(r2)

        return pd.DataFrame(
            {
                "model": models_array,
                "admin_level_name": admin_level_names_array,
                "region_name": region_names_array,
                "rmse": rmses,
                "mae": maes,
                "r2": r2s,
            }
        )

    def create_model_performance_by_region_geodataframe(self) -> RegionGeoPlotter:
        """Join pd.DataFrame object stored in `RegionAnalysis.df` with the a
        GeoDataFrames for the admin_level at `RegionGeoPlotter.gdf` in order
        to create spatial plots of the model performances in different regions.
        """
        assert self.regional_mean_metrics is not None, (
            "require "
            "`RegionAnalysis.df`. Has `RegionAnalysis.analyze`"
            "been run? Run that first!"
        )

        # create the region geoplotter object
        geoplotter = RegionGeoPlotter(data_folder=self.data_dir, country="kenya")
        geoplotter.read_shapefiles()
        geoplotter.merge_all_model_performances_gdfs(
            all_models_df=self.regional_mean_metrics
        )
        return geoplotter

    def _base_analyze_single(
        self,
        admin_level_name: str,
        region_da: Optional[xr.DataArray] = None,
        region_lookup: Optional[Dict] = None,
        region_group_name: Optional[str] = None,
        landcover_das: Optional[List[xr.DataArray]] = None,
    ) -> Optional[pd.DataFrame]:
        # RUN CHECKS FOR CORRECT INPUTS
        if self.admin_boundaries:
            assert region_da is not None, (
                "require `region_da`" "argument to run administrative region analysis"
            )
            assert region_lookup is not None, (
                "require `region_lookup`"
                "argument to run administrative region analysis"
            )
            assert region_group_name is not None, (
                "require `region_group_name`"
                "argument to run administrative region analysis"
            )
        else:
            assert landcover_das is not None, (
                "If running landcover region analysis"
                "require the `landcover_das` argument to run the analysis"
            )

        print(f"* Analyzing for {admin_level_name} *")
        all_model_dfs = []
        for model in self.models:
            print(f"\n** Analyzing for {model}-{admin_level_name} **")
            # create the filename
            if not (self.out_dir / model).exists():
                (self.out_dir / model).mkdir(exist_ok=True, parents=True)

            dfs = []
            true_data_paths = [f for f in self.features_dir.glob("*/y.nc")]
            # convert this to funciton
            for true_data_path in true_data_paths:
                # load the required data
                true_da = self.load_true_data(true_data_path)
                dt = self.read_xr_datetime(true_da)
                preds_data_path = self.get_pred_data_on_timestep(
                    datetime=dt, model=model
                )
                if not preds_data_path.exists():
                    # if it's not there
                    warnings.warn(
                        f"{preds_data_path.parents[0] / preds_data_path.name} does not exist"
                    )
                    continue
                pred_da = self.load_prediction_data(preds_data_path)

                # compute the statistics - different for landcover vs. admin regions
                if self.admin_boundaries:
                    (
                        datetimes,
                        region_name,
                        predicted_mean_value,
                        true_mean_value,
                    ) = self.compute_mean_statistics(  # type: ignore
                        region_da=region_da,
                        true_da=true_da,
                        pred_da=pred_da,
                        region_lookup=region_lookup,
                        datetime=dt,
                    )
                else:
                    (
                        datetimes,
                        region_name,
                        predicted_mean_value,
                        true_mean_value,
                    ) = self.compute_mean_statistics(  # type: ignore
                        landcover_das, true_da=true_da, pred_da=pred_da, datetime=dt
                    )

                # store as pandas object and add to
                dfs.append(
                    pd.DataFrame(
                        {
                            "admin_level_name": [
                                admin_level_name for _ in range(len(datetimes))
                            ],
                            "model": [model for _ in range(len(datetimes))],
                            "datetime": datetimes,
                            "region_name": region_name,
                            "predicted_mean_value": predicted_mean_value,
                            "true_mean_value": true_mean_value,
                        }
                    )
                )

            if dfs == []:
                print(f"No matching time data found for {model}")
                print(f"Contents of {model} dir:")
                print(f"\t{[f.name for f in (self.models_dir / model).iterdir()]}")

            else:
                df = pd.concat(dfs).reset_index()
                df = df.sort_values(by=["datetime"])
                output_filepath = (
                    self.out_dir / model / f"{model}_{admin_level_name}.csv"
                )
                df.to_csv(output_filepath)
                print(f"** Written {model} csv to {output_filepath.as_posix()} **")
                all_model_dfs.append(df)

        if all_model_dfs != []:
            all_model_df = pd.concat(all_model_dfs).reset_index()
            all_model_df = all_model_df.sort_values(by=["datetime"]).drop(
                columns=["index", "level_0"]
            )
            return all_model_df
        else:
            print("No DataFrames Created")
            return None

    def compute_metrics(
        self, compute_global_errors: bool = True, compute_regional_errors: bool = True
    ) -> None:
        if compute_global_errors:
            self.global_mean_metrics = self.compute_global_error_metrics()
            fname = f"global_error_metrics_{self.experiment}"
            fname += "_admin" if self.admin_boundaries else "_landcover"
            fname = f"{fname}.csv"
            self.global_mean_metrics.to_csv(self.out_dir / fname)
            print("\n* Assigned Global Error Metrics to `self.global_mean_metrics` *")
            print(f"* Written csv to data/analysis/region_analysis/{fname} *")

        if compute_regional_errors:
            self.regional_mean_metrics = self.compute_regional_error_metrics()
            fname = f"regional_error_metrics_{self.experiment}"
            fname += "_admin" if self.admin_boundaries else "_landcover"
            fname = f"{fname}.csv"
            self.regional_mean_metrics.to_csv(self.out_dir / fname)
            print(
                "\n* Assigned Regional Error Metrics to `self.regional_mean_metrics` *"
            )
            print(f"* Written csv to data/analysis/region_analysis/{fname} *")
