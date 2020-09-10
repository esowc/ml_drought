import shutil
from pathlib import Path
import time
import json
import pandas as pd
import re
from datetime import datetime


def get_data_path() -> Path:
    # if the working directory is alread ml_drought don't need ../data
    if "/home/tommy" in Path(".").absolute().as_posix():
        # on AWS machine
        data_path = Path("/cats/datastore/data")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
    else:
        if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
            data_path = Path("data")
        elif Path(".").absolute().as_posix().split("/")[-3] == "ml_drought":
            data_path = Path("../../data")
        else:
            data_path = Path("../data")
    return data_path


def _rename_directory(
    from_path: Path, to_path: Path, with_datetime: bool = False
) -> None:
    if with_datetime:
        dt = time.gmtime()
        dt_str = f"{dt.tm_year}_{dt.tm_mon:02}_{dt.tm_mday:02}:{dt.tm_hour:02}{dt.tm_min:02}{dt.tm_sec:02}"
        name = "/" + dt_str + "_" + to_path.as_posix().split("/")[-1]
        to_path = "/".join(to_path.as_posix().split("/")[:-1]) + name
        to_path = Path(to_path)
    shutil.move(from_path.as_posix(), to_path.as_posix())
    print(f"MOVED {from_path} to {to_path}")


def get_results(model_dir: Path, print_output: bool = True) -> pd.DataFrame:
    """ Display the results from the results.json """

    def _get_persistence_for_group(x):
        return x.loc[x.model == "previous_month"].total_rmse

    # create a dataframe for the results in results.json
    result_paths = [p for p in model_dir.glob("*/*/results.json")]
    date_regex = r"\d{4}_\d{2}_\d{2}:\d{6}_"
    experiments = [re.sub(date_regex, "", p.parents[1].name) for p in result_paths]
    df = pd.DataFrame({"experiment": experiments})

    # match the date_str if in the experiment name
    date_matches = [re.match(date_regex, p.parents[1].name) for p in result_paths]

    datetimes = []
    for dt in date_matches:
        if dt is None:
            datetimes.append(None)
        else:
            datetimes.append(
                pd.to_datetime(datetime.strptime(dt.group(), "%Y_%m_%d:%H%M%S_"))
            )

    df["time"] = datetimes

    df["model"] = [p.parents[0].name for p in result_paths]
    result_dicts = [json.load(open(p, "rb")) for p in result_paths]
    df["total_rmse"] = [d["total"] for d in result_dicts]

    persistence_rmses = (
        df.groupby("experiment").apply(_get_persistence_for_group).reset_index()
    )

    # merge values for that experiment
    df = df.merge(persistence_rmses.drop(columns="level_1"), on="experiment").rename(
        columns=dict(total_rmse_y="previous_month_score")
    )

    #
    df["outperform_baseline"] = df.total_rmse_x < df.previous_month_score

    if print_output:
        for i, row in df.iterrows():
            persistence_score = (
                persistence_rmses["total_rmse"]
                .loc[persistence_rmses.experiment == row.experiment]
                .values
            )

            print(
                f"Experiment: {row.experiment}\n"
                f"Model: {row.model}\n"
                f"Persistence RMSE: {persistence_score}\n"
                f"RMSE: {row.total_rmse}\n"
            )

    return df


def _base_increment_folder(data_path: Path, dir: str):
    old_paths = [d for d in data_path.glob(f"*_{dir}*")]
    if old_paths == []:
        integer = 0
    else:
        old_max = max([int(p.name.split("_")[0]) for p in old_paths])
        integer = old_max + 1

    _rename_directory(
        from_path=data_path / f"{dir}",
        to_path=data_path / f"{integer}_{dir}",
        with_datetime=False,
    )


def rename_features_dir(data_path: Path, dir: str):
    """increment the features dir by 1"""
    _base_increment_folder(data_path, dir="features")


def rename_models_dir(data_path: Path):
    """increment the models dir by 1"""
    _base_increment_folder(data_path, dir="models")
