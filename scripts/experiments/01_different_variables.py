"""
run_different_variables.py

All vars:
['pev', 'sp', 't2m', 'tp', 'VCI', 'precip', 'ndvi', 'E', 'Eb', 'SMroot', 'SMsurf',]

# NOTE: p84.162 == 'vertical integral of moisture flux'

Experiment ['VCI', 'E'] Static: False
"""
from itertools import combinations
from pathlib import Path
from typing import List
import sys

sys.path.append("../..")

from _base_models import persistence, regression, linear_nn, rnn, earnn

from scripts.utils import _rename_directory, get_data_path
import logging

logging.basicConfig(filename="01_different_variables.log", level=logging.DEBUG)


def rename_model_experiment_file(
    data_dir: Path, vars_: List[str], static: bool
) -> None:
    vars_joined = "_".join(vars_)
    from_path = data_dir / "models" / "one_month_forecast"
    if static:
        to_path = data_dir / "models" / f"one_month_forecast_{vars_joined}_YESstatic"
    else:
        to_path = data_dir / "models" / f"one_month_forecast_{vars_joined}_NOstatic"

    # with_datetime ensures that unique
    _rename_directory(from_path, to_path, with_datetime=True)


def run_all_models_as_experiments(
    vars_to_include: List[str],
    ignore_vars: List[str],
    static: bool,
    run_regression: bool = True,
):
    print(f"Experiment {vars_to_include} Static: {static}")

    # RUN EXPERIMENTS
    if run_regression:
        regression(ignore_vars=ignore_vars, include_static=static)

    if static:
        # 'embeddings' or 'features'
        try:
            linear_nn(ignore_vars=ignore_vars, static="embeddings")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.debug(
                f"\n{'*'*10}\n FAILED: LinearNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )
            logging.debug(e)

        try:
            rnn(ignore_vars=ignore_vars, static="embeddings")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.debug(
                f"\n{'*'*10}\n FAILED: RNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )
            logging.debug(e)

        try:
            earnn(pretrained=False, ignore_vars=ignore_vars, static="embeddings")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.debug(
                f"\n{'*'*10}\n FAILED: EALSTM for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )
            logging.debug(e)

    else:
        try:
            linear_nn(ignore_vars=ignore_vars, static=None)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.debug(
                f"\n{'*'*10}\n FAILED: LinearNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )
            logging.debug(e)

        try:
            rnn(ignore_vars=ignore_vars, static=None)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.debug(
                f"\n{'*'*10}\n FAILED: RNN for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )
            logging.debug(e)

        try:
            # NO NEED to run the EALSTM without static data because
            # just equivalent to the RNN
            earnn(pretrained=False, ignore_vars=ignore_vars, static=None)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.debug(
                f"\n{'*'*10}\n FAILED: EALSTM for vars={vars_to_include} static={static}\n{'*'*10}\n"
            )
            logging.debug(e)

    # RENAME DIRECTORY
    data_dir = get_data_path().absolute()
    rename_model_experiment_file(data_dir, vars_to_include, static)
    print(f"Experiment {vars_to_include} finished")


def get_already_run(data_dir: Path, vars_joined: str, static: bool):
    already_run = [
        m
        for m in (data_dir / "models").glob(
            f"*{vars_joined}_{'YES' if static else 'NO'}static"
        )
    ]

    return already_run


if __name__ == "__main__":
    ignore_vars = None
    always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb"]
    all_vars = [
        "VCI",
        "precip",
        "t2m",
        "pev",
        "E",
        "SMsurf",
        "SMroot",
        "Eb",
        "sp",
        "tp",
        "ndvi",
        "p84.162",
    ]
    important_vars = ["VCI", "precip", "t2m", "pev", "E", "SMsurf", "SMroot"]
    additional_vars = ["precip", "t2m", "pev", "E", "SMsurf", "SMroot"]

    # create ALL combinations of features (VCI + ...)
    expts_to_run = []
    for n_combinations in range(len(additional_vars)):
        expts_to_run.extend(
            [["VCI"] + list(c) for c in combinations(additional_vars, n_combinations)]
        )

    # # add variables in one at a time
    # for i in range(len(important_vars)):
    #     vars_to_include = important_vars[:-(i)]
    #     if vars_to_include == []:
    #         continue

    for vars_to_include in expts_to_run[4:]:
        print(f'\n{"-" * 10}\nRunning experiment with: {vars_to_include}\n{"-" * 10}')

        vars_to_exclude = [v for v in important_vars if v not in vars_to_include]
        ignore_vars = always_ignore_vars + vars_to_exclude
        print(ignore_vars)

        # run experiments
        for static in [True, False]:
            vars_joined = "_".join(vars_to_include)
            already_run = get_already_run(data_dir, vars_joined, static)
            # CHECK if experiment has already been run
            if already_run != []:
                print(
                    f'\n{"-" * 10}\Experiment ALREADY RUN: {vars_to_include} static:{static}\n{"-" * 10}',
                    "Skipping to next experiment ...",
                )
                continue

            try:
                run_all_models_as_experiments(
                    vars_to_include, ignore_vars, static=static, run_regression=False
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(
                    f'\n{"-" * 10}\nExperiment FAILED: {vars_to_include} static:{static}\n{"-" * 10}'
                )
                logging.debug(
                    f'\n{"-" * 10}\nExperiment FAILED: {vars_to_include} static:{static}\n{"-" * 10}'
                )
                logging.debug(e)

                # SAVE DIRECTORY anyway (some models may have run)
                data_dir = get_data_path()
                rename_model_experiment_file(data_dir, vars_to_include, static)
                print(f"Experiment {vars_to_include} finished")
