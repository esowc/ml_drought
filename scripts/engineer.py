import sys

sys.path.append("..")

from src.engineer import Engineer
from scripts.utils import get_data_path


def engineer(
    experiment="one_month_forecast",
    process_static=True,
    pred_months=12,
    test_years=[y for y in range(2011, 2019)],
):

    engineer = Engineer(
        get_data_path(), experiment=experiment, process_static=process_static
    )
    engineer.engineer(
        test_year=[y for y in range(2015, 2019)],
        target_variable="VCI",
        pred_months=pred_months,
        expected_length=pred_months,
    )


def engineer_static():
    Engineer.engineer_static_only(get_data_path())


if __name__ == "__main__":
    engineer(pred_months=3, process_static=False)
    # engineer(pred_months=12, experiment='nowcast')
    # engineer_static()
