import sys

sys.path.append("../..")

from _base_models import parsimonious, regression, linear_nn, rnn, earnn
from scripts.utils import _rename_directory, get_data_path

if __name__ == "__main__":
    # important_vars = ["VCI", "precip", "t2m", "pev", "p0005", "SMsurf", "SMroot"]
    # always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "E", "p0001"]
    important_vars = ["precip", "t2m", "pev", "E", "SMsurf", "SMroot"]
    always_ignore_vars = ["ndvi", "p84.162", "sp", "tp", "Eb", "VCI"]

    parsimonious()
    # regression(ignore_vars=always_ignore_vars)
    # gbdt(ignore_vars=always_ignore_vars)
    # linear_nn(ignore_vars=always_ignore_vars)
    # rnn(ignore_vars=always_ignore_vars)
    earnn(
        experiment="one_month_forecast",
        include_pred_month=True,
        surrounding_pixels=None,
        pretrained=False,
        explain=False,
        static="features",
        ignore_vars=always_ignore_vars,
        num_epochs=50,
        early_stopping=5,
        hidden_size=256,
        static_embedding_size=64,
    )

    # rename the output file
    data_path = get_data_path()

    _rename_directory(
        from_path=data_path / "models" / "one_month_forecast",
        to_path=data_path / "models" / "one_month_forecast_NO_VCI",
    )
