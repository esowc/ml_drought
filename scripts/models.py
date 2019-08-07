import sys
sys.path.append('..')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

from src.analysis import plot_shap_values
from src.models import (Persistence, LinearRegression,
                        LinearNetwork, RecurrentNetwork,
                        EARecurrentNetwork, load_model)
from src.models.neural_networks.base import NNBase
from src.models.data import DataLoader


idx_to_input = {
    0: 'historical',
    1: 'pred_months',
    2: 'latlons',
    3: 'current',
    4: 'yearly_aggs',
    5: 'static'
}


def _make_nn_input(x, start_idx):
    """
    Returns a list of tensors, as is required
    by the shap explainer
    """

    output_tensors = []
    output_tensors.append(x.historical[start_idx: start_idx + 3])
    # one hot months
    one_hot_months = NNBase._one_hot_months(x.pred_months[start_idx: start_idx + 3])
    output_tensors.append(one_hot_months[start_idx: start_idx + 3])
    output_tensors.append(x.latlons[start_idx: start_idx + 3])
    if x.current is None:
        output_tensors.append(torch.zeros(3, 1))
    else:
        output_tensors.append(x.current[start_idx: start_idx + 3])
    # yearly aggs
    output_tensors.append(x.yearly_aggs[start_idx: start_idx + 3])
    # static data
    if x.static is None:
        output_tensors.append(torch.zeros(3, 1))
    else:
        output_tensors.append(x.static[start_idx: start_idx + 3])
    return output_tensors


def parsimonious(
    experiment='one_month_forecast',
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = Persistence(data_path, experiment=experiment)
    predictor.evaluate(save_preds=True)


def regression(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearRegression(
        data_path, experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    experiment=experiment,
                                    shuffle_data=False, mode='test')
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explain_hist, explain_add = predictor.explain(val.x)

    np.save('shap_regression_historical.npy', explain_hist)
    np.save('shap_regression_add.npy', explain_add)
    np.save('shap_x_hist.npy', val.x.historical)
    np.save('shap_x_add.npy', val.x.pred_months)

    with open('variables.txt', 'w') as f:
        f.write(str(val.x_vars))

    # plot the variables
    with (data_path / f'features/{experiment}/normalizing_dict.pkl').open('rb') as f:
        normalizing_dict = pickle.load(f)

    for variable in val.x_vars:
        plt.clf()
        plot_shap_values(
            val.x.historical[0], explain_hist[0], val.x_vars,
            normalizing_dict, variable, normalize_shap_plots=True,
            show=False
        )
        plt.savefig(f'{variable}_linear_regression.png', dpi=300, bbox_inches='tight')


def linear_nn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = LinearNetwork(
        layer_sizes=[100], data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # The code below is commented out because of a bug in the shap deep Explainer which
    # prevents it from working. It has been fixed in master, but not deployed yet:
    # https://github.com/slundberg/shap/pull/684

    # test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
    #                                 shuffle_data=False, mode='test', to_tensor=True)
    # key, val = list(next(iter(test_arrays_loader)).items())[0]
    #
    # explain_hist, explain_add = predictor.explain([val.x.historical[:3], val.x.pred_months[:3]])
    # print(explain_hist.shape)
    # np.save('shap_linear_network_hist.npy', explain_hist)
    # np.save('shap_linear_network_add.npy', explain_add)
    #
    # np.save('shap_x_linear_network_hist.npy', val.x.historical[:3])
    # np.save('shap_x_linear_network_add.npy', val.x.pred_months[:3])
    #
    # with open('variables_linear_network.txt', 'w') as f:
    #     f.write(str(val.x_vars))
    #
    # # plot the variables
    # with (data_path / 'features/normalizing_dict.pkl').open('rb') as f:
    #     normalizing_dict = pickle.load(f)
    #
    # for variable in val.x_vars:
    #     plt.clf()
    #     plot_shap_values(val.x[0].numpy(), explain_hist[0], val.x_vars, normalizing_dict, variable,
    #                      normalize_shap_plots=True, show=False)
    #     plt.savefig(f'{variable}_linear_network.png', dpi=300, bbox_inches='tight')


def rnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels
    )
    predictor.train(num_epochs=50, early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    # See above; we need to update the shap version before this can be explained


def earnn(
    experiment='one_month_forecast',
    include_pred_month=True,
    surrounding_pixels=1,
    pretrained=True
):
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=128,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels
        )
        predictor.train(num_epochs=50, early_stopping=5)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f'models/{experiment}/ealstm/model.pkl')

    # See above; we need to update the shap version before this can be explained
    test_arrays_loader = DataLoader(data_path=data_path, batch_file_size=1,
                                    shuffle_data=False, mode='test', to_tensor=True,
                                    static=True)
    key, val = list(next(iter(test_arrays_loader)).items())[0]

    explain_inputs = _make_nn_input(val.x, start_idx=0)
    explain_arrays = predictor.explain(explain_inputs)
    for idx, shap_array in enumerate(explain_arrays):
        np.save(f'shap_ealstm_valie_{idx_to_input[idx]}.npy', shap_array)
        np.save(f'shap_ealstm_input_{idx_to_input[idx]}.npy', explain_inputs[idx].cpu().numpy())


if __name__ == '__main__':
    # parsimonious()
    # regression()
    # linear_nn()
    # rnn()
    earnn(pretrained=True)
