import numpy as np
from pathlib import Path
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import shap

from typing import Any, Dict, Tuple, Optional

from .base import ModelBase
from .data import DataLoader, train_val_mask


class LinearRegression(ModelBase):

    model_name = 'linear_regression'

    def __init__(self, data_folder: Path = Path('data'),
                 batch_size: int = 1, num_epochs: int = 1,
                 early_stopping: Optional[int] = None) -> None:
        super().__init__(data_folder, batch_size)

        self.num_epochs = num_epochs
        self.early_stopping = early_stopping
        self.explainer: Optional[shap.LinearExplainer] = None

    def train(self) -> None:
        print(f'Training {self.model_name}')

        if self.early_stopping is not None:
            len_mask = len(DataLoader._load_datasets(self.data_path, mode='train',
                                                     shuffle_data=False))
            train_mask, val_mask = train_val_mask(len_mask, 0.3)

            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train', mask=train_mask)
            val_dataloader = DataLoader(data_path=self.data_path,
                                        batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='train', mask=val_mask)
            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train')
        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor()

        for epoch in range(self.num_epochs):
            train_rmse = []
            for x, y in train_dataloader:
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                self.model.partial_fit(x, y.ravel())

                train_pred_y = self.model.predict(x)
                train_rmse.append(np.sqrt(mean_squared_error(y, train_pred_y)))
            if self.early_stopping is not None:
                val_rmse = []
                for x, y in val_dataloader:
                    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                    val_pred_y = self.model.predict(x)
                    val_rmse.append(np.sqrt(mean_squared_error(y, val_pred_y)))

            print(f'Epoch {epoch + 1}, train RMSE: {np.mean(train_rmse)}')

            if self.early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f'Val RMSE: {epoch_val_rmse}')
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == self.early_stopping:
                        print('Early stopping!')
                        return None

    def explain(self, x: Any) -> np.ndarray:

        if self.model is None:
            self.train()

        if self.explainer is None:
            mean = self._calculate_big_mean()
            self.explainer: shap.LinearExplainer = shap.LinearExplainer(
                self.model, (mean, None), feature_dependence='independent')

        return self.explainer.shap_values(x)

    def save_model(self) -> None:

        if self.model is None:
            self.train()
            self.model: linear_model.LinearRegression

        coefs = self.model.coef_
        np.save(self.model_dir / 'model.npy', coefs)

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        test_arrays_loader = DataLoader(data_path=self.data_path, batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='test')

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        if self.model is None:
            self.train()
            self.model: linear_model.SGDRegressor

        for dict in test_arrays_loader:
            for key, val in dict.items():
                preds = self.model.predict(val.x.reshape(val.x.shape[0],
                                                         val.x.shape[1] * val.x.shape[2]))
                preds_dict[key] = preds
                test_arrays_dict[key] = {'y': val.y, 'latlons': val.latlons}

        return test_arrays_dict, preds_dict

    def _calculate_big_mean(self) -> np.ndarray:
        """
        Calculate the mean of the training data in batches.

        For now, we don't calculate the covariance matric, since it wouldn't fit in
        memory either
        """
        train_dataloader = DataLoader(data_path=self.data_path,
                                      batch_file_size=1,
                                      shuffle_data=False, mode='train')

        means, sizes = [], []
        for x, _ in train_dataloader:
            # first, flatten x
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
            sizes.append(x.shape[0])
            means.append(x.mean(axis=0))

        total_size = sum(sizes)
        weighted_means = [mean * size / total_size for mean, size in zip(means, sizes)]
        return sum(weighted_means)
