"""Evaluation module for the project"""

import pandas as pd # type: ignore
from typing import Callable, Optional, Union, Dict, List
from tqdm.auto import tqdm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

class ModelWrapper:
    def __init__(self, model, model_type, target_column_name):
        self.model = model
        self.model_type = model_type
        self.target_column_name = target_column_name

    def fit(self, X_train, y_train):
        if self.model_type == "autots":
            # Combine the X_train and y_train along columns into a single dataframe
            train_df = pd.concat([X_train, y_train], axis=1)
            self.model.fit_data(train_df) # Fit the model to the new data without retraining the whole thing.
            
        if self.model_type == 'AutoReg':
            lags = ar_select_order(
                endog=y_train,
                exog=X_train,
                maxlag=12,
                trend='ct'
            )
            
            model = AutoReg(
                endog=y_train,
                exog=X_train,
                trend='ct',
                lags=lags.ar_lags
            ).fit()
            
            self.model = model
            

    def forecast(
        self,
        steps,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ):
        if self.model_type == "autots":
            return self.model.predict(forecast_length=steps).forecast[
                self.target_column_name
            ]

        if self.model_type == 'AutoReg':
            return self.model.predict(
                exog_oos=X_test,
                start=len(self.model.fittedvalues),
                end=len(self.model.fittedvalues) + steps - 1
            )
        
    def __repr__(self) -> str:
        try:
            return self.model.__repr__()
        except TypeError:
            return str(self.model)

class Evaluator:
    def __init__(self):
        pass

    def prequential_block(
        self,
        model: ModelWrapper,
        loss_function: Callable,
        features: pd.Series | pd.DataFrame,
        targets: pd.Series | pd.DataFrame,
        block_size: int = 1,
        forecast_horizon: int = 1,
        train_proportion: float = 0.7,
    ) -> Dict[str, List[float | pd.Series]]:
        """
        Evaluate a model prequentially with a block size and a forecast horizon
        """

        assert len(features) == len(targets)

        n_train = int(train_proportion * len(features))
        n_test = len(features) - n_train

        scores = []
        predictions = []
        for i in tqdm(range(0, n_test, block_size), desc='Running evaluation over expanding blocks'):
            # split data into train and test
            X_train, X_test = (
                features.iloc[: n_train + i],
                features.iloc[n_train + i : n_train + i + forecast_horizon],
            )
            
            y_train, y_test = (
                targets.iloc[: n_train + i],
                targets.iloc[n_train + i : n_train + i + forecast_horizon],
            )

            if len(y_test) < forecast_horizon:
                continue
            
            # fit the model
            model.fit(X_train, y_train)

            forecast = model.forecast(steps=forecast_horizon, X_test=X_test)
            
            score = float(loss_function(y_test[:forecast_horizon], forecast))
                        
            predictions.append(forecast)
            
            scores.append(score)

        return {
            "scores": scores,
            "predictions": predictions,
        }
