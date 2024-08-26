"""Evaluation module for the project"""

import pandas as pd # type: ignore
from typing import Callable, Optional, Union, Dict, List
from tqdm.auto import tqdm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from darts import TimeSeries

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
            
        if self.model_type == 'autoreg':
            lags = ar_select_order(
                endog=y_train,
                exog=X_train,
                maxlag=12,
                trend='c'
            )
            
            model = AutoReg(
                endog=y_train,
                exog=X_train,
                trend='c',
                lags=lags.ar_lags
            ).fit()
            
            self.model = model
            
        if self.model_type == 'darts':
            darts_X_train = TimeSeries.from_dataframe(X_train)
            darts_Y_train = TimeSeries.from_series(y_train)
            
            self.model.fit(
                series=darts_Y_train,
                past_covariates=darts_X_train
            )

    def forecast(
        self,
        steps,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
    ):
        if self.model_type == "autots":
            return self.model.predict(forecast_length=steps).forecast[
                self.target_column_name
            ]

        if self.model_type == 'autoreg':
            fcst = self.model.predict(
                exog_oos=X_test,
                start=len(X_train),
                end=len(X_train) + steps - 1
            )
            
            return fcst 
        
        if self.model_type == 'darts':
            darts_X_test = TimeSeries.from_dataframe(X_test) # type: ignore
            return self.model.predict(
                n=steps,
                past_covariates=darts_X_test
            ).pd_series()
        
        
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
        forecast_horizon: int = 3,
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
                print("Skipping block due to insufficient Y")
                continue
            
            # fit the model
            model.fit(X_train, y_train)

            if model.model_type == 'darts': # Expand the X_test to include the past covariates for all the model lags
                max_lag = min(model.model.lags['past'])
                lags_df = X_train[max_lag-1:]
                X_test = pd.concat([lags_df, X_test], axis=0)
                X_test = X_test.sort_index()
            
            forecast = model.forecast(steps=forecast_horizon, X_test=X_test, X_train=X_train) # type: ignore
            score = float(loss_function(y_test[:forecast_horizon], forecast))

            # Prepend the final training real result to the forecast at the beginning
            y_train_final = pd.Series(y_train.iloc[-1], index=[y_train.index[-1]])
            forecast = pd.concat([y_train_final, forecast], axis=0)
            forecast = forecast.sort_index()
            
            predictions.append(forecast)
            
            scores.append(score)

        return {
            "scores": scores,
            "predictions": predictions,
        }
