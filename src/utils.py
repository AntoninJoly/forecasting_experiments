import numpy as np
import pandas as pd
from darts.models import RNNModel

def arima_forecast(ARIMA_model, df, periods=24):
    fitted, confint = ARIMA_model.predict(n_periods=periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    return fitted_series, lower_series, upper_series


def sarimax_forecast(SARIMAX_model, df, periods=24):
    forecast_df = pd.DataFrame({"month_index":pd.date_range(df.index[-1], periods = periods, freq='MS').month},
                    index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS'))

    fitted, confint = SARIMAX_model.predict(n_periods=periods, return_conf_int=True, exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    
    return fitted_series, lower_series, upper_series

def metrics_darts(pred, act):
    from darts.metrics import mape, rmse, r2_score, mae, mse
    gt = act.slice_intersect(pred)
    pred = pred.slice_intersect(gt)
    res = {'mae': round(mae(gt, pred),3),
           'mse': round(mse(gt, pred),3),
           'rmse': round(rmse(gt, pred),3),
           'r2_score': round(r2_score(gt, pred),3)}
    return res

def metrics_sklearn(gt, pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    res = {'mae': round(mean_absolute_error(gt, pred),3),
           'mse': round(mean_squared_error(gt, pred, squared=True),3),
           'rmse': round(mean_squared_error(gt, pred, squared=False),3),
           'r2_score': round(r2_score(gt, pred),3)}
    return res

def fit_it(model, train, val, flavor):
    res = model.fit(train,
                    future_covariates=covariates,
                    val_series=val,
                    val_future_covariates=covariates,
                    verbose=False)
    return res

def fit_model(flavor, ts, train, val, covariates, periodicity):
    model = RNNModel(model=flavor,
                     # model_name=flavor + str(' RNN'),
                     input_chunk_length=periodicity,
                     training_length=6,
                     hidden_dim=20,
                     batch_size=16,
                     n_epochs=50,
                     dropout=0.2,
                     optimizer_kwargs={'lr': 1e-3},
                     log_tensorboard=False,
                     random_state=42,
                     force_reset=True)
    model.fit(train, future_covariates=covariates, val_series=val, val_future_covariates=covariates, verbose=False)
    return model

# def split_into_sequences(data, seq_len):
#     n_seq = len(data) - seq_len + 1
#     return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

# def get_train_test_sets(data, seq_len, train_frac):
#     sequences = split_into_sequences(data, seq_len)
#     n_train = int(sequences.shape[0] * train_frac)
#     x_train = sequences[:n_train, :-1, :]
#     y_train = sequences[:n_train, -1, :]
#     x_test = sequences[n_train:, :-1, :]
#     y_test = sequences[n_train:, -1, :]
#     return x_train, y_train, x_test, y_test