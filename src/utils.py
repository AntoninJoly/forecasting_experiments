import numpy as np
import pandas as pd

def forecast(ARIMA_model, periods=24):
    fitted, confint = ARIMA_model.predict(n_periods=periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    return fitted_series, lower_series, upper_series


def sarimax_forecast(SARIMAX_model, periods=24):
    forecast_df = pd.DataFrame({"month_index":pd.date_range(df.index[-1], periods = periods, freq='MS').month},
                    index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS'))

    fitted, confint = SARIMAX_model.predict(n_periods=periods, return_conf_int=True, exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = periods, freq='MS')
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    
    return fitted_series, lower_series, upper_series