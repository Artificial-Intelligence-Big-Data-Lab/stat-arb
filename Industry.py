import numpy as np
import pandas as pd
from datetime import date
import copy
import os
from Company import Asset
from abc import ABCMeta, abstractmethod, abstractproperty

class Industry(Asset):

    def __init__(self, symbol, folder):
        self.__folder = folder
        if ('.csv' not in symbol):
            symbol = symbol + '.csv'
        self.__symbol = symbol
        self.__LR_columns = []
        self._TI_columns = []
        self.index_column = 'Date'
        self.categorical_column = 'Asset_ID'
        self.label = 'OC'
        self.return_label = 'OC Percent'
    
    def __read_data(self, data_type):
        self.__location = self.__folder + '/' + data_type+"_"+self.__symbol.replace(" ","_")
        
        if not os.path.exists(self.__location):
            return pd.DataFrame()
        else:
            
            returns = pd.read_csv(self.__location, parse_dates=True)
            returns['Date'] = pd.to_datetime(returns['Date'], dayfirst=True)
        return returns

    @property
    def symbol(self):
        return self.__symbol

    @property
    def lagged_returns_columns(self):
        return self.__LR_columns

    @property
    def technical_indicators_colums(self):
        return self._TI_columns


    def create_lagged_series(self, series_orig:pd.DataFrame=pd.DataFrame(), lags=5):
        series = pd.DataFrame()
        
        if (series_orig is None or series_orig.empty==True):
            series_orig = self.__read_data("LR")

        if (series_orig.empty == True):
            return series
        series_orig = series_orig.reset_index()
        series['Date'] = series_orig['Date']
        series['Open'] = series_orig['Open']
        series['Close']=series_orig['Close']
        series[self.label] = series_orig[self.label]
        series[self.return_label]=series_orig[self.return_label]
        excluded_columns = ['Title','Open','Close']
        categories = {k: v for v, k in enumerate(series_orig['Title'].unique())}

        lag_days_array = np.array(np.arange(1, lags+1))
        for lag in lag_days_array:
            close_column_name = "Close_lag_{0}".format(lag)
            open_column_name = "Open_lag_{0}".format(lag)
            result_column_name = "OC_Percent_lag_{0}".format(lag)
            series[result_column_name] = (series_orig[close_column_name] - series_orig[open_column_name]) / series_orig[open_column_name]
            
        series['Title'] = series_orig['Title']
        series['Asset_ID'] = series['Title'].map(categories)
        series['Asset_ID'] = series['Asset_ID'].astype('category')
        
        train_cols = [col for col in series.columns if col not in [self.categorical_column, self.index_column, self.label, self.return_label] and col not in excluded_columns ]
        self.companies = series.Title.unique()
        self.__LR_columns = train_cols
        return series

    def create_tehnical_indicators(self,series_orig:pd.DataFrame=pd.DataFrame()):

        series = pd.DataFrame()

        if (series_orig is None or series_orig.empty==True):
            series_orig = self.__read_data("TI")

        if (series_orig.empty == True):
            return series

        series_orig = series_orig.reset_index()    
        series['Date'] = series_orig['Date']
        series['Open'] = series_orig['Open']
        series['Close']=series_orig['Close']
        series[self.label] = series_orig[self.label]
        series[self.return_label]=series_orig[self.return_label]
        categories = {k: v for v, k in enumerate(series_orig['Title'].unique())}

    
        series['AvgExp'] = series_orig['AvgExp']
        series['%K'] = series_orig['%K']
        series['ROC'] = series_orig['ROC']
        series['RSI'] = series_orig['RSI']
        series['ACCUMULATION_DISTRIBUTION'] = series_orig['ACCUMULATION_DISTRIBUTION']
        series['MACD'] = series_orig['MACD']
        series['WILLIAMS'] = series_orig['WILLIAMS']
        series['DISPARITY_5'] = series_orig['DISPARITY_5']
        series['DISPARITY_10'] = series_orig['DISPARITY_10']
        series['Title'] = series_orig['Title']
        series['Asset_ID'] = series['Title'].map(categories)
        series['Asset_ID'] = series['Asset_ID'].astype('category')
        excluded_columns = ['Title', 'Open', 'Close']

        train_cols = [col for col in series.columns if col not in [self.categorical_column, self.index_column, self.label, self.return_label] and col not in excluded_columns ]

        self._TI_columns=train_cols
        return series

    