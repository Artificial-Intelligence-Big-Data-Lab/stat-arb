import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

class CustomFixTimeSeriesSplit:

        def split(self, series: pd.DataFrame):
                copy = series.copy()
                copy['Date_int'] = series.Date.dt.strftime("%Y%m%d").astype(int)
                dates = np.sort(copy.Date_int.unique())
                length = len(dates)
                
                windows_length = int(length / 3)
                cv2 = [
                        (copy.loc[copy['Date_int'].isin(dates[0 : 2 * windows_length])].index.values, copy.loc[copy['Date_int'].isin(dates[2 * windows_length:length])].index.values),
                        (copy.loc[copy['Date_int'].isin(dates[0 : windows_length])].index.values, copy.loc[copy['Date_int'].isin(dates[windows_length : 2 * windows_length])].index.values),
                        (copy.loc[copy['Date_int'].isin(dates[0:windows_length])].index.values, copy.loc[copy['Date_int'].isin(dates[windows_length:2*windows_length])].index.values),
                        (copy.loc[copy['Date_int'].isin(dates[windows_length:2 * windows_length])].index.values, copy.loc[copy['Date_int'].isin(dates[2 * windows_length:length])].index.values),
                        (copy.loc[copy['Date_int'].isin(dates[0 : int(length/2)])].index.values, copy.loc[copy['Date_int'].isin(dates[int(length/2):length])].index.values)
                ]
                return cv2

class CustomTimeSeriesSplit:
        def __init__(self, n_splits=5, max_train_size=None):
                self.n_splits = n_splits
                self.max_train_size = max_train_size
                
        
        def split(self, series: pd.DataFrame):
                copy = series.copy()
                copy.reset_index(inplace=True, drop=True)
                copy['Date_int'] = copy.Date.dt.strftime("%Y%m%d").astype(int)
                dates = np.sort(copy.Date_int.unique())
                tcsv = TimeSeriesSplit( n_splits=self.n_splits, max_train_size= self.max_train_size )
                for train_index, test_index in tcsv.split(dates):
                        yield (copy.loc[copy['Date_int'].isin(dates[train_index])].index.values,copy.loc[copy['Date_int'].isin(dates[test_index])].index.values)