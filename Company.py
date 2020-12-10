from abc import ABCMeta, abstractmethod, abstractproperty
class Asset(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_lagged_series(self, lags=5):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")

    def create_tehnical_indicators(self):
        raise NotImplementedError("Should implement generate_signals()")

    @abstractmethod
    def create_train_test_set(self,lags=5):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")
    
    @property
    @abstractmethod
    def lagged_returns_columns(self):
        pass

    @property
    @abstractmethod
    def technical_indicators_colums(self):
        pass

    @abstractproperty
    def symbol(self):
        pass

import numpy as np
import pandas as pd
from datetime import date
import copy
import os
class Company(Asset):
    __mask = np.vectorize(lambda x, threshold: x if np.isnan(x) or abs(x) > threshold else threshold)
    __CLOSE='Adj Close'
    
    def __init__(self, symbol, folder):
        self.__folder = folder
        self.__symbol = symbol
        self.__location = folder + '/' + symbol
        
        if not os.path.exists(self.__location):
            self.returns = pd.DataFrame()
        else:
            
            returns = pd.read_csv(self.__location, parse_dates=True)
            returns['Date'] = pd.to_datetime(returns['Date'], dayfirst=True)
            returns = returns.set_index('Date')
            returns['return'] = returns[self.__CLOSE].pct_change()*100.0
        
            self.returns = returns