import os
import pandas as pd
import numpy as np

import datetime
import time
from pandas.tseries.offsets import BDay

import ast
from pathlib import Path

from utils import (timeit,print_info)
from Industry import Industry

class Environment():

    __LR_DATA_FOLDER='./industry_data_lag/'
    __TI_DATA_FOLDER='./industry_data/'
    __INDUSTRY_GROUPING_FILE ='../data/company_data.csv'
    __SINGLE_FOLDER = '/single_data/'
    __GICS_SECTOR = 'GICS Sector'

    __parameters = {}
    __non_empty_companies = []
    
    def __set_data_folder_path(self):
        
        if not os.path.exists(self.__INDUSTRY_GROUPING_FILE):
            self.__INDUSTRY_GROUPING_FILE = './data/company_data.csv'
            
    def __get_file_names(self, companies_mask):
        __DATA_FOLDER='./data/'
        file_names = []
        for files in os.listdir(__DATA_FOLDER):
            if files.endswith(".csv") and ("SP500_FULL_NATIVE" not in files) and ("sp500_cet" not in files):
                if (len(companies_mask) != 0):
                    if (files in companies_mask):
                        file_names.append(files)
                else:
                    file_names.append(files)
        return file_names
    
    def __init__(self, parameters, companies_mask=[], industry=None):
        self.__parameters = parameters
        self.__test_folder = parameters['method'] + '/test/'
        self.__walks = 0
        SYMBOL='Symbol'
            
        self.__set_data_folder_path()
        file_names= self.__get_file_names(companies_mask)

        grouped_companies_data = pd.read_csv(self.__INDUSTRY_GROUPING_FILE, ',', parse_dates=True)
        grouped_companies_data[SYMBOL]= grouped_companies_data[SYMBOL].apply(lambda x: "{0}.csv".format(x))
        
        if (industry is not None):
            if  industry not in grouped_companies_data[self.__GICS_SECTOR].unique():
                raise ValueError('Invalid industry')
            grouped_companies_data = grouped_companies_data.loc[(grouped_companies_data[self.__GICS_SECTOR] == industry) & (grouped_companies_data[SYMBOL].isin(file_names))]
        self._active_industry = industry
        grouped_companies_data = grouped_companies_data.loc[(grouped_companies_data[SYMBOL].isin(file_names))]

        grouped_companies_data.set_index(SYMBOL, inplace=True)

        self._industry_information = grouped_companies_data[[self.__GICS_SECTOR]]
        self.__set_file_names(self._industry_information.index.values, companies_number = parameters['companies'])
        
    
    def __set_file_names(self, file_names, companies_number=0):
        if (companies_number == 0):
            self.__companies_no = len(file_names)
        else:
            self.__companies_no = companies_number
        industry = self.industries[0]
        self.__file_names = self._industry_information[self._industry_information[self.__GICS_SECTOR]==industry].index[:self.__companies_no].values
            
    
    @property
    def industries(self):
        if self._active_industry is None:
            return self._industry_information[self.__GICS_SECTOR].unique()
        else:
            return [self._active_industry]*1

    @property
    def company_names(self):
        return self.__file_names

    @property
    def data_folder(self):
        return  self.__parameters['method'] + self.__SINGLE_FOLDER
    
    @property
    def companies_number(self):        
        return self.__companies_no

    @property
    def test_folder(self):
        return self.__test_folder

    @property
    def output_folder(self):
        output_folder =   self.__parameters['method'] + "/StatisticalArbitrage/" + self.__parameters["start_date"].strftime('%Y-%m-%d') + "_ARIMASINGLE280_output"
        my_file = Path(output_folder)
        if not my_file.exists():
            os.mkdir(output_folder)
        return output_folder

    @property
    def walk(self):
        return self.__walks
        
    @property
    def has_walk(self):
        if hasattr(self,"_next_date") and self._next_date is not None:
            self.__parameters["start_date"] = self._next_date
        result = self.__parameters["start_date"].year < 2018 and (self.__walks < self.__parameters['walks'] or self.__parameters['walks'] == 0)
        return result

    def companies_by_industry(self, industry):
        return self._industry_information.loc[self._industry_information[self.__GICS_SECTOR] == industry].index.intersection(self.__file_names).values
        
    def get_LR_and_trunk_it_before_start_date(self, fnombre, start_date):
        return self.__get_data_and_trunk_it_before_start_date( fnombre, self.__LR_DATA_FOLDER,"LR", start_date)
        
    def get_TI_and_trunk_it_before_start_date(self, fnombre, start_date):
        return self.__get_data_and_trunk_it_before_start_date(fnombre,self.__TI_DATA_FOLDER,"TI", start_date)
        
    def __get_data_and_trunk_it_before_start_date(self, fnombre,folder, data_type, start_date):
        if ('.csv' not in fnombre):
            fnombre = fnombre + ".csv"
            
        df = self.__read_dataframe_for_company(folder,fnombre)
        print_info("File location {0} for industry {1}".format( folder + fnombre, fnombre), file="stdout", flush=True)
        df['Industry'] = fnombre
    
        
        self.__parameters["start_date"] = start_date
        
        start_date = self.__parameters["start_date"]
        days_lag = self.__parameters['lag'] #account for technical indicators which should be otherwise considered
        nr_forecasting = self.__parameters["test"]
        days_training = self.__parameters["train"]
        days_validation = self.__parameters['val']

        print("TRONCO PRIMA DI START_DATE:{0}".format(start_date))

        first_date_index = df[df.Date.gt(start_date - pd.DateOffset(1))].index[0]
        first_date = df.iloc[first_date_index]['Date']

        begin = df.iloc[max(first_date_index,0)]['Date']
        end = start_date + (days_training + days_validation + nr_forecasting) * BDay()
    
        df = df.set_index('Date')
        df.sort_index(inplace=True)
        df= df.truncate(before=begin)
        df = df.dropna()

        dates = np.sort(np.unique(df.index.values))
        dates = dates[0:days_training + days_validation + nr_forecasting]
        df = df.loc[dates]


        write_path = self.get_write_folder(data_type, fnombre)
        if ((not df.empty) and (first_date >= begin) and (first_date <= end) and (df.shape[0] >= days_lag + days_training + nr_forecasting)): 
            df.to_csv(write_path)
            print_info('Set length {0} for company {1}'.format(df.shape,fnombre), file="stdout", flush=True)
    
        if (len(df.index) >= nr_forecasting + days_lag):
            return pd.Timestamp(dates[nr_forecasting]), df
        else:
            return pd.datetime.max, df

    def get_write_folder(self, data_type, fnombre):
        if ('.csv' not in fnombre):
            fnombre = fnombre + '.csv'
            
        write_path = self.data_folder + data_type + "_" + fnombre.replace(" ", "_")
        print_info('Write path {0}'.format(write_path), file="stdout", flush=True)
        if os.path.exists(write_path):
            os.remove(write_path)
        return write_path

    def __read_dataframe_for_company(self, folder, fnombre):
        fnombre = fnombre.replace(" ", "_")
        location = folder + fnombre
        if os.path.exists(location):
            df = pd.read_csv(location, ',', parse_dates=True)
            df['Date'] = df.Date.astype('datetime64')
            return df
        return pd.DataFrame()

    @timeit
    def create_feature_dataframe(self, **kwargs):
        print('Building features dataframe')
        
        print_info('TRUNCATE BEFORE {0}'.format(self.__parameters["start_date"]), file="stdout", flush=True )
        dates = []
        feature_names = []
        feature_names2=[]
        non_empty_industries = []
        empty_industries = []
        empty_companies=[]
        
        for industry in self.industries:
            
            data_LR, LR_df = self.get_LR_and_trunk_it_before_start_date(industry, self.__parameters["start_date"])
            data_TI, TI_df = self.get_TI_and_trunk_it_before_start_date(industry, self.__parameters["start_date"])
            assert data_LR == data_TI, "data mismatch"
            dates.append(data_LR)

            asset = Industry(industry, self.data_folder)
            df_LR = asset.create_lagged_series(LR_df,self.__parameters['lag'])
            df_TI = asset.create_tehnical_indicators(TI_df)

            if df_LR is not None and df_LR.empty == False and df_TI is not None and df_TI.empty == False:

                assert df_LR.shape[0]>= self.__parameters["train"] + self.__parameters["test"] + self.__parameters["val"], "Invalid shape for {0}".format(industry)
                print_info('Set length LR {0} for industry {1}'.format(df_LR.shape, industry), file="stdout", flush=True)
                print_info('Set length TI {0} for industry {1}'.format( df_TI.shape, industry), file="stdout", flush=True)
                non_empty_industries.append(industry)
                feature_names.append(asset.lagged_returns_columns)
                feature_names2.append(asset.technical_indicators_colums)
                write_path_lr = self.get_write_folder("LR", industry)
                df_LR.to_csv(write_path_lr, index=False)
                write_path_ti = self.get_write_folder("TI", industry)
                df_TI.to_csv(write_path_ti,  index=False)
        
            else:
                empty_industries.append(industry)
                print_info( "Empty set for {0}".format (industry), file="stdout", flush=True)

        next_date = min(dates)
        feature_names = np.unique(feature_names)
        print_info("Empty set for {0}".format( empty_industries), file="stdout", flush=True)
        print_info("Not empty set for {0}".format(non_empty_industries), file="stdout", flush=True)
        self._next_date = next_date
        self.__walks = self.__walks + 1
        return {'LR':feature_names, 'TI':np.unique(feature_names2)}


    def write_predictions_to_test(self, title, data: pd.DataFrame):
        data.to_csv("{0}{1}.csv".format(self.__test_folder,title))