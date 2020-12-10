import datetime

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


import read_parameters as r
from utils import (setup_folders,_cleanup_test_folder,safe_print,print_info)
from environment import Environment

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from modeling import Models
from os import path
from statArbRegression import StatArbRegression

from collections import namedtuple

Set = namedtuple('Set', ['X', 'label'])

if __name__ == "__main__":

    prediction_method, prediction_params = r.get_parameters()

    # strategy1 = StatArbRegression(pd.DataFrame(), 'Predicted_OC_perc',folder='../GB/StatisticalArbitrage/2003-01-01_ARIMASINGLE280_output/')
    # strategy1.compute_metrics(output_folder = '../GB/test')
    # strategy1.generate_signals(output_folder = '../GB/test')
    # strategy1.plot_returns(outputfolder='../GB/test', parameters=prediction_params)

    
    env = Environment(prediction_params)
    
    setup_folders(prediction_params["method"])

    print(prediction_params)

    next_date = prediction_params["start_date"]
    
    print(env.companies_number)

    overall_bias_variance = pd.DataFrame()
    overall_chosen_methods = pd.DataFrame()

    
    while env.has_walk:

        _cleanup_test_folder(env.test_folder, env.company_names)
        _cleanup_test_folder(env.data_folder, env.company_names)

        features = env.create_feature_dataframe()
        
        dftot = pd.DataFrame()
        dftot_val=pd.DataFrame()
        bias_variance = pd.DataFrame()
        
        model = Models(features, prediction_params, env.data_folder, env.walk)

        for industry in env.industries:
            print_info("Predictions for {0}".format(industry), file="stdout", flush=True)
            df_val, df, metrics_by_company, metrics_groups = model.predict(industry)
            
            if (not df.empty):
                dftot = pd.concat([dftot, df], axis=0)

            if (not df_val.empty):
                dftot_val = pd.concat([dftot_val, df_val], axis=0)

            if (not metrics_by_company.empty):
                bias_variance = pd.concat([bias_variance, metrics_by_company], axis=0)

            if (not metrics_groups.empty):
                overall_chosen_methods = pd.concat([overall_chosen_methods, metrics_groups], axis=0)

        overall_bias_variance = pd.concat([overall_bias_variance, bias_variance], axis=0)        
        bias_variance.to_csv(env.output_folder+'/bias_variance.csv')
        strategy1 = StatArbRegression(dftot, 'Predicted_OC_perc')
        strategy1.compute_metrics(output_folder = env.output_folder)
        strategy1.generate_signals(output_folder = env.output_folder)
        strategy1.plot_returns(outputfolder=env.output_folder, parameters=prediction_params)
        dftot_val.to_csv(env.output_folder+'/totale_validation.csv')

    overall_bias_variance.to_csv("./" +prediction_params['method']+'/StatisticalArbitrage/bias_variance.csv')
    overall_chosen_methods.to_csv("./" +prediction_params['method']+'/StatisticalArbitrage/chosen_methods.csv')    