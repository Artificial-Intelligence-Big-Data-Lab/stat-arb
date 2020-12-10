import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, KFold, cross_val_score, train_test_split
import forecasting_metrics as metrics
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from CustomCrossValidator import CustomTimeSeriesSplit, CustomFixTimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
from utils import (setup_folders, _cleanup_test_folder, safe_print, print_info)
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from bias_variance_decomposition import bias_variance_decomp,BiasVarianceDecompose,copy_info

from collections import namedtuple

Set = namedtuple('Set', ['X', 'y'])

class Models():
    def __init__(self, features:dict, parameters, folder,walk):
        self.feature_types = features
        self.parameters = parameters
        self.folder = folder
        self.index_column = 'Date'
        self.categorical_column = 'Asset_ID'
        self.label = 'OC'
        self.return_label = 'OC Percent'
        self.trainig = parameters["train"]
        self.test = parameters["test"]
        self.scale = parameters["scale"]
        self.validation = parameters["val"]
        self.method = parameters['method'].replace('.', '').replace('/', '')
        self.walk = walk
        self.scaler_y = None
        self.test_folder = '{0}/test/'.format(parameters['method'])
        self.categorical = parameters['categorical']

    def predict(self,industry):
        self.industry = industry
        result = pd.DataFrame()
        metrics_by_company = pd.DataFrame(columns=['walk_no', 'method_id', 'type', "start date", "Title", "Industry", 'mse1','mse2','mse_val','mse_test','avg_bias','avg_var'])
        prediction_dates=[]
        for data_type in self.feature_types.keys():
            df = self.__read_data(data_type)
            df = df.groupby(['Title']).filter(lambda x: len(x) ==(self.trainig+self.validation+self.test))

            if (result.empty):
                result = df.copy()
 
            print_info("Predictions for industry {0} features {1} shape {2}".format(industry, data_type, df.shape), file="stdout", flush=True)

            features = self.feature_types[data_type]
            print_info("Features {0}".format(features), file="stdout", flush=True)
            training_index, validation_index, test_index = self._split_train_validation_test(df)
            training, validation, test = df.loc[training_index], df.loc[validation_index], df.loc[test_index]

            validation_dates = df.loc[validation_index]['Date'].unique()
            test_dates = df.loc[test_index]['Date'].unique()
            dates = np.concatenate((validation_dates,test_dates))
            assert ((np.alltrue(dates == prediction_dates)) or (len(prediction_dates)==0)), "Invalid dates"
            prediction_dates = dates


            print_info("Predictions for industry {0} shapes: {1} {2} {3}".format(industry, training.shape,validation.shape,test.shape), file="stdout", flush=True)
            train_series, validation_series, test_series = self.feature_engineering(training, validation, test, features, 'industry')
            print_info("Predictions for industry {0} shapes: train {1} target {2} validation {3} test {4}".format(industry, train_series.X.shape,train_series.y.shape, validation_series.X.shape,test_series.shape), file="stdout", flush=True)
            
            _, _, model = self.get_params(train_series.X,train_series.y,  categorical_feature='auto')
            

            predictions_validation = model.predict(validation_series.X.values)
            predictions_test = model.predict(test_series.values)


            if (self.scale):
                predictions_test = self.scaler_y.inverse_transform(np.array(predictions_test).reshape(-1, 1)).ravel()
                predictions_validation = self.scaler_y.inverse_transform(np.array(predictions_validation).reshape(-1, 1)).ravel()
                
            label= self.get_label("industry",data_type,self.method)
            df.iloc[:]['Predicted'] = np.nan
        
            df.loc[test_index, 'Predicted'] = predictions_test
            df.loc[validation_index, 'Predicted'] = predictions_validation



            result['Predicted_OC_perc_'+label] = np.nan
            result['Expected_OC_perc_'+label] = np.nan
            self.__set_predictions_to_result(result, df, test_dates, validation_dates, metrics_by_company, label, industry)
            metrics_df = BiasVarianceDecompose(model,validation_index,test_index,self.scaler_y).decompose(validation_series.X,df, self.label, label)

            m = copy_info(metrics_df,self.walk,label,industry,self.method,self.parameters['start_date'])
            metrics_by_company = metrics_by_company.append(m, ignore_index=True)

            label= self.get_label("company",data_type,self.method)

            result['Predicted_OC_perc_'+label] = np.nan
            result['Expected_OC_perc_'+label] = np.nan
            
            for title, group in df.groupby('Title'):
                company_df = group.copy()
                print_info("Predictions for industry {0} company {1} features {2}".format(industry,title, data_type), file="stdout", flush=True)
                training_index, validation_index, test_index = self._split_train_validation_test(company_df)
                if ((len(training_index) == 0) or (len(validation_index) == 0) or (len(test_index) == 0)):
                    continue
                training, validation, test = company_df.loc[training_index], company_df.loc[validation_index], company_df.loc[test_index]
                train_series, validation_series, test_series=self.feature_engineering(training,validation,test,features,'company')
                score, opt_parameters, model = self.get_params(train_series.X, train_series.y[self.label], categorical_feature='auto')

                predictions_validation = model.predict(validation_series.X)
                predictions_test = model.predict(test_series)

                if (self.scale):
                    predictions_test = self.scaler_y.inverse_transform(np.array(predictions_test).reshape(-1, 1)).ravel()
                    predictions_validation = self.scaler_y.inverse_transform(np.array(predictions_validation).reshape(-1, 1)).ravel()
                    
                company_df.iloc[:]['Predicted'] = np.nan
            
                company_df.loc[test_index, 'Predicted'] = predictions_test
                company_df.loc[validation_index, 'Predicted'] = predictions_validation
                self.__set_predictions_to_result(result, company_df, company_df.loc[test_index]['Date'].unique(), company_df.loc[validation_index]['Date'].unique(), metrics_by_company, label, industry)
                metrics_df = BiasVarianceDecompose(model,validation_index,test_index,self.scaler_y).decompose(validation_series.X, company_df, self.label, label)

                m = copy_info(metrics_df, self.walk, label, industry, self.method, self.parameters['start_date'])
                metrics_by_company = metrics_by_company.append(m, ignore_index=True)

            #TODO: write predict per company - verify if train,valid,test for each company
            #TODO: write parameters for each model
       
        available_metrics = [metric for metric in metrics_by_company.columns if 'mse' in metric]
        chosen_metrics=pd.DataFrame()
        for metric in available_metrics:
            metrics_sorted = metrics_by_company.sort_values([metric], ascending=[True])
            metrics_groups = metrics_sorted.groupby(['Title', 'walk_no', 'type'], as_index=False).nth(0)
            resul_label = self.method + '_' + metric
            
            result['Predicted_OC_perc_' + resul_label] = np.nan
            metrics_groups['metrics'] = metric
            chosen_metrics= pd.concat([metrics_groups,chosen_metrics],axis=0,ignore_index=True)

            for groups in metrics_groups.itertuples():
                title = groups.Title
                method_id = groups.method_id
                result.loc[result['Title'] == title,'Predicted_OC_perc_' + resul_label] = result[result['Title'] == title]['Predicted_OC_perc_' + method_id].values
            

        for title, group in result.groupby('Title'):
            copy = group.copy()
            size = len(copy)
            copy = copy[(size-(self.test+self.validation)):size]
            copy.to_csv(self.test_folder + title, index=False)
            
        return result[result['Date'].isin(validation_dates)], result[result['Date'].isin(test_dates)], metrics_by_company, chosen_metrics

    def __set_predictions_to_result(self,result:pd.DataFrame,df:pd.DataFrame,test_dates,validation_dates,metrics_by_company,label,industry):

            
        for title, group in df.groupby('Title'): 
            snippet_test = group[group['Date'].isin(test_dates)].copy()
            snippet_validation = group[group['Date'].isin(validation_dates)].copy()

            if (snippet_test.empty or snippet_validation.empty):
                continue
            if (len(result.loc[(result['Date'].isin(test_dates)) & (result['Title'] == title)]) == 0):
                continue
            if (len(result.loc[(result['Date'].isin(validation_dates)) & (result['Title'] == title)]) == 0):
                continue

            result.loc[(result['Date'].isin(test_dates)) & (result['Title']==title), 'Predicted_OC_perc_'+label] =  snippet_test['Predicted'].values
            result.loc[(result['Date'].isin(test_dates)) & (result['Title']==title), 'Expected_OC_perc_'+label] =  snippet_test[self.label].values


            # print_info("Shapes {0} {1}".format(result.loc[(result['Date'].isin(validation_dates)) & (result['Title']==title)].shape ,validation_dates.shape), file="stdout", flush=True)
            result.loc[(result['Date'].isin(validation_dates)) & (result['Title']==title), 'Predicted_OC_perc'+label] = snippet_validation['Predicted'].values
            result.loc[(result['Date'].isin(validation_dates)) & (result['Title']==title), 'Expected_OC_perc'+label] = snippet_validation[self.label].values

                

    def get_label(self,level, data_type, method):
        label = ''
        if (level=='industry'):
            label = label + "I"
        else:
            label = label + "C"
        label = label + method
        label = label + data_type
        
        return label
    
    def feature_engineering(self, training, validation, test, features,level):
        if (self.scale):
            scaler_x, self.scaler_y, series_scaled_x, series_scaled_y = self.scale_data(training[features], training[self.label])
            train_series = pd.DataFrame(series_scaled_x, columns=features)
            target_series = pd.DataFrame(series_scaled_y, columns=[self.label])
        else:
            train_series = pd.DataFrame(training[features].values, columns=features)
            target_series = pd.DataFrame(training[self.label].values, columns=[self.label])

        train_series = train_series.assign(Date=training.Date.values)

        if (self.scale):
            validation_series = pd.DataFrame(scaler_x.transform(validation[features]), columns=features,index=validation.index)
            dummy = np.column_stack((self.scaler_y.transform(validation[self.label].values.reshape(-1,1)).ravel(),validation['Open'].values))
            validation_target_series = pd.DataFrame(dummy, columns=[self.label,'Open'],index=validation.index)
            
            test_series = pd.DataFrame(scaler_x.transform(test[features]), columns=features)
        else:
            validation_series = pd.DataFrame(validation[features], columns=features)
            validation_target_series = pd.DataFrame(validation[self.label], columns=[self.label])
            
            test_series = pd.DataFrame(test[features].values, columns=features)

        if (level=='industry' and self.categorical):
            train_series = train_series.assign(Asset_ID=training[self.categorical_column].values)
            train_series[self.categorical_column] = train_series[self.categorical_column].astype('category')
            
            validation_series = validation_series.assign(Asset_ID=validation[self.categorical_column])
            validation_series[self.categorical_column] = validation_series[self.categorical_column].astype('category')
            
            test_series = test_series.assign(Asset_ID=test[self.categorical_column].values)
            test_series[self.categorical_column] = test_series[self.categorical_column].astype('category')

        return Set(X=train_series, y= target_series), Set(X = validation_series, y=validation_target_series) ,test_series

    def scale_data(self,train_x, train_y):
        scaler_x = MinMaxScaler(feature_range=(-1, 1))
        scaler_x.fit(train_x)
        scaled_x = scaler_x.transform(train_x)
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        scaler_y.fit(np.array(train_y).reshape(-1,1))
        scaled_y = scaler_y.transform(np.array(train_y).reshape(-1,1))
        return scaler_x,scaler_y,scaled_x,scaled_y

    def _split_train_validation_test(self,series:pd.DataFrame):
        copy = series.copy()
        copy['Date_int'] = series.Date.dt.strftime("%Y%m%d").astype(int)
    
        dates = np.sort(copy.Date_int.unique())
        train_dates_index = dates[0:self.trainig]
        validation_dates_index = dates[self.trainig:self.trainig+self.validation]
        test_dates_index = dates[self.trainig+self.validation:]
        return copy.loc[copy['Date_int'].isin(train_dates_index)].index, copy.loc[copy['Date_int'].isin(validation_dates_index)].index, copy.loc[copy['Date_int'].isin(test_dates_index)].index

    def __read_data(self, data_type):
        self.__location = self.folder +  data_type+"_" + self.industry.replace(" ","_")+".csv"
        
        if not os.path.exists(self.__location):
            return pd.DataFrame()
        else:
            
            returns = pd.read_csv(self.__location, parse_dates=True)
            returns['Date'] = pd.to_datetime(returns['Date'], dayfirst=True)
            returns = returns.sort_values(['Date','Title'],ascending=[True,True])
        return returns

    def get_params(self,train_x,train_y, early_stopping_rounds=100, max_train_size=None, categorical_feature='auto'): 
    
        model,grid_params=self.get_model()


        custom_validation_technique = CustomTimeSeriesSplit(n_splits=10).split(train_x)
        train_x = train_x.drop(['Date'], axis=1)
        print_info("Predictions for industry  shapes: train {0} target {1} v".format(train_x.shape, train_y.shape), file="stdout", flush=True)
        # Create the grid
        grid = GridSearchCV(
        model,
        grid_params,
        n_jobs=-1,
        cv = custom_validation_technique,
        verbose=True,
        refit=True,
        scoring=make_scorer(mean_squared_error),error_score=np.nan)

        grid.fit(train_x.values, train_y.values.ravel())
        
        return_params = grid.best_params_
        return grid.best_score_, return_params, grid.best_estimator_
        
    def get_model(self):
            
        rf_model = RandomForestRegressor(oob_score=True, random_state=42)
    
        # Create parameters to search
        rf_grid_params = {
        'n_estimators':[100,200,300,400],
        'max_depth': [5],
        'min_samples_leaf': [3],
	'min_samples_split': [3],
        }

        gb_model = lgb.LGBMRegressor(boosting_type= 'gbdt')
        
        # Create parameters to search
        gb_grid_params = {
        'n_jobs':[-1],
        'learning_rate': [0.01, 0.03],
        'num_leaves': [70, 80, 100],
        'n_estimators':[100],
        'max_depth': [3,8],
        'colsample_bytree':[.8],
        'subsample':[.8],
        'reg_alpha':[.1],
        'reg_lambda':[.01],
        'min_split_gain':[.01],
        'min_child_weight':[2],
        }

        svm_model = SVR(shrinking=False)


  
        svm_grid_params = {
        'C': [8,10,15],
        'kernel':['rbf'],
        'max_iter': [1e5], 
        'tol': [1e-2],
        'gamma': [0.1,0.15]
        }

        if ( 'GB' in self.method):
            return gb_model, gb_grid_params
        elif (self.method == 'SVR'):
            return svm_model, svm_grid_params
        elif (self.method=='RF'):
                return rf_model,rf_grid_params

