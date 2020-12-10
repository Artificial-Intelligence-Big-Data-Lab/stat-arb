import pandas as pd
import numpy as np
import glob
import os
import sys
import os
import sys
from pathlib import Path

import shutil
import ast

import time

import numpy as np



def setup_folders(prediction_method):

    if not Path('./'+prediction_method+'/').exists():
        os.mkdir('./'+prediction_method+'/')

    if not Path('./' + prediction_method + '/single_data/').exists():
        os.mkdir('./' + prediction_method + '/single_data/')

    test_folder = './'+prediction_method+'/test/'
    if not Path(test_folder).exists():
        os.mkdir(test_folder)
    
    statistical_arbitrage_folder= './'+prediction_method+'/StatisticalArbitrage/'

    if not Path(statistical_arbitrage_folder).exists():
        os.mkdir(statistical_arbitrage_folder)
    else:

        for the_file in os.listdir(statistical_arbitrage_folder):
            file_path = os.path.join(statistical_arbitrage_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)


def _cleanup_test_folder(test_folder,files):
    for fnombre in files:
        if os.path.exists(test_folder + fnombre):
            os.remove(test_folder + fnombre)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__qualname__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            f = "stdout"
            safe_print("""[INFO] start_method  %r - %2.2f [ms]""" % (method.__qualname__, (te - ts) * 1000), file=f, flush=True)
           
        return result
    return timed

def safe_print(*objects, **kwargs):
    """Safe print function for backwards compatibility."""
    # Get stream
    file = kwargs.pop('file', sys.stdout)
    if isinstance(file, str):
        file = getattr(sys, file)

    # Get flush
    flush = kwargs.pop('flush', False)

    # Print
    print(*objects, file=file, **kwargs)

    # Need to flush outside print function for python2 compatibility
    if flush:
        file.flush()

def print_info(message='', **kwargs):
    safe_print('[INFO] {0} {1}'.format (time.strftime("%d-%b-%Y %H:%M:%S"), message), **kwargs)


def print_time(t0, message='', **kwargs):
    """Utility function for printing time"""
    if len(message) > 0:
        message += ' | '

    m, s = divmod(time() - t0, 60)
    h, m = divmod(m, 60)

    safe_print(message + '%02d:%02d:%02d' % (h, m, s), **kwargs)

def get_folders(folder_path):
    folders = []
    stri = "./{0}/StatisticalArbitrage/*-*ARIMASINGLE2*_output".format(folder_path)

    for folder in glob.glob(stri):
        print (folder)
        folders.append(folder)
    return folders

def computecumsum(series):
    xspredsimple = np.array(series.values.flatten().tolist())
    mean_return_by_day=np.array(series.values.flatten().tolist()).mean()
    xspred = xspredsimple.cumsum()
    ipred = np.argmax(np.maximum.accumulate(xspred) - xspred)
    if ipred==0:
        jpred = 0
    else:
        jpred = np.argmax(xspred[:ipred])

    mddpred = xspred[jpred]-xspred[ipred]
    
    computed_return=sum(series.values.flatten().tolist())
    romad=sum(series.values.flatten().tolist())/mddpred

    return mddpred,computed_return,romad,xspred,ipred,jpred,mean_return_by_day

def compute_yearly_drawdown(series,window=252):

    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first 252 days data have an expanding window
    Roll_Max = series.rolling(window, min_periods=1).max()
    Daily_Drawdown =series/Roll_Max

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    return Daily_Drawdown, Max_Daily_Drawdown


def compute_buy_and_hold_companies(df_input:pd.DataFrame,active=5,sort_by_labels={'Expected':'Expected_OC_perc','Predicted':'Predicted_OC_perc','Metric':{'Name':None,'Order':None}}, industry_information = None, prediction_method=None, mask=None):
    
    active_range = set(range(0, active, 1))
    dftot=df_input.copy()

    if (industry_information is not None and len(industry_information) != 0):
        dftot = dftot[dftot['Title'].isin(industry_information)]

    df_unuseful = dftot.groupby('Date').filter(lambda x: len(x) < active)

    print(df_unuseful.index.unique())
    dftot = dftot.groupby('Date').filter(lambda x: len(x) >= active)

    
   
    dftot_1 = dftot.sort_values([sort_by_labels['Expected']], ascending=False)
    dftot_1['Rank']=dftot_1.groupby('Date')[sort_by_labels['Expected']].rank(ascending=False, method='dense')
    #print "QUIDFTOT_1:",dftot_1
    peggiori_exp = dftot_1.groupby('Date').nth(active_range)

    if (prediction_method is not None):
        peggiori_exp.to_csv('{0}_peggiori_expected.csv'.format(prediction_method))

   
    dftot_2 = dftot.sort_values([sort_by_labels['Expected']], ascending=True)
    dftot_2['Rank']= -dftot_2.groupby('Date')[sort_by_labels['Expected']].rank(ascending=True, method='dense')
    migliori_exp = dftot_2.groupby('Date').nth(active_range)
    

    if (prediction_method is not None):
        migliori_exp.to_csv('{0}_migliori_expected.csv'.format(prediction_method))

    incr_bydate_exp = migliori_exp.groupby('Date').sum()
    #print "incr_bydate_exp",incr_bydate_exp
    decr_bydate_exp = peggiori_exp.groupby('Date').sum()

    # peggiori da rendere negatvi (?)
    #print "decr_bydate_exp",decr_bydate_exp

    valore_giornaliero_exp = ( - incr_bydate_exp['Expected_OC_perc'] + decr_bydate_exp['Expected_OC_perc'])/(2*active)*100
    valore_giornaliero_exp.index = pd.to_datetime(valore_giornaliero_exp.index)

    if (prediction_method is not None):
        valore_giornaliero_exp.to_csv('{0}_valore_giornaliero_expected.csv'.format(prediction_method))

    if (mask is not None):
        dftot = dftot[dftot.apply(mask,axis=1)]

    if ('Metric'in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric']['Name'] is not None:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']], ascending=[False, sort_by_labels['Metric']['Order']])
    else:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[False])


    dftot_3['Rank']=dftot_3.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=False, method='dense')
    peggiori_pred = dftot_3.groupby('Date').nth(active_range)
    
    
    if (prediction_method is not None):
        peggiori_pred.to_csv('{0}_peggiori_predicted.csv'.format(prediction_method))

    # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    if ('Metric'in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric']['Name'] is not None:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']], ascending=[True, sort_by_labels['Metric']['Order']])
    else:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[True])
    
    dftot_4['Rank'] = - dftot_4.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=True, method='dense')
    migliori_pred = dftot_4.groupby('Date').nth(active_range)
    
    
    if (prediction_method is not None):
        migliori_pred.to_csv('{0}_migliori_predicted.csv'.format(prediction_method))

    # calcolo del valore percentuale sui valori predetti
    incr_bydate_pred = migliori_pred.groupby('Date').sum()
    decr_bydate_pred = peggiori_pred.groupby('Date').sum()
  

    valore_giornaliero_pred = (- incr_bydate_pred['Expected_OC_perc'] + decr_bydate_pred['Expected_OC_perc'])/(2*active)*100
    valore_giornaliero_pred.index = pd.to_datetime(valore_giornaliero_pred.index)

    return valore_giornaliero_pred,valore_giornaliero_exp,migliori_exp,migliori_pred,peggiori_exp,peggiori_pred


def compute_buy_and_hold_companies2(dftot:pd.DataFrame,active=5,sort_by_labels={'Expected':'Expected_OC_perc','Predicted':'Predicted_OC_perc'}):
    active_range = set(range(0, active, 1))
    
    dftot = dftot.groupby('Date').filter(lambda x: len(x) >= active)
    dftot_1 = dftot.sort_values([sort_by_labels['Expected']], ascending=False); 
    #print "QUIDFTOT_1:",dftot_1
    migliori_exp = dftot_1.groupby('Date').nth(active_range); 
    #print "MIGL:",migliori_exp

    # Calcolo dei 5 titoli peggiori sui risultati attesi (expected)
    dftot_2 = dftot.sort_values([sort_by_labels['Expected']], ascending=True); 
    peggiori_exp = dftot_2.groupby('Date').nth(active_range); 


    dftot_3 = dftot.sort_values([sort_by_labels['Predicted']], ascending=False); 
    migliori_pred = dftot_3.groupby('Date').nth(active_range); 

    # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    dftot_4 = dftot.sort_values([sort_by_labels['Predicted']], ascending=True); 
    peggiori_pred = dftot_4.groupby('Date').nth(active_range); 

    incr_bydate_exp = migliori_exp.groupby('Date').sum()
    #print "incr_bydate_exp",incr_bydate_exp
    decr_bydate_exp = peggiori_exp.groupby('Date').sum()

    # peggiori da rendere negatvi (?)
    #print "decr_bydate_exp",decr_bydate_exp

    valore_giornaliero_exp = (incr_bydate_exp['Expected_OC_perc'] - decr_bydate_exp['Expected_OC_perc'])/(2*active)*100
    valore_giornaliero_exp.index = pd.to_datetime(valore_giornaliero_exp.index)

    # calcolo del valore percentuale sui valori predetti
    incr_bydate_pred = migliori_pred.groupby('Date').sum()
    decr_bydate_pred = peggiori_pred.groupby('Date').sum()
  
    # peggiori da rendere negatvi (?)
    #decr_bydate_pred = (decr_bydate_pred)
    #print "incr_bydate_pred",incr_bydate_pred
    #print "decr_bydate_pred",decr_bydate_pred


    valore_giornaliero_pred = (incr_bydate_pred['Expected_OC_perc'] - decr_bydate_pred['Expected_OC_perc'])/(2*active)*100
    valore_giornaliero_pred.index = pd.to_datetime(valore_giornaliero_pred.index)
    
    return valore_giornaliero_pred,valore_giornaliero_exp,migliori_exp,migliori_pred,peggiori_exp,peggiori_pred

def compute_ranks_peggiori(df_input: pd.DataFrame, sort_by_labels={'Expected': 'Expected_OC_perc', 'Predicted': 'Predicted_OC_perc', 'Metric': {'Name': None, 'Order': None}}, industry_information=None, prediction_method=None, mask=None):
    dftot=df_input.copy()

    if (industry_information is not None and len(industry_information) != 0):
        dftot = dftot[dftot['Title'].isin(industry_information)]
   
    dftot_1 = dftot.sort_values([sort_by_labels['Expected']], ascending=False)
    dftot_1['Rank'] = dftot_1.groupby('Date')[sort_by_labels['Expected']].rank(ascending=False, method='dense')
    
    if (mask is not None):
        dftot = dftot[dftot.apply(mask, axis=1)]
        
    if ('Metric'in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric']['Name'] is not None:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']], ascending=[False, sort_by_labels['Metric']['Order']])
    else:
        dftot_3 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[False])


    dftot_3['Rank']=dftot_3.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=False, method='dense')
    if ('Date' in dftot_1.columns):
        dftot_1.set_index('Date', inplace=True)

    if ('Date' in dftot_1.columns):
        dftot_3.set_index('Date', inplace=True)
    return dftot_1, dftot_3
    
def compute_ranks_migliori(df_input:pd.DataFrame,active=5,sort_by_labels={'Expected':'Expected_OC_perc','Predicted':'Predicted_OC_perc','Metric':{'Name':None,'Order':None}}, industry_information = None, prediction_method=None, mask=None):
    
   
    dftot=df_input.copy()

    if (industry_information is not None and len(industry_information) != 0):
        dftot = dftot[dftot['Title'].isin(industry_information)]

    df_unuseful = dftot.groupby('Date').filter(lambda x: len(x) < active)

    print(df_unuseful.index.unique())
    dftot = dftot.groupby('Date').filter(lambda x: len(x) >= active)

   
    dftot_2 = dftot.sort_values([sort_by_labels['Expected']], ascending=True)
    dftot_2['Rank']= -dftot_2.groupby('Date')[sort_by_labels['Expected']].rank(ascending=True, method='dense')


    if (mask is not None):
        dftot = dftot[dftot.apply(mask,axis=1)]

        # Calcolo dei 5 titoli peggiori sui risultati predetti (predicted)
    if ('Metric'in sort_by_labels) and ('Name' in sort_by_labels['Metric']) and sort_by_labels['Metric']['Name'] is not None:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted'], sort_by_labels['Metric']['Name']], ascending=[True, sort_by_labels['Metric']['Order']])
    else:
        dftot_4 = dftot.sort_values([sort_by_labels['Predicted']], ascending=[True])
    
    dftot_4['Rank'] = - dftot_4.groupby('Date')[sort_by_labels['Predicted']].rank(ascending=True, method='dense')

    return dftot_2,dftot_4
   