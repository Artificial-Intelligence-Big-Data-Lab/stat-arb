import pandas as pd
import sys, argparse
def get_prediction_type(argument):
    switcher = {
        1: "GB",
        2: "ARIMA",
        3: "SVM",
        4: "RF",
    }
    if argument in switcher.values():
        return argument
    else:
        return "ARIMA"


def parse_arguments():
    start_date = pd.Timestamp(sys.argv[1])
    days_training = int(sys.argv[2])
    nr_forecasting = int(sys.argv[3])
    days_validation=int(sys.argv[4])
    companies_number = int(sys.argv[5])
    prediction_method = get_prediction_type(sys.argv[6])
    refit = False
    scale = False
    
    if len(sys.argv) == 10 and sys.argv[9]=='scale':
        scale = True
            
    numer_of_walks = int(sys.argv[7])

    if prediction_method!="ARIMA" :
        if len(sys.argv)<=7:
            days_lag=10
        else:
            days_lag=int(sys.argv[8])
    else:
        days_lag=1


    prediction_params = {
        'start_date':start_date,
        'train': days_training,
        'val': days_validation,
        'test':nr_forecasting,
        'companies':companies_number,
        'lag':days_lag,
        'walks':numer_of_walks,
        'method':'../' + prediction_method,
        'refit': refit,
        'scale':scale
        }
    return prediction_method, prediction_params
    
def get_parameters(arguments=None):
    p = argparse.ArgumentParser()
    p.add_argument("--keyvalues", action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", dest='keyvalues')
    if arguments is None:
        arguments=sys.argv[1:]
    args = p.parse_args(arguments)
    d = dict(args.keyvalues)

    prediction_params = {
        'start_date':pd.Timestamp(d['start_date']),
        'train': int(d['train']),
        'val': int(d['val']),
        'test': int(d['test']),
        'companies': int(d['companies']),
        'lag': int(d['lag']),
        'walks': int(d['walks']),
        'method': './' + d['method'],
        'nn':int(d['nn']),
        'refit': False,

        }
    if ('scale' in d.keys()):
        prediction_params['scale'] = d['scale'] in ['True', 'true']
    else:
        prediction_params['scale'] = False
        
    if ('categorical' in d.keys()):
        prediction_params['categorical'] = d['categorical'] in ['True', 'true']
    else:
        prediction_params['categorical'] = False
        

    return prediction_params['method'], prediction_params


class StoreDictKeyPair(argparse.Action):
     def __init__(self, option_strings, dest, nargs=None, **kwargs):
         self._nargs = nargs
         super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
     def __call__(self, parser, namespace, values, option_string=None):
         keyvalues = {}
         print ("values: {}".format(values))
         for kv in values:
             k,v = kv.split("=")
             keyvalues[k] = v
         setattr(namespace, self.dest, keyvalues)
