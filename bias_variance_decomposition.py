import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def _draw_boostrap_from_sample_data(rng, dates):
    size = len(dates)
    sample_indices = np.arange(size)
    bootstrap_indices = rng.choice(sample_indices,
                                   size=int(size/2),
                                   replace=False)
    sorted_boostrap_indices = np.array(sorted(bootstrap_indices))
    return dates[sorted_boostrap_indices],sorted_boostrap_indices

def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices,
                                   size=sample_indices.shape[0],
                                   replace=True)
    return X[bootstrap_indices], y[bootstrap_indices]

def compute_measures(predictions, y):
    all_pred = np.concatenate(predictions).ravel().reshape(predictions.size, -1)
    y = np.concatenate(y).ravel().reshape(y.size, -1)
    main_exp = np.mean(y, axis=0)
    avg_expected_loss=((all_pred-y)**2).mean()
    avg_expected_loss2 = np.apply_along_axis(
            lambda x:
            ((x - main_exp)**2).mean(),
            axis=1,
            arr=all_pred).mean()

    main_predictions = np.mean(all_pred, axis=0)

    avg_bias = np.sum((main_predictions - main_exp)**2) / main_exp.size
    avg_var = np.sum((main_predictions - all_pred)** 2) / all_pred.size


    return avg_expected_loss, avg_expected_loss2, avg_bias, avg_var

class BiasVarianceDecompose(object):
    def __init__(self,estimator,validation_index,test_index,scaler_y=None):
        self.scaler_y = scaler_y
        self.estimator = estimator
        self.validation_index = validation_index
        self.test_index = test_index
        

    def decompose(self, X_validation:pd.DataFrame,y_validation:pd.DataFrame, label, method, num_rounds=50, random_seed=123):
        if ('Date' not in y_validation.columns):
            raise NotImplementedError('Not supported without date')
        if ('Title' not in y_validation.columns):
            raise NotImplementedError('Not supported without Title')
        rng = np.random.RandomState(random_seed)
        all_pred = pd.DataFrame(columns=['Title', 'Predicted','Expected'])
        dates = y_validation.loc[self.validation_index]['Date'].unique()
        dates_test = y_validation.loc[self.test_index]['Date'].unique()
        dates_test=dates_test[:int(len(dates_test)/2)]
    
        result = pd.DataFrame(columns=['Title','avg_loss1','avg_loss2','loss_overall','loss_test','avg_bias','avg_var'])

        for i in range(num_rounds):
            boostrap_dates, _ = _draw_boostrap_from_sample_data(rng, dates)
            y = y_validation[y_validation['Date'].isin(boostrap_dates)].copy()
            X = X_validation.loc[y.index].copy()
            

            predictions = self.estimator.predict(X.values)
            if (self.scaler_y):
                predictions = self.scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).ravel()

            y['Predicted'] = predictions / y['Open']
            for title,group in y.groupby(['Title']):
                all_pred.loc[len(all_pred)] = {'Title': title, 'Predicted': group['Predicted'].values,'Expected':group[label].values}
        companies = y_validation.loc[self.validation_index]['Title'].unique()
        y_val = y_validation.loc[self.validation_index]
        y_test = y_validation[y_validation['Date'].isin(dates_test)]
        
        for title in companies:
            
            loss1, loss2, bias, var = compute_measures(all_pred[all_pred['Title'] == title]['Predicted'].values, all_pred[all_pred['Title'] == title]['Expected'].values)
            mse_overall_val = compute_overall_loss(y_val.loc[y_val['Title']==title], label)
            mse_overall_test = compute_overall_loss(y_test.loc[y_test['Title']==title], label)
            
            result.loc[len(result)]={'Title':title,'avg_loss1':loss1,'avg_loss2':loss2,'loss_overall':mse_overall_val,'loss_test':mse_overall_test,'avg_bias':bias,'avg_var':var}

        return result

def compute_overall_loss(snippet_validation,label):
    predicted = snippet_validation['Predicted'].values / snippet_validation['Open'].values
    expected = snippet_validation[label].values / snippet_validation['Open'].values
    mse = mean_squared_error(expected, predicted)
    return mse


def bias_variance_decomp(estimator, X_train, y_train, X_test, y_test,
                         loss='0-1_loss', num_rounds=50, random_seed=None):
    """
    estimator : object
        A classifier or regressor object or class implementing a `fit`
        `predict` method similar to the scikit-learn API.
    X_train : array-like, shape=(num_examples, num_features)
        A training dataset for drawing the bootstrap samples to carry
        out the bias-variance decomposition.
    y_train : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.
    X_test : array-like, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.
    y_test : array-like, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.
    loss : str (default='0-1_loss')
        Loss function for performing the bias-variance decomposition.
        Currently allowed values are '0-1_loss' and 'mse'.
    num_rounds : int (default=200)
        Number of bootstrap rounds for performing the bias-variance
        decomposition.
    random_seed : int (default=None)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.
    Returns
    ----------
    avg_expected_loss, avg_bias, avg_var : returns the average expected
        average bias, and average bias (all floats), where the average
        is computed over the data points in the test set.
    """
    supported = ['0-1_loss', 'mse']
    if loss not in supported:
        raise NotImplementedError('loss must be one of the following: %s' %
                                  supported)

    rng = np.random.RandomState(random_seed)

    # all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=np.int)
    all_pred=[]

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)
        pred = estimator.fit(X_boot, y_boot).predict(X_test)
        all_pred=np.concatenate((all_pred,pred),axis=0)

    if loss == '0-1_loss':
        main_predictions = np.apply_along_axis(lambda x:
                                               np.argmax(np.bincount(x)),
                                               axis=0,
                                               arr=all_pred)

        avg_expected_loss = (main_predictions != y_test).sum()/y_test.size

        avg_expected_loss = np.apply_along_axis(lambda x:
                                                (x != y_test).mean(),
                                                axis=1,
                                                arr=all_pred).mean()

        avg_bias = np.sum(main_predictions != y_test) / y_test.size

        var = np.zeros(pred.shape)

        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int)
        var /= num_rounds

        avg_var = var.sum()/y_test.shape[0]

    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x:
            ((x - y_test)**2).mean(),
            axis=1,
            arr=all_pred).mean()

        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = np.sum((main_predictions - y_test)**2) / y_test.size
        avg_var = np.sum((main_predictions - all_pred)**2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var

def copy_info(metrics_df: pd.DataFrame, walk, method_label, industry,method,start_date):
    new=pd.DataFrame(columns=['walk_no', 'method_id', 'type', "start date", "Title", "Industry", 'mse1','mse2','mse_val','mse_test','avg_bias','avg_var'])

    new['mse2'] = metrics_df['avg_loss2']
    new['mse1']=metrics_df['avg_loss1']
    new['mse_val'] = metrics_df['loss_overall']
    new['mse_test']=metrics_df['loss_test']
    new['Title'] = metrics_df['Title']
    # new['mse_test'] =mse_test
    new['avg_bias'] = metrics_df['avg_bias']
    new['avg_var'] = metrics_df['avg_var']
    new['Industry'] = industry
    new['walk_no'] = walk
    new['method_id'] = method_label
    new['type'] = method
    new['start date'] = start_date
    return new