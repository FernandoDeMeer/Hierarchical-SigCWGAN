import numpy as np
import pandas as pd
import torch

class Pipeline:
    def __init__(self, steps):
        """ Pre- and postprocessing pipeline. This class employs StandardScalerTS and allows for other scalers. """
        self.steps = steps

    def transform(self, x, until=None):
        x = x.clone()
        for n, step in self.steps:
            if n == until:
                break
            x = step.transform(x)
        return x

    def inverse_transform(self, x, until=None):
        for n, step in self.steps[::-1]:
            if n == until:
                break
            x = step.inverse_transform(x)
        return x

class StandardScalerTS():
    """ Standard scales a given (indexed) input vector along the specified axis. """

    def __init__(self, axis=(1)):
        self.mean = None
        self.std = None
        self.axis = axis

    def transform(self, x):
        if self.mean is None:
            self.mean = torch.mean(x, dim=self.axis)
            self.std = torch.std(x, dim=self.axis)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def inverse_transform(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def rolling_window(x, x_lag, add_batch_dim=True):
    if add_batch_dim:
        x = x[None, ...]
    return torch.cat([x[:, t:t + x_lag] for t in range(x.shape[1] - x_lag)], dim=0)

def rolling_window_crossdim(x, base_dims, target_dims,p, q ):
    rolling_windows = rolling_window(x=x,x_lag=p+q, add_batch_dim= False)
    x_input = rolling_windows[:,:p,np.concatenate((base_dims,target_dims))]
    x_input_base = rolling_windows[:,:p,base_dims]
    x_output = rolling_windows[:,-q:,np.concatenate((base_dims,target_dims))]
    return x_input,x_output,x_input_base


def get_rawandpreprocessed_data(data_type):
    data_function_name = 'get_{}_dataset'.format(data_type)
    try:
        possibles = globals().copy()
        possibles.update(locals())
        data_function = possibles.get(data_function_name)
    except:
        pass

    if data_type == 'futures':
        pipeline, x_real_raw, x_real = get_futures_dataset()
    elif data_type == 'ETFS':
        pipeline, x_real_raw, x_real = get_ETFS_dataset()
    else:
        try:
            pipeline, x_real_raw, x_real = data_function()
        except:
            raise NotImplementedError('Dataset %s not valid' % data_type)
    return pipeline, x_real_raw, x_real

def get_data_sigcwgan(data_type, p, q,):
    pipeline, x_real_raw, x_real = get_rawandpreprocessed_data(data_type)
    assert x_real.shape[0] == 1
    x_real = rolling_window(x_real[0], p + q)
    return x_real


def get_data_cross_dim_sigcwgan(dataset, base_dims, target_dims, p, q):
    pipeline, x_real_raw, x_real = get_rawandpreprocessed_data(dataset)
    assert x_real.shape[0] == 1
    x_input,x_output,x_input_base = rolling_window_crossdim(x_real, base_dims, target_dims, p, q)

    return x_input,x_output,x_input_base

def get_data_base_sigcwgan(dataset, p, q, clusters_last_added_array, **data_params):
    pipeline, x_real_raw, x_real = get_rawandpreprocessed_data(dataset)

    assert x_real.shape[0] == 1

    x_real = x_real[0,:,clusters_last_added_array]
    x_real = rolling_window(x_real, p + q)
    x_input = x_real[:,:p]
    x_output = x_real[:,-q:]

    return x_input,x_output

def get_ETFS_dataset():
    """
    Get historical time series of a set of ETFS.
    """

    ETFS = pd.read_csv('../data/ETFUniverse.csv', index_col=0)

    ETF_prices = np.log(np.array(ETFS))

    data_raw = (ETF_prices[1:] - ETF_prices[:-1]).reshape(1, -1, ETF_prices.shape[1])

    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)

    return pipeline, data_raw, data_preprocessed

def get_futures_dataset():
    """
    Get historical time series of a set of futures.
    """
    try:
        FUTURES = pd.read_csv('src/data/futures.csv', index_col=0, header=0)
    except:
        FUTURES = pd.read_csv('../data/futures.csv', index_col=0, header=0)

    FUTURES_prices = FUTURES.loc['03/05/2000':]

    # FUTURES_prices = FUTURES.loc['01/01/2020':'31/12/2020']

    #we use absolute returns because of negative oil prices in April 2020

    data_raw = np.array(np.divide(np.diff(FUTURES_prices,axis=0),FUTURES_prices[:-1]))

    data_raw = data_raw.reshape(1, -1, data_raw.shape[1])

    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)

    return pipeline, data_raw, data_preprocessed


def get_futures_extended_dataset():
    """
    Get historical time series of a set of futures.
    """
    try:
        FUTURES = pd.read_csv('src/data/futures_extended.csv', index_col=0, header=0)
    except:
        FUTURES = pd.read_csv('../data/futures_extended.csv', index_col=0, header=0)

    FUTURES_prices = FUTURES.loc['03/05/2000':]

    # FUTURES_prices = FUTURES.loc['01/01/2020':'31/12/2020']

    #we use absolute returns because of negative oil prices in April 2020

    data_raw = np.array(np.divide(np.diff(FUTURES_prices,axis=0),FUTURES_prices[:-1]))

    data_raw = data_raw.reshape(1, -1, data_raw.shape[1])

    data_raw = torch.from_numpy(data_raw).float()
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    data_preprocessed = pipeline.transform(data_raw)

    return pipeline, data_raw, data_preprocessed

