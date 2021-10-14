import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from src.lib.utils import load_pickle, to_numpy
from src.lib.test_metrics import test_metrics
from src.lib.base import is_multivariate
from src.lib.plot import plot_matrix


def sample_sig_fake_hiergan(sig_config, x_past,x_fake):
    x_fake = x_fake.repeat(sig_config.mc_size, 1, 1).requires_grad_()
    sigs_fake_future = sig_config.compute_sig_future(x_fake)
    sigs_fake_ce = sigs_fake_future.reshape(sig_config.mc_size, x_past.size(0), -1).mean(0)
    return sigs_fake_ce, x_fake


def compute_predictive_score(x_past, x_future, x_fake):
    size = x_fake.shape[0]
    X = to_numpy(x_past.reshape(size, -1))
    Y = x_fake.reshape(size, -1)
    size = x_past.shape[0]
    X_test = X.copy()
    Y_test = to_numpy(x_future[:, :1].reshape(size, -1))
    model = LinearRegression()
    model.fit(X, Y)  # TSTR
    r2_tstr = model.score(X_test, Y_test)
    model = LinearRegression()
    model.fit(X_test, Y_test)  # TRTR
    r2_trtr = model.score(X_test, Y_test)
    return dict(r2_tstr=r2_tstr, r2_trtr=r2_trtr, predictive_score=np.abs(r2_trtr - r2_tstr))

def compute_test_metrics(x_fake, x_real):
    res = dict()
    res['abs_metric'] = test_metrics['abs_metric'](x_real)(x_fake).item()
    res['acf_id_lag=1'] = test_metrics['acf_id'](x_real, max_lag=2)(x_fake).item()
    res['kurtosis'] = test_metrics['kurtosis'](x_real)(x_fake).item()
    res['skew'] = test_metrics['skew'](x_real)(x_fake).item()
    if is_multivariate(x_real):
        res['cross_correl'] = test_metrics['cross_correl'](x_real)(x_fake).item()
    return res

def rolling_window_base_dim(base_dim, p):
    rolling_windows = []
    for scenario in range(base_dim.shape[0]):
        rolling_windows_scenario = [base_dim[scenario][p*t:p*(t+1)].unsqueeze(0)
                                    for t in range(int(base_dim.shape[1]/p))]
        rolling_windows.append(torch.cat(rolling_windows_scenario, dim=0).unsqueeze(0))
    return torch.cat(rolling_windows,dim=0)

def returns_to_prices(rtns):

    gen_prices = np.ones((rtns.shape[0],rtns.shape[1]+1,rtns.shape[2]))

    for i in range(rtns.shape[1]):
        gen_prices[:,i+1] = gen_prices[:,i]*(1 + rtns[:,i])

    return gen_prices



