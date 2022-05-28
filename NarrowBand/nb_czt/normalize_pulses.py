# Standard library imports
from copy import deepcopy
from attr import validate

# Third-party library imports
from natsort import natsorted
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy import signal
import seaborn as sns
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook
#from tqdm.dask import TqdmCallback

# Local application imports
import NarrowBand.analysis_pd.df_processing as dfproc

def correct_positive_pulses(df, column = 'signal', drop_columns = ['sample', 'time']):

    # attempt to make more efficient function (using groupby instead of for loops), but logic is trickier
    df2 = deepcopy(df)
    df2.drop(drop_columns, inplace = True)

    if "phantom" in df.columns:
        # columns to group by df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df.groupby(["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
            df2.groupby(["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"]).agg(
                        max=pd.NamedAgg(column=column, aggfunc='max'), min=pd.NamedAgg(column=column, aggfunc='min'))
        else:
            df.groupby(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
            df2.groupby(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"]).agg(
                        max=pd.NamedAgg(column=column, aggfunc='max'), min=pd.NamedAgg(column=column, aggfunc='min'))
    else:
        df.groupby(["cal_type", "date", "rep", "iter", "attLO", "attRF"])
        df2.groupby(["cal_type", "date", "rep", "iter", "attLO", "attRF"]).agg(
                        max=pd.NamedAgg(column=column, aggfunc='max'), min=pd.NamedAgg(column=column, aggfunc='min'))

        df2['multiplier'] = -1.0 if np.abs(df2['min']) > np.abs(df2['max']) else 1.0

        dfout = df.merge(df2['multiplier'], how='left', validate = 'many_to_one')

        dfout['signal'] = dfout['signal'].multiply(dfout['multiplier'], axis='columns', level=None, fill_value=None)
        dfout.drop('multipier', inplace=True)
        dfout.reset_index(inplace = True)

    return dfout


def correct_positive_pulses3(df, ref, column = 'signal', percent_tol = 0.05):

    if "phantom" in df.columns:
        # splits df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

    processed = []

    for data in tqdm(df_list):
        data.reset_index(inplace=True)
        xcorr = signal.correlate((data[column] - data[column].mean()) / np.std(data[column]), (ref - np.mean(ref)) / np.std(ref), mode='full', method='auto') / min(len(data[column]),len(ref)) # normalized cross-correlation
        if np.abs((1 - percent_tol) * np.max(xcorr)) > np.abs(np.min(xcorr)):
            data['sign'] = 1
        else:
            data['sign'] = -1
        # data[column] =  data[column] - data[column].mean()
        # data['max'] = data[column].max()
        # data['min'] = data[column].min()
        # data['sd'] = data[column].std(axis = 0, ddof=1)

        processed.append(data)

    dfout = pd.concat(processed, axis=0, ignore_index = True)
    # dfout[column] = np.where(dfout['min'].abs().to_numpy() > dfout['max'].abs().to_numpy() + tol*dfout['max'].abs().to_numpy(), -dfout[column], dfout[column])
    # dfout[column] = np.where((1 - percent_tol) * dfout['max'].abs().to_numpy() > dfout['min'].abs().to_numpy() + abs_tol, dfout[column], -dfout[column])
    # dfout.drop(labels= ['index', 'max', 'min'], inplace = True, errors = 'ignore', axis=1)
    dfout[column] = dfout[column] * dfout['sign']
    dfout.drop(labels= ['index', 'sign'], inplace = True, errors = 'ignore', axis=1)

    return dfout

def normalize_pulses(df, column = 'signal', use_sd = False, ddof=0):

    if "phantom" in df.columns:
        # splits df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

    processed = []

    for data in tqdm(df_list):
        data.reset_index(inplace=True)
        if use_sd:
            data['normalization'] = data[column].std(ddof=ddof)
            data[column] =  (data[column] - data[column].mean()) / data[column].std(ddof=ddof)
        else:
            data[column] =  data[column] - data[column].mean()
            data['abs_max'] = data[column].abs().max()

        processed.append(data)

    dfout = pd.concat(processed, axis=0, ignore_index = True)
    if not use_sd:
        dfout[column] = dfout[column] / dfout['abs_max']
        data.rename({'abs_max' : 'normalization'} , inplace=True, axis='columns')
    dfout.drop(labels= ['index'], inplace = True, errors = 'ignore', axis=1)

    return dfout

def normalize2positive_pulses(df, column = 'signal', percent_tol = 0.15, abs_tol = 0.01):

    if "phantom" in df.columns:
        # splits df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"])
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

    processed = []

    for data in tqdm(df_list):
        data.reset_index(inplace=True)
        data[column] =  data[column] - data[column].mean()
        data['abs_max'] = data[column].abs().max()
        data['max'] = data[column].max()
        data['min'] = data[column].min()
        # data['sd'] = data[column].std(axis = 0, ddof=1).values

        processed.append(data)

    dfout = pd.concat(processed, axis=0, ignore_index = True)
    dfout[column] = dfout[column] / dfout['abs_max']
    dfout['max'] = dfout['max'] / dfout['abs_max']
    dfout['min'] = dfout['min'] / dfout['abs_max']
    # dfout[column] = np.where(dfout['min'].abs().to_numpy() > dfout['max'].abs().to_numpy() + tol*dfout['max'].abs().to_numpy(), -dfout[column], dfout[column])
    dfout[column] = np.where((1 - percent_tol) * dfout['max'].abs().to_numpy() > dfout['min'].abs().to_numpy() + abs_tol, dfout[column], -dfout[column])
    data.rename({'abs_max' : 'normalization'} , inplace=True, axis='columns')
    dfout.drop(labels= ['index', 'max', 'min'], inplace = True, errors = 'ignore', axis=1)

    return dfout