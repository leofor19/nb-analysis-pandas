# Python 3.8.12
# 2022-02-16

# Version 1.1.2
# Latest update 2022-03-10

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

    Description:
            Module for aligning signals in the time domain or frequency domain.

            Main functionality mirror MATLAB alignsignals function (using xcorr as basic principle).

    Dependencies::
                    numpy
                    scipy
"""
# Standard library imports
from copy import deepcopy
# from datetime import datetime
import itertools as it
# import json
# import os
# import os.path
# from pathlib import Path
# import re
# import sys
# import warnings

# Third-party library imports
# import dask.dataframe as dd
# from dask.diagnostics import ProgressBar
# from turtle import onclick
# import matplotlib.pyplot as plt
# from natsort import natsorted
import numpy as np
import pandas as pd
import scipy.signal as signal
# from scipy import sparse
# import seaborn as sns
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook
#from tqdm.dask import TqdmCallback

# Local application imports
# import fft_window
# from NarrowBand.nb_czt import czt
# from NarrowBand.nb_czt.fft_window import fft_window
# import NarrowBand.analysis_pd.df_antenna_space as dfant
import NarrowBand.align_signals.align_signals as alsig
import NarrowBand.analysis_pd.df_compare as dfcomp
import NarrowBand.analysis_pd.df_processing as dfproc
# import NarrowBand.analysis_pd.df_data_essentials as nbd
# from NarrowBand.analysis_pd.uncategorize import uncategorize

def df_simple_align_signals(df, column = 'signal', sort_col = 'time', max_delay = None, truncate = True, peak_center = 0):
    """Centers signals of a dataframes by prepending or appending zeroes.

    Based on MATLAB alignsignals function, finding delay using cross-correlation.

    Inputs are expected to be 1-D arrays.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    df : Pandas df or list of df
        input DataFrame or list of DataFrames to align
    df_ref : Pandas df
        input DataFrame with reference signal(s)
    column : str or list of str, optional
        DataFrame column(s) to align, by default 'signal'
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True

    Returns
    -------
    df_out : Pandas df
        output DataFrame with aligned columns in relation to df_ref
    """

    if not isinstance(column, list):
        column = [column]

    if "phantom" in df.columns:
        # splits df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

    processed = []

    for data in tqdm(df_list):
        data.reset_index(inplace=True)
        for p in tqdm(data.pair.unique(), leave = False):
            for c in tqdm(column, leave = False):
                data.loc[data.pair.eq(p), c] = alsig.align_signals(data.loc[data.pair.eq(p), c], max_delay = max_delay, truncate=truncate, output_delay=False, 
                                                                        assigned_delay=None, method = 'simple', peak_center = peak_center)
        processed.append(data)

    df_out = pd.concat(processed, axis=0)

    return df_out

def df_align_signals(df, df_ref, column = 'signal', sort_col = 'time', max_delay = None, truncate = True):
    """Align signals between two dataframes by delaying earliest signal.

    Based on MATLAB alignsignals function, finding delay using cross-correlation.

    Inputs are expected to be 1-D arrays.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    df : Pandas df or list of df
        input DataFrame or list of DataFrames to align
    df_ref : Pandas df
        input DataFrame with reference signal(s)
    column : str or list of str, optional
        DataFrame column(s) to align, by default 'signal'
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True

    Returns
    -------
    df_out : Pandas df
        output DataFrame with aligned columns in relation to df_ref
    """

    # check if df_ref is simple (not an actual scan) and allows it (so we can use ideal pulses, for example)
    if ("cal_type" not in df_ref.columns) and ("phantom" not in df_ref.columns):
        pass
    else:
        # initial check if ref is a single scan, otherwise interrupt execution

        if ("cal_type" not in df_ref.columns):
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"]
        elif ("phantom" in df_ref.columns):
            cols = ["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"]
        else:
            cols = ["cal_type", "date", "rep", "iter", "attLO", "attRF"]

        ref_list = dfproc.split_df(df_ref, groups = cols)

        if len(ref_list) != 1:
            tqdm.write('Reference df_ref contains more than one scan! Please select a single scan.')
            return 0

    if not isinstance(column, list):
        column = [column]

    if ("phantom" in df.columns) and ("phantom" in df_ref.columns):
        # retain only intersection of antenna pairs and frequencies
        df, df_ref = dfcomp.remove_non_intersection(df, df_ref, column = "pair")
    ## not sure if frequencies and times should be removed as well, which would remove samples from the data
    # if ("freq".casefold() in df.columns) and ("freq".casefold() in df_ref.columns):
    #     df, df_ref = dfcomp.remove_non_intersection(df, df_ref, column = "freq")
    # elif ("time".casefold() in df.columns) and ("time".casefold() in df_ref.columns):
    #     df, df_ref = dfcomp.remove_non_intersection(df, df_ref, column = "time")

    # if ("freq".casefold() in df.columns) and ("freq".casefold() in df_ref.columns):
    #     if ("distances".casefold() in df.columns) and ("distances".casefold() in df_ref.columns):
    #         cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "distances"]
    #     else:
    #         cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq"]
    # elif ("time".casefold() in df.columns) and ("time".casefold() in df_ref.columns):
    #     if ("distances".casefold() in df.columns) and ("distances".casefold() in df_ref.columns):
    #         cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "time", "distances"]
    #     else:
    #         cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "time"]
    # else:
    #     print("Dataframes types are incompatible (Frequency Domain vs. Time Domain).")
    #     return 0
    # df = df.set_index(keys = cols, drop=True).sort_index()
    # df_ref = df_ref.set_index(keys = cols, drop=True).sort_index()

    # c = pd.DataFrame().reindex(columns=df.columns)

    # initially checks if df is phantom-based or plain calibrations types 1-3, routines are different
    if "phantom" in df.columns:
        # splits df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

    processed = []

    for data in tqdm(df_list):
        # in_process = []
        data.reset_index(inplace=True)
        data, _ = direct_df_align_signals(data, df_ref, column = column, sort_col = sort_col, max_delay = max_delay, truncate = truncate, fixed_df2=True, method = 'matlab', peak_center = 0)

        processed.append(data)

    df_out = pd.concat(processed, axis=0)

    return df_out

def direct_df_align_signals(df1, df2, column = 'signal', sort_col = 'time', max_delay = None, truncate = True, fixed_df2 = True, method = 'matlab', peak_center = 0):
    """Align signals between two dataframes by delaying earliest signal, assumes both dataframes contain a single scan.

    Note that by default ph_fixed_df2 = True, so it keeps df2 fixed whenever df1 and df2 are of same type (phantom vs. calibrations 2-3).

    When one df is a calibration 2-3, thus without antenna pairs, the calibration is fixed by default.

    Based on MATLAB alignsignals function, finding delay using cross-correlation.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    df1 : Pandas df
        input DataFrame 1
    df2 : Pandas df
        input DataFrame 2
    column : str or list of str, optional
        DataFrame column(s) to align, by default 'signal'
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True
    fixed_df2 : bool, optional
        set to True to ensure shift occurs only to df1 when both dataframes are of same type (phantom vs. calibration 2-3), by default True
        if True, uses align_signals with method 'primary'
        if False, uses align_signals with method 'matlab' or another set by input method
    method: str, optional
        align signals method to be used when fixed_df2 = False, by default 'matlab'
        available methods:
            'matlab': MATLAB method, always delays earliest signal
            'primary' or 'fixed': only alters s1, keeping s2 fixed
            'simple' or 'max_peak': centers max of each signal, with optional adjustment using input peak_center
    peak_center : int, optional
        optional offset to center of array (for 'simple' method only), by default 0

    Returns
    -------
    out1 : np.array
        output aligned array 1
    out2 : np.array
        output aligned array 2
    """

    ph_cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx"]
    cal_cols = ["cal_type", "date", "rep", "iter", "attLO", "attRF"]

    if not isinstance(column, list):
        column = [column]

    # df1 = df1.sort_values(by=sort_col)
    # df2 = df2.sort_values(by=sort_col)
    # df1 = df1.reset_index(drop = False)
    # df2 = df2.reset_index(drop = False)

    if ("pair" in df1.columns) and ("pair" in df2.columns):
        # alignment between phantom scan pulses
        # remove unmatched pairs
        df1, df2 = dfcomp.remove_non_intersection(df1, df2, column = "pair")
        cols = ph_cols + [sort_col]
        df1 = df1.sort_values(by=cols)
        df2 = df2.sort_values(by=cols)
        df1 = df1.reset_index(drop = False)
        df2 = df2.reset_index(drop = False)

        for p in df1.pair.unique():
            for c in tqdm(column, leave = False):
                if fixed_df2:
                    # making sure that the df2 pulse stays fixed, only df1 is shifted (when truncate = True)
                    df1.loc[df1.pair.eq(p), c], df2.loc[df2.pair.eq(p), c] = alsig.align_signals(df1.loc[df1.pair.eq(p), c], df2.loc[df2.pair.eq(p), c], max_delay = max_delay, 
                                                                                                            truncate=truncate, output_delay=False, assigned_delay=None, method= 'primary')
                else:
                    df1.loc[df1.pair.eq(p), c], df2.loc[df2.pair.eq(p), c] = alsig.align_signals(df1.loc[df1.pair.eq(p), c], df2.loc[df2.pair.eq(p), c], max_delay = max_delay,
                                                                                                            truncate=truncate, output_delay=False, assigned_delay=None, method = method, peak_center = peak_center)
    elif ("pair" in df1.columns):
        cols1 = ph_cols + [sort_col]
        if "cal_type" in df2.columns:
            cols2 = cal_cols + [sort_col]
        else:
            cols2 = [sort_col]
        df1 = df1.sort_values(by=cols1)
        df2 = df2.sort_values(by=cols2)
        df1 = df1.reset_index(drop = False)
        df2 = df2.reset_index(drop = False)
        # alignment between one phantom scan and one calibration pulse
        for p in df1.pair.unique():
            for c in tqdm(column, leave = False):
                # making sure that the calibration pulse stays fixed, the phantom scan data is shifted (when truncate = True)
                df1.loc[df1.pair.eq(p), c], df2.loc[:, c] = alsig.align_signals(df1.loc[df1.pair.eq(p), c].to_numpy(), df2.loc[:, c].to_numpy(), max_delay = max_delay, 
                                                                                        truncate=truncate, output_delay=False, assigned_delay=None, method= 'primary')
    elif ("pair" in df2.columns):
        if "cal_type" in df1.columns:
            cols1 = cal_cols + [sort_col]
        else:
            cols1 = [sort_col]
        cols2 = ph_cols + [sort_col]
        df1 = df1.sort_values(by=cols1)
        df2 = df2.sort_values(by=cols2)
        df1 = df1.reset_index(drop = False)
        df2 = df2.reset_index(drop = False)
        # alignment between one phantom scan and one calibration pulse
        for p in df2.pair.unique():
            for c in tqdm(column, leave = False):
                # making sure that the calibration pulse stays fixed, the phantom scan data is shifted (when truncate = True)
                df2.loc[df2.pair.eq(p), c], df1.loc[:, c] = alsig.align_signals(df2.loc[df2.pair.eq(p), c].to_numpy(), df1.loc[:, c].to_numpy(), max_delay = max_delay, 
                                                                                        truncate=truncate, output_delay=False, assigned_delay=None, method= 'primary')
    else:
        if "cal_type" in df1.columns:
            cols1 = cal_cols + [sort_col]
        else:
            cols1 = [sort_col]
        if "cal_type" in df2.columns:
            cols2 = cal_cols + [sort_col]
        else:
            cols2 = [sort_col]
        df1 = df1.sort_values(by=cols1)
        df2 = df2.sort_values(by=cols2)
        df1 = df1.reset_index(drop = False)
        df2 = df2.reset_index(drop = False)
        for c in tqdm(column, leave = False):
            # alignment between calibration pulses
            if fixed_df2:
                # making sure that df2 pulse stays fixed, the phantom scan data is shifted (when truncate = True)
                df1.loc[:, c], df2.loc[:, c] = alsig.align_signals(df1.loc[:, c].to_numpy(), df2.loc[:, c].to_numpy(), max_delay = max_delay, truncate=truncate, 
                                                                                output_delay=False, assigned_delay=None, method= 'primary')
            else:
                df1.loc[:, c], df2.loc[:, c] = alsig.align_signals(df1.loc[:, c].to_numpy(), df2.loc[:, c].to_numpy(), max_delay = max_delay, truncate=truncate, 
                                                                                output_delay=False, assigned_delay=None, method = method, peak_center = peak_center)

    df1 = df1.set_index('index', drop=True)
    df2 = df2.set_index('index', drop=True)

    return df1, df2

def df_align_signals_same_distance(df, column = 'signal', sort_col = 'time', max_delay = None, truncate = True, method = 'matlab', decimals = 4, peak_center = 0):
    """Align signals for antennas with same distance in-between on a same dataframe by delaying earliest signal.

    Based on MATLAB alignsignals function, finding delay using cross-correlation.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    df1 : Pandas df
        input DataFrame 1
    df2 : Pandas df
        input DataFrame 2
    column : str or list of str, optional
        DataFrame column(s) to align, by default 'signal'
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True
    method: str, optional
        align signals method, by default 'matlab'
        available methods:
            'matlab': MATLAB method, always delays earliest signal
            'primary' or 'fixed': only alters s1, keeping s2 fixed
            'simple' or 'max_peak': centers max of each signal, with optional adjustment using input peak_center
    decimals : int, optional
        number of decimals cases for np.arounding values, in particular after conversion, by default 4
    peak_center : int, optional
        optional offset to center of array (for 'simple' method only), by default 0

    Returns
    -------
    df : np.array
        output DataFrame with aligned signals for antennas with same distance in-between
    """

    if 'pair' not in df.columns:
        tqdm.write('DataFrame does not have \'pair\' column.')
        return 0

    if not isinstance(column, list):
        column = [column]

    df_list = dfproc.dfsort_pairs(df, sort_type = "between_antennas", decimals = decimals, out_distances = True, out_as_list = True)

    processed = []

    for data in tqdm(df_list):

        data = data.sort_values(by=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", sort_col])
        data = data.reset_index(drop = False)

        for d in tqdm(data.distances.unique().to_list(), leave = False):
            pairs = data.loc[data.distances.eq(d)].pair.unique().to_list()
            p1 = pairs[0]
            for p2 in tqdm(pairs[1:], leave = False):
                for c in tqdm(column, leave = False):
                    data.loc[data.pair.eq(p1), c], data.loc[data.pair.eq(p2), c] = alsig.align_signals(data.loc[data.pair.eq(p1), c], data.loc[data.pair.eq(p2), c], max_delay = max_delay, 
                                                                                                            truncate=truncate, output_delay=False, assigned_delay=None, method= method, 
                                                                                                            peak_center = peak_center)
                p1 = deepcopy(p2)

        data = data.set_index('index', drop=True)

        processed.append(data)

    df = pd.concat(processed, axis = 0)

    return df