# Python 3.8.12
# 2022-02-07

# Version 1.0.0
# Latest update 2022-02-07

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
# from copy import deepcopy
# from datetime import datetime
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
# import pandas as pd
import scipy.signal as signal
# from scipy import sparse
# import seaborn as sns
# from tqdm import tqdm # when using terminal
# from tqdm.notebook import tqdm # when using Jupyter Notebook
#from tqdm.dask import TqdmCallback

# Local application imports
# import fft_window
# from NarrowBand.nb_czt import czt
# from NarrowBand.nb_czt.fft_window import fft_window
# import NarrowBand.analysis_pd.df_antenna_space as dfant
# import NarrowBand.analysis_pd.df_processing as dfproc
# import NarrowBand.analysis_pd.df_data_essentials as nbd
# from NarrowBand.analysis_pd.uncategorize import uncategorize

def find_delay_old(s1, s2, max_delay = None):
    """Find delay between two signals using cross-correlation.

    Note that the delay is an integer, indicating the lag by number of samples.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Inputs are expected to be 1-D arrays.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    s2 : np.array-like
        input array 2
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None

    Returns
    -------
    delay : int
        integer output delay, signifying lag by number of samples
    """
    xcorr = signal.correlate(s1, s2, mode='full', method='auto')
    lags = signal.correlation_lags(np.size(s1), np.size(s2), mode='full')
    delay = lags[np.argmax(xcorr)]

    if max_delay is None:
        max_delay = np.min([len(s1),len(s2)])

    if delay > max_delay:
        delay = max_delay

    return delay

def find_delay(s1, s2, max_delay = None):
    """Find delay between two signals using cross-correlation.

    Note that the delay is an integer, indicating the lag by number of samples.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Inputs are expected to be 1-D arrays.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    s2 : np.array-like
        input array 2
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None

    Returns
    -------
    delay : int
        integer output delay, signifying lag by number of samples
    """
    xcorr1 = signal.correlate(s1, s2, mode='full', method='auto')
    lags1 = signal.correlation_lags(np.size(s1), np.size(s2), mode='full')
    delay1 = lags1[np.argmax(xcorr1)]

    xcorr2 = signal.correlate(s2, s1, mode='full', method='auto')
    lags2 = signal.correlation_lags(np.size(s2), np.size(s1), mode='full')
    delay2 = lags2[np.argmax(xcorr2)]

    if (np.abs(delay1) > np.abs(delay2)):
        delay = -delay2
    elif (np.abs(delay1) == np.abs(delay2)):
        delay = np.abs(delay1)
    else:
        delay = delay1

    if max_delay is None:
        max_delay = np.min([len(s1),len(s2)])

    if (delay > max_delay) and (delay > 0):
        delay = max_delay
    elif (delay < -max_delay) and (delay < 0):
        delay = -max_delay

    return delay

def align_signals(s1, s2, max_delay = None, truncate = True, output_delay = False):
    """Align two signals by delaying earliest signal.

    Based on MATLAB alignsignals function, finding delay using cross-correlation.

    Inputs are expected to be 1-D arrays.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    s2 : np.array-like
        input array 2
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True
    output_delay : bool, optional
        set to True to also return delay value, by default False

    Returns
    -------
    out1 : np.array
        output aligned array 1
    out2 : np.array
        output aligned array 2
    delay : int, optional
        returned if output_delay is set to True
        integer output delay, signifying lag by number of samples
    """
    delay = find_delay(s1, s2, max_delay = max_delay)

    if delay == 0:
        out1 = s1
        out2 = s2

    elif delay > 0:
        if truncate:
            out1 = s1[:-delay]
            out2 = s2

            # prepending zeroes
            out1 = np.pad(out1, (delay, 0), mode='constant', constant_values = 0)

        else:
            out1 = s1
            # prepending zeroes
            out2 = np.pad(s2, (delay, 0), mode='constant', constant_values = 0)

    else:
        if truncate:
            out1 = s1
            out2 = s2[:delay]

            # prepending zeroes
            out2 = np.pad(out2, ((-delay), 0), mode='constant', constant_values = 0)

        else:
            out2 = s2
            # prepending zeroes
            out1 = np.pad(s1, ((-delay), 0), mode='constant', constant_values = 0)

    if output_delay:
        return  out1, out2, delay
    else:
        return out1, out2