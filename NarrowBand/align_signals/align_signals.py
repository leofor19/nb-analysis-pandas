# Python 3.8.12
# 2022-02-07

# Version 1.1.0
# Latest update 2022-02-22

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
# import NarrowBand.analysis_pd.df_compare as dfcomp
# import NarrowBand.analysis_pd.df_processing as dfproc
# import NarrowBand.analysis_pd.df_data_essentials as nbd
# from NarrowBand.analysis_pd.uncategorize import uncategorize

def find_delay_old(s1, s2, max_delay = None):
    """(Depracated) Find delay between two signals using cross-correlation.

    Note that the delay is an integer, indicating the lag by number of samples.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Inputs are expected to be 1-D arrays.

    This version (find_delay_old) used only one direction for the cross-correlation, which can result in wrong values.

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
    delay = -lags[np.argmax(xcorr)]

    if max_delay is None:
        max_delay = np.min([len(s1),len(s2)])

    if delay > max_delay:
        delay = max_delay

    return delay

def find_delay_simple_max_peak(s1, max_delay = None, peak_center = 0):
    """Find delay between two signals by centering highest absolute peak at peak_center.

    Note that the delay is an integer, indicating the lag by number of samples.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Inputs are expected to be 1-D arrays.

    Use input peak_center to select the desired offset for the highest absolute peak. A value of 0 means the center of the array (int(len(s1)/2)).

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    peak_center : int or None, optional
        optional maximum tolerated delay, by default None

    Returns
    -------
    delay : int
        integer output delay, signifying lag by number of samples
    """

    delay = -np.argmax(np.abs(s1)) - int(peak_center)

    if max_delay is None:
        max_delay = np.min([len(s1)])

    if (delay > max_delay) and (delay > 0):
        delay = max_delay
    elif (delay < -max_delay) and (delay < 0):
        delay = -max_delay

    return delay

def find_delay(s1, s2, max_delay = None, out_xcorr = False):
    """Find delay between two signals using cross-correlation.

    Note that the delay is an integer, indicating the lag by number of samples.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Inputs are expected to be 1-D arrays.

    The function applies cross-correlation in both directions, which reduces delay calculation errors for signals 
    with very small number of samples.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    s2 : np.array-like
        input array 2
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    out_xcorr : bool, optional
        set to True to output cross-correlation, by default False

    Returns
    -------
    delay : int
        integer output delay, signifying lag by number of samples
    xcorr_out : np.array, optional
        returned if out_xcorr is set to True
        cross-correlation result between arrays
    """
    s1 = np.real_if_close(s1, tol = 10000)
    s2 = np.real_if_close(s2, tol = 10000)
    xcorr1 = signal.correlate(s1, s2, mode='full', method='auto')
    lags1 = signal.correlation_lags(np.size(s1), np.size(s2), mode='full')
    delay1 = -lags1[np.argmax(xcorr1)]

    xcorr2 = signal.correlate(s2, s1, mode='full', method='auto')
    lags2 = signal.correlation_lags(np.size(s2), np.size(s1), mode='full')
    delay2 = -lags2[np.argmax(xcorr2)]

    if (np.abs(delay1) > np.abs(delay2)):
        delay = -delay2
        xcorr_out = xcorr2
    else:
        delay = delay1
        xcorr_out = xcorr1

    if max_delay is None:
        max_delay = np.min([len(s1),len(s2)])

    if (delay > max_delay) and (delay > 0):
        delay = max_delay
    elif (delay < -max_delay) and (delay < 0):
        delay = -max_delay

    if out_xcorr:
        return delay, xcorr_out
    else:
        return delay

def align_signals(s1, s2, max_delay = None, truncate = True, output_delay = False, assigned_delay = None, out_xcorr = False):
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
    assigned_delay : int or None, optional
        optional value for delay bypassing find_delay function, by default None
    out_xcorr : bool, optional
        set to True to output cross-correlation, by default False

    Returns
    -------
    out1 : np.array
        output aligned array 1
    out2 : np.array
        output aligned array 2
    delay : int, optional
        returned if output_delay is set to True
        integer output delay, signifying lag by number of samples
    xcorr_out : np.array, optional
        returned if out_xcorr is set to True
        cross-correlation result between arrays
    """
    if isinstance(assigned_delay, int):
        # use assigned_delay if it's an integer value
        delay = assigned_delay
    elif out_xcorr:
        delay, xcorr_out = find_delay(s1, s2, max_delay = max_delay, out_xcorr = out_xcorr)
    else:
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
        if out_xcorr:
            return  out1, out2, delay, xcorr_out
        else:
            return  out1, out2, delay
    elif out_xcorr:
        return  out1, out2, xcorr_out
    else:
        return out1, out2