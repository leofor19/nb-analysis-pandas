# Python 3.8.12
# 2022-02-07

# Version 1.2.2
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
    xcorr = signal.correlate((s1 - np.mean(s1)) / np.std(s1), (s2 - np.mean(s2)) / np.std(s2), mode='full', method='auto') / min(len(s1),len(s2)) # normalized cross-correlation
    lags = signal.correlation_lags(np.size(s1), np.size(s2), mode='full')
    delay = -lags[np.argmax(np.abs(xcorr))]

    if max_delay is None:
        max_delay = min(len(s1),len(s2))

    if delay > max_delay:
        delay = max_delay

    return delay

def find_delay_simple_max_peak(s1, max_delay = None, peak_center_offset = 0, peakLowerBound = None, peakUpperBound = None, align_power = False):
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
    peak_center_offset : int, optional
        optional offset to center of array, by default 0
    peakLowerBound: int or None, optional
        optional lower bound position for array max (useful for multiple pulses with known expected position), by default None
    peakUpperBound: int or None, optional
        optional upper bound position for array max (useful for multiple pulses with known expected position), by default None
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

    Returns
    -------
    delay : int
        integer output delay, signifying lag by number of samples
    """
    if (peakLowerBound is not None) or (peakUpperBound is not None):
        if peakLowerBound is None:
            peakLowerBound = 0
        selection = s1[peakLowerBound:peakUpperBound]
    else:
        selection = s1

    if align_power:
        s1 = s1**2

    array_center = len(s1) // 2

    delay = array_center -np.argmax(np.abs(selection)) - int(peak_center_offset)

    if peakLowerBound is not None:
        delay = delay - peakLowerBound

    if max_delay is None:
        max_delay = len(s1)

    if (delay > max_delay) and (delay > 0):
        delay = max_delay
    elif (delay < -max_delay) and (delay < 0):
        delay = -max_delay

    return delay

def simple_center_signal_max_peak(s1, max_delay = None, truncate = True, output_delay = False,
                                     peak_center_offset = 0, peakLowerBound = None, peakUpperBound = None, align_power = False):
    """Center max peak of single signal by prepending or appending zeroes.

    Inputs are expected to be 1-D arrays.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True
    output_delay : bool, optional
        set to True to also return delay value, by default False
    peak_center_offset : int, optional
        optional offset to center of array, by default 0
    peakLowerBound: int or None, optional
        optional lower bound position for array max (useful for multiple pulses with known expected position), by default None
    peakUpperBound: int or None, optional
        optional upper bound position for array max (useful for multiple pulses with known expected position), by default None
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

    Returns
    -------
    out1 : np.array
        output aligned array 1
    delay : int, optional
        returned if output_delay is set to True
        integer output delay, signifying lag by number of samples
    """

    delay = find_delay_simple_max_peak(s1, max_delay = max_delay, peak_center_offset = peak_center_offset, peakLowerBound = peakLowerBound,
                                         peakUpperBound = peakUpperBound, align_power = align_power)

    if delay == 0:
        out1 = s1

    elif delay > 0:
        if truncate:
            out1 = s1[:-delay]
            # prepending zeroes
            out1 = np.pad(out1, (delay, 0), mode='constant', constant_values = 0)

        else:
            # appending zeroes
            out1 = np.pad(s1, (delay, 0), mode='constant', constant_values = 0)

    else:
        if truncate:
            out1 = s1[-delay:]
            # appending zeroes
            out1 = np.pad(out1, (0, -delay), mode='constant', constant_values = 0)

        else:
            # appending zeroes
            out1 = np.pad(s1, (0, -delay), mode='constant', constant_values = 0)

    if output_delay:
        return  out1, delay
    else:
        return out1

def simple_align_signals_max_peak(s1, s2 = None, max_delay = None, truncate = True, output_delay = False,
                                     peak_center_offset = 0, peakLowerBound = None, peakUpperBound = None, align_power = False):
    """Center max peaks of two signal by prepending or appending zeroes.

    Inputs are expected to be 1-D arrays.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    s2 : np.array-like, optional
        input array 2, by default None
    max_delay : int or None, optional
        optional maximum tolerated delay, by default None
    truncate : bool, optional
        set to True to maintain size of input arrays (truncating and prepending zeroes) or to False to implement delay by prepending with zeroes, by default True
    output_delay : bool, optional
        set to True to also return delay value, by default False
    assigned_delay : int or None, optional
        optional value for delay bypassing find_delay function, by default None
    peak_center_offset : int, optional
        optional offset to center of array, by default 0
    peakLowerBound: int or None, optional
        optional lower bound position for array max (useful for multiple pulses with known expected position), by default None
    peakUpperBound: int or None, optional
        optional upper bound position for array max (useful for multiple pulses with known expected position), by default None
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

    Returns
    -------
    out1 : np.array
        output aligned array 1
    out2 : np.array
        output aligned array 2
    delay_diff : int, optional
        returned if output_delay is set to True
        integer output delay, signifying lag by number of samples
        for this function, signifies delay2 - delay1
    """

    out1, delay1 = simple_center_signal_max_peak(s1, max_delay = max_delay, truncate = truncate, output_delay = True, peak_center_offset = peak_center_offset,
                                                 peakLowerBound = peakLowerBound, peakUpperBound = peakUpperBound, align_power = align_power)
    if s2 is not None:
        out2, delay2 = simple_center_signal_max_peak(s2, max_delay = max_delay, truncate = truncate, output_delay = True, peak_center_offset = peak_center_offset,
                                                    peakLowerBound = peakLowerBound, peakUpperBound = peakUpperBound, align_power = align_power)
    else:
        out2 = None
        delay2 = 0

    if output_delay:
        delay_diff = delay2 - delay1
        return  out1, out2, delay_diff
    else:
        return out1, out2

def find_delay(s1, s2, max_delay = None, out_xcorr = False, align_power = False):
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
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

    Returns
    -------
    delay : int
        integer output delay, signifying lag by number of samples
    xcorr_out : np.array, optional
        returned if out_xcorr is set to True
        cross-correlation result between arrays
    """
    if align_power:
        s1 = s1**2
        s2 = s2**2
    else:
        s1 = np.real_if_close(s1, tol = 10000)
        s2 = np.real_if_close(s2, tol = 10000)

    if max_delay is None:
        max_delay = min(len(s1),len(s2))

    xcorr = signal.correlate((s1 - np.mean(s1)) / np.std(s1), (s2 - np.mean(s2)) / np.std(s2), mode='full', method='auto') / min(len(s1),len(s2)) # normalized cross-correlation
    lags = signal.correlation_lags(np.size(s1), np.size(s2), mode='full')

    xcorr_limited = np.where(np.abs(lags) < max_delay, xcorr, 0)
    delay = -lags[np.argmax(np.abs(xcorr_limited))]

    # delay = -lags[np.argmax(np.abs(xcorr))]

    # xcorr2 = signal.correlate((s2 - np.mean(s2)) / np.std(s2), (s1 - np.mean(s1)) / np.std(s1), mode='full', method='auto') / min(len(s1),len(s2)) # normalized cross-correlation
    # lags2 = signal.correlation_lags(np.size(s2), np.size(s1), mode='full')
    # delay2 = -lags2[np.argmax(xcorr2)]

    # if (np.abs(delay1) > np.abs(delay2)):
    #     delay = -delay2
    #     xcorr_out = xcorr2
    # else:
    #     delay = delay1
    #     xcorr_out = xcorr1
    # delay = delay1
    # xcorr_out = xcorr1

    # if max_delay is None:
    #     max_delay = min(len(s1),len(s2))

    # if (delay > max_delay) and (delay > 0):
    #     delay = max_delay
    # elif (delay < -max_delay) and (delay < 0):
    #     delay = -max_delay

    if out_xcorr:
        return delay, xcorr
    else:
        return delay

def matlab_align_signals(s1, s2, max_delay = None, truncate = True, output_delay = False, assigned_delay = None, out_xcorr = False, align_power = False):
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
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

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
        delay, xcorr_out = find_delay(s1, s2, max_delay = max_delay, out_xcorr = out_xcorr, align_power = align_power) # normalized cross-correlation
    else:
        delay = find_delay(s1, s2, max_delay = max_delay, out_xcorr = out_xcorr, align_power = align_power) # normalized cross-correlation

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
            out1 = np.pad(s1, (delay, 0), mode='constant', constant_values = 0)
            # prepending zeroes
            out2 = s2

    else:
        if truncate:
            out1 = s1
            out2 = s2[:delay]

            # prepending zeroes
            out2 = np.pad(out2, (-delay, 0), mode='constant', constant_values = 0)

        else:
            out1 = s1
            # prepending zeroes
            out2 = np.pad(s2, (-delay, 0), mode='constant', constant_values = 0)

    if output_delay:
        if out_xcorr:
            return  out1, out2, delay, xcorr_out
        else:
            return  out1, out2, delay
    elif out_xcorr:
        return  out1, out2, xcorr_out
    else:
        return out1, out2

def primary_align_signals(s1, s2_fixed, max_delay = None, truncate = True, output_delay = False, assigned_delay = None, out_xcorr = False, align_power = False):
    """Align two signals by delaying or anticipating first signal, second signal remains fixed.

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
    truncate : bool, optional (default True)
        set to True to maintain size of input arrays (truncating and prepending/appending zeroes) or to False to implement delay by only prepending/appending with zeroes, by default True
    output_delay : bool, optional
        set to True to also return delay value, by default False
    assigned_delay : int or None, optional
        optional value for delay bypassing find_delay function, by default None
    out_xcorr : bool, optional
        set to True to output cross-correlation, by default False
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

    Returns
    -------
    out1 : np.array
        output aligned array 1
    out2 : np.array
        output unchanged array 2
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
        delay, xcorr_out = find_delay(s1, s2_fixed, max_delay = max_delay, out_xcorr = out_xcorr, align_power = align_power) # normalized cross-correlation
    else:
        delay = find_delay(s1, s2_fixed, max_delay = max_delay, align_power = align_power)

    if delay == 0:
        out1 = s1
        out2 = s2_fixed

    elif delay > 0:
        if truncate:
            out1 = s1[:-delay]
            # prepending zeroes
            out1 = np.pad(out1, (delay, 0), mode='constant', constant_values = 0)
            out2 = s2_fixed

        else:
            # appending zeroes
            out1 = np.pad(s1, (delay, 0), mode='constant', constant_values = 0)
            out2 = s2_fixed

    else:
        if truncate:
            out1 = s1[-delay:]
            # appending zeroes
            out1 = np.pad(out1, (0, -delay), mode='constant', constant_values = 0)
            out2 = s2_fixed

        else:
            # appending zeroes
            out1 = np.pad(s1, (0, -delay), mode='constant', constant_values = 0)
            out2 = s2_fixed

    if output_delay:
        if out_xcorr:
            return  out1, out2, delay, xcorr_out
        else:
            return  out1, out2, delay
    elif out_xcorr:
        return  out1, out2, xcorr_out
    else:
        return out1, out2

def align_signals(s1, s2 = None, max_delay = None, truncate = True, output_delay = False, assigned_delay = None, out_xcorr = False, method = 'primary',
                     peak_center_offset = 0, peakLowerBound = None, peakUpperBound = None, align_power = False):
    """Align two signals.

    Based on MATLAB alignsignals function, finding delay using cross-correlation.

    Default method is 'primary', which keeps s2 fixed, prepending or appending s1 with zeroes. 

    Other option is 'matlab', which delays earliest signal (either s1 or s2), only prepending zeroes.

    Inputs are expected to be 1-D arrays.

    If max_delay is None, the maximum tolerated delay is the smallest length of the input arrays.

    Parameters
    ----------
    s1 : np.array-like
        input array 1
    s2 : np.array-like, optional
        input array 2, by default None
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
        not used for 'simple' method
    method: str, optional
        align signals method, by default 'primary'
        available methods:
            'matlab': MATLAB method, always delays earliest signal
            'primary' or 'fixed': only alters s1, keeping s2 fixed
            'simple' or 'max_peak': centers max of each signal, with optional adjustment using input peak_center
    peak_center_offset : int, optional
        optional offset to center of array (for 'simple' method only), by default 0
    peakLowerBound: int or None, optional
        optional lower bound position for array max (useful for multiple pulses with known expected position), by default None
        (for 'simple' method only)
    peakUpperBound: int or None, optional
        optional upper bound position for array max (useful for multiple pulses with known expected position), by default None
        (for 'simple' method only)
    align_power : bool, optional
        set to True to perform alignment using power of signals, by default False

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

    if (method.casefold() == 'primary') or (method.casefold() == 'fixed'):
        output = primary_align_signals(s1, s2_fixed = s2, max_delay = max_delay, truncate = truncate, output_delay = output_delay, assigned_delay = assigned_delay, 
                                        out_xcorr = out_xcorr, align_power = align_power)
    elif (method.casefold() == 'matlab'):
        output = matlab_align_signals(s1, s2 = s2, max_delay = max_delay, truncate = truncate, output_delay = output_delay, assigned_delay = assigned_delay,
                                         out_xcorr = out_xcorr, align_power = align_power)
    elif (method.casefold() == 'simple') or (method.casefold() == 'max_peak'):
        out_xcorr = False
        output = simple_align_signals_max_peak(s1, s2 = s2, max_delay = max_delay, truncate = truncate, output_delay = output_delay, 
                                                peak_center_offset = peak_center_offset, peakLowerBound = peakLowerBound, peakUpperBound = peakUpperBound, align_power = align_power)
    else:
        print("Align Signals method not available. Please select 'primary' , 'matlab' or 'simple'.")
        return 0

    # output[0] = out1, output[1] = out2, output[2] = delay, output[3] = xcorr_out
    if output_delay:
        if out_xcorr:
            return  output[0], output[1], output[2], output[3]
        else:
            return  output[0], output[1], output[2]
    elif out_xcorr:
        # for this case output[2] = xcorr_out
        return  output[0], output[1], output[2]
    else:
        return output[0], output[1]