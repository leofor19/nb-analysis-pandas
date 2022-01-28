# Python 3.8.12
# 2021-12-09

# Version 1.0.1
# Latest update 2022-01-28

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

    Description:
            Module for performing Inverse Chirp-Z Transform (ICZT) operations on Pandas DataFrames for the narrow band system.

            Expands on Python module czt by John Garrett (https://github.com/garrettj403/CZT/).

            Also builds upon previous Python 2.7 FFT modules (NarrowBand/analysis/sim_fft.py).

    Dependencies::
                    czt
                    dask
                    natsort
                    numpy
                    pandas
                    pathlib
                    scipy
                    tqdm
                    NarrowBand.analysis.df_antenna_space
                    NarrowBand.analysis_pd.df_data_essentials
                    NarrowBand.analysis.df_processing
                    NarrowBand.analysis_pd.uncategorize
"""
# Standard library imports
from copy import deepcopy
from datetime import datetime
import json
import os
import os.path
from pathlib import Path
import re
import sys
import warnings

# Third-party library imports
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
#import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
# from scipy import sparse
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook
#from tqdm.dask import TqdmCallback
from yaspin import yaspin

# Local application imports
# import fft_window
from NarrowBand.nb_czt import czt
from NarrowBand.nb_czt.fft_window import fft_window
# import NarrowBand.analysis_pd.df_antenna_space as dfant
import NarrowBand.analysis_pd.df_processing as dfproc
import NarrowBand.analysis_pd.df_data_essentials as nbd
from NarrowBand.analysis_pd.uncategorize import uncategorize

def zeroes_for_fft(data_ch1, data_ch2, freqs, f_step = 12.5*1.e6/1, fft_num_samples = 2**16):
    """Return simulated rfft from Q and I channel data at respective input frequencies, padding all other bins with zeroes.

    Parameters
    ----------
    data_ch1 : ndarray
        data from channel 1 (Q-channel)
    data_ch2 : ndarray
        data from channel 2 (I-channel)
    freqs : ndarray
        frequencies array (in MHz)
    f_step : float, optional
        frequency step of the fft (in Hz), preferably in the form 12.5*1.e6 / N to include bins at multiples of 12.5 MHz, by default 12.5*1.e6/2
    fft_num_samples : int, optional
        number of samples for the fft, preferably powers of 2, by default 2**16

    Returns
    -------
    fft_data : ndarray
        simulated fft data
    fft_freqs : ndarray
        frequencies array (in Hz)
    f_step : ndarray
        frequency step of the fft (in Hz)
    """

    #f_step = (12.5*1e6/4)
    #f_step = (12.5*1e6/2)
    fft_freqs = np.arange(0,fft_num_samples*f_step,f_step)

    fft_data = np.zeros(fft_freqs.shape, dtype=complex)

    for i, freq in enumerate(freqs*1e6):
        fft_data[(fft_freqs == freq)] = complex((data_ch2[i]),(-data_ch1[i]))

    return fft_data, fft_freqs, f_step

def generate_freq_array(freqs, max_freq = None, freq_step = None, min_freq = None, fscale = 1e6):
    """Generate array of frequencies for ICZT.

    Parameters
    ----------
    freqs : numpy array-like
        array of available frequencies (in MHz)
    max_freq : float, optional
        set to None for use of maximum frequency of freqs array or else set a value, by default None
    freq_step : float, optional
        set to None for use of minimum frequency difference within freqs array or else set a value, by default None
    min_freq : float, optional
        set to None for use of 0 or the minimum frequency of freqs array (negative values), otherwise set a value, by default None
    fscale : float, optional
        scale for frequencies, by default 1e6 (MHz)

    Returns
    -------
    freqs : numpy arrays
        array of frequencies for ICZT (from 0 to max_freq using freq_step)
    """

    if (freq_step == None) or (max_freq == None):
        freqs = np.unique(freqs)

    if freq_step == None:
        # finds smallest frequency step required
        diffs = np.diff(np.sort(freqs))
        freq_step = diffs[diffs>0].min()

    if max_freq == None:
        max_freq = freqs.max()

    if min_freq == None:
        if freqs.min() > 0:
            min_freq = 0
        else:
            min_freq = freqs.min()

    out_freqs = np.arange(min_freq, max_freq*fscale, step = freq_step*fscale)
    return out_freqs

def place_czt_value(czt_data, f, freqs, df, pair = None, quadrant = 1, I=2, Q=1, signal='voltage', fscale = 1e6):
    """Place czt signal complex value in correct position in array.

    Includes possible "correction" for Complex Plane quadrant, as well as selection of dataframe column and which channel is I/Q-channel.

    Parameters
    ----------
    f : float
        frequency to assign values
    freqs : np.array-like
        array of all frequencies
    df : Pandas df
        df with original narrowband data
    czt_data : np.array-like
        array in contruction with czt signals
    pair : None or str
        antenna pair to assign values (for phantom scans only), by default None
    quadrant : int, optional
        Complex Plane quadrant "correction", by default 1
    I : int, optional
        number of df data channel to be assigned as I-ch, by default 2
    Q : int, optional
        number of df data channel to be assigned as Q-ch, by default 1
    signal : str, optional
        data column identifier string, by default 'voltage'
    fscale : float, optional
        scale for frequencies, by default 1e6 (MHz)

    Returns
    -------
    czt_data_out : np.array-like
        array updated with czt signal values
    """
    if ('voltage' in signal.lower()) and (df.voltage_unit.unique() == 'mV'):
        scale = 1e-3
    else:
        scale = 1

    # quadrant "correction" for the Complex Plane (aiming to place in 1st Quadrant)
    if quadrant == 2:
        s1 = -1 * scale
        s2 = +1 * scale
    elif quadrant == 3:
        s1 = -1 * scale
        s2 = -1 * scale
    elif quadrant == 4:
        s1 = +1 * scale
        s2 = -1 * scale
    else:
        # first quadrant (default)
        s1 = +1 * scale
        s2 = +1 * scale

    # df column strings assignment
    Icol = "".join((signal.lower(),"_ch",str(I)))
    Qcol = "".join((signal.lower(),"_ch",str(Q)))

    czt_data_out = deepcopy(czt_data)

    if ("pair" in df.columns) and (pair is not None):
        czt_data_out = np.where(freqs == f*fscale, complex((s1 * df.loc[(df.pair.eq(pair)) & (df.freq.eq(f)), Icol]), (s2 * df.loc[(df.pair.eq(pair)) & (df.freq.eq(f)), Qcol])), czt_data)
    elif ("pair" in df.columns):
        raise ValueError('Input \'pair\' value expected for phantom scans.')
    else:
        czt_data_out = np.where(freqs == f*fscale, complex((s1 * df.loc[(df.freq.eq(f)), Icol]), (s2 * df.loc[(df.freq.eq(f)), Qcol])), czt_data)

    return czt_data_out

def apply_fft_window(td_df, window_type = 'hann'):
    """Apply window to time-domain for reducing sidelobes due to FFT of incoherently sampled data.

        Window type is a case INsensitive string and can be one of:
            'Hamming', 'Hann', 'Blackman', 'BlackmanExact', 'BlackmanHarris70',
            'FlatTop', 'BlackmanHarris92'
            The default window type is 'Hann'.

        Window is to be applied to time-domain signal.

    Parameters
    ----------
    td_df : Pandas df
        dataframe with time-domain signals
    window_type : str, optional
        string with window type (as explained above), by default 'hann'

    Returns
    -------
    td_df_out : Pandas df
        dataframe after window application
    """
    td_df_out = deepcopy(td_df)
    # if dataset contains antenna pairs, it will follow phantom data scan format, otherwise it contains cal_type 1-3
    if ('pair' in td_df.columns):
        for ph in tqdm(td_df.phantom.unique()):
            for plug in tqdm(td_df.plug.unique(), leave=False):
                for date in tqdm(td_df.date.unique(), leave=False):
                    for rep in tqdm(td_df.rep.unique(), leave=False):
                        for ite in tqdm(td_df.iter.unique(), leave=False):
                            for p in tqdm(td_df.pair.unique(), leave=False):
                                data = td_df.loc[(td_df.phantom.eq(ph)) & (td_df.plug.eq(plug)) & (td_df.date.eq(date)) & (td_df.rep.eq(rep)) & (td_df.iter.eq(ite)) & (td_df.pair.eq(p)),:]
                                window = fft_window(data.signal.size, window_type=window_type)
                                td_df_out.loc[(td_df.phantom.eq(ph)) & (td_df.plug.eq(plug)) & (td_df.date.eq(date)) & (td_df.rep.eq(rep)) & (td_df.iter.eq(ite)) 
                                                & (td_df.pair.eq(p)), "signal"] = window * data.loc[:, "signal"]
    else:
        for cal_type in tqdm(td_df.cal_type.unique()):
            for date in tqdm(td_df.date.unique(), leave=False):
                for rep in tqdm(td_df.rep.unique(), leave=False):
                    for ite in tqdm(td_df.iter.unique(), leave=False):
                        for p in tqdm(td_df.pair.unique(), leave=False):
                            window = fft_window(td_df.loc[:, "signal"].size, window_type=window_type)
                            td_df_out.loc[:, "signal"] = window * td_df.loc[:, "signal"]
    return td_df_out

def df_to_freq_domain(df, max_freq = None, freq_step = None, min_freq = None, conj_sym=True, auto_complex_plane = True, quadrant = 1, I=2, Q=1, signal='voltage', fscale = 1e6, verbose = False):
    """Convert phantom data scan DataFrame into new Dataframe with reconstructed CZT signals.

    The new DataFrame contains columns for each antenna pair, for which there are two subcolumns: frequencies and simulated CZT signal.

    The reconstructed CZT signal is the frequency domain representation of the scans, composed of zeroes for unused frequencies 
    and complex value measurements (I-channel - j*Q-channel) for each input frequency used.

    Parameters
    ----------
    df : Pandas df
        phantom data scan DataFrame
    max_freq : float, optional
        set to None for use of maximum frequency within df or else set a value, by default None
    freq_step : float, optional
        set to None for use of minimum frequency difference within df or else set a value, by default None
    min_freq : float, optional
        set to None for use of 0 or the minimum frequency of freqs array (negative values), otherwise set a value, by default None
    conj_sym: bool, optional (default True)
        set to True to convert FD signal to conjugate symmetrical (to force real signal), by default True
    auto_complex_plane: bool, optional (default True)
        set to True to automatically estimate Complex Plane quadrant and I-ch, Q-ch number, by default True
    quadrant : int, optional
        Complex Plane quadrant "correction", by default 1
    I : int, optional
        number of df data channel to be assigned as I-ch, by default 2
    Q : int, optional
        number of df data channel to be assigned as Q-ch, by default 1
    signal : str, optional
        data column identifier string, by default 'voltage'
    fscale : float, optional
        scale for frequencies, by default 1e6 (MHz)
    verbose: bool, optional
        verbosity for auto_complex_plane
        set to True to print quadrant, I-ch and Q-ch, by default False

    Returns
    -------
    df_out : Pandas df
        output DataFrame with reconstructed CZT signals (frequency domain)
    """
    if auto_complex_plane:
        quadrant, I, Q = auto_detect_complex_plane(df, signal = signal, verbose = verbose)

    # initially checks if df is phantom-based or plain calibrations types 1-3, routines are different
    if "phantom" in df.columns:
        # splits df in lists of dfs per grouping, criteria depending on 'calibration type 4' (phantom with Tx-off) or 'phantom scan'
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])

        processed = []
        in_process = []

        for data in tqdm(df_list):
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale)
            czt_data_shape = np.zeros(freqs.shape, dtype=complex)
            for pair in tqdm(data.pair.unique(), leave=False):
                # creates frequency and CZT data arrays per antenna pair
                czt_data = deepcopy(czt_data_shape)
                for f in tqdm(freq, leave=False):
                    czt_data = place_czt_value(czt_data = czt_data, f = f, freqs = freqs, df = data, pair = pair, quadrant = quadrant, I=I, Q=Q, signal=signal)
                if conj_sym:
                    freqs_out, czt_data = conjugate_symmetric(freqs, czt_data)
                else:
                    freqs_out = freqs
                fd_data = pd.concat([pd.DataFrame({'freq': freqs_out}),  pd.DataFrame({'czt': czt_data})], axis=1)
                # fd_data.columns = pd.MultiIndex.from_product([[pair],['freq','czt']])
                fd_data["pair"] = pair
                fd_data["Tx"] = int(data.loc[data.pair.eq(pair), 'Tx'].unique())
                fd_data["Rx"] = int(data.loc[data.pair.eq(pair), 'Rx'].unique())
                in_process.append(fd_data)
            czt_df = pd.concat(in_process, axis=0)

            czt_df["phantom"] = data.phantom[0]
            czt_df["angle"] = data.angle[0]
            czt_df["plug"] = data.plug[0]
            czt_df["date"] = data.date[0]
            czt_df["rep"] = data.rep[0]
            czt_df["iter"] = data.iter[0]
            czt_df["attLO"] = data.attLO[0]
            czt_df["attRF"] = data.attRF[0]
            if "cal_type" in data.columns:
                czt_df["cal_type"] = data.cal_type[0]

            processed.append(czt_df)
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

        processed = []

        for data in tqdm(df_list):
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale)
            czt_data = np.zeros(freqs.shape, dtype=complex)
            for f in tqdm(freq, leave=False):
                czt_data = place_czt_value(czt_data = czt_data, f = f, freqs = freqs, df = data, pair = None, quadrant = quadrant, I=I, Q=Q, signal=signal)
            if conj_sym:
                    freqs_out, czt_data = conjugate_symmetric(freqs, czt_data)
            else:
                freqs_out = freqs
            czt_df = pd.DataFrame({'freq': freqs_out, 'czt': czt_data})

            czt_df["cal_type"] = data.cal_type[0]
            czt_df["date"] = data.date[0]
            czt_df["rep"] = data.rep[0]
            czt_df["iter"] = data.iter[0]
            czt_df["attLO"] = data.attLO[0]
            czt_df["attRF"] = data.attRF[0]

            processed.append(czt_df)

    df_out = pd.concat(processed, axis=0, ignore_index=True)
    return df_out

def array_invert_to_time_domain(freqs, czt_data):

    f, d = conjugate_symmetric(freqs, czt_data)

    N = len(czt_data)/2
    time, sig_t = czt.freq2time(f, d)
    return time, N*sig_t

def df_invert_to_time_domain(df, max_freq = None, freq_step = None, t = ' auto', min_freq = None, conj_sym=True, auto_complex_plane = True, 
                                quadrant = 1, I=2, Q=1, signal='voltage', fscale = 1e6, verbose = False):
    """Convert phantom data scan DataFrame into new Dataframe with converted ICZT signals.

    The new DataFrame contains columns for each antenna pair, for which there are two subcolumns: frequencies and converted ICZT signal.

    The ICZT signal is the time domain representation of the scans, converted using the Inverse Chirp-Z Transform algorithm.

    Parameters
    ----------
    df : Pandas df
        phantom data scan DataFrame
    max_freq : float, optional
        set to None for use of maximum frequency within df or else set a value, by default None
    freq_step : float, optional
        set to None for use of minimum frequency difference within df or else set a value, by default None
    t : np.ndarray or str or None
        optional time array for output signal, by default 'auto'
        set to 'auto' for use of auto_time_array() function
        set to None for standard FFT time sweep
    min_freq : float, optional
        set to None for use of 0 or the minimum frequency of freqs array (negative values), otherwise set a value, by default None
    conj_sym: bool, optional (default True)
        set to True to convert FD signal to conjugate symmetrical (to force real signal), by default True
    auto_complex_plane: bool, optional (default True)
        set to True to automatically estimate Complex Plane quadrant and I-ch, Q-ch number, by default True
    quadrant : int, optional
        Complex Plane quadrant "correction", by default 1
    I : int, optional
        number of df data channel to be assigned as I-ch, by default 2
    Q : int, optional
        number of df data channel to be assigned as Q-ch, by default 1
    signal : str, optional
        data column identifier string, by default 'voltage'
    fscale : float, optional
        scale for frequencies, by default 1e6 (MHz)
    verbose: bool, optional
        set to True to print quadrant, I-ch and Q-ch, by default False

    Returns
    -------
    df_out : Pandas df
        output DataFrame with ICZT signals (time domain)
    """

    if auto_complex_plane:
        quadrant, I, Q = auto_detect_complex_plane(df, signal = signal, verbose = verbose)

    if "phantom" in df.columns:
        if "cal_type" in df.columns:
            df_list = dfproc.split_df(df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
        else:
            df_list = dfproc.split_df(df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])

        processed = []
        in_process = []

        for data in tqdm(df_list):
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale)
            czt_data_shape = np.zeros(freqs.shape, dtype=complex)
            for pair in tqdm(data.pair.unique(), leave=False):
                czt_data = deepcopy(czt_data_shape)
                for f in tqdm(freq, leave=False):
                    czt_data = place_czt_value(czt_data = czt_data, f = f, pair = pair, freqs = freqs, df = data, quadrant = quadrant, I=I, Q=Q, signal=signal)
                if conj_sym:
                    freqs_out, czt_data = conjugate_symmetric(freqs, czt_data)
                else:
                    freqs_out = freqs
                if t == 'auto':
                    t2 = auto_time_array(freqs_out, start = 0, multiple=1)
                else:
                    t2 = t
                N = len(czt_data)/2
                time, sig_t = czt.freq2time(freqs_out, czt_data, t = t2)
                td_data = pd.concat([pd.DataFrame({'time': time}),  pd.DataFrame({'signal': N*sig_t})], axis=1)
                # td_data.columns = pd.MultiIndex.from_product([[pair],['time','signal']])
                td_data["pair"] = pair
                td_data["Tx"] = int(data.loc[data.pair.eq(pair), 'Tx'].unique())
                td_data["Rx"] = int(data.loc[data.pair.eq(pair), 'Rx'].unique())
                in_process.append(td_data)
            iczt_df = pd.concat(in_process, axis=0)

            iczt_df["phantom"] = data.phantom[0]
            iczt_df["angle"] = data.angle[0]
            iczt_df["plug"] = data.plug[0]
            iczt_df["date"] = data.date[0]
            iczt_df["rep"] = data.rep[0]
            iczt_df["iter"] = data.iter[0]
            iczt_df["attLO"] = data.attLO[0]
            iczt_df["attRF"] = data.attRF[0]
            if "cal_type" in data.columns:
                iczt_df["cal_type"] = data.cal_type[0]

            processed.append(iczt_df)

    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

        processed = []
        for data in tqdm(df_list):
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale)
            czt_data = np.zeros(freqs.shape, dtype=complex)
            for f in tqdm(freq, leave=False):
                czt_data = place_czt_value(czt_data = czt_data, f = f, freqs = freqs, df = data, pair = None, quadrant = quadrant, I=I, Q=Q, signal=signal)
            if conj_sym:
                    freqs_out, czt_data = conjugate_symmetric(freqs, czt_data)
            else:
                freqs_out = freqs
            if t == 'auto':
                t2 = auto_time_array(freqs_out, start = 0, multiple=1)
            else:
                t2 = t
            N = len(czt_data)/2
            time, sig_t = czt.freq2time(freqs_out, czt_data, t = t2)
            iczt_df = pd.DataFrame({'time': time, 'signal': N*sig_t})

            iczt_df["cal_type"] = data.cal_type[0]
            iczt_df["date"] = data.date[0]
            iczt_df["rep"] = data.rep[0]
            iczt_df["iter"] = data.iter[0]
            iczt_df["attLO"] = data.attLO[0]
            iczt_df["attRF"] = data.attRF[0]

            processed.append(iczt_df)

    df_out = pd.concat(processed, axis=0, ignore_index=True)
    return df_out

def czt_df_invert_to_time_domain(czt_df, t = None, conj_sym=True):
    """Convert CZT DataFrame into new Dataframe with converted ICZT signals.

    The new DataFrame contains columns for each antenna pair, for which there are two subcolumns: frequencies and converted ICZT signal.

    The ICZT signal is the time domain representation of the scans, converted using the Inverse Chirp-Z Transform algorithm.

    Parameters
    ----------
    czt_df : Pandas df
        DataFrame with reconstructed CZT signals (frequency domain)
    t : np.ndarray
        time for output signal, optional, defaults to standard FFT time sweep
    conj_sym: bool, optional (default True)
        set to True to convert FD signal to conjugate symmetrical (to force real signal), by default True

    Returns
    -------
    iczt_df : Pandas df
        output DataFrame with ICZT signals (time domain)
    """
    if "phantom" in czt_df.columns:
        if "cal_type" in czt_df.columns:
            df_list = dfproc.split_df(czt_df, groups=["cal_type", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])
        else:
            df_list = dfproc.split_df(czt_df, groups=["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"])


        processed = []
        in_process = []

        for data in tqdm(df_list):
            for p in data.pair:
                scan = data.loc[:, data.pair.eq(p)]
                if conj_sym:
                    freqs, czt_data = conjugate_symmetric(scan.freq,scan.czt)
                else:
                    freqs = data.freq.to_numpy()
                    czt_data = data.czt.to_numpy()
                if t == 'auto':
                    t2 = auto_time_array(freqs*1e6, start = 0, multiple=1)
                else:
                    t2 = t
                N = len(czt_data)/2
                time, sig_t = czt.freq2time(freqs, czt_data, t=t2)
                td_data = pd.concat([pd.DataFrame({'time': time}),  pd.DataFrame({'signal': N*sig_t})], axis=1)
                # td_data.columns = pd.MultiIndex.from_product([[p],['time','signal']])
                td_data["pair"] = p
                td_data["Tx"] = int(data.loc[data.pair.eq(p), 'Tx'].unique())
                td_data["Rx"] = int(data.loc[data.pair.eq(p), 'Rx'].unique())
                in_process.append(td_data)
            iczt_df = pd.concat(in_process, axis=0)

            iczt_df["phantom"] = data.phantom[0]
            iczt_df["angle"] = data.angle[0]
            iczt_df["plug"] = data.plug[0]
            iczt_df["date"] = data.date[0]
            iczt_df["rep"] = data.rep[0]
            iczt_df["iter"] = data.iter[0]
            iczt_df["attLO"] = data.attLO[0]
            iczt_df["attRF"] = data.attRF[0]
            if "cal_type" in data.columns:
                iczt_df["cal_type"] = data.cal_type[0]

            processed.append(iczt_df)

    else:
        df_list = dfproc.split_df(czt_df, groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

        processed = []
        for data in tqdm(df_list):
            if conj_sym:
                freqs, czt_data = conjugate_symmetric(data.freq,data.czt)
            else:
                freqs = data.freq.to_numpy()
                czt_data = data.czt.to_numpy()
            if t == 'auto':
                    t2 = auto_time_array(freqs, start = 0, multiple=1)
            else:
                t2 = t
            N = len(czt_data)/2
            time, sig_t = czt.freq2time(freqs, czt_data, t=t2)
            iczt_df = pd.DataFrame({'time': time, 'signal': N*sig_t})

            iczt_df["cal_type"] = data.cal_type[0]
            iczt_df["date"] = data.date[0]
            iczt_df["rep"] = data.rep[0]
            iczt_df["iter"] = data.iter[0]
            iczt_df["attLO"] = data.attLO[0]
            iczt_df["attRF"] = data.attRF[0]

            processed.append(iczt_df)

    df_out = pd.concat(processed, axis=0, ignore_index=True)
    return df_out

def conjugate_symmetric(x, y):
    """Return conjugate symmetric array for y and correspondent x array with negative values.

    Conjugate symmetric vectors respect the relation y[x] = y*[-x]. In order for an IFFT signal to be real valued, the FFT signal needs to be conjugate symmetric.

    The function first checks if the lengths of x and y are the same, then checks if x only has positive values.
    If these conditions are not met and y isn't already conjugate symmetric, the funtion is interrupted
    and returns 0. If there are negative values for x *and* y is already conjugate symmetric, it returns the unaltered arrays.

    If x is strictly positive and missing zero, both x and y arrays are extended with extra samples (zeroes for y).

    Finally, both arrays are extended, x with negative values and y with conjugate symmetric values.

    Parameters
    ----------
    x : array-like
        array to include negative values (time or frequency)
    y : array-like
        array to become conjugate symmetric (signal)

    Returns
    -------
    x_out : array-like
        array with negative values
    y_out : array-like
        conjugate symmetric array
    """
    if len(x) != len(y):
        print("x and y arrays have different lengths!")
        return 0

    if np.min(x) < 0:
        if check_symmetric(y):
            # y is already conjugate symmetric, returns unchanged arrays
            return x, y
        else:
            print("x array already has negative values but not conjugate symmetric, please verify.")
            return 0
    elif np.round(np.min(x)) != 0:
        print("x array missing zero, padding arrays.")
        # gets minimum step value
        diffs = np.diff(np.sort(x))
        step = diffs[diffs>0].min()
        # extra values for x, starting from 0
        extra = np.arange(0, x[0], step = step)
        x = np.concatenate((extra, x))
        # pads y array with zeroes to match the new x array
        y = np.concatenate((np.zeros(len(extra)), y))
    else:
        pass

    # allocating arrays, y_out needs to be explicitly complex
    n = len(x) - 1
    x_out = np.zeros(2*n + 1)
    y_out = np.zeros(2*n + 1, dtype=complex)
    reversed = y[::-1]
    reversed_x = x[::-1]

    # # obtaining step value for x
    # nx = len(x)
    # xspan = x[-1] - x[0]
    # dx = xspan / (nx - 1)  # more accurate than x[1] - x[0]

    # output arrays in natural order
    # x_out[0:n] = np.arange(-x[-1], 0, step= dx)
    x_out[0:n] = - reversed_x[:-1]
    x_out[n:] = x
    y_out[0:n+1] = np.conjugate(reversed[:n+1])
    y_out[n:] = y

    return x_out, y_out

def conjugate_symmetric_fft_format(x, y):
    """Return FFT-format conjugate symmetric array for y and correspondent x array with negative values.

    FFT-format convention has a specific order, so that y[0] is the DC component/average, y[1:n/2+2] corresponds to increasing positive values of x
    and y[n/2+1:] corresponds to negative values of x.

    This format may be necessary for IFFT functions.

    Parameters
    ----------
    x : array-like
        array to include negative values (time or frequency)
    y : array-like
        array to become conjugate symmetric (signal)

    Returns
    -------
    x_out : array-like
        array with negative values
    y_out : array-like
        conjugate symmetric array in FFT-format
    """
    if np.min(x) >= 0:
        # converts y array to conjugate symmetric
        x,y = conjugate_symmetric(x, y)

    # rearranges order of elements, so that negative values of x are after the positive values
    n = int((len(x) - 1)/2)
    x_out = np.concatenate((x[n:], x[0:n]))
    y_out = np.concatenate((y[n:], y[0:n]))

    return x_out, y_out

def auto_time_array(f, start = 0, multiple=1):

    if f[0] < 0:
        # Input frequency array
        nf = int(len(f)/2)
        fspan = f[-1] - 0
        df = fspan / (nf - 1)  # more accurate than f[1] - f[0]
    else:
        # Input frequency array
        nf = len(f)
        fspan = f[-1] - f[0]
        df = fspan / (nf - 1)  # more accurate than f[1] - f[0]

    # Output time array (from cbz.freq2time())
    # Default to FFT time sweep
    t = np.fft.fftshift(np.fft.fftfreq(nf, df/multiple))

    # # final time is chosen *multiple* divided by max frequency
    # max_time = multiple / (8*df)

    # # time step is 1 divided by *multiple* times the frequency step
    # tstep = 1/(4*multiple*f[-1])

    # t = np.arange(start, max_time, step = tstep)
    return t

def check_symmetric(a, tol=1e-8):
    """Check if array is conjugate symmetric.

    Adapted from https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy

    Parameters
    ----------
    a : array-like
        array to check
    tol : float, optional
        tolerance, by default 1e-8

    Returns
    -------
    bool
       True if conjugate symmetric, False otherwise
    """
    return np.linalg.norm(a-np.conjugate(a[::-1]), np.Inf) < tol;

def auto_detect_complex_plane(df, signal="voltage", verbose=False):
    """Estimate Complex Plane definitions (quadrant, I-ch, Q-ch).

    Selects the channel with highest absolute median value as in-phase (I-ch), the lower as quadrature (Q-ch).
    Afterwards, identifies the Complex Plane quadrant by verifying the signal of the medians.
    It is convenient for the ICZT functions to reflect values to the 1st Quadrant (positive real and complex values).

    Parameters
    ----------
    df : Pandas df
        input dataframe
    signal : str, optional
        column base name for data, by default "voltage"
    verbose: bool, optional
        set to True to print quadrant, I-ch and Q-ch, by default False

    Returns
    -------
    quadrant : int
        estimated quadrant in the complex plane (between 1 and 4)
    I : int
        estimated number of in-phase channel (1 or 2)
    Q : int
        estimated number of quadrature channel (1 or 2)
    """
    col1 = "".join((signal,"_ch1"))
    col2 = "".join((signal,"_ch2"))

    # uses median of entire column as estimate
    est1 = df[col1].median()
    est2 = df[col2].median()

    if 10*np.abs(est1) < np.abs(est2):
        # reverses I and Q channels due to higher value in ch2
        I = 2
        Q = 1
        if (est2 < 0) and (est1 > 0):
            quadrant = 2
        elif (est2 < 0) and (est1 < 0):
            quadrant = 3
        elif (est2 > 0) and (est1 < 0):
            quadrant = 4
        else:
            quadrant = 1
    else:
        I = 1
        Q = 2
        if (est1 < 0) and (est2 > 0):
            quadrant = 2
        elif (est1 < 0) and (est2 < 0):
            quadrant = 3
        elif (est1 > 0) and (est2 < 0):
            quadrant = 4
        else:
            quadrant = 1

    if verbose:
        print(f"Median Ch. 1: {est1}, Median Ch. 2: {est2}")
        print(f"Quadrant: {quadrant}, I-channel: {I}, Q-channel: {Q}")

    return quadrant, I, Q