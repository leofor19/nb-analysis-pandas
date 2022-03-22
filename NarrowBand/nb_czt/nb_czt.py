# Python 3.8.12
# 2021-12-09

# Version 1.1.0
# Latest update 2022-03-12

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
from calendar import c
from copy import deepcopy
from datetime import datetime
import json
import os
import os.path
from pathlib import Path
import re
import sys
import warnings
from attr import validate

# Third-party library imports
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import matplotlib.pyplot as plt
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

def generate_freq_array(freqs, max_freq = None, freq_step = None, min_freq = None, fscale = 1e6, extra_freqs = 0):
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
    extra_freqs: int, optional
        extra frequencies (zeroes) around those present in freqs, by default 0

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
        max_freq = freqs.max() + freq_step*(extra_freqs + 1)

    if min_freq == None:
        # if freqs.min() > 0:
        #     min_freq = 0
        # else:
        #     min_freq = freqs.min()
        min_freq = freqs.min() - freq_step*extra_freqs

    out_freqs = np.arange(min_freq*fscale, max_freq*fscale, step = freq_step*fscale)
    return out_freqs

def auto_time_array(f, periods = 1, step_division = 1, start = None, max_time = None, tstep = None):
    """Generate default ifft time array.

    When setting start, max_time or tstep as None, uses default FFT time array as basis by performing [np.fft.fftshift(np.fft.fftfreq(nf, df/step_division))].

    For more automated functionality, only alter periods and step_division.

    Function applies np.arange for periods different from 1 and for cases with user-defined start, max_time or tstep.

    Parameters
    ----------
    f : np.array-like
        input array with frequencies
    periods : int, optional
        number of periods to use, by default 1
    step_division : int, optional
        integer to divide time step, by default 1
    start : None or float, optional
        initial time position, by default None
    max_time : None or float, optional
        final time position, by default None
    tstep : None or float, optional
        time step, by default None

    Returns
    -------
    t : np.array
        output time array for converting signal to Time Domain
    """

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
    if (tstep is None) or (start is None) or (max_time is None) :
        t = np.fft.fftshift(np.fft.fftfreq(nf, df/step_division))

    # adjustments to default IFFT array
    if (start is not None) or (max_time is not None) or (periods != 1):
        periods = np.around(periods)
        if tstep is None:
            tstep = (t[-1] - t[0]) / (len(t)) / step_division
        if start is None:
            start = t[0] * periods
        if max_time is None:
            max_time = t[-1] * periods

        t = np.arange(start, max_time, step = tstep)

    # # final time is chosen *multiple* divided by max frequency
    # max_time = multiple / (8*df)

    # # time step is 1 divided by *multiple* times the frequency step
    # tstep = 1/(4*multiple*f[-1])

    # t = np.arange(start, max_time, step = tstep)
    return t

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

def apply_fft_window(td_df, window_type = 'hann', use_scipy=False, column = 'signal', conj_sym = False ):
    """Apply window to time-domain for reducing sidelobes due to FFT of incoherently sampled data.

        Window type is a case INsensitive string and can be one of:
            'Hamming', 'Hann', 'Blackman', 'BlackmanExact', 'BlackmanHarris70',
            'FlatTop', 'BlackmanHarris92'
            The default window type is 'Hann'.

        Window is to be applied to time-domain signal.

        Can receive directly a window as a Numpy array (for window_type), which requires correct array length.

        Can also apply scipy.signal.get_window(), which has more options. Some of these require tuples for window_type.
        Please see Scipy docs for further information (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)

    Parameters
    ----------
    td_df : Pandas df
        dataframe with time-domain signals
    window_type : str or nd.ndarray-like, optional
        string with window type (as explained above), by default 'hann'
        can also receive a Numpy array for direct window definition, but requires attention to the length!
    use_scipy: bool, optional
        set to True to use scipy.signal.get_window function, by default False
        see Scipy docs for further information (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html)
        some window types require tuples for window_type
    column: str or List[str], optional
        df column to apply window, by default 'signal'
    conj_sym: bool, optional
        set to True to apply window separately to negative and positive frequencies, by default False

    Returns
    -------
    td_df_out : Pandas df
        dataframe after window application
    """
    td_df_out = deepcopy(td_df)

    if ~isinstance(column, list):
        column = [column]

    if window_type is None:
        return td_df_out
    else:
        # if dataset contains antenna pairs, it will follow phantom data scan format, otherwise it contains cal_type 1-3
        if ('pair' in td_df.columns):
            for ph in tqdm(td_df.phantom.unique()):
                for plug in tqdm(td_df.plug.unique(), leave=False):
                    for date in tqdm(td_df.date.unique(), leave=False):
                        for rep in tqdm(td_df.rep.unique(), leave=False, disable = True):
                            for ite in tqdm(td_df.iter.unique(), leave=False, disable = True):
                                for p in tqdm(td_df.pair.unique(), leave=False, disable = True):
                                    data = td_df.loc[(td_df.phantom.eq(ph)) & (td_df.plug.eq(plug)) & (td_df.date.eq(date)) & (td_df.rep.eq(rep)) & (td_df.iter.eq(ite)) & (td_df.pair.eq(p)),:]
                                    for c in column:
                                        if conj_sym:
                                            if use_scipy:
                                                window = signal.get_window(window=window_type, Nx = (data[c].size - 1)/2, fft_bins = True)
                                            elif isinstance(window_type,np.ndarray):
                                                window = window_type
                                            else:
                                                window = fft_window((data[c].size - 1)/2, window_type=window_type)
                                            window = np.concatenate((window,np.zeros(1),window))
                                        else:
                                            if use_scipy:
                                                window = signal.get_window(window=window_type, Nx = data[c].size, fft_bins = True)
                                            elif isinstance(window_type,np.ndarray):
                                                window = window_type
                                            else:
                                                window = fft_window(data[c].size, window_type=window_type)
                                        td_df_out.loc[(td_df.phantom.eq(ph)) & (td_df.plug.eq(plug)) & (td_df.date.eq(date)) & (td_df.rep.eq(rep)) & (td_df.iter.eq(ite)) 
                                                        & (td_df.pair.eq(p)), c] = data[c].multiply(window, axis = 0)
        else:
            for cal_type in tqdm(td_df.cal_type.unique()):
                for date in tqdm(td_df.date.unique(), leave=False):
                    for rep in tqdm(td_df.rep.unique(), leave=False):
                        for ite in tqdm(td_df.iter.unique(), leave=False):
                            data = td_df.loc[(td_df.cal_type.eq(cal_type)) & (td_df.date.eq(date)) & (td_df.rep.eq(rep)) & (td_df.iter.eq(ite)),:]
                            for c in column:
                                if conj_sym:
                                    if use_scipy:
                                        window = signal.get_window(window=window_type, Nx = (data[c].size - 1)/2, fft_bins = True)
                                    elif isinstance(window_type,np.ndarray):
                                        window = window_type
                                    else:
                                        window = fft_window((data[c].size - 1)/2, window_type=window_type)
                                    window = np.concatenate((window,np.zeros(1),window))
                                else:
                                    if use_scipy:
                                        window = signal.get_window(window=window_type, Nx = data[c].size, fft_bins = True)
                                    elif isinstance(window_type,np.ndarray):
                                        window = window_type
                                    else:
                                        window = fft_window(data[c].size, window_type=window_type)
                                td_df_out.loc[(td_df.cal_type.eq(cal_type)) & (td_df.date.eq(date)) & (td_df.rep.eq(rep)) & (td_df.iter.eq(ite)), c] = data[c].multiply(window, axis = 0)
        return td_df_out

def df_to_freq_domain(df, max_freq = None, freq_step = None, min_freq = None, conj_sym_partial = False, conj_sym=False, auto_complex_plane = False,
                        quadrant = 1, I=2, Q=1, signal='voltage', fscale = 1e6, extra_freqs = 0, verbose = False):
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
    conj_sym_partial: bool, optional (default False)
        set to True to convert FD signal to conjugate symmetrical (to force real signal) without zero padding, by default False
    conj_sym: bool, optional (default False)
        set to True to convert FD signal to conjugate symmetrical (to force real signal) with zero padding, by default False
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
    extra_freqs: int, optional
        extra frequencies (zeroes) around those present in freqs, by default 0
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

        for data in tqdm(df_list):
            in_process = []
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale, extra_freqs = extra_freqs)
            czt_data_shape = np.zeros(freqs.shape, dtype=complex)
            for pair in tqdm(data.pair.unique(), leave=False, disable = True):
                # creates frequency and CZT data arrays per antenna pair
                czt_data = deepcopy(czt_data_shape)
                for f in tqdm(freq, leave=False, disable = True):
                    czt_data = place_czt_value(czt_data = czt_data, f = f, freqs = freqs, df = data, pair = pair, quadrant = quadrant, I=I, Q=Q, signal=signal)
                if conj_sym_partial:
                    freqs_out, czt_data = conjugate_symmetric_partial(freqs, czt_data)
                elif conj_sym:
                    freqs_out, czt_data = conjugate_symmetric(freqs, czt_data)
                else:
                    freqs_out = freqs
                fd_data = pd.concat([pd.DataFrame({'freq': freqs_out}),  pd.DataFrame({'czt': czt_data})], axis=1)
                # fd_data.columns = pd.MultiIndex.from_product([[pair],['freq','czt']])
                fd_data["pair"] = pair
                fd_data["Tx"] = int(data.loc[data.pair.eq(pair), 'Tx'].unique())
                fd_data["Rx"] = int(data.loc[data.pair.eq(pair), 'Rx'].unique())
                if "distances" in data.columns:
                    fd_data["distances"] = data.loc[data.pair.eq(pair), 'distances'].unique()[0]
                in_process.append(fd_data)

            czt_df = pd.concat(in_process, axis=0)

            czt_df["phantom"] = int(data.phantom.unique())
            czt_df["angle"] = int(data.angle.unique())
            czt_df["plug"] = int(data.plug.unique())
            czt_df["date"] = data.date.unique()[0]
            czt_df["rep"] = int(data.rep.unique())
            czt_df["iter"] = int(data.iter.unique())
            czt_df["attLO"] = data.attLO.unique()[0]
            czt_df["attRF"] = data.attRF.unique()[0]
            czt_df["digital_unit"] = data.digital_unit.unique()[0]
            if data.voltage_unit.unique()[0] == 'mV':
                czt_df["voltage_unit"] = 'V'
            else:
                czt_df["voltage_unit"] = data.voltage_unit.unique()[0]
            if "cal_type" in data.columns:
                czt_df["cal_type"] = data.cal_type.unique()[0]

            processed.append(czt_df)
    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

        processed = []

        for data in tqdm(df_list):
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale, extra_freqs = extra_freqs)
            czt_data = np.zeros(freqs.shape, dtype=complex)
            for f in tqdm(freq, leave=False, disable = True):
                czt_data = place_czt_value(czt_data = czt_data, f = f, freqs = freqs, df = data, pair = None, quadrant = quadrant, I=I, Q=Q, signal=signal)
            if conj_sym_partial:
                freqs_out, czt_data = conjugate_symmetric_partial(freqs, czt_data)
            elif conj_sym:
                freqs_out, czt_data = conjugate_symmetric(freqs, czt_data)
            else:
                freqs_out = freqs
            czt_df = pd.DataFrame({'freq': freqs_out, 'czt': czt_data})

            czt_df["cal_type"] = int(data.cal_type.unique())
            czt_df["date"] = data.date.unique()[0]
            czt_df["rep"] = int(data.rep.unique())
            czt_df["iter"] = int(data.iter.unique())
            czt_df["attLO"] = data.attLO.unique()[0]
            czt_df["attRF"] = data.attRF.unique()[0]
            czt_df["digital_unit"] = data.digital_unit.unique()[0]
            if data.voltage_unit.unique()[0] == 'mV':
                czt_df["voltage_unit"] = 'V'
            else:
                czt_df["voltage_unit"] = data.voltage_unit.unique()[0]

            processed.append(czt_df)

    df_out = pd.concat(processed, axis=0, ignore_index=True)
    return df_out

def array_invert_to_time_domain(freqs, czt_data, t = None, df = None):

    if not check_symmetric(czt_data, tol=1e-8):
        freqs, czt_data = conjugate_symmetric(freqs, czt_data)

    if t == 'auto':
        t =  t = auto_time_array(freqs, step_division = 1)

    # N = int(len(czt_data)/2)
    time, sig_t = czt.freq2time(freqs, czt_data, t = t, df = None)
    return time, sig_t

def iczt_spectral_zoom(freqs, czt_data, length = None, fs = None, f1 = None, f2 = None, t = 'auto'):

    if not check_symmetric(czt_data, tol=1e-8):
        freqs, czt_data = conjugate_symmetric(freqs, czt_data)

    if length is None:
        length = len(czt_data)
    if fs is None:
        fs = freqs[-1] - freqs[0] / (len(freqs) - 1)
    if f1 is None:
        f1 = freqs[0]
    if f2 is None:
        f2 = freqs[-1]

    w = np.exp( -2j * np.pi * (f2-f1) / (length * fs) )
    a = np.exp( 2j * np.pi * f1 / fs )

    if t == 'auto':
        t =  t = auto_time_array(freqs, periods = 1, step_division = 1, tstep = 1 / fs)

    # Phase correction
    phase = np.exp(-2j * np.pi * t[0] * freqs)

    N = int(len(czt_data)/2)

    signal = czt.iczt(czt_data, N=length, W = w, A = a, simple=True, t_method="scipy", f_method="numpy")

    signal = np.real_if_close(signal)

    return signal

def df_invert_to_time_domain(df, max_freq = None, freq_step = None, t = 'auto', min_freq = None, conj_sym_partial=False, conj_sym=False, auto_complex_plane = False, 
                                quadrant = 1, I=2, Q=1, signal='voltage', fscale = 1e6, extra_freqs = 0, verbose = False, periods = 1, step_division = 1, tstep = 5e-10):
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
    conj_sym_partial: bool, optional (default False)
        set to True to convert FD signal to conjugate symmetrical (to force real signal) without zero padding, by default False
        when both conj_sym_part and conj_sym are set to False, extracts real part of the TD signal automatically
    conj_sym: bool, optional (default False)
        set to True to convert FD signal to conjugate symmetrical (to force real signal) with zero padding, by default False
        when both conj_sym_part and conj_sym are set to False, extracts real part of the TD signal automatically
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
    extra_freqs: int, optional
        extra frequencies (zeroes) around those present in freqs, by default 0
    verbose: bool, optional
        set to True to print quadrant, I-ch and Q-ch, by default False
    periods : int, optional
        number of periods to use, by default 1
    step_division : int, optional
        integer to divide time step, by default 1
    tstep : None or float, optional
        time step, by default 5e-10

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

        for data in tqdm(df_list):
            in_process = []
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale, extra_freqs = extra_freqs)
            czt_data_shape = np.zeros(freqs.shape, dtype=complex)
            for pair in tqdm(data.pair.unique(), leave=False, disable = True):
                czt_data = deepcopy(czt_data_shape)
                for f in tqdm(freq, leave=False, disable = True):
                    czt_data = place_czt_value(czt_data = czt_data, f = f, pair = pair, freqs = freqs, df = data, quadrant = quadrant, I=I, Q=Q, signal=signal)
                if conj_sym_partial:
                    freqs_out, czt_data = conjugate_symmetric_partial(freqs.to_numpy(), czt_data.to_numpy())
                elif conj_sym:
                    freqs_out, czt_data = conjugate_symmetric(freqs.to_numpy(), czt_data.to_numpy())
                else:
                    freqs_out = freqs
                if t == 'auto':
                    t2 = auto_time_array(freqs_out, periods = periods, step_division = step_division, start = None, tstep = tstep)
                else:
                    t2 = t
                # N = int(len(czt_data)/2)
                time, sig_t = czt.freq2time(freqs_out, czt_data, t = t2)
                if (not conj_sym) and (not conj_sym_partial):
                    sig_t = np.real(sig_t)
                    if freqs[0] >= 0:
                        sig_t = 2*sig_t # magnitude correction for using only positive frequencies
                else:
                    sig_t = np.real_if_close(sig_t, tol = 1e-5) # to favor real signals
                td_data = pd.concat([pd.DataFrame({'sample': np.arange(0,len(sig_t))}), pd.DataFrame({'time': time}),  pd.DataFrame({'signal': sig_t})], axis=1)
                # td_data.columns = pd.MultiIndex.from_product([[pair],['time','signal']])
                td_data["pair"] = pair
                td_data["Tx"] = int(data.loc[data.pair.eq(pair), 'Tx'].unique())
                td_data["Rx"] = int(data.loc[data.pair.eq(pair), 'Rx'].unique())
                if "distances" in data.columns:
                    td_data["distances"] = data.loc[data.pair.eq(p), 'distances'].unique()[0]
                in_process.append(td_data)

            iczt_df = pd.concat(in_process, axis=0)

            iczt_df["phantom"] = int(data.phantom.unique())
            iczt_df["angle"] = int(data.angle.unique())
            iczt_df["plug"] = int(data.plug.unique())
            iczt_df["date"] = data.date.unique()[0]
            iczt_df["rep"] = int(data.rep.unique())
            iczt_df["iter"] = int(data.iter.unique())
            iczt_df["attLO"] = data.attLO.unique()[0]
            iczt_df["attRF"] = data.attRF.unique()[0]
            iczt_df["digital_unit"] = data.digital_unit.unique()[0]
            if data.voltage_unit.unique()[0] == 'mV':
                iczt_df["voltage_unit"] = 'V'
            else:
                iczt_df["voltage_unit"] = data.voltage_unit.unique()[0]
            if "cal_type" in data.columns:
                iczt_df["cal_type"] = data.cal_type.unique()[0]


            processed.append(iczt_df)

    else:
        df_list = dfproc.split_df(df.loc[df.cal_type.ne(1)], groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

        processed = []
        for data in tqdm(df_list):
            data.reset_index(inplace=True)
            freq = data.freq.unique()
            freqs = generate_freq_array(freq, max_freq = max_freq, freq_step = freq_step, min_freq = min_freq, fscale = fscale, extra_freqs = extra_freqs)
            czt_data = np.zeros(freqs.shape, dtype=complex)
            for f in tqdm(freq, leave=False, disable = True):
                czt_data = place_czt_value(czt_data = czt_data, f = f, freqs = freqs, df = data, pair = None, quadrant = quadrant, I=I, Q=Q, signal=signal)
            if conj_sym_partial:
                freqs_out, czt_data = conjugate_symmetric_partial(freqs.to_numpy(), czt_data.to_numpy())
            elif conj_sym:
                freqs_out, czt_data = conjugate_symmetric(freqs.to_numpy(), czt_data.to_numpy())
            else:
                freqs_out = freqs
            if t == 'auto':
                t2 = auto_time_array(freqs_out, periods = periods, step_division = step_division, start = None, tstep = tstep)
            else:
                t2 = t
            # N = int(len(czt_data)/2)
            time, sig_t = czt.freq2time(freqs_out, czt_data, t = t2)
            if (not conj_sym) and (not conj_sym_partial):
                sig_t = np.real(sig_t)
                if freqs[0] >= 0:
                    sig_t = 2*sig_t # magnitude correction for using only positive frequencies
            else:
                sig_t = np.real_if_close(sig_t, tol = 1e-5) # to favor real signals
            iczt_df = pd.DataFrame({'sample': np.arange(0,len(sig_t)), 'time': time, 'signal': sig_t})

            iczt_df["cal_type"] = data.cal_type.unique()[0]
            iczt_df["date"] = data.date.unique()[0]
            iczt_df["rep"] = int(data.rep.unique())
            iczt_df["iter"] = int(data.iter.unique())
            iczt_df["attLO"] = data.attLO.unique()[0]
            iczt_df["attRF"] = data.attRF.unique()[0]
            iczt_df["digital_unit"] = data.digital_unit.unique()[0]
            if data.voltage_unit.unique()[0] == 'mV':
                iczt_df["voltage_unit"] = 'V'
            else:
                iczt_df["voltage_unit"] = data.voltage_unit.unique()

            processed.append(iczt_df)

    df_out = pd.concat(processed, axis=0, ignore_index=True)
    return df_out

def czt_df_invert_to_time_domain(czt_df, t = None, conj_sym_partial = False, conj_sym = False, periods = 1, step_division = 1, tstep = 5e-10):
    """Convert CZT DataFrame into new Dataframe with converted ICZT signals.

    The new DataFrame contains columns for each antenna pair, for which there are two subcolumns: frequencies and converted ICZT signal.

    The ICZT signal is the time domain representation of the scans, converted using the Inverse Chirp-Z Transform algorithm.

    Parameters
    ----------
    czt_df : Pandas df
        DataFrame with reconstructed CZT signals (frequency domain)
    t : np.ndarray
        time for output signal, optional, defaults to standard FFT time sweep
    conj_sym_partial: bool, optional (default False)
        set to True to convert FD signal to conjugate symmetrical (to force real signal) without zero padding, by default False
        when both conj_sym_part and conj_sym are set to False, extracts real part of the TD signal automatically
    conj_sym: bool, optional (default True)
        set to True to convert FD signal to conjugate symmetrical (to force real signal), by default False
        when both conj_sym_part and conj_sym are set to False, extracts real part of the TD signal automatically
    periods : int, optional
        number of periods to use, by default 1
    step_division : int, optional
        integer to divide time step, by default 1
    tstep : None or float, optional
        time step, by default 5e-10

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

        for data in tqdm(df_list):
            data = deepcopy(data)
            data.reset_index(inplace=True)
            in_process = []
            for p in data.pair.unique():
                scan = data.loc[(data.pair.eq(p))]
                if conj_sym_partial:
                    freqs, czt_data = conjugate_symmetric_partial(scan.freq.to_numpy(),scan.czt.to_numpy())
                elif conj_sym:
                    freqs, czt_data = conjugate_symmetric(scan.freq.to_numpy(),scan.czt.to_numpy())
                else:
                    freqs = scan.freq.to_numpy()
                    czt_data = scan.czt.to_numpy()
                if t == 'auto':
                    t2 = auto_time_array(freqs*1e6, periods = periods, step_division = step_division, start = None, tstep = tstep)
                else:
                    t2 = t
                # N = int(len(czt_data)/2)
                time, sig_t = czt.freq2time(freqs, czt_data, t=t2)
                if (not conj_sym) and (not conj_sym_partial):
                    sig_t = np.real(sig_t)
                    if freqs[0] >= 0:
                        sig_t = 2*sig_t # magnitude correction for using only positive frequencies
                else:
                    sig_t = np.real_if_close(sig_t, tol = 1e-5) # to favor real signals
                td_data = pd.concat([pd.DataFrame({'sample': np.arange(0,len(sig_t))}), pd.DataFrame({'time': time}),  pd.DataFrame({'signal': sig_t})], axis=1)
                # td_data.columns = pd.MultiIndex.from_product([[p],['time','signal']])
                td_data["pair"] = p
                td_data["Tx"] = int(data.loc[data.pair.eq(p), 'Tx'].unique())
                td_data["Rx"] = int(data.loc[data.pair.eq(p), 'Rx'].unique())
                if "distances" in data.columns:
                    td_data["distances"] = data.loc[data.pair.eq(p), 'distances'].unique()[0]
                in_process.append(td_data)

            iczt_df = pd.concat(in_process, axis=0, ignore_index=True)

            iczt_df["phantom"] = int(data.phantom.unique())
            iczt_df["angle"] = int(data.angle.unique())
            iczt_df["plug"] = int(data.plug.unique())
            iczt_df["date"] = data.date.unique()[0]
            iczt_df["rep"] = int(data.rep.unique())
            iczt_df["iter"] = int(data.iter.unique())
            iczt_df["attLO"] = data.attLO.unique()[0]
            iczt_df["attRF"] = data.attRF.unique()[0]
            iczt_df["digital_unit"] = data.digital_unit.unique()[0]
            if data.voltage_unit.unique()[0] == 'mV':
                iczt_df["voltage_unit"] = 'V'
            else:
                iczt_df["voltage_unit"] = data.voltage_unit.unique()[0]
            if "cal_type" in data.columns:
                iczt_df["cal_type"] = data.cal_type.unique()[0]

            processed.append(iczt_df)

    else:
        df_list = dfproc.split_df(czt_df, groups=["cal_type", "date", "rep", "iter", "attLO", "attRF"])

        processed = []
        for data in tqdm(df_list):
            if conj_sym_partial:
                freqs, czt_data = conjugate_symmetric_partial(data.freq.to_numpy(),data.czt.to_numpy())
            elif conj_sym:
                freqs, czt_data = conjugate_symmetric(data.freq.to_numpy(),data.czt.to_numpy())
            else:
                freqs = data.freq.to_numpy()
                czt_data = data.czt.to_numpy()
            if t == 'auto':
                    t2 = auto_time_array(freqs, periods = periods, step_division = step_division, start = None, tstep = tstep)
            else:
                t2 = t
            # N = int(len(czt_data)/2)
            time, sig_t = czt.freq2time(freqs, czt_data, t=t2)
            if (not conj_sym) and (not conj_sym_partial):
                sig_t = np.real(sig_t)
                if freqs[0] >= 0:
                    sig_t = 2*sig_t # magnitude correction for using only positive frequencies
            else:
                sig_t = np.real_if_close(sig_t, tol = 1e-5) # to favor real signals
            iczt_df = pd.DataFrame({'sample': np.arange(0,len(sig_t)), 'time': time, 'signal': sig_t})

            iczt_df["cal_type"] = data.cal_type.unique()[0]
            iczt_df["date"] = data.date.unique()[0]
            iczt_df["rep"] = int(data.rep.unique())
            iczt_df["iter"] = int(data.iter.unique())
            iczt_df["attLO"] = data.attLO.unique()[0]
            iczt_df["attRF"] = data.attRF.unique()[0]
            iczt_df["digital_unit"] = data.digital_unit.unique()[0]
            if data.voltage_unit.unique()[0] == 'mV':
                iczt_df["voltage_unit"] = 'V'
            else:
                iczt_df["voltage_unit"] = data.voltage_unit.unique()[0]

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
        tqdm.write("x and y arrays have different lengths!")
        return 0

    if np.min(x) < 0:
        if check_symmetric(y):
            # y is already conjugate symmetric, returns unchanged arrays
            return x, y
        else:
            tqdm.write("x array already has negative values but not conjugate symmetric, please verify.")
            return 0
    elif np.round(np.min(x)) != 0:
        tqdm.write("x array missing zero, padding arrays.")
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

def conjugate_symmetric_partial(x, y, verbose = False):
    """Return conjugate symmetric array for y and correspondent x array with negative values, but only with requested bins and x = 0.

    Conjugate symmetric vectors respect the relation y[x] = y*[-x]. In order for an IFFT signal to be real valued, the FFT signal needs to be conjugate symmetric.

    Output is in FFT-format convention, so that y[0] is the DC component/average, y[1:n/2+2] corresponds to increasing positive values of x
    and y[n/2+1:] corresponds to negative values of x.

    The function first checks if the lengths of x and y are the same, then checks if x only has positive values.
    If these conditions are not met and y isn't already conjugate symmetric, the funtion is interrupted
    and returns 0. If there are negative values for x *and* y is already conjugate symmetric, it returns the unaltered arrays.

    If x is strictly positive and missing zero, both x and y arrays are extended with an extra zero sample.

    Finally, both arrays are extended, x with negative values and y with conjugate symmetric values.

    Parameters
    ----------
    x : array-like
        array to include negative values (time or frequency)
    y : array-like
        array to become conjugate symmetric (signal)
    verbose: bool, optional
        set to True to output "x array missing zero, padding arrays." message when needed, by default False

    Returns
    -------
    x_out : array-like
        array with negative values
    y_out : array-like
        conjugate symmetric array
    """
    if len(x) != len(y):
        tqdm.write("x and y arrays have different lengths!")
        return 0

    if np.min(x) < 0:
        if check_symmetric(y):
            # y is already conjugate symmetric, returns unchanged arrays
            return x, y
        else:
            if verbose:
                tqdm.write("x array already has negative values but not conjugate symmetric, please verify.")
            return x, y
    elif np.round(np.min(x)) != 0:
        if verbose:
            tqdm.write("x array missing zero, padding arrays.")
        x = np.concatenate((np.zeros(1), x))
        # pads y array with zeroes to match the new x array
        y = np.concatenate((np.zeros(1), y))
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
    x_out[0:n] = - reversed_x[:-1]
    x_out[n:] = x
    y_out[0:n+1] = np.conjugate(reversed[:n+1])
    y_out[n:] = y

    # # output arrays in FFT convention
    # x_out[0:n+1] = x
    # x_out[n+1:] = - reversed_x[:-1]
    # y_out[0:n+1] = y
    # y_out[n+1:] = np.conjugate(reversed[:-1])

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

def check_symmetric(a, tol=1e-8):
    """Check if array is conjugate symmetric.

    Note that this doesn't seem to work for FFT-format array.

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
    return np.linalg.norm(a-np.conjugate(a[::-1]), np.Inf) < tol

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
        tqdm.write(f"Median Ch. 1: {est1}, Median Ch. 2: {est2}")
        tqdm.write(f"Quadrant: {quadrant}, I-channel: {I}, Q-channel: {Q}")

    return quadrant, I, Q

def convert_to_DMAS_format(df):
    """Convert Pandas DataFrame of Time-Domain data to numpy array in DMAS algorithm compatible format.

    3-D array has shape (Tx_number , Rx_number , number_of_samples).

    Parameters
    ----------
    df : Pandas df
        input DataFrame with Time-Domain data

    Returns
    -------
    matrix_out: np.array
        3-D numpy array compatible with DMAS algorithm
    """
    # forces first unique phantom scan
    df = df.loc[df.phantom.eq(df.phantom.unique()[0]) & df.angle.eq(df.angle.unique()[0]) & df.date.eq(df.date.unique()[0]) & 
                df.rep.eq(df.rep.unique()[0]) & df.iter.eq(df.iter.unique()[0])]

    # identifies number of samples
    N = df.loc[df.phantom.eq(df.phantom.unique()[0]) & df.angle.eq(df.angle.unique()[0]) & df.date.eq(df.date.unique()[0]) & 
                df.rep.eq(df.rep.unique()[0]) & df.iter.eq(df.iter.unique()[0]) & df.pair.eq(df.pair.unique()[0]), 'signal'].size

    matrix_out = np.zeros((16,16,N), dtype=float)

    for Tx in np.arange(1,17):
        for Rx in np.arange(1,17):
            if f"({Tx},{Rx})" in df.pair.unique():
                matrix_out[Tx-1,Rx-1,:] = df.loc[df.Tx.eq(Tx) & df.Rx.eq(Rx), 'signal'].values
    return matrix_out

def generate_Mat_file_name(df):
    """Generate .mat file name from DataFrame info.

    Parameters
    ----------
    df : Pandas df
        input dataframe

    Returns
    -------
    filename: str
        .mat file name
    """

    file_name = f"Phantom_{df.phantom.unique()[0]}_Ang_{df.angle.unique()[0]}_Date_{df.date.unique()[0]}_Plug_{df.plug.unique()[0]}_Rep_{df.rep.unique()[0]}_Iter_{df.iter.unique()[0]}.mat"

    return file_name

def export_to_DMAS_Matlab(df, main_path="C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Mat/".format(datetime.now().strftime("%Y_%m_%d")), file_name=None):
    """Save .mat file from phantom scan DataFrame.

    Please select specific phantom scan (phantom, angle, date, rep, iter, plug) before exporting.

    Parameters
    ----------
    df : Pandas df
        input pantom scan dataframe
    main_path : str, optional
        file path to save .mat file, by default "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Mat/".format(datetime.now().strftime("%Y_%m_%d"))
    file_name : str, optional
        .mat file name, by default None
        if None, automatically generates file name from dataframe info.
    """
    if file_name is None:
        file_name = generate_Mat_file_name(df)
    file_path = "".join((main_path, file_name))

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    matrix = convert_to_DMAS_format(df)

    savemat(file_path, {'Scan': matrix})
    tqdm.write(f"Matlab file written: {file_path}")
    return

def plot_complex_quadrants(df, pair = None, I = 1, Q = 2, fscale = 1e6, t = 'auto', save_figure_path=None, window_type = 'hann'):

    if pair:
        df = df.loc[df.pair.eq(pair)]

    out = []

    for quad in np.arange(1,5):
        czt = df_invert_to_time_domain(df, max_freq = None, freq_step = None, t = t, min_freq = None, conj_sym=True, auto_complex_plane = False, 
                                quadrant = quad, I=I, Q=Q, signal='voltage', fscale = fscale, verbose = False)
        czt['quadrant'] = quad
        czt = apply_fft_window(czt, window_type = window_type)

        out.append(czt)

    dfout = pd.concat(out)

    dfout['signal'] = np.real(dfout.signal)

    dfout['scan'] = 'Rep ' + dfout.rep.astype(str) + ' Iter ' + dfout.iter.astype(str)

    if 'plug' in dfout.columns:
        p = 'plug'
    else:
        p = None

    # Save a palette to a variable:
    #palette = sns.color_palette("bright")
    palette = sns.color_palette("colorblind")

    # Use palplot and pass in the variable:
    # Set the palette using the name of a palette:
    sns.set_palette(palette)
    sns.set(rc={'figure.figsize':(20,10)}, font_scale=1.5)

    g = sns.relplot(data=dfout, x="time", y="signal", hue='scan', size=None, style= p, row=None, col="quadrant", col_wrap=2,
                row_order=None, col_order=None, palette=None, hue_order=None, hue_norm=None, sizes=None, size_order=None, 
                size_norm=None, markers=None, dashes=None, style_order=None, legend='auto', kind='line', height=5, aspect=1, 
                facet_kws={'sharey': False, 'sharex': True}, units=None)

    if save_figure_path: # checks if empty value (such as 0, '', [], None)
        if not os.path.exists(os.path.dirname(save_figure_path)):
            os.makedirs(os.path.dirname(save_figure_path))
        plt.savefig(save_figure_path, dpi=800)
    else:
        plt.show()
    return

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