# Python 3.10.2
# 2022-02-25

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
 Description:
        Module for basic data manipulation of the ultrawideband (UWB) system.
        The two main functions are narrow_band_data_read and data_read, the former outputs only the data
        while the latter also outputs number of samples, time/frequency arrays and sampling rate.

Functions::

        uwb_data_read : reads ultrawideband system data file and returns dataframe or numpy array.

        data_read : reads narrow band system data file and returns data and time/frequency arrays plus number of samples and sampling rate.

Written by: Leonardo Fortaleza
"""
# Standard library imports
import os
import os.path
from pathlib import Path
import re
from contextlib import redirect_stderr
import io

# Third party imports
from natsort import natsorted
import numpy as np
import pandas as pd
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook

def uwb_data_read(file_name, output_numpy = False, nafill = 0, keep_default_na = True, on_bad_lines = 'warn'):
    """Read ultrawideband system data file and return data array.

    The function reads a PicoScope output .txt file, returning the data in array (or matrix) form.

    Parameters
    ----------
    file_name : str
        file name and path for .txt file in PicoScope format
    output_numpy : bool, optional
        set to True to output numpy array instead of Pandas DataFrame, by default False
    fillna: None or any, default 0
        set to None to maintain NaNs (if keep_default_na is True), otherwise set to value that replaces NaNs.
    keep_default_na: bool, default True
        Whether or not to include the default NaN values when parsing the data. Depending on whether na_values is passed in, the behavior is as follows:

        If keep_default_na is True, and na_values are specified, na_values is appended to the default NaN values used for parsing.

        If keep_default_na is True, and na_values are not specified, only the default NaN values are used for parsing.

        If keep_default_na is False, and na_values are specified, only the NaN values specified na_values are used for parsing.

        If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.

        Note that if na_filter is passed in as False, the keep_default_na and na_values parameters will be ignored.
    on_bad_lines: {'error', 'warn', 'skip'} or callable, default 'warn'
        Specifies what to do upon encountering a bad line (a line with too many fields). Allowed values are:

        'error', raise an Exception when a bad line is encountered.

        'warn', raise a warning when a bad line is encountered and skip that line.

        'skip', skip bad lines without raising or warning when they are encountered.

    Returns
    ----------
    data : Pandas df or ndarray
        Pnadas DataFrame or array with PicoScope output values, rows are samples and columns are separate channel
    """

    f = io.StringIO()
    with redirect_stderr(f):
        data = pd.read_csv(file_name, delim_whitespace=True, header=None, names = ['raw_signal_ch1', 'raw_signal_ch2'],
                        keep_default_na = keep_default_na, on_bad_lines = on_bad_lines)
    if f.getvalue(): # checking warnings/errors for more information
        tqdm.write(f"Parsing errors on {Path(file_name).stem}: {f.getvalue()}")

    if nafill is not None:
        data.fillna(nafill, inplace=True) # Replaces NaN with fillna (default 0)

    if output_numpy:
        data = data.to_numpy()

    return data

def uwb_scan_folder_sweep(main_path):
    """Find all phantom measurement .txt files in a given directory path.

    In addition to a list with the string pathnames of each file found, the ndarray meta_index and list meta_freq are generated.
    The ndarray meta_index associates parameters of Phantom, Angle, Plug, Tx, Rx, Iteration and Repetition to each file while the list meta_freq
    does the same for the Frequency.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to .adc measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    ph_path : str, optional
        wildcard phantom folder path, by default "Phantom*/"

    Returns
    ----------
    list of str
        list with full pathnames to each .adc file found
    meta_index : ndarray of int
        2-D list of parameters associated to each .adc file found
        Parameters are Phantom, Angle, Plug, Tx, Rx, Iteration and Repetition
    meta_freq : list of str
        list of string formatted frequencies associated to each .adc file found
    """

    a = natsorted(Path(main_path).glob("*.txt"))
    path_list = [str(x) for x in a]

    ## meta_index is matrix with rows in the form [Antenna Pair Tx, Antenna Pair Rx]

    meta_index = [[int(re.search(r'^.*?_[Aa](\d+)',str(i)).group(0)),
                int(re.search(r'^.*?_[Aa][\d+]_[Aa](\d+)', str(i)).group(0))]
                for i in path_list]
    meta_index = np.array(meta_index)

    return path_list, meta_index

def data_read(file_name):
    """Read narrow band system data file and return data and time/frequency arrays plus number of samples and sampling rate integers.

    The function identifies wether the file is .adc or .fft and extracts the data and time/freq values for each sample, returning it
    in array (or matrix) form. The number of samples and sampling rate are also returned as integers.

    Parameters
    ----------
    file_name : str
       file name and path for .adc or .fft file in PScope format

    Returns
    ----------
    data : ndarray of int
        2-D array with ADC output values, rows are samples and columns are separate channel
    time: ndarray of float
        2-D array with time values for each ADC sample, rows are samples and columns are separate channel
        output for .adc input files only
    freq : ndarray of float
        2-D array with time values for each ADC sample, rows are samples and columns are separate channel
        output for .fft input files only
    nsamples : int
        number of samples (per channel)
    srate : float
        sampling rate in Msps
    """

    extension = os.path.splitext(file_name)[1]

    if extension == '.fft': #for .ffr files
        #start = timer()
        try:
            csv_reader1 = pd.read_csv(file_name, sep ='\s+', skiprows = 0, nrows = 1, header = None, usecols = [2,3])
        except:
            print("\n C Engine Error in: ", file_name)
            csv_reader1 = pd.read_csv(file_name, sep ='\s+', skiprows = 0, nrows = 1, header = None, usecols = [2,3], engine = 'python')
        nsamples, srate = csv_reader1.to_numpy()[0]
        csv_reader = pd.read_csv(file_name, sep = '\s+', skiprows = 0, header = None)
        csv_reader.drop(csv_reader.tail(1).index,inplace=True)# droping last row "End", faster than read_csv with skipfooter = 'True'
        data = csv_reader.to_numpy()
        data = np.delete(data, axis = 1, obj = 1)# MUCH faster than using read_csv with sep = ', ,' due to engine = 'python'
        data = data.astype(float)
        freq = np.linspace(0,(srate*1e6)/2,len(data[:,1]))
        #end = timer()
        #print "Duration:" , end-start, " seconds"
        return data, freq, nsamples, srate

    else: # for .adc files (default)
        #start = timer()
        try:
            csv_reader1 = pd.read_csv(file_name, sep =',', skiprows = 4, nrows = 1, header = None, usecols = [2,6])
        except:
            print("\n C Engine Error in: ", file_name)
            csv_reader1 = pd.read_csv(file_name, sep =',', skiprows = 4, nrows = 1, header = None, usecols = [2,6], engine = 'python')
        nsamples, srate = csv_reader1.to_numpy()[0]
        csv_reader = pd.read_csv(file_name, sep = ',', skiprows = 6, header = None)
        csv_reader.drop(csv_reader.tail(1).index,inplace=True)# droping last row "End", faster than read_csv with skipfooter = 'True'
        data = csv_reader.to_numpy()
        data = np.delete(data, axis = 1, obj = 1)# MUCH faster than using read_csv with sep = ', ,' due to engine = 'python'
        data = data.astype(float)
        time = np.linspace(0,len(data)/(srate*1e6),len(data[:,1]))
        #end = timer()
        #print "Duration:" , end-start, " seconds"
        return data, time, nsamples, srate
