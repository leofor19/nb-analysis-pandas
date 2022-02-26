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

# Third party imports
from natsort import natsorted
import numpy as np
import pandas as pd

def uwb_data_read(file_name, output_numpy = False):
    """Read ultrawideband system data file and return data array.

    The function reads a PicoScope output .txt file, returning the data in array (or matrix) form.

    Parameters
    ----------
    file_name : str
        file name and path for .txt file in PicoScope format
    output_numpy : bool, optional
        set to True to output numpy array instead of Pandas DataFrame, by default False

    Returns
    ----------
    data : Pandas df or ndarray
        Pnadas DataFrame or array with PicoScope output values, rows are samples and columns are separate channel
    """

    data = pd.read_csv(file_name, sep=" ", header=None)
    data.columns = ['signal_ch1', 'signal_ch2']

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
            csv_reader1 = pd.read_csv(file_name, sep =',', skiprows = 1, nrows = 1, header = None, usecols = [2,3])
        except:
            print("\n C Engine Error in: ", file_name)
            csv_reader1 = pd.read_csv(file_name, sep =',', skiprows = 1, nrows = 1, header = None, usecols = [2,3], engine = 'python')
        nsamples, srate = csv_reader1.to_numpy()[0]
        csv_reader = pd.read_csv(file_name, sep = ',', skiprows = 2, header = None)
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
