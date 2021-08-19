# Python 3.7
# 2020-12-10

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
 Description:
        Module for basic data manipulation of the narrow band system.
        The two main functions are narrow_band_data_read and data_read, the former outputs only the data
        while the latter also outputs number of samples, time/frequency arrays and sampling rate.

Functions::

        narrow_band_data_read : reads narrow band system data file and returns data array.

        data_read : reads narrow band system data file and returns data and time/frequency arrays plus number of samples and sampling rate.

Written by: Leonardo Fortaleza
"""
# Standard library imports
import os

# Third party imports
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from timeit import default_timer as timer

# Local application imports
#from ReceiverFFT import ReceiverFFT as rfft


def narrow_band_data_read(file_name):
    """Read narrow band system data file and return data array.

    The function identifies wether the file is .adc or .fft and extracts the data values, returning it
    in array (or matrix) form.

    Parameters
    ----------
    file_name : str
        file name and path for .adc or .fft file in PScope format

    Returns
    ----------
    data : ndarray of int
        2-D array with ADC output values, rows are samples and columns are separate channel
    """

    extension = os.path.splitext(file_name)[1]

    if extension == '.fft': #for .ffr files
        #start = timer()
        try:
            csv_reader = pd.read_csv(file_name, sep = ',', skiprows = 2, header = None)
        except:
            print("\n C Engine Error in: ", file_name)
            csv_reader = pd.read_csv(file_name, sep = ',', skiprows = 2, header = None, engine = 'python')
        csv_reader.drop(csv_reader.tail(1).index,inplace=True)# droping last row "End", faster than read_csv with skipfooter = 'True'
        data = csv_reader.to_numpy()
        data = np.delete(data, axis = 1, obj = 1)# MUCH faster than using read_csv with sep = ', ,' due to engine = 'python'
        data = data.astype(float)
        #end = timer()
        #print "Duration:" , end-start, " seconds"
        return data

    else: # for .adc files (default)
        #start = timer()
        try:
            csv_reader = pd.read_csv(file_name, sep = ',', skiprows = 6, header = None)
        except:
            print("\n C Engine Error in: ", file_name)
            csv_reader = pd.read_csv(file_name, sep = ',', skiprows = 6, header = None, engine = 'python')
        csv_reader.drop(csv_reader.tail(1).index,inplace=True)# droping last row "End", faster than read_csv with skipfooter = 'True'
        data = csv_reader.to_numpy()
        data = np.delete(data, axis = 1, obj = 1)# MUCH faster than using read_csv with sep = ', ,' due to engine = 'python'
        data = data.astype(float)
        #end = timer()
        #print "Duration:" , end-start, " seconds"
        return data

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
