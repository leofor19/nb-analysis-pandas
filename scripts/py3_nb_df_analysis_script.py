# Standard library imports
from datetime import datetime
import os
import os.path
import sys
#import warnings

# Third-party library imports
import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sn
import statsmodels as stm

# Local application imports
#from ..NarrowBand.analysis_pd import df_processing as dfproc
sys.path.insert(1, os.path.abspath('{}/OneDrive - McGill University/Documents McGill/2019.2/LabTools Scripts/2021_08_12 Adaptations 14/'.format(os.environ['USERPROFILE'])))
from NarrowBand.analysis_pd import df_processing as dfproc

# Main location path of Pandas DataFrame files (.parquet)

main_path = "{}/OneDrive - McGill University/Narrow Band Data1/PScope/".format(os.environ['USERPROFILE'])

# Dates

#meas_dates = ["2020_09_14"]
#meas_dates = ["2019_08_29", "2019_09_05", "2019_09_06", "2019_09_30", "2019_10_30", "2019_11_08", "2019_11_13",
#                 "2020_01_17", "2020_01_20", "2020_01_21", "2020_01_22", "2020_01_23"]

meas_dates = ["2021_08_16"]

#meas_dates = ["2020_09_14", "2020_09_15", "2020_09_16", "2020_09_21", "2020_09_23"]

#meas_dates = ["2020_09_15", "2020_09_16"]
#meas_dates = ["2020_09_21", "2020_09_23"]

#meas_dates = ["2019_11_13"]
#meas_dates = ["2019_08_29", "2019_09_05", "2019_09_06", "2019_09_30", "2019_10_30"]
#meas_dates = ["2020_01_17", "2020_01_20", "2020_01_21", "2020_01_22", "2020_01_23"]
#meas_dates = ["2020_01_23"]

# Recursively sweep through sub-folders

is_recursive = False

# File format ("parquet" or "csv")

file_format = "parquet"

# Sub-Folder

sub_folder = "Conv/"

# Processed path

processed_path = "Processed/DF 03/"

# Converted path

conv_path = "Conv/"

# Output path for post-processed files. Typically includes sub-folder for current date.

#means_path = "{}/OneDrive - McGill University/Narrow Band Data1/Analysis/2020_05_22/DF v4_0/Set Means/2020_05_26 Phantom Set Means.parquet".format(os.environ['USERPROFILE'])
#means_path = "{}/OneDrive - McGill University/Narrow Band Data1/Analysis/2020_05_27/DF v4_1/Set Means/2020_05_27 Phantom Set Means.parquet".format(os.environ['USERPROFILE'])

#agg_path = "{}/OneDrive - McGill University/Narrow Band Data1/Analysis/2020_05_22/DF v4_0/Set Agg Means by Rep/2020_05_26 Phantom Set Agg Means by Rep.parquet".format(os.environ['USERPROFILE'])

#agg_path1 = "{}/OneDrive - McGill University/Narrow Band Data1/Analysis/2020_05_22/DF v4_0/Set Agg Means/2020_05_26 Phantom Set Agg Means.parquet".format(os.environ['USERPROFILE'])

# Correction or conversion value, for instance round(1.5e3/8192,4) = 0.1831 mV per ADC unit

correction = np.around(1.0e3/8192,4)

# Decimals

decimals = 4

#columns = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq","raw_digital_ch1", #"raw_digital_ch2", "digital_ch1", "digital_ch2", "voltage_mag", "voltage_phase", "c_digital_ch1", "c_digital_ch2", #"c_voltage_mag", "c_voltage_phase", "n_digital_ch1", "n_digital_ch2", "n_voltage_mag", "n_voltage_phase"]

#columns = ["meas_number", "meas_order","phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "digital_ch1_mean", "digital_ch2_mean", "voltage_mag", "c_voltage_mag", "n_voltage_mag"]

#columns_agg = ["phantom", "angle", "plug", "date", "rep", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "digital_ch1_mean", "digital_ch2_mean", "voltage_mag", "c_voltage_mag", "n_voltage_mag"]

#columns_agg1 = ["phantom", "angle", "plug", "date", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "digital_ch1_mean", "digital_ch2_mean", "voltage_mag", "c_voltage_mag", "n_voltage_mag"]

#columns_set = ["meas_number", "phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "time", "digital_ch1", "digital_ch2", "voltage_mag", "c_voltage_mag", "n_voltage_mag"]

#-------------------------------------------------------------------------------------------------------------------------------------------------
# USING THE FUNCTIONS FOR DATA PROCESSING

# Process the data, generating Pandas Data Frames with aggregated data "scan data set" files in processed_path sub-folder (only needs to be performed once for new measurement sets)


#dfproc.cal_data_read2pandas(meas_dates, main_path = main_path, cal_path = "Calibration/", processed_path = processed_path, correction = correction, 
#                            conv_path = conv_path, decimals = decimals, save_format=file_format)

for date in meas_dates:
    dfproc.cal_data_pd_compile(date, main_path = main_path, cal_path = "Calibration/", processed_path = processed_path, conv_path = conv_path, 
                            file_format=file_format, is_recursive=is_recursive)

    dfproc.cal_data_pd_agg(date, main_path = main_path, cal_path = "Calibration/", processed_path = processed_path, correction = correction, 
                            conv_path = conv_path, decimals = decimals, save_format=file_format)

#dfproc.case_data_read2pandas(meas_dates, main_path = main_path, sub_folder = "", processed_path = processed_path, correction = correction,
#                             conv_path = conv_path, decimals = decimals, cal_rep = 1, save_format=file_format)

#for date in meas_dates:
#    dfproc.data_pd_agg(date, main_path = main_path, sub_folder = sub_folder, processed_path = processed_path, correction = correction, conv_path = conv_path, 
#                    decimals = decimals, file_format=file_format, is_recursive= is_recursive)