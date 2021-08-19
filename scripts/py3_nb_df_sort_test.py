# Standard library imports
from datetime import datetime
import os
import os.path
import sys
#import warnings

# Third-party library imports
import matplotlib.pyplot as plt
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from natsort import natsorted
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sn
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import OneHotEncoder
#from sklearn-pandas import DataFrameMapper
import statsmodels as stm
import statsmodels.api as sm
from tqdm.notebook import trange, tqdm

# Local application imports
#from ..NarrowBand.analysis_pd import df_processing as dfproc
sys.path.insert(1, os.path.abspath('C:/Users/leofo/OneDrive - McGill University/Documents McGill/2019.2/LabTools Scripts/2021_03_05 Adaptations 12/'))
import NarrowBand.analysis_pd.df_antenna_space as dfant
from NarrowBand.analysis_pd import df_processing as dfproc
from NarrowBand.analysis_pd import df_compare as dfcomp
from NarrowBand.analysis_pd.uncategorize import uncategorize

# Main location path of Pandas DataFrame files (.parquet)

main_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/"

# Location path of comparison Pandas DataFrame files (.parquet)

compared_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/2021_04_20/Comps Means/"
compared_path_d = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/2021_04_20/Comps Decl Means/"

# Dates

meas_dates = ["2020_09_14", "2020_09_15", "2020_09_16", "2020_09_21", "2020_09_23"]
#meas_dates = ["2020_09_14"]

# Recursively sweep through sub-folders

is_recursive = False

# File format ("parquet" or "csv")

file_format = "parquet"

# Sub-Folder (decluttered)

sub_folder1 = "Decluttered/"

# Sub-Folder (regular)

sub_folder2 = "Means/"

# Processed path

processed_path = "Processed/DF/"

# Converted path

conv_path = "Conv/"

# Output path for post-processed files. Typically includes sub-folder for current date.

out_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Plots/Box/".format(datetime.now().strftime("%Y_%m_%d"))

# Correction or conversion value, for instance round(1.5e3/8192,4) = 0.1831 mV per ADC unit

correction = round(1.5e3/8192,4)

# Decimals

decimals = 4

# Compare between different attRF:

different_attRF = True

# Compare between different attLO (NOT recommended):

different_attLO = False

# Compare between different phantoms:

between_phantoms = False

# Decluttered using Average Trace Subtraction
#main_paths1 = ["".join((main_path,d,"/",processed_path, sub_folder1)) for d in meas_dates]
df_list1 = dfproc.df_collect(compared_path_d, is_recursive=is_recursive, file_format=file_format, columns=None)
df1 = pd.concat(df_list1)

df1 = df1[df1.phantom_1.ne(0)]

df1 = dfproc.dfsort_pairs_compared(df1, reference_point = "tumor", sort_type = "between_antennas", decimals = 4, out_distances = True, select_ref=1)