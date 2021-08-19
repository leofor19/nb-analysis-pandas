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
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sn
import statsmodels as stm

# Local application imports
#from ..NarrowBand.analysis_pd import df_processing as dfproc
sys.path.insert(1, os.path.abspath('C:/Users/leofo/OneDrive - McGill University/Documents McGill/2019.2/LabTools Scripts/2021_03_05 Adaptations 12/'))
from NarrowBand.analysis_pd import df_processing as dfproc
from NarrowBand.analysis_pd import df_compare as dfcomp
from NarrowBand.analysis_pd.uncategorize import uncategorize

# Main location path of Pandas DataFrame files (.parquet)

main_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/"

# Dates

meas_dates = ["2020_09_14", "2020_09_15", "2020_09_16", "2020_09_21", "2020_09_23"]

# Recursively sweep through sub-folders

is_recursive = False

# File format ("parquet" or "csv")

file_format = "parquet"

# Sub-Folder

sub_folder = "Means/"

# Processed path

processed_path = "Processed/DF/"

# Converted path

conv_path = "Conv/"

# Output path for post-processed files. Typically includes sub-folder for current date.

out_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Comps/".format(datetime.now().strftime("%Y_%m_%d"))

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

#-------------------------------------------------------------------------------------------------------------------------------------------------
# USING THE FUNCTIONS FOR DATA PROCESSING

# Process the data, generating Pandas Data Frames with aggregated data "scan data set" files in processed_path sub-folder (only needs to be performed once for new measurement sets)
#

if __name__ == "__main__":

    # Dask Distributed initialization, displaying link for dashboard monitoring
    client = Client()
    print("Dask Dashboard:  http://localhost:{}/status \n".format(client.scheduler_info()['services']['dashboard']))

    pathlist = ["".join((main_path,d,"/",processed_path,sub_folder)) for d in meas_dates]

    ddf_list = dfproc.dd_collect(pathlist, is_recursive=is_recursive, file_format=file_format, columns=None)
    ddf = dd.concat(ddf_list)
    ddf = ddf.apply(lambda x: uncategorize(x), axis=1, meta=ddf)
    df = ddf.compute()
    if "index" in df.columns:
        df.drop("index", axis=1, inplace=True)

    dfcomp.save_pairwise_comparisons(df, out_path = out_path, different_attRF = different_attRF, different_attLO = different_attLO, between_phantoms = between_phantoms,
                                        correction = correction, save_format = file_format)

    client.shutdown()