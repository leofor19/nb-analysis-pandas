# Python 3.9

# uses nb38 environment

# Using DF 05 data sets

# 2022/07/19

"""Script for performing Average Trace Subtraction decluttering on phantom data scan DataFrames in the Time Domain (TD), after
performing time-domain signal alignment (via cross-correlation) and normalizion.

This version performs a routine through multiple parquet files in the specified dates/folder.

"""
# %%
# Standard library imports
from datetime import datetime
import os
import os.path
import sys

# Third-party library imports
from natsort import natsorted, natsort_keygen
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

# Local application imports
#from ..NarrowBand.analysis_pd import df_processing as dfproc
# sys.path.insert(1, os.path.abspath('C:/Users/leofo/OneDrive - McGill University/Documents McGill/2019.2/LabTools Scripts/2021_12_03 Adaptations 15/'))
sys.path.insert(1, os.path.abspath('C:/Users/leofo/Documents/Github/nb-analysis-pandas/'))
import NarrowBand.align_signals.df_align_signals as dfal
from NarrowBand.analysis_pd import df_processing as dfproc
from NarrowBand.analysis_pd import df_compare as dfcomp
from NarrowBand.analysis_pd import df_declutter as dfdec
from NarrowBand.analysis_pd.safe_arange import safe_arange
from NarrowBand.nb_czt import czt as czt

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %%
## USER INPUT SECTION

## Dates for file selection:

#meas_dates = ["2020_09_14"]
#meas_dates = ["2019_08_29", "2019_09_05", "2019_09_06", "2019_09_30", "2019_10_30", "2019_11_08", "2019_11_13",
#                 "2020_01_17", "2020_01_20", "2020_01_21", "2020_01_22", "2020_01_23"]

# meas_dates = ["2021_08_16"]

#meas_dates = ["2020_09_14", "2020_09_15", "2020_09_16", "2020_09_21", "2020_09_23"]

#meas_dates = ["2020_09_15", "2020_09_16"]
#meas_dates = ["2020_09_21", "2020_09_23"]

#meas_dates = ["2019_11_13"]
#meas_dates = ["2019_08_29", "2019_09_05", "2019_09_06", "2019_09_30", "2019_10_30"]
#meas_dates = ["2020_01_17", "2020_01_20", "2020_01_21", "2020_01_22", "2020_01_23"]
#meas_dates = ["2020_01_23"]

meas_dates = ["2019_08_29", "2019_09_05", "2019_09_06", "2019_09_30", "2019_10_30", "2019_11_08", "2019_11_13",
                "2020_01_17", "2020_01_20", "2020_01_21", "2020_01_22", "2020_01_23",
                "2020_09_14", "2020_09_15", "2020_09_16", "2020_09_21", "2020_09_23",
                "2021_08_16", "2021_08_18", "2021_08_16", "2021_08_18"]

# Main location path of Pandas DataFrame files (.parquet)

main_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/"

# Output path for post-processed files. Typically includes sub-folder for current date.

out_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/TD Decluttered/".format(datetime.now().strftime("%Y_%m_%d"))

## FFT Window Type ('hann', 'BlackmanHarris92', None, etc):
window_type = 'hann'

## Reference Pulse Parameters (in Hz):

fstep = 12.5e6 # frequency step in Hz (also used for CZT, should match scan data)
fmin = 2.0e9 # minimum frequency in Hz
fmax = 2.1e9 # minimum frequency in Hz
ref_mag = 1 # absolute magnitude of each frequency component

fscale = 1e6 # for CZT functions

## Target Time Array for Time Domain Signal:
#       A good choice is: safe_arange(0, 40e-9 + 5e-10, step = 5e-10)

target_time = safe_arange(0, 40e-9 + 5e-10, step = 5e-10)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %%
for date in tqdm(meas_dates):

    data_path = "".join((main_path, f"{date}/Processed/DF 05/TD/{date} Phantom Set Means CZT TD.parquet"))
    # %%
    # # Phantom Scan Signal Alignment (DataFrame)

    df = pd.read_parquet(data_path, engine='pyarrow')

    df = dfproc.dfsort_pairs(df, sort_type = 'between_antennas', out_distances = True, out_as_list = False)

    # ref_df = pd.read_parquet(ref_path, engine='pyarrow')
    # ref_df = ref_df.loc[ref_df.rep.eq(1) & ref_df.iter.eq(1)]
    # dfa = dfal.df_align_signals(df, ref_df, column = ['signal' ,'magnitude'], sort_col = 'sample', max_delay = 10, truncate = True, align_power = False)
    # # dfa2 = dfal.df_align_signals(df, ref_df, column = 'power', sort_col = 'sample', max_delay = 10, truncate = True, align_power = False)
    # # dfa.loc[:,'power'] = dfa2.loc[:,'power']

    dfa = dfal.df_align_signals_same_distance(df,  column = ['signal' ,'magnitude'], sort_col = 'sample', max_delay = 10, truncate = True, align_power = True)

    # del dfout

    dfa = dfproc.dfsort_pairs(dfa, sort_type = 'between_antennas', out_distances = True, out_as_list= False)

    # %%
    # # Clutter

    clutter = dfdec.avg_trace_clutter(dfa, progress_bar = True, center='mean')

    # %%
    # # Average Trace Decluttering

    # decluttered = dfdec.subtract_clutter(dfa, clutter, column=['magnitude', 'power', 'signal'])
    decluttered = dfdec.subtract_clutter(dfa, clutter, column=['signal'])

    # decluttered.loc[:, 'mag2dB'] = 10*np.log10(decluttered.loc[:, 'magnitude']**2)
    # dfa.loc[:, 'mag2dB'] = 10*np.log10(dfa.loc[:, 'magnitude']**2)

    # decluttered.loc[:, 'power_dB'] = 10*np.log10(decluttered.loc[:, 'signal']**2)
    # dfa.loc[:, 'power_dB'] = 10*np.log10(dfa.loc[:, 'signal']**2)

    decluttered.loc[:, 'sig2power'] = decluttered.loc[:, 'signal']**2
    dfa.loc[:, 'sig2power'] = dfa.loc[:, 'signal']**2

    decluttered.loc[:, 'sig2dB'] = 10*np.log10(decluttered.loc[:, 'signal']**2)
    dfa.loc[:, 'sig2dB'] = 10*np.log10(dfa.loc[:, 'signal']**2)

    # %%
    # # Save files

    out_path_data = "".join((main_path, f'{date}/Processed/DF 05/TD Decluttered/{date} Phantom Set Means TD Avg Trace Decluttered.parquet'))
    out_path_clutter = "".join((main_path, f'{date}/Processed/DF 05/TD Decluttered/{date} Phantom Set Means TD Avg Trace Clutter.parquet'))
    if not os.path.exists(os.path.dirname(out_path_data)):
        os.makedirs(os.path.dirname(out_path_data))
    decluttered.reset_index(drop=True).to_parquet(out_path_data, engine='pyarrow')
    clutter.reset_index(drop=True).to_parquet(out_path_clutter, engine='pyarrow')
    tqdm.write(f"\nSaved file: {out_path_data}        ")
