# Python 3.9

# uses nb38 environment

# Using DF 05 data sets

# 2022/04/28

"""Script for converting narrowband (NB) data scans (from Pandas DataFrames) from Frequency Domain (FD) to Time Domain (TD) via Chirp-Z Transform (CZT),
as well as perform time-domain signal alignment (via cross-correlation) and normalizion.

This version operates on specific parquet files.

"""
# %%
# Standard library imports
from datetime import datetime
import os
import os.path
import sys

# Third-party library imports
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

# Local application imports
sys.path.insert(1, os.path.abspath('C:/Users/leofo/Documents/Github/nb-analysis-pandas/'))
import NarrowBand.align_signals.df_align_signals as dfal
from NarrowBand.analysis_pd import df_processing as dfproc
from NarrowBand.analysis_pd.safe_arange import safe_arange
from NarrowBand.nb_czt import nb_czt as nbczt
from NarrowBand.nb_czt.fft_window import fft_window
from NarrowBand.nb_czt import czt as czt

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %%
## USER INPUT SECTION

## Date for file selection:

date = "2020_09_18"

## Main location path of Pandas DataFrame files (.parquet)

# main_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/"
data_path = f"C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/{date}/Processed/DF 05/Means/{date} Phantom Set Means.parquet"

## Main location path of Pandas DataFrame files (.parquet) for Calibration Type 3 (Tx bypassed)

cal_path3 = f"C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/{date}/Processed/DF 05/Calibration/Means/{date} Calibration Processed Means Type 3.parquet"

## Frequency subset for calibration pulses (can be set to None for full set):

# freq_subset = None
freq_subset = [2012.5,2025,2037.5,2050,2062.5,2075,2087.5,2100]

## Specific Subsets of Calibration Selections (set None to use all available)

# cal_reps = None
# cal_iters = None
cal_reps = [1]
cal_iters = [1]

## Specific Subsets of Phantom Scan Selections (set None to use all available)

# phs = None
# ph_plugs = None
# ph_reps = None
# ph_iters = None
# ph_ant_pairs = None
phs = [2]
ph_plugs = [2, 38]
ph_reps = [1,2]
ph_iters = [1]
ph_ant_pairs = ['(7,13)', '(2,5)', '(5,2)', '(1,6)', '(6,1)', '(3,8)', '(8,3)']
# ph_ant_pairs = ['(12,14)', '(14,12)', '(4,15)', '(15,4)', '(4,11)', '(11,4)', '(14,16)', '(16,14)', '(10,12)', '(12,10)', '(9,15)', '(15,9)', '(9,11)', '(11,9)', '(10,16)', '(16,10)']

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

# Output path for post-processed files. Typically includes sub-folder for current date.

out_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/TD/".format(datetime.now().strftime("%Y_%m_%d"))

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# %%
## Reference Ideal Pulse

# note that it uses start = fmin - fstep and stop = fmax + 2*fstep (effectively ending in fmax + fstep) in case fft window zeroes edges
freq = np.arange(fmin - fstep, fmax + 2*fstep, step = 12.5e6)
spectrum = np.where( np.abs(freq) >= fmin, ref_mag, 0)
win = fft_window(len(spectrum), window_type = window_type)
spectrum = win * spectrum
time, sig = czt.freq2time(freq, spectrum, t = target_time)


ideal = pd.DataFrame({'time': time, 'signal': 2*np.real(sig)})
ideal['signal'] = np.real_if_close(ideal['signal'].to_numpy())

# %%
## Reference Calibration Type 3 Pulse

cal_df3 = pd.read_parquet(cal_path3, engine='pyarrow')

# Subsets of the data
if freq_subset is not None:
    cal_df3 = cal_df3.loc[(cal_df3.freq.isin(freq_subset)) & cal_df3.rep.eq(1) & cal_df3.iter.eq(1)]
if cal_reps is not None:
    cal_df3 = cal_df3.loc[cal_df3.rep.isin(cal_reps)]
if cal_iters is not None:
    cal_df3 = cal_df3.loc[cal_df3.iter.isin(cal_iters)]

df3 = nbczt.df_to_freq_domain(cal_df3, max_freq= cal_df3.freq.max() + 2*fstep/fscale, freq_step= fstep/fscale, min_freq= cal_df3.freq.min() - fstep/fscale, conj_sym=False, auto_complex_plane = False, 
                                quadrant = 1, I=2, Q=1, signal='voltage', fscale = fscale, verbose = False)

df3 = nbczt.apply_fft_window(df3, window_type = window_type, column = ['czt'])

dfout31 = nbczt.czt_df_invert_to_time_domain(df3, t= target_time, conj_sym=False)

dfout31 = dfal.df_align_signals(dfout31, ideal, column = 'signal', sort_col = 'time', max_delay = None, truncate = True, align_power = True)

# dfout31, _ = dfal.direct_df_align_signals(dfout31.loc[dfout31.rep.eq(1) & dfout31.iter.eq(1)], ideal, column = 'signal', sort_col = 'time', max_delay = None, truncate = True, fixed_df2 = True)

dfout31 = nbczt.normalize_pulses(dfout31, column = 'signal', use_sd= False)

dfout3 = dfal.df_trim2starting_zero(dfout31, ycolumn = 'signal', xcolumn = ['sample', 'time'], pad_end=True)

dfout3['magnitude'] = dfout3['signal'].abs()
dfout3['power'] = dfout3['signal']**2

if not os.path.exists(os.path.dirname(out_path)):
    os.makedirs(os.path.dirname(out_path))
out_path_cal = "".join((out_path, f'Calibration/{date} Calibration Means Type 3 CZT TD.parquet'))
dfout3.reset_index().to_parquet(out_path_cal, engine='pyarrow')
tqdm.write(f"\nSaved file: {out_path_cal}        ")

# %%
## Phantom Scan Data

df = pd.read_parquet(data_path, engine='pyarrow')

# Subsets of the data
if phs is not None:
    df = df.loc[df.phantom.isin(phs)]
if ph_plugs is not None:
    df = df.loc[df.plug.isin(ph_plugs)]
if ph_reps is not None:
    df = df.loc[df.rep.isin(ph_reps)]
if ph_iters is not None:
    df = df.loc[df.iter.isin(ph_iters)]
if ph_ant_pairs is not None:
    df = df.loc[df.pair.isin(ph_ant_pairs)]

df = dfproc.dfsort_pairs(df, sort_type = 'between_antennas', out_distances = True, out_as_list = False)

df1 = nbczt.df_to_freq_domain(df, max_freq= df.freq.max() + 2*fstep/fscale, freq_step= fstep/fscale, min_freq= df.freq.min() - fstep/fscale, conj_sym=False, auto_complex_plane = False,
                                 quadrant = 1, I=2, Q=1, signal='voltage', fscale = fscale, verbose = False)

df1 = nbczt.apply_fft_window(df1, window_type = window_type, column = ['czt'])

dfout1 = nbczt.czt_df_invert_to_time_domain(df1, t= target_time, conj_sym=False)

dfout1 = nbczt.normalize_pulses(dfout1, column = 'signal')
# dfout1 = nbczt.normalize_pulses(dfout1, column = 'signal', use_sd = True, ddof=0)

dfout = dfal.df_trim2starting_zero(dfout1, ycolumn = 'signal', xcolumn = ['sample', 'time'], pad_end = False, pad_zeroes = True)

dfout['magnitude'] = dfout['signal'].abs()
dfout['power'] = dfout['signal']**2

dfout1['magnitude'] = dfout1['signal'].abs()
dfout1['power'] = dfout1['signal']**2

# del df, df1

if not os.path.exists(os.path.dirname(out_path)):
    os.makedirs(os.path.dirname(out_path))
out_path_data = "".join((out_path, f'{date} Phantom Set Means CZT TD.parquet'))
dfout1.reset_index(drop=True).to_parquet(out_path_data, engine='pyarrow')
tqdm.write(f"\nSaved file: {out_path_data}        ")