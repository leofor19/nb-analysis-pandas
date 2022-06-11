# Python 3.9

# uses nb38 environment

# Using DF 04 data sets

# 2022/04/29 UNFINISHED

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
from tqdm import tqdm

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

## Date for file selection:

date = "2020_09_18"

## Main location path of Pandas DataFrame files (.parquet)

# main_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/"
data_path = f"C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/PScope/{date}/Processed/DF 04/TD/{date} Phantom Set Means CZT TD.parquet"

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

out_path_data = "".join((out_path, f'{date} Phantom Set Means TD Avg Trace Decluttered.parquet'))
out_path_clutter = "".join((out_path, f'{date} Phantom Set Means TD Avg Trace Clutter.parquet'))
if not os.path.exists(os.path.dirname(out_path_data)):
    os.makedirs(os.path.dirname(out_path_data))
decluttered.reset_index().to_parquet(out_path_data, engine='pyarrow')
clutter.reset_index().to_parquet(out_path_clutter, engine='pyarrow')
tqdm.write(f"\nSaved file: {out_path_data}        ")

# %%


# %%
da1 = dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(1)]
da2 = dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(1)]
da3 = dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(2)]
da4 = dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(2)]

# %%
dec1 = decluttered.loc[decluttered.plug.eq(2) & decluttered.rep.eq(1)]
dec2 = decluttered.loc[decluttered.plug.eq(38) & decluttered.rep.eq(1)]
dec3 = decluttered.loc[decluttered.plug.eq(2) & decluttered.rep.eq(2)]
dec4 = decluttered.loc[decluttered.plug.eq(38) & decluttered.rep.eq(2)]

# %%
diff21 = dfcomp.specific_comparison(da2, da1, comp_columns='signal')
for col in diff21:
    if "_1" in col:
        diff21[col.rstrip("_1")] = diff21[col]
        diff21.drop(col, axis = 1, inplace = True)
    elif "_2" in col:
        diff21.drop(col, axis = 1, inplace = True)

diff23 = dfcomp.specific_comparison(da2, da3, comp_columns='signal')
for col in diff23:
    if "_1" in col:
        diff23[col.rstrip("_1")] = diff23[col]
        diff23.drop(col, axis = 1, inplace = True)
    elif "_2" in col:
        diff23.drop(col, axis = 1, inplace = True)

diff13 = dfcomp.specific_comparison(da1, da3, comp_columns='signal')
for col in diff13:
    if "_1" in col:
        diff13[col.rstrip("_1")] = diff13[col]
        diff13.drop(col, axis = 1, inplace = True)
    elif "_2" in col:
        diff13.drop(col, axis = 1, inplace = True)

diff42 = dfcomp.specific_comparison(da4, da2, comp_columns='signal')
for col in diff42:
    if "_1" in col:
        diff42[col.rstrip("_1")] = diff42[col]
        diff42.drop(col, axis = 1, inplace = True)
    elif "_2" in col:
        diff42.drop(col, axis = 1, inplace = True)

diff43 = dfcomp.specific_comparison(da4, da3, comp_columns='signal')
for col in diff43:
    if "_1" in col:
        diff43[col.rstrip("_1")] = diff43[col]
        diff43.drop(col, axis = 1, inplace = True)
    elif "_2" in col:
        diff43.drop(col, axis = 1, inplace = True)

# %%


# %%
decsum1 = dec1.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum2 = dec2.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum3 = dec3.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum4 = dec4.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

dasum1 = da1.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum2 = da2.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum3 = da3.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum4 = da4.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

# %%
diff21['sig2power'] = diff21.signal_diff**2
diff23['sig2power'] = diff23.signal_diff**2
diff13['sig2power'] = diff13.signal_diff**2
diff42['sig2power'] = diff42.signal_diff**2
diff43['sig2power'] = diff43.signal_diff**2

diffsum21 = diff21.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
diffsum23 = diff23.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
diffsum13 = diff13.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
diffsum42 = diff42.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
diffsum43 = diff43.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

diffsum21['power_dB'] = 10*np.log10(diffsum21['sig2power'].values)
diffsum23['power_dB'] = 10*np.log10(diffsum23['sig2power'].values)
diffsum13['power_dB'] = 10*np.log10(diffsum13['sig2power'].values)
diffsum42['power_dB'] = 10*np.log10(diffsum42['sig2power'].values)
diffsum43['power_dB'] = 10*np.log10(diffsum43['sig2power'].values)

diffsum21.sort_values('pair', inplace = True, key= natsort_keygen())
diffsum23.sort_values('pair', inplace = True, key= natsort_keygen())
diffsum13.sort_values('pair', inplace = True, key= natsort_keygen())
diffsum42.sort_values('pair', inplace = True, key= natsort_keygen())
diffsum43.sort_values('pair', inplace = True, key= natsort_keygen())


# %%
decsum1['power_dB'] = 10*np.log10(decsum1['sig2power'].values)
decsum2['power_dB'] = 10*np.log10(decsum2['sig2power'].values)
decsum3['power_dB'] = 10*np.log10(decsum3['sig2power'].values)
decsum4['power_dB'] = 10*np.log10(decsum4['sig2power'].values)


dasum1['power_dB'] = 10*np.log10(dasum1['sig2power'].values)
dasum2['power_dB'] = 10*np.log10(dasum2['sig2power'].values)
dasum3['power_dB'] = 10*np.log10(dasum3['sig2power'].values)
dasum4['power_dB'] = 10*np.log10(dasum4['sig2power'].values)

# %%
from natsort import natsort_keygen

decsum1.sort_values('pair', inplace = True, key= natsort_keygen())
decsum2.sort_values('pair', inplace = True, key= natsort_keygen())
decsum3.sort_values('pair', inplace = True, key= natsort_keygen())
decsum4.sort_values('pair', inplace = True, key= natsort_keygen())

dasum1.sort_values('pair', inplace = True, key= natsort_keygen())
dasum2.sort_values('pair', inplace = True, key= natsort_keygen())
dasum3.sort_values('pair', inplace = True, key= natsort_keygen())
dasum4.sort_values('pair', inplace = True, key= natsort_keygen())

ats1 = decsum1.loc[:,['phantom', 'angle', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']]
diff1 = dasum1.loc[:,['phantom', 'angle', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']]

for col in ['sig2power', 'power_dB']:
    ats1[col] = decsum2[col].values - decsum1[col].values
    diff1[col] = diffsum21[col].values - decsum1[col].values

ats1.rename({'power_dB' : 'SCR'}, axis = 1, inplace = True)
diff1.rename({'power_dB' : 'SCR'}, axis = 1, inplace = True)

# %%
ats1

# %%
diff1

# %%
ats1.loc[ats1.SCR.gt(0), 'SCR'].count()

# %%
ats1.loc[ats1.SCR.lt(0), 'SCR'].count()

# %%
diff1.loc[diff1.SCR.gt(0), 'SCR'].count()

# %%
diff1.loc[diff1.SCR.lt(0), 'SCR'].count()

# %%
ats1.SCR.mean()

# %%
ats1.SCR.median()

# %%
ats1.SCR.max()

# %%
ats1.SCR.min()

# %%
diff1.SCR.mean()

# %%
diff1.SCR.median()

# %%
diff1.SCR.max()

# %%
diff1.SCR.min()

# %%
print(dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(1) & dfa.pair.eq("(1,6)"), 'abs_max'].max())
print(dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(1) & dfa.pair.eq("(1,6)"), 'abs_max'].max())

# %%
print(dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(1) & dfa.distance.eq(0.1306), 'abs_max'].unique())
print(dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(1) & dfa.distance.eq(0.1306), 'abs_max'].unique())

# %%


# %%
da11 = dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(1) & dfa.time.between(5e-9,30e-9)]
da22 = dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(1) & dfa.time.between(5e-9,30e-9)]
da33 = dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(2) & dfa.time.between(5e-9,30e-9)]
da44 = dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(2) & dfa.time.between(5e-9,30e-9)]

dec11 = decluttered.loc[decluttered.plug.eq(2) & decluttered.rep.eq(1) & decluttered.time.between(5e-9,30e-9)]
dec22 = decluttered.loc[decluttered.plug.eq(38) & decluttered.rep.eq(1) & decluttered.time.between(5e-9,30e-9)]
dec33 = decluttered.loc[decluttered.plug.eq(2) & decluttered.rep.eq(2) & decluttered.time.between(5e-9,30e-9)]
dec44 = decluttered.loc[decluttered.plug.eq(38) & decluttered.rep.eq(2) & decluttered.time.between(5e-9,30e-9)]

# %%
decsum11 = dec11.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum22 = dec22.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum33 = dec33.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum44 = dec44.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

dasum11 = da11.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum22 = da22.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum33 = da33.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum44 = da44.groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

# %%
decsum11['power_dB'] = 10*np.log10(decsum11['sig2power'].values)
decsum22['power_dB'] = 10*np.log10(decsum22['sig2power'].values)
decsum33['power_dB'] = 10*np.log10(decsum33['sig2power'].values)
decsum44['power_dB'] = 10*np.log10(decsum44['sig2power'].values)


dasum11['power_dB'] = 10*np.log10(dasum11['sig2power'].values)
dasum22['power_dB'] = 10*np.log10(dasum22['sig2power'].values)
dasum33['power_dB'] = 10*np.log10(dasum33['sig2power'].values)
dasum44['power_dB'] = 10*np.log10(dasum44['sig2power'].values)

# %%
from natsort import natsort_keygen

decsum11.sort_values('pair', inplace = True, key= natsort_keygen())
decsum22.sort_values('pair', inplace = True, key= natsort_keygen())
decsum33.sort_values('pair', inplace = True, key= natsort_keygen())
decsum44.sort_values('pair', inplace = True, key= natsort_keygen())

dasum11.sort_values('pair', inplace = True, key= natsort_keygen())
dasum22.sort_values('pair', inplace = True, key= natsort_keygen())
dasum33.sort_values('pair', inplace = True, key= natsort_keygen())
dasum44.sort_values('pair', inplace = True, key= natsort_keygen())

ats11 = decsum11.loc[:,['phantom', 'angle', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']]
diff11 = dasum11.loc[:,['phantom', 'angle', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']]

for col in ['sig2power', 'power_dB']:
    ats11[col] = decsum22[col].values - decsum11[col].values
    diff11[col] = dasum22[col].values - decsum11[col].values

ats11.rename({'power_dB' : 'SCR'}, axis = 1, inplace = True)
diff11.rename({'power_dB' : 'SCR'}, axis = 1, inplace = True)

# %%
ats11

# %%
diff11

# %%


# %%


# %%


# %%
ats = dfcomp.specific_comparison(dec2, dec1, comp_columns=['signal', 'power_dB'])
diff = dfcomp.specific_comparison(da2, da1, comp_columns=['signal', 'power_dB'])

# %%
ats.rename({'power_dB_diff' : 'scr'}, axis = 1, inplace = True)
diff.rename({'power_dB_diff' : 'scr'}, axis = 1, inplace = True)

# %%
ats2 = dfcomp.specific_comparison(decsum2, decsum1, comp_columns=['signal', 'power_dB'])
diff2 = dfcomp.specific_comparison(dasum2, dasum1, comp_columns=['signal', 'power_dB'])

ats2.rename({'power_dB_diff' : 'scr'}, axis = 1, inplace = True)
diff2.rename({'power_dB_diff' : 'scr'}, axis = 1, inplace = True)

# %%
dfa['sig2power'] = dfa['signal'] **2
decluttered['sig2power'] = decluttered['signal'] **2

# %%
decsum1 = decluttered.loc[decluttered.plug.eq(2) & decluttered.rep.eq(1)].groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
decsum2 = decluttered.loc[decluttered.plug.eq(38) & decluttered.rep.eq(1)].groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

dasum1 = dfa.loc[dfa.plug.eq(2) & dfa.rep.eq(1)].groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()
dasum2 = dfa.loc[dfa.plug.eq(38) & dfa.rep.eq(1)].groupby(['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']).agg({'sig2power': np.sum}).reset_index()

# %%
decsum1['power_dB'] = 10*np.log10(decsum1['sig2power'])
decsum2['power_dB'] = 10*np.log10(decsum2['sig2power'])

dasum1['power_dB'] = 10*np.log10(dasum1['sig2power'])
dasum2['power_dB'] = 10*np.log10(dasum2['sig2power'])

# %%
ats3 = dfcomp.specific_comparison(decsum2, decsum1, comp_columns=['signal', 'sig2power', 'power_dB'])
diff3 = dfcomp.specific_comparison(dasum2, dasum1, comp_columns=['signal', 'sig2power', 'power_dB'])

# %%
ats3 = decsum1.loc[:,['phantom', 'angle', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']]
diff3 = dasum1.loc[:,['phantom', 'angle', 'date', 'rep', 'iter', 'pair', 'attLO', 'attRF']]

# %%
from natsort import natsort_keygen

decsum1.sort_values('pair', inplace = True, key= natsort_keygen)
decsum2.sort_values('pair', inplace = True, key= natsort_keygen)
dasum1.sort_values('pair', inplace = True, key= natsort_keygen)
dasum2.sort_values('pair', inplace = True, key= natsort_keygen)

ats3.sort_values('pair', inplace = True, key= natsort_keygen)
diff3.sort_values('pair', inplace = True, key= natsort_keygen)

for col in ['sig2power', 'power_dB']:
    ats3[col] = decsum2[col] - decsum1[col]
    diff3[col] = dasum2[col] - dasum1[col]

ats3.rename({'power_dB' : 'scr'}, axis = 1, inplace = True)
diff3.rename({'power_dB' : 'scr'}, axis = 1, inplace = True)

# %%
ats3

# %%
diff3
