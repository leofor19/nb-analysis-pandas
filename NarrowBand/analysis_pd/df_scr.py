# Python 3.10
# 2022-04-29

# Version 0.0.1
# Latest update 2022-04-29

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

    Description:
            Module for performing per antenna pair signal-to-clutter ratio (SCR) between two phantom scans on Pandas DataFrames for the narrow band or ultrawideband system.

            Functions are updated to be more general and flexible than in the NarrowBand.analysis_pd.df_processing module.

    Dependencies::
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
import os.path

# Third-party library imports
from natsort import natsort_keygen
import numpy as np
# from tqdm import tqdm # when using terminal
from tqdm.autonotebook import tqdm # when using Jupyter Notebook

# Local application imports

def scr_per_pair(df, df_ref, data_col = ['sig2power'], info_cols = ['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF']):

    if not isinstance(data_col, list):
        data_col = [data_col]

    dfsum = df.groupby(info_cols).agg({data_col: np.sum}).reset_index()
    dfsum_ref = df_ref.groupby(info_cols).agg({col: np.sum for col in data_col}).reset_index()

    dfsum['power_dB'] = 10*np.log10(dfsum[data_col].values)
    dfsum_ref['power_dB'] = 10*np.log10(dfsum_ref[data_col].values)

    dfsum.sort_values('pair', inplace = True, key= natsort_keygen())
    dfsum_ref.sort_values('pair', inplace = True, key= natsort_keygen())

    scr_df = dfsum.loc[:, [elem for elem in info_cols if elem not in ['Tx', 'Rx']]] # Tx and Rx create problems here

    data_col.append('power_dB')

    for col in data_col:
        scr_df[col] = dfsum[col].values - dfsum_ref[col].values

    scr_df.rename({'power_dB' : 'SCR'}, axis = 1, inplace = True)

    return scr_df