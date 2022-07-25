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
import matplotlib.pyplot as plt
from natsort import natsort_keygen
import numpy as np
import pandas as pd
import seaborn as sns
# from tqdm import tqdm # when using terminal
from tqdm.autonotebook import tqdm # when using Jupyter Notebook

# Local application imports
from NarrowBand.analysis_pd import df_antenna_space as dfant
from NarrowBand.analysis_pd import df_processing as dfproc


def scr_per_pair(df, df_ref, data_col = ['sig2power'], info_cols = ['phantom', 'angle', 'plug', 'date', 'rep', 'iter', 'pair', 'Tx', 'Rx', 'attLO', 'attRF'],
                    noise_level_dB = -70):

    if not isinstance(data_col, list):
        data_col = [data_col]

    dfsum = df.groupby(info_cols).agg({col: np.sum for col in data_col}).reset_index()
    dfsum_ref = df_ref.groupby(info_cols).agg({col: np.sum for col in data_col}).reset_index()

    for col in data_col:
        if len(data_col) == 1:
            dbname = 'power_dB'
        else:
            dbname = "".join((col,"_dB"))
        # Decibel conversion, values <=0 stay at noise_level_dB (warnings unnecessary)
        with np.errstate(divide='ignore', invalid='ignore'):
            dfsum.loc[:, dbname] = 10*np.log10(dfsum.loc[:, col].values, where= dfsum.loc[:, col] > 0, out = (noise_level_dB/10) *np.ones(dfsum.loc[:, col].size));
            dfsum_ref.loc[:, dbname] = 10*np.log10(dfsum_ref.loc[:, col].values, where= dfsum_ref.loc[:, col] > 0, out = (noise_level_dB/10) *np.ones(dfsum_ref.loc[:, col].size));

    dfsum.sort_values('pair', inplace = True, key= natsort_keygen())
    dfsum_ref.sort_values('pair', inplace = True, key= natsort_keygen())

    scr_df = dfsum.loc[:, [elem for elem in info_cols if elem not in ['Tx', 'Rx']]] # Tx and Rx create problems here

    if len(data_col) == 1:
        data_col.append('power_dB')
    else:
        colcopy = data_col.copy()
        for col in data_col:
            data_col.append("".join((col,"_dB")))

    for col in data_col:
        scr_df[col] = dfsum[col].values - dfsum_ref[col].values

    if 'pair' in info_cols:
        scr_df.loc[:, 'Tx'] = scr_df.pair.astype('str').str.extract('(\d+),').astype(int).values
        scr_df.loc[:, 'Rx'] = scr_df.pair.astype('str').str.extract(',(\d+)').astype(int).values

    scr_df.rename({'power_dB' : 'SCR'}, axis = 1, inplace = True)

    return scr_df

def scr_sort_pairs(scr_df, reference_point = "tumor", sort_type = "distance", decimals = 4, out_distances = True, out_as_list = True, narrowband = True, array_config = 'hemisphere'):
    """Return list of dataframes with antenna pair column as categorical, sorted by distance to reference point.

    The input dataframe is split into a list of dataframes, according to phantom and angle.
    These elements change the spatial configuration within the hemisphere, leading to different distances for sorting.

    Parameters
    ----------
    scr_df : DataFrame or list of Dataframes
        SCR scan data set or list of DataFrames
    reference_point: str or array-like, default "tumor"
        input "tumor", "tumour" or "plug" to calculate based on position of plug inside phantom, otherwise receives a 3-D coordinate in form [x,y,z]
    sort_type : str, default "distance"
       available selections:
            "distance": calculate minimum travel distance between each antenna in a pair and the reference point
            "mean-point": calculate mean-point between antennas in a pair and its distance to the reference point
            "between_antennas": direct distance between antennas, disregarding the reference point


    decimals : int, default 4
        number of decimals for rounding
    out_distances: bool, optional
        set to True to provide optional return column with distances, by default False
    out_as_list: bool, optional
        set to True to output list, otherwise concatenate back to DataFrame, by default True
    narrowband : bool, optional
            set to True to switch positions of antennas 4 and 13, by default True
            this is used for the narrowband system due to switching circuit manufacturing error
    array_config : str, optional
        selection for array configuration, by default 'hemisphere'
        options:
        'hemisphere': ring array as in 3-D printed hemisphere
        'hybrid': hybrid bra (mix between ring and cross)
        'ring': ring bra (similar to hemisphere, but varying radius)

    Returns
    ----------
    df_list: list of DataFrame or DataFrame
        list with DataFrames split by phantom, angle with the "pair" column sorted (when out_as_list is True)
        or concatenated DataFrame
    """

    df_list = list([])

    if not isinstance(scr_df, list):
        scr_df = [scr_df]

    for df1 in tqdm(scr_df):
        for ph in tqdm(df1.phantom.drop_duplicates().to_list(), leave= False):
            for ang in tqdm(df1.loc[df1.phantom.eq(ph), "angle"].drop_duplicates().to_list(), leave= False):

                df_list.append(df1.loc[df1.phantom.eq(ph) & df1.angle.eq(ang)])

                full_pairs = dfproc.allpairs2list(df_list[-1])

                if out_distances:
                    sorted_pairs, distance = dfant.sort_pairs(phantom=ph, angle=ang, selected_pairs = full_pairs, reference_point = reference_point, sort_type = sort_type,
                                                                        decimals = decimals, out_distances=True, narrowband = narrowband, array_config = array_config)
                else:
                    sorted_pairs = dfant.sort_pairs(phantom=ph, angle=ang, selected_pairs = full_pairs, reference_point = reference_point, sort_type = sort_type, decimals = decimals, out_distances=False,
                                                    narrowband = narrowband, array_config = array_config)

                sorted_pairs = [ "".join(("(", str(p[0]), ",", str(p[1]), ")")) for p in sorted_pairs]

                if out_distances:
                    d = dict(zip(sorted_pairs,distance))
                    dist = df_list[-1].loc[:,"pair"].apply(lambda x: d.get(x)).copy()
                    df_list[-1] = df_list[-1].assign(distance=dist.values)

                df_list[-1].loc[:,'pair'] = pd.Categorical(df_list[-1].loc[:,"pair"], ordered=True, categories=sorted_pairs)

                # checking the x-axis column
                if df_list[-1].columns.str.contains('SCR', case=False, na=False).all():
                    xlabel = 'SCR'
                else:
                    xlabel = 'power_dB'

                # verification of existence of values to be sorted in dataframe, maintaining sorting order
                sort_list = ["phantom", "angle", "plug", "attLO", "attRF", "date", "rep", "iter", "pair", xlabel]
                intersection = [x for x in sort_list if x in frozenset(df_list[-1].columns)]

                df_list[-1] = df_list[-1].sort_values(intersection, inplace=False, ignore_index=True).copy()

    if not out_as_list:
        df_list = pd.concat(df_list, axis = 0)

    return df_list

def scr_heatmap(scr_df, value_col = 'SCR', vmin = -1, vmax = 1, figsize= (20,10), font_scale=1, save_fig = False, dpi = 600,
                output_folder = 'C:/Users/leofo/OneDrive - McGill University/Documents McGill/Thesis/Figures/Ch 6/',
                file_name = 'SCR_Heatmap.pdf',
                **kwargs):
    """Generate heatmap from SCR dataframe.

    Can either plot figure or save to file.

    Parameters
    ----------
    scr_df : Pandas DataFrame
        SCR DataFrame
    value_col: str, optional
        column with values, by default 'SCR'
    vmin : int, optional
        minimum colorbar value, by default -1
    vmax : int, optional
        maximum colorbar value, by default 1
    figsize : tuple, optional
        figure dimensions in inches, by default (20,10)
    font_scale : int, optional
        font scale, by default 1
    save_fig : bool, optional
        set to True to save figure, by default False
    dpi : int, optional
        dots per inch, by default 600
    output_folder : str, optional
        output file folder, by default 'C:/Users/leofo/OneDrive - McGill University/Documents McGill/Thesis/Figures/Ch 6/'
    file_name : str, optional
        output file name with format, by default 'SCR_Heatmap.pdf'
    """
    scr_table = scr_df.pivot(index = 'Tx', columns = 'Rx', values = value_col)
    sns.set(rc={'figure.figsize':figsize}, font_scale = font_scale)

    plt.figure()
    g = sns.heatmap(scr_table, annot=True, fmt='.2f', center=None, vmin = vmin, vmax = vmax, cbar_kws={'label': 'SCR (dB)'},
                    **kwargs)
    # g.invert_yaxis()

    if save_fig:
        if not os.path.exists(os.path.dirname(output_folder)):
            os.makedirs(os.path.dirname(output_folder))
        g.figure.savefig("".join((output_folder, file_name)), dpi=dpi, bbox_inches="tight")
    else:
        plt.show()