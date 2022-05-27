# Python 3.10
# 2022-03-04

# Version 0.0.2
# Latest update 2022-03-07

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

    Description:
            Module for performing decluttering on Pandas DataFrames for the narrow band or ultrawideband system.

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
from copy import deepcopy
from datetime import datetime
import json
import os
import os.path
from pathlib import Path
import sys

# Third-party library imports
# import dask.dataframe as dd
# from dask.diagnostics import ProgressBar
#import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook
#from tqdm.dask import TqdmCallback
from yaspin import yaspin

# Local application imports
import NarrowBand.analysis_pd.df_antenna_space as dfant
import NarrowBand.analysis_pd.df_processing as dfproc
import NarrowBand.analysis_pd.df_data_essentials as nbd
from NarrowBand.analysis_pd.uncategorize import uncategorize

def simple_declutter(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']),  processed_path = "Processed/DF 04/",
                        sub_folder = "Means/", correction = np.around(1.0e3/8192,4), decimals = 4, file_format="parquet", parquet_engine= 'pyarrow', is_recursive= False, center = 'mean'):
    """Perform Average Trace Subtraction on Pandas DataFrame "phantom data set files".

    Uses a simple clutter rejection technique in the form of a spatial filter.

    Calculates average singal per frequency/time for antenna pairs with a same distance (assuming the clutter is constant for such pairs).

    Next, subtracts this mean clutter from the original signals.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    sub_folder : str, optional
        sub-folder for file sweep, by default ""
        searched location will be main_path + "Phantom*/" + sub_folder
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/DF"
        final location will be main_path + processed_path
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    conv_path : str, optional
        sub-folder for JSON files of converted data (if correction != 1), by default "Converted/"
    decimals : int, optional
        number of decimals cases for np.arounding values, in particular after conversion, by default 2
    file_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    center: str, optional
        center tendency option between 'mean' or 'median', by default 'mean'
    """

    if not isinstance(date, list):
        date = [date]

    main_paths = []

    if "mean".casefold() in sub_folder.casefold():

        columns = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq","raw_digital_ch1_mean", "raw_digital_ch2_mean",
                "digital_ch1_mean", "digital_ch2_mean", "n_digital_ch1_mean", "n_digital_ch2_mean", "digital_unit"]
    else:
        columns = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq","raw_digital_ch1", "raw_digital_ch2",
                "digital_ch1", "digital_ch2", "n_digital_ch1", "n_digital_ch2", "digital_unit"]

    for d in tqdm(date):

        main_paths = ["".join((main_path,d,"/",processed_path, sub_folder))]

        out_path = "".join((main_path,d,"/",processed_path,"Decluttered/{0} Phantom Set Decluttered.parquet".format(d)))

        df_list = dfproc.df_collect(main_paths, is_recursive=is_recursive, file_format=file_format, columns=columns)
        if ~df_list[0].columns.str.contains('distances', case=False, na=False).any():
            df_list = dfproc.dfsort_pairs(df_list, reference_point = "tumor", sort_type = "between_antennas", decimals = 4, out_distances = True)

        decl_list = []
        tqdm.pandas()

        for df in tqdm(df_list, leave=False):
            dunit = df.digital_unit.head(1).item()
            df = df.progress_apply(lambda x: uncategorize(x), axis=1)
            if "index" in df.columns:
                df.drop("index", axis=1, inplace=True)
            if "voltage_unit" in df.columns:
                #vunit = data.voltage_unit.unique().tolist()[0]
                # remove "voltage_mag" or "voltage_phase" columns
                df = df.loc[:,~df.columns.str.contains('^(?=.*voltage)(?=.*mag|.*phase).*', case=False, na=False)]
            df = df.loc[:,~df.columns.str.contains('subject', case=False, na=False)]
            df = df.loc[:,~df.columns.str.contains('(?=.*digital)(?!.*ch[12]).*', case=False, na=False)]
            clutter = avg_trace_clutter(df, progress_bar = False, center = center)

            df.set_index(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", "freq", "pair", "Tx", "Rx"], inplace=True)
            #pairs = df["pair"]
            #TX = df["Tx"]
            #RX = df["Rx"]
            #df.drop(["pair", "Tx", "Rx"], axis = 1, inplace = True, errors = "ignore")

            if "mean".casefold() in sub_folder.casefold():
                keys = {"raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0, "digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 8, "n_digital_ch2_mean": 8}
            else:
                 keys = {"raw_digital_ch1": 0, "raw_digital_ch2": 0, "digital_ch1": 0, "digital_ch2": 0, "n_digital_ch1": 8, "n_digital_ch2": 8}
            df = df.subtract(clutter, fill_value=0, axis = 0).round(keys).reset_index()

            #df["pair"] = pairs
            #df["Tx"] = TX
            #df["Rx"] = RX
            df["digital_unit"] = dunit
            dfproc.df_convert_values(df, correction = correction)

            if "iter" in df.columns:
                df["subject"] = [" ".join((a, "Phantom", b, "Angle", c, "Rep", d, "Iter", e))
                                                for a,b,c,d,e in zip(df.date.astype(str), df.phantom.astype(str), df.angle.astype(str), df.rep.astype(str), df.iter.astype(str))]
            elif "rep" in df.columns:
                df["subject"] = [" ".join((a, "Phantom", b, "Angle", c, "Rep", d))
                                                for a,b,c,d in zip(df.date.astype(str), df.phantom.astype(str), df.angle.astype(str), df.rep.astype(str))]
            else:
                df["subject"] = [" ".join((a, "Phantom", b, "Angle", c))
                                                for a,b,c in zip(df.date.astype(str), df.phantom.astype(str), df.angle.astype(str))]

            decl_list.append(df)

        dfw = pd.concat(decl_list)
        dfproc.convert2category(dfw, cols = ["phantom", "angle", "plug", "date", "rep", "iter", "pair", "Tx", "Rx", "freq", "digital_unit", "voltage_unit", "subject"])
        if "index" in dfw.columns:
            dfw.drop("index", axis=1, inplace=True)

        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        dfw.reset_index().to_parquet(out_path, engine=parquet_engine)
        tqdm.write(f"\nSaved file: {out_path}        ")

def subtract_clutter(df, clutter, column = 'signal'):
    """Subtract clutter from phantom scan dataframe.

    Parameters
    ----------
    df : Pandas df
        input phantom scan DataFrame
    clutter : Pandas df
       input clutter DataFrame (such as output from avg_trace_clutter function)
    column : str or List[str], optional
       column to perform subtraction, by default 'signal'

    Returns
    -------
    df: Pandas df
        output DataFrame after clutter subtraction
    """
    if not isinstance(column, list):
        column = [column]

    # checking the x-axis column
    if df.columns.str.contains('time', case=False, na=False).any():
        xlabel = 'time'
    elif df.columns.str.contains('samples', case=False, na=False).any():
        xlabel = 'samples'
    else:
        xlabel = 'freq'
    dunit = df.digital_unit.head(1).item()

    clutter = deepcopy(clutter)

    tqdm.pandas()
    df = df.progress_apply(lambda x: uncategorize(x), axis=1)
    if "index" in df.columns:
        df.drop("index", axis=1, inplace=True)
    if "level_0" in df.columns:
        df.drop("level_0", axis=1, inplace=True)
    if "voltage_unit" in df.columns:
        #vunit = data.voltage_unit.unique().tolist()[0]
        # remove "voltage_mag" or "voltage_phase" columns
        vunit = df.voltage_unit.head(1).item()
        df.drop('voltage_unit', axis = 1, inplace = True)
        df = df.loc[:,~df.columns.str.contains('^(?=.*voltage)(?=.*mag|.*phase).*', case=False, na=False)]
    df = df.loc[:,~df.columns.str.contains('subject', case=False, na=False)]
    df = df.loc[:,~df.columns.str.contains('(?=.*digital)(?!.*ch[12]).*', case=False, na=False)]

    # df.set_index(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", xlabel, "pair", "Tx", "Rx"], inplace=True)
    # df.set_index(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", xlabel], inplace=True)
    # clutter.set_index(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", xlabel], inplace=True)
    for col in column:
        clutter.rename(columns={col: "".join((col,"_clutter"))}, inplace = True, errors='ignore')
    # df.merge(clutter['clutter'], how='left', validate = 'many_to_one', left_index = True).reset_index()
    df = pd.merge(df, clutter, how='left', validate = 'many_to_one', 
                left_on = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", xlabel],
                right_on = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", xlabel]).reset_index()

    # if "mean".casefold() in df.columns:
    #     keys = {"raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0, "digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 8, "n_digital_ch2_mean": 8}
    # else:
    #         keys = {"raw_digital_ch1": 0, "raw_digital_ch2": 0, "digital_ch1": 0, "digital_ch2": 0, "n_digital_ch1": 8, "n_digital_ch2": 8}
    # # df = df.subtract(clutter, fill_value=0, axis = 0).round(keys).reset_index()
    # df['signal'] = df['signal'].subtract(df['clutter'], fill_value=0, axis = 0).round(keys).reset_index()
    # clutter_cols = [col for col in df.columns if "_clutter" in col]
    for c in column:
        df[c] = df[c].subtract(df["".join((c,"_clutter"))], fill_value=0, axis = 0)

    df["digital_unit"] = dunit
    if vunit:
        df["voltage_unit"] = vunit

    df.reset_index(inplace=True)

    return df

def avg_trace_clutter(df, progress_bar = True, center='mean', out_as_list = False):
    """Calculate average trace cluttter per distance between antennas and per frequency.

    Calculates average singal per frequency/time for antenna pairs with a same distance (assuming the clutter is constant for such pairs).

    Only uses digital signals.

    Parameters
    ----------
    df : Pandas df or list of df
        input DataFrame or list of dataframes
    progress_bar: bool, optional
        set to True to display tqdm progress bar, by default True
    center: str, optional
        center tendency option between 'mean' or 'median', by default 'mean'
    out_as_list: bool, optional
        set to True to output list, otherwise concatenate back to DataFrame, by default False

    Returns
    -------
    clutter: Pandas df or list of df
        DataFrame(s) with average trace clutter values
        multi-indexed using ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", "freq"]
    """
    if not isinstance(df, list):
        df = [df]

    if ~df[0].columns.str.contains('distances', case=False, na=False).any():
        df = dfproc.dfsort_pairs(df, reference_point = "tumor", sort_type = "between_antennas", decimals = 4, out_distances = True)

    if center == 'median':
        agg_center = 'median'
    else:
        agg_center = 'mean'

    clutter = []
    tqdm.pandas()
    for data in tqdm(df, disable = ~progress_bar):
        data = data.progress_apply(lambda x: uncategorize(x), axis=1)
        if "index" in data.columns:
            data.drop("index", axis=1, inplace=True)
        if "level_0" in data.columns:
            data.drop("level_0", axis=1, inplace=True)
        if "voltage_unit" in data.columns:
            #vunit = data.voltage_unit.unique().tolist()[0]
            # remove "voltage_mag" or "voltage_phase" columns
            data = data.loc[:,~data.columns.str.contains('^(?=.*voltage)(?=.*mag|.*phase).*', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('subject', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('(?=.*digital)(?!.*ch[12]).*', case=False, na=False)]

        # checking the x-axis column
        if data.columns.str.contains('time', case=False, na=False).any():
            xlabel = 'time'
        elif data.columns.str.contains('samples', case=False, na=False).any():
            xlabel = 'samples'
        else:
            xlabel = 'freq'

        c = data.drop(["Tx","Rx"], axis = 1).groupby(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", xlabel], observed=True).agg([agg_center])
        c.columns = [x[0] if isinstance(x,tuple) else x for x in c.columns.ravel()]
        if "index" in c.columns:
            c.drop("index", axis=1, inplace=True)
        if "digital_ch1_mean".casefold() in c.columns:
            keys = {"raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0, "digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 8, "n_digital_ch2_mean": 8}
        else:
            keys = {"raw_digital_ch1": 0, "raw_digital_ch2": 0, "digital_ch1": 0, "digital_ch2": 0, "n_digital_ch1": 8, "n_digital_ch2": 8}
        c = c.round(keys)
        c.reset_index(inplace=True)
        clutter.append(c)

    if len(clutter) == 1:
        clutter = clutter[0]
    elif ~out_as_list:
        clutter = pd.concat(clutter, axis = 0)

    return clutter