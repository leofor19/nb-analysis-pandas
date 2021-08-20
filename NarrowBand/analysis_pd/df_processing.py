# Python 3.8
# 2021-03-05

# Version 1.2.3
# Latest update 2021-08-19

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

    Description:
            Module for performing data processing on Pandas DataFrames for the narrow band system.

    Dependencies::
                    dask
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
from datetime import datetime
import json
import os
import os.path
from pathlib import Path
import re
import sys
import warnings

# Third-party library imports
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
#import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
from tqdm import tqdm
#from tqdm.dask import TqdmCallback
from yaspin import yaspin

# Local application imports
import NarrowBand.analysis_pd.df_antenna_space as dfant
#import NarrowBand.analysis_pd.df_processing as dfproc
import NarrowBand.analysis_pd.df_data_essentials as nbd
from NarrowBand.analysis_pd.uncategorize import uncategorize

# To supress warnings.warn("Non-categorical multi-index is likely brittle", UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

cal_dtypes1 = {
                "voltage_unit": str, "digital_unit": str, "obs": str, "samp_rate": "Float64", "nsamples": "int64", "iter": "int32", "rep": "int32", "date": str,
                "cal_type": "int32", "attRF": str, "attLO": str, "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "Float64",
                "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

cal_dtypes2 = {
                "voltage_unit": str, "digital_unit": str, "obs": str, "samp_rate": "Float64", "nsamples": "int64", "iter": "int32", "rep": "int32", "date": str,
                "cal_type": "int32", "attRF": str, "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "Float64",
                "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

cal_dtypes3 = {
                "voltage_unit": str, "digital_unit": str, "obs": str, "samp_rate": "Float64", "nsamples": "int64", "iter": "int32", "rep": "int32", "date": str,
                "cal_type": "int32", "attRF": "int32", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "Float64",
                "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }


cal_dtypes4 = {
                "voltage_unit": str, "digital_unit": str, "obs": str, "samp_rate": "Float64", "nsamples": "int64", "phantom": str, "plug": str, "iter": "int32", 
                "rep": "int32", "date": str, "attRF": "int32", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", 
                "freq": "Float64", "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

ph_dtypes = {
                "voltage_unit": str, "digital_unit": str, "obs": str, "samp_rate": "Float64", "nsamples": "int64", "phantom": str, "plug": str, "iter": "int32", 
                "rep": "int32", "date": str, "attRF": "int32", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", 
                "freq": "Float64", "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

def dd_collect(path_list, is_recursive = False, file_format="parquet", columns=None, check_key=None, check_value=None, parquet_engine= 'pyarrow'):
    """Collect all Dask DataFrame files in given directory path list.

    Returns a list with the DataFrame files found.

    Parameters
    ----------
    main_path : list of str
        list of paths to measurement files
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False
    file_format: str, optional
        target file format ("parquet" or "csv"), by default "parquet"
    column: list of str, optional
        list of columns to read, by default None (reads all columns)
    chcek_key: str, optional
        key to check value, by default None
    chcek_value: number or str, optional
        value to check, by default None
    parquet_engine: str, optional
        Parquet reader library to use, by default 'pyarrow'
        Options include: ‘auto’, ‘fastparquet’, ‘pyarrow’.
        The ‘auto’ option selects the FastParquetEngine if fastparquet is installed (and ArrowDatasetEngine otherwise).
        If ‘pyarrow’ or ‘pyarrow-dataset’ is specified, the ArrowDatasetEngine (which leverages the pyarrow.dataset API) will be used.

    Returns
    ----------
    df_list: list of df
        1-D list with the Pandas DataFrame files found
    """

    if not isinstance(path_list, list):
        path_list = [path_list]

    df_paths = natsorted(df_sweep(path_list, is_recursive, file_format))

    df_list = []
    if file_format.casefold() == "parquet":
        for p in tqdm(df_paths):
            df_list.append(dd.read_parquet(p, engine=parquet_engine, columns=columns))
            tqdm.write(f"\rOpened file: {p}        ", end="\r")
    elif file_format.casefold() == "csv":
        for p in tqdm(df_paths):
            df_list.append(dd.read_csv(p, cols = columns, low_memory=False))
            tqdm.write(f"\rOpened file: {p}        ", end="\r")

    if check_key is not None:
        # df_list = [df for df in df_list if df.compute()[check_key].eq(check_value).all()]
        df_list = [df.loc[df[check_key] == check_value] for df in df_list]

    tqdm.write("", end="\n")

    return df_list

def df_collect(path_list, is_recursive = False, file_format="parquet", columns=None, specifier=None, parquet_engine= 'pyarrow'):
    """Collect all Pandas DataFrame files in given directory path list.

    Returns a list with the DataFrame files found.

    Parameters
    ----------
    main_path : list of str
        list of paths to measurement files
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False
    file_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    columns: list of str, optional
        list of columns to read, by default None (reads all columns)
    specifier: str, optional
        string to include only specific files, for instance 'Calibration Processed Means Type 3.parquet', by default None (includes all file paths)
    parquet_engine: str, optional
        Parquet reader library to use, by default 'pyarrow'
        Options include: ‘auto’, ‘fastparquet’, ‘pyarrow’.
        If ‘auto’, then the option io.parquet.engine is used.
        The default io.parquet.engine behavior is to try ‘pyarrow’,
        falling back to ‘fastparquet’ if ‘pyarrow’ is unavailable.

    Returns
    ----------
    df_list: list of df
        1-D list with the Pandas DataFrame files found
    """

    if not isinstance(path_list, list):
        path_list = [path_list]

    df_paths = natsorted(df_sweep(path_list, is_recursive, file_format))

    if specifier:
        df_paths = [p for p in df_paths if specifier in p]

    df_list = []
    if file_format.casefold() == "parquet":
        for p in tqdm(df_paths):
            df_list.append(pd.read_parquet(p, engine=parquet_engine, columns=columns))
            tqdm.write(f"\rOpened file: {p}        ", end="\r")
    elif file_format.casefold() == "csv":
        for p in tqdm(df_paths):
            df_list.append(pd.read_csv(p, usecols = columns, low_memory=False))
            tqdm.write(f"\rOpened file: {p}        ", end="\r")

    tqdm.write("",end="\n")

    return df_list

def df_sweep(path_list, is_recursive = False, file_format="parquet"):
    """Find all Pandas DataFrame files in given directory path list.

    Returns a list with the string pathnames of each DataFrame file found.

    Parameters
    ----------
    main_path : list of str
        list of paths main path to measurement files
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False
    file_format: str
        target file format (either "parquet" or "csv"), by default "parquet"

    Returns
    ----------
    df_paths: list of str
        1-D list with full pathnames to each Pandas DataFrame file found
    """

    if is_recursive:
        target = "".join(("**/*.",file_format))
    else:
        target = "".join(("*.",file_format))

    if not isinstance(path_list, list):
        path_list = [path_list]

    df_paths = []
    for p in path_list:
        df_paths.extend([str(i) for i in list(Path(p).glob(target))])
    df_paths = natsorted(df_paths)

    return df_paths

def dB2Volts(gain_dB):
    """Convert gain in dB to Volts/Volts.

    Parameters
    ----------
    gain_dB : int or float
        gain in dB

    Returns
    ----------
    float
        gain in Volts/Volts
    """

    if isinstance(gain_dB, str):
        # for cases such as attRF = "grounded"
        gain_dB = 0.0

    return 10.0**(gain_dB/20.0)

def calculate_power_dBm(df, Z = 50.0, noise_floor = -108):
    """Calculates power in dBm and outputs to df["power_dBm"].

    Compensates for mV if that is the only "voltage_unit" found.

    Zero values are replaced by a noise floor estimation, by default -108 dBm. This avoids NaN results.

    Parameters
    ----------
    df : Pandas dataframe
        dataframe with I and Q voltage channels
    Z: int or float (optional)
        impedance in ohms, by default 50.0 ohms
    noise_floor: float (optional)
        estimated noise floor in dBm to replace 0 magnitude values, by default 0.1252
    """

    calculate_power(df, Z = Z)

    df["power_dBm"] = power2dBm(df["power"], noise_floor = noise_floor)

def calculate_power(df, Z = 50.0):
    """Calculates power in Watts and outputs to df["power"].

    Compensates for mV if that is the only "voltage_unit" found.

    Otherwise, directly calculates (Q**2 + I**2) / Z.

    Parameters
    ----------
    df : Pandas dataframe
        dataframe with I and Q voltage channels
    Z: int or float (optional)
        impedance in ohms, by default 50.0 ohms
    """

    q = df.voltage_ch1
    i = df.voltage_ch2

    if df.voltage_unit.unique() == 'mV' or 'converted by factor 0.1221':

        df["power"] = ((q * 1e-3)**2 + (i  * 1e-3)**2) / Z

    else:

        df["power"] = (q**2 + i**2) / Z

def power2dBm(power_W, noise_floor = -108):
    """Convert power in W to dBm.

    Zero values are replaced by a noise floor estimation, by default -108 dBm.

    This avoids NaN results.

    Parameters
    ----------
    power_W : int or float
        power in W
    noise_floor: float (optional)
        estimated noise floor in dBm to replace 0 magnitude values, by default 0.1252

    Returns
    ----------
    dBm
        power in dBm
    """
    power_W[power_W == 0] = noise_floor

    return 10.0 * np.log10(power_W, where= power_W > 0) + 30

def milivolts2dBm(milivolts, Z = 50.0, noise_floor = 0.1252):
    """Convert mV to dBm.

    Zero values are replaced by a noise floor estimation, by default 0.1252 mV.

    This avoids NaN results. Should be used only for specific use cases (digital resolution and noise floor can be estimated).

    Parameters
    ----------
    milivolts : int or float
        voltage rms in mV
    Z: int or float (optional)
        impedance in ohms, by default 50 ohms
    noise_floor: float (optional)
        estimated noise floor in mV to replace 0 magnitude values, by default 0.1252

    Returns
    ----------
    dBm
        power in dBm
    """

    arg = (( (milivolts* 1e-3)**2 ) * 1000) / Z

    arg[arg == 0] = noise_floor * 1e-3

    return 10.0 * np.log10(arg, where= arg > 0)

def volts2dBm(volts, Z = 50.0, noise_floor = 0.1252):
    """Convert V to dBm.

    Zero values are replaced by a noise floor estimation, by default 0.1252 mV.

    This avoids NaN results. Should be used only for specific use cases (digital resolution and noise floor can be estimated).

    Parameters
    ----------
    volts : int or float
        voltage rms in mV
    Z: int or float (optional)
        impedance in ohms, by default 50 ohms
    noise_floor: float (optional)
        estimated noise floor in mV to replace 0 magnitude values, by default 0.1252

    Returns
    ----------
    dBm
        power in dBm
    """

    arg = (( (volts)**2 ) * 1000) / Z

    arg[arg == 0] = noise_floor * 1e-3

    return 10.0 * np.log10(arg, where= arg > 0)

def intersectpairs2list(df):
    """Return sorted list of the antenna pairs intersection between all dates within a dataframe.

    The pairs provided have measurements for all included dates.

    Parameters
    ----------
    df : DataFrame
        phantom scan data set

    Returns
    ----------
    inter_pairs: list
        sorted list with intersection of antenna pairs through all dates within the dataframe
    """
    full_pairs = list([])

    for date in df.date.drop_duplicates().to_list():
        if not 'pair' in df.columns:
            pairs = df.loc[df.date.eq(date),"Tx"].astype(str) + "," + df.loc[df.date.eq(date),"Rx"].astype(str)
            pairs = pairs.drop_duplicates().to_list()
            pairs = set([tuple(map(int, p.split(','))) for p in pairs])
        else:
            pairs = df[df.date.eq(date)]["pair"].drop_duplicates().to_list()
            pairs = set([tuple(map(int, p.lstrip("(").rstrip(")").split(','))) for p in pairs])

        full_pairs.append(pairs)

    inter_pairs = natsorted(set(full_pairs[0]).intersection(*full_pairs))

    return inter_pairs

def allpairs2list(df):
    """Return sorted list of all unique antenna pairs within a dataframe.

    Parameters
    ----------
    df : DataFrame
        phantom scan data set

    Returns
    ----------
    full_pairs: list
        sorted list with unique antenna pairs within the dataframe
    """
    full_pairs = set([])

    for date in df.date.drop_duplicates().to_list():
        if not 'pair' in df.columns:
            pairs = df.loc[df.date.eq(date),"Tx"].astype(str) + "," + df.loc[df.date.eq(date),"Rx"].astype(str)
            pairs = pairs.drop_duplicates().to_list()
            pairs = set([tuple(map(int, p.split(','))) for p in pairs])
        else:
            pairs = df[df.date.eq(date)]["pair"].drop_duplicates().to_list()
            pairs = set([tuple(map(int, p.lstrip("(").rstrip(")").split(','))) for p in pairs])

        full_pairs = full_pairs | pairs

    full_pairs = natsorted(full_pairs)

    return full_pairs

def allpairs2list2(df, select_ref = 1):
    """Return sorted list of all unique antenna pairs within a dataframe.

    Parameters
    ----------
    df : DataFrame
        phantom scan comparison data set
    select_ref: int
        which element of the comparison pairs to use as reference for sorting (1 or 2), by default 1
        this is NOT the phantom number, but refers to the compared pair (column on DataFrame), usually both phantoms are the same

    Returns
    ----------
    full_pairs: list
        sorted list with unique antenna pairs within the dataframe
    """
    full_pairs = set([])

    if "date" not in df.columns:
        d = "".join(("date_",f"{select_ref}"))
    else:
        d = "date"

    for date in df[d].drop_duplicates().to_list():
        if not 'pair' in df.columns:
            pairs = df.loc[df[d].eq(date),"Tx"].astype(str) + "," + df.loc[df[d].eq(date),"Rx"].astype(str)
            pairs = pairs.drop_duplicates().to_list()
            pairs = set([tuple(map(int, p.split(','))) for p in pairs])
        else:
            pairs = df[df[d].eq(date)]["pair"].drop_duplicates().to_list()
            pairs = set([tuple(map(int, p.lstrip("(").rstrip(")").split(','))) for p in pairs])
        full_pairs = full_pairs | pairs

    full_pairs = natsorted(full_pairs)

    return full_pairs

def dfsort_pairs(df, reference_point = "tumor", sort_type = "distance", decimals = 4, out_distances = False):
    """Return list of dataframes with antenna pair column as categorical, sorted by distance to reference point.

    The input dataframe is split into a list of dataframes, according to phantom and angle.
    These elements change the spatial configuration within the hemisphere, leading to different distances for sorting.

    Parameters
    ----------
    df : DataFrame or list of Dataframes
        phantom scan data set or list of DataFrames
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

    Returns
    ----------
    df_list: list of DataFrame
        list with DataFrames split by phantom, angle with the "pair" column sorted
    """

    df_list = list([])

    if not isinstance(df, list):
        df = [df]

    for df1 in tqdm(df):
        for ph in tqdm(df1.phantom.drop_duplicates().to_list(), leave= False):
            for ang in tqdm(df1.loc[df1.phantom.eq(ph), "angle"].drop_duplicates().to_list(), leave= False):

                df_list.append(df1.loc[df1.phantom.eq(ph) & df1.angle.eq(ang)])

                full_pairs = allpairs2list(df_list[-1])

                if out_distances:
                    sorted_pairs, distances = dfant.sort_pairs(phantom=ph, angle=ang, selected_pairs = full_pairs, reference_point = reference_point, sort_type = sort_type,
                                                                        decimals = decimals, out_distances=True)
                else:
                    sorted_pairs = dfant.sort_pairs(phantom=ph, angle=ang, selected_pairs = full_pairs, reference_point = reference_point, sort_type = sort_type, decimals = decimals, out_distances=False)

                sorted_pairs = [ "".join(("(", str(p[0]), ",", str(p[1]), ")")) for p in sorted_pairs]

                if out_distances:
                    d = dict(zip(sorted_pairs,distances))
                    dist = df_list[-1].loc[:,"pair"].apply(lambda x: d.get(x)).copy()
                    df_list[-1] = df_list[-1].assign(distances=dist.values)

                df_list[-1].loc[:,'pair'] = pd.Categorical(df_list[-1].loc[:,"pair"], ordered=True, categories=sorted_pairs)

                # verification of existance of values to be sorted in dataframe, maintaining sorting order
                sort_list = ["phantom", "angle", "plug", "attLO", "attRF", "date", "rep", "iter", "pair", "freq"]
                intersection = [x for x in sort_list if x in frozenset(df_list[-1].columns)]

                df_list[-1] = df_list[-1].sort_values(intersection, inplace=False, ignore_index=True).copy()

    return df_list

def dfsort_pairs_compared(df, reference_point = "tumor", sort_type = "distance", decimals = 4, out_distances = False, select_ref = 1):
    """Return list of dataframes with antenna pair column as categorical, sorted by distance to reference point.

    The input dataframe is split into a list of dataframes, according to phantom and angle.
    These elements change the spatial configuration within the hemisphere, leading to different distances for sorting.

    This function if for compared scans DataFrames (has phantom_1 and phantom_2 values).

    Parameters
    ----------
    df : DataFrame or list of Dataframes
        compared phantom scan data set or list of DataFrames
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
    select_ref: int
        which element of the comparison pairs to use as reference for sorting (1 or 2), by default 1
        this is NOT the phantom number, but refers to the compared pair (column on DataFrame), usually both are the same

    Returns
    ----------
    df_list: list of DataFrame
        list with DataFrames split by phantom, angle with the "pair" column sorted
    """

    p = "".join(("phantom_",f"{select_ref}"))
    a = "".join(("angle_",f"{select_ref}"))

    df_list = list([])

    if not isinstance(df, list):
        df = [df]

    for df1 in tqdm(df):
        for ph in tqdm(df1[p].drop_duplicates().to_list(), leave= False):
            for ang in tqdm(df1.loc[df1[p].eq(ph), a].drop_duplicates().to_list(), leave= False):

                df_list.append(df1.loc[df1[p].eq(ph) & df1[a].eq(ang)])

                full_pairs = allpairs2list2(df_list[-1], select_ref = select_ref)

                if out_distances:
                    sorted_pairs, distances = dfant.sort_pairs(phantom=ph, angle=ang, selected_pairs = full_pairs, reference_point = reference_point, sort_type = sort_type,
                                                                        decimals = decimals, out_distances=True)
                else:
                    sorted_pairs = dfant.sort_pairs(phantom=ph, angle=ang, selected_pairs = full_pairs, reference_point = reference_point, sort_type = sort_type, decimals = decimals, out_distances=False)

                sorted_pairs = [ "".join(("(", str(p[0]), ",", str(p[1]), ")")) for p in sorted_pairs]

                if out_distances:
                    d = dict(zip(sorted_pairs,distances))
                    dist = df_list[-1].loc[:,"pair"].apply(lambda x: d.get(x)).copy()
                    df_list[-1] = df_list[-1].assign(distances=dist.values)

                df_list[-1].loc[:,'pair'] = pd.Categorical(df_list[-1].loc[:,"pair"], ordered=True, categories=sorted_pairs)

                # verification of existance of values to be sorted in dataframe, maintaining sorting order
                sort_list = ["phantom", "angle", "plug", "attLO", "attRF", "date", "rep", "iter", "pair", "freq"]
                intersection = [x for x in sort_list if x in frozenset(df_list[-1].columns)]

                df_list[-1] = df_list[-1].sort_values(intersection, inplace=False, ignore_index=True).copy()

    return df_list

def pivot_for_multivariate(df, index=None, columns=list(["pair","freq"]), values=list(["voltage_mag"])):
    """Return dataframe pivoted to have each measurement for an antenna pair + frequency combination in a separate column.

    This facilitates multivariate analysis (e.g. MANOVA).

    Parameters
    ----------
    df : DataFrame
        phantom scan data set
    index : int, str, tuple, or list, default None
        Only remove the given levels from the index. Removes all levels by default.
    columns : column, Grouper, array, or list of the previous, default ["pair","freq"]
        columns to unstack, producing new variables
        If an array is passed, it must be the same length as the data. The list can contain any of the other types (except list).
        Keys to group by on the pivot table column. If an array is passed, it is being used as the same manner as column values.
    values : column to aggregate, default ["voltage_mag"]
        values columns

    Returns
    ----------
    df2: DataFrame
        DataFrame with each variable for antenna pair + frequency in a separate column
    """

    df2 = df.drop(["Tx", "Rx"], axis=1)
    df2.freq = df2.freq.astype(str)

    # checks if dataframe is full (has time column) or aggregated (providing mean values) by iter or rep

    if "time" in df2.columns:
        keys = list(["phantom", "angle", "plug", "attLO", "attRF", "date", "rep", "iter", "meas_number", "pair", "freq", "time"])

    elif "iter" in df2.columns:
        keys = list(["phantom", "angle", "plug", "attLO", "attRF", "date", "rep", "iter", "meas_number", "pair", "freq"])

    elif "rep" in df2.columns:
        keys = list(["phantom", "angle", "plug", "attLO", "attRF", "date", "rep", "pair", "freq"])
    else:
        keys = list(["phantom", "angle", "plug", "attLO", "attRF", "date", "pair", "freq"])
    df2 = df2[keys + values]
    df2 = df2.set_index(keys=keys).unstack(columns)

    df2.columns = ['_'.join(x).rstrip('_')  if isinstance(x,tuple) else x for x in df2.columns.ravel()]

    if index == None:
        df2.reset_index(inplace=True)
    else:
        df2.set_index(keys=index, inplace=True)

    return df2

# Added on 29/08/2020
# From data_processing.py
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

def cal_data_read2pandas(dates, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", 
                    processed_path = "Processed/DF/", correction = np.around(1.0e3/8192,4), conv_path = "Converted/", decimals = 4,
                    save_format="parquet", parquet_engine= 'pyarrow'):
    """Generate Pandas DataFrame "calibration file" from PScope .adc files.

    First, sweeps trough main_path + date_path + cal_path folders and lists all .adc files found.
    Then, parameters are extracted from types 1, 2 and 3 calibration files.

    Parameters
    ----------
    dates : str or list of str
        date(s) in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        sub-folder with calibration files, by default "Calibration/"
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/DF/"
        final location will be main_path + processed_path + cal_path
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    conv_path : str, optional
        sub-folder for JSON files of converted data (if correction != 1), by default "Converted/"
    decimals : int, optional
        number of decimals cases for np.arounding values, in particular after conversion, by default 2
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    parquet_engine: str, optional
        Parquet reader library to use, by default 'pyarrow'
        Options include: ‘auto’, ‘fastparquet’, ‘pyarrow’.
        If ‘auto’, then the option io.parquet.engine is used.
        The default io.parquet.engine behavior is to try ‘pyarrow’,
        falling back to ‘fastparquet’ if ‘pyarrow’ is unavailable.
    """

    if not isinstance(dates, list):
        dates = [dates]

    for date in tqdm(dates):

        date_path = _date2path(date)

        paths, meta_index, meta_freq = cal_folder_sweep(date = date_path, main_path = main_path, cal_path = cal_path)

        # collects measurement configuration parameters from JSON files

        ConfigSet = collect_json_data(date_path, main_path= main_path, sub_folder = "Config", json_type = "calibration configuration parameters", 
                                                is_recursive= False)

        # Calibration type 1

        data = np.empty((0,2))
        time = np.empty((0,1))

        ite = meta_index[0][:,0]
        rep = meta_index[0][:,1]

        for x, i in enumerate(tqdm(paths[0], leave=False)):

            data, time , nsamples, srate = nbd.data_read(i)

            if x == 0:

                attLO, attRF, obs, freqs = check_configuration_parameters_cal(ConfigSet = ConfigSet, date = date_path, cal_type = 1, rep = rep[x], ite = ite[x])

                cal_df1_list = [pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"])]
                cal_df1_list[-1]["time"] = time
                cal_df1_list[-1]["freq"] = 0
                cal_df1_list[-1].insert(0,"attLO",attLO)
                cal_df1_list[-1].insert(0,"attRF",attRF)
                cal_df1_list[-1].insert(0,"cal_type",1)
                cal_df1_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                cal_df1_list[-1].insert(0, "rep", rep[x])
                cal_df1_list[-1].insert(0, "iter", ite[x])
                cal_df1_list[-1].insert(0, "nsamples", nsamples)
                cal_df1_list[-1].insert(0, "samp_rate", srate*1e6)
                cal_df1_list[-1].insert(0, "obs", obs)
                cal_df1_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")


            if x != 0:
                if rep[x] != rep[x-1] or ite[x] != ite[x-1]:

                    # close data set, save it to file and start new data set dictionary

                    cal_df1 = pd.concat(cal_df1_list)

                    convert2category(cal_df1, cols = ["date", "rep", "iter", "cal_type", "digital_unit", "attLO", "attRF", "freq", "samp_rate", "nsamples", "obs"])

                    if correction != 1:
                        # if there is a correction or conversion factor, saves the data in converted form as well

                        if correction == np.around(1.0e3/8192,4):
                            cal_df1.insert(0, "voltage_unit", "mV")
                        elif correction == np.around(1.0/8192,7):

                            cal_df1.insert(0, "voltage_unit", "V")
                        else:
                            cal_df1.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                        cal_df1["voltage_ch1"] = cal_df1["raw_digital_ch1"]*correction
                        cal_df1["voltage_ch2"] = cal_df1["raw_digital_ch2"]*correction
                        cal_df1["voltage_mag"] = np.sqrt(np.square(cal_df1["voltage_ch1"]) + np.square(cal_df1["voltage_ch2"]))
                        cal_df1["voltage_phase"] = np.arctan2(cal_df1["voltage_ch1"], cal_df1["voltage_ch2"])

                        convert2category(cal_df1, cols = ["voltage_unit"])

                        base_path = main_path + date_path + processed_path + cal_path + conv_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))
                        file_title = " ".join((date_path.replace( "/",""), "Calibration Type 1 V", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                    else:

                        base_path = main_path + date_path + processed_path + cal_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))
                        file_title = " ".join((date_path.replace( "/",""), "Calibration Type 1 D", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                    if save_format.casefold() == "parquet":
                        if parquet_engine == 'pyarrow':
                            cal_df1.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, index= False)
                        else:
                            cal_df1.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                    else:
                        cal_df1.to_csv(base_path + file_title + ".csv")
                    tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                    del cal_df1_list[:]
                    del file_title, base_path, cal_df1

                    attLO, attRF, obs, freqs = check_configuration_parameters_cal(ConfigSet = ConfigSet, date = date_path, cal_type = 1, rep = rep[x], ite = ite[x])


                cal_df1_list.append(pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"]))
                cal_df1_list[-1]["time"] = time
                cal_df1_list[-1]["freq"] = 0
                cal_df1_list[-1].insert(0,"attLO",attLO)
                cal_df1_list[-1].insert(0,"attRF",attRF)
                cal_df1_list[-1].insert(0,"cal_type",1)
                cal_df1_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                cal_df1_list[-1].insert(0, "rep", rep[x])
                cal_df1_list[-1].insert(0, "iter", ite[x])
                cal_df1_list[-1].insert(0, "nsamples", nsamples)
                cal_df1_list[-1].insert(0, "samp_rate", srate*1e6)
                cal_df1_list[-1].insert(0, "obs", obs)
                cal_df1_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            data = np.empty((0,2))
            time = np.empty((0,1))

            if x == len(paths[0])-1 :
                # close data set, save it to file and start new data set dictionary

                cal_df1 = pd.concat(cal_df1_list)
                convert2category(cal_df1, cols = ["date", "rep", "iter", "cal_type", "digital_unit", "attLO", "attRF", "freq", "samp_rate", "nsamples", "obs"])

                if correction != 1:
                    # if there is a correction or conversion factor, saves the data in converted form as well

                    if correction == np.around(1.0e3/8192,4):
                        cal_df1.insert(0, "voltage_unit", "mV")
                    elif correction == np.around(1.0/8192,7):

                        cal_df1.insert(0, "voltage_unit", "V")
                    else:
                        cal_df1.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                    cal_df1["voltage_ch1"] = cal_df1["raw_digital_ch1"]*correction
                    cal_df1["voltage_ch2"] = cal_df1["raw_digital_ch2"]*correction
                    cal_df1["voltage_mag"] = np.sqrt(np.square(cal_df1["voltage_ch1"]) + np.square(cal_df1["voltage_ch2"]))
                    cal_df1["voltage_phase"] = np.arctan2(cal_df1["voltage_ch1"], cal_df1["voltage_ch2"])

                    convert2category(cal_df1, cols = ["voltage_unit"])

                    base_path = main_path + date_path + processed_path + cal_path + conv_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))
                    file_title = " ".join((date_path.replace( "/",""), "Calibration Type 1 V", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                else:

                    base_path = main_path + date_path + processed_path + cal_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))
                    file_title = " ".join((date_path.replace( "/",""), "Calibration Type 1 D", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                if save_format.casefold() == "parquet":
                    if parquet_engine == 'pyarrow':
                        cal_df1.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, index= False)
                    else:
                        cal_df1.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                else:
                    cal_df1.to_csv(base_path + file_title + ".csv")
                tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                del cal_df1_list[:]
                del file_title, base_path, cal_df1

        # Calibration type 2

        data = np.empty((0,2))
        time = np.empty((0,1))

        freq = np.asarray([i.replace('_','.') for i in meta_freq[1]], dtype = float)
        frequencies = np.unique(freq)

        ite = meta_index[1][:,0]
        rep = meta_index[1][:,1]

        for x, i in enumerate(tqdm(paths[1], leave=False)):

            data, time , nsamples, srate = nbd.data_read(i)

            if x == 0:

                attLO, attRF, obs, freqs = check_configuration_parameters_cal(ConfigSet = ConfigSet, date = date_path, cal_type = 2, rep = rep[x], ite = ite[x])

                cal_df2_list = [pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"])]
                cal_df2_list[-1]["time"] = time
                cal_df2_list[-1]["freq"] = freq[x]
                cal_df2_list[-1].insert(0,"attLO", attLO)
                cal_df2_list[-1].insert(0,"attRF", attRF)
                cal_df2_list[-1].insert(0,"cal_type", 2)
                cal_df2_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                cal_df2_list[-1].insert(0, "rep", rep[x])
                cal_df2_list[-1].insert(0, "iter", ite[x])
                cal_df2_list[-1].insert(0, "nsamples", nsamples)
                cal_df2_list[-1].insert(0, "samp_rate", srate*1e6)
                cal_df2_list[-1].insert(0, "obs", obs)
                cal_df2_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            if x != 0:
                if rep[x] != rep[x-1] or ite[x] != ite[x-1]:

                    # close data set, save it to file and start new data set dictionary

                    cal_df2 = pd.concat(cal_df2_list)

                    cal_df2["attLO"] = pd.to_numeric(cal_df2["attLO"])
                    convert2category(cal_df2, cols = ["date", "rep", "iter", "freq", "cal_type", "digital_unit", "attRF", "freq", "samp_rate", "nsamples", "obs"])

                    if correction != 1:
                        # if there is a correction or conversion factor, saves the data in converted form as well

                        if correction == np.around(1.0e3/8192,4):
                            cal_df2.insert(0, "voltage_unit", "mV")
                        elif correction == np.around(1.0/8192,7):

                            cal_df2.insert(0, "voltage_unit", "V")
                        else:
                            cal_df2.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                        cal_df2["voltage_ch1"] = cal_df2["raw_digital_ch1"]*correction
                        cal_df2["voltage_ch2"] = cal_df2["raw_digital_ch2"]*correction
                        cal_df2["voltage_mag"] = np.sqrt(np.square(cal_df2["voltage_ch1"]) + np.square(cal_df2["voltage_ch2"]))
                        cal_df2["voltage_phase"] = np.arctan2(cal_df2["voltage_ch1"], cal_df2["voltage_ch2"])

                        convert2category(cal_df2, cols = ["voltage_unit"])

                        base_path = main_path + date_path + processed_path + cal_path + conv_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))
                        file_title = " ".join((date_path.replace( "/",""), "Calibration Type 2 V", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                    else:

                        base_path = main_path + date_path + processed_path + cal_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))
                        file_title = " ".join((date_path.replace( "/",""), "Calibration Type 2 D", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                    if save_format.casefold() == "parquet":
                        if parquet_engine == 'pyarrow':
                            cal_df2.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, index= False)
                        else:
                            cal_df2.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                    else:
                        cal_df2.to_csv(base_path + file_title + ".csv")
                    tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                    del cal_df2_list[:]
                    del file_title, base_path, cal_df2

                    attLO, attRF, obs, freqs = check_configuration_parameters_cal(ConfigSet = ConfigSet, date = date_path, cal_type = 2, rep = rep[x], ite = ite[x])

                cal_df2_list.append(pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"]))
                cal_df2_list[-1]["time"] = time
                cal_df2_list[-1]["freq"] = freq[x]
                cal_df2_list[-1].insert(0,"attLO", attLO)
                cal_df2_list[-1].insert(0,"attRF", attRF)
                cal_df2_list[-1].insert(0,"cal_type", 2)
                cal_df2_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                cal_df2_list[-1].insert(0, "rep", rep[x])
                cal_df2_list[-1].insert(0, "iter", ite[x])
                cal_df2_list[-1].insert(0, "nsamples", nsamples)
                cal_df2_list[-1].insert(0, "samp_rate", srate*1e6)
                cal_df2_list[-1].insert(0, "obs", obs)
                cal_df2_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            data = np.empty((0,2))
            time = np.empty((0,1))

            if x == len(paths[1])-1 :
                # close data set, save it to file

                cal_df2 = pd.concat(cal_df2_list)

                cal_df2["attLO"] = pd.to_numeric(cal_df2["attLO"])
                convert2category(cal_df2, cols = ["date", "rep", "iter", "freq", "cal_type", "digital_unit", "attRF", "freq", "samp_rate", "nsamples", "obs"])

                if correction != 1:
                    # if there is a correction or conversion factor, saves the data in converted form as well

                    if correction == np.around(1.0e3/8192,4):
                        cal_df2.insert(0, "voltage_unit", "mV")
                    elif correction == np.around(1.0/8192,7):

                        cal_df2.insert(0, "voltage_unit", "V")
                    else:
                        cal_df2.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                    cal_df2["voltage_ch1"] = cal_df2["raw_digital_ch1"]*correction
                    cal_df2["voltage_ch2"] = cal_df2["raw_digital_ch2"]*correction
                    cal_df2["voltage_mag"] = np.sqrt(np.square(cal_df2["voltage_ch1"]) + np.square(cal_df2["voltage_ch2"]))
                    cal_df2["voltage_phase"] = np.arctan2(cal_df2["voltage_ch1"], cal_df2["voltage_ch2"])

                    convert2category(cal_df2, cols = ["voltage_unit"])

                    base_path = main_path + date_path + processed_path + cal_path + conv_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))
                    file_title = " ".join((date_path.replace( "/",""), "Calibration Type 2 V", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                else:

                    base_path = main_path + date_path + processed_path + cal_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))
                    file_title = " ".join((date_path.replace( "/",""), "Calibration Type 2 D", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                if save_format.casefold() == "parquet":
                    if parquet_engine == 'pyarrow':
                        cal_df2.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, index= False)
                    else:
                        cal_df2.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                else:
                    cal_df2.to_csv(base_path + file_title + ".csv")
                tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                del cal_df2_list[:]
                del file_title, base_path, cal_df2

        # Calibration type 3

        freq = np.asarray([i.replace('_','.') for i in meta_freq[2]], dtype = float)
        frequencies = np.unique(freq)

        ite = meta_index[2][:,0]
        rep = meta_index[2][:,1]

        data = np.empty((0,2))
        time = np.empty((0,1))

        for x, i in enumerate(tqdm(paths[2], leave=False)):

            data, time , nsamples, srate = nbd.data_read(i)

            if x == 0:

                attLO, attRF, obs, _ = check_configuration_parameters_cal(ConfigSet = ConfigSet, date = date_path, cal_type = 3, rep = rep[x], ite = ite[x])

                cal_df3_list = [pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"])]
                cal_df3_list[-1]["time"] = time
                cal_df3_list[-1]["freq"] = freq[x]
                cal_df3_list[-1].insert(0,"attLO", attLO)
                cal_df3_list[-1].insert(0,"attRF", attRF)
                cal_df3_list[-1].insert(0,"cal_type", 3)
                cal_df3_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                cal_df3_list[-1].insert(0, "rep", rep[x])
                cal_df3_list[-1].insert(0, "iter", ite[x])
                cal_df3_list[-1].insert(0, "nsamples", nsamples)
                cal_df3_list[-1].insert(0, "samp_rate", srate*1e6)
                cal_df3_list[-1].insert(0, "obs", obs)
                cal_df3_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            if x != 0:
                if rep[x] != rep[x-1] or ite[x] != ite[x-1]:

                    # close data set, save it to file and start new data set dictionary

                    cal_df3 = pd.concat(cal_df3_list)

                    cal_df3["attLO"] = pd.to_numeric(cal_df3["attLO"])
                    cal_df3["attRF"] = pd.to_numeric(cal_df3["attRF"])
                    convert2category(cal_df3, cols = ["date", "rep", "iter", "freq", "cal_type", "digital_unit", "freq", "samp_rate", "nsamples", "obs"])

                    if correction != 1:
                        # if there is a correction or conversion factor, saves the data in converted form as well

                        if correction == np.around(1.0e3/8192,4):
                            cal_df3.insert(0, "voltage_unit", "mV")
                        elif correction == np.around(1.0/8192,7):

                            cal_df3.insert(0, "voltage_unit", "V")
                        else:
                            cal_df3.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                        cal_df3["voltage_ch1"] = cal_df3["raw_digital_ch1"]*correction
                        cal_df3["voltage_ch2"] = cal_df3["raw_digital_ch2"]*correction
                        cal_df3["voltage_mag"] = np.sqrt(np.square(cal_df3["voltage_ch1"]) + np.square(cal_df3["voltage_ch2"]))
                        cal_df3["voltage_phase"] = np.arctan2(cal_df3["voltage_ch1"], cal_df3["voltage_ch2"])

                        convert2category(cal_df3, cols = ["voltage_unit"])

                        base_path = main_path + date_path + processed_path + cal_path + conv_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))
                        file_title = " ".join((date_path.replace( "/",""), "Calibration Type 3 V", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                    else:
                        base_path = main_path + date_path + processed_path + cal_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))

                        file_title = " ".join((date_path.replace( "/",""), "Calibration Type 3 D", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                    if save_format.casefold() == "parquet":
                        if parquet_engine == 'pyarrow':
                            cal_df3.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, index= False)
                        else:
                            cal_df3.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                    else:
                        cal_df3.to_csv(base_path + file_title + ".csv")
                    tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                    del cal_df3_list[:]
                    del file_title, base_path, cal_df3

                    attLO, attRF, obs, freqs = check_configuration_parameters_cal(ConfigSet = ConfigSet, date = date_path, cal_type = 3, rep = rep[x], ite = ite[x])

                cal_df3_list.append(pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"]))
                cal_df3_list[-1]["time"] = time
                cal_df3_list[-1]["freq"] = freq[x]
                cal_df3_list[-1].insert(0,"attLO", attLO)
                cal_df3_list[-1].insert(0,"attRF", attRF)
                cal_df3_list[-1].insert(0,"cal_type", 3)
                cal_df3_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                cal_df3_list[-1].insert(0, "rep", rep[x])
                cal_df3_list[-1].insert(0, "iter", ite[x])
                cal_df3_list[-1].insert(0, "nsamples", nsamples)
                cal_df3_list[-1].insert(0, "samp_rate", srate*1e6)
                cal_df3_list[-1].insert(0, "obs", obs)
                cal_df3_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            data = np.empty((0,2))
            time = np.empty((0,1))

            if x == len(paths[2])-1 :
                # close data set, save it to file

                cal_df3 = pd.concat(cal_df3_list)

                cal_df3["attLO"] = pd.to_numeric(cal_df3["attLO"])
                cal_df3["attRF"] = pd.to_numeric(cal_df3["attRF"])
                convert2category(cal_df3, cols = ["date", "rep", "iter", "freq", "cal_type", "digital_unit", "freq", "samp_rate", "nsamples", "obs"])

                if correction != 1:
                    # if there is a correction or conversion factor, saves the data in converted form as well

                    if correction == np.around(1.0e3/8192,4):
                        cal_df3.insert(0, "voltage_unit", "mV")
                    elif correction == np.around(1.0/8192,7):

                        cal_df3.insert(0, "voltage_unit", "V")
                    else:
                        cal_df3.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                    cal_df3["voltage_ch1"] = cal_df3["raw_digital_ch1"]*correction
                    cal_df3["voltage_ch2"] = cal_df3["raw_digital_ch2"]*correction
                    cal_df3["voltage_mag"] = np.sqrt(np.square(cal_df3["voltage_ch1"]) + np.square(cal_df3["voltage_ch2"]))
                    cal_df3["voltage_phase"] = np.arctan2(cal_df3["voltage_ch1"], cal_df3["voltage_ch2"])

                    convert2category(cal_df3, cols = ["voltage_unit"])

                    base_path = main_path + date_path + processed_path + cal_path + conv_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))
                    file_title = " ".join((date_path.replace( "/",""), "Calibration Type 3 V", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                else:

                    base_path = main_path + date_path + processed_path + cal_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))
                    file_title = " ".join((date_path.replace( "/",""), "Calibration Type 3 D", "Rep", str(rep[x-1]), "Iter", str(ite[x-1])))

                if save_format.casefold() == "parquet":
                    if parquet_engine == 'pyarrow':
                        cal_df3.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, index= False)
                    else:
                        cal_df3.to_parquet(base_path + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                else:
                    cal_df3.to_csv(base_path + file_title + ".csv")
                tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")
                tqdm.write("".join(("\r ",date_path," Calibration Data processing finished!                                                                                       ")))

                del cal_df3_list[:]
                del file_title, base_path, cal_df3

def cal_data_pd_compile(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", 
                    processed_path = "Processed/DF/", conv_path = "Conv/", file_format="parquet", parquet_engine= 'pyarrow', is_recursive=False):
    """Compiles Pandas DataFrame "calibration files" into single parquet file for each calibration type (1-3).

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        sub-folder with calibration files, by default "Calibration/"
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/DF"
        final location will be main_path + processed_path + cal_path)
    conv_path : str, optional
        sub-folder for files of converted data, by default "Conv/"
    file_format: str, optional
        file format for dataframes, by default "parquet"
    is_recursive: bool, optional
        boolean for recursive folder sweep, by default "False"
    """

    path_list = list_paths(date, main_path = main_path, cal_path = cal_path, processed_path = processed_path, conv_path = conv_path)

    df1 = dd.concat(dd_collect(path_list, is_recursive, file_format, check_key="cal_type", check_value=1), axis=0).compute().reset_index(drop=True)
    df2 = dd.concat(dd_collect(path_list, is_recursive, file_format, check_key="cal_type", check_value=2), axis=0).compute().reset_index(drop=True)
    df3 = dd.concat(dd_collect(path_list, is_recursive, file_format, check_key="cal_type", check_value=3), axis=0).compute().reset_index(drop=True)

    df2["attLO"] = pd.to_numeric(df2["attLO"])
    df3["attLO"] = pd.to_numeric(df3["attLO"])
    df3["attRF"] = pd.to_numeric(df3["attRF"])

    convert2category(df1, cols = ["date", "rep", "iter", "cal_type", "digital_unit", "voltage_unit", "attLO", "attRF", "freq", "samp_rate", "nsamples", "obs"])
    convert2category(df2, cols = ["date", "rep", "iter", "cal_type", "freq", "digital_unit", "voltage_unit", "attRF", "freq", "samp_rate", "nsamples", "obs"])
    convert2category(df3, cols = ["date", "rep", "iter", "cal_type", "freq", "digital_unit", "voltage_unit", "freq", "samp_rate", "nsamples", "obs"])

    if isinstance(date, list):
        out_path= "{0}/OneDrive - McGill University/Narrow Band Data1/Analysis/{1}/DF/Calibration/Comp/{1} Compiled Calibration Type NUM.parquet".format(os.environ['USERPROFILE'], datetime.now().strftime("%Y_%m_%d"))
    else:
        out_path= "".join((main_path,date,"/",processed_path,cal_path,"Comp/{} Compiled Calibration Type NUM.parquet".format(date)))

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if parquet_engine == 'pyarrow':
        df1.to_parquet(out_path.replace("NUM","1"), engine=parquet_engine, index= False)
    else:
        df1.to_parquet(out_path.replace("NUM","1"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path.replace("NUM","1")}        ')

    if parquet_engine == 'pyarrow':
        df2.to_parquet(out_path.replace("NUM","2"), engine=parquet_engine, index= False)
    else:
        df2.to_parquet(out_path.replace("NUM","2"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path.replace("NUM","2")}        ')

    if parquet_engine == 'pyarrow':
        df3.to_parquet(out_path.replace("NUM","3"), engine=parquet_engine, index= False)
    else:
        df3.to_parquet(out_path.replace("NUM","3"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path.replace("NUM","3")}        ')

def cal_data_pd_agg(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", 
                    processed_path = "Processed/DF/", correction = np.around(1.0e3/8192,4), conv_path = "Converted/", decimals = 4,
                    save_format="parquet", parquet_engine= 'pyarrow'):
    """Generates aggregated Pandas DataFrame "calibration files" for each calibration type.

    First, sweeps trough main_path + date_path + cal_path folders and lists all .adc files found.
    Then, parameters are extracted from types 1, 2 and 3 calibration files.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        sub-folder with calibration files, by default "Calibration/"
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/"
        final location will be main_path + processed_path + cal_path)
    confidence : float, optional
        tolerance for confidence interval estimation, by default 0.95
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    conv_path : str, optional
        sub-folder for JSON files of converted data (if correction != 1), by default "Converted/"
    decimals : int, optional
        number of decimals cases for np.arounding values, in particular after conversion, by default 2
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    """

    if isinstance(date, list):
        out_path1= "{0}/OneDrive - McGill University/Narrow Band Data1/Analysis/{1}/DF/Calibration/Means/Calibration Processed Means Type NUM.parquet".format(os.environ['USERPROFILE'], datetime.now().strftime("%Y_%m_%d"))
        out_path2= "{0}/OneDrive - McGill University/Narrow Band Data1/Analysis/{1}/DF/Calibration/Means Agg/Calibration Processed Agg Means Type NUM.parquet".format(os.environ['USERPROFILE'], datetime.now().strftime("%Y_%m_%d"))
    else:
        out_path1= "".join((main_path,date,"/",processed_path,cal_path,"Means/{} Calibration Processed Means Type NUM.parquet".format(date)))
        out_path2= "".join((main_path,date,"/",processed_path,cal_path,"Means Agg/{} Calibration Processed Agg Means Type NUM.parquet".format(date)))

    main_path = "".join((main_path,date,"/",processed_path,cal_path,"Comp/",date," Compiled Calibration Type NUM.parquet"))

    df = dd.read_parquet(main_path.replace("NUM","1"), engine=parquet_engine, dtypes={
                    "voltage_unit": "category", "digital_unit": "category", "obs": "category", "samp_rate": "category", "nsamples": "category", "iter": "category", "rep": "category", "date": "category",
                    "cal_type": "category", "attRF": "category", "attLO": "category", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "category",
                    "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                    })
    df = df.compute()
    df1 = df.groupby(["date", "attLO", "attRF", "samp_rate", "nsamples","cal_type", "digital_unit", "voltage_unit", "obs", "rep", "freq"], observed=True).agg(digital_ch1 = ("raw_digital_ch1", "mean"),
                    digital_ch2 = ("raw_digital_ch2","mean"), std_digital_ch1 =("raw_digital_ch1", "std"), std_digital_ch2 = ("raw_digital_ch2", "std"))
    df1 = df1.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)

    df1 = df1.round({"digital_ch1": 0, "digital_ch2": 0})
    df1["voltage_ch1"] = correction*df1["digital_ch1"]
    df1["voltage_ch2"] = correction*df1["digital_ch2"]
    df1["voltage_mag"] = np.sqrt(np.square(df1["voltage_ch1"]) + np.square(df1["voltage_ch2"]))
    df1["voltage_phase"] = np.arctan2(df1["voltage_ch1"], df1["voltage_ch2"])

    convert2category(df1, cols = ["date", "rep", "cal_type", "digital_unit", "voltage_unit", "attLO", "attRF", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path1)):
        os.makedirs(os.path.dirname(out_path1))
    if parquet_engine == 'pyarrow':
        df1.to_parquet(out_path1.replace("NUM","1"), engine=parquet_engine, index= False)
    else:
        df1.to_parquet(out_path1.replace("NUM","1"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path1.replace("NUM","1")}        ')

    df1 = df.groupby(["date", "attLO", "attRF", "samp_rate", "nsamples","cal_type", "digital_unit", "voltage_unit", "obs", "freq"], observed=True).agg(digital_ch1 = ("raw_digital_ch1", "mean"),
                    digital_ch2 = ("raw_digital_ch2","mean"), std_digital_ch1 =("raw_digital_ch1", "std"), std_digital_ch2 = ("raw_digital_ch2", "std"))
    df1 = df1.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)

    df1 = df1.round({"digital_ch1": 0, "digital_ch2": 0})
    df1["voltage_ch1"] = correction*df1["digital_ch1"]
    df1["voltage_ch2"] = correction*df1["digital_ch2"]
    df1["voltage_mag"] = np.sqrt(np.square(df1["voltage_ch1"]) + np.square(df1["voltage_ch2"]))
    df1["voltage_phase"] = np.arctan2(df1["voltage_ch1"], df1["voltage_ch2"])

    convert2category(df1, cols = ["date", "cal_type", "freq", "digital_unit", "voltage_unit"])

    if not os.path.exists(os.path.dirname(out_path2)):
        os.makedirs(os.path.dirname(out_path2))
    if parquet_engine == 'pyarrow':
        df1.to_parquet(out_path2.replace("NUM","1"), engine=parquet_engine, index= False)
    else:
        df1.to_parquet(out_path2.replace("NUM","1"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path2.replace("NUM","1")}        ')

    df["subject"] = df["date"].astype(str) + " Rep " + df["rep"].astype(str) + " Iter " + df["iter"].astype(str)

    df = dd.read_parquet(main_path.replace("NUM","2"), engine=parquet_engine, dtypes={
                        "voltage_unit": "category", "digital_unit": "category", "obs": str, "samp_rate": "category", "nsamples": "category", "iter": "category", "rep": "category", "date": str,
                        "cal_type": "category", "attRF": "category", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "category",
                        "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                        })
    df = df.compute()

    df21 = df.groupby(["date", "attLO", "attRF", "samp_rate", "nsamples","cal_type", "digital_unit", "voltage_unit", "obs", "rep", "freq"], observed=True).agg(digital_ch1 = ("raw_digital_ch1", "mean"),
                        digital_ch2 = ("raw_digital_ch2","mean"), std_digital_ch1 =("raw_digital_ch1", "std"), std_digital_ch2 = ("raw_digital_ch2", "std"))
    df21 = df21.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)

    df21 = df21.round({"digital_ch1": 0, "digital_ch2": 0})
    df21["voltage_ch1"] = correction*df21["digital_ch1"]
    df21["voltage_ch2"] = correction*df21["digital_ch2"]
    df21["voltage_mag"] = np.sqrt(np.square(df21["voltage_ch1"]) + np.square(df21["voltage_ch2"]))
    df21["voltage_phase"] = np.arctan2(df21["voltage_ch1"], df21["voltage_ch2"])

    convert2category(df21, cols = ["date", "rep", "cal_type", "freq", "digital_unit", "voltage_unit", "attRF", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path1)):
        os.makedirs(os.path.dirname(out_path1))
    if parquet_engine == 'pyarrow':
        df21.to_parquet(out_path1.replace("NUM","2"), engine=parquet_engine, index= False)
    else:
        df21.to_parquet(out_path1.replace("NUM","2"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path1.replace("NUM","2")}        ')

    df22 = df.groupby(["date", "attLO", "attRF", "samp_rate", "nsamples","cal_type", "digital_unit", "voltage_unit", "obs", "freq"], observed=True).agg(digital_ch1 = ("raw_digital_ch1", "mean"),
                        digital_ch2 = ("raw_digital_ch2","mean"), std_digital_ch1 =("raw_digital_ch1", "std"), std_digital_ch2 = ("raw_digital_ch2", "std"))
    df22 = df22.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)

    df22 = df22.round({"digital_ch1": 0, "digital_ch2": 0})
    df22["voltage_ch1"] = correction*df22["digital_ch1"]
    df22["voltage_ch2"] = correction*df22["digital_ch2"]
    df22["voltage_mag"] = np.sqrt(np.square(df22["voltage_ch1"]) + np.square(df22["voltage_ch2"]))
    df22["voltage_phase"] = np.arctan2(df22["voltage_ch1"], df22["voltage_ch2"])

    convert2category(df22, cols = ["date", "cal_type", "freq", "digital_unit", "voltage_unit", "attRF", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path2)):
        os.makedirs(os.path.dirname(out_path2))
    if parquet_engine == 'pyarrow':
        df22.to_parquet(out_path2.replace("NUM","2"), engine=parquet_engine, index= False)
    else:
        df22.to_parquet(out_path2.replace("NUM","2"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path2.replace("NUM","2")}        ')

    df = dd.read_parquet(main_path.replace("NUM","3"), engine=parquet_engine, dtypes={
                        "voltage_unit": "category", "digital_unit": "category", "obs": "category", "samp_rate": "category", "nsamples": "category", "iter": "category", "rep": "category", "date": "category",
                        "cal_type": "category", "attRF": "int32", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "category",
                        "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                        })
    df = df.compute().reset_index(drop=True)

    df3 = df.groupby(["date", "attLO", "attRF", "samp_rate", "nsamples","cal_type", "digital_unit", "voltage_unit", "obs", "rep", "freq"], observed=True).agg(digital_ch1 = ("raw_digital_ch1", "mean"), 
                    digital_ch2 = ("raw_digital_ch2","mean"), std_digital_ch1 =("raw_digital_ch1", "std"), std_digital_ch2 = ("raw_digital_ch2", "std"))
    df3 = df3.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)

    df3["attLO"] = pd.to_numeric(df3["attLO"])
    df3["attRF"] = pd.to_numeric(df3["attRF"])

    df21 = df21.set_index(["date","attLO", "nsamples", "rep", "freq"])
    df3 = df3.set_index(["date","attLO", "nsamples", "rep", "freq"])

    df3["c_digital_ch1"] = df3["digital_ch1"] - df21["digital_ch1"]
    df3["c_digital_ch2"] = df3["digital_ch2"] - df21["digital_ch2"]

    df3 = df3.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)
    df21 = df21.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)

    for d in df3.date.drop_duplicates().tolist():
        if df3.loc[df3.date.eq(d), "rep"].astype(int).max() > df21.loc[df21.date.eq(d), "rep"].astype(int).max():
            extra_rep = df3.loc[df3.date.eq(d), "rep"].astype(int).max()
            df3.loc[df3.date.eq(d) & df3.rep.eq(extra_rep), "c_digital_ch1"] = df3.loc[df3.date.eq(d) & df3.rep.eq(extra_rep), "digital_ch1"] - df21.loc[df3.date.eq(d) & df21.rep.eq(extra_rep - 1), "digital_ch1"].values
            df3.loc[df3.date.eq(d) & df3.rep.eq(extra_rep), "c_digital_ch2"] = df3.loc[df3.date.eq(d) & df3.rep.eq(extra_rep), "digital_ch2"] - df21.loc[df3.date.eq(d) & df21.rep.eq(extra_rep - 1), "digital_ch2"].values

    df3 = df3.round({"digital_ch1": 0, "digital_ch2": 0, "c_digital_ch1": 0, "c_digital_ch2": 0})
    df3["voltage_ch1"] = correction*df3["c_digital_ch1"]
    df3["voltage_ch2"] = correction*df3["c_digital_ch2"]
    df3["voltage_mag"] = np.sqrt(np.square(df3["voltage_ch1"]) + np.square(df3["voltage_ch2"]))
    df3["voltage_phase"] = np.arctan2(df3["voltage_ch1"], df3["voltage_ch2"])
    df3 = df3.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)
    df21 = df21.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)

    convert2category(df3, cols = ["date", "rep", "cal_type", "freq", "digital_unit", "voltage_unit", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path1)):
        os.makedirs(os.path.dirname(out_path1))
    if parquet_engine == 'pyarrow':
        df3.to_parquet(out_path1.replace("NUM","3"), engine=parquet_engine, index= False)
    else:
        df3.to_parquet(out_path1.replace("NUM","3"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path1.replace("NUM","3")}        ')

    df_ref = df3.groupby(["attLO", "attRF", "freq"], observed=True).agg(digital_ch1 = ("digital_ch1", "mean"), digital_ch2 = ("digital_ch2","mean"), 
                    std_digital_ch1 = ("digital_ch1", "std"), std_digital_ch2 = ("digital_ch2", "std"), c_digital_ch1 = ("c_digital_ch1", "mean"), 
                    c_digital_ch2 = ("c_digital_ch2","mean"), std_c_digital_ch1 =("c_digital_ch1", "std"), std_c_digital_ch2 = ("c_digital_ch2", "std"))
    df_ref.reset_index()
    df_ref = df_ref.round({"digital_ch1": 0, "digital_ch2": 0, "c_digital_ch1": 0, "c_digital_ch2": 0})
    df_ref["voltage_ch1"] = correction*df_ref["c_digital_ch1"]
    df_ref["voltage_ch2"] = correction*df_ref["c_digital_ch2"]
    df_ref["voltage_mag"] = np.sqrt(np.square(df_ref["voltage_ch1"]) + np.square(df_ref["voltage_ch2"]))
    df_ref["voltage_phase"] = np.arctan2(df_ref["voltage_ch1"], df_ref["voltage_ch2"])
    df_ref = df_ref.reset_index().sort_values(by=["attLO", "attRF", "freq"], ignore_index=True)

    convert2category(df_ref, cols = ["freq"])

    if not os.path.exists(os.path.dirname(out_path1)):
        os.makedirs(os.path.dirname(out_path1))
    if parquet_engine == 'pyarrow':
        df_ref.to_parquet(out_path1.replace("NUM","3 Ref"), engine=parquet_engine, index= False)
    else:
        df_ref.to_parquet(out_path1.replace("NUM","3 Ref"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path1.replace("NUM","3 Ref")}        ')

    df4 = df3.drop(columns= ["digital_ch1", "digital_ch2", "voltage_ch1", "voltage_ch2", "voltage_mag", "voltage_phase"])

    for d in df3["date"].drop_duplicates().tolist():
        for rep in df3.loc[df3["date"] == d, "rep"].drop_duplicates().tolist():
            for attLO in df3.loc[(df3["date"] == d) & (df3["rep"] == rep), "attLO"].drop_duplicates().tolist():
                for attRF in df3.loc[(df3["date"] == d) & (df3["rep"] == rep) & (df3["attLO"] == attLO), "attRF"].drop_duplicates().tolist():

                    if attRF in df_ref.loc[df_ref.attLO.eq(attLO), "attRF"].values:
                        attRF_ref = attRF
                    elif attLO == 20:
                        attRF_ref = 25
                    else:
                        attRF_ref = df_ref.loc[df_ref.attLO.eq(attLO), "attRF"].iloc[0]

                    to_subtract = df3.loc[(df3["date"] == d) & (df3["rep"] == rep) & (df3["attLO"] == attLO) & (df3["attRF"] == attRF)].drop(columns= ["digital_ch1", "digital_ch2", "voltage_ch1", "voltage_ch2", "voltage_mag", "voltage_phase"])
                    freqs = to_subtract.freq.drop_duplicates().tolist()

                    df4.loc[(df4["date"] == d) & (df4["rep"] == rep) & (df3["attLO"] == attLO) & (df3["attRF"] == attRF), "offset_ch1"] = dB2Volts(attRF - attRF_ref) * to_subtract["c_digital_ch1"].values - df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch1"].values
                    df4.loc[(df4["date"] == d) & (df4["rep"] == rep) & (df3["attLO"] == attLO) & (df3["attRF"] == attRF), "offset_ch2"] = dB2Volts(attRF - attRF_ref) * to_subtract["c_digital_ch2"].values - df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch2"].values
                    df4.loc[(df4["date"] == d) & (df4["rep"] == rep) & (df3["attLO"] == attLO) & (df3["attRF"] == attRF), "attRF"] = attRF_ref

                    df4.loc[(df4["date"] == d) & (df4["rep"] == rep), "multiplier_ch1"] = dB2Volts(attRF - attRF_ref)*to_subtract["c_digital_ch1"] / df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch1"].values
                    df4.loc[(df4["date"] == d) & (df4["rep"] == rep), "multiplier_ch2"] = dB2Volts(attRF - attRF_ref)*to_subtract["c_digital_ch2"] / df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch2"].values


    df4 = df4.round({"offset_ch1": 0, "offset_ch2": 0})
    df4["voltage_ch1"] = correction*df4["offset_ch1"]
    df4["voltage_ch2"] = correction*df4["offset_ch2"]
    df4["voltage_mag"] = np.sqrt(np.square(df4["voltage_ch1"]) + np.square(df4["voltage_ch2"]))
    df4["voltage_phase"] = np.arctan2(df4["voltage_ch1"], df4["voltage_ch2"])

    df3 = df3.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)
    df4 = df4.reset_index().sort_values(by=["date", "rep", "freq"], ignore_index=True)

    del df4["level_0"]
    del df4["index"]

    convert2category(df4, cols = ["date", "rep", "cal_type", "freq", "digital_unit", "voltage_unit", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path1)):
        os.makedirs(os.path.dirname(out_path1))
    if parquet_engine == 'pyarrow':
        df4.to_parquet(out_path1.replace("NUM","3 Offsets"), engine=parquet_engine, index= False)
    else:
        df4.to_parquet(out_path1.replace("NUM","3 Offsets"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path1.replace("NUM","3 Offsets")}        ')

    del df4

    df3 = df.groupby(["date", "attLO", "attRF", "samp_rate", "nsamples","cal_type", "digital_unit", "voltage_unit", "obs", "freq"], observed=True).agg(digital_ch1 = ("raw_digital_ch1", "mean"), 
                    digital_ch2 = ("raw_digital_ch2","mean"), std_digital_ch1 =("raw_digital_ch1", "std"), std_digital_ch2 = ("raw_digital_ch2", "std"))
    df3 = df3.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)

    df22 = df22.set_index(["date","attLO", "nsamples", "freq"])
    df3 = df3.set_index(["date","attLO", "nsamples", "freq", "attRF"])

    df3["c_digital_ch1"] = df3["digital_ch1"] - df22["digital_ch1"]
    df3["c_digital_ch2"] = df3["digital_ch2"] - df22["digital_ch2"]

    df3 = df3.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)
    df22 = df22.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)

    df3 = df3.round({"digital_ch1": 0, "digital_ch2": 0, "c_digital_ch1": 0, "c_digital_ch2": 0})
    df3["voltage_ch1"] = correction*df3["c_digital_ch1"]
    df3["voltage_ch2"] = correction*df3["c_digital_ch2"]
    df3["voltage_mag"] = np.sqrt(np.square(df3["voltage_ch1"]) + np.square(df3["voltage_ch2"]))
    df3["voltage_phase"] = np.arctan2(df3["voltage_ch1"], df3["voltage_ch2"])

    convert2category(df3, cols = ["date", "cal_type", "freq", "digital_unit", "voltage_unit", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path2)):
        os.makedirs(os.path.dirname(out_path2))
    if parquet_engine == 'pyarrow':
        df3.to_parquet(out_path2.replace("NUM","3"), engine=parquet_engine, index= False)
    else:
        df3.to_parquet(out_path2.replace("NUM","3"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path2.replace("NUM","3")}        ')

    convert2category(df_ref, cols = ["freq"])

    if not os.path.exists(os.path.dirname(out_path2)):
        os.makedirs(os.path.dirname(out_path2))
    if parquet_engine == 'pyarrow':
        df_ref.to_parquet(out_path2.replace("NUM","3 Ref"), engine=parquet_engine, index= False)
    else:
        df_ref.to_parquet(out_path2.replace("NUM","3 Ref"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path2.replace("NUM","3 Ref")}        ')

    df4 = df3.drop(columns= ["digital_ch1", "digital_ch2", "voltage_ch1", "voltage_ch2", "voltage_mag", "voltage_phase"])

    for d in df3["date"].drop_duplicates().tolist():
        for attLO in df3.loc[df3["date"] == d, "attLO"].drop_duplicates().tolist():
            for attRF in df3.loc[(df3["date"] == d) & (df3["attLO"] == attLO), "attRF"].drop_duplicates().tolist():

                if attRF in df_ref.loc[df_ref.attLO.eq(attLO), "attRF"].values:
                    attRF_ref = attRF
                elif attLO == 20:
                    attRF_ref = 25
                else:
                    attRF_ref = df_ref.loc[df_ref.attLO.eq(attLO), "attRF"].iloc[0]

                to_subtract = df3.loc[(df3["date"] == d) & (df3["attLO"] == attLO) & (df3["attRF"] == attRF)].drop(columns= ["digital_ch1", "digital_ch2", "voltage_ch1", "voltage_ch2", "voltage_mag", "voltage_phase"])
                freqs = to_subtract.freq.drop_duplicates().tolist()
                df4.loc[(df4["date"] == d) & (df4["attLO"] == attLO) & (df4["attRF"] == attRF), "offset_ch1"] = dB2Volts(attRF - attRF_ref) * to_subtract["c_digital_ch1"].values - df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch1"].values
                df4.loc[(df4["date"] == d) & (df4["attLO"] == attLO) & (df4["attRF"] == attRF), "offset_ch2"] = dB2Volts(attRF - attRF_ref) * to_subtract["c_digital_ch2"].values - df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch2"].values
                df4.loc[(df4["date"] == d) & (df4["attLO"] == attLO) & (df4["attRF"] == attRF), "attRF"] = attRF_ref

                df4.loc[(df4["date"] == d) & (df4["attLO"] == attLO) & (df4["attRF"] == attRF), "multiplier_ch1"] = dB2Volts(attRF - attRF_ref) * to_subtract["c_digital_ch1"].values / df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch1"].values
                df4.loc[(df4["date"] == d) & (df4["attLO"] == attLO) & (df4["attRF"] == attRF), "multiplier_ch2"] = dB2Volts(attRF - attRF_ref) * to_subtract["c_digital_ch2"].values / df_ref.loc[(df_ref.attLO.eq(attLO)) & (df_ref.attRF.eq(attRF)) & (df_ref.freq.isin(freqs)), "c_digital_ch2"].values

    df4 = df4.round({"offset_ch1": 0, "offset_ch2": 0})

    df4["voltage_ch1"] = correction*df4["offset_ch1"]
    df4["voltage_ch2"] = correction*df4["offset_ch2"]
    df4["voltage_mag"] = np.sqrt(np.square(df4["voltage_ch1"]) + np.square(df4["voltage_ch2"]))
    df4["voltage_phase"] = np.arctan2(df4["voltage_ch1"], df4["voltage_ch2"])

    df3 = df3.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)
    df4 = df4.reset_index().sort_values(by=["date", "attLO", "attRF", "freq"], ignore_index=True)

    convert2category(df4, cols = ["date", "cal_type", "freq", "digital_unit", "voltage_unit", "freq", "samp_rate", "nsamples", "obs"])

    if not os.path.exists(os.path.dirname(out_path2)):
        os.makedirs(os.path.dirname(out_path2))
    if parquet_engine == 'pyarrow':
        df4.to_parquet(out_path2.replace("NUM","3 Offsets"), engine=parquet_engine, index= False)
    else:
        df4.to_parquet(out_path2.replace("NUM","3 Offsets"), engine=parquet_engine, object_encoding='utf8', write_index= False)
    tqdm.write(f'\nSaved file: {out_path2.replace("NUM","3 Offsets")}        ')

def cal4_data_read2pandas(dates, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", 
                    processed_path = "Processed/DF/", correction = np.around(1.0e3/8192,4), conv_path = "Converted/", decimals = 4,
                    save_format="parquet", parquet_engine= 'pyarrow'):
    """Generate Pandas DataFrame "calibration file" for calibration type 4 from PScope .adc files.

    This data processing step should be performed after cal_data_pd_agg().

    Uses case_data_read2pandas() with custom inputs.

    Parameters
    ----------
    dates : str or list of str
        date(s) in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        sub-folder with calibration files, by default "Calibration/"
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/DF/"
        final location will be main_path + processed_path + cal_path
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    conv_path : str, optional
        sub-folder for JSON files of converted data (if correction != 1), by default "Converted/"
    decimals : int, optional
        number of decimals cases for np.arounding values, in particular after conversion, by default 2
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    parquet_engine: str, optional
        Parquet reader library to use, by default 'pyarrow'
        Options include: ‘auto’, ‘fastparquet’, ‘pyarrow’.
        If ‘auto’, then the option io.parquet.engine is used.
        The default io.parquet.engine behavior is to try ‘pyarrow’,
        falling back to ‘fastparquet’ if ‘pyarrow’ is unavailable.
    """

    if not isinstance(dates, list):
        dates = [dates]

    extra_type = "Type 4/"

    for date in tqdm(dates):

        date_path = _date2path(date)

        base_path4 = "".join((main_path, date_path, cal_path, extra_type))
        if os.path.exists(os.path.dirname(base_path4)):

            # Calibration type 4 - uses case_data_read2pandas() with specific options

            case_data_read2pandas(dates, main_path = main_path, sub_folder1 = "".join((cal_path, extra_type)),
                        processed_path = "".join((processed_path, cal_path)), correction = correction, conv_path = conv_path,
                        decimals = decimals, save_format=save_format, parquet_engine=parquet_engine, cal_option = 4)

        else:
            tqdm.write(f'No Calibration Type 4 files found for {date}!', end='\n')

def cal4_data_pd_agg(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", 
                    processed_path = "Processed/DF/", correction = np.around(1.0e3/8192,4), conv_path = "Converted/", decimals = 4,
                    save_format="parquet", parquet_engine= 'pyarrow'):
    """Generates aggregated Pandas DataFrame "calibration files" for calibration type 4.

    Uses data_pd_agg() with custom inputs.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        sub-folder with calibration files, by default "Calibration/"
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/"
        final location will be main_path + processed_path + cal_path)
    confidence : float, optional
        tolerance for confidence interval estimation, by default 0.95
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    conv_path : str, optional
        sub-folder for JSON files of converted data (if correction != 1), by default "Converted/"
    decimals : int, optional
        number of decimals cases for np.arounding values, in particular after conversion, by default 2
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    parquet_engine: str, optional
        Parquet reader library to use, by default 'pyarrow'
        Options include: ‘auto’, ‘fastparquet’, ‘pyarrow’.
        If ‘auto’, then the option io.parquet.engine is used.
        The default io.parquet.engine behavior is to try ‘pyarrow’,
        falling back to ‘fastparquet’ if ‘pyarrow’ is unavailable.
    """

    # Calibration type 4 - uses data_pd_agg()

    data_pd_agg(date, main_path = main_path, sub_folder = cal_path, 
                    processed_path = processed_path, correction = correction, conv_path = conv_path, decimals = decimals,
                    save_format=save_format, parquet_engine=parquet_engine, is_recursive= False, cal_option = 4)

def case_data_read2pandas(dates, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), sub_folder1 = "", sub_folder2 = "",
                    processed_path = "Processed/DF/", correction = np.around(1.0e3/8192,4), conv_path = "Converted/",
                     decimals = 4, save_format="parquet", parquet_engine= 'pyarrow', cal_option = 0):
    """Generate Pandas DataFrame "scan data set" files from PScope .adc files, in both digital scale and, optionally, voltages.

    This version includes values for magnitude and phase of the recorded means.

    First, sweeps trough main_path + date_path + phantom folders and lists all .adc files found. Then parameters are extracted and calibrated
    using calibration type 2 data (RF grounded and LO with input frequency).

    Parameters
    ----------
    dates : str, list of str
        date(s) in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    sub_folder2 : str, optional
        sub-folder for file sweep (after date path), by default ""
        searched location will be main_path + sub_folder1 + date_path + "Phantom*/" + sub_folder2
    sub_folder2 : str, optional
        sub-folder for file sweep (after phantom info), by default ""
        searched location will be main_path + sub_folder1 + date_path + "Phantom*/" + sub_folder2
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
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    cal_option: int, optional
        if different than 0, appends "Calibration Type ##" to start of filenames, by default 0
        (for use with calibration type 4)
    """

    if not isinstance(dates, list):
        dates = [dates]

    for date in tqdm(dates):

        date_path = _date2path(date)

        paths, meta_index, meta_freq = phantom_folder_sweep(date = "".join((date_path, sub_folder1)) , main_path = main_path, ph_path = "".join(("Phantom*/" , sub_folder2)) )

        # collects measurement configuration parameters from JSON files

        if cal_option == 0:
            ConfigSet = collect_json_data(date = date_path, main_path= main_path, sub_folder = "Config", json_type = "measurement configuration parameters", 
                                                    is_recursive= False)
        else:
            ConfigSet1 = collect_json_data(date = date_path, main_path= main_path, sub_folder = "Config", json_type = "calibration configuration parameters", 
                                                    is_recursive= False)
            ConfigSet = []

            for conf in ConfigSet1:
                if conf["cal_type"] == cal_option:
                    ConfigSet.append(conf)

        phantom = meta_index[:,0]
        angle = meta_index[:,1]
        plug = meta_index[:,2]
        rep = meta_index[:,6]
        ite = meta_index[:,5]
        pair = np.array([(meta_index[i,3],meta_index[i,4]) for i in np.arange(meta_index.shape[0])])

        freq = np.asarray([i.replace('_','.') for i in meta_freq], dtype = float)

        phantoms = np.unique(phantom)
        angles = np.unique(angle)
        plugs = np.unique(plug)
        reps = np.unique(rep)
        iters = np.unique(ite)
        pairs_uniq = np.unique(pair, axis=0)

        frequencies = np.unique(freq)

        # Uses calibration type 2 data for offset removal (in digital scale). Currently assumes Rep 1 for calibration.

        if cal_option == 0:
            cal_mean_2 = calibration_mean_dataframe(date = date_path, main_path = main_path, processed_path = processed_path, cal_type = 2)
            cal_mean_3 = calibration_mean_dataframe(date = date_path, main_path = main_path, processed_path = processed_path, cal_type = 3)
        else:
            cal_mean_2 = calibration_mean_dataframe(date = date_path, main_path = main_path, processed_path = processed_path, cal_path = "Means Agg/", cal_type = 2)
            cal_mean_3 = calibration_mean_dataframe(date = date_path, main_path = main_path, processed_path = processed_path, cal_path = "Means Agg/", cal_type = 3)

        data = np.empty((0,2))
        time = np.empty((0,1))

        for x, i in enumerate(tqdm(paths, leave=False)):

            data, time , nsamples, srate = nbd.data_read(i)

            if x == 0:

                attLO, attRF, obs, freqs, pairs = check_configuration_parameters_case(ConfigSet = ConfigSet, date = date_path, phantom = phantom[x], angle = angle[x], plug = plug[x], rep = rep[x], ite = ite[x])

                df_list = [pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"])]
                df_list[-1]["time"] = time
                df_list[-1]["freq"] = freq[x]
                df_list[-1]["Tx"] = pair[x,0]
                df_list[-1]["Rx"] = pair[x,1]
                df_list[-1].insert(0,"attLO", attLO)
                df_list[-1].insert(0,"attRF", attRF)
                df_list[-1].insert(0,"phantom", phantom[x])
                df_list[-1].insert(0,"angle", angle[x])
                df_list[-1].insert(0,"plug", plug[x])
                df_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                df_list[-1].insert(0, "rep", rep[x])
                df_list[-1].insert(0, "iter", ite[x])
                df_list[-1].insert(0, "nsamples", nsamples)
                df_list[-1].insert(0, "samp_rate", srate*1e6)
                df_list[-1].insert(0, "obs", obs)
                df_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            if x != 0:
                if phantom[x] != phantom[x-1] or angle[x] != angle[x-1] or plug[x] != plug[x-1] or rep[x] != rep[x-1] or ite[x] != ite[x-1]:

                    # close data set, save it to file and start new data set dictionary

                    df = pd.concat(df_list)
                    df.set_index("freq", drop=False, append=True, inplace=True)

                    # Calibration type 2 offset subtraction (implicitly includes type 1)

                    cal_mean_2sub = cal_mean_2.loc[(cal_mean_2["date"] == df["date"].iloc[0]) & (cal_mean_2["attLO"] == df["attLO"].iloc[0])]
                    cal_mean_2sub.set_index(keys="freq", drop=True, append=False, inplace=True)
                    df["digital_ch1"] = df["raw_digital_ch1"].subtract(cal_mean_2sub["digital_ch1"], level="freq", fill_value=0)
                    df["digital_ch2"] = df["raw_digital_ch2"].subtract(cal_mean_2sub["digital_ch2"], level="freq", fill_value=0)
                    df = df.round({"digital_ch1": 0, "digital_ch2": 0})

                    df.reset_index(level="freq", inplace=True, drop=True)

                    # Tentative Calibration type 3 normalization

                    cal_mean_3sub = cal_mean_3.loc[(cal_mean_3["date"] == df["date"].iloc[0]) & (cal_mean_3["attLO"] == df["attLO"].iloc[0])]
                    attRF = dB2Volts(df["attRF"].iloc[0] - cal_mean_3sub["attRF"].iloc[0])

                    df.set_index(keys="freq", drop=False, append=True, inplace=True)
                    cal_mean_3sub.set_index(keys="freq", drop=True, append=False, inplace=True)

                    df["n_digital_ch1"] = df["digital_ch1"].divide(cal_mean_3sub["c_digital_ch1"], level="freq", fill_value=0)
                    df["n_digital_ch2"] = df["digital_ch2"].divide(cal_mean_3sub["c_digital_ch2"], level="freq", fill_value=0)

                    df.reset_index(level="freq", inplace=True, drop=True)
                    df.sort_values(by=["date", "attLO", "attRF","Tx","Rx","freq","time"], ignore_index=True)

                    # Some manipulation on dataframes

                    df["pair"] = "(" + df["Tx"].astype(str) + "," + df["Rx"].astype(str) + ")"
                    df["subject"] = [" ".join((a, "Phantom", b, "Angle", c, "Rep", d, "Iter", e))
                                        for a,b,c,d,e in zip(df.date.astype(str), df.phantom.astype(str), df.angle.astype(str), df.rep.astype(str), df.iter.astype(str))]
                    df["attLO"] = pd.to_numeric(df["attLO"])
                    df["attRF"] = pd.to_numeric(df["attRF"])
                    convert2category(df, cols = ["date", "rep", "iter", "phantom", "angle", "plug", "freq", "digital_unit", "pair", "subject", "freq", "samp_rate", "nsamples", "obs"])

                    if correction != 1:
                            # if there is a correction or conversion factor, saves the data in converted form as well
                        if correction == np.around(1.0e3/8192,4):
                            df.insert(0, "voltage_unit", "mV")
                        elif correction == np.around(1.0/8192,7):
                            df.insert(0, "voltage_unit", "V")
                        else:
                            df.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                        df["voltage_ch1"] = df["digital_ch1"]*correction
                        df["voltage_ch2"] = df["digital_ch2"]*correction
                        df["voltage_mag"] = np.sqrt(np.square(df["voltage_ch1"]) + np.square(df["voltage_ch2"]))
                        df["voltage_phase"] = np.arctan2(df["voltage_ch1"], df["voltage_ch2"])

                        df["n_voltage_ch1"] = attRF*df["n_digital_ch1"]*correction
                        df["n_voltage_ch2"] = attRF*df["n_digital_ch2"]*correction
                        df["n_voltage_mag"] = np.sqrt(np.square(df["n_voltage_ch1"]) + np.square(df["n_voltage_ch2"]))
                        df["n_voltage_phase"] = np.arctan2(df["n_voltage_ch1"], df["n_voltage_ch2"])

                        convert2category(df, cols = ["voltage_unit"])

                        base_path = main_path + date_path + processed_path + conv_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))

                        if cal_option == 0:
                            file_title = "Phantom {0:} Angle {1:} Plug {2:} Rep {3:} Iter {4:} V".format(phantom[x-1], angle[x-1], plug[x-1], rep[x-1], ite[x-1])
                        else:
                            file_title = "Calibration Type {0:} Phantom {1:} Angle {2:} Plug {3:} Rep {4:} Iter {5:} V".format(cal_option, phantom[x-1], angle[x-1], 
                                            plug[x-1], rep[x-1], ite[x-1])
                            df["cal_type"] = cal_option


                        if save_format.casefold() == "parquet":
                            if parquet_engine == 'pyarrow':
                                df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, index= False)
                            else:
                                df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                        else:
                            df.to_csv(base_path + date_path.replace( "/"," ") + file_title + ".csv")
                        tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                    else:
                        base_path = main_path + date_path + processed_path
                        if not os.path.exists(os.path.dirname(base_path)):
                            os.makedirs(os.path.dirname(base_path))

                        if cal_option == 0:
                            file_title = "Phantom {0:} Angle {1:} Plug {2:} Rep {3:} Iter {4:} D".format(phantom[x-1], angle[x-1], plug[x-1], rep[x-1], ite[x-1])
                        else:
                            file_title = "Calibration Type {0:} Phantom {1:} Angle {2:} Plug {3:} Rep {4:} Iter {5:} D".format(cal_option, phantom[x-1], angle[x-1], 
                                            plug[x-1], rep[x-1], ite[x-1])
                            df["cal_type"] = cal_option

                        if save_format.casefold() == "parquet":
                            if parquet_engine == 'pyarrow':
                                df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, index= False)
                            else:
                                df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                        else:
                            df.to_csv(base_path + date_path.replace( "/"," ") + file_title + ".csv")
                        tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                    if x == len(paths) - 1:
                        tqdm.write("".join(("\r ",date_path," Data processing finished!")))

                    del df_list[:]
                    del file_title, base_path

                    attLO, attRF, obs, freqs, pairs = check_configuration_parameters_case(ConfigSet = ConfigSet, date = date_path, phantom = phantom[x], angle = angle[x], plug = plug[x], rep = rep[x], ite = ite[x])

                df_list.append(pd.DataFrame(data, columns=["raw_digital_ch1","raw_digital_ch2"]))
                df_list[-1]["time"] = time
                df_list[-1]["freq"] = freq[x]
                df_list[-1]["Tx"] = pair[x,0]
                df_list[-1]["Rx"] = pair[x,1]
                df_list[-1].insert(0,"attLO", attLO)
                df_list[-1].insert(0,"attRF", attRF)
                df_list[-1].insert(0,"phantom", phantom[x])
                df_list[-1].insert(0,"angle", angle[x])
                df_list[-1].insert(0,"plug", plug[x])
                df_list[-1].insert(0, "date", date_path.replace( "/","").replace("_","-"))
                df_list[-1].insert(0, "rep", rep[x])
                df_list[-1].insert(0, "iter", ite[x])
                df_list[-1].insert(0, "nsamples", nsamples)
                df_list[-1].insert(0, "samp_rate", srate*1e6)
                df_list[-1].insert(0, "obs", obs)
                df_list[-1].insert(0, "digital_unit", "ADC digital scale [-8192,8192]")

            data = np.empty((0,2))
            time = np.empty((0,1))

            if x == len(paths)-1 :
                # close data set, save it to file and finish

                df = pd.concat(df_list)
                df.set_index("freq", drop=False, append=True, inplace=True)

                # Calibration type 2 offset subtraction (implicitly includes type 1)

                cal_mean_2sub = cal_mean_2.loc[(cal_mean_2["date"] == df["date"].iloc[0]) & (cal_mean_2["attLO"] == df["attLO"].iloc[0])]
                cal_mean_2sub.set_index(keys="freq", drop=True, append=False, inplace=True)

                df["digital_ch1"] = df["raw_digital_ch1"].subtract(cal_mean_2sub["digital_ch1"], level="freq", fill_value=0)
                df["digital_ch2"] = df["raw_digital_ch2"].subtract(cal_mean_2sub["digital_ch2"], level="freq", fill_value=0)
                df = df.round({"digital_ch1": 0, "digital_ch2": 0})

                df.reset_index(level="freq", drop=True, inplace=True)

                # Tentative Calibration type 3 normalization

                cal_mean_3sub = cal_mean_3.loc[(cal_mean_3["date"] == df["date"].iloc[0]) & (cal_mean_3["attLO"] == df["attLO"].iloc[0])]
                attRF = dB2Volts(df["attRF"].iloc[0] - cal_mean_3sub["attRF"].iloc[0])

                df.set_index(keys="freq", drop=False, append=True, inplace=True)
                cal_mean_3sub.set_index(keys="freq", drop=True, append=False, inplace=True)

                df["n_digital_ch1"] = df["digital_ch1"].divide(cal_mean_3sub["c_digital_ch1"], level="freq", fill_value=0)
                df["n_digital_ch2"] = df["digital_ch2"].divide(cal_mean_3sub["c_digital_ch2"], level="freq", fill_value=0)

                df.reset_index(level="freq", inplace=True, drop=True)
                df.sort_values(by=["date", "attLO", "attRF","Tx","Rx","freq","time"], ignore_index=True)

                # Some manipulation on dataframes

                df["pair"] = "(" + df["Tx"].astype(str) + "," + df["Rx"].astype(str) + ")"
                df["subject"] = [" ".join((a, "Phantom", b, "Angle", c, "Rep", d, "Iter", e))
                                        for a,b,c,d,e in zip(df.date.astype(str), df.phantom.astype(str), df.angle.astype(str), df.rep.astype(str), df.iter.astype(str))]
                df["attLO"] = pd.to_numeric(df["attLO"])
                df["attRF"] = pd.to_numeric(df["attRF"])
                convert2category(df, cols = ["date", "rep", "iter", "phantom", "angle", "plug", "freq", "digital_unit", "pair", "subject", "freq", "samp_rate", "nsamples", "obs"])

                if correction != 1:
                    # if there is a correction or conversion factor, saves the data in converted form as well

                    if correction == np.around(1.0e3/8192,4):
                        df.insert(0, "voltage_unit", "mV")
                    elif correction == np.around(1.0/8192,7):
                        df.insert(0, "voltage_unit", "V")
                    else:
                        df.insert(0, "voltage_unit", "converted by factor {}".format(correction))

                    df["voltage_ch1"] = df["digital_ch1"]*correction
                    df["voltage_ch2"] = df["digital_ch2"]*correction
                    df["voltage_mag"] = np.sqrt(np.square(df["voltage_ch1"]) + np.square(df["voltage_ch2"]))
                    df["voltage_phase"] = np.arctan2(df["voltage_ch1"], df["voltage_ch2"])

                    df["n_voltage_ch1"] = attRF*df["n_digital_ch1"]*correction
                    df["n_voltage_ch2"] = attRF*df["n_digital_ch2"]*correction
                    df["n_voltage_mag"] = np.sqrt(np.square(df["n_voltage_ch1"]) + np.square(df["n_voltage_ch2"]))
                    df["n_voltage_phase"] = np.arctan2(df["n_voltage_ch1"], df["n_voltage_ch2"])

                    convert2category(df, cols = ["voltage_unit"])

                    base_path = main_path + date_path + processed_path + conv_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))

                    if cal_option == 0:
                        file_title = "Phantom {0:} Angle {1:} Plug {2:} Rep {3:} Iter {4:} V".format(phantom[x-1], angle[x-1], plug[x-1], rep[x-1], ite[x-1])
                    else:
                        file_title = "Calibration Type {0:} Phantom {1:} Angle {2:} Plug {3:} Rep {4:} Iter {5:} V".format(cal_option, phantom[x-1], angle[x-1], 
                                        plug[x-1], rep[x-1], ite[x-1])
                        df["cal_type"] = cal_option

                    if save_format == "parquet":
                        if parquet_engine == 'pyarrow':
                            df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, index= False)
                        else:
                            df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                    else:
                        df.to_csv(base_path + date_path.replace( "/"," ") + file_title + ".csv")
                    tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")

                else:
                    base_path = main_path + date_path + processed_path
                    if not os.path.exists(os.path.dirname(base_path)):
                        os.makedirs(os.path.dirname(base_path))

                    if cal_option == 0:
                        file_title = "Phantom {0:} Angle {1:} Plug {2:} Rep {3:} Iter {4:} D".format(phantom[x-1], angle[x-1], plug[x-1], rep[x-1], ite[x-1])
                    else:
                        file_title = "Calibration Type {0:} Phantom {1:} Angle {2:} Plug {3:} Rep {4:} Iter {5:} D".format(cal_option, phantom[x-1], angle[x-1], 
                                        plug[x-1], rep[x-1], ite[x-1])
                        df["cal_type"] = cal_option

                    if save_format.casefold() == "parquet":
                        if parquet_engine == 'pyarrow':
                            df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, index= False)
                        else:
                            df.to_parquet(base_path + date_path.replace( "/"," ") + file_title + ".parquet", engine=parquet_engine, object_encoding='utf8', write_index= False)
                    else:
                        df.to_csv(base_path + date_path.replace( "/"," ") + file_title + ".csv")
                    tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")
                tqdm.write("".join(("\r ",date_path," Data processing finished!                                                                                                      ")))

                del df_list[:]
                del file_title, base_path

def data_pd_agg(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), sub_folder = "", 
                    processed_path = "Processed/DF/", correction = np.around(1.0e3/8192,4), conv_path = "Conv/", decimals = 4,
                    save_format="parquet", parquet_engine= 'pyarrow', is_recursive= False, cal_option = 0):
    """Generates aggregated Pandas DataFrame "phantom data set files".

    First, sweeps trough main_path + date_path folders and lists all .adc files found.
    Then, parameters are extracted from types 1, 2 and 3 calibration files.

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
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    """

    #if isinstance(date, list):
        #out_path1= "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{0}/DF/Means/{0} Phantom Set Means.parquet".format(datetime.now().strftime("%Y_%m_%d"))
        #out_path2  = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{0}/Means Agg by Rep/{0} Phantom Set Agg Means by Rep.parquet".format(datetime.now().strftime("%Y_%m_%d"))
        #out_path3 = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{0}/DF/Means Agg/{0} Phantom Set Means Agg.parquet".format(datetime.now().strftime("%Y_%m_%d"))

    #else:
    if not isinstance(date, list):
        #out_path1 = "".join((main_path,date,"/",processed_path,"Means/{0} Phantom Set Means.parquet".format(date)))
        #out_path2 = "".join((main_path,date,"/",processed_path,"Means Agg by Rep/{0} Phantom Set Agg Means by Rep.parquet".format(date)))
        #out_path3 = "".join((main_path,date,"/",processed_path,"Means Agg/{0} Phantom Set Means Agg.parquet".format(date)))

        date = [date]

    main_paths = []

    columns = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq","raw_digital_ch1", "raw_digital_ch2",
            "digital_ch1", "digital_ch2", "n_digital_ch1", "n_digital_ch2", "digital_unit"]

    for d in tqdm(date):

        main_paths = ["".join((main_path,d,"/",processed_path, sub_folder, conv_path))]

        out_path1 = "".join((main_path,d,"/",processed_path,"Means/{0} Phantom Set Means.parquet".format(d)))
        out_path2 = "".join((main_path,d,"/",processed_path,"Means Agg by Rep/{0} Phantom Set Agg Means by Rep.parquet".format(d)))
        out_path3 = "".join((main_path,d,"/",processed_path,"Means Agg/{0} Phantom Set Means Agg.parquet".format(d)))

        if cal_option == 0:
            ddf_list = dd_collect(main_paths, is_recursive=is_recursive, file_format=save_format, columns=columns, parquet_engine=parquet_engine)
        else:
            ddf_list = dd_collect(main_paths, is_recursive=is_recursive, file_format=save_format, columns=columns + ["cal_type"], check_key="cal_type", check_value= cal_option,
                                    parquet_engine=parquet_engine)

        df_list1 = []
        df_list2 = []
        df_list3 = []

        for ddf in tqdm(ddf_list, leave=False):
            dunit = ddf.digital_unit.compute().head(1).item()

            if cal_option != 0:
                cal_type = ddf.cal_type.unique()
                ddf = ddf.drop("cal_type", axis=1)

            ddf = ddf.apply(lambda x: uncategorize(x), axis=1, meta=ddf)
            print(ddf.columns)
            ddf1 = ddf.drop("digital_unit", axis=1).groupby(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq"], observed=True).agg(['mean', 'std'])

            df1 = ddf1.compute()
            df1 = df1.reset_index()
            df1.columns = ['_'.join(x).rstrip('_')  if isinstance(x,tuple) else x for x in df1.columns.ravel()]
            df1 = df1.round({"digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 0, "n_digital_ch2_mean": 0, "raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0})

            df1.sort_values(by=["date","rep", "iter","pair","freq"], ignore_index=True)

            df1["subject"] = [" ".join((a, "Phantom", b, "Angle", c, "Rep", d, "Iter", e))
                                            for a,b,c,d,e in zip(df1.date.astype(str), df1.phantom.astype(str), df1.angle.astype(str), df1.rep.astype(str), df1.iter.astype(str))]

            df1["digital_unit"] = dunit
            df_convert_values(df1, correction = correction)

            if cal_option != 0:
                df1["cal_type"] = cal_type

            df_list1.append(df1)

            ddf2 = ddf.drop(["iter", "digital_unit"], axis=1).groupby(["phantom", "angle", "plug", "date", "rep", "attLO", "attRF", "pair", "Tx", "Rx", "freq"], observed=True).agg(['mean','std'])

            df2 = ddf2.compute()
            df2 = df2.reset_index()
            df2.columns = ['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in df2.columns.ravel()]
            df2 = df2.round({"digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 0, "n_digital_ch2_mean": 0, "raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0})

            df2.sort_values(by=["date","rep","pair","freq"], ignore_index=True)
            df2["subject"] = [" ".join((a, "Phantom", b, "Angle", c, "Rep", d))
                                            for a,b,c,d in zip(df2.date.astype(str), df2.phantom.astype(str), df2.angle.astype(str), df2.rep.astype(str))]

            df2["digital_unit"] = dunit
            df_convert_values(df2, correction = correction)

            if cal_option != 0:
                df2["cal_type"] = cal_type

            df_list2.append(df2)

            ddf3 = ddf.drop(["rep","iter","digital_unit"], axis=1).groupby(["phantom", "angle", "plug", "date", "attLO", "attRF", "pair", "Tx", "Rx", "freq"], observed=True).agg(['mean','std'])

            df3 = ddf3.compute()
            df3 = df3.reset_index()
            df3.columns = ['_'.join(x).rstrip('_') if isinstance(x,tuple) else x for x in df3.columns.ravel()]
            df3 = df3.round({"digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 0, "n_digital_ch2_mean": 0, "raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0})

            df3.sort_values(by=["date","pair","freq"], ignore_index=True)
            df3["subject"] = [" ".join((a, "Phantom", b, "Angle", c))
                                            for a,b,c in zip(df3.date.astype(str), df3.phantom.astype(str), df3.angle.astype(str))]

            df3["digital_unit"] = dunit
            df_convert_values(df3, correction = correction)

            if cal_option != 0:
                df3["cal_type"] = cal_type

            df_list3.append(df3)

        dfw = pd.concat(df_list1)
        convert2category(dfw, cols = ["phantom", "angle", "plug", "date", "rep", "iter", "pair", "Tx", "Rx", "freq", "digital_unit", "voltage_unit"])

        if not os.path.exists(os.path.dirname(out_path1)):
            os.makedirs(os.path.dirname(out_path1))
        dfw.reset_index().to_parquet(out_path1, engine=parquet_engine)
        tqdm.write(f"\nSaved file: {out_path1}        ")

        dfw = pd.concat(df_list2)
        convert2category(dfw, cols = ["phantom", "angle", "plug", "date", "rep", "pair", "Tx", "Rx", "freq", "digital_unit", "voltage_unit"])

        if not os.path.exists(os.path.dirname(out_path2)):
            os.makedirs(os.path.dirname(out_path2))
        dfw.reset_index().to_parquet(out_path2, engine=parquet_engine)
        tqdm.write(f"\nSaved file: {out_path2}        ")

        dfw = pd.concat(df_list3)
        convert2category(dfw, cols = ["phantom", "angle", "plug", "date", "pair", "Tx", "Rx", "freq", "digital_unit", "voltage_unit"])

        if not os.path.exists(os.path.dirname(out_path3)):
            os.makedirs(os.path.dirname(out_path3))
        dfw.reset_index().to_parquet(out_path3, engine=parquet_engine)
        tqdm.write(f"\nSaved file: {out_path3}        ")

def df_convert_values(df, correction = np.around(1.0e3/8192,4)):
    """Generates converted columns from digital values on Pandas DataFrame "phantom data set files".

    Parameters
    ----------
    df : Data Frame (Pandas or Dask)
        input Data Frame
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
    """
    attRF = dB2Volts(df["attRF"] - 25)

    df["voltage_ch1"] = df["digital_ch1_mean"]*correction
    df["voltage_ch2"] = df["digital_ch2_mean"]*correction
    df["voltage_mag"] = np.sqrt(np.square(df["voltage_ch1"]) + np.square(df["voltage_ch2"]))
    df["voltage_phase"] = np.arctan2(df["voltage_ch1"], df["voltage_ch2"])
    df["raw_voltage_ch1"] = df["raw_digital_ch1_mean"]*correction
    df["raw_voltage_ch2"] = df["raw_digital_ch2_mean"]*correction
    df["raw_voltage_mag"] = np.sqrt(np.square(df["raw_voltage_ch1"]) + np.square(df["raw_voltage_ch2"]))
    df["raw_voltage_phase"] = np.arctan2(df["raw_voltage_ch1"], df["raw_voltage_ch2"])
    df["n_voltage_ch1"] = attRF*df["n_digital_ch1_mean"]*correction
    df["n_voltage_ch2"] = attRF*df["n_digital_ch2_mean"]*correction
    df["n_voltage_mag"] = np.sqrt(np.square(df["n_voltage_ch1"]) + np.square(df["n_voltage_ch2"]))
    df["n_voltage_phase"] = np.arctan2(df["n_voltage_ch1"], df["n_voltage_ch2"])

    if correction == np.around(1.0e3/8192,4):
        df["voltage_unit"] = "mV"
    elif correction == np.around(1.0/8192,7):
        df["voltage_unit"] = "V"
    else:
        df["voltage_unit"] =  "converted by factor {}".format(correction)


def calibration_mean_dataframe(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']),
                     cal_path = "Calibration/Means Agg/", processed_path = "Processed/DF/", cal_type = 2, parquet_engine= 'pyarrow'):
    """Return array with mean_ch1 and mean_ch2 of a calibration file (in digital scale).

    Parameters
    ----------
    date : str
       subfolder(s) by measurement date in the form "YYYY_MM_DD/"
    main_path : str
        main path to files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_type : int, optional
        cslibration type, by default 2

    Returns
    -------
    cal_mean : ndarray
        contains two sub-arrays, [...,0] for mean_ch1 and [...,1] for mean_ch2 of the calibration file
    """
    date_path = _date2path(date)

    if cal_type not in (1,2,3):
        tqdm.write(f"Error! Invalid Calibration Type: {cal_type}")
        return

    if cal_type == 1:
        columns = ["cal_type", "date" "digital_ch1", "digital_ch2"]
    elif cal_type == 2:
        columns = ["cal_type", "date", "attLO", "freq", "digital_ch1", "digital_ch2"]
    else:
        columns = ["cal_type", "date", "attLO", "attRF", "freq", "digital_ch1", "digital_ch2", "c_digital_ch1", "c_digital_ch2"]

    path_list = "".join((main_path, date_path, processed_path, cal_path))
    cal_path = "".join((main_path, date_path, processed_path, cal_path, "".join((date_path.replace( "/",""), " Calibration Processed Agg Means Type ", str(cal_type), ".parquet"))))

    if os.path.exists(cal_path):

        df = pd.read_parquet(cal_path, columns=columns, engine=parquet_engine)
        data_mean = df
    else:

        df_list = df_sweep(path_list = path_list, is_recursive = False, file_format="parquet")

        df = [pd.read_parquet(c_path, columns=columns, engine=parquet_engine) for c_path in df_list]

        df = pd.concat(df, axis=0)

        if cal_type == 3:
            data_mean = df["cal_type" == cal_type, ].groupby(by=columns[1:-2], observed=True).mean().round({"digital_ch1": 0, "digital_ch2": 0, "c_digital_ch1": 0, "c_digital_ch2": 0})
        else:
            data_mean = df["cal_type" == cal_type].groupby(by=columns[1:-2], observed=True).mean().round({"digital_ch1": 0, "digital_ch2": 0})
        data_mean.reset_index(inplace=True)
        if cal_type == 3:
            data_mean.drop("attRF")

    return data_mean

def simple_declutter(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']),  processed_path = "Processed/DF/",
                        sub_folder = "Means/", correction = np.around(1.0e3/8192,4), decimals = 4, file_format="parquet", parquet_engine= 'pyarrow', is_recursive= False, center = 'mean'):
    """Perform Average Trace Subtraction on Pandas DataFrame "phantom data set files".

    Uses a simple clutter rejection technique in the form of a spatial filter.

    Calculates average singal per frequency for antenna pairs with a same distance (assuming the clutter is constant for such pairs).

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

        df_list = df_collect(main_paths, is_recursive=is_recursive, file_format=file_format, columns=columns)
        if ~df_list[0].columns.str.contains('distances', case=False, na=False).all():
            df_list = dfsort_pairs(df_list, reference_point = "tumor", sort_type = "between_antennas", decimals = 4, out_distances = True)

        decl_list = []
        tqdm.pandas()

        for df in tqdm(df_list, leave=False):
            dunit = df.digital_unit.head(1).item()
            df = df.progress_apply(lambda x: uncategorize(x), axis=1)
            if "index" in df.columns:
                df.drop("index", axis=1, inplace=True)
            if "voltage_unit" in df.columns:
                #vunit = data.voltage_unit.unique().tolist()[0]
                # remove "voltage" columns, only digital used to reduce uncertainty
                df = df.loc[:,~df.columns.str.contains('^(?=.*voltage)(?!.*mag).*', case=False, na=False)]
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
            df_convert_values(df, correction = correction)

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
        convert2category(dfw, cols = ["phantom", "angle", "plug", "date", "rep", "iter", "pair", "Tx", "Rx", "freq", "digital_unit", "voltage_unit", "subject"])
        if "index" in dfw.columns:
            dfw.drop("index", axis=1, inplace=True)

        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        dfw.reset_index().to_parquet(out_path, engine=parquet_engine)
        tqdm.write(f"\nSaved file: {out_path}        ")

def avg_trace_clutter(df, progress_bar = True, center='mean'):
    """Calculate average trace cluttter per distance between antennas and per frequency.

    Calculates average singal per frequency for antenna pairs with a same distance (assuming the clutter is constant for such pairs).

    Only uses digital signals.

    Parameters
    ----------
    df : Pandas df or list of df
        input DataFrame or list of dataframes
    progress_bar: bool, optional
        set to True to display tqdm progress bar, by default True
    center: str, optional
        center tendency option between 'mean' or 'median', by default 'mean'

    Returns
    -------
    clutter: Pandas df or list of df
        DataFrame(s) with average trace clutter values
        multi-indexed using ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", "freq"]
    """
    if not isinstance(df, list):
        df = [df]

    if ~df[0].columns.str.contains('distances', case=False, na=False).all():
        df = dfsort_pairs(df, reference_point = "tumor", sort_type = "between_antennas", decimals = 4, out_distances = True)

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
        if "voltage_unit" in data.columns:
            #vunit = data.voltage_unit.unique().tolist()[0]
            # remove "voltage" columns, only digital used to reduce uncertainty
            data = data.loc[:,~data.columns.str.contains('s(?=.*voltage)(?!.*mag).*', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('subject', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('(?=.*digital)(?!.*ch[12]).*', case=False, na=False)]
        c = data.drop(["Tx","Rx"], axis = 1).groupby(["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "distances", "freq"], observed=True).agg([agg_center])
        c.columns = [x[0] if isinstance(x,tuple) else x for x in c.columns.ravel()]
        if "index" in c.columns:
            c.drop("index", axis=1, inplace=True)
        if "digital_ch1_mean".casefold() in c.columns:
            keys = {"raw_digital_ch1_mean": 0, "raw_digital_ch2_mean": 0, "digital_ch1_mean": 0, "digital_ch2_mean": 0, "n_digital_ch1_mean": 8, "n_digital_ch2_mean": 8}
        else:
             keys = {"raw_digital_ch1": 0, "raw_digital_ch2": 0, "digital_ch1": 0, "digital_ch2": 0, "n_digital_ch1": 8, "n_digital_ch2": 8}
        c = c.round(keys)
        clutter.append(c)

    if len(clutter) == 1:
        clutter = clutter[0]

    return clutter

def convert2category(df, cols = ["date", "rep", "iter"], natsort=True):
    """Convert selected dataframe columns to category dtype.

    If a column name is not found (KeyError), displays a message and continues with other columns.

    Parameters
    ----------
    df : Pandas DataFrame
        input dataframe
    cols : list of str, optional
        list of column names to convert to categorical dtype, by default ["date", "rep", "iter"]
    natsort : bool, optional
        set to True to use natural sorting for each category, default True
    """

    for c in cols:
        try:
            if natsort:
                df[c] = pd.Categorical(df[c], ordered=True, categories=natsorted(df[c].unique()))
            else:
                df[c] = pd.Categorical(df[c])
        except KeyError:
            tqdm.write(f"{c} is not a column in the dataframe.")

def dd_collect_formatted(path_list, dftype = 0, is_recursive = False, file_format="parquet", parquet_engine= 'pyarrow', columns=None, check_key=None, check_value=None):
    """Collect all Dask DataFrame files in given directory path list.

    This function explicitly reads columns in previously determined dtypes. Returns a list with the DataFrame files found.

    Parameters
    ----------
    main_path : list of str
        list of paths main path to measurement files
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False
    file_format: str
        target file format ("parquet" or "csv"), by default "parquet"
    column: list of str
        list of columns to read, by default None (reads all columns)
    chcek_key: str
        key to check value, by default None
    chcek_value: number or str
        value to check, by default None

    Returns
    ----------
    df_list: list of df
        1-D list with the Pandas DataFrame files found
    """

    if dftype == 1:

        dtypes = {
                "voltage_unit": "category", "digital_unit": "category", "obs": "category", "samp_rate": "category", "nsamples": "category", "iter": "category", "rep": "category", "date": "category",
                "cal_type": "category", "attRF": "category", "attLO": "category", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "category",
                "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

    elif dftype == 2:

        dtypes = {
                "voltage_unit": "category", "digital_unit": "category", "obs": "category", "samp_rate": "category", "nsamples": "category", "iter": "category", "rep": "category", "date": "category",
                "cal_type": "category", "attRF": "category", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "category",
                "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

    elif dftype == 3:

        dtypes = {
                "voltage_unit": "category", "digital_unit": "category", "obs": "category", "samp_rate": "category", "nsamples": "category", "iter": "category", "rep": "category", "date": "category",
                "cal_type": "category", "attRF": "int32", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", "freq": "category",
                "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64"
                }

    else:

        dtypes = {
                "voltage_unit": "category", "digital_unit": "category", "obs": "category", "samp_rate": "category", "nsamples": "category", "phantom": "category", "plug": "category", "iter": "category", 
                "rep": "category", "date": "category", "attRF": "int32", "attLO": "int32", "raw_digital_ch1": "int64", "raw_digital_ch2": "int64", "time": "Float64", 
                "freq": "category", "voltage_ch1": "Float64", "voltage_ch2": "Float64", "voltage_mag": "Float64" , "voltage_phase": "Float64",
                "n_voltage_ch1": "Float64", "n_voltage_ch2": "Float64", "n_voltage_mag": "Float64" , "n_voltage_phase": "Float64", "subject": "category"
                }

    df_paths = natsorted(df_sweep(path_list, is_recursive, file_format))

    df_list = []
    if file_format.casefold() == "parquet":
        for p in df_paths:
            df_list.append(dd.read_parquet(p, engine=parquet_engine, columns=columns, dtypes=dtypes))
            tqdm.write(f"\rOpened file: {p}        ", end="\r")
    elif file_format.casefold() == "csv":
        for p in df_paths:
            df_list.append(dd.read_csv(p, cols = columns, dtypes=dtypes, low_memory=False))
            tqdm.write(f"\rOpened file: {p}        ", end="\r")

    if check_key is not None:
        df_list = [df for df in df_list if df.compute()[check_key].eq(check_value).all()]

    return df_list


def list_paths(dates, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", 
                    processed_path = "Processed/DF/", conv_path = "Conv/", file_format="parquet"):
    """Generate list of folder paths using given dates.

    Parameters
    ----------
    dates : list of str,
        list of dates in format "YYYY_MM_DD" or date folders "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        sub-folder with calibration files, by default "Calibration/"
    processed_path : str, optional
        sub-folder for output JSON files, by default "Processed/"
        final location will be main_path + processed_path + cal_path)
    conv_path : str, optional
        sub-folder for JSON files of converted data, by default "Conv/"

    Returns
    ----------
    path_list : list of str,
        list of folder paths
    """

    if not isinstance(dates, list):
        dates = [dates]

    path_list = ["/".join((main_path,date.rstrip("/"),processed_path,cal_path,conv_path)) for date in dates]
    return path_list

def cal_folder_sweep(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/"):
    """Find all calibration .adc files in a given directory path.

    In addition to a list with the string pathnames of each file found, the lists meta_index and meta_freq are generated,
    containing elements for calibration types 1, 2 and 3. The ndarray meta_index associate parameters of Iteration and Repetition to each file
    while the list meta_freq does the same for the Frequency.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to .adc measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        calibration folder path, by default "Calibration/"
    cal_type : int, optional
        number describing the calibration type, by default 1. Possible types:
        :1: LO and RF grounded with a 50 ohm terminator
        :2: RF grounded with a 50 ohm terminator, LO active with frequencies set by freq_range
        :3: RF receives Tx and Rx connected directly (without antennas), using frequencies set by freq_range

    Returns
    ----------
    path_list : list of list of str
        list with 3 list elements (one for each calibration type) with full pathnames to each .adc file found
    meta_index : list of ndarrays of int
        list with 3 elements (one per cal_type), each a 2-D array of parameters associated to each .adc file found
        Parameters are Iteration and Repetition
    meta_freq : list of list of str
        list with 3 elements (one per cal_type), each a 1-D list of string formatted frequencies associated to each .adc file found
        the element representing type 1 is an empty list
    """
    path_list = [[],[],[]]
    meta_index = [[],[],[]]
    meta_freq = [[],[],[]]

    date_path = _date2path(date)

    for i, cal_type in enumerate(range(1,4)):
        if cal_type == 1:
            path_list[i], meta_index[i] = _single_cal_folder_sweep(date= date_path, main_path= main_path, cal_path= cal_path, cal_type= cal_type)
        else:
            path_list[i], meta_index[i], meta_freq[i] = _single_cal_folder_sweep(date= date_path, main_path= main_path, cal_path= cal_path, cal_type= cal_type)

    return path_list, meta_index, meta_freq

def phantom_folder_sweep(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), ph_path = "Phantom*/"):
    """Find all phantom measurement .adc files in a given directory path.

    In addition to a list with the string pathnames of each file found, the ndarray meta_index and list meta_freq are generated.
    The ndarray meta_index associates parameters of Phantom, Angle, Plug, Tx, Rx, Iteration and Repetition to each file while the list meta_freq
    does the same for the Frequency.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to .adc measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    ph_path : str, optional
        wildcard phantom folder path, by default "Phantom*/"

    Returns
    ----------
    list of str
        list with full pathnames to each .adc file found
    meta_index : ndarray of int
        2-D list of parameters associated to each .adc file found
        Parameters are Phantom, Angle, Plug, Tx, Rx, Iteration and Repetition
    meta_freq : list of str
        list of string formatted frequencies associated to each .adc file found
    """

    date_path = _date2path(date)

    a = natsorted(Path(main_path + date_path).glob(ph_path+"**/*.adc"))
    path_list = [str(x) for x in a]

    ## meta_index is matrix with rows in the form [Phantom, Angle, Plug, Antenna Pair Tx, Antenna Pair Rx, Iteration, Repetition]

    meta_index = [[int(re.search(r'(?<=[Pp][hH][aA][nN][tT][oO][mM](\s))(\d+)',str(i)).group(0)),
                int(re.search(r'(\d+)(?=(\s)[dD][eE][gG])',str(i)).group(0)),
                int(re.search(r'(?<=[Pp][lL][uU][gG](\s))(\d+)',str(i)).group(0)),
                int(re.search(r'(?<=[Tt][xX])(\s?)(\d+)',str(i)).group(0)),
                int(re.search(r'(?<=[Rr][xX])(\s?)(\d+)',str(i)).group(0)),
                int(re.search(r'(?<=[Ii][tT][eE][rR](\s))(\d+)',str(i)).group(0)),
                int(re.search(r'(?<=[Rr][eE][pP](\s))(\d+)',str(i)).group(0))]
                for i in path_list]
    meta_freq = [re.search(r'(\d+\_?\d+)(?=[Mm][Hh][zZ])',str(i)).group(0) for i in path_list]
    meta_index = np.array(meta_index)

    return path_list, meta_index, meta_freq

def check_configuration_parameters_cal(ConfigSet, date, cal_type = 1, rep = 1, ite = 1):
    """Extract attLO, attRF and obs from "calibration configuration parameters", when available.

    Parameters
    ----------
    ConfigSet : list of dict
        list with dictionaries originated from "calibration configuration parameters"
    date : str
        date : str,
        date in format "YYYY_MM_DD"
    cal_type : int, optional
        calibration type, by default 1
    rep : int, optional
        repetition number, by default 1
    ite : int, optional
        iteration number, by default 1

    Returns
    ----------
    attLO : int or str
        attenuator value in dB or string "check logs or notes"
    attRF : int or str
        attenuator value in dB or string "check logs or notes"
    obs: str
        observation string
    freqs: list of float
        list of input frequencies in MHz
        returns None if there is no configuration file.
    """

    date = date.replace("/","")

    Config = next((item for item in ConfigSet if (item["date"] == date) and (item["cal_type"] == cal_type) and (item["rep"] == rep) and ((item["iter"] == ite))), None)

    if Config is not None:
        attRF = str(Config["attRF"])
        attLO = str(Config["attLO"])
        obs = Config["obs"] if Config["obs"] != "" else _assign_cal_obs(cal_type = cal_type)
        freqs = Config["freq_range"]
        freqs = [float(fre.replace("_",".")) for fre in freqs]
    else:
        attLO, attRF = _assign_cal_att(cal_type = cal_type, date = date, rep = rep)
        obs = _assign_cal_obs(cal_type = cal_type)
        freqs = None

    return attLO, attRF, obs, freqs

def check_configuration_parameters_case(ConfigSet, date, phantom = 1, angle = 0, plug = 2, rep = 1, ite = 1):
    """Extract attLO, attRF and obs from "measurement configuration parameters", when available.

    Parameters
    ----------
    ConfigSet : list of dict
        list with dictionaries originated from "measurement configuration parameters"
    date : str
        date : str,
        date in format "YYYY_MM_DD"
    phantom : int, optional
        phantom number, by default 1
    angle: int, optional
        angle position in degrees, by default 0
    plug: int, opt
        plug number, by default 2
    rep : int, optional
        repetition number, by default 1
    ite : int, optional
        iteration number, by default 1

    Returns
    ----------
    attLO : int or str
        attenuator value in dB or string "check logs or notes"
    attRF : int or str
        attenuator value in dB or string "check logs or notes"
    obs: str
        observation string
    freqs: list of float
        list of input frequencies in MHz
        returns None if there is no configuration file.
    pairs: ndarray of int
        list of antenna pairs
        returns None if there is no configuration file.
    """

    date = date.replace("/","")

    Config = next((item for item in ConfigSet if (item["date"] == date) and (item["Phantom"] == phantom) and (item["Angle"] == angle) and (item["rep"] == rep) and ((item["iter"] == ite))), None)

    if Config is not None:
        attRF = Config["attRF"]
        attLO = Config["attLO"]
        obs = Config["obs"] if Config["obs"] != "" else _assign_case_obs(date = date, phantom= phantom, plug = plug, rep = rep)
        freqs = Config["freq_range"]
        freqs = [float(fre.replace("_",".")) for fre in freqs]
        pairs = np.array(Config["pairs"])
    else:
        attLO, attRF = _assign_case_att(date = date, phantom = phantom)
        obs = _assign_case_obs(date = date, phantom= phantom, plug = plug, rep = rep)
        freqs = None
        pairs = None

    return attLO, attRF, obs, freqs, pairs

def collect_configuration_files(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), sub_folder= "Config/",
                            is_recursive = False):
    """Collect JSON configuration files for both calibration and phantom measurements.

    This function should be used to extract the configuration paramenters when processing .adc files into JSON "calibration file" and "scan data set" files.

    Parameters
    ----------
    date : str or list of str
        subfolder(s) by measurement date in the form "YYYY_MM_DD" or "YYYY_MM_DD/"
    main_path : str, optional
        main path to files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    sub_folder : str, optional
        specific sub-path to JSON "configuration parameters" files, by default "Config/"
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False

    Returns
    ----------
    CalConfigSet : list of dict
        list of dictionaries containing found "calibration configuration parameters" information
    MeasConfigSet : list of dict
        list of dictionaries containing found "measurement configuration parameters" information
    """

    CalConfigSet = collect_json_data(date, main_path= main_path, sub_folder = sub_folder, json_type = "calibration configuration parameters", 
                                            is_recursive= is_recursive)

    MeasConfigSet = collect_json_data(date, main_path= main_path, sub_folder = sub_folder, json_type = "measurement configuration parameters", 
                                            is_recursive= is_recursive)

    return CalConfigSet, MeasConfigSet

def json_sweep(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']),
                sub_folder = "Processed/Converted/", is_recursive = False):
    """Find all json files in given directory path.

    Returns a list with the string pathnames of each .json file found.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    sub_folder : str, optional
        specific sub-path to JSON "scan data set" files, by default "Processed/Converted/"
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False

    Returns
    ----------
    json_paths: list of str
        1-D list with full pathnames to each .json file found
    """

    date_path = _date2path(date)

    if is_recursive:
        a = natsorted(Path("".join([main_path, date_path, sub_folder])).glob("**/*.json"))
        json_paths = [str(i) for i in a]
    else:
        a = natsorted(Path("".join([main_path, date_path, sub_folder])).glob("*.json"))
        json_paths = [str(i) for i in a]

    return json_paths


def collect_json_data(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), json_type = "scan data set",
                        sub_folder = "Processed/Converted/", is_recursive = False):
    """Find JSON files and load those of a given type to a list of dictionaries.

    Parameters
    ----------
    date : str or list of str
        subfolder(s) by measurement date in the form "YYYY_MM_DD" or "YYYY_MM_DD/"
    main_path : str, optional
        main path to files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    json_type : str, optional
        json file value for the key "type", by default "scan data set"
    sub_folder : str, optional
        specific sub-path to JSON "scan data set" files, by default "Processed/Converted/"
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False

    Returns
    -------
    data_sets : list of dict
        list of dictionaries containing found "scan data set" information
    """

    data_sets = []

    date_path = _date2path(date)

    if not isinstance(date_path, list):
        date_path = [date_path]
    for dp in date_path:
            paths = json_sweep(date = dp, main_path = main_path, sub_folder = sub_folder, is_recursive= is_recursive)
            for p in paths:
                with open(p, 'r') as fp:
                    data = json.load(fp)
                if data["type"].lower() == json_type.lower():
                    data_sets.append(data)

    return data_sets

def collect_json_data_simple(main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), json_type = "scan data set",
                        sub_folder = "Processed/Converted/", is_recursive = False):
    """Find JSON files and load those of a given type to a list of dictionaries.

    Does not use date input, collecting files from main_path + sub_folder.

    Parameters
    ----------
    main_path : str, optional
        main path to files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    json_type : str, optional
        json file value for the key "type", by default "scan data set"
    sub_folder : str, optional
        specific sub-path to JSON "scan data set" files, by default "Processed/Converted/"
    is_recursive : bool, optional
        set to True to make sweep recursive (access folders inside the sub_folder path), by default False

    Returns
    -------
    data_sets : list of dict
        list of dictionaries containing found "scan data set" information
    """

    data_sets = []

    a = natsorted(Path("".join([main_path, sub_folder])).glob("*.json"))
    json_paths = [str(i) for i in a]

    for p in json_paths:
        with open(p, 'r') as fp:
            data = json.load(fp)
        if data["type"].lower() == json_type.lower():
            data_sets.append(data)

    return data_sets

def revert_gains_Rx(df, gaindB = -18.2):
    """Revert gains of voltage columns to values at Rx input.

    Default gaindB is -18.2 dB (entire Rx chain reverted) plus the attenuators used.

    Converts the columns 'voltage_ch1', 'voltage_ch2', 'voltage_mag',
            'raw_voltage_ch1', 'raw_voltage_ch2', 'raw_voltage_mag',
            'n_voltage_ch1', 'n_voltage_ch2', 'n_voltage_mag', 'power', 'power_dBm'.

    Parameters
    ----------
    df : Pandas dataframe
        dataframe with columns to convert
    gaindB : int or float
        gain in dB to apply, by default -18.2
        (entire Rx chain reverted)
    """

    # df["gaindB"] = -14 + 1.8 - 10 + 4 + df.attRF
    # default gaindB -14 + 1.8 - 10 + 4 = -18.2

    df["gaindB"] = gaindB + df.attRF.astype(float)

    v_cols = ['voltage_ch1', 'voltage_ch2', 'voltage_mag',
            'n_voltage_ch1', 'n_voltage_ch2', 'n_voltage_mag']

    for col in v_cols:
        df[col] = round( (10**(df["gaindB"]/20)) * df[col], 4)

    w_cols = ['power']

    for col in w_cols:
        if col in df.columns:
            df.loc[:, col] = round( (10 ** (df["gaindB"] / 10)) * df[col] * 0.001,4)

    dB_cols = ['power_dBm']

    for col in dB_cols:
        if col in df.columns:
            df.loc[:, col] = df["gaindB"] + df[col]

def revert_gains_Tx(df, gaindB = -14):
    """Revert gains of voltage columns to Tx values before switching matrix.

    Default gaindB is -14 dB (only DC Receiver gain reverted) plus the attenuators used.

    Converts the columns 'voltage_ch1', 'voltage_ch2', 'voltage_mag', 'power', 'power_dBm'.

    Parameters
    ----------
    df : Pandas dataframe
        dataframe with columns to convert
    gaindB : int or float
        gain in dB to apply, by default -14
        (DC receiver gain reversed)
    """
    # df["gaindB"] = -14 + df["attRF"].astype(float)
    # default gaindB -14 (only DC Receiver)

    df.loc[:, "gaindB"] = gaindB + df["attRF"].astype(float)

    v_cols = ['voltage_ch1', 'voltage_ch2', 'voltage_mag']

    for col in v_cols:
        df.loc[:, col] = round( (10 ** (df["gaindB"] / 20)) * df[col] * 0.001,4)

    w_cols = ['power']

    for col in w_cols:
        if col in df.columns:
            df.loc[:, col] = round( (10 ** (df["gaindB"] / 10)) * df[col] * 0.001,4)

    dB_cols = ['power_dBm']

    for col in dB_cols:
        if col in df.columns:
            df.loc[:, col] = df["gaindB"] + df[col]

def _str2folder_path(x):
    """Return string or string list x with a forward slash "/" at the end.

    Verifies if the string ends with a forward slash "/" to designate a folder and inserts one if needed.

    Parameters
    ----------
    x : str or list of str
        name(s) to designate folder(s)

    Returns
    ----------
    folder_path : str
        folder name with forward slash "/"
    """

    if isinstance(x,list):
        folder_path = []
        for d in x:
            folder_path.append(d) if d[-1] == "/" else folder_path.append("".join([d,"/"]))
    else:
        folder_path = x if x[-1] == "/" else "".join([x,"/"])

    return folder_path

def _date2path(date):
    """Return string or string list date in the folder path format "YYYY_MM_DD/".

    Parameters
    ----------
    date : str or list of str
        date(s) to designate folder(s), in the format "YYYY_MM_DD" or "YYYY/MM/DD" or "YYYY-MM-DD"

    Returns
    ----------
    date_path : str
        date folder name(s) in format "YYYY_MM_DD/"
    """

    if isinstance(date,list):
        date_path = []
        for d in date:
            date_path.append(d.replace("/","_").replace("-","_"))
    else:
        date_path = date.replace("/","_").replace("-","_")

    date_path = _str2folder_path(date)

    return date_path

def _single_cal_folder_sweep(date, main_path = "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE']), cal_path = "Calibration/", cal_type = 1):
    """Find calibration .adc files of type cal_type in a given directory path.

    In addition to a list with the string pathnames of each file found, the ndarray meta_index and (for types 2 and 3) list meta_freq are generated.
    The ndarray meta_index associates parameters of Iteration and Repetition to each file while the list meta_freq does the same for the Frequency.

    Parameters
    ----------
    date : str,
        date in format "YYYY_MM_DD" or date folder "YYYY_MM_DD/"
    main_path : str, optional
        main path to .adc measurement files, by default "{}/OneDrive - McGill University/Documents McGill/Data/PScope/".format(os.environ['USERPROFILE'])
    cal_path : str, optional
        calibration folder path, by default "Calibration/"
    cal_type : int, optional
        number describing the calibration type, by default 1. Possible types:
        :1: LO and RF grounded with a 50 ohm terminator
        :2: RF grounded with a 50 ohm terminator, LO active with frequencies set by freq_range
        :3: RF receives Tx and Rx connected directly (without antennas), using frequencies set by freq_range

    Returns
    ----------
    path_list : list of str
        list with full pathnames to each .adc file found
    meta_index : ndarray of int
        2-D array of parameters associated to each .adc file found
        Parameters are Iteration and Repetition
    meta_freq : list of str
        list of string formatted frequencies associated to each .adc file found
    """
    date_path = _date2path(date)

    a = natsorted(Path(main_path + date_path).glob(cal_path + "Type " + str(cal_type) + "/**/*.adc"))
    path_list = [str(x) for x in a]


    ## meta_index for calibration measurements is matrix with rows in the form [Iteration, Repetition]

    meta_index = np.array([[int(re.search(r'(?<=[Ii][tT][eE][rR](\s))(\d+)',str(i)).group(0)),
                int(re.search(r'(?<=[Rr][eE][pP](\s))(\d+)',str(i)).group(0))]
                for i in path_list])

    if cal_type != 1:
        meta_freq = [re.search(r'(\d+\_?\d+)(?=[Mm][Hh][zZ])',str(i)).group(0) for i in path_list]
        return path_list, meta_index, meta_freq
    else:
        return path_list, meta_index

def _assign_cal_att(cal_type, date, rep = 1):
    """Assign LO and RF attenuator values for calibrations on 29/08/2019, 05/09/2019, 06/09/2019 and 30/09/2019.

    Function assigns known attenuator values because experiments before 23/09/2019 did not have JSON files with this information.

    Parameters
    ----------
    cal_type : int
        calibration type number
    date : str
        date in format "YYYY_MM_DD"
    rep : int, optional
        repetition or re-positioning number, by default 1

    Returns
    ----------
    attLO: int or str
        attenuator value in dB or string "check logs or notes"
    attRF: int or str
        attenuator value in dB or string "check logs or notes"
    """

    valid_dates = ["2019_08_29", "2019_09_05", "2019_09_06", "2019_09_30"]

    date = date.replace("/","")

    if cal_type == 1:
        attLO = "grounded"
        attRF = "grounded"
    elif cal_type == 2 and date in valid_dates:
        if rep == 1:
            attLO = str(20)
            attRF = "grounded"
        elif rep == 2 and date in valid_dates:
            attLO = str(9)
            attRF = "grounded"
        else:
            attLO = "check logs or notes"
            attRF = "check logs or notes"
    elif cal_type == 3:
        if rep == 1 and (date == "2019_08_29" or date == "2019_09_06" or date == "2019_09_30"):
            attLO = str(20)
            attRF = str(25)
        elif rep == 1 and date == "2019_09_05":
            attLO = str(20)
            attRF = str(27)
        elif rep == 2 and (date == "2019_08_29" or date == "2019_09_06"):
            attLO = str(9)
            attRF = str(25)
        elif rep == 2 and date == "2019_09_05":
            attLO = str(9)
            attRF = str(27)
        elif rep == 3 and date == "2019_09_05":
            attLO = str(9)
            attRF = str(21)
        else:
            attLO = "check logs or notes"
            attRF = "check logs or notes"
    else:
        attLO = "check logs or notes"
        attRF = "check logs or notes"

    return attLO, attRF

def _assign_case_att(date, phantom):
    """Assign LO and RF attenuator values for experiments on 29/08/2019, 05/09/2019 and 06/09/2019.

    Function assigns known attenuator values because experiments before 23/09/2019 did not have JSON files with this information.

    Parameters
    ----------
    date : str
        measurement date in the form "YYYY_MM_DD"
    phantom : int
        phantom number

    Returns
    ----------
    attLO: int or str
        attenuator value in dB or string "check logs or notes"
    attRF: int or str
        attenuator value in dB or string "check logs or notes"
    """

    date = date.replace("/","")

    if phantom == 1:
        attLO = 20
        attRF = 9
    elif phantom == 14:
        if date == "2019_08_29":
            attLO = 20
            attRF = 9
        else:
            attLO = 20
            attRF = 0
    else:
        attLO = "check logs or notes"
        attRF = "check logs or notes"

    return attLO, attRF

def _assign_case(plug, rep):
    """Assign baseline or tumor title for phantom scan.

    Parameters
    ----------
    plug : int
        plug number
    rep : int or list of int or tuple of int
        re-positioning or repetition, either a single number or a list of numbers

    Returns
    ----------
    case: str
        case identifier (Baseline  or Tumor + Repetition number)
    """

    if isinstance(rep, (list,tuple)):
        rep = ','.join(str(i) for i in rep)

    if plug == 2:
        case =  "Baseline " + str(rep)
    else:
        case = "Tumor " + str(rep)

    return case

def _assign_case_obs(date, phantom, plug, rep):
    """Assign observations for experiments on 29/08/2019, 05/09/2019 and 06/09/2019.

    Function assigns known observations because experiments before 23/09/2019 did not have JSON files with this information.

    Parameters
    ----------
    date : str
        measurement date in the form "YYYY_MM_DD"
    phantom : int
        phantom number
    plug : int
        plug number
    rep : int
        repetition or re-positioning number

    Returns
    ----------
    obs: str
        observation string
    """

    date = date.replace("/","")

    if rep == 1:
        if plug in (0,1,2):
            obs = "Initial phantom measurement - baseline."
        elif plug in range(3,39):
            obs = "Plug changed to tumor."
    elif rep == 2:
        if date == "2019_08_29":
            obs = "Consecutive measurement, no plug change."
        elif date == "2019_09_05" or "2019_09_06":
            obs = "Same plug removed and placed back."
        else:
            obs = ""
    elif rep == 3:
        if date == "2019_09_05" or "2019_09_06":
            obs = "Phantom repositioned. Plug replaced."
        else:
            obs = ""
    else:
        obs = ""

    return obs

def _assign_cal_obs(cal_type):
    """Assign observations for calibrations on 29/08/2019, 05/09/2019 and 06/09/2019.

    Parameters
    ----------
    cal_type : int
        calibration type

    Returns
    ----------
    obs: str
        observation string
    """

    if cal_type == 1:
        obs = "Type 1: Both LO and RF grounded with 50 ohm terminators. No frequency input."
    elif cal_type == 2:
        obs = "Type 2: RF grounded with 50 ohm terminator, LO connected to frequency synthesizer."
    elif cal_type == 3:
        obs = "Type 3: RF connected to Rx-Tx directly by cables (bypassing antennas) and LO connected to frequency syntesizer."
    else:
        obs = ""

    return obs
