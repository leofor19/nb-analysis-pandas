# Python 3.8
# 2021-12-03

# Version 1.1.0

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

    Description:
            Module for performing comparisons between scans on Pandas DataFrames for the narrow band system.

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
import itertools as it
import json
import os
import os.path
from pathlib import Path
# import re
import warnings

# Third-party library imports
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
#from dask.distributed import Client, LocalCluster
#import matplotlib.pyplot as plt
from natsort import natsorted
import numpy as np
from numpy.core.numeric import zeros_like
import pandas as pd
# from tqdm import tqdm
from tqdm.notebook import tqdm
#from tqdm.dask import TqdmCallback
# from yaspin import yaspin

# Local application imports
import NarrowBand.analysis_pd.df_antenna_space as dfant
import NarrowBand.analysis_pd.df_processing as dfproc
import NarrowBand.analysis_pd.df_data_essentials as nbd
from NarrowBand.analysis_pd.uncategorize import uncategorize
from numpyencoder.numpyencoder import NumpyEncoder

# To supress warnings.warn("Non-categorical multi-index is likely brittle", UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

#client = Client()

def save_pairwise_comparisons(df, out_path = "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Comps/".format(datetime.now().strftime("%Y_%m_%d")), 
                                different_attRF = True, different_attLO = False, between_phantoms = False, between_dates = True, correction = np.around(1.0e3/8192,4), save_format="parquet", ftype=1):
    """Generate dataframe files comparing pairs of phantom scans on Pandas data frames.

    Possible to select whether to allow comparison between different attRF, attLO, phantoms/angles or dates.

    Comparing between different attRFs and dates is selected by default.

    Comparing between different attLO is NOT reccomended, because the gain variation on the final recorded measurements is unpredictable.

    Files format can be parquet or csv.

    Parameters
    ----------
    df : Pandas df or list of df
        input dataframe or list of dataframes
    different_attRF : bool, optional
        set to true to enable different attRF comparisons, by default True
    different_attLO : bool, optional
            set to true to enable different attLO comparisons (not reccomended), by default False
    between_phantoms : bool, optional
            set to true to enable comparisons between different phantoms or positions (angles), by default False
    between_dates : bool, optional
            set to true to enable comparisons between different dates, by default True
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    save_format: str
        target file format (either "parquet" or "csv"), by default "parquet"
    ftype: int, optional
        pairwise comparison function selection, by default 1
        1: pairwise_comparisons - differences are obtained using I-Q channel directly, then converted to magnitude/phase
        2: pairwise_comparisons2 - differences are obtained from magnitudes/phases instead of I-Q channel
        3: pairwsise_comparisons_of_medians - extracts means/medians per date before comparing, differences are obtained from magnitudes/phases instead of I-Q channel
    """
    if ftype == 1:

        compared, groups_list = pairwise_comparisons(df, different_attRF = different_attRF, different_attLO = different_attLO, between_phantoms = between_phantoms, between_dates = between_dates,
                                                    correction = correction)
    elif ftype == 2:

        compared, groups_list = pairwise_comparisons2(df, different_attRF = different_attRF, different_attLO = different_attLO, between_phantoms = between_phantoms, between_dates = between_dates,
                                                    correction = correction)
    else:

        compared, groups_list = pairwsise_comparisons_of_medians(df, different_attRF = different_attRF, different_attLO = different_attLO, between_phantoms = between_phantoms, between_dates = between_dates,
                                                    correction = correction, center = 'median')

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    groups_file = "".join((out_path,"{} groups.json".format(datetime.now().strftime("%Y_%m_%d"))))

    with open(groups_file, 'w') as f:
        json.dump(groups_list, f, ensure_ascii=False, cls=NumpyEncoder)
        #for item in groups_list:
        #    for pair in item:
        #       f.write(f'{pair}\n')

    df_file = "".join((out_path,"{} Scan Comparisons NUM.parquet".format(datetime.now().strftime("%Y_%m_%d"))))

    for i, df in enumerate(tqdm(compared)):
        if save_format.casefold() == "parquet":
            df.to_parquet(df_file.replace("NUM", f'{i:02d}'), engine='fastparquet', object_encoding='utf8')
        else:
            df.to_csv(df_file.replace("NUM", f'{i:02d}').replace("parquet","csv"))

def pairwise_comparisons(df, different_attRF = True, different_attLO = False, between_phantoms = False, between_dates = True, correction = np.around(1.0e3/8192,4)):
    """Compare pairs of phantom scans on Pandas data frames.

    Possible to select whether to allow comparison between different attRF, attLO, phantoms/angles or dates.

    Comparing between different attLO is NOT reccomended, because the gain variation on the final recorded measurements is unpredictable.

    Parameters
    ----------
    df : Pandas df or list of df
        input dataframe or list of dataframes
    different_attRF : bool, optional
        set to true to enable different attRF comparisons, by default True
    different_attLO : bool, optional
            set to true to enable different attLO comparisons (not reccomended), by default False
    between_phantoms : bool, optional
            set to true to enable comparisons between different phantoms or positions (angles), by default False
    between_dates : bool, optional
            set to true to enable comparisons between different dates, by default True
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files

    Returns
    ----------
    group_list: list
        list of groups, composed of permutations of dataframes grouped by columns ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"]

        In case there are different attRF, attLO or phantoms and the respective inputs are set to False, it is a list of lists. 

        Each sub-list contains tuples with the column values.
    """

    if not isinstance(df, list):
        df = [df]

    if not between_phantoms:
        groups = []
        for data in df:
            if data.phantom.nunique() > 1:
                g = data.groupby(by = ["phantom"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not between_dates:
        groups = []
        for data in df:
            if data.date.nunique() > 1:
                g = data.groupby(by = ["date"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not different_attLO:
        groups = []
        for data in df:
            if data.attLO.nunique() > 1:
                g = data.groupby(by = ["attLO"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not different_attRF:
        groups = []
        for data in df:
            if data.attRF.nunique() > 1:
                g = data.groupby(by = ["attRF"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    else:
        df = [attenuation_match(data, decimals = 0, correction = correction) for data in df]

    groups_list = []
    compared = []

    for data in tqdm(df):
        if "voltage_unit" in data.columns:
            vunit = data.voltage_unit.unique().tolist()[0]
            # remove "voltage" columns, only digital used to reduce uncertainty
            data = data.loc[:,~data.columns.str.contains('^(?=.*voltage)(?!.*mag).*', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('subject', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('unit', case=False, na=False)]

        data_gr = data.groupby(by = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"], observed=True)
        groups = [name for name,unused_df in data_gr]

        g_out = []

        for p in it.permutations(groups,2):
            # excluding reversed/redundant pairs, i.e. if (A,B) then no need for (B,A)
            if p <= p[::-1]:
                g_out.append(p)

        groups_list.append([g_out])

        if len(g_out) < 1:
            # skip group comparison for less than one group
            continue

        if "distances".casefold() in data.columns:
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "distances"]
        else:
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq"]
        data = data.set_index(keys = cols, drop=True).sort_index()
        c_list = []

        for g in g_out:
            attRF = max(g[0][7],g[1][7])
            c = data.loc[g[0]].subtract(data.loc[g[1]], fill_value=0)
            # relative differences (using voltage units)
            c["rel1_diff_ch1"] = 100.00*c.digital_ch1_mean.divide(data.loc[g[0],"voltage_mag"], fill_value=0)*correction
            c["rel1_diff_ch2"] = 100.00*c.digital_ch2_mean.divide(data.loc[g[0],"voltage_mag"], fill_value=0)*correction
            c["rel2_diff_ch1"] = -100.00*c.digital_ch1_mean.divide(data.loc[g[1],"voltage_mag"], fill_value=0)*correction
            c["rel2_diff_ch2"] = -100.00*c.digital_ch2_mean.divide(data.loc[g[1],"voltage_mag"], fill_value=0)*correction

            col_names = ["phantom_1", "angle_1", "plug_1", "date_1", "rep_1", "iter_1", "attLO_1", "attRF_1", "phantom_2", "angle_2", "plug_2", "date_2", "rep_2", "iter_2", "attLO_2", "attRF_2"]
            values = g[0] + g[1]

            for i, col in enumerate(col_names):
                c[col] = values[i]

            c = name_case_comparisons(c)
            c_list.append(c.copy())

        c = pd.concat(c_list)

        c["voltage_ch1"] = c["digital_ch1_mean"]*correction
        c["voltage_ch2"] = c["digital_ch2_mean"]*correction
        c["voltage_mag"] = np.sqrt(np.square(c["voltage_ch1"]) + np.square(c["voltage_ch2"]))
        c["voltage_phase"] = np.arctan2(c["voltage_ch1"], c["voltage_ch2"])
        c["raw_voltage_ch1"] = c["raw_digital_ch1_mean"]*correction
        c["raw_voltage_ch2"] = c["raw_digital_ch2_mean"]*correction
        c["raw_voltage_mag"] = np.sqrt(np.square(c["raw_voltage_ch1"]) + np.square(c["raw_voltage_ch2"]))
        c["raw_voltage_phase"] = np.arctan2(c["raw_voltage_ch1"], c["raw_voltage_ch2"])
        c["n_voltage_ch1"] = dB2Volts(attRF - 25)*c["n_digital_ch1_mean"]*correction
        c["n_voltage_ch2"] = dB2Volts(attRF - 25)*c["n_digital_ch2_mean"]*correction
        c["n_voltage_mag"] = np.sqrt(np.square(c["n_voltage_ch1"]) + np.square(c["n_voltage_ch2"]))
        c["n_voltage_phase"] = np.arctan2(c["n_voltage_ch1"], c["n_voltage_ch2"])
        c["rel1_diff_mag"] = np.sqrt(np.square(c["rel1_diff_ch1"]) + np.square(c["rel1_diff_ch2"]))
        c["rel1_diff_phase"] = np.arctan2(c["rel1_diff_ch1"], c["rel1_diff_ch2"])
        c["rel2_diff_mag"] = np.sqrt(np.square(c["rel2_diff_ch1"]) + np.square(c["rel2_diff_ch2"]))
        c["rel2_diff_phase"] = np.arctan2(c["rel2_diff_ch1"], c["rel2_diff_ch2"])

        if correction == np.around(1.0e3/8192,4):
            c["voltage_unit"] = "mV"
        elif correction == np.around(1.0/8192,4):
            c["voltage_unit"] = "V"
        else:
            c["voltage_unit"] =  "converted by factor {}".format(correction)

        if "digital_eq_ch1" in c.columns:
            c["voltage_eq_ch1"] = c["digital_eq_ch1"]*correction
            c["voltage_eq_ch2"] = c["digital_eq_ch2"]*correction
            c["voltage_eq_mag"] = np.sqrt(np.square(c["voltage_eq_ch1"]) + np.square(c["voltage_eq_ch2"]))
            c["voltage_eq_phase"] = np.arctan2(c["voltage_eq_ch1"], c["voltage_eq_ch2"])
            c["raw_voltage_eq_ch1"] = c["raw_digital_eq_ch1"]*correction
            c["raw_voltage_eq_ch2"] = c["raw_digital_eq_ch2"]*correction
            c["raw_voltage_eq_mag"] = np.sqrt(np.square(c["raw_voltage_eq_ch1"]) + np.square(c["raw_voltage_eq_ch2"]))
            c["raw_voltage_eq_phase"] = np.arctan2(c["raw_voltage_eq_ch1"], c["raw_voltage_eq_ch2"])

            if correction == np.around(1.0e3/8192,4):
                c["eq_voltage_unit"] = "mV"
            elif correction == np.around(1.0/8192,4):
                c["eq_voltage_unit"] = "V"
            else:
                c["eq_voltage_unit"] =  "converted by factor {}".format(correction)

        compared.append(c.reset_index())

    return compared, groups_list


def pairwise_comparisons2(df, different_attRF = True, different_attLO = False, between_phantoms = False, between_dates = True, correction = np.around(1.0e3/8192,4)):
    """Compare pairs of phantom scans on Pandas data frames. This version computes differences in magnitude and phase directly instead of going back to I-Q vector.

    Possible to select whether to allow comparison between different attRF, attLO, phantoms/angles or dates.

    Comparing between different attLO is NOT reccomended, because the gain variation on the final recorded measurements is unpredictable.

    Parameters
    ----------
    df : Pandas df or list of df
        input dataframe or list of dataframes
    different_attRF : bool, optional
        set to true to enable different attRF comparisons, by default True
    different_attLO : bool, optional
            set to true to enable different attLO comparisons (not reccomended), by default False
    between_phantoms : bool, optional
            set to true to enable comparisons between different phantoms or positions (angles), by default False
    between_dates : bool, optional
            set to true to enable comparisons between different dates, by default True
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files

    Returns
    ----------
    group_list: list
        list of groups, composed of permutations of dataframes grouped by columns ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"]

        In case there are different attRF, attLO or phantoms and the respective inputs are set to False, it is a list of lists. 

        Each sub-list contains tuples with the column values.
    """

    if not isinstance(df, list):
        df = [df]

    if not between_phantoms:
        groups = []
        for data in df:
            if data.phantom.nunique() > 1:
                g = data.groupby(by = ["phantom"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not between_dates:
        groups = []
        for data in df:
            if data.date.nunique() > 1:
                g = data.groupby(by = ["date"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not different_attLO:
        groups = []
        for data in df:
            if data.attLO.nunique() > 1:
                g = data.groupby(by = ["attLO"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not different_attRF:
        groups = []
        for data in df:
            if data.attRF.nunique() > 1:
                g = data.groupby(by = ["attRF"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    else:
        df = [attenuation_match(data, decimals = 0, correction = correction) for data in df]

    groups_list = []
    compared = []

    for data in tqdm(df):
        if "voltage_unit" in data.columns:
            vunit = data.voltage_unit.unique().tolist()[0]
        data = data.loc[:,~data.columns.str.contains('subject', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('unit', case=False, na=False)]

        data_gr = data.groupby(by = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"], observed=True)
        groups = [name for name,unused_df in data_gr]

        g_out = []

        for p in it.permutations(groups,2):
            # excluding reversed/redundant pairs, i.e. if (A,B) then no need for (B,A)
            if p <= p[::-1]:
                g_out.append(p)

        groups_list.append([g_out])

        if len(g_out) < 1:
            # skip group comparison for less than one group
            continue

        if "distances".casefold() in data.columns:
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "distances"]
        else:
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq"]
        data = data.set_index(keys = cols, drop=True).sort_index()
        c_list = []

        for g in g_out:
            attRF = max(g[0][7],g[1][7])
            c = data.loc[g[0]].subtract(data.loc[g[1]], fill_value=0)
            # relative differences (using voltage units)
            c["rel1_diff"] = 100.00*c.voltage_mag_mean.divide(data.loc[g[0],"voltage_mag"], fill_value=0)
            c["rel2_diff"] = -100.00*c.voltage_mag_mean.divide(data.loc[g[1],"voltage_mag"], fill_value=0)

            col_names = ["phantom_1", "angle_1", "plug_1", "date_1", "rep_1", "iter_1", "attLO_1", "attRF_1", "phantom_2", "angle_2", "plug_2", "date_2", "rep_2", "iter_2", "attLO_2", "attRF_2"]
            values = g[0] + g[1]

            for i, col in enumerate(col_names):
                c[col] = values[i]

            c = name_case_comparisons(c)
            c_list.append(c.copy())

        c = pd.concat(c_list)

        #c["voltage_ch1"] = c["digital_ch1_mean"]*correction
        #c["voltage_ch2"] = c["digital_ch2_mean"]*correction
        #c["voltage_mag"] = np.sqrt(np.square(c["voltage_ch1"]) + np.square(c["voltage_ch2"]))
        #c["voltage_phase"] = np.arctan2(c["voltage_ch1"], c["voltage_ch2"])
        #c["raw_voltage_ch1"] = c["raw_digital_ch1_mean"]*correction
        #c["raw_voltage_ch2"] = c["raw_digital_ch2_mean"]*correction
        #c["raw_voltage_mag"] = np.sqrt(np.square(c["raw_voltage_ch1"]) + np.square(c["raw_voltage_ch2"]))
        #c["raw_voltage_phase"] = np.arctan2(c["raw_voltage_ch1"], c["raw_voltage_ch2"])
        #c["n_voltage_ch1"] = dB2Volts(attRF - 25)*c["n_digital_ch1_mean"]*correction
        #c["n_voltage_ch2"] = dB2Volts(attRF - 25)*c["n_digital_ch2_mean"]*correction
        #c["n_voltage_mag"] = np.sqrt(np.square(c["n_voltage_ch1"]) + np.square(c["n_voltage_ch2"]))
        #c["n_voltage_phase"] = np.arctan2(c["n_voltage_ch1"], c["n_voltage_ch2"])
        #c["rel1_diff_mag"] = np.sqrt(np.square(c["rel1_diff_ch1"]) + np.square(c["rel1_diff_ch2"]))
        #c["rel1_diff_phase"] = np.arctan2(c["rel1_diff_ch1"], c["rel1_diff_ch2"])
        #c["rel2_diff_mag"] = np.sqrt(np.square(c["rel2_diff_ch1"]) + np.square(c["rel2_diff_ch2"]))
        #c["rel2_diff_phase"] = np.arctan2(c["rel2_diff_ch1"], c["rel2_diff_ch2"])

        if correction == np.around(1.0e3/8192,4):
            c["voltage_unit"] = "mV"
        elif correction == np.around(1.0/8192,4):
            c["voltage_unit"] = "V"
        else:
            c["voltage_unit"] =  "converted by factor {}".format(correction)

        if "digital_eq_ch1" in c.columns:
        #    c["voltage_eq_ch1"] = c["digital_eq_ch1"]*correction
        #    c["voltage_eq_ch2"] = c["digital_eq_ch2"]*correction
        #    c["voltage_eq_mag"] = np.sqrt(np.square(c["voltage_eq_ch1"]) + np.square(c["voltage_eq_ch2"]))
        #    c["voltage_eq_phase"] = np.arctan2(c["voltage_eq_ch1"], c["voltage_eq_ch2"])
        #    c["raw_voltage_eq_ch1"] = c["raw_digital_eq_ch1"]*correction
        #    c["raw_voltage_eq_ch2"] = c["raw_digital_eq_ch2"]*correction
        #    c["raw_voltage_eq_mag"] = np.sqrt(np.square(c["raw_voltage_eq_ch1"]) + np.square(c["raw_voltage_eq_ch2"]))
        #    c["raw_voltage_eq_phase"] = np.arctan2(c["raw_voltage_eq_ch1"], c["raw_voltage_eq_ch2"])

            if correction == np.around(1.0e3/8192,4):
                c["eq_voltage_unit"] = "mV"
            elif correction == np.around(1.0/8192,4):
                c["eq_voltage_unit"] = "V"
            else:
                c["eq_voltage_unit"] =  "converted by factor {}".format(correction)

        compared.append(c.reset_index())

    return compared, groups_list

def pairwsise_comparisons_of_medians(df, different_attRF = True, different_attLO = False, between_phantoms = False, between_dates = True, correction = np.around(1.0e3/8192,4), center = 'median'):
    """Compare pairs of phantom scans on Pandas data frames, using the median/mean for each date.

    This version aggregates over all repetitions and iterations for a single date, attemtping to identify the central tendency.

    Possible to select whether to allow comparison between different attRF, attLO, phantoms/angles or dates.

    Comparing between different attLO is NOT reccomended, because the gain variation on the final recorded measurements is unpredictable.

    Parameters
    ----------
    df : Pandas df or list of df
        input dataframe or list of dataframes
    different_attRF : bool, optional
        set to true to enable different attRF comparisons, by default True
    different_attLO : bool, optional
            set to true to enable different attLO comparisons (not reccomended), by default False
    between_phantoms : bool, optional
            set to true to enable comparisons between different phantoms or positions (angles), by default False
    between_dates : bool, optional
            set to true to enable comparisons between different dates, by default True
    correction : float, optional
        conversion scale factor for digital scale data, by default np.around(1.0e3/8192,4)
        default is equivalent to 0.1831 mV per ADC unit
        set to 1 for no converted files
    center: str, optional
        center tendency option between 'mean' or 'median', by default 'mean'

    Returns
    ----------
    group_list: list
        list of groups, composed of permutations of dataframes grouped by columns ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF"]

        In case there are different attRF, attLO or phantoms and the respective inputs are set to False, it is a list of lists.

        Each sub-list contains tuples with the column values.
    """

    if not isinstance(df, list):
        df = [df]

    if center == 'median':
        agg_center = 'median'
    else:
        agg_center = 'mean'

    if not between_phantoms:
        groups = []
        for data in df:
            if data.phantom.nunique() > 1:
                g = data.groupby(by = ["phantom"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not between_dates:
        groups = []
        for data in df:
            if data.date.nunique() > 1:
                g = data.groupby(by = ["date"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not different_attLO:
        groups = []
        for data in df:
            if data.attLO.nunique() > 1:
                g = data.groupby(by = ["attLO"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    if not different_attRF:
        groups = []
        for data in df:
            if data.attRF.nunique() > 1:
                g = data.groupby(by = ["attRF"], observed=True)
                groups = groups + [g.get_group(x).reset_index(drop=True) for x in g.groups]
            else:
                groups = groups + [data]
        df = groups

    else:
        df = [attenuation_match(data, decimals = 0, correction = correction) for data in df]

    groups_list = []
    compared = []

    for data in tqdm(df):
        if "voltage_unit" in data.columns:
            vunit = data.voltage_unit.unique().tolist()[0]
        data = data.loc[:,~data.columns.str.contains('subject', case=False, na=False)]
        data = data.loc[:,~data.columns.str.contains('unit', case=False, na=False)]

        data_gr = data.groupby(by = ["phantom", "angle", "plug", "date", "attLO", "attRF"], observed=True)
        groups = [name for name,unused_df in data_gr]

        g_out = []

        for p in it.permutations(groups,2):
            # excluding reversed/redundant pairs, i.e. if (A,B) then no need for (B,A)
            if p <= p[::-1]:
                g_out.append(p)

        groups_list.append([g_out])

        if len(g_out) < 1:
            # skip group comparison for less than one group
            continue

        if "distances".casefold() in data.columns:
            cols = ["phantom", "angle", "plug", "date", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "distances"]
        else:
            cols = ["phantom", "angle", "plug", "date", "attLO", "attRF", "pair", "Tx", "Rx", "freq"]
        data = data.drop(["rep", "iter"], axis = 1).groupby(by = cols, observed=True).agg(['median', 'mean', 'std', 'var']).reset_index()
        data.columns = ['_'.join(e.replace("_mean", "") for e in x).rstrip('_') if isinstance(x,tuple) else x for x in data.columns.ravel()]
        cols_to_sub = data.columns[data.columns.str.contains('(?:median|mean).*', case=False, na=False)].tolist()
        cols_to_copy = data.columns[data.columns.str.contains('(?:std|var).*', case=False, na=False)].tolist()
        data = data.set_index(keys = cols, drop=True).sort_index()
        c_list = []

        for g in g_out:
            attRF = max(g[0][5],g[1][5])
            c = data.xs(g[0], drop_level = True)[cols_to_sub].subtract(data.xs(g[1], drop_level = True)[cols_to_sub], fill_value=0)
            #c = data.xs(g[0], level = cols_to_sub, axis=1).subtract(data.xs(g[1], level = cols_to_sub, axis=1), fill_value=0)

            # relative differences (using voltage units)
            c["rel1_diff_median"] = 100.00*c.voltage_mag_median.divide(data.loc[g[0],"voltage_mag_median"], fill_value=0)
            c["rel2_diff_median"] = -100.00*c.voltage_mag_median.divide(data.loc[g[1],"voltage_mag_median"], fill_value=0)

            c["rel1_diff_mean"] = 100.00*c.voltage_mag_mean.divide(data.loc[g[0],"voltage_mag_mean"], fill_value=0)
            c["rel2_diff_mean"] = -100.00*c.voltage_mag_mean.divide(data.loc[g[1],"voltage_mag_mean"], fill_value=0)

            for col in cols_to_copy:
                c[col + '_1'] = data.loc[g[0], col]
                c[col + '_2'] = data.loc[g[1], col]

            col_names = ["phantom_1", "angle_1",  "plug_1", "date_1","attLO_1", "attRF_1", "phantom_2", "angle_2", "plug_2", "date_2", "attLO_2", "attRF_2"]
            values = g[0] + g[1]

            for i, col in enumerate(col_names):
                c[col] = values[i]

            c = name_case_comparisons(c)
            c_list.append(c.copy())

        c = pd.concat(c_list)

        #c["voltage_ch1"] = c["digital_ch1_mean"]*correction
        #c["voltage_ch2"] = c["digital_ch2_mean"]*correction
        #c["voltage_mag"] = np.sqrt(np.square(c["voltage_ch1"]) + np.square(c["voltage_ch2"]))
        #c["voltage_phase"] = np.arctan2(c["voltage_ch1"], c["voltage_ch2"])
        #c["raw_voltage_ch1"] = c["raw_digital_ch1_mean"]*correction
        #c["raw_voltage_ch2"] = c["raw_digital_ch2_mean"]*correction
        #c["raw_voltage_mag"] = np.sqrt(np.square(c["raw_voltage_ch1"]) + np.square(c["raw_voltage_ch2"]))
        #c["raw_voltage_phase"] = np.arctan2(c["raw_voltage_ch1"], c["raw_voltage_ch2"])
        #c["n_voltage_ch1"] = dB2Volts(attRF - 25)*c["n_digital_ch1_mean"]*correction
        #c["n_voltage_ch2"] = dB2Volts(attRF - 25)*c["n_digital_ch2_mean"]*correction
        #c["n_voltage_mag"] = np.sqrt(np.square(c["n_voltage_ch1"]) + np.square(c["n_voltage_ch2"]))
        #c["n_voltage_phase"] = np.arctan2(c["n_voltage_ch1"], c["n_voltage_ch2"])
        #c["rel1_diff_mag"] = np.sqrt(np.square(c["rel1_diff_ch1"]) + np.square(c["rel1_diff_ch2"]))
        #c["rel1_diff_phase"] = np.arctan2(c["rel1_diff_ch1"], c["rel1_diff_ch2"])
        #c["rel2_diff_mag"] = np.sqrt(np.square(c["rel2_diff_ch1"]) + np.square(c["rel2_diff_ch2"]))
        #c["rel2_diff_phase"] = np.arctan2(c["rel2_diff_ch1"], c["rel2_diff_ch2"])

        if correction == np.around(1.0e3/8192,4):
            c["voltage_unit"] = "mV"
        elif correction == np.around(1.0/8192,4):
            c["voltage_unit"] = "V"
        else:
            c["voltage_unit"] =  "converted by factor {}".format(correction)

        if "digital_eq_ch1" in c.columns:
            c["voltage_eq_ch1"] = c["digital_eq_ch1"]*correction
            c["voltage_eq_ch2"] = c["digital_eq_ch2"]*correction
            c["voltage_eq_mag"] = np.sqrt(np.square(c["voltage_eq_ch1"]) + np.square(c["voltage_eq_ch2"]))
            c["voltage_eq_phase"] = np.arctan2(c["voltage_eq_ch1"], c["voltage_eq_ch2"])
            c["raw_voltage_eq_ch1"] = c["raw_digital_eq_ch1"]*correction
            c["raw_voltage_eq_ch2"] = c["raw_digital_eq_ch2"]*correction
            c["raw_voltage_eq_mag"] = np.sqrt(np.square(c["raw_voltage_eq_ch1"]) + np.square(c["raw_voltage_eq_ch2"]))
            c["raw_voltage_eq_phase"] = np.arctan2(c["raw_voltage_eq_ch1"], c["raw_voltage_eq_ch2"])

            if correction == np.around(1.0e3/8192,4):
                c["eq_voltage_unit"] = "mV"
            elif correction == np.around(1.0/8192,4):
                c["eq_voltage_unit"] = "V"
            else:
                c["eq_voltage_unit"] =  "converted by factor {}".format(correction)

        compared.append(c.reset_index())

    return compared, groups_list

def name_case_comparisons(df):
    """Create "cases" column on Pandas data frame, naming them, such as "baseline-baseline".

    List of possible cases:

    - "baseline-baseline"
    - "baseline-tumor"
    - "tumor-tumor"
    - "air-air"
    - "baseline-air"
    - "tumor-air"
    - "baseline1-baseline2" (different baseline plugs)
    - "tumor1-tumor2" (different tumor plugs)

    Parameters
    ----------
    df : Pandas df
        input dataframe

    Returns
    ----------
    df: Pandas df
        dataframe with "cases" column
    """

    baselines = list(range(1,8)) + [22,302]
    tumors = list(range(10,22)) + list(range(30,39)) + [310]

    if df.plug_1.equals(df.plug_2):
        if df.plug_1.isin(baselines).all():
            df["cases"] = "baseline-baseline"
        elif df.plug_1.isin(tumors).all():
            df["cases"] = "tumor-tumor"
        elif df.plug_1.eq(0).all():
            df["cases"] = "air-air"
    elif df.plug_1.eq(0).all() or df.plug_2.eq(0).all():
        if df.plug_1.isin(baselines).all() or df.plug_2.isin(tumors).all():
            df["cases"] = "baseline-air"
        elif df.plug_1.isin(tumors).all() or df.plug_2.isin(tumors).all():
            df["cases"] = "tumor-air"
    elif df.plug_1.isin(baselines).all() and df.plug_2.isin(baselines).all():
        df["cases"] = "baseline1-baseline2"
    elif df.plug_1.isin(tumors).all() and df.plug_2.isin(tumors).all():
        df["cases"] = "tumor1-tumor2"
    else:
        df["cases"] = "baseline-tumor"

    return df


def specific_comparison(scan1, scan2, comp_columns = "power_dBm"):
    """Perform comparison between two phantom scans.

    Comparison consists of a subtraction (scan1[comp_column] - scan2[comp_column]).

    Parameters
    ----------
    scan1 : Pandas df
        input dataframe 1
    scan2 : Pandas df
        input dataframe 2
    comp_column : str or list of str
        column name(s) to be compared

    Returns
    ----------
    df: Pandas df
        dataframe with "diff" column
    """

    if not isinstance(comp_columns, list):
        comp_columns = [comp_columns]

    # retain only intersection of antenna pairs and frequencies
    scan1, scan2 = remove_non_intersection(scan1, scan2, column = "pair")
    if ("freq".casefold() in scan1.columns) and ("freq".casefold() in scan1.columns):
        scan1, scan2 = remove_non_intersection(scan1, scan2, column = "freq")
    elif ("time".casefold() in scan1.columns) and ("time".casefold() in scan1.columns):
        scan1, scan2 = remove_non_intersection(scan1, scan2, column = "time")

    scan1, scan2 = attenuation_match2(scan1, scan2, decimals = 0, correction = np.around(1.0e3/8192,4))

    if "power_dBm" in comp_columns:
        # check if power_dBm is present and otherwise calculates it
        if (not scan1.columns.isin(["power_dBm"]).all()):
            dfproc.calculate_power_dBm(scan1, Z = 50.0, noise_floor = -108, inplace=True)
        elif (not scan1.columns.isin(comp_columns).all()):
            print(f"Some comparison column {comp_columns} not present in first dataframe!")
            return 0
        if (not scan2.columns.isin(["power_dBm"]).all()):
            dfproc.calculate_power_dBm(scan2, Z = 50.0, noise_floor = -108, inplace=True)
        elif (not scan1.columns.isin(comp_columns).all()):
            print(f"Some comparison column {comp_columns} not present in second dataframe!")
            return 0

    if ("freq".casefold() in scan1.columns) and ("freq".casefold() in scan1.columns):
        if ("distances".casefold() in scan1.columns) and ("distances".casefold() in scan2.columns):
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq", "distances"]
        else:
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "freq"]
    elif ("time".casefold() in scan1.columns) and ("time".casefold() in scan1.columns):
        if ("distances".casefold() in scan1.columns) and ("distances".casefold() in scan2.columns):
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "time", "distances"]
        else:
            cols = ["phantom", "angle", "plug", "date", "rep", "iter", "attLO", "attRF", "pair", "Tx", "Rx", "time"]
    else:
        print("Dataframes types are incompatible (Frequency Domain vs. Time Domain).")
        return 0
    scan1 = scan1.set_index(keys = cols, drop=True).sort_index()
    scan2 = scan2.set_index(keys = cols, drop=True).sort_index()

    c = pd.DataFrame().reindex(columns=scan1.columns)

    scan1_gr = scan1.groupby(by = cols[0:8], observed=True)
    groups1 = [name for name,unused_df in scan1_gr]
    scan2_gr = scan2.groupby(by = cols[0:8], observed=True)
    groups2 = [name for name,unused_df in scan2_gr]

    g_out = it.product(groups1,groups2)

    c_list = []

    for g in tqdm(g_out):

        # attRF = max(g[0][7],g[1][7])
        for col in comp_columns:
            c.loc[:,"_".join((col,"diff"))] = scan1.loc[g[0],col].subtract(scan2.loc[g[1],col], fill_value=0)

        col_names = ["phantom_1", "angle_1", "plug_1", "date_1", "rep_1", "iter_1", "attLO_1", "attRF_1", "phantom_2", "angle_2", "plug_2", "date_2", "rep_2", "iter_2", "attLO_2", "attRF_2"]
        values = g[0] + g[1]

        for i, coln in enumerate(col_names):
            c[coln] = values[i]

        c = name_case_comparisons(c)
        c_list.append(c.copy())

    res_df = pd.concat(c_list)
    res_df.dropna(axis=1, how='all', inplace=True)
    res_df.reset_index(inplace=True)

    return res_df

def remove_non_intersection(df1, df2, column = "pair", inplace=False):
    """Perform intersection on column between two dataframes and removes rows not in common.

    Parameters
    ----------
    df1 : Pandas df
        input dataframe 1
    df2 : Pandas df
        input dataframe 2
    column : str, optional
        column name, by default "pair"
    inplace : bool, optional
        set to True to perform in-place, by default False

    Returns (when inplace = False)
    ----------
    df1: Pandas df
        dataframe 1 with only intersected elements from selected column
    df2: Pandas df
        dataframe 2 with only intersected elements from selected column
    """

    col1 = set(df1[column].unique())
    col2 = set(df2[column].unique())

    intersection = natsorted(list(col1.intersection(col2)))

    df1 = df1.loc[df1[column].isin(intersection)]
    df2 = df2.loc[df2[column].isin(intersection)]

    if not inplace:
        return df1, df2

def split(df, group):
    """Return list of dataframes split in groups.

    From Jeff Mandell on Stackoverflow:

    https://stackoverflow.com/questions/23691133/split-pandas-dataframe-based-on-groupby

    Parameters
    ----------
    df : Pnadas dataframe
        input datafrane
    group : list
        list of groups to split

    Returns
    -------
    list
        list of split groups
    """
    gb = df.groupby(group)
    return [gb.get_group(x) for x in gb.groups]

def attenuation_match(df, decimals = 0, correction = np.around(1.0e3/8192,4)):
    """Output dataframe with new data columns after matching RF attenuation to the highest case for stats.

        The match is performed by leveling to the highest "attRF" value (lowest output voltages), in order to prevent extrapolations
        beyond the measurement resolution.

        Parameters
        ----------
        df : Pandas dataframe
            input dataframe
        decimals : int, optional
            number of decimals for rounding, by default 0
        correction : float, optional
            value for magnitude conversion, by default np.around(1.0e3/8192,4)

        Returns
        ----------
        df : Pandas dataframe
            dataframe with new columns for data equalized  to the max attRF
    """

    if df.attRF.nunique() == 1:
        return df
    else:
        max_attRF = df.attRF.max()

        df["attRF_eq"] = max_attRF

        df["digital_eq_ch1"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["digital_ch1_mean"]
        df["digital_eq_ch2"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["digital_ch2_mean"]
        df["raw_digital_eq_ch1"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["raw_digital_ch1_mean"]
        df["raw_digital_eq_ch2"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["raw_digital_ch2_mean"]

        #df["digital_eq_ch1"] = df.digital_eq_ch1.round(0)
        #df["digital_eq_ch2"] = df.digital_eq_ch2.round(0)
        #df["raw_digital_eq_ch1"] = df.raw_digital_eq_ch1.round(0)
        #df["raw_digital_eq_ch2"] = df.raw_digital_eq_ch2.round(0)

        if correction != 1:
            df["voltage_eq_ch1"] = df["digital_eq_ch1"]*correction
            df["voltage_eq_ch2"] = df["digital_eq_ch2"]*correction
            df["voltage_eq_mag"] = np.sqrt(np.square(df["voltage_eq_ch1"]) + np.square(df["voltage_eq_ch2"]))
            df["voltage_eq_phase"] = np.arctan2(df["voltage_eq_ch1"], df["voltage_eq_ch2"])
            df["raw_voltage_eq_ch1"] = df["raw_digital_eq_ch1"]*correction
            df["raw_voltage_eq_ch2"] = df["raw_digital_eq_ch2"]*correction
            df["raw_voltage_eq_mag"] = np.sqrt(np.square(df["raw_voltage_eq_ch1"]) + np.square(df["raw_voltage_eq_ch2"]))
            df["raw_voltage_eq_phase"] = np.arctan2(df["raw_voltage_eq_ch1"], df["raw_voltage_eq_ch2"])

            if correction == np.around(1.0e3/8192,4):
                df["eq_voltage_unit"] = "mV"
            elif correction == np.around(1.0/8192,4):
                df["eq_voltage_unit"] = "V"
            else:
                df["eq_voltage_unit"] =  "converted by factor {}".format(correction)

        return df

def attenuation_match2(df1, df2, decimals = 0, correction = np.around(1.0e3/8192,4), convert2power_dBm = True):
    """Output 2 dataframes with new data columns after matching RF attenuation to the highest case for stats.

        The match is performed by leveling to the highest "attRF" value (lowest output voltages), in order to prevent extrapolations
        beyond the measurement resolution.

        Parameters
        ----------
        df1 : Pandas dataframe
            input dataframe 1
        df2 : Pandas dataframe
            input dataframe 2
        decimals : int, optional
            number of decimals for rounding, by default 0
        correction : float, optional
            value for magnitude conversion, by default np.around(1.0e3/8192,4)

        Returns
        ----------
        df : Pandas dataframe
            dataframe with new columns for data equalized  to the max attRF
    """

    if (df1.attRF.nunique() == 1) and (df2.attRF.nunique() == 1) and (df1.attRF.unique() == df2.attRF.unique()):
        return df1, df2
    else:
        max_attRF = max(df1.attRF.max(),df2.attRF.max())

        for df in [df1, df2]:

            df["attRF_eq"] = max_attRF

            df["digital_eq_ch1"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["digital_ch1_mean"]
            df["digital_eq_ch2"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["digital_ch2_mean"]
            df["raw_digital_eq_ch1"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["raw_digital_ch1_mean"]
            df["raw_digital_eq_ch2"] =  (dB2Volts(df["attRF"] - max_attRF)) * df["raw_digital_ch2_mean"]

            #df["digital_eq_ch1"] = df.digital_eq_ch1.round(0)
            #df["digital_eq_ch2"] = df.digital_eq_ch2.round(0)
            #df["raw_digital_eq_ch1"] = df.raw_digital_eq_ch1.round(0)
            #df["raw_digital_eq_ch2"] = df.raw_digital_eq_ch2.round(0)

            if correction != 1:
                df["voltage_eq_ch1"] = df["digital_eq_ch1"]*correction
                df["voltage_eq_ch2"] = df["digital_eq_ch2"]*correction
                df["voltage_eq_mag"] = np.sqrt(np.square(df["voltage_eq_ch1"]) + np.square(df["voltage_eq_ch2"]))
                df["voltage_eq_phase"] = np.arctan2(df["voltage_eq_ch1"], df["voltage_eq_ch2"])
                df["raw_voltage_eq_ch1"] = df["raw_digital_eq_ch1"]*correction
                df["raw_voltage_eq_ch2"] = df["raw_digital_eq_ch2"]*correction
                df["raw_voltage_eq_mag"] = np.sqrt(np.square(df["raw_voltage_eq_ch1"]) + np.square(df["raw_voltage_eq_ch2"]))
                df["raw_voltage_eq_phase"] = np.arctan2(df["raw_voltage_eq_ch1"], df["raw_voltage_eq_ch2"])

                if correction == np.around(1.0e3/8192,4):
                    df["eq_voltage_unit"] = "mV"
                elif correction == np.around(1.0/8192,4):
                    df["eq_voltage_unit"] = "V"
                else:
                    df["eq_voltage_unit"] =  "converted by factor {}".format(correction)

                if convert2power_dBm == True:
                    dfproc.calculate_power_dBm(df, Z = 50.0, noise_floor = -108, inplace=True, voltage_col = "voltage_eq")

        return df1, df2

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