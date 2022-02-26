# Python 3.10.2
# 2022-02-25

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
 Description:
        Module for performing data processing on Pandas DataFrames for the ultrawideband system.

Classes::

        Scan_settings : Class for providing phantom scan info.

Functions::

        uwb_data_read : reads ultrawideband system data file and returns dataframe or numpy array.

        uwb_data_read2pandas : reads ultrawideband system data file and generates dataframe "scan data set" files from PicoScope .txt files, either saves it to a parquet file or outputs the dataframe.

        uwb_scan_folder_sweep : finds all phantom measurement .txt files in a given directory path. used for 

Written by: Leonardo Fortaleza
"""
# Standard library imports
import datetime
import os
import os.path
from pathlib import Path
import re

# Third party imports
from natsort import natsorted
import numpy as np
import pandas as pd
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook

class Scan_settings:
    """Class for providing phantom scan info.

    Can be initiated by performing:

    s = Scan_settings(phantom = 1, angle = 0, plug = 2, rep = 1, iter = 1, sampling_rate = 160e9, date = '2020-01-24', attRF = 0, HP_amp = 35, LNA_amp = 25, sig_names = ['transmission', 'signal'],
                    obs = '')

    Defaults include:
        phantom = 1
        angle = 0
        plug = 2
        rep = 1
        iter = 1
        sampling_rate = 160e9
        date = '2020-01-24'
        att = 0 # [dB]
        HP_amp = 35 # [dB]
        LNA_amp = 25 # [dB]
        sig_names = ['transmission', 'signal'] # (list of column names)
        obs = '' # (string for notes and further info)
    """
    def __init__(self, **kwargs):

        # Predefine attributes with default values
        self.phantom = 1
        self.angle = 0
        self.plug = 2
        self.rep = 1
        self.iter = 1
        self.sampling_rate = 160e9

        self.date = '2020-01-24'

        self.attRF = 0
        self.HP_amp = 35
        self.LNA_amp = 25

        self.sig_names = ['transmission', 'signal']
        self.obs = ''

        self.update_values(kwargs)

    def update_values(self, **kwargs):
        """Update class with keyword arguments.

        Checks for unexpected arguments, allowing them but displaying a warning.

        Raises
        ------
        Warning
           When unexpected keyword arguments are provided. Settings are applied but might be unused.
        """
        # get a list of all predefined values directly from __dict__
        allowed_keys = list(self.__dict__.keys())

        # # Update __dict__ but only for keys that have been predefined 
        # # (silently ignore others)
        # self.__dict__.update((key, value) for key, value in kwargs.items() 
        #                      if key in allowed_keys)

        # # To NOT silently ignore rejected keys
        # rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        # if rejected_keys:
        #     raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

        # Update __dict__  for any keys, but provides Warning

        # To NOT silently ignore rejected keys
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise Warning("Unexpected arguments in constructor Scan_settings:{}. Settings were applied but might be unused.".format(rejected_keys))
        self.__dict__.update(kwargs)


def uwb_data_read(file_name, output_numpy = False, col_names = ['transmission', 'signal']):
    """Read ultrawideband system data file and return data array.

    The function reads a PicoScope output .txt file, returning the data in array (or matrix) form.

    Parameters
    ----------
    file_name : str
        file name and path for .txt file in PicoScope format
    output_numpy : bool, optional
        set to True to output numpy array instead of Pandas DataFrame, by default False

    Returns
    ----------
    data : Pandas df or ndarray
        Pnadas DataFrame or array with PicoScope output values, rows are samples and columns are separate channel
    """

    data = pd.read_csv(file_name, sep=" ", header=None)
    data.columns = col_names

    if output_numpy:
        data = data.to_numpy()

    return data

def uwb_scan_folder_sweep(main_path):
    """Find all phantom measurement .txt files in a given directory path.

    In addition to a list with the string pathnames of each file found, the ndarray meta_index is generated.
    The ndarray meta_index associates parameters of Tx, Rx to each file (which is typically in the format 'sig_A[Tx]_A[Rx]_hw16.txt').

    Parameters
    ----------
    main_path : str
        main path (folder path) to .txt measurement files

    Returns
    ----------
    list of str
        list with full pathnames to each .txt file found
    meta_index : ndarray of int
        2-D list of parameters associated to each .txt file found
        Parameters are Tx, Rx
    """

    a = natsorted(Path(main_path).glob("*.txt"))
    path_list = [str(x) for x in a]

    ## meta_index is matrix with rows in the form [Antenna Pair Tx, Antenna Pair Rx]

    meta_index = [[int(re.search(r'^.*?_[Aa](\d+)',str(i)).group(0)),
                int(re.search(r'^.*?_[Aa][\d+]_[Aa](\d+)', str(i)).group(0))]
                for i in path_list]
    meta_index = np.array(meta_index)

    return path_list, meta_index

def uwb_data_read2pandas(main_path, out_path = "{}/OneDrive - McGill University/Documents McGill/Data/UWB/".format(os.environ['USERPROFILE']),
                            processed_path = "Processed/DF/", settings = None, save_file = True, save_format="parquet", parquet_engine= 'pyarrow', **kwargs):
    """Generate Pandas DataFrame "scan data set" files from PicoScope .txt files.

    In order to gather metadata, requires prior manual instancing of class Scan_settings, otherwise uses default values for Phantom, Angle, Plug, etc. 
    (Data files for UWB were not formatted to optimize automated data processing of multiple data scans, unlike NB files.)

    By default, saves DataFrame in parquet format to [out_path + processed_path], but can output the DataFrame if save_file is set to False.

    Parameters
    ----------
    main_path : str
        main path to measurement files
    out_path : str, optional
        initial path for output dataframe files, by default "{}/OneDrive - McGill University/Documents McGill/Data/UWB/".format(os.environ['USERPROFILE'])
    processed_path : str, optional
        sub-folder for output files, by default "Processed/DF"
        final location will be out_path + processed_path
    settings : class Scan_settings, optional
        receives Scan_settings class file, by default None
        kwargs apply to this class file.
        if None is provided, defaults include:
            phantom = 1
            angle = 0
            plug = 2
            rep = 1
            iter = 1
            sample_rate = 160e9
            date = '2020-01-24'
            att = 0 # [dB]
            HP_amp = 35 # [dB]
            LNA_amp = 25 # [dB]
    save_file : bool, optional
        set to False to output DataFrame instead of saving to file, by default True
    save_format: str, optional
        target file format (either "parquet" or "csv"), by default "parquet"
    parquet_engine: str, optional
        Parquet reader library to use, by default 'pyarrow'
        Options include: 'auto', 'fastparquet', 'pyarrow'.
        If 'auto', then the option io.parquet.engine is used.
        The default io.parquet.engine behavior is to try 'pyarrow',
        falling back to 'fastparquet' if 'pyarrow' is unavailable.
    **kwargs : dict, optional
        keyword arguments to apply to settings class (see 'settings' input above).

    Returns (optional)
    ----------
    df : Pandas df, optional
        function by default saves a file
        in case save_file is set to False, then outputs the resulting df
    """
    if settings is None:
        settings = Scan_settings(kwargs)
    elif kwargs:
        settings.update_values(kwargs)

    path_list, meta_index = uwb_scan_folder_sweep(main_path)

    data_list = []

    for p, i in tqdm(enumerate((path_list))):
        data = uwb_data_read(p, output_numpy = False, col_names = settings.sig_names)
        data['sample'] = np.arange(0, np.size(data, axis = 0))
        df["nsamples"] = np.size(data, axis = 0)
        data['Tx'] = meta_index[i, 0]
        data['Rx'] = meta_index[i, 1]
        data_list.append(data)

    df = pd.concat(data_list, axis = 0)

    df["system"] = 'UWB'
    df["file_type"] = "scan data set"

    df["phantom"] = settings.phantom
    df["angle"] = settings.angle
    df["plug"] = settings.plug
    df["date"] = settings.date
    df["rep"] = settings.rep
    df["iter"] = settings.iter

    df["attRF"] = settings.attRF
    df["HP_amp"] = settings.HP_amp
    df["LNA_amp"] = settings.LNA_amp

    df['samp_rate'] = settings.sampling_rate
    df['time'] = df['sample'] * (1/settings.sample_rate)
    df["obs"] = settings.obs

    if save_file:

        file_title = f"{settings.date} Phantom {settings.phantom} Angle {settings.angle} Plug {settings.plug} Rep {settings.rep} Iter {settings.iter}"

        if save_format.casefold() == "parquet":
            if parquet_engine == 'pyarrow':
                df.to_parquet("".join((out_path,processed_path, file_title, ".parquet")), engine=parquet_engine, index= False)
            else:
                df.to_parquet("".join((out_path,processed_path, file_title, ".parquet")), engine=parquet_engine, object_encoding='utf8', write_index= False)
        else:
            df.to_csv("".join((out_path,processed_path, file_title, ".csv")))
        tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")
        return

    else:
        return df