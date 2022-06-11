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
from contextlib import redirect_stderr
import io

# Third party imports
from scipy import signal
from natsort import natsorted
import numpy as np
import pandas as pd
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook

# Local imports
from UWB.analysis_pd.matlab_bandpass import matlab_bandpass

class Scan_settings:
    """Class for providing phantom scan info.

    Can be initiated by performing:

    s = Scan_settings(phantom = 1, angle = 0, plug = 2, rep = 1, iter = 1, sampling_rate = 160e9, date = '2020-01-24', attRF = 0, HP_amp = 35, LNA_amp = 25, 
                        f_low = 1.7e9, f_high = 4e9, sig_names = ['raw_signal', 'raw_transmission'], array_config = 'hemisphere', obs = '')

    Defaults include:
        phantom = 1
        angle = 0
        plug = 2
        rep = 1
        iter = 1
        sampling_rate = 160e9 # [samples/second]
        f_low = 1.7e9 # [Hertz]
        f_high = 4e9 # [Hertz]
        date = '2020-01-24'
        att = 0 # [dB]
        HP_amp = 35 # [dB]
        LNA_amp = 25 # [dB]
        sig_names = ['raw_signal', 'raw_transmission'] # (list of column names)
        array_config = 'hemisphere'
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

        self.f_low = 1.7e9
        self.f_high = 4e9

        self.date = '2020-01-24'

        self.attRF = 0
        self.HP_amp = 35
        self.LNA_amp = 25

        self.sig_names = ['raw_signal', 'raw_transmission']
        self.array_config = 'hemisphere'
        self.obs = ''

        self.update_values(**kwargs)

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


def uwb_data_read(file_name, output_numpy = False, col_names = ['raw_signal', 'raw_transmission'], nafill = 0, keep_default_na = True, on_bad_lines = 'warn'):
    """Read ultrawideband system data file and return data array.

    The function reads a PicoScope output .txt file, returning the data in array (or matrix) form.

    Parameters
    ----------
    file_name : str
        file name and path for .txt file in PicoScope format
    output_numpy : bool, optional
        set to True to output numpy array instead of Pandas DataFrame, by default False
    fillna: None or any, default 0
        set to None to maintain NaNs (if keep_default_na is True), otherwise set to value that replaces NaNs.
    keep_default_na: bool, default True
        Whether or not to include the default NaN values when parsing the data. Depending on whether na_values is passed in, the behavior is as follows:

        If keep_default_na is True, and na_values are specified, na_values is appended to the default NaN values used for parsing.

        If keep_default_na is True, and na_values are not specified, only the default NaN values are used for parsing.

        If keep_default_na is False, and na_values are specified, only the NaN values specified na_values are used for parsing.

        If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.

        Note that if na_filter is passed in as False, the keep_default_na and na_values parameters will be ignored.
    on_bad_lines: {'error', 'warn', 'skip'} or callable, default 'warn'
        Specifies what to do upon encountering a bad line (a line with too many fields). Allowed values are:

        'error', raise an Exception when a bad line is encountered.

        'warn', raise a warning when a bad line is encountered and skip that line.

        'skip', skip bad lines without raising or warning when they are encountered.

    Returns
    ----------
    data : Pandas df or ndarray
        Pnadas DataFrame or array with PicoScope output values, rows are samples and columns are separate channel
    """

    f = io.StringIO()
    with redirect_stderr(f):
        data = pd.read_csv(file_name, delim_whitespace=True, header=None,  names = col_names,
                        keep_default_na = keep_default_na, on_bad_lines = on_bad_lines)
    if f.getvalue(): # checking warnings/errors for more information
        tqdm.write(f"Parsing errors on {Path(file_name).stem}: {f.getvalue()}")

    if nafill is not None:
        data.fillna(nafill, inplace=True) # Replaces NaN with fillna (default 0)

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

    meta_index = [[int(re.search(r'^.*?_[Aa](\d+)',str(i)).group(1)),
                int(re.search(r'^.*?_[Aa]\d+_[Aa](\d+)', str(i)).group(1))]
                for i in path_list]
    meta_index = np.array(meta_index)

    return path_list, meta_index

def uwb_filter_signals(df, input_col_names = ['raw_signal'], output_col_names = ['signal'], start_pt = None, nSamples = 1000, detrend = True):
    """Apply bandpass and detrend filters to uwb system raw signals, with optional prior windowing of samples.

    Uses matlab_bandpass to emulate MATLAB's bandpass function. Detrend is performed with scipy.signal.detrend(x, type = 'linear').

    Attention: if start_pt and nSamples are used to window, the output DataFrame has dropped unused rows.

    Attempt to optimize function.

    Parameters
    ----------
    df : Pandas df
        input DataFrame with UWB data.
    input_col_names : list, optional
        input column names to filter, by default ['raw_signal']
    output_col_names : list, optional
        output column names for filtered signal, by default ['signal']
        IndexError occurs if len(output_col_names) ~= len(input_col_names).
    start_pt : int, optional
        initial sample for optional signal windowing, by default None
    nSamples : int, optional
        number of samples for optional signal windowing, by default 1000
        used only if start_pt is not None.
    detrend : bool, optional
        set to False to only apply bandpass filter, skipping subsequent linear detrending, by default True

    Returns
    -------
    Pandas df
        output DataFrame with UWB data including filtered signal
        Note: if start_pt and nSamples are used to window, the output DataFrame has dropped unused rows.
    """
    if (start_pt is not None):
        df = df.loc[df.samples.between(start_pt, start_pt + nSamples, inclusive=True)]

    f_low = df.loc[:, "f_low"].unique()[0]
    f_high = df.loc[:, "f_high"].unique()[0]
    samp_rate = df.loc[:, "samp_rate"].unique()[0]

    for i, col in enumerate(input_col_names): # performs routine for each input column
        # first extracts data to numpy array of shape (# of samples, # of pairs)
        # IMPORTANT: np.reshape and later np.flatten need to use Fortran order ('F'), otherwise signals get mixed up
        x = df[col].to_numpy(dtype = np.float64, copy = True).reshape(df[col].count() // df.pair.nunique(), df.pair.nunique(), order = 'F')
        rd = matlab_bandpass(x,
                                fpass = [f_low, f_high],
                                fs = samp_rate)

        if detrend:
            data = signal.detrend(rd, axis = 0, type = 'linear')
            data = data.flatten(order = 'F')
        else:
            data = rd.flatten(order = 'F')

        try:
            df.loc[:, output_col_names[i]] = data
        except IndexError:
            tqdm.write("IndexError: output_col_names index does not match input_col_names!")
            return -1

    return df

def uwb_data_read2pandas(main_path, out_path = "{}/OneDrive - McGill University/Documents McGill/Data/UWB/".format(os.environ['USERPROFILE']),
                            processed_path = "Processed/DF/", settings = None, save_file = True, save_format="parquet", parquet_engine= 'pyarrow',
                            nafill = 0, keep_default_na = True, on_bad_lines = 'warn', out_df = False,
                            **kwargs):
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
            sample_rate = 160e9 # [samples/second]
            f_low = 2e9 # [Hertz]
            f_high = 4e9 # [Hertz]
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
    fillna: None or any, default 0
        set to None to maintain NaNs (if keep_default_na is True), otherwise set to value that replaces NaNs.
    keep_default_na: bool, default True
        Whether or not to include the default NaN values when parsing the data. Depending on whether na_values is passed in, the behavior is as follows:

        If keep_default_na is True, and na_values are specified, na_values is appended to the default NaN values used for parsing.

        If keep_default_na is True, and na_values are not specified, only the default NaN values are used for parsing.

        If keep_default_na is False, and na_values are specified, only the NaN values specified na_values are used for parsing.

        If keep_default_na is False, and na_values are not specified, no strings will be parsed as NaN.

        Note that if na_filter is passed in as False, the keep_default_na and na_values parameters will be ignored.
    on_bad_lines: {'error', 'warn', 'skip'} or callable, default 'warn'
        Specifies what to do upon encountering a bad line (a line with too many fields). Allowed values are:

        'error', raise an Exception when a bad line is encountered.

        'warn', raise a warning when a bad line is encountered and skip that line.

        'skip', skip bad lines without raising or warning when they are encountered.
    out_df : bool, default False
        set to True to output DataFrame even with save_file == True
    **kwargs : dict, optional
        keyword arguments to apply to settings class (see 'settings' input above).

    Returns (optional)
    ----------
    df : Pandas df or str, optional
        function by default saves a file and option
        in case save_file is set to False, then outputs the resulting df
    """
    if settings is None:
        settings = Scan_settings(**kwargs)
    if kwargs:
        settings.update_values(**kwargs)

    path_list, meta_index = uwb_scan_folder_sweep(main_path)

    data_list = []

    for i, p in enumerate(tqdm(path_list)):
        data = uwb_data_read(Path(p), output_numpy = False, col_names = settings.sig_names, nafill=nafill, keep_default_na = keep_default_na, on_bad_lines = on_bad_lines)
        data['samples'] = np.arange(0, np.size(data, axis = 0))
        data["nsamples"] = np.size(data, axis = 0)
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
    df['pair'] = "(" + df["Tx"].astype(str) + "," + df["Rx"].astype(str) + ")"

    df["attRF"] = settings.attRF
    df["HP_amp"] = settings.HP_amp
    df["LNA_amp"] = settings.LNA_amp

    df['samp_rate'] = settings.sampling_rate
    df['time'] = df['samples'] * (1/settings.sampling_rate)
    df['array_config'] = settings.array_config
    df["obs"] = settings.obs

    df['f_low'] = settings.f_low
    df['f_high'] = settings.f_high

    if save_file:
        out_path = "".join((out_path,processed_path))
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        file_title = f"{settings.date} Phantom {settings.phantom} Angle {settings.angle} Plug {settings.plug} Rep {settings.rep} Iter {settings.iter}"

        if save_format.casefold() == "parquet":
            if parquet_engine == 'pyarrow':
                file_path = "".join((out_path, file_title, ".parquet"))
                df.to_parquet(file_path, engine=parquet_engine, index= False)
            else:
                file_path = "".join((out_path, file_title, ".parquet"))
                df.to_parquet(file_path, engine=parquet_engine, object_encoding='utf8', write_index= False)
        else:
            file_path = "".join((out_path, file_title, ".csv"))
            df.to_csv(file_path)
        tqdm.write("".join(("\r Saved DataFrame file for: ", file_title, "          ")), end="")
        if out_df:
            return df
        else:
            return
    else:
        return df