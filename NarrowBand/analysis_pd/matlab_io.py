# Standard library imports
from datetime import datetime
import os
import os.path

# Third-party library imports
import numpy as np
from pymatreader import read_mat
from scipy.io import savemat
# from tqdm import tqdm # when using terminal
from tqdm.notebook import tqdm # when using Jupyter Notebook
#from tqdm.dask import TqdmCallback

# Local application imports


def convert_to_DMAS_format(df, signal_col = 'signal'):
    """Convert Pandas DataFrame of Time-Domain data to numpy array in DMAS algorithm compatible format.

    3-D array has shape (Tx_number , Rx_number , number_of_samples).

    Parameters
    ----------
    df : Pandas df
        input DataFrame with Time-Domain data
    signal_col : str, optional
        column name with signal of interest, by default 'signal'

    Returns
    -------
    matrix_out: np.array
        3-D numpy array compatible with DMAS algorithm
    """
    # forces first unique phantom scan
    df = df.loc[df.phantom.eq(df.phantom.unique()[0]) & df.angle.eq(df.angle.unique()[0]) & df.date.eq(df.date.unique()[0]) & 
                df.rep.eq(df.rep.unique()[0]) & df.iter.eq(df.iter.unique()[0])]

    # identifies number of samples
    N = df.loc[df.phantom.eq(df.phantom.unique()[0]) & df.angle.eq(df.angle.unique()[0]) & df.date.eq(df.date.unique()[0]) & 
                df.rep.eq(df.rep.unique()[0]) & df.iter.eq(df.iter.unique()[0]) & df.pair.eq(df.pair.unique()[0]), signal_col].size

    matrix_out = np.zeros((16,16,N), dtype=float)

    for Tx in np.arange(1,17):
        for Rx in np.arange(1,17):
            if f"({Tx},{Rx})" in df.pair.unique():
                matrix_out[Tx-1,Rx-1,:] = df.loc[df.Tx.eq(Tx) & df.Rx.eq(Rx), signal_col].values
    return matrix_out

def generate_Mat_file_name(df):
    """Generate .mat file name from DataFrame info.

    Parameters
    ----------
    df : Pandas df
        input dataframe

    Returns
    -------
    filename: str
        .mat file name
    """

    file_name = f"Phantom_{df.phantom.unique()[0]}_Ang_{df.angle.unique()[0]}_Date_{df.date.unique()[0]}_Plug_{df.plug.unique()[0]}_Rep_{df.rep.unique()[0]}_Iter_{df.iter.unique()[0]}.mat"

    return file_name

def export_to_DMAS_Matlab(df, main_path="C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Mat/".format(datetime.now().strftime("%Y_%m_%d")), file_name=None,
                            signal_col = 'signal'):
    """Save .mat file from phantom scan DataFrame.

    Please select specific phantom scan (phantom, angle, date, rep, iter, plug) before exporting.

    Parameters
    ----------
    df : Pandas df
        input pantom scan dataframe
    main_path : str, optional
        file path to save .mat file, by default "C:/Users/leofo/OneDrive - McGill University/Narrow Band Data1/Analysis/{}/Mat/".format(datetime.now().strftime("%Y_%m_%d"))
    file_name : str, optional
        .mat file name, by default None
        if None, automatically generates file name from dataframe info.
    signal_col : str, optional
        column name with signal of interest, by default 'signal'
    """
    if file_name is None:
        file_name = generate_Mat_file_name(df)
    file_path = "".join((main_path, file_name))

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    matrix = convert_to_DMAS_format(df, signal_col = signal_col)

    savemat(file_path, {'Scan': matrix})
    tqdm.write(f"Matlab file written: {file_path}")
    return

def import_from_Matlab(filename):
    data = read_mat(filename) # data is a dict

    # for key, value in data.items():
    #     unpacked = [key, ]
    #     # see https://stackoverflow.com/questions/40581247/how-to-unpack-a-dictionary-of-list-of-dictionaries-and-return-as-grouped-tupl

    return data