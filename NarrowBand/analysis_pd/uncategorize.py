"""

    Uncategorize function to remove category dtype on Pandas dataframes and restore previous dtype.

    Usage:

        >> df = df.apply(lambda x: uncategorize(x), axis=0)


    Obtained from user toto_tico on StackOverflow:

    https://stackoverflow.com/questions/62834653/how-to-remove-all-categorical-columns-from-a-pandas-dataframe
"""
import pandas as pd


def uncategorize(col):
    """Remove category dtype from Pandas dataframe columns, restore previous dtype.

    Usage:

        >> df = df.apply(lambda x: uncategorize(x), axis=0)

    Parameters
    ----------
    col : Pandas Dataframe column
        Pandas Dataframe column

    Returns
    ----------
    col : Pandas Dataframe column
        Pandas Dataframe column without category dtype
    """
    if col.dtype.name == 'category':
        try:
            return col.astype(col.cat.categories.dtype)
            #return pd.to_numeric(col)
        except:
            # In case there is pd.NA (pandas >= 1.0), Int64 should be used instead of int64
            return col.astype(col.cat.categories.dtype.name.title())
            #return col.astype(str)
    else:
        return col