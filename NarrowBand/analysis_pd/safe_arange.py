import numpy as np

def safe_arange(start, stop, step, dtype = None):
    """Safer implementation of numpy.arange for floats.

    Applies start and stop normalized by step to avoid floating point issues.

    However, not guaranteed to solve all issues with numpy.arange.

    Parameters
    -----------
    start: integer or real
        Start of interval. The interval includes this value. The default start value is 0.
        Not optional in this implementation.

    stop: integer or real
        End of interval. The interval does not include this value, except in some cases where step is not an integer and floating point round-off affects the length of out.

    step: integer or real, optional
        Spacing between values. For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. The default step size is 1. If step is specified as a position argument, start must also be given.
        Not optional in this implementation.

    dtype: dtype
        The type of the output array. If dtype is not given, infer the data type from the other input arguments.

    Returns
    -----------
    arange: ndarray
        Array of evenly spaced values.

    For floating point arguments, the length of the result is ceil((stop - start)/step). Because of floating point overflow, this rule may result in the last element of out being greater than stop.

    See https://stackoverflow.com/questions/47243190/numpy-arange-how-to-make-precise-array-of-floats
    """
    return step * np.arange(start / step, stop / step, dtype = dtype)