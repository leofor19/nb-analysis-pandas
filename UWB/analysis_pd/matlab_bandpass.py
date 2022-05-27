""" Description:
        Function that emulates MATLAB's bandpass function.

Adapted from https://dsp.stackexchange.com/questions/68921/python-equivalent-code-for-matlab-bandpass-function

"""

from scipy.signal import filtfilt, firwin, remez, kaiser_atten, kaiser_beta, kaiserord

def matlab_bandpass(signal, fpass, fs):
    """Emulates MATLAB bandpass filter.

    Parameters
    ----------
    signal : ndarray-like
        input signal
    fpass : tuple
        passband frequencies (lowcut, highcut), in Hertz
    fs : float
        sampling rate

    Returns
    -------
    filtered_signal
        signal filtered by bandpass filter
    """

    lowcut = fpass[0] #In Hertz
    highcut = fpass[1] #In Hertz
    stopbbanAtt = 60  #stopband attenuation of 60 dB.
    steepness = 0.85 # MATLAB default
    nyq = 0.5*fs
    width = (1 - steepness)*nyq # This sets the cutoff width in Hertz
    ntaps, gb = kaiserord(stopbbanAtt, width/nyq)
    atten = kaiser_atten(ntaps, width/nyq)
    beta = kaiser_beta(atten)
    a = 1.0
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero = False, window=('kaiser', beta), scale=False)
    filtered_signal = filtfilt(taps, a, signal)

    return filtered_signal