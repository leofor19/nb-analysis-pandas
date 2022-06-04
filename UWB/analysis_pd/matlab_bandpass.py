""" Description:
        Function that emulates MATLAB's bandpass function.

Adapted from https://dsp.stackexchange.com/questions/68921/python-equivalent-code-for-matlab-bandpass-function

"""

# from scipy.signal import sosfiltfilt, filtfilt, firwin, ellip, ellipord, butter, buttord, remez, kaiser_atten, kaiser_beta, kaiserord
from scipy import signal

def matlab_bandpass(x, fpass, fs):
    """Emulates MATLAB bandpass filter.

    Parameters
    ----------
    x : ndarray-like
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
    gpass = 0.1
    steepness = 0.85 # MATLAB default
    nyq = 0.5*fs

    wp = [lowcut, highcut]
    ws = [steepness*lowcut, nyq*(1-steepness) + steepness*highcut]

    ## Kaiser filter
    # width = (1 - steepness)*nyq # This sets the cutoff width in Hertz
    # ntaps, gb = kaiserord(stopbbanAtt, width/nyq)
    # atten = kaiser_atten(ntaps, width/nyq)
    # beta = kaiser_beta(atten)
    # a = 1.0
    # # taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero = False, window=('kaiser', beta), scale=False)
    # taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero = False, scale=False)
    # filtered_signal = filtfilt(taps, a, x)


    ## Elliptical (Cauer) filter
    # ord, wn = signal.ellipord(wp = wp, ws = ws, gpass = gpass, gstop = stopbbanAtt, fs = fs)
    # sos = signal.ellip(ord, gpass, stopbbanAtt, wn, btype = 'bandpass', output = 'sos', fs = fs)
    ## ord, wn = buttord([lowcut, highcut], [steepness*lowcut, nyq*(1-steepness) + steepness*highcut], gpass, stopbbanAtt, fs)
    ## sos = butter(ord, wn, btype = 'bandpass', output = 'sos', fs = fs)

    ## IIRfilter
    sos = signal.iirdesign(wp = wp, ws = ws, gpass = gpass, gstop = stopbbanAtt, analog=False, ftype='ellip', output='sos', fs=fs)

    filtered_signal = signal.sosfiltfilt(sos, x, axis = 0, padtype = 'odd')

    return filtered_signal