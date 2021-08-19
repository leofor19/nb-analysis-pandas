# Python 2.7
# 2020-09-04

# Version 4.0.2
# Last updated on 2021-08-11

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Description:
        Module for controlling the narrow band system. The main function is to initiate a frequency sweep, recording and optionally plotting the data collected
        by the DC Receiver.

Class::
        Dc1513bAa(dc890.Demoboard): defined for communication with the D890B demo board.

Functions::

        ant_sweep : performs scans for selected antenna pairs, for all selected input frequencies
                    in order order antenna pair switching -> frequency switching

        ant_sweep_alt : performs scans for selected antenna pairs, for all selected input frequencies
                        in order frequency switching -> antenna pair switching

        cal_system : performs a calibration scan, for which there are 3 types

Inner functions::

        _generate_file_path

        _generate_file_path2

        _generate_cal_file_path

        _save_json_exp

        _save_json_cal

Written by: Leonardo Fortaleza

With code by Anne-Marie Zaccarin on Dc1513bAa(dc890.Demoboard) and its data collection, adapted from Linear Technology.
"""
# Standard library imports
import copy
from datetime import datetime
import itertools as it
import json
import os, sys
import time

# checks proper folder for Linear Lab Tools and adds to path
lltpath = '%UserProfile%/Documents/Analog Devices/linear_lab_tools64/python/'
if not os.path.exists(os.path.dirname(lltpath)):
    lltpath = '%UserProfile%/Documents/linear_technology/linear_lab_tools64/python/'
sys.path.insert(1,lltpath)
sys.path.insert(1, os.path.abspath('%UserProfile%/OneDrive - McGill University/Documents McGill/2019.2/LabTools Scripts/2021_05_25 Adaptations 13/'))

# Third-party imports
import numpy as np
from timeit import default_timer as timer
from tqdm.auto import tqdm, trange

# Local  Linear Technology imports
#import llt.common.constants as consts
#import llt.common.dc890 as dc890
#import llt.common.functions as funcs

# Local application imports
#from NarrowBand.ReceiverFFT import ReceiverFFT as rfft
#from NarrowBand.SwitchingMatrix import switching_matrix as swm
#from NarrowBand.Transmitter_LTC6946 import ltc6946_serial as fsynth


#class Dc1513bAa(dc890.Demoboard):
#    """
#        A DC890 demo board with settings for the DC1513B-AA.
#    """
#
#    def __init__(self, spi_registers, verbose = False):
#        dc890.Demoboard.__init__(self,
#                                dc_number           = 'DC_1513B-AA',
#                                fpga_load           = 'CMOS',
#                                num_channels        = 2,
#                                is_positive_clock   = False,
#                                num_bits            = 14,
#                               alignment           = 14,
#                                is_bipolar          = True,
#                                spi_reg_values      = spi_registers,
#                                verbose             = verbose)

ch0 = np.random.randint(-8191,8192, size=1024)
ch1 = np.random.randint(-8191,8192, size=1024)

def ant_sweep(meas_parameters, window = 'hann', do_plot = False, do_FFT = False, save_json = True, display=False):
    """Execute frequency sweep and data acquisition, recording files for time and frequency domain.

    Performs narrow band system measurements by setting discrete input frequencies with the LTC6946 PLL Frequency Synthesizer
    and acquiring data with the LTM9004 DC Receiver.

    This function uses serial control for the frequency synthesizer and performs in order: switch antenna pair  ->  switch frequency. 

    Parameters
    ----------
    meas_parameters : dict
        dictionary containing several measurement parameters for the experiment (see details after parameters)
    window : str, optional
        FFT window to be used (see fft_window module), by default 'hann'
    do_plot : bool, optional
        set True to plot the data on the terminal, by default False
    do_FFT : bool, optional
        set True to record the FFT, by default False
    save_json : bool, optional
        set True to save JSON dictionary file with experiment configuration, by default True

    For the meas_parameters dictionary:
    ----------------------------------------

    freq_range: list or tuple of strings
        list or tuple of strings containing the input frequency range in MHz, with underscores "_" replacing dots "."

    data_file: str
        string with generic file name and path for the time domain data file to be written,
        using placeholders for several details such as "FREQ" for the current frequency

    date: str
        string in the format "yyyy_mm_dd" (placeholder in file names is "DATE")

    Phantom: int
        phantom number (placeholder in file names is "PHA")

    Angle: int
        phantom rotation angle (placeholder in file names is "ANG")

    Plug: int
        plug number (placeholder in file names is "PLU")

    rep: int
        user determined number counting the repetitions of measurement after repositioning,
        requires calling the function again for each repetition (placeholder in file names is "REP")

    iter: int
        number of iteations to be performed of full frequency sweeps (for checking repeatability of measurements)
        (placeholder in file names is "ITE")

    window: str
        FFT window to be used, default is 'hann' (see fft_window module)
    """

    start = timer()

    if save_json:
        meas_parameters["start"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for key in ["cal_data_file", "cal_fft_file"]:
            del meas_parameters[key]

        meas_parameters["type"] =  "measurement configuration parameters"

    ite = meas_parameters["iter"]

    pairs = meas_parameters["pairs"]

    _generate_file_path(meas_parameters = meas_parameters)

    num_samples = meas_parameters["num_samples"]
    spi_registers = meas_parameters["spi_registers"]
    verbose = meas_parameters["verbose"]

    freq_range = meas_parameters["freq_range"]

    window = meas_parameters["fft_window"]

    #fctrl = fsynth.DC590B()

    #with Dc1513bAa(spi_registers, verbose) as controller:
    pbar = tqdm(range(1,ite+1), leave= True)

    for j in pbar:
        pbar.set_description("Iteration: %i" % j)
        ite_start = timer()
        for (TX, RX) in tqdm(pairs, leave= False):
            #print "Switching to Pair: Tx - ", TX, ", Rx - ", RX,  "\n"
            #swm.set_pair(TX, RX)
            pbar2 = tqdm( range(0,len(freq_range)) , leave= False)
            for i in pbar2:
                f_cur = freq_range[i]
                #tqdm.write("Switching to Pair: Tx - ", TX, ", Rx - ", RX, " @ ", f_cur, " MHz        ",)
                pbar2.set_description("Tx - %2i Rx -% 2i @ %s MHz" % (TX, RX, f_cur))
                #print "\rSwitching to input frequency: {} MHz \n".format(f_cur)
                #if display:
                #    print "\rSwitching to Pair: Tx - ", TX, ", Rx - ", RX, " @ ", f_cur, " MHz        ",
                #fctrl.freq_set(freq = f_cur, verbose=verbose)
                #time.sleep(0.1)
                data_file= _generate_file_path2(meas_parameters = meas_parameters, antenna_pair = "Tx {0:d} Rx {1:d}".format(TX,RX))
                if not os.path.exists(os.path.dirname(data_file.replace("ITE",str(j)))):
                    os.makedirs(os.path.dirname(data_file.replace("ITE",str(j))))
                #ch0,ch1 = controller.collect(num_samples, consts.TRIGGER_NONE)
                if do_plot:
                    #print "\rPlotting for input frequency: {} MHz".format(f_cur),
                    #rfft.plot_channels(controller.get_num_bits(), window,
                    #                    ch0, ch1,
                    #                    verbose=verbose)
                    pass
                save_for_pscope(data_file.replace("FREQ",f_cur).replace("ITE",str(j)), 14, True, num_samples,
                                        'DC_1513B-AA', 'LTM9004', ch0, ch1,)
                pass
                if do_FFT:
                    #rfft.save_for_pscope_fft(data_file.replace(".adc",".fft").replace("FREQ",f_cur).replace("ITE",str(j)), controller.num_bits, controller.is_bipolar,num_samples,
                    #                    'DC_1513B-AA', 'LTM9004', window, ch0, ch1)
                    pass

        if save_json and j != ite:
            ite_end = timer()
            meas_parameters["iter_duration"] = ite_end - ite_start
            _save_json_exp(meas_parameters = meas_parameters, iteration = j)

    #fctrl.freq_set(freq = "0", verbose=verbose)
    end = timer()
    meas_parameters["meas_duration"] = str(end - start)
    #print "\rArray Measurement Time: ", meas_parameters["meas_duration"], "seconds        \n"

    if save_json:
        meas_parameters["end"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _save_json_exp(meas_parameters = meas_parameters)

def _generate_file_path(meas_parameters):
    """Alter the dictionary value for the key "data_file" with current values for date, phantom, angle, plug and rep.

    Parameters
    ----------
    meas_parameters : dict
        dictionary with "measurement configuration parameters"
    """
    #fft_file = meas_parameters["fft_file"]

    date = meas_parameters["date"]
    phantom = meas_parameters["Phantom"]
    angle = meas_parameters["Angle"]
    plug = meas_parameters["Plug"]
    repetition = meas_parameters["rep"]


    meas_parameters["data_file"] = meas_parameters["data_file"].replace("DATE", date).replace("PHA",str(phantom)).replace("ANG",str(angle)).replace("PLU",str(plug)).replace("REP",str(repetition))

def _generate_file_path2(meas_parameters, antenna_pair = "Tx 15 Rx 16", file_path_key = "data_file"):
    """Output the data_file full path with the current antenna pair values.

    Parameters
    ----------
    meas_parameters : dict
        dictionary with "measurement configuration parameters"
    antenna_pair : str, optional
        string in the format: "Tx # Rx #", by default "Tx 15 Rx 16"
    file_path_key: str, optional
        string with file path, by default "data_file"

    Returns
    ----------
    str
        data_file path with placeholders "ANTPAIR" replaced with current values
    """

    data_file = copy.copy(meas_parameters[file_path_key])


    data_file = data_file.replace("ANTPAIR",str(antenna_pair))

    return data_file

def _save_json_exp(meas_parameters, config_folder = "Config/", iteration = None):
    """Save "measurement configuration parameters" dictionary to JSON file.

    Parameters
    ----------
    meas_parameters : dict
        dictionary with "measurement configuration parameters"
    config_folder: str, optional
        sub-folder to place the JSON configuration file, by default "Config/"
    iteration: int or None, optional
        current iteration value, by default None
        when None, receives meas_parameter["iter"]
    """

    if iteration is None:
        iteration = meas_parameters["iter"]
        original_it = None
    else:
        original_it = meas_parameters["iter"]
        meas_parameters["iter"] = iteration

    out_path = meas_parameters["data_file"].partition("Phantom ")[0] + config_folder
    file_name = os.path.basename(meas_parameters["data_file"]).replace("ANTPAIR FREQMHz","").replace("ITE", str(iteration)).replace(".adc",".json")

    #file_name = meas_parameters["file_name"].replace(".adc",".json")

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    with open(out_path + file_name, 'w') as fp:
        json.dump(meas_parameters, fp, sort_keys=True, indent=4)
    print "\r Saved JSON file for: ", file_name
    if original_it is not None:
        meas_parameters["iter"] = original_it

def save_for_pscope(out_path = 'data.adc', num_bits = 14, is_bipolar = True, num_samples = 1*1024, dc_num = 'DC_1513B-AA',
						 ltc_num = 'LTM9004', *data):
	"""Save data in PScope .adc format (csv type file).

	As defined by Linear Technology.

	Keyword arguments (all optional except for data)::
		out-path -- file path in string format
		num_bits -- integer number of bits on the receiver ADC (default is 14, which is the case for the LTM9004)
		is_bipolar -- boolean stating if the ADC operates with both negative and positive values (True) or just positive values (False)
				Default is True, which is the case for the LTM9004
		num_samples -- integer number of samples setting for the ADC on the receiver 
					(multiples of 1024 in powers of 2, up to 64*1024=65,536)
					Default is 1024
		dc_num -- string with the name/number of the demonstration circuit (default is 'DC_1513B-AA', used with the LTM9004)
		ltc_num -- string with the name/number of the device (default is 'LTM9004')
		*data -- array with collected data in the time domain, each column representing a channel
	"""
	num_channels = len(data)
	if num_channels < 0 or num_channels > 16:
		raise ValueError("pass in a list for each channel (between 1 and 16)")

	full_scale = 1 << num_bits
	if is_bipolar:
		min_val = -full_scale / 2
		max_val = full_scale / 2
	else:
		min_val = 0
		max_val = full_scale

	sample_rate = 125.0
	with open(out_path, 'w') as out_file:
		out_file.write('Version,115\n')
		out_file.write('Retainers,0,{0:d},{1:d},1024,0,{2:0.15f},1,1\n'.format(num_channels, num_samples, sample_rate))
		out_file.write('Placement,44,0,1,-1,-1,-1,-1,10,10,1031,734\n')
		out_file.write('DemoID,' + dc_num + ',' + ltc_num + ',0\n')
		for i in range(num_channels):
			out_file.write(
				'RawData,{0:d},{1:d},{2:d},{3:d},{4:d},{5:0.15f},{3:e},{4:e}\n'.format(
					i+1, num_samples, num_bits, min_val, max_val, sample_rate ))
		for samp in range(num_samples):
			out_file.write(str(data[0][samp]))
			for ch in range(1, num_channels):
				out_file.write(', ,' + str(data[ch][samp]))
			out_file.write('\n')
		out_file.write('End\n')

def my_write_func():
    return 1

if __name__ == '__main__':

    now = datetime.now()

    #PAIRS = [(TX, RX) for RX in range(1,17) for TX in range(1,17) if TX != RX]
    #PAIRS = [(TX, RX) for RX in range(1,17) for TX in range(1,17) if TX != RX and TX != 13 and RX != 13]
    #PAIRS = [(1,2), (2,3), (3,5), (4,5), (5,6), (6,7), (7,8),
    #            (1,8), (1,9), (1,16), (2,10), (2,11), (3,11), (3,12), (4,9), (4,10), (4,11), (4,12), (4,14), (5,12),
    #            (6,14), (6,15), (7,15), (7,16), (8,15), (8,16), (9,10), (9,14), (9,15), (9,16), (10,11), (10,12), (10,14),
    #            (11,12), (14,15), (14,16), (15,16)
    #            ]

    #pairs = [(1,2), (2,3), (3,5), (5,6), (7,8), (9,10), (10,11), (11,12), (14,15), (15,16),
	#		(1,6), (2,5), (4,10), (4,12), (4,14), (7,8), (9,14), (9,16)
    #        ]
    #pairs_rev = [tuple(reversed(t)) for t in pairs]
    #extra_pairs = [(3, 13), (5,13)]

    #pairs.extend(pairs_rev)
    #pairs.extend(extra_pairs)

    # 225 pairs - excludes Tx = 13

    Tx = range(1,13) + range(14,17)
    Rx = range(1,17)
    pairs = [(x, y) for x, y in it.product(Tx,Rx) if x != y]

    # MeasParameters should be reset every loop of rep (repetition), except for new iterations!

    MeasParameters ={
                    "num_samples" : 1024,
                    "spi_registers" : [],
                    "verbose" : False, # DC590B controller board verbosity

                    "samp_rate" : 125*1e6,
                    "fft_window" : "hann",

                    #"data_file" : "%UserProfile%/Documents/Documents McGill/Data/PScope/DATE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Phantom PHA Plug PLU ANG deg Rep REP Iter ITE.adc",
                    #"fft_file" : "%UserProfile%/Documents/Documents McGill/Data/PScope/DATE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Phantom PHA Plug PLU ANG deg Rep REP Iter ITE.fft",

                    #"cal_data_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Calibration/Type TYPE/Rep REP/Iter ITE/Calibration Iter ITE.adc".format(os.environ['USERPROFILE']),
                    #"cal_fft_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Calibration/Type TYPE/Rep REP/Iter ITE/Calibration Iter ITE.fft".format(os.environ['USERPROFILE']),

                    "data_file" : "{}//OneDrive - McGill University/Documents McGill/2021.3/Python Test/DATE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Phantom PHA Plug PLU ANG deg Rep REP Iter ITE.adc".format(os.environ['USERPROFILE']),
                    "fft_file" : "{}//OneDrive - McGill University/Documents McGill/2021.3/Python Test/DATE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Phantom PHA Plug PLU ANG deg Rep REP Iter ITE.fft".format(os.environ['USERPROFILE']),

                    "cal_data_file" : "{}/OneDrive - McGill University/Documents McGill/2021.3/Python Test/DATE/Calibration/Type TYPE/Rep REP/Iter ITE/Calibration Iter ITE.adc",
                    "cal_fft_file" : "{}/OneDrive - McGill University/Documents McGill/2021.3/Python Test/DATE/Calibration/Type TYPE/Rep REP/Iter ITE/Calibration Iter ITE.fft",

                    "folder_path" : None,
                    "file_name" : None,
                    "fft_file_name" : None,


                    "date" : now.strftime("%Y_%m_%d"),

                    "Phantom" : 1,
                    "Angle" : 0,
                    "Plug" : 2,

                    "rep" : 4,
                    "iter" : 1,

                    "freq_range" : ("2000",
                                    "2012_5",
                                    "2025",
                                    "2037_5",
                                    "2050",
                                    "2062_5",
                                    "2075",
                                    "2087_5",
                                    "2100",
                                    "2112_5",
                                    "2125",
                                    "2137_5",
                                    "2150",
                                    "2162_5",
                                    "2175",
                                    "2187_5",
                                    "2200"),

                    "pairs" : pairs,

                    "attLO" : 20,
                    "attRF" : 9,

                    "obs" : "",

                    "system" : "narrow band",
                    "type" : "measurement configuration parameters"
                    }

    #cal_system(meas_parameters = MeasParameters, cal_type  = 1, do_plot = False, do_FFT = False, save_json = True)

    ant_sweep(meas_parameters = MeasParameters, do_plot = False, do_FFT = False, save_json = True, display = False)

    #ch0 = np.random.randint(-8191,8192, size=1024)
    #ch1 = np.random.randint(-8191,8192, size=1024)

    #write_path =  "{}//OneDrive - McGill University/Documents McGill/2021.3/Python Test/Write Test File.adc".format(os.environ['USERPROFILE'])
    #start = time.time()
    #save_for_pscope(write_path, 14, True, 1*1024, 'DC_1513B-AA', 'LTM9004', ch1, ch2)
    #end = time.time()
    #print "Write Time: {}".format(end-start)