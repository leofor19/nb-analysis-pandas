# Python 2.7
# 2021-05-19

# Version 4.0.3
# Last updated on 2021-08-15

# Leonardo Fortaleza (leonardo.fortaleza@mail.mcgill.ca)

"""
Written by: Leonardo Fortaleza

 Description:
		 Script for performing measurements with the narrow band system using the Python modules.
"""
# Standard library imports
from datetime import datetime
import itertools as it
import os, sys

# in case the modules need to be inserted in system path (script folder outside rest of module):
# may replace 'os.path.dirname(os.path.abspath(__file__))' with 'os.path.abspath('%UserProfile%/Documents/nb-control/')' or another path as required
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

# Local application imports
import NarrowBand.system as nbsys

now = datetime.now()

freq_cal = ("2000",
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
            "2200")

# 225 pairs - excludes Tx = 13

Tx = range(1,13) + range(14,17)
Rx = range(1,17)
pairs = [(x, y) for x, y in it.product(Tx,Rx) if x != y]

# 36 pairs (?)

#pairs = [(1,2), (2,3), (3,5), (5,6), (7,8), (9,10), (10,11), (11,12), (14,15), (15,16),
#			(1,6), (2,5), (4,10), (4,12), (4,14), (7,8), (9,14), (9,16)
#            ]
#pairs_rev = [tuple(reversed(t)) for t in pairs]
#extra_pairs = [(3, 13), (5,13)]

#pairs.extend(pairs_rev)
#pairs.extend(extra_pairs)

# 28 pairs

#pairs = [(1,6), (1,7), (1,8), (1,9), (1,14), (1,15), (1,16), (6,7), (6,8), (6,9), (6,14), (6,15), (6,16), (7,8), (7,9), (7,14), (7,15), (7,16),
#         (8,9), (8,14), (8,15), (8,16), (9,14), (9,15), (9,16), (14,15), (14,16), (15,16)
#            ]
#pairs_rev = [tuple(reversed(t)) for t in pairs]
#pairs.extend(pairs_rev)


MeasParameters ={
                    "num_samples" : 1024,
                    "spi_registers" : [],
                    "verbose" : False, # DC590B controller board verbosity

                    "samp_rate" : 125*1e6,
                    "fft_window" : "hann",

                    "data_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Phantom PHA Plug PLU ANG deg ANTPAIR FREQMHz Rep REP Iter ITE.adc".format(os.environ['USERPROFILE']),
                    "fft_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Phantom PHA Plug PLU ANG deg ANTPAIR FREQMHz Rep REP Iter ITE.fft".format(os.environ['USERPROFILE']),

                    "cal_data_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Calibration/Type TYPE/Rep REP/Iter ITE/Calibration Type TYPE Rep REP Iter ITE.adc".format(os.environ['USERPROFILE']),
                    "cal_fft_file" : "{}/Documentsy/Documents McGill/Data/PScope/DATE/Calibration/Type TYPE/Rep REP/Iter ITE/Calibration Type TYPE Rep REP Iter ITE.fft".format(os.environ['USERPROFILE']),

                    "cal_ph_data_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Calibration/Type TYPE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Calibration Type TYPE Phantom PHA Plug PLU ANG deg FREQMHz ANTPAIR Rep REP Iter ITE.adc".format(os.environ['USERPROFILE']),
                    "cal_ph_fft_file" : "{}/Documents/Documents McGill/Data/PScope/DATE/Calibration/Type TYPE/Phantom PHA/ANG deg/Plug PLU/Rep REP/Iter ITE/Calibration Type TYPE Phantom PHA Plug PLU ANG deg FREQMHz ANTPAIR Rep REP Iter ITE.fft".format(os.environ['USERPROFILE']),

                    "date" : now.strftime("%Y_%m_%d"),

                    "Phantom" : 1,
                    "Angle" : 0,
                    "Plug" : 2,

                    "rep" : 1,
                    "iter" : 1,

                    "freq_range" : ("2012_5",
                                    "2025",
                                    "2037_5",
                                    "2050",
                                    "2062_5",
                                    "2075",
                                    "2087_5",
                                    "2100"),

                    "pairs" : pairs,

                    "attLO" : 20,
                    "attRF" : 9,

                    "obs" : "",

                    "system" : "narrow band",
                    "type" : "measurement configuration parameters",
                    }

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Measurement routine starts here!

# Edit the pertinent parts here!

# 1 - Phantom details (Phantom, Angle, Plug, Repetition), how many sequential Iterations, observations.
# 2 - Calibration sequence (types 1, 2, 3, 4)
# 3 - Phantom scan sequence
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Phantom details - Reinforcing values
# IMPORTANT: When using Phantom 1, RF needs 6 dB attenuator, LO can use 20 dB attenuator (could be a bit less - remembering to use amplified Tx for LO)
MeasParameters["Phantom"] = 0
MeasParameters["Angle"] = 0
MeasParameters["Plug"] = 0

MeasParameters["rep"] = 1
MeasParameters["iter"] = 5

#MeasParameters["obs"] = "Baseline-only Initialization."
MeasParameters["obs"] = "Empty hemisphere (air)."
#MeasParameters["obs"] = "Baseline-Tumour Progress with phantom and plug re-position."

AntPair = "Tx 15 Rx 16"


""" First calibration round:

	Both LO and RF grounded with 50 ohm terminators.
	"""

MeasParameters["attLO"] = "grounded"
MeasParameters["attRF"] = "grounded"

nbsys.cal_system(meas_parameters = MeasParameters, cal_type  = 1, do_plot = False, do_FFT = False, save_json = True)

""" Second calibration round:

	RF grounded with 50 ohm terminator, LO connected to frequency synthesizer.
	LO can use 20 dB attenuator.
    Remember to use 50-ohm terminator on splitter Tx.
	"""

#MeasParameters["attLO"] = 20
#MeasParameters["attRF"] = "grounded"
#MeasParameters["freq_range"] = freq_cal


#nbsys.cal_system(meas_parameters = MeasParameters, cal_type  = 2, do_plot = False, do_FFT = False, save_json = True)

""" Third calibration round:

	RF connected to Rx-Tx directly by cables (bypassing antennas) and LO connected to frequency syntesizer.
	Use RF with 25 dB attenuator, LO can use 20 dB attenuator.
	"""

#MeasParameters["attLO"] = 20
#MeasParameters["attRF"] = 22
#MeasParameters["freq_range"] = freq_cal

#nbsys.cal_system(meas_parameters = MeasParameters, cal_type  = 3, do_plot = False, do_FFT = False, save_json = True)

""" Fourth calibration round:

    Environmental noise scan. Tx after splitter with 50-ohm terminator, switching matrix Tx also terminated.

	RF connected to Rx antennas and LO connected to frequency syntesizer.
	Phantom 1: Use RF with 9 dB attenuator, LO can use 20 dB attenuator.

	Phantoms with skin: Use RF without attenuator, LO can use 20 dB attenuator
	"""

#MeasParameters["attLO"] = 20
#MeasParameters["attRF"] = 9 # skinless phantoms (1)
#MeasParameters["attRF"] = 0 # phantoms with skin

#nbsys.cal_system(meas_parameters = MeasParameters, cal_type  = 4, do_plot = False, do_FFT = False, save_json = True)

""" Actual measurements:

	RF connected to Rx antennas and LO connected to frequency syntesizer.
	Phantom 1: Use RF with 9 dB attenuator, LO can use 20 dB attenuator.

	Phantoms with skin: Use RF without attenuator, LO can use 20 dB attenuator.
	"""

#MeasParameters["attLO"] = 20
#MeasParameters["attRF"] = 9 # skinless phantoms (1)
#MeasParameters["attRF"] = 0 # phantoms with skin

#nbsys.ant_sweep(meas_parameters = MeasParameters, do_plot = False, do_FFT = False, save_json = True, display = False)