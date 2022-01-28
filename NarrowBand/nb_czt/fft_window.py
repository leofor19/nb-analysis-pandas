"""
	Copyright (c) 2015, Linear Technology Corp.(LTC)
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, 
	   this list of conditions and the following disclaimer.
	2. Redistributions in binary form must reproduce the above copyright 
	   notice, this list of conditions and the following disclaimer in the 
	   documentation and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
	LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
	CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
	SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
	INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
	ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
	POSSIBILITY OF SUCH DAMAGE.

	The views and conclusions contained in the software and documentation are 
	those of the authors and should not be interpreted as representing official
	policies, either expressed or implied, of Linear Technology Corp.

	Ported by Leonardo Fortaleza from Linear Technology provided Matlab script fft_window.m.

	Description:
		 Generates a window for reducing sidelobes due to FFT of incoherently sampled data.

		Window type is a case INsensitive string and can be one of:
			'Hamming', 'Hann', 'Blackman', 'BlackmanExact', 'BlackmanHarris70',
			'FlatTop', 'BlackmanHarris92'
			The default window type is 'Hann'
"""
import ctypes
import math
from math import pi as pi


import numpy as np


def fft_window(n, window_type = 'hann'):
	"""Generate a window for reducing sidelobes due to FFT of incoherently sampled data.

		Window type is a case INsensitive string and can be one of:
			'Hamming', 'Hann', 'Blackman', 'BlackmanExact', 'BlackmanHarris70',
			'FlatTop', 'BlackmanHarris92'
			The default window type is 'Hann'
	"""
	if window_type.lower() == "hann".lower():
		win = one_cos(n, 0.50, 0.50, 1.632993)

	elif window_type.lower() == 'hamming'.lower():
		win = one_cos(n, 0.54, 0.46, 1.586303)

	elif window_type.lower() == 'blackman'.lower():
		win = two_cos(n, 0.42, 0.50, 0.08, 1.811903)

	elif window_type.lower() == 'blackmanexact'.lower():
		win = two_cos(n, 0.42659071, 0.49656062, 0.07684867, 1.801235)

	elif window_type.lower() == 'blackmanharris70'.lower():
		win = two_cos(n, 0.42323, 0.49755, 0.07922, 1.807637) # from: www.mathworks.com/access/helpdesk/help/toolbox/signal/window.shtml

	elif window_type.lower() == 'flattop'.lower():
		win = two_cos(n, 0.2810639, 0.5208972, 0.1980399, 2.066037)

	elif window_type.lower() == 'blackmanharris92'.lower():
		win = three_cos(n, 0.35875, 0.48829, 0.14128, 0.01168, 1.968888) # from: www.mathworks.com/access/helpdesk/help/toolbox/signal/window.shtml

	else:
		MessageBox = ctypes.windll.user32.MessageBoxW
		MessageBox(None, u'Unexpected window type {}'.format(window_type), u"Error", 0)
		pass

	return win


def one_cos(n, a0, a1, norm):
	n_1 = n - 1
	"""t = np.true_divide(np.arange(n_1+1, dtype='f'), n_1)"""
	t = np.arange(n_1+1, dtype='f')/n_1
	win1 = a0 - a1*np.cos(2*pi*t)
	win = win1*norm
	return win

def two_cos(n, a0, a1, a2, norm):
	n_1 = n - 1
	"""t = np.true_divide(np.arange(n_1+1, dtype='f'), n_1)"""
	t = np.arange(n_1+1, dtype='f')/n_1
	win1 = a0 - a1*np.cos(2*pi*t) + a2*np.cos(4*pi * t)
	win = win1*norm
	return win

def three_cos(n, a0, a1, a2, a3, norm):
	n_1 = n - 1
	"""t = np.true_divide(np.arange(n_1+1, dtype='f'), n_1)"""
	t = np.arange(n_1+1, dtype='f')/n_1
	win1 = a0 - a1*np.cos(2*pi * t) + a2*np.cos(4*pi * t) - a3*np.cos(6*pi * t)
	win = win1*norm
	return win

if __name__ == '__main__':
	#window = 'NONE'
	#window = ""
	window = 'Hann'
	#window = 'Hamming'
	#window = 'Blackman'
	#window = 'BlackmanExact'
	#window = 'BlackmanHarris70'
	#window = 'FlatTop'
	#window = 'BlackmanHarris92'

	n = 10
	win = fft_window(n,window)
	print("Window =", win)