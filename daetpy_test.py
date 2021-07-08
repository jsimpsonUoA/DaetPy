'''
Test script for developing the DaetPy class

April 2020
'''

import numpy as np
import matplotlib.pyplot as plt

from main import DaetPy
import plotting as plot



dp = DaetPy('daet_test/',50,200,pump_sensor='PolytecOFV5000',probe_sensor='PolytecOFV5000X')

reference_min = 3.5e3
reference_max = 6.5e3
reference_use_times = True
bandpass=(1e3,1e6)

reference_waveform = dp.generate_reference_waveform(bandpass=bandpass,reference_min=reference_min,reference_max=reference_max, reference_use_times=reference_use_times)
reference_t0 = dp.pick_reference_t0(reference_waveform)

lags = dp.compute_time_delays(reference_waveform, bandpass=bandpass,tmin=2,tmax=20)

plot.pump_dv_plot(dp, lags / reference_t0 * -100.)

plot.animated_probe_waveforms(dp, reference_waveform, update_interval=0.05,repeat_delay=0., bandpass=bandpass, normalise=True, demean=True, figsize=(11,8), \
                                color='r', ylim=(-1.2,1.2),xlim=(0,20), ylab='Amplitude (a.u.)', legend=True, legend_loc='upper right')

