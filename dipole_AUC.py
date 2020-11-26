#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:21:50 2020
@author: mdclarke

Find peak within a window for dipole moment, then compute area under the curve (AUC)
"""
import mne
from os import path as op
import numpy as np
import matplotlib.pyplot as plt

### define time window of interest
window_min = 0.07
window_max = 0.14

path = '/storage/Maggie/'
fname = op.join(path, 'soma2_387_hand_100window_xfit_waves.fif')

dip = mne.read_dipole(fname)
data = dip.data[0]
times = dip.times
assert len(data) == len(times)
time_mask = (times>window_min) & (times<=window_max)
data_mask = data[time_mask]

# Find index of maximum value idx
peak_idx = np.where(data == np.amax(data_mask))

# time around peak for AUC calculation
tmin = times[peak_idx] - 0.015 
tmax = times[peak_idx] + 0.015

auc = np.sum(np.abs(data)) * len(data) * (1. / dip.info['sfreq'])
print('AUC value: %s' % auc)

# plot channel with peak and AUC window
plt.figure()
plt.plot(times, data)
plt.axvline(times[peak_idx], linestyle='-', color='r')
plt.axvline(window_min, linestyle='dotted', color='silver')
plt.axvline(window_max, linestyle='dotted', color='silver')
plt.axvspan(tmin, tmax, alpha=0.15, color='r')
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.title('dipole peak within window')
plt.show()
