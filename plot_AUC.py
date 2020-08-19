#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:21:50 2020

@author: mdclarke

"""
import mne
from os import path as op
import numpy as np
from mne.utils import _time_mask
import matplotlib.pyplot as plt

### change this for lip condition
hemi = 'left' # left or right
###

left = [] # left hemi sensors for lip condition
right = [] # right hemi sensors for lip condition
full = [] # hand/foot sensors

path = '/storage/Maggie/'
fname = op.join(path, 'Locations_40-sss_eq_soma3_408-ave.fif')

evoked = mne.read_evokeds(fname)[0]

if evoked.comment == 'lip' and hemi == 'left':
  evoked.pick_channels(ch_names=left)
if evoked.comment == 'lip' and hemi == 'right':
  evoked.pick_channels(ch_names=right)
if evoked.comment == 'foot' or 'hand':
  evoked.pick_channels(ch_names=full)
else:
  print("No condition named in file")

evoked.data

# find channel with highest value
max_ch = np.where(evoked.data == evoked.data.max())[0]
max_ch_name = evoked.info['ch_names'][max_ch[0]]
print(max_ch_name)
evoked.pick_channels([max_ch_name])

# pick peak, define time window for peak here
peak = evoked.get_peak(return_amplitude=True,
                      mode='abs', tmin=0.07, tmax=0.140) # early lip window

tmin = peak[1] - 0.015 # take 15 ms around peak
tmax = peak[1] + 0.015
time_mask = _time_mask(times=evoked.times, tmin=tmin, tmax=tmax,
                       sfreq=evoked.info['sfreq'], raise_error=True)
pick = mne.pick_types(evoked.info, meg=True)
data = evoked.data[pick[0], time_mask] 
auc = np.sum(np.abs(data)) * len(data) * (1. / evoked.info['sfreq'])

print('AUC value: %s' % auc)
print('peak latency (s): %s' % peak[1])

# plot channel with peak and AUC window
plt.figure()
plt.plot(evoked.times, evoked.data[0])
plt.axvline(peak[1], linestyle='-', color='r')
plt.axvspan(tmin, tmax, alpha=0.15, color='r')
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.suptitle('Max channel: %s' % max_ch_name)
plt.show()
