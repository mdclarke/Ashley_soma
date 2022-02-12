#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:55:06 2022
​
@author: ashdrew
"""
import numpy as np
import mne
from mne import read_evokeds
from mne.viz import plot_arrowmap

path = '/Users/ashdrew/Soma_Data/grandaves/'
fname = path + 'GrandAve_8infants_handgroup_40_Locations_N8-ave.fif'
lpf = 40
condition = 'hand'
evoked = read_evokeds(fname, condition=condition)
data = evoked.data
times = evoked.times
assert len(data[1] == len(times))

evoked_mag = evoked.copy().pick_types(meg='mag')
evoked_grad = evoked.copy().pick_types(meg='grad')

# take peak time
#max_time_idx = np.abs(evoked_mag.data).mean(axis=0).argmax()

# take chosen time - change min and max values here:
mask = np.where(np.logical_and(evoked.times>=0.119, evoked.times<=0.120))

print('CHECK THAT I AM THE RIGHT NUMBER ASHLEY!!!! %s' %evoked.times[mask])
squeeze = np.squeeze(evoked_mag.data[:, mask])
plot_arrowmap(squeeze, evoked_mag.info)
