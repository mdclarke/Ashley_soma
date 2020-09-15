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
hemi = 'right' # left or right
tmin,tmax = 0.300,0.400
###

left = ['MEG0342','MEG0343','MEG0323','MEG0322','MEG0332','MEG0333','MEG0643','MEG0642','MEG0212','MEG0213','MEG0223','MEG0222','MEG0412','MEG0413','MEG0423','MEG0422','MEG0632','MEG0633','MEG0242','MEG0243','MEG0233','MEG0232','MEG0442','MEG0443','MEG0433','MEG0432','MEG0712','MEG0713','MEG1612','MEG1613','MEG1623','MEG1622','MEG1812','MEG1813','MEG1823','MEG1822','MEG0742','MEG0743','MEG1642','MEG1643','MEG1633','MEG1632','MEG1842','MEG1843','MEG1833','MEG1832','MEG2012','MEG2013']
right = ['MEG1033','MEG1032','MEG1242','MEG1243','MEG1233','MEG1232','MEG1222','MEG1223','MEG1042','MEG1043','MEG1113','MEG1112','MEG1122','MEG1123','MEG1313','MEG1312','MEG1322','MEG1323','MEG0722','MEG0723','MEG1143','MEG1142','MEG1132','MEG1133','MEG1343','MEG1342','MEG1332','MEG1333','MEG0732','MEG0733','MEG2213','MEG2212','MEG2222','MEG2223','MEG2413','MEG2412','MEG2422','MEG2423','MEG2243','MEG2242','MEG2232','MEG2233','MEG2443','MEG2442','MEG2432','MEG2433','MEG2022','MEG2023']
full = ['MEG0342','MEG0343','MEG0323','MEG0322','MEG0332','MEG0333','MEG0643','MEG0642','MEG0212','MEG0213','MEG0223','MEG0222','MEG0412','MEG0413','MEG0423','MEG0422','MEG0632','MEG0633','MEG0242','MEG0243','MEG0233','MEG0232','MEG0442','MEG0443','MEG0433','MEG0432','MEG0712','MEG0713','MEG1612','MEG1613','MEG1623','MEG1622','MEG1812','MEG1813','MEG1823','MEG1822','MEG0742','MEG0743','MEG1642','MEG1643','MEG1633','MEG1632','MEG1842','MEG1843','MEG1833','MEG1832','MEG2012','MEG2013','MEG1033','MEG1032','MEG1242','MEG1243','MEG1233','MEG1232','MEG1222','MEG1223','MEG1042','MEG1043','MEG1113','MEG1112','MEG1122','MEG1123','MEG1313','MEG1312','MEG1322','MEG1323','MEG0722','MEG0723','MEG1143','MEG1142','MEG1132','MEG1133','MEG1343','MEG1342','MEG1332','MEG1333','MEG0732','MEG0733','MEG2213','MEG2212','MEG2222','MEG2223','MEG2413','MEG2412','MEG2422','MEG2423','MEG2243','MEG2242','MEG2232','MEG2233','MEG2443','MEG2442','MEG2432','MEG2433','MEG2022','MEG2023']

path = '/Users/ashdrew/Documents/Soma_2/KasgaDownload082320/'
fname = op.join(path, 'Locations_40-sss_eq_soma3_437-2-ave.fif')

evoked = mne.read_evokeds(fname)[0]

if evoked.comment == 'lip' and hemi == 'left':
  evoked.pick_channels(ch_names=left)
if evoked.comment == 'lip' and hemi == 'right':
  evoked.pick_channels(ch_names=right)
if evoked.comment == 'foot' and hemi == 'left':
  evoked.pick_channels(ch_names=left)
if evoked.comment == 'foot' and hemi == 'right':
  evoked.pick_channels(ch_names=right) 
if evoked.comment == 'hand'and hemi == 'left':
  evoked.pick_channels(ch_names=left)
if evoked.comment == 'hand'and hemi == 'right':
  evoked.pick_channels(ch_names=right)
else:
  print("No condition named in file")
ev = evoked.copy()
ev.crop(tmin, tmax)
peak = ev.get_peak(return_amplitude=True,
                      mode='abs') # early lip window value

tmin = peak[1] - 0.015 # take 15 ms around peak
tmax = peak[1] + 0.015
time_mask = _time_mask(times=evoked.times, tmin=tmin, tmax=tmax,
                       sfreq=evoked.info['sfreq'], raise_error=True)
pick = mne.pick_types(evoked.info, meg=True)
data = evoked.data[pick[0], time_mask] 
auc = np.sum(np.abs(data)) * len(data) * (1. / evoked.info['sfreq'])

print('AUC value: %s' % auc)
print('peaklatency [seconds]: %s' % peak[1]) 

evoked.pick_channels(ch_names=[peak[0]]) # plot w peak channel

# plot channel with peak and AUC window
plt.figure()
plt.plot(evoked.times, evoked.data[0])
plt.axvline(peak[1], linestyle='-', color='r')
plt.axvspan(tmin, tmax, alpha=0.15, color='r')
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.suptitle('Max channel: %s' % peak[0])
plt.show()
