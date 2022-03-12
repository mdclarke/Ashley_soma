#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:21:26 2022

@author: mdclarke

Find channel with highest peak and plot latency 
"""

import mne
from os import path as op
import numpy as np
import matplotlib.pyplot as plt

path = '/storage/' # change this to shared path where grand average files live

evoked = mne.read_evokeds(op.join(path, 
                                  'GrandAve_8infants_lipgroup_40_Locations_N8-ave(1).fif'))[0]

### change these ###
hemi = 'right' # left or right
tmin,tmax = 0.300,0.400 # time window where you want to find largest peak
###

left = ['MEG0342','MEG0343','MEG0323','MEG0322','MEG0332',
        'MEG0333','MEG0643','MEG0642','MEG0212','MEG0213',
        'MEG0223','MEG0222','MEG0412','MEG0413','MEG0423',
        'MEG0422','MEG0632','MEG0633','MEG0242','MEG0243',
        'MEG0233','MEG0232','MEG0442','MEG0443','MEG0433',
        'MEG0432','MEG0712','MEG0713','MEG1612','MEG1613',
        'MEG1623','MEG1622','MEG1812','MEG1813','MEG1823',
        'MEG1822','MEG0742','MEG0743','MEG1642','MEG1643',
        'MEG1633','MEG1632','MEG1842','MEG1843','MEG1833',
        'MEG1832','MEG2012','MEG2013']
right = ['MEG1033','MEG1032','MEG1242','MEG1243','MEG1233',
         'MEG1232','MEG1222','MEG1223','MEG1042','MEG1043',
         'MEG1113','MEG1112','MEG1122','MEG1123','MEG1313',
         'MEG1312','MEG1322','MEG1323','MEG0722','MEG0723',
         'MEG1143','MEG1142','MEG1132','MEG1133','MEG1343',
         'MEG1342','MEG1332','MEG1333','MEG0732','MEG0733',
         'MEG2213','MEG2212','MEG2222','MEG2223','MEG2413',
         'MEG2412','MEG2422','MEG2423','MEG2243','MEG2242',
         'MEG2232','MEG2233','MEG2443','MEG2442','MEG2432',
         'MEG2433','MEG2022','MEG2023']

ev = evoked.copy()

if hemi == 'left':
  ev.pick_channels(ch_names=left)
if hemi == 'right':
  ev.pick_channels(ch_names=right)
else:
  print("Choose left ot right hemispshere")

# find channel with largest peak 
max_ch = np.where(ev.data == max(ev.data.min(), ev.data.max(), key=abs))[0]
max_ch_name = ev.info['ch_names'][max_ch[0]]
print(max_ch_name)
ev = ev.pick_channels([max_ch_name])
peak = ev.get_peak(return_amplitude=True,
                      mode='abs') # early lip window value
# plot all chanels
evoked.plot_topo()

# plot chosen channel with peak latency
plt.figure()
plt.plot(ev.times, ev.data[0])
plt.axvline(peak[1], linestyle='-', color='r')
plt.axhline(0, linestyle='--', color='k')
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.suptitle('Max channel: %s' % peak[0])
plt.show()

# plot sensor location on head
mne.viz.plot_sensors(ev.info, show_names=True)
