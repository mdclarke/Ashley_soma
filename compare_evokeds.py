#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:25:53 2019

@author: ashdrew
"""

import mne
from os import path as op

path = '/Users/ashdrew/Documents/Soma_2/'

# filenames
fname1 = op.join(path, 'grandaves', 'GrandAve_8infants_liponlygroup_40_Locations_N8-ave.fif')
fname2 = op.join(path, 'grandaves', 'GrandAve_8infants_handonlygroup_40_Locations_N8-ave.fif')
fname3 = op.join(path, 'grandaves', 'GrandAve_8infants_footonlygroup_40_Locations_N8-ave.fif')
 
# read evoked files  
lip = mne.read_evokeds(fname1)[0]
lip.pick_types(meg='grad')
hand = mne.read_evokeds(fname2)[0]
hand.pick_types(meg='grad')
foot = mne.read_evokeds(fname3)[0]
foot.pick_types(meg='grad')

# make dict
ev_dict = {'lip': lip, 'hand': hand, 'foot': foot}

# plot
mne.viz.plot_compare_evokeds(ev_dict,legend=True) # plot averaged waveforms
mne.viz.plot_compare_evokeds(ev_dict,legend=True, axes='topo') # plot waveform per channel
