#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 16:41:17 2021

@author: mdclarke

Plot (interactive) somatosensory evoked grand averages
"""
import mne
import os.path as op

path = '/home/mdclarke/Desktop/soma_files' #change this to shared path where grand average files live

ev1 = mne.read_evokeds(op.join(path, 'GrandAve_8infants_handgroup_40_Locations_N8-ave.fif'))[0]
assert ev1.comment
ev2 = mne.read_evokeds(op.join(path, 'GrandAve_8infants_footgroup_40_Locations_N8-ave.fif'))[0]
assert ev2.comment
ev3 = mne.read_evokeds(op.join(path, 'GrandAve_8infants_lipgroup_40_Locations_N8-ave.fif'))[0]
assert ev3.comment

evokeds_list = [ev1, ev2, ev3]

for e in evokeds_list:
    e.pick_types(meg='grad')

# plot - is interactive so if you click on a channel it will blow it up for you
mne.viz.plot_evoked_topo(evokeds_list, color=['r','k', 'C0'], title='Infant Somatosensory')
