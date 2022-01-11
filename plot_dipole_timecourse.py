#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:15:42 2022

@author: mdclarke

Plot somatosensory dipole timecourses with GOF
"""
import mne
import matplotlib.pyplot as plt
import os.path as op
import numpy as np

path = '/Users/ashdrew/Soma_Data/TWA/dips/'
dips = [408, 409] # enter subject numbers for dipoles you want plotted here

for d in dips:
    dip_fname = op.join(path, 'soma3_%s.dip' %d)

    dip = mne.read_dipole(dip_fname)
    gof = dip.gof
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    best_dip_idx = gof.argmax()
    max_gof = gof[best_dip_idx]
    best_dip_time = dip.times[best_dip_idx]

    # plot each dipole as a function of time with latency of highest GOF%
    dip.plot_amplitudes()
    plt.plot(dip.times, gof, color='red', alpha=0.5, label='GOF', linewidth=1)
    plt.axvline(x = best_dip_time, color = 'b', 
                label = 'Max GOF: %d' %max_gof + '%', alpha=0.5, 
                linestyle='dashed')
    plt.title('dipole %d' %d)
    plt.legend()
