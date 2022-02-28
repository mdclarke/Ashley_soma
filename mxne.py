!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:35:26 2022

@author: mdclarke
"""
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne

## change subject ##
subj = 'soma3_437'

left = ['MEG0623', 'MEG0622', 'MEG0621', 'MEG0643', 'MEG0642', 'MEG0641', 
        'MEG0332', 'MEG0333', 'MEG0331', 'MEG0323', 'MEG0322', 'MEG0321', 
        'MEG0342', 'MEG0343', 'MEG0341', 'MEG0123', 'MEG0122', 'MEG0121', 
        'MEG0112', 'MEG0113', 'MEG0111', 'MEG0632', 'MEG0633', 'MEG0631', 
        'MEG0423', 'MEG0422', 'MEG0421', 'MEG0412', 'MEG0413', 'MEG0411',
        'MEG0223', 'MEG0222', 'MEG0221', 'MEG0212', 'MEG0213', 'MEG0211', 
        'MEG0133', 'MEG0132', 'MEG0131', 'MEG0142', 'MEG0143', 'MEG0141',
        'MEG0712', 'MEG0713', 'MEG0711', 'MEG0433', 'MEG0432', 'MEG0431',
        'MEG0442', 'MEG0443', 'MEG0441', 'MEG0233', 'MEG0232', 'MEG0231', 
        'MEG0242', 'MEG0243', 'MEG0241', 'MEG1513', 'MEG1512', 'MEG1511', 
        'MEG1542', 'MEG1543', 'MEG1541', 'MEG0742', 'MEG0743', 'MEG0741',
        'MEG1823', 'MEG1822', 'MEG1821', 'MEG1812', 'MEG1813', 'MEG1811', 
        'MEG1623', 'MEG1622', 'MEG1621', 'MEG1612', 'MEG1613', 'MEG1611',
        'MEG1523', 'MEG1522', 'MEG1521', 'MEG1532', 'MEG1533', 'MEG1531', 
        'MEG1833', 'MEG1832', 'MEG1831', 'MEG1842', 'MEG1843', 'MEG1841',
        'MEG1633', 'MEG1632', 'MEG1631', 'MEG1642', 'MEG1643', 'MEG1641', 
        'MEG1723', 'MEG1722', 'MEG1721', 'MEG1532', 'MEG1533', 'MEG1531',
        'MEG2012', 'MEG2013', 'MEG2011', 'MEG1913', 'MEG1912', 'MEG1911', 
        'MEG1942', 'MEG1943', 'MEG1941', 'MEG1733', 'MEG1732', 'MEG1731',
        'MEG1712', 'MEG1713', 'MEG1711', 'MEG2043', 'MEG2042', 'MEG2041']
right = []

## select channel selection ##
hemi = left

subjects_dir = '/storage/Maggie/anat/subjects/'
path = '/storage/'

fname_evoked = op.join(path, 'inverse', 
                       'Locations_40-sss_eq_%s-ave.fif' %subj)
fname_src = op.join(subjects_dir, '%s' %subj, 'bem', '%s-oct-6-src.fif' %subj)
fname_cov = op.join(path, 'covariance', '%s-40-sss-cov.fif' %subj)
fname_trans = op.join(path, 'trans', '%s-trans.fif' %subj)
fname_epochs = op.join(path, 'epochs', 'All_40-sss_%s-epo.fif' %subj)
fname_bem = op.join(subjects_dir, '%s' %subj, 'bem', '%s-5120-bem-sol.fif' %subj)

epochs = mne.read_epochs(fname_epochs)
cov = mne.read_cov(fname_cov)
trans = mne.read_trans(fname_trans)
src = mne.read_source_spaces(fname_src)
bem = mne.read_bem_solution(fname_bem)

evoked = mne.read_evokeds(fname_evoked, condition='lip')
if hemi == left:
    evoked.pick_channels(left)
elif hemi == right:
    evoked.pick_channels(right)
else:
    print('choose left or right channel selection')
    
forward = mne.make_forward_solution(evoked.info, trans, src, bem)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, forward, cov)

stc_mne = mne.minimum_norm.apply_inverse(evoked, inv)
alpha = 20
weights_min = 5
noise_cov = mne.read_cov(fname_cov)

stc_mxne_dip = mne.inverse_sparse.mixed_norm(
    evoked, forward, noise_cov, alpha,
    weights=stc_mne, weights_min=weights_min,
    n_mxne_iter=5, return_as_dipoles=True, verbose=True)

for d in stc_mxne_dip:
    gof = d.gof
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})

    best_dip_idx = gof.argmax()
    max_gof = gof[best_dip_idx]
    best_dip_time = d.times[best_dip_idx]

    # plot each dipole as a function of time with latency of highest GOF%
    d.plot_amplitudes()
    plt.plot(d.times, gof, color='red', alpha=0.5, label='GOF', linewidth=1)
    plt.axvline(x = best_dip_time, color = 'b', 
                label = 'Max GOF: %d' %max_gof + '%', alpha=0.5, 
                linestyle='dashed')
    plt.title('dipole %s' %d)
    plt.legend()
    
for d in stc_mxne_dip:
    d.plot_locations(trans, subj, idx='gof')

stc_mxne, residual = mne.inverse_sparse.mixed_norm(
    evoked, forward, noise_cov, alpha,
    weights=stc_mne, weights_min=weights_min,
    n_mxne_iter=5, return_residual=True, verbose=True)

evoked.pick_types(meg=True)
fig, axes = plt.subplots(2, 2)
evoked.plot(axes=axes[:, 0])
residual.plot(axes=axes[:, 1])
for ii in range(2):
    max_ = max(ax.get_ylim()[1] for ax in axes[ii])
    min_ = min(ax.get_ylim()[0] for ax in axes[ii])
    for ax in axes[ii]:
        ax.set_ylim([min_, max_])
kwargs = dict(src=inv['src'], subjects_dir=subjects_dir, initial_time=0.124,
              views=['axial', 'sagittal', 'coronal'], view_layout='horizontal',
              size=(900, 300), show_traces=0.5, surface='pial')
with mne.viz.use_3d_backend('pyvista'):
    brain_mne = stc_mne.plot(**kwargs)
    brain_mxne = stc_mxne.plot(**kwargs)
