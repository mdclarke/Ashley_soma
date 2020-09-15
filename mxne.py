import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
fname_evoked = 'soma3_437/inverse/Locations_40-sss_eq_soma3_437-ave.fif'
fname_inv = 'soma3_437/inverse/soma3_437-40-sss-meg-free-inv.fif'
fname_fwd = 'soma3_437/forward/soma3_437-sss-fwd.fif'
fname_cov = 'soma3_437/covariance/soma3_437-40-sss-cov.fif'
subjects_dir = '/Users/ashdrew/anatomy/subjects/'

old_time = 1599591877  # 15:04 EST Sept 8 2020

# rerun with fixed MNE if necessary
subject = 'soma3_437'
check_file = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
cfg = mne.coreg.read_mri_cfg(subject, subjects_dir)
if os.stat(check_file).st_mtime < old_time:
    kwargs = {key: cfg[key] for key in ('subject_from', 'scale')}
    kwargs.update(subject_to=subject, subjects_dir=subjects_dir, verbose=True,
                  overwrite=True)
    mne.coreg.scale_mri(**kwargs)
scale = np.concatenate([cfg['scale'], [1.]])
src_mri_t = mne.read_source_spaces(
    op.join(subjects_dir, subject, 'bem', f'{subject}-vol5-src.fif')
)[0]['src_mri_t']

evoked = mne.read_evokeds(fname_evoked, condition='lip')
inv = mne.minimum_norm.read_inverse_operator(fname_inv)
if os.stat(fname_inv).st_mtime < old_time:
    inv['src'][0]['src_mri_t'] = src_mri_t
stc_mne = mne.minimum_norm.apply_inverse(evoked, inv)
alpha = 20
weights_min = 5
noise_cov = mne.read_cov(fname_cov)
forward = mne.read_forward_solution(fname_fwd)
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
    brain_mne = stc_mne.plot_3d(**kwargs)
    brain_mxne = stc_mxne.plot_3d(**kwargs)
