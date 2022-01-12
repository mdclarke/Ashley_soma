#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SOMA_MXNE.PY

Source space analysis of Soma subjects, using mixed norm modeling. Currently
set for surface models. Adapted from code in mxne.py. Called from Soma_MXxx
notebooks.

Created on Thu Mar 4 2021.
@authors: SMB and AD based on original code by EL and MC.
"""

import os
import numpy as np
import re
import matplotlib.pyplot as plt

from dataclasses import make_dataclass
from io import StringIO
from numpy.testing import assert_allclose

from nilearn.plotting import plot_anat
from nilearn.datasets import load_mni152_template

import mne


# Global variables #
mni_template = load_mni152_template()
kwplot = dict(surf='inflated', hemi='split', views=('lateral','medial'))

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle[7] = '#5f7f6f'          # replace grey with olive
color_cycle.extend(['#ffffff']*8)   # pad with white
color_comment = ('COLOR CODE - 0:blue; 1:orange, 2:green, 3:red, 4:purple, '
                '5:brown, 6:pink, 7:olive, 8:yellow, 9:cyan, 10+:white')
use_parc = 'aparc'


# Main and auxiliary functions #
def create_mxne(subjects, p, extra_tag=''):
    '''Make and save mixed-norm STCs for a list of subjects.
    '''            
    for subject in subjects:
        # Define needed files and parameters #
        fname_evk = f'Locations_40-sss_eq_{subject}-ave.fif'
        fname_inv = f'{subject}-40-sss-meg-free-inv.fif'
        fname_fwd = f'{subject}-sss-fwd.fif'
        fname_cov = f'{subject}-40-sss-cov.fif'
        cond = p.stc_params['condition']
        
        # Make adjustments for older subject files #
        # Skipped, as this is likely not necessary for surface spaces.
        
        # Load saved data and model components for the subject
        fname = os.path.join(p.work_dir, subject, p.inverse_dir, fname_evk)
        evoked = mne.read_evokeds(fname, condition=cond)
        fname = os.path.join(p.work_dir, subject, p.inverse_dir, fname_inv)
        inv = mne.minimum_norm.read_inverse_operator(fname)
        fname = os.path.join(p.work_dir, subject, p.cov_dir, fname_cov)
        noise_cov = mne.read_cov(fname)
        fname = os.path.join(p.work_dir, subject, p.forward_dir, fname_fwd)
        forward = mne.read_forward_solution(fname)
    
        # Compute the mixed-norm solution as STCs #
        stc_mne = mne.minimum_norm.apply_inverse(evoked, inv)
        min_, max_ = stc_mne.data.min(), stc_mne.data.max()
        mean_ = stc_mne.data.mean()
        print('Initial STC min/max/mean are: ', min_, max_, mean_)
        
        stc_mxne, residual_mxne = mne.inverse_sparse.mixed_norm(
            evoked, forward, noise_cov, p.stc_params['mxne_alpha'],
            weights=stc_mne, weights_min=p.stc_params['mxne_weight'],
            n_mxne_iter=p.stc_params['mxne_iter'], return_residual=True,
            return_as_dipoles=False, verbose=False)
        dipole_mxne = mne.inverse_sparse.mixed_norm(
            evoked, forward, noise_cov, p.stc_params['mxne_alpha'],
            weights=stc_mne, weights_min=p.stc_params['mxne_weight'],
            n_mxne_iter=p.stc_params['mxne_iter'], return_residual=False,
            return_as_dipoles=True, verbose=False)
        
        # Save the mixed-norm STCs #
        stc_dir = os.path.join(p.work_dir, subject, p.stc_dir)
        if not os.path.isdir(stc_dir):
            os.mkdir(stc_dir)
            print(f'  Created STC directory for {subject}.')
        
        output_stem = os.path.join(stc_dir, subject + '_' + cond + extra_tag)
        stc_mxne.save(output_stem, ftype='stc', verbose=None)

        # Also save meta data with numpy method #
        gof_mxne = []
        for dip in dipole_mxne:
            for s in subjects:
                gof_mxne.append(dip.gof)
                dip.save('/Users/ashdrew/Soma_Data/TWA/dips/%s.dip' %s)
        save_array = np.empty((2,), dtype=np.ndarray)
        save_array[0] = gof_mxne
        save_array[1] = residual_mxne
        
        # Save indivdiual dipoles
        for s in subjects:
          for ii in range(len(dipole_mxne)):
            dipole_mxne[ii].save('/Users/ashdrew/Soma_Data/TWA/dips/%s_%d.dip' %(s, ii), overwrite=True)
        
        save_file = output_stem + '.npy'
        np.save(save_file, save_array, fix_imports=False, overwrite=True)

        print(f'\nProcessed source spectra for {len(subjects)} subjects.')


def analyze_dipoles(stc, gof_list, evoked, noise_cov, t_range=(None,None)):
    '''Single-subject analysis.'''
    n_dipoles, n_times = stc.data.shape
    assert len(evoked.times) == n_times
    assert len(gof_list[0]) == n_times
    assert len(gof_list) == n_dipoles
    assert n_dipoles < 15, 'Too many dipoles; is this a mixed-norm STC?'
    
    t_range = list(t_range)
    if t_range[0] is None:
        t_range[0] = evoked.times[0]
    if t_range[1] is None:
        t_range[1] = evoked.times[-1]
    t_a, t_b = evoked.time_as_index(t_range)
    
    white_mat, _ = mne.cov.compute_whitener(noise_cov, evoked.info)
    evoked_w = np.dot(white_mat, evoked.data)
    var_wgts = np.linalg.norm(evoked_w, axis=0)  # SMB 21.01.04: removed .data
    var_wgts /= var_wgts.max()
    
    Results = make_dataclass('Results', [('mean', float), ('peak', float),
                                         ('pidx', int)])  # for storage
    gof_results = [Results(0,0,0) for i in range(n_dipoles)]
    amp_results = [Results(0,0,0) for i in range(n_dipoles)]

    for di, gof in enumerate(gof_list):
        gof = gof[t_a:t_b+1]
        amp = stc.data[di, t_a:t_b+1]
        # Simple means over user-chosen time range #
        gof_results[di].mean = gof.mean()
        amp_results[di].mean = amp.mean()
        # Peak values over time range #
        gof_wgtd = gof * var_wgts[t_a:t_b+1]
        pidx = np.argmax(gof_wgtd)
        gof_results[di].pidx = pidx + t_a
        gof_results[di].peak = gof[pidx]
        pidx = np.argmax(amp)
        amp_results[di].pidx = pidx + t_a
        amp_results[di].peak = amp[pidx]
    
    return (gof_results, amp_results)


def sort_dipoles(results: (list, list)):
    '''Utility function to sort dipoles based on goodness-of-fit.
    Input is a tuple of gof and amplitude lists of custom Results dataclass,
    one list entry per dipole. Output is the sorted tuple and sorting
    index.
    '''
    gof_results, amp_results = results
    n_dip = len(gof_results)
    
    sort_vec = [gof_results[i].peak for i in range(n_dip)]
    sort_idx = np.argsort(sort_vec)
    sort_idx = sort_idx[::-1]  # now descending order
    
    tmp_results = [gof_results[ii] for ii in sort_idx]
    gof_results2 = tmp_results
    tmp_results = [amp_results[ii] for ii in sort_idx]
    amp_results2 = tmp_results
    results_sorted = (gof_results2, amp_results2)
    
    return results_sorted, sort_idx


def which_label(vertex, hemi, label_set):
    '''In: vertex; Out: label.'''
    for label in [ll for ll in label_set if ll.hemi == hemi]:
        if vertex in label.vertices:
            return label
    return None


def create_mxne_summary(subjects, p, morph_subject=None, n_display=None,
                        pattern_in='', pattern_out='_mxne',
                        path_out='./', title='%s Dipoles', ):
    '''Create a report and spreadsheet about mixed-norm dipoles.'''
    src_file = os.path.join(p.subjects_dir, 'fsaverage',
                             'bem', 'fsaverage-ico-5-src.fif')
    src_fsavg = mne.read_source_spaces(src_file)
    src_file = os.path.join(p.subjects_dir, '14mo_surr',
                             'bem', '14mo_surr-oct-6-src.fif')
    src_14mo = mne.read_source_spaces(src_file)
    
    mni_labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1',
                                            subjects_dir=p.subjects_dir)
    labels = mne.read_labels_from_annot('14mo_surr', use_parc,
                                            subjects_dir=p.subjects_dir)

    for subject in subjects:
        # Load STCs and other saved data #
        cond = p.stc_params['condition']
        stc_path = os.path.join(p.work_dir, subject, p.stc_dir)
        stc_stem = subject + '_' + cond + pattern_in
        stc_file = os.path.join(stc_path, stc_stem)
        if not os.path.isfile(stc_file + '-lh.stc'):
            print(f'** STC file matching {stc_stem} not found ********.\n')
            continue
        stc_mxne = mne.read_source_estimate(stc_file)
        n_dipoles, n_times = stc_mxne.data.shape
        
        meta_data = np.load(stc_file + '.npy', allow_pickle=True)
        gof_mxne = meta_data[0]
        residual_mxne = meta_data[1]
        
        evk_path = os.path.join(p.work_dir, subject, p.inverse_dir)
        evk_file = f'Locations_40-sss_eq_{subject}-ave.fif'
        evk_file = os.path.join(evk_path, evk_file)
        evoked = mne.read_evokeds(evk_file, condition=cond, kind='average')
        evoked.pick_types(meg=True)

        cov_path = os.path.join(p.work_dir, subject, p.cov_dir)
        cov_file = f'{subject}-40-sss-cov.fif'
        cov_file = os.path.join(cov_path, cov_file)
        cov = mne.read_cov(cov_file)
        
        trans_path = os.path.join(p.work_dir, subject, p.trans_dir)
        trans_file = f'{subject}-trans.fif'
        trans_file = os.path.join(trans_path, trans_file)
        trans = mne.read_trans(trans_file, verbose=False)
        
        fwd_path = os.path.join(p.work_dir, subject, p.forward_dir)
        fwd_file = f'{subject}-sss-fwd.fif'
        fwd_file = os.path.join(fwd_path, fwd_file)
        fwd = mne.read_forward_solution(fwd_file, verbose=False)
        
        assert fwd['src'][0]['nuse'] == src_14mo[0]['nuse']
        assert fwd['src'][1]['nuse'] == src_14mo[1]['nuse']
        
        # Run analysis on the dipoles, then sort then by goodness-of-fit #
        results = analyze_dipoles(stc_mxne, gof_mxne, evoked, cov,
                                  p.stc_params['gof_t_range'])
        results, sort_idx = sort_dipoles(results)  # stc still unsorted
        gof_results, amp_results = results
        assert len(gof_results) == n_dipoles
        
        # Collect info for the top dipoles, in order #
        n_show = n_dipoles
        if n_display:
            n_show = min(n_display, n_show)
        n_left = len(stc_mxne.vertices[0])  # .data stacked lh then rh
            
        postop, mnitop, wavtop = [], [], []
        for i in range(n_dipoles):
            di = sort_idx[i]
            hemid = int(di >= n_left)
            vidx = di - hemid * n_left
            vert = stc_mxne.vertices[hemid][vidx]
            pos = fwd['src'][hemid]['rr'][vert]
            postop.append(pos)
            mni = mne.vertex_to_mni(vert, hemid, subject,
                                    subjects_dir=p.subjects_dir)
            mnitop.append(mni)
            wav = stc_mxne.data[di, :]
            wavtop.append(wav)
        assert wav[amp_results[i].pidx] == amp_results[i].peak  # check last

        # Make various figures #
        figure_list, figure_info, figure_comment = [], [], []
        
        # 1) Top dipoles in one set of surface maps.
        if morph_subject:
            src_subject = morph_subject
            caption = 'Surface Plots | ' + morph_subject
        else:
            src_subject = subject
            caption = 'Surface Plots | Coreg.' 
        fig_surface = make_surfaceplots(stc_mxne, src_subject, p.subjects_dir,
                                        sort_idx, parc=use_parc)
        
        figure_list.append(fig_surface)
        figure_info.append([caption, 'Surface Plots'])
        figure_comment.append(color_comment)
        
        # 2) Top dipoles in 3D slices (non-morphed and MNI).
        mri_file = os.path.join(p.subjects_dir, subject, 'mri', 'T1.mgz')
        
        postop_mri = mne.head_to_mri(postop, mri_head_t=trans, subject=subject,
                                     subjects_dir=p.subjects_dir)
        postop_mni = mne.head_to_mni(postop, mri_head_t=trans, subject=subject,
                                     subjects_dir=p.subjects_dir)
        assert_allclose(mnitop[0], postop_mni[0], atol=0.01)
        assert_allclose(mnitop[-1], postop_mni[-1], atol=0.01)
        
        fig_orthog1 = make_orthogplots(mri_file, postop_mri[:n_show])
        fig_orthog2 = make_orthogplots(mni_template, postop_mni[:n_show])
        
        figure_list.append(fig_orthog1)
        figure_info.append(['Orthogonal Plots | Coreg.', 'Orthogonal Plots'])
        figure_comment.append(None)
        figure_list.append(fig_orthog2)
        figure_info.append(['Orthogonal Plots | MNI)', 'Orthogonal Plots'])
        figure_comment.append(f'Top {n_show} of {n_dipoles} dipoles '
                              'displayed.')
        
        # 3) Top dipoles' time waveforms.
        fig_wav = make_sourcewavs(wavtop, stc_mxne.times,
                                   p.stc_params['gof_t_range'])
        figure_list.append(fig_wav)
        figure_info.append(['STC Time Course', 'Temporal Waveforms'])
        figure_comment.append(None)
        
        # 4) Evoked and residual waveforms (averages across sensors)
        fig_sensor = make_sensorwavs(evoked, residual_mxne)
        
        figure_list.append(fig_sensor)
        figure_info.append(['Sensor Time Course', 'Temporal Waveforms'])
        figure_comment.append(None)
        
        # Determine 14-mo surrogate "aparc" label for each dipole #
        labels_stc = []  # note these are not gof-ordered
        for hh, hemi in enumerate(('lh', 'rh')):
            for vert in stc_mxne.vertices[hh]:
                label = which_label(vert, hemi, labels)
                if label:
                    labels_stc.append(label.name)
                else:
                    labels_stc.append('no_label')
        
        # Expand the sparse STC so it can be morphed (X='expanded') #
        # v_lh = fwd['src'][0]['vertno']   # the full source space
        # v_rh = fwd['src'][1]['vertno']
        # n_vtotal = vertices = len(v_lh) + len(v_rh)
        # data_mxneX = np.zeros((n_vtotal, n_times))
        # idx_vrts = np.isin(v_lh, stc_mxne.vertices[0])
        # idx_vrts = np.where(idx_vrts)[0]
        # data_mxneX[idx_vrts, :] = stc_mxne.data[:n_left, :]
        # idx_vrts = np.isin(v_rh, stc_mxne.vertices[1])
        # idx_vrts = np.where(idx_vrts)[0]
        # data_mxneX[idx_vrts, :] = stc_mxne.data[n_left:, :]
        # stc_mxneX = mne.SourceEstimate(data_mxneX, [v_lh, v_rh],
        #     tmin=stc_mxne.tmin, tstep=stc_mxne.tstep,
        #     subject=stc_mxne.subject)
        
        # Determine fsaverage "HCPMMP1" labels for each dipole #
        # Note: 'sparse' doesn't give a 1-to-1 mapping.
        morph_fcn = mne.compute_source_morph(stc_mxne, src_to=src_fsavg, 
                        smooth='nearest', spacing=None, warn=False,
                        subjects_dir=p.subjects_dir, niter_sdr=(), sparse=True,
                        subject_from=subject, subject_to='fsaverage')
        mlabels_stc = []  # like above, but now for fsaverage:HCPMMP1
        
        verts_mni = []
        for di in range(n_dipoles):
            stc_temp = stc_mxne.copy()  # zero all but dipole of interest
            stc_temp.data = np.zeros((n_dipoles, n_times))
            stc_temp.data[di, :] = stc_mxne.data[di, :]
            
            mstc_temp = morph_fcn.apply(stc_temp)
            vidx = np.where(mstc_temp.data[:,0] > 0)[0]
            vidx_lh = [i for i in vidx if i < n_left]  # don't assume hemi
            vidx_rh = [i-n_left for i in vidx if i >= n_left]
            verts_byhemi = [None, None]
            verts_byhemi[0] = mstc_temp.vertices[0][vidx_lh]
            verts_byhemi[1] = mstc_temp.vertices[1][vidx_rh]
            verts_mni.append(verts_byhemi)
            
            cnt = 0
            for verts, hemi, prefix in zip(verts_byhemi, ['lh', 'rh'],
                                            ['L_', 'R_']):
                if not verts:
                    continue
                vert = verts[0]  # should only be one with sparse arg.
                lbl = which_label(vert, hemi, mni_labels)
                if lbl:
                    lbl = lbl.name
                else:
                    lbl = 'no_label'
                lbl = re.sub(rf"^{prefix}", "", lbl)
                lbl = re.sub(r"_ROI", "", lbl)
                cnt += 1
            assert cnt == 1  # only one hemisphere should be valid
            mlabels_stc.append(lbl)

        # Create formatted tables for a report section #
        # SMB: Saving as string objects in case they can be added to report.
        strobj1 = StringIO()  # TABLE 1: sorted gof and amplitude info
        sprint = lambda *x: print(*x, file=strobj1, end='')
        
        ff = '<8.2f'   # format: center on 8-char field, 2 decimal places
        sprint(f'{"Dip #":^6} {"Peak/Mean Amp":<16} '
              f'{"Peak/Mean GOF":<16} {"GOF Time":<8}\n')
        for i in range(n_dipoles):
            amp_m = 1e9 * amp_results[i].mean
            amp_p = 1e9 * amp_results[i].peak
            gof_m = gof_results[i].mean
            gof_p = gof_results[i].peak
            time_p = evoked.times[gof_results[i].pidx]
            sprint(f'{i:^6} {amp_p:{ff}}{amp_m:{ff}} '
                   f'{gof_p:{ff}}{gof_m:{ff}} '
                   f'{time_p:{"<8.3f"}}\n')
        sprint('\n')
        
        strobj2 = StringIO()  # TABLE 2: coordinate and label info
        sprint = lambda *x: print(*x, file=strobj2, end='')
        
        ff = '<20'
        sprint(f'{"Dip #":^6} {"14mo Coord":{ff}} {"MNI Coord":{ff}} '
               f'{"14mo Aparc | Fsavg HCPMMP1":{ff}}\n')
        for i in range(n_dipoles):
            di = sort_idx[i]
            hemid = int(di >= n_left)
            # hemi = 'rh' if hemid else 'lh'
            vidx = di - hemid * n_left
            vert = stc_mxne.vertices[hemid][vidx]
            coord = src_14mo[hemid]['rr'][vert] * 1000
            coord_str = ' '.join([f'{x:.1f}' for x in coord])
            
            vert = verts_mni[di][hemid][0]  # just the first one
            coord = src_fsavg[hemid]['rr'][vert] * 1000
            mcoord_str = ' '.join([f'{x:.1f}' for x in coord])
            
            sprint(f'{i:^6} {coord_str:{ff}} {mcoord_str:{ff}} '
                   f'{labels_stc[di]:{ff}}\n {"":<47} '
                   f'{mlabels_stc[di]:{ff}}\n')
        
        # Print out the tables #
        print(f'\nGOF-sorted dipole info for {subject}:')
        strobj1.seek(0)
        print(strobj1.read())
        strobj1.close()
        
        print(f'\nGOF-sorted position info for {subject}:')
        strobj2.seek(0)
        print(strobj2.read())
        strobj2.close()

        # Compile all figures into a report #
        print(f'Compiling report for {subject}.')
        
        if not os.path.exists(path_out):
            os.mkdir(path_out)
        if '%s ' in title:
            title_use = title.replace('%s ', 'Group')
        else:
            title_use = title

        report = mne.Report(title=title_use, image_format='png')
        for fig, info, cstr in zip(figure_list, figure_info, figure_comment):
            report.add_figs_to_section(fig, captions=info[0], scale=1.0,
                    section=info[1], comments = cstr)
        
        report_file = os.path.join(path_out, subject + pattern_out + '.html')
        report.save(report_file, open_browser=False, overwrite=True)


def make_orthogplots(mri_param, pos_array):
    '''Plot dipoles from pos_array (#dipoles x 3 coords) into one figure.'''
    fig, axset = plt.subplots(len(pos_array), 1,
                              figsize=(14, 5*len(pos_array)),squeeze=False) #added squeeze=False
    
    for i, pos in enumerate(pos_array):
        nl = plot_anat(mri_param, cut_coords=pos, figure=fig, axes=axset[i,0],
                       title=f'Dip {i}', black_bg=True); #changed axes=axset from [i] to [i,0]
        nl.close()
    
    return fig


def make_surfaceplots(stc, src_name, anatomy_dir, sort_idx, parc=None):
    '''Inflated surface plots for left and right hemispheres.'''
    lh_num = len(stc.vertices[0])
    hemi_str = ['lh', 'rh']
    
    brain = mne.viz.Brain(src_name, subjects_dir=anatomy_dir, **kwplot)
    brain.clear_glyphs()
    if parc:
        brain.add_annotation(parc, color='grey')
    for i, di in enumerate(sort_idx):
        hemid = di >= lh_num  # bool, but works as int
        vert = stc.vertices[hemid][di - hemid * lh_num]
        brain.add_foci(vert, coords_as_verts=True, hemi=hemi_str[hemid],
                       color=color_cycle[i])
    
    fig = brain.screenshot()
    brain.close()
    
    return fig


def make_sourcewavs(waveforms, times, t_range=None):
    '''STC amplitude waveform.'''
    fig = plt.figure(figsize=[14,5], edgecolor='blue')
    for i, wav in enumerate(waveforms):
        plt.plot(times, wav, color=color_cycle[i])
    
    legend_str = [f'Dipole {ii}' for ii in range(len(waveforms))]
    plt.figlegend(labels=legend_str)
    
    if t_range:
        ya, yb = plt.ylim()
        t1, t2 = t_range
        plt.plot((t1, t1), (ya, yb), color='grey', linestyle='--')
        plt.plot((t2, t2), (ya, yb), color='grey', linestyle='--')
    plt.ylim((ya, yb))   # in case it just changed
    
    return fig


def make_sensorwavs(evoked, residual):
    '''Evoked waveforms and residuals to model fit.'''
    evoked.pick_types(meg=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8)) #added squeeze=False
    evoked.plot(axes=axes[:, 0], show=False);
    residual.plot(axes=axes[:, 1], show=False);
    for ii in range(2):
        max_ = max(ax.get_ylim()[1] for ax in axes[ii])
        min_ = min(ax.get_ylim()[0] for ax in axes[ii])
        for ax in axes[ii]:
            ax.set_ylim([min_, max_])
    axes[0, 0].set_xlabel(''); axes[0, 1].set_xlabel('')
    axes[0, 0].set_title('Evoked Grad');
    axes[1, 0].set_title('Evoked Mag');
    axes[0, 1].set_title('Residual Grad');
    axes[1, 1].set_title('Residual Mag');
    
    return fig


def create_mxne_spreadsheet():
    '''Single dipole summary for all subjects including table from above.
    '''
    pass
