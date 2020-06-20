"""
extract data from Schaefer parcellation for aim1 data

take in an fmriprepped func file

use nideconv to deconvolve on schaefer region data

"""

import os
import nideconv
import pandas as pd
from deconv_utils import extract_timecourse_from_nii
import argparse
from templateflow import api as tflow
import json
import glob


def get_args():
    parser = argparse.ArgumentParser(description='aim1 timelocked responses')
    parser.add_argument('-f', '--filename', required=True)
    parser.add_argument('--templateflow_home',
                        help='templateflow home directory')
    parser.add_argument('--nparcels', default=400,
                        help='number of parcels in parcellation')
    parser.add_argument('--nnetworks', default=17,
                        help='number of Yeo networks (7 or 17)')
    parser.add_argument('--atlas', default='Schaefer2018')
    parser.add_argument('--atlas_resolution', default=2)
    parser.add_argument('--TR', default=0.68)
    parser.add_argument('--use_confounds', default=True)
    parser.add_argument('--overwrite', default=False,
                        help='overwrite existing timecourse derivatives')
    parser.add_argument('--hpf_cutoff', default=88,
                        help='cutoff period for high-pass filter (seconds)')
    args = parser.parse_args()

    if 'TEMPLATEFLOW_HOME' not in os.environ:
        if args.templateflow_home is not None:
            os.environ['TEMPLATEFLOW_HOME'] = args.templateflow_home
        else:
            raise Exception('Templateflow home must be specified as environ var or argument')

    return(args)


def get_file_info(args):
    info_dict = args.__dict__
    # assumes this is an fmriprep preprocessed bold image
    full_split = info_dict['filename'].split('/')
    assert full_split[-1].find('preproc_bold.nii.gz') > -1
    assert full_split[-5] == 'fmriprep'
    info_dict['basedir'] = '/'.join(full_split[:-6])
    info_dict['deriv_base'] = '/'.join(full_split[:-5])
    filename_split = os.path.basename(info_dict['filename']).split('_')
    for i in filename_split:
        if '-' not in i:
            continue
        i_split = i.split('-')
        info_dict[i_split[0]] = i_split[1]
    return(info_dict)


def setup_derivative_dirs(info_dict, makedirs=True):
    basedirs = {'parcel_base': 'parcellation',
                'deconv_base': 'deconvolution'}
    for basedir in basedirs:
        info_dict[basedir] = os.path.join(
            info_dict['deriv_base'],
            basedirs[basedir]
        )
        if makedirs and not os.path.exists(info_dict[basedir]):
            os.mkdir(info_dict[basedir])

        dirtype = basedir.replace('_base', '')
        info_dict[f'{dirtype}_sub'] = os.path.join(
            info_dict[basedir],
            f"sub-{info_dict['sub']}",
            'func'
        )
        if makedirs and not os.path.exists(info_dict[f'{dirtype}_sub']):
            try:
                os.makedirs(info_dict[f'{dirtype}_sub'])
            except FileExistsError:
                pass
    return(info_dict)


def get_atlas_path(info_dict, args):
    info_dict['atlas_desc'] = f'{args.nparcels}Parcels{args.nnetworks}Networks'
    info_dict['atlas_path'] = str(
        tflow.get(info_dict['space'],
                  desc=info_dict['atlas_desc'],
                  resolution=args.atlas_resolution,
                  atlas=args.atlas))
    return(info_dict)


def save_parcellation_derivatives(timecourses, info_dict):
    timecourses.to_csv(info_dict['parcel_deriv_path'], sep='\t', index=False)
    deriv_sidecar = info_dict['parcel_deriv_path'].replace('.tsv', '.json')
    assert deriv_sidecar != info_dict['parcel_deriv_path']
    with open(deriv_sidecar, 'w') as f:
        json.dump(info_dict, f, indent=4)
    return(info_dict)


def get_parcel_derivative_path(info_dict):
    deriv_filename = f'sub-{info_dict["sub"]}_task-{info_dict["task"]}_run-{info_dict["run"]}_atlas-{info_dict["atlas"]}_desc-{info_dict["atlas_desc"]}_confounds-{info_dict["use_confounds"]}_timeseries.tsv'
    info_dict['parcel_deriv_path'] = os.path.join(
        info_dict['parcel_sub'],
        deriv_filename
    )
    return(info_dict)


def extract_timecourses(info_dict):
    confounds, info_dict = get_confounds(info_dict) if info_dict['use_confounds'] else (None, info_dict)
    timecourses = extract_timecourse_from_nii(
        info_dict['atlas_path'],
        info_dict['filename'],
        t_r=info_dict['TR'],
        atlas_type='labels',
        low_pass=None,
        high_pass=1. / info_dict['hpf_cutoff'],
        confounds=confounds)
    info_dict = save_parcellation_derivatives(timecourses, info_dict)
    return(info_dict)


def get_confounds(info_dict, confounds_to_include=None):
    if confounds_to_include is None:
        confounds_to_include = ['framewise_displacement', 'a_comp_cor_00',
                                'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
                                'a_comp_cor_04', 'a_comp_cor_05', 'a_comp_cor_06',
                                'a_comp_cor_07', 'trans_x', 'trans_y', 'trans_z',
                                'rot_x', 'rot_y', 'rot_z']
    fmriprep_funcdir = os.path.dirname(info_dict['filename'])
    confound_file = glob.glob(os.path.join(
        fmriprep_funcdir,
        f'sub-{info_dict["sub"]}_ses-{info_dict["ses"]}_task-{info_dict["task"]}_run-{info_dict["run"]}_desc-confounds_regressors.tsv'
    ))
    assert len(confound_file) == 1
    info_dict['confound_file'] = confound_file[0]
    confounds = pd.read_csv(info_dict['confound_file'], sep='\t').fillna(0)
    return((confounds[confounds_to_include].values, info_dict))


def setup_events(info_dict):
    info_dict['events_file'] = os.path.join(
        info_dict['basedir'],
        f'sub-{info_dict["sub"]}/ses-{info_dict["ses"]}/func/sub-{info_dict["sub"]}_ses-{info_dict["ses"]}_task-{info_dict["task"]}_run-{info_dict["run"]}_events.tsv')
    events = pd.read_csv(info_dict['events_file'], sep='\t')
    events_dict = {}
    trial_types = events.loc[:, 'trial_type'].unique()
    for trial_type in trial_types:
        condition_events = events.query('trial_type == "%s"' % trial_type)
        events_dict[trial_type] = condition_events.onset.values
    return((events_dict, info_dict))


def setup_fir_fitter(info_dict, events_dict, timecourses, windowlen=20):
    windowlen = windowlen - windowlen % info_dict['TR']
    rf = nideconv.ResponseFitter(
        input_signal=timecourses,
        sample_rate=1 / info_dict['TR'])
    for trial_type in events_dict:
        rf.add_event(event_name=trial_type,
                     onsets=events_dict[trial_type],
                     interval=[0, windowlen])
    return(rf)


def get_evoked_responses(rf):
    rf.fit()
    return(rf.get_timecourses())


def save_deconv_derivatives(evoked_responses, info_dict):
    info_dict = get_deconv_derivative_path(info_dict)
    evoked_responses.to_csv(info_dict['deconv_deriv_path'], sep='\t', index=False)
    deriv_sidecar = info_dict['deconv_deriv_path'].replace('.tsv', '.json')
    assert deriv_sidecar != info_dict['deconv_deriv_path']
    with open(deriv_sidecar, 'w') as f:
        json.dump(info_dict, f, indent=4)


def get_deconv_derivative_path(info_dict):
    deriv_filename = f'sub-{info_dict["sub"]}_task-{info_dict["task"]}_run-{info_dict["run"]}_atlas-{info_dict["atlas"]}_desc-{info_dict["atlas_desc"]}_confounds-{info_dict["use_confounds"]}_deconvolved.tsv'
    info_dict['deconv_deriv_path'] = os.path.join(
        info_dict['deconv_sub'],
        deriv_filename
    )
    return(info_dict)


if __name__ == '__main__':

    args = get_args()
    info_dict = get_file_info(args)
    info_dict = setup_derivative_dirs(info_dict)
    info_dict = get_atlas_path(info_dict, args)
    info_dict = get_parcel_derivative_path(info_dict)

    if not os.path.exists(info_dict['parcel_deriv_path']) or args.overwrite:
        info_dict = extract_timecourses(info_dict)
    timecourses = pd.read_csv(info_dict['parcel_deriv_path'], sep='\t')
    event_dict, info_dict = setup_events(info_dict)
    rf = setup_fir_fitter(info_dict, event_dict, timecourses)
    evoked_responses = get_evoked_responses(rf)
    save_deconv_derivatives(evoked_responses, info_dict)
