"""
create images for deconvolved timeseries
within each atlas region
"""

import os
from glob import glob
import pandas as pd
import collections
import numpy as np
import json
from templateflow import api as tflow
import nibabel
import nilearn.input_data

def spec_dict_to_string(spec_dict):
    spec_string = ['%s-%s' % (k, spec_dict[k]) for k in spec_dict]
    return('_'.join(spec_string))


def get_deconv_file_info_from_json(filename):
    with open(filename.replace('.tsv', '.json')) as f:
        info_dict = json.load(f)
    info_dict['filename'] = filename
    full_split = info_dict['filename'].split('/')
    info_dict['basedir'] = '/'.join(full_split[:-6])
    info_dict['deriv_base'] = '/'.join(full_split[:-5])
    info_dict['desc'] = info_dict['atlas_desc']
    return(info_dict)


def check_spec_match(info_dict, spec_dict):
    spec_match = True
    for key in spec_dict:
        if info_dict[key] != spec_dict[key]:
            # print('mismatch:', info_dict[key], spec_dict[key])
            spec_match = False
    return(spec_match)


def get_atlas_info(spec_dict, templateflow_home):
    atlas_info_file = os.path.join(
        templateflow_home,
        f'tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_atlas-{spec_dict["atlas"]}_desc-{spec_dict["desc"]}_dseg.tsv')
    atlas_info = pd.read_csv(atlas_info_file, sep='\t')
    atlas_info['network'] = [i.split('_')[2] for i in atlas_info.name]
    return(atlas_info)


def simplify_deconv_data(df):
    # the data from nideconv have multiple rows with identical values
    # which we drop to save time later on
    data_df = df.copy()
    # since time varies, we need to set it to the index so it won't be counted
    data_df.index = data_df.time
    del data_df['time']
    data_df = data_df.drop_duplicates()
    return(data_df)


def smooth_timeseries(df, smoothing_window=3):
    status_vars = ('time', 'event_type')
    data_df = df.copy()
    smoothed_data = None

    # smooth separately for each event type since they
    # are concatenated in the dataset
    for event_type in data_df.event_type.unique():
        event_data = data_df.query('event_type == "%s"' % event_type)
        event_data_copy = event_data.copy()
        for sv in status_vars:
            del event_data_copy[sv]

        event_data_smoothed = event_data_copy.rolling(5, min_periods=1).mean()
        for sv in status_vars:
            event_data_smoothed[sv] = event_data[sv]
        
        if smoothed_data is None:
            smoothed_data = event_data
        else:
            smoothed_data = pd.concat((smoothed_data, event_data_smoothed))

    return(smoothed_data)


def get_combined_data(datafiles, info_dict, spec_dict):
    combined_data = None
    for task in datafiles:
        print(f'found {len(datafiles[task])} datasets for task {task}')
        for datafile in datafiles[task]:
            data = pd.read_csv(datafile, sep='\t')
            data = data.rename(columns={"event type": "event_type"})
            del data['covariate']
            data = data.fillna(0)
            data_smooth = smooth_timeseries(data)
            data_smooth['subcode'] = info_dict[datafile]['sub']
            data_smooth['task'] = task
            #data_smooth['time'] = data_smooth.index
            if combined_data is None:
                combined_data = data_smooth
            else:
                combined_data = pd.concat((combined_data, data_smooth))

    return(pd.melt(combined_data,
           id_vars=['event_type', 'subcode', 'task', 'time'],
           var_name='network', value_name='response'))


def get_mean_response(combined_data):
   return(combined_data.groupby(['task', 'event_type', 'time', 'network']).mean().reset_index())


def get_matching_datafiles(resultfiles, spec_dict):
    datafiles = collections.defaultdict(lambda: [])
    info_dict = {}
    for filename in resultfiles:
        idict = get_deconv_file_info_from_json(filename)
        if check_spec_match(idict, spec_dict):
            info_dict[filename] = idict
            datafiles[info_dict[filename]['task']].append(filename)
    return((datafiles, info_dict))


def map_response_to_image(mean_response, task, atlas_path, image_dir):
    atlas_masker = nilearn.input_data.NiftiMasker()
    atlas_img = atlas_masker.fit(str(atlas_path))
    atlas_img_nib = nibabel.load(str(atlas_path))
    atlas_data = atlas_img.transform(str(atlas_path))
    task_response = mean_response.query('task == "%s"' % task)
    timepoints = task_response.time.unique()
    for event_type in task_response.event_type.unique():
        respdata = np.zeros((len(timepoints), atlas_data.shape[1]))
        event_response = task_response.query('event_type == "%s"' % event_type)
        for network in task_response.network.unique():
            network_idx = int(network) + 1
            network_vox = atlas_data == network_idx
            n_network_vox = np.sum(network_vox)
            respdata[:, network_vox[0, :]] = np.repeat(
                event_response.loc[event_response.network==network, 'response'].values[:, np.newaxis], n_network_vox, axis=1)
        event_response_image = atlas_img.inverse_transform(respdata)
        zooms = list(event_response_image.header.get_zooms())
        zooms[3] = 0.68
        event_response_image.header.set_zooms(zooms)
        event_response_image.to_filename(os.path.join(image_dir, '%s_%s.nii.gz' % (task, event_type)))
                
            


if __name__ == "__main__":

    basedir = '/Users/poldrack/data_unsynced/uh2/BIDS_data/derivatives/deconvolution'
    templateflow_home = os.environ['TEMPLATEFLOW_HOME']
    # set the intended values to match on here
    spec_dict = {'use_ridge': False,
                 'atlas': 'Schaefer2018',
                 'use_confounds': True,
                 'desc': '400Parcels17Networks',
                 'nparcels': 400,
                 'nnetworks': 17,
                 'atlas': 'Schaefer2018',
                 'atlas_resolution': 2,
                 'space': 'MNI152NLin2009cAsym',
                 'atlas_desc': '400Parcels17Networks'}

    atlas_path =  tflow.get(spec_dict['space'],
                  desc=spec_dict['atlas_desc'],
                  resolution=spec_dict['atlas_resolution'],
                  atlas=spec_dict['atlas'])

    image_dir = os.path.join(basedir, 'images')
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    resultfiles = glob(os.path.join(basedir, 'sub*/func/*_deconvolved.tsv'))

    spec_string = spec_dict_to_string(spec_dict)

    datafiles, info_dict = get_matching_datafiles(resultfiles, spec_dict)

    combined_data_long = get_combined_data(datafiles, info_dict, spec_dict)

    mean_response = get_mean_response(combined_data_long)

    for task in mean_response.task.unique():
        print('writing files for', task)
        map_response_to_image(mean_response, task, atlas_path, image_dir)