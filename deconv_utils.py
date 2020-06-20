"""
util function for deconvolution
adapted from nideconv by Henry Jones
"""

import pandas as pd
import numpy as np
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker


# HELPERS
# taken from:
# https://github.com/VU-Cog-Sci/nideconv/blob/d7f5c6a71e4cae4159c38b18b72e335a51437493/nideconv/utils/roi.py#L261
def _make_psc(data):
    mean_img = image.mean_img(data)

    # Replace 0s for numerical reasons
    mean_data = mean_img.get_data()
    mean_data[mean_data == 0] = 1
    denom = image.new_img_like(mean_img, mean_data)

    return image.math_img('data / denom[..., np.newaxis] * 100 - 100',
                          data=data, denom=denom)


# modified from:
# https://github.com/VU-Cog-Sci/nideconv/blob/d7f5c6a71e4cae4159c38b18b72e335a51437493/nideconv/utils/roi.py#L8
def extract_timecourse_from_nii(atlas,
                                nii,
                                mask=None,
                                confounds=None,
                                atlas_type=None,
                                t_r=None,
                                low_pass=None,
                                high_pass=1. / 128,
                                *args,
                                **kwargs):
    """
    Extract time courses from a 4D `nii`, one for each label 
    or map in `atlas`,
    This method extracts a set of time series from a 4D nifti file
    (usually BOLD fMRI), corresponding to the ROIs in `atlas`.
    It also performs some minimal preprocessing using 
    `nilearn.signal.clean`.
    It is especially convenient when using atlases from the
    `nilearn.datasets`-module.
    Parameters
    ----------
    atlas: str  
        Path to 3D atlas image to be passed into NiftiLabelsMasker
    nii: 4D niimg-like object
        This NiftiImage contains the time series that need to
        be extracted using `atlas`
    mask: 3D niimg-like object
        Before time series are extracted, this mask is applied,
        can be useful if you want to exclude non-gray matter.
    confounds: CSV file or array-like, optional
        This parameter is passed to nilearn.signal.clean. Please 
        see the related documentation for details.
        shape: (number of scans, number of confounds)
    t_r, float, optional
        Repetition time of `nii`. Can be important for
        temporal filtering.
    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details
    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details
    Examples
    --------
    >>> from nilearn import datasets
    >>> data = '/data/ds001/derivatives/fmriprep/sub-01/func/sub-01_task-checkerboard_bold.nii.gz'
    >>> atlas = datasets.fetch_atlas_pauli_2017()
    >>> ts = extract_timecourse_from_nii(atlas,
                                         data,
                                         t_r=1.5)
    >>> ts.head()
    """

    standardize = kwargs.pop('standardize', False)
    detrend = kwargs.pop('detrend', False)

    masker = NiftiLabelsMasker(atlas,
                               mask_img=mask,
                               standardize=standardize,
                               detrend=detrend,
                               t_r=t_r,
                               low_pass=low_pass,
                               high_pass=high_pass,
                               *args, **kwargs)

    data = _make_psc(nii)

    results = masker.fit_transform(data,
                                   confounds=confounds)

    if t_r is None:   # hold over from original
        t_r = 1

    # build up index with TR increments
    index = pd.Index(np.arange(0,
                               t_r * data.shape[-1],
                               t_r),
                     name='time')

    try:  # occassionaly results has one less TR than original data
        out_df = pd.DataFrame(results,
                              index=index)
    except ValueError:
        out_df = pd.DataFrame(results,
                              index=index[:-1])        
    return(out_df)
