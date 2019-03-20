# -*- coding: utf-8 -*-

"""
@author: raphael.bacher@gipsa-lab.fr
"""

import numpy as np
import yaml

DEFAULT_PARAMS = {
    # Beta parameter for HST PSF Moffat model
    'beta_hst': 1.6,
    # FHWM parameter for HST PSF Moffat
    'fwhm_hst': 0.085,
    # number of spectral blocks wherein MUSE FSF is considered constant
    'nBands': 10,
    # slope and intercept parameters of MUSE FSF fwhm spectral evolution
    # model: fhwm =  a + b * lambda
    'fsf_a_muse': 0.869,
    'fsf_b_muse': -3.401e-05,
    # Beta parameter for MUSE FSF Moffat model
    'fsf_beta_muse': 2.8,
    # mean wavelegnth to be used for grouping (where white MUSE image is used)
    'fsf_wavelength': 7000,
    # absolute cut value for convolution during grouping
    'cut': 0.005,
    # Get mask containing a fraction alpha_cut of total map intensity
    'alpha_cut': 0.999,
    # regularize deblending
    'regul': True,
    # size of median filter for separating continuum from lines in the
    # regularization process
    'filt_w': 101,
    # minimal width of a group bounding box
    'min_width': 6,
    # additionnal margin in pixels around a ground bounding box
    'margin_bbox': 3,
    # minimal number of sky pixels in a bounding box to estimate correctly
    # the background
    'min_sky_pixels': 20,
}


def load_settings(settings_file):
    """Load the YAML settings, and substitute keys from the 'vars' block."""
    with open(settings_file, 'r') as f:
        conftext = f.read()
    conf = yaml.safe_load(conftext)
    conftext = conftext.format(**conf.get('vars', {}))
    return yaml.safe_load(conftext)


class Params(dict):

    def __init__(self, **kwargs):
        # initialize dict with DEFAULT_PARAMS, and override with the provided
        # parameters (kwargs)
        super().__init__(**DEFAULT_PARAMS)
        self.update(kwargs)
        # this allows to access dict items as attributes
        self.__dict__ = self

    @property
    def fwhm_muse(self):
        return self.fsf_a_muse + self.fsf_b_muse * self.fsf_wavelength

    @property
    def alpha_hst(self):
        # expressed in MUSE pixels
        return np.sqrt((self.fwhm_hst / 0.2 * 15) ** 2 /
                       (4 * (2**(1 / self.beta_hst) - 1)))
