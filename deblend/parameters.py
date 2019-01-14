# -*- coding: utf-8 -*-

"""
@author: raphael.bacher@gipsa-lab.fr
"""

import os
import numpy as np

# get default files for hst response filters

DEFAULT_HSTFILTER606 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F606W_81.dat')
DEFAULT_HSTFILTER775 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F775W_81.dat')
DEFAULT_HSTFILTER814 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F814W_81.dat')
DEFAULT_HSTFILTER850 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F850LP_81.dat')


class Params():

    def __init__(self,
                 beta_hst=1.6, # Beta parameter for HST PSF Moffat model
                 fwhm_hst=0.085, # FHWM parameter for HST PSF Moffat 
                 nBands=10, # number of spectral blocks wherein MUSE FSF is considered constant
                 fsf_a_muse = 0.869, # parameter "a" of MUSE FSF fwhm spectral evolution model : fhwm = a + b * lambda
                 fsf_b_muse = -3.401e-05, # parameter "b" of MUSE FSF fwhm spectral evolution model : fhwm = a + b * lambda
                 fsf_beta_muse = 2.8, #Beta parameter for MUSE FSF Moffat model
                 fsf_wavelength=7000, # mean wavelegnth to be used for grouping (where white MUSE image is used)
                 cut = 0.005, # absolute cut value for convolution during grouping
                 min_width = 6, # minimal width of a group bounding box
                 min_sky_pixels = 20, # minimal number of sky pixels in a bounding box to estimate correctly the background
                 listFiltName=[DEFAULT_HSTFILTER606, # hst spectral filter response
                               DEFAULT_HSTFILTER775,
                               DEFAULT_HSTFILTER814,
                               DEFAULT_HSTFILTER850],
                 ):

        
        

        # params for grouping objects
        
        ## first for convolution kernel
        self.fsf_a_muse = fsf_a_muse
        self.fsf_b_muse = fsf_b_muse
        self.fsf_beta_muse = fsf_beta_muse
        
        self.fsf_wavelength = fsf_wavelength
        
        self.beta_hst = beta_hst
        self.fwhm_hst = fwhm_hst
        
        ## computed values
        self.fwhm_muse = fsf_a_muse + fsf_b_muse*fsf_wavelength 
        
        ### expressed in MUSE pixels
        self.alpha_hst = np.sqrt((self.fwhm_hst / 0.2 * 15)
                                ** 2 / (4 * (2**(1 / self.beta_hst) - 1)))
        
        ## then for cutting the convolution
        self.cut = cut
        
        ## minimal width of a group bounding box
        self.min_width = min_width
        self.min_sky_pixels = min_sky_pixels
        
        # Misc
        self.nBands = nBands
        self.listFiltName = listFiltName