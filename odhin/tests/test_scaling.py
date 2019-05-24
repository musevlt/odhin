# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""

import os

import numpy as np
import scipy.optimize as so
import scipy.signal as ssl
import scipy.sparse.linalg as sla
from scipy.interpolate import interp1d
from skimage.measure import label, regionprops

import astropy.io.fits as pyfits
import astropy.units as units
from deblend.deblend_utils import convertIntensityMap
from mpdaf.obj import Cube, Image, Spectrum

from .parameters import Params
from .regularization import medfilt, regulDeblendFunc

imHST = Image("../data/hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")
imMUSE = Image("../data/IMAGE_UDF-10.fits")
segmap = Image("../data/segmentation_map_rafelski_2015.fits")
fwhm = 0.7230970999999999
betaFSF = 2.8
kernel = debl.listTransferKernel[0]
