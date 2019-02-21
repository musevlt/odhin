# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""

from mpdaf.obj import Cube, Image, Spectrum
import scipy.signal as ssl
import scipy.sparse.linalg as sla
import numpy as np
from scipy.interpolate import interp1d
import scipy.optimize as so
import os
import astropy.units as units
import astropy.io.fits as pyfits
from .regularization import regulDeblendFunc, medfilt
from .parameters import Params
from skimage.measure import regionprops, label

from deblend.deblend_utils import convertIntensityMap
from mpdaf.obj import Cube, Image

imHST = Image("../data/hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")
imMUSE = Image("../data/IMAGE_UDF-10.fits")
segmap = Image("../data/segmentation_map_rafelski_2015.fits")
fwhm = 0.7230970999999999
betaFSF = 2.8
kernel = debl.listTransferKernel[0]
