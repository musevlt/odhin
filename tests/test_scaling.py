# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""


from mpdaf.obj import Image

imHST = Image("../data/hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")
imMUSE = Image("../data/IMAGE_UDF-10.fits")
segmap = Image("../data/segmentation_map_rafelski_2015.fits")
fwhm = 0.7230970999999999
betaFSF = 2.8
kernel = debl.listTransferKernel[0]
