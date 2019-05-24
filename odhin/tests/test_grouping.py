from mpdaf.obj import Cube, Image

from .grouping import convertBboxToHR, doGrouping

imHST = Image("../data/hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")
imMUSE = Image("../data/IMAGE_UDF-10.fits")
segmap = Image("../data/segmentation_map_rafelski_2015.fits")
#fwhm = imHST.getFWHM()
#betaFSF = debl.betaFSF

#kernel = calcTransferKernel()


def test_convertBboxToHR():
    imHR = imHST
    imLR = imMUSE
    bbox = [10,50,10,50]
    imHR.wcs.sky2pix(imLR.wcs.pix2sky([10,10]))
    # assert convertBboxToHR(bbox,imHR,imLR) == 
