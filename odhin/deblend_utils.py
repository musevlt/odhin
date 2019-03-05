# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""


import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
import astropy.units as u
import math
from scipy import ndimage
from photutils import create_matching_kernel, TopHatWindow, SegmentationImage


def block_sum(ar, fact):
    """
    Subsample a matrix *ar* by a integer factor *fact* using sums.
    """
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy // fact[1] * (X // fact[0]) + Y // fact[1]
    res = ndimage.sum(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx // fact[0], sy // fact[1])
    return res


def generatePSF_HST(alphaHST, betaHST, shape=(375, 375), shapeMUSE=(25, 25)):
    """
    Generate PSF HST at MUSE resolution with Moffat model.
    To increase precision (as this PSF is sharp) the construction is made on a larger array
    then subsampled.
    """
    PSF_HST_HR = generateMoffatIm(
        shape=shape,
        center=(shape[0] // 2, shape[1] // 2),
        alpha=alphaHST,
        beta=betaHST,
        dim=None,
    )
    factor = (shape[0] // shapeMUSE[0], shape[1] // shapeMUSE[1])
    PSF_HST = block_sum(PSF_HST_HR, (factor[0], factor[1]))
    PSF_HST[PSF_HST < 0.001] = 0
    PSF_HST = PSF_HST / np.sum(PSF_HST)
    return PSF_HST


def getMainSupport(u, alpha=0.999):
    """
    Get mask containing a fraction alpha of total map intensity.
    """
    mask = np.zeros_like(u).astype(bool)
    for i, row in enumerate(u):
        s = np.cumsum(np.sort(row)[::-1])
        keepNumber = np.sum(s < alpha * s[-1])
        mask[i, np.argsort(row, axis=None)[-keepNumber:]] = True
    return mask


def apply_resampling_window(im):
    """Multiply the FFT of an image with the blackman anti-aliasing window
    that would be used by default by the MPDAF Image.resample()
    function for the specified image grid.
    Parameters
    ----------
    im  : numpy.ndarray
       The resampled image.
    """

    # Create an array which, for each pixel in the FFT image, holds
    # the radial spatial-frequency of the pixel center, divided by the
    # Nyquist folding frequency (ie. 0.5 cycles/image_pixel).
    shape = im.shape
    imfft = np.fft.rfft2(im)
    f = np.sqrt(
        (np.fft.rfftfreq(shape[1]) / 0.5)
        ** 2
        + (np.fft.fftfreq(shape[0]) / 0.5)[np.newaxis, :].T
        ** 2
    )

    # Scale the FFT by the window function, calculating the blackman
    # window function as a function of frequency divided by its cutoff
    # frequency.

    imfft *= np.where(
        f <= 1.0, 0.42 + 0.5 * np.cos(np.pi * f) + 0.08 * np.cos(2 * np.pi * f), 0.0
    )
    return np.fft.irfft2(imfft, shape)


def convertFilt(filt, wave=None, x=None):
    """
    Resample response of HST filter on the spectral grid of MUSE

    filt: 2d array (wavelength,amplitude) of response filter
    wave: mpdaf wave object
    """
    if wave is not None:
        x = wave.coord()
    f = interp1d(filt[:, 0], filt[:, 1], fill_value=0, bounds_error=False)
    return np.array(f(x))


def calcFSF(a, b, beta, listLambda, center=(12, 12), shape=(25, 25), dim="MUSE"):
    """
    Build list of FSF images (np arrays) from parameters a,b and beta.
    fwhm=a+b*lmbda, and beta is the Moffat parameter.
    """
    listFSF = []
    for lmbda in listLambda:
        fwhm = a + b * lmbda

        alpha = np.sqrt(fwhm ** 2 / (4 * (2 ** (1 / beta) - 1)))
        listFSF.append(generateMoffatIm(center, shape, alpha, beta, dim=dim))
    return listFSF


def Moffat(r, alpha, beta):
    """
    Compute Moffat values for array of distances *r* and Moffat parameters *alpha* and *beta*
    """
    return (beta - 1) / (math.pi * alpha ** 2) * (1 + (r / alpha) ** 2) ** (-beta)


def generateMoffatIm(center=(12, 12), shape=(25, 25), alpha=2, beta=2.5,
                     dx=0., dy=0., dim="MUSE"):
    """
    Generate Moffat FSF image
    By default alpha is supposed to be given in arsec, if not it is given in
    MUSE pixel. a,b allow to decenter slightly the Moffat image.
    """
    ind = np.indices(shape)
    r = np.sqrt(((ind[0] - center[0] + dx) ** 2 + ((ind[1] - center[1] + dy)) ** 2))
    if dim == "MUSE":
        r = r * 0.2
    elif dim == "HST":
        r = r * 0.03
    res = Moffat(r, alpha, beta)
    res = res / np.sum(res)
    return res


def normalize(a, axis=-1, order=2, returnCoeff=False):
    """
    Normalize array along axis
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    if returnCoeff is True:
        return a / np.expand_dims(l2, axis), np.expand_dims(l2, axis)
    else:
        return a / np.expand_dims(l2, axis)


def convertIntensityMap(intensityMap, muse, hst, fwhm, beta, imPSFMUSE):
    """
    Parameters
    ----------
    intensityMap : `ndarray`
        The matrix of intensity maps (one row per object) at HST resolution
    hst : `mpdaf.obj.Image`
       The HST image to be resampled.
    muse : `mpdaf.obj.Image` of `mpdaf.obj.Cube`
       The MUSE image or cube to use as the template for the HST image.
    fwhm : `float`
        fwhm of MUSE FSF
    beta : `float`
        Moffat beta parameter of MUSE FSF

    Returns
    -------
    intensityMapMuse : `ndarray`
       The matrix of intensity maps (one row per object) at MUSE resolution
       (and convolved by MUSE FSF or HST-MUSE transfert function)
    """
    intensityMapMuse = np.zeros((intensityMap.shape[0], muse.data.size))
    hst_ref = hst.copy()

    imPSFMUSE = imPSFMUSE / np.sum(imPSFMUSE)
    for i in range(intensityMap.shape[0]):
        hst_ref.data = intensityMap[i].reshape(hst_ref.shape)

        hst_ref.data = fftconvolve(hst_ref.data, imPSFMUSE, mode="same")
        hst_ref_muse = regrid_hst_like_muse(hst_ref, muse, inplace=False,
                                            antialias=False)
        hst_ref_muse = rescale_hst_like_muse(hst_ref_muse, muse, inplace=True)
        hst_ref_muse.mask[:] = False

        intensityMapMuse[i] = hst_ref_muse.data.flatten()
    return intensityMapMuse


# functions taken from muse_analysis and slighlty modified


def regrid_hst_like_muse(hst, muse, inplace=True, antialias=False):
    """
    Resample an HST image onto the spatial coordinate grid of a given
    MUSE image or MUSE cube.

    Parameters
    ----------
    hst : `mpdaf.obj.Image`
       The HST image to be resampled.
    muse : `mpdaf.obj.Image` of `mpdaf.obj.Cube`
       The MUSE image or cube to use as the template for the HST image.
    inplace : bool
       (This defaults to True, because HST images tend to be large)
       If True, replace the contents of the input HST image object
       with the resampled image.
       If False, return a new Image object that contains the resampled
       image.

    Returns
    -------
    out : `mpdaf.obj.Image`
       The resampled HST image.

    """

    # Operate on a copy of the input image?

    if not inplace:
        hst = hst.copy()

    # If a MUSE cube was provided, extract a single-plane image to use
    # as the template.

    if muse.ndim > 2:
        muse = muse[0, :, :]

    # Mask the zero-valued blank margins of the HST image.

    # "
    # Removed here because we work on intensity maps with lot of zeros
    # np.ma.masked_inside(hst.data, -1e-10, 1e-10, copy=False)
    #########

    # Resample the HST image onto the same coordinate grid as the MUSE
    # image.
    return hst.align_with_image(
        muse, cutoff=0.0, flux=True, inplace=False, antialias=antialias
    )


class HstFilterInfo(object):
    """An object that contains the filter characteristics of an HST
    image.

    Parameters
    ----------
    hst : `mpdaf.obj.Image` or str
       The HST image to be characterized, or the name of an HST filter,
       such as "F606W.

    Attributes
    ----------
    filter_name : str
       The name of the filter.
    abmag_zero : float
       The AB magnitude that corresponds to the zero electrons/s\n
       from the camera (see ``https://archive.stsci.edu/prepds/xdf``).
    photflam : float
       The calibration factor to convert pixel values in\n
       electrons/s to erg cm-2 s-1 Angstrom-1.\n
       Calculated using::

         photflam = 10**(-abmag_zero/2.5 - 2*log10(photplam) - 0.9632)

       which is a rearranged form of the following equation from\n
       ``http://www.stsci.edu/hst/acs/analysis/zeropoints``::

         abmag_zero = -2.5 log10(photflam)-5 log10(photplam)-2.408

    photplam : float
       The pivot wavelength of the filter (Angstrom)\n
       (from ``http://www.stsci.edu/hst/acs/analysis/bandwidths``)
    photbw : float
       The effective bandwidth of the filter (Angstrom)\n
       (from ``http://www.stsci.edu/hst/acs/analysis/bandwidths``)

    """

    # Create a class-level dictionary of HST filter characteristics.
    # See the documentation of the attributes for the source of these
    # numbers.

    _filters = {
        "F606W": {
            "abmag_zero": 26.51,
            "photplam": 5921.1,
            "photflam": 7.73e-20,
            "photbw": 672.3,
        },
        "F775W": {
            "abmag_zero": 25.69,
            "photplam": 7692.4,
            "photflam": 9.74e-20,
            "photbw": 434.4,
        },
        "F814W": {
            "abmag_zero": 25.94,
            "photplam": 8057.0,
            "photflam": 7.05e-20,
            "photbw": 652.0,
        },
        "F850LP": {
            "abmag_zero": 24.87,
            "photplam": 9033.1,
            "photflam": 1.50e-19,
            "photbw": 525.7,
        },
    }

    def __init__(self, hst):

        # If an image has been given, get the name of the HST filter
        # from the FITS header of the HST image, and convert the name
        # to upper case. Otherwise get it from the specified filter
        # name.

        if isinstance(hst, str):
            self.filter_name = hst.upper()
        elif "FILTER" in hst.primary_header:
            self.filter_name = hst.primary_header["FILTER"].upper()
        elif (
            "FILTER1" in hst.primary_header
            and hst.primary_header["FILTER1"] != "CLEAR1L"
        ):
            self.filter_name = hst.primary_header["FILTER1"].upper()
        elif (
            "FILTER2" in hst.primary_header
            and hst.primary_header["FILTER2"] != "CLEAR2L"
        ):
            self.filter_name = hst.primary_header["FILTER2"].upper()

        # Get the dictionary of the characteristics of the filter.

        if self.filter_name in self.__class__._filters:
            info = self.__class__._filters[self.filter_name]

        # Record the characteristics of the filer.

        self.abmag_zero = info["abmag_zero"]
        self.photflam = info["photflam"]
        self.photplam = info["photplam"]
        self.photbw = info["photbw"]


def rescale_hst_like_muse(hst, muse, inplace=True):
    """Rescale an HST image to have the same flux units as a given MUSE image.

    Parameters
    ----------
    hst : `mpdaf.obj.Image`
       The HST image to be resampled.
    muse : `mpdaf.obj.Image` or `mpdaf.obj.Cube`
       A MUSE image or cube with the target flux units.
    inplace : bool
       (This defaults to True, because HST images tend to be large)
       If True, replace the contents of the input HST image object
       with the rescaled image.
       If False, return a new Image object that contains the rescaled
       image.

    Returns
    -------
    out : `mpdaf.obj.Image`
       The rescaled HST image.

    """

    # Operate on a copy of the input image?

    if not inplace:
        hst = hst.copy()

    # Get the characteristics of the HST filter.

    filt = HstFilterInfo(hst)

    # Calculate the calibration factor needed to convert from
    # electrons/s in the HST image to MUSE flux-density units.

    cal = filt.photflam * u.Unit("erg cm-2 s-1 Angstrom-1").to(muse.unit)

    # Rescale the HST image to have the same units as the MUSE image.

    hst.data *= cal
    if hst.var is not None:
        hst.var *= cal ** 2
    hst.unit = muse.unit

    return hst


def getBlurKernel(imHR, imLR, sizeKer, returnImBlurred=False, cut=0.00001):
    """
    Compute convolution kernel between two images (typically one from HST
    and one from MUSE).

    Parameters:
    -----------
    imLR,imHR - blurred and sharp images respectively
    szKer - 2 element vector specifying the size of the required kernel

    Returns:
    -------
    kernel - the recovered kernel,
    imLRsynth - the sharp image convolved with the recovered kernel

    """
    window = TopHatWindow(0.35)
    kernel = create_matching_kernel(imHR, imLR, window=window)
    imLRsynth = fftconvolve(imHR, kernel, "same")
    residuals = np.sum((imLR - imLRsynth) ** 2)
    if residuals > 0.1 * np.sum(imLR ** 2):
        print("Warning : residuals are strong, maybe the linear inversion "
              "is not constrained enough.")
        print(residuals, np.sum(imLR ** 2))

    kernel[kernel < cut] = 0.
    if returnImBlurred is True:
        return kernel, imLRsynth

    return kernel


def _getLabel(segmap=None):
    """
    Create a new segmap with contiguous indices
    """
    label_image = np.zeros(segmap.shape, dtype='int')
    i = 0
    for k in sorted(set(segmap.flatten())):
        label_image[segmap == k] = i
        i = i + 1

    return label_image


def calcMainKernelTransfert(params, imHST):
    """
    Build transfert kernel between an HR kernel and LR kernel (defined in paramaters)
    """
    # get parameters
    fsf_beta_muse = params.fsf_beta_muse
    fwhm_muse = params.fwhm_muse
    fwhm_hst = params.fwhm_hst
    beta_hst = params.beta_hst

    dy, dx = imHST.get_step(unit=u.arcsec)

    shape = np.array(imHST.shape)

    # get odd shape
    shape_1 = shape // 2 * 2 + 1
    center = shape_1 // 2

    # Build "distances to center" matrix.
    ind = np.indices(shape_1)
    rsq = ((ind[0] - center[0]) * dx)**2 + (((ind[1] - center[1])) * dy)**2

    # Build HST FSF
    asq_hst = fwhm_hst**2 / 4.0 / (2.0**(1.0 / beta_hst) - 1.0)
    psf_hst = 1.0 / (1.0 + rsq / asq_hst)**beta_hst
    psf_hst = psf_hst / np.sum(psf_hst)

    # Build MUSE FSF
    asq = fwhm_muse**2 / 4.0 / (2.0**(1.0 / fsf_beta_muse) - 1.0)
    im_muse = 1.0 / (1.0 + rsq / asq)**fsf_beta_muse
    im_muse = im_muse / np.sum(im_muse)

    return getBlurKernel(imHR=psf_hst, imLR=im_muse, sizeKer=(21, 21))


def createIntensityMap(imHR, segmap, imLR, kernel_transfert, params=None):
    """
    Create intensity maps from HST images and segmentation map.

    Parameters
    ----------
    imHR: the reference high resolution image
    segmap : the segmantation map at high resolution
    imLR : image at the targeted low resolution
    kernel_transfert : convolution kernel from imHR to imLR
    params : additionnal parameters

    """
    if params is None:
        # FIXME: params = Params()
        raise ValueError('this case is not implemented!')

    # create image of label reindexed squentially from 0 to nSources
    segmapIm = SegmentationImage(segmap.data)
    try:
        # new method in photutils v0.5
        relabel_func = segmapIm.relabel_consecutive
    except AttributeError:
        # old method
        relabel_func = segmapIm.relabel_sequential

    relabel_func()
    labelHR = segmapIm.data

    intensityMapHR = np.zeros(imHR.shape)
    intensityMapHR[labelHR > 0] = np.maximum(
        imHR.data[segmap.data > 0], 10**(-9))  # avoid negative abundances

    intensityMapLRConvol = convertIntensityMap(
        intensityMapHR[None, :], imLR, imHR, params.fwhm_muse,
        params.fsf_beta_muse, kernel_transfert
    ).reshape(imLR.shape)

    return intensityMapLRConvol


def modifSegmap(segmap, cat):
    """
    Avoid discrepancy between the catalog and the segmentation map
    (there are some segmap objects missing from the Rafelski 2015 catalog).
    """
    segmap2 = segmap.copy()
    keys_cat = cat['ID']
    keys = np.unique(segmap.data.data)
    for k in keys:
        if k not in keys_cat:
            # to log somewhere
            # print(k)
            ####
            segmap2.data[segmap.data == k] = 0
    return segmap2


# def get_cat_from_segmap(segmap):
#     """
#     Get IDs and center directly from a segmap (thus removing the need for an external catalag)
#     """
#     # To be build
#     pass


def extractHST(imHST, imMUSE, rot=True):
    """
    Extract HST image corresponding to MUSE image.
    """
    centerpix = (imMUSE.shape[0] // 2, imMUSE.shape[1] // 2)
    center = imMUSE.wcs.pix2sky(centerpix)[0]
    step = imMUSE.get_step(u.arcsec)
    stepHST = imHST.get_step(u.arcsec)
    size = (step[1] * imMUSE.shape[1], step[0] *
            imMUSE.shape[0])  # width then height
    # size[0] is width so second dimension of the array
    sizeHST = (size[0] / stepHST[1], size[1] / stepHST[0])
    ext_size = (size[1] + 10, size[0] + 10)
    ext_sizeHST = (ext_size[0] / stepHST[1], ext_size[1] / stepHST[0])
    # centerHST = imHST.wcs.sky2pix(center, nearest=True)[0]

    if rot is True:
        imHST_tmp = imHST.subimage(center, ext_sizeHST, unit_size=None)
        imHST_tmp = imHST_tmp.rotate(imMUSE.get_rot())
        imHST2 = imHST_tmp.subimage(center, sizeHST, unit_size=None)
    else:
        imHST2 = imHST.subimage(center, sizeHST, unit_size=None)
    return imHST2
