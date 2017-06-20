# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:53:50 2016

@author: raphael.bacher@gipsa-lab.fr
"""
import numpy as np
import scipy.signal as ssl
from numpy import ma
from scipy.ndimage.filters import convolve
import scipy.optimize as so
from scipy.interpolate import interp1d
import astropy.units as units
import astropy.io.fits as pyfits
import math
from muse_analysis.imphot import fit_image_photometry, rescale_hst_like_muse
from scipy import ndimage

import matplotlib.pyplot as plt

def block_sum(ar, fact):
    """
    Subsample a matrix *ar* by a integer factor *fact* using sums.
    """
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact[1] * (X/fact[0]) + Y/fact[1]
    res = ndimage.sum(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact[0], sy/fact[1])
    return res


def generatePSF_HST(alphaHST, betaHST, shape=(375, 375), shapeMUSE=(25, 25)):
    """Generate PSF HST at MUSE resolution with Moffat model.
    To increase precision (as this PSF is sharp) the construction is made on a larger array
    then subsampled.
    """
    PSF_HST_HR = generateMoffatIm(shape=shape, center=(shape[0]/2, shape[1]/2),
                                      alpha=alphaHST, beta=betaHST, dim=None)
    factor = (shape[0]/shapeMUSE[0], shape[1]/shapeMUSE[1])
    PSF_HST = block_sum(PSF_HST_HR, (factor[0], factor[1]))
    PSF_HST[PSF_HST < 0.001] = 0
    PSF_HST = PSF_HST/np.sum(PSF_HST)
    return PSF_HST

def getMainSupport(u, alpha=0.999):
    """
    Get mask containing fraction alpha of total map intensity.
    """
    mask = np.zeros_like(u).astype(bool)
    for i,row in enumerate(u):
        s=np.cumsum(np.sort(row)[::-1])
        keepNumber=np.sum(s<alpha*s[-1])
        mask[i,np.argsort(row,axis=None)[-keepNumber:]]=True
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
    f = np.sqrt((np.fft.rfftfreq(shape[1]) / 0.5)**2 +
                (np.fft.fftfreq(shape[0]) / 0.5)[np.newaxis,:].T**2)

    # Scale the FFT by the window function, calculating the blackman
    # window function as a function of frequency divided by its cutoff
    # frequency.

    imfft *= np.where(f <= 1.0, 0.42 + 0.5 * np.cos(np.pi * f) + 0.08 * np.cos(2*np.pi * f), 0.0)
    return np.fft.irfft2(imfft,shape)

def convertFilt(filt,wave=None,x=None):
    """
    Resample response of HST filter on the spectral grid of MUSE

    filt: 2d array (wavelength,amplitude) of response filter
    wave: mpdaf wave object
    """
    if wave is not None:
        x = wave.coord()
    f=interp1d(filt[:,0],filt[:,1],fill_value=0,bounds_error=False)
    return np.array(f(x))


def calcFSF(a, b, beta, listLambda, center=(12, 12), shape=(25, 25), dim='MUSE'):
    """
    Build list of FSF images (np arrays) from parameters a,b and beta.
    fwhm=a+b*lmbda, and beta is the Moffat parameter.
    """
    listFSF = []
    for lmbda in listLambda:
        fwhm = a+b*lmbda

        alpha = np.sqrt(fwhm**2/(4*(2**(1/beta)-1)))
        listFSF.append(generateMoffatIm(center, shape, alpha, beta, dim=dim))
    return listFSF


def Moffat(r, alpha, beta):
    """
    Compute Moffat values for array of distances *r* and Moffat parameters *alpha* and *beta*
    """
    return (beta-1)/(math.pi*alpha**2)*(1+(r/alpha)**2)**(-beta)

def generateMoffatIm(center=(12,12),shape=(25,25),alpha=2,beta=2.5,dx=0.,dy=0.,dim='MUSE'):
    """
    By default alpha is supposed to be given in arsec, if not it is given in MUSE pixel.
    a,b allow to decenter slightly the Moffat image.
    """
    ind = np.indices(shape)
    r = np.sqrt(((ind[0]-center[0]+dx)**2 + ((ind[1]-center[1]+dy))**2))
    if dim == 'MUSE':
        r = r*0.2
    elif dim == 'HST':
        r = r*0.03
    res = Moffat(r, alpha, beta)
    res = res/np.sum(res)
    return res


def normalize(a, axis=-1, order=2, returnCoeff=False):
    """
    normalize array along axis
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    if returnCoeff is True:
        return a / np.expand_dims(l2, axis), np.expand_dims(l2, axis)
    else:
        return a / np.expand_dims(l2, axis)


def getSpatialShift(imMUSE_0, imHST_0, beta_muse, fwhm_muse,):
    """
    Parameters
    ----------

    imMUSE_0 : `mpdaf.obj.Image`
        The MUSE image or cube against which the HST image must be aligned.
    imHST_0 : `mpdaf.obj.Image`
          The HST image to be shifted.
    fwhm_muse : `float`
        fwhm of MUSE FSF
    beta_muse : `float`
        Moffat beta parameter of MUSE FSF



    Returns
    -------
    shift : `(float,float)`
       Shift in (x,y) to apply to HST image

    """
    imHST_1 = imHST_0.copy()
    regrid_hst_like_muse(imHST_1, imMUSE_0, inplace=True)
    rescale_hst_like_muse(imHST_1, imMUSE_0, inplace=True)
    res = fit_image_photometry(imHST_1, imMUSE_0, fix_beta=beta_muse,
                               fix_fwhm=fwhm_muse, fix_dx=None, fix_dy=None, display=False)[0]

    return (res.dx.value, res.dy.value)



def convertIntensityMap(intensityMap,muse,hst,fwhm,beta,shift,antialias=False,psf_hst=True):
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
    shift : `(float,float)`
        Shifts to apply to HST image
    antialias : `bool`
        Use antialising filter or not.
        Default to False (because a broad convolution is applied afterwards)
    psf_hst : `bool`
        Use HST-MUSE transfert function instead of MUSE FSF. Default to True


    Returns
    -------
    intensityMapMuse : `ndarray`
       The matrix of intensity maps (one row per object) at MUSE resolution
       (and convolved by MUSE FSF or HST-MUSE transfert function)
    """
    intensityMapMuse = np.zeros((intensityMap.shape[0], muse.data.size))
    hst_ref = hst.copy()

    for i in xrange(intensityMap.shape[0]):
        hst_ref.data = intensityMap[i].reshape(hst_ref.shape)
        hst_ref_muse = regrid_hst_like_muse(hst_ref, muse, inplace=False,
                                            antialias=antialias)

        hst_ref_muse = rescale_hst_like_muse(hst_ref_muse, muse, inplace=False)
        hst_ref_muse.mask[:] = False
        im = getHSTIm(hst_ref_muse, muse, fwhm, beta, ddx=shift[0],
                    ddy=shift[1], taper=0, psf_hst=psf_hst)
        intensityMapMuse[i] = im.flatten()
    return intensityMapMuse

def convertIntensityMapV2(intensityMap,muse,hst,fwhm,beta,shift,antialias=False,psf_hst=True):
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
    shift : `(float,float)`
        Shifts to apply to HST image
    antialias : `bool`
        Use antialising filter or not.
        Default to False (because a broad convolution is applied afterwards)
    psf_hst : `bool`
        Use HST-MUSE transfert function instead of MUSE FSF. Default to True


    Returns
    -------
    intensityMapMuse : `ndarray`
       The matrix of intensity maps (one row per object) at MUSE resolution
       (and convolved by MUSE FSF or HST-MUSE transfert function)
    """
    intensityMapMuse = np.zeros((intensityMap.shape[0], muse.data.size))
    hst_ref = hst.copy()


    #tmp_dir='./tmp/'
    #imPSFMUSE = pyfits.open(tmp_dir+'kernel_%s.fits'%fwhm)[0].data
    #imPSFMUSE=imPSFMUSE/np.sum(imPSFMUSE)
    #alpha= fwhm / 2.0 / np.sqrt((2.0**(1.0 / beta) - 1.0))
    #print alpha,beta
    #imPSFMUSE2 = generateMoffatIm(center=(50,50),shape=(101,101),alpha=alpha,beta=beta,dim='HST')
    #return imPSFMUSE,imPSFMUSE2

    if psf_hst == True: #get HST-MUSE transfert function
        tmp_dir='./tmp/'
        imPSFMUSE = pyfits.open(tmp_dir+'kernel_%s.fits'%fwhm)[0].data
        imPSFMUSE=imPSFMUSE/np.sum(imPSFMUSE)
    else:
        alpha= fwhm / 2.0 / np.sqrt((2.0**(1.0 / beta) - 1.0))
        imPSFMUSE = generateMoffatIm(center=(50,50),shape=(101,101),alpha=alpha,beta=beta,dim='HST')
    for i in xrange(intensityMap.shape[0]):
        hst_ref.data = intensityMap[i].reshape(hst_ref.shape)


        hst_ref.data = ssl.fftconvolve(hst_ref.data,imPSFMUSE,mode='same')
        hst_ref_muse = regrid_hst_like_muse(hst_ref, muse, inplace=False,
                                            antialias=antialias)

        hst_ref_muse = rescale_hst_like_muse(hst_ref_muse, muse, inplace=False)
        hst_ref_muse.mask[:] = False

        intensityMapMuse[i] = hst_ref_muse.data.flatten()
    return intensityMapMuse



### functions taken from muse_analysis and slighlty modified


def regrid_hst_like_muse(hst, muse, inplace=True, antialias=True):
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
        muse = muse[0,:,:]

    # Mask the zero-valued blank margins of the HST image.
    #### Removed here because we work on intensity maps with lot of zeros
    #np.ma.masked_inside(hst.data, -1e-10, 1e-10, copy=False)


    # Resample the HST image onto the same coordinate grid as the MUSE
    # image.

    return hst.align_with_image(muse, cutoff=0.0, flux=True, inplace=True,antialias=antialias)

def getHSTIm(hst,muse,fwhm=0.67, beta=2.8,ddx=0.0,ddy=0.0,scale=1,bg=0,
                         margin=2.0,taper=9,psf_hst=False):

    dy, dx = muse.get_step(unit=units.arcsec)

    hst_data = hst._data.copy()
    if hst.mask is not ma.nomask:
        hst_mask = hst.mask.copy()
    else:
        hst_mask = ~np.isfinite(hst_data)

    # Convert the initial offsets to pixels, and round them to the nearest
    # integer number of pixels.
    init_dx = 0.
    init_dy = 0.
    pxoff = int(np.floor(init_dx / dx + 0.5))
    pyoff = int(np.floor(init_dy / dy + 0.5))

    # Compute the angular size of the above pixel offsets.

    xshift = pxoff * dx
    yshift = pyoff * dy
    subtracted = np.ma.median(muse.data)
    mask = hst_mask

    # Copy both image arrays into masked array containers that use the
    # above mask.

    hdata = ma.array(data=hst_data, mask=mask, copy=False)

    hdata = ma.filled(hdata, 0.0)

    # When a position offset in the image plane is performed by
    # applying a linearly increasing phase offset in the Fourier
    # domain of an FFT, the result is that features in the image
    # that get shifted off the right edge of the image reappear at the
    # left edge of the image, and vice versa. By appending a margin of
    # zeros to the right edge, and later discarding this margin, we
    # ensure that features that are shifted off the left or right
    # edges of the image end up in this discarded area, rather than
    # appearing at the opposite edge of the visible image. Appending a
    # similar margin to the Y axis has the same effect for vertical
    # shifts.

    shape = np.asarray(muse.shape) + np.ceil(np.abs(
        margin / muse.get_step(unit=units.arcsec) ) ).astype(int)

    # Round the image dimensions up to integer powers of two, to
    # ensure that an efficient FFT implementation is used.

    shape = (2**np.ceil(np.log(shape)/np.log(2.0))).astype(int)

    # Extract the dimensions of the expanded Y and X axes.

    ny,nx = shape

    # Compute the slice needed to extract the original area from
    # expanded arrays of the above shape.

    sky_slice = [slice(0,muse.shape[0]), slice(0,muse.shape[1])]


    # Zero-pad the HST image array to have the new shape.

    tmp = np.zeros(shape)
    tmp[sky_slice] = hdata
    hdata = tmp

    # Pad the mask array to have the same dimensions, with padded
    # elements being masked.

    tmp = np.ones(shape, dtype=bool)
    tmp[sky_slice] = mask
    mask = tmp

    # Compute a multiplicative pixel scaling image that inverts the
    # mask to be 1 for pixels that are to be kept, and 0 for pixels
    # that are to be removed, then smoothly bevel the edges of the
    # areas of ones to reduce the effects of sharp edges on the fitted
    # position offsets.

    if taper >= 2:
        weight_img = _bevel_mask(~mask, 2*(taper//2)+1)
    else:
        weight_img = (~mask).astype(float)

    # Also obtain the FFT of the mask for fitting the background flux
    # offset in the Fourier plane.

    weight_fft = np.fft.rfft2(weight_img)


    hdata *= weight_img

    # Obtain the Fourier transforms of the MUSE and HST images. The
    # Fourier transform is hermitian, because the images have no
    # imaginary parts, so we only calculate half of the Fourier
    # transform plane by using rfft2 instead of fft2().

    hfft = np.fft.rfft2(hdata)


    # Compute the spatial frequencies along the X and Y axes
    # of the above power spectra.

    fx = np.fft.rfftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dy)

    # Get 2D grids of the x,y spatial-frequency coordinates of each pixel
    # in the power spectra.

    fx2d,fy2d = np.meshgrid(fx,fy)

    # Calculate the frequency interval of the FFTs along the
    # X and Y axes.

    dfx = 1.0 / (nx * dx)
    dfy = 1.0 / (ny * dy)

    # Get a 2D array of the radius of each image pixel center relative
    # to pixel [0,0] (the spatial origin of the FFT algorithm).

    rsq = np.fft.fftfreq(nx, dfx)**2 + \
          np.fft.fftfreq(ny, dfy)[np.newaxis,:].T**2


    hfft = _xy_moffat_model_fn(fx2d, fy2d, rsq, hfft, weight_fft, subtracted,
                               xshift, yshift,
                               ddx,
                               ddy,
                               bg,
                               scale,
                               fwhm,
                               beta,psf_hst).view(dtype=complex)

    # Invert the best-fit FFTs to obtain the best-fit MUSE and HST images.


    hst_im = np.fft.irfft2(hfft)[sky_slice]
    return hst_im

def _xy_moffat_model_fn(fx, fy, rsq, hstfft, wfft, subtracted, xshift, yshift,
                        dx, dy, bg, scale, fwhm, beta, psf_hst=False):

    """This function is designed to be passed to lmfit to fit the FFT of
    an HST image to the FFT of a MUSE image on the same coordinate
    grid.

    It takes the FFT of an HST image and changes it as follows:

    1. It multiplies the FFT by the scale parameter, to change its flux
       calibration.
    2. It multiplies the FFT by the FFT of a circularly symmetric moffat
       function of the specified fwhm and beta parameter. This is
       equivalent to convolving the original HST image with a
       Moffat PSF with the specified characteristics.
    3. It multiplies the FFT by phase gradients that are equivalent to
       shifting the original HST image by dx and dy arcseconds, to
       correct any position offset.
    4. It adds bg*wfft to change the zero-offset of the original HST
       image by bg.

    Parameters:
    -----------
    fx : numpy.ndarray
       The X-axis spatial-frequency coordinate (cycles/arcsec) of each
       pixel in the HST FFT.
    fy : numpy.ndarray
       The Y-axis spatial-frequency coordinate (cycles/arcsec) of each
       pixel in the HST FFT.
    rsq : numpy.ndarray
       The radius-squared of each pixel of an image of the same
       dimensions as the image that was forier transformed to obtain
       hstfft, relative to the center of pixel [0,0], in the same
       spatial units as fwhm. Note that the x offsets of each pixel
       along the x axis, relative to pixel index 0, can be calculated
       using np.fft.fftfreq(nx,1.0/(nx*dx)), where nx is the number of
       pixels along the x axis, and dx is the pixel width in
       arcseconds.  Similarly for the y offsets of each pixel.
    hstfft : numpy.ndarray
       The FFT of the HST image.
    wfft : numpy.ndarray
       The FFT of the weighting array that has been applied to the
       pixels of the image.
    subtracted : float
       A constant background value that has was already subtracted from
       the MUSE image before it was FFT'd.
    xshift : int
       The distance that the HST image was shifted along the x-axis, before
       the fit (arcsec).
    yshift : int
       The distance that the HST image was shifted along the y-axis, before
       the fit (arcsec).
    dx: float
       The position offset along the X axis of the image.
    dy: float
       The position offset along the Y axis of the image.
    bg : float
       The background offset to add to the HST image (in the Fourier plane
       this is added to the pixel at the origin, which contains the sum
       of the flux in the image plane.)
    scale : float
       The scale factor to multiply the HST FFT pixels by.
    fwhm : float
       The full-width at half maximum of the Moffat function in the image
       plane that will be convolved with the HST image.
    beta : float
       The term due to scattering in the atmosphere that widens
       the wings of the PSF compared to a Gaussian. A common
       choice for this valus is 2.0.
    psf_hst : bool
        If True, the psf used is not the whole MUSE PSF but the transfert
        function between MUSE and HST

    Returns:
    --------
    out : numpy.ndarray
       The modified value of the HST FFT to be compared to the
       MUSE FFT. The fitter can't cope with complex elements,
       so this function returns an array of two float elements (real, imag)
       for each element of the FFT. To convert this back to a complex array,
       note that you can follow it with .view(dtype=complex)

    """

    # A center-normalized Moffat function is defined as follows:
    #
    #   y(x,y) = 1 / (1 + (x**2 + y**2) / a**2)**beta
    #
    # Calculate a**2 from the FWHM and beta.

    asq = fwhm**2 / 4.0 / (2.0**(1.0 / beta) - 1.0)

    # Compute an image of a Moffat function centered at pixel 0,0.
    im = 1.0 / (1.0 + rsq / asq)**beta

    if psf_hst is True: # get HST-MUSE transfert function corresponding to MUSE fwhm
        tmp_dir='./tmp/'
        im=pyfits.open(tmp_dir+'kernel_%s.fits'%fwhm)[0].data

        #center transfert function at pixel 0,0
        im=np.roll(im,(im.shape[1]/2+1,im.shape[1]/2+1),axis=(0,1))

    # Obtain the discrete Fourier Transform of the Moffat function.
    # The function is even, meaning that its Fourier transform is
    # entirely real, so also discard the imaginary parts.

    moffat_ft = np.real(np.fft.rfft2(im))

    # Normalize it to have unit volume in the image plane.

    moffat_ft /= moffat_ft[0,0]

    # If the HST FFT array has been flattened to make it compatible with
    # lmfit, do the same to moffat_ft.

    if hstfft.ndim == 1:
        moffat_ft = moffat_ft.ravel()

    # Precompute the image shifting coefficients.

    argx = -2.0j * np.pi * (dx - xshift)
    argy = -2.0j * np.pi * (dy - yshift)

    # Create the model to compare with the MUSE FFT. This is the HST FFT
    # scaled by the fitted scaling factor, smoothed to a lower resolution
    # by the above 2D Moffat function, and shifted by dx and dy.

    #model = (bg - subtracted) * wfft + hstfft * scale * moffat_ft * np.exp(argx*fx + argy*fy)
    model =  hstfft * scale * moffat_ft * np.exp(argx*fx + argy*fy)

    # The model-fitting function can't handle complex numbers, so
    # return the complex FFT model as an array of alternating real and
    # imaginary floats.

    return model.view(dtype=float)

def _bevel_mask(mask, width):
    """Return a floating point image that is a smoothed version of a
    boolean mask array. It is important that pixels that are zero in
    the input masked array remain zero in the smoothed version. If we
    just convolved the mask with a smoothing kernel, this would smear
    non-zero values into the region of the zeros, so instead we first
    erode the edges of the masked regions using a square kernel, then
    apply a square smoothing kernel of the same size to smooth down to
    the edges of the original masked areas.

    Parameters
    ----------
    mask   :   numpy.ndarray
       A 2D array of bool elements.
    width  :   int
       The width of the bevel in pixels.

    Returns
    -------
    out    :   numpy.ndarray
       A 2D array of float elements.

    """

    # First shave off 'width' pixels from all edges of the mask,
    # and return the result as a floating point array.


    im = ndimage.morphology.binary_erosion(mask, structure=np.ones((width, width))).astype(float)

    # Compute a [width,width] smoothing convolution mask.

    w = np.blackman(width+2)[1:width+1]
    m = w * w[np.newaxis,:].T
    m /= m.sum()

    # Smooth the eroded edges of the mask.

    return convolve(im, m)





