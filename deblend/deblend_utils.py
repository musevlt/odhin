# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:53:50 2016

@author: raphael
"""
import numpy as np
import scipy
from scipy.interpolate import interp1d
import math


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
    shape=im.shape
    imfft=np.fft.rfft2(im)
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


def calcFSF(a,b,beta,listLambda,center=(12,12),shape=(25,25),dim='MUSE'):
    """
    Build list of FSF images (np arrays) from parameters a,b and beta. fwhm=a+b*lmbda, and beta is the Moffat parameter.
    """
    listFSF=[]
    for lmbda in listLambda:
        fwhm=a+b*lmbda
        alpha=np.sqrt(fwhm**2/(4*(2**(1/beta)-1)))
        listFSF.append(generateMoffatIm(center,shape,alpha,beta,dim=dim))
    return listFSF

def Moffat(r,alpha,beta):
    return (beta-1)/(math.pi*alpha**2)*(1+(r/alpha)**2)**(-beta)
    
def generateMoffatIm(center=(12,12),shape=(25,25),alpha=2,beta=2.5,a=0.,b=0.,dim='MUSE'):
    """
    by default alpha is supposed to be given in arsec, if not it is given in MUSE pixel.
    a,b allow to decenter slightly the Moffat image.
    """
    ind=np.indices(shape)
    r=np.sqrt(((ind[0]-center[0]+a)**2 + ((ind[1]-center[1]+b))**2))
    if dim=='MUSE':
        r=r*0.2
    elif dim=='HST':
        r=r*0.03
    res=Moffat(r,alpha,beta)
    res=res/np.sum(res)
    return res
    
def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)