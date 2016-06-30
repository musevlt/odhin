# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:53:50 2016

@author: raphael
"""
import numpy as np
import scipy
from scipy.interpolate import interp1d
import math


def convertFilt(filt,cube=None,x=None):
    """
    Resample response of HST filter on the spectral grid of MUSE
    
    filt: 2d array (wavelength,amplitude) of response filter
    cube: mpdaf cube, provides the wave sampling grid
    """
    if cube is not None:
        x = cube.wave.coord()
    f=interp1d(filt[:,0],filt[:,1],fill_value=0,bounds_error=False)
    return np.array(f(x))


def calcFSF(a,b,beta,listLambda):
    """
    Build list of FSF images (np arrays) from parameters a,b and beta. fwhm=a+b*lmbda, and beta is the Moffat parameter.
    """
    listFSF=[]
    for lmbda in listLambda:
        fwhm=a+b*lmbda
        alpha=np.sqrt(fwhm**2/(4*(2**(1/beta)-1)))
        listFSF.append(generateMoffatIm((15,15),(31,31),alpha,beta))
    return listFSF

def Moffat(r,alpha,beta):
    return (beta-1)/(math.pi*alpha**2)*(1+(r/alpha)**2)**(-beta)
    
def generateMoffatIm(center=(12,12),shape=(25,25),alpha=2,beta=2.5,a=0.,b=0.,dim='arcsec'):
    """
    by default alpha is supposed to be given in arsec, if not it is given in MUSE pixel.
    a,b allow to decenter slightly the Moffat image.
    """
    ind=np.indices(shape)
    r=np.sqrt(((ind[0]-center[0]+a)**2 + ((ind[1]-center[1]+b))**2))
    if dim=='arcsec':
        r=r*0.2
    res=Moffat(r,alpha,beta)
    res=res/np.sum(res)
    return res