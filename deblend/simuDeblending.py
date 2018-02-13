# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:04:39 2016

@author: raphael
"""





import numpy as np
import scipy.signal as ssl
from scipy.stats import multivariate_normal
from .downsampling import downsampling
from .deblend_utils import generateMoffatIm
import os
import astropy.io.fits as pyfits

class SimuDeblending:

    def __init__(self, listCenter, listSpectra, listRadius, listIntens,
                 shapeLR=np.array([41,41]), shapeHR=np.array([161,161]),
                                 PSFMuse=None, FiltResp=None, listHiddenSources=[],genFromCubeHR=True):
        """
        FSFMuse: at high resolution
        """
        self.LBDA = listSpectra[0].shape[0]
        self.shapeHR = shapeHR
        self.shapeLR = shapeLR
        self.d = self.shapeHR/self.shapeLR
        self.nbS = int(len(listCenter))
        self.listCenter = listCenter
        self.listRadius = listRadius
        self.listSpectra = listSpectra  # s*l
        self.listHiddenSources = listHiddenSources
        self.DicSources = np.zeros((self.LBDA, self.nbS))
        for k, spec in enumerate(listSpectra):
            self.DicSources[:, k] = self.listSpectra[k]

        # for now : spectral filter response =1

        if PSFMuse is None:
            self.PSFMuse = generateMoffatIm(shape=(21, 21), center=(10, 10), alpha=4, beta=2.8,dim=None)
        else:
            self.PSFMuse = PSFMuse  # f*f
        #self.generatePSFMatrixHR()
        self.generatePSFMatrixLR()
        self.generateImHR()  # N1*N2

        self.CubeHR = self.generateCubeHR()
        self.ImHR2 = np.sum(self.CubeHR,axis=0)

        #convolve by HST PSF
        self.fwhm_hst=0.085
        self.betaHST=1.6
        a_hst = np.sqrt(self.fwhm_hst**2 / 4.0 / (2.0**(1.0 / self.betaHST) - 1.0))
        self.PSFHST = generateMoffatIm(shape=(101, 101), center=(50, 50), alpha=a_hst, beta=self.betaHST,dim='HST')
        self.ImHR2=ssl.fftconvolve(self.ImHR2,self.PSFHST,mode='same')
        for k in range(self.nbS):
            self.mapAbundances[k]=ssl.fftconvolve(self.mapAbundances[k].reshape(self.shapeHR), self.PSFHST,mode='same').flatten()
        self._generateHSTMUSE_transfert_PSF()
        tmp_dir='./tmp/'
        imTransfertHSTMUSE = pyfits.open(tmp_dir+'kernel_simu.fits')[0].data
        imTransfertHSTMUSE=imTransfertHSTMUSE/np.sum(imTransfertHSTMUSE)

        self.subMatrix = np.zeros((self.ImHR.size, self.shapeLR[0]*self.shapeLR[1]))

        self.CubeLR = np.zeros((self.LBDA,self.shapeLR[0],self.shapeLR[1]))
        #self.mapAbundancesLR = np.zeros((self.nbS, self.shapeLR[0]*self.shapeLR[1]))
        self.mapAbundancesConvol = np.zeros((self.nbS, self.shapeHR[0]*self.shapeHR[1]))
        self.mapAbundancesLRConvol = np.zeros((self.nbS, self.shapeLR[0]*self.shapeLR[1]))

        for k in range(self.nbS):
            self.mapAbundancesConvol[k]=ssl.fftconvolve(self.mapAbundances[k].reshape(self.shapeHR), imTransfertHSTMUSE, mode='same').flatten()
            self.mapAbundancesLRConvol[k]=downsampling(self.mapAbundancesConvol[k].reshape(self.shapeHR), self.shapeLR).flatten()

            #self.mapAbundancesLR[k]=downsampling(self.mapAbundances[k].reshape(self.shapeHR), self.shapeLR).flatten()
            #self.mapAbundancesLRConvol[k]=ssl.fftconvolve(self.mapAbundancesLR[k].reshape(self.shapeLR), self.PSFMuse, mode='same').flatten()

        if genFromCubeHR==True:
            self.CubeLR,self.downsamplingMatrix,self.Mh, self.Mv = \
                       downsampling(self.CubeHR, self.shapeLR, returnMatrix=True)#downsampling
            self.CubeLR = ssl.fftconvolve(self.CubeLR, self.PSFMuse[np.newaxis,:,:], mode='same') #l*n1*n2
        else:
            _,self.downsamplingMatrix,self.Mh, self.Mv = \
                    downsampling(self.CubeHR[0:1], self.shapeLR, returnMatrix=True)#downsampling
            self.CubeLR = np.dot(self.mapAbundancesLRConvol.T,self.DicSources.T).T.reshape(self.LBDA,self.shapeLR[0],self.shapeLR[1])

        #convolve by HST PSF (after construction of the cubeLR to simulated the presence of two unrelated PSF one for HST, one for MUSE)
        #for k in range(self.nbS):
        #    self.mapAbundances[k]=ssl.fftconvolve(self.mapAbundances[k].reshape(self.shapeHR), self.PSFHST,mode='same').flatten()

    def _generateHSTMUSE_transfert_PSF(self):
        """
        Generate HST-MUSE transfert PSF using astconvolve
        """
        hst = self.ImHR2

        dy= dx = 0.03

        shape = np.asarray(hst.shape).astype(int)

        # get odd shape
        shape_1 = shape//2 *2 +1
        center=shape_1//2
        # Extract the dimensions of the expanded Y and X axes.
        ind = np.indices(shape_1)
        rsq=((ind[0]-center[0])*dx)**2 + (((ind[1]-center[1]))*dy)**2
        betaHST = 1.6
        fwhm_hst=0.085

        asq_hst = fwhm_hst**2 / 4.0 / (2.0**(1.0 / betaHST) - 1.0)
        psf_hst = 1.0 / (1.0 + rsq / asq_hst)**betaHST
        psf_hst=psf_hst/np.sum(psf_hst)
        im=generateMoffatIm(shape=shape_1, center=center, alpha=4*0.2, beta=2.8,dim='HST')

        tmp_dir='./tmp/'
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise
        pyfits.writeto(tmp_dir+'wider.fits',im,overwrite=True)
        pyfits.writeto(tmp_dir+'sharper.fits',psf_hst,overwrite=True)
        os.system('export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH')
        os.system('astconvolve --kernel=%ssharper.fits --makekernel=%s %swider.fits --output=%skernel_simu.fits'%(tmp_dir,np.maximum(im.shape[0],im.shape[1])//2-1,tmp_dir,tmp_dir))

    def generateImHR(self):
        """
        """
        self.ImHR = np.zeros(self.shapeHR)
        self.mapAbundances = np.zeros((self.nbS, self.ImHR.size))
        self.ImHR_hide = np.zeros(self.shapeHR)
        self.mapAbundances_hide = np.zeros((self.nbS-len(self.listHiddenSources),
                                            self.ImHR.size))
        self.label = np.zeros(self.shapeHR)+len(self.listCenter)
        self.spectraTot = np.zeros((self.nbS, self.LBDA))
        x, y = np.mgrid[0:self.shapeHR[0], 0:self.shapeHR[1]]
        for k, center in enumerate(self.listCenter):
            zHalo = multivariate_normal.pdf(np.swapaxes([x, y], 0, 2),
                                            mean=center, cov=[[self.listRadius[k], 0],
                                                              [0,  self.listRadius[k]]])
            zHalo = zHalo*1/np.max(zHalo)
            zHalo[zHalo < 0.1] = 0
            self.label[zHalo > 0] = k
            self.ImHR = self.ImHR+zHalo
            self.mapAbundances[k] = zHalo.flatten()
            self.spectraTot[k] = np.sum(self.mapAbundances[k])*self.listSpectra[k]
            if k not in self.listHiddenSources:
                self.ImHR_hide = self.ImHR+zHalo
                self.mapAbundances_hide[k] = zHalo.flatten()

    def generateCubeHR(self):
        cubeHR = np.zeros((self.LBDA, self.shapeHR[0], self.shapeHR[1]))
        for k, spec in enumerate(self.listSpectra):
            cubeHR[:,self.label==k] = spec[:,np.newaxis]*np.tile(self.ImHR[self.label==k],\
                  (self.LBDA,1))
        return cubeHR

    def generatePSFMatrixLR(self):
        shapeFSF = self.PSFMuse.shape

        self.matrixPSF = np.zeros((self.shapeLR[0],self.shapeLR[1], self.shapeLR[0]*self.shapeLR[1]))
        for i in range(self.shapeLR[0]):
            for j in range(self.shapeLR[1]):
                self.matrixPSF[max(0,i-shapeFSF[0]//2):min(self.shapeLR[0],i+shapeFSF[0]//2+1),
                               max(0,j-shapeFSF[1]//2):min(self.shapeLR[1],j+shapeFSF[1]//2+1),j+self.shapeLR[0]*i]\
                               = self.PSFMuse[int(max(0,shapeFSF[0]//2-i)):int(min(shapeFSF[0], self.shapeLR[0]+shapeFSF[0]*1/2.-i)),
                               int(max(0,shapeFSF[1]//2-j)):int(min(shapeFSF[1], self.shapeLR[1]+shapeFSF[1]*1/2.-j))]
        self.matrixPSF = self.matrixPSF.reshape((self.shapeLR[0]*self.shapeLR[1],\
                                                 self.shapeLR[0]*self.shapeLR[1]))

    def generatePSFMatrixHR(self):
        shapeFSF=self.PSFMuse.shape

        self.matrixPSF=np.zeros((self.shapeHR[0],self.shapeHR[1],self.shapeHR[0]*self.shapeHR[1]))
        for i in range(self.shapeHR[0]):
            for j in range(self.shapeHR[1]):
                self.matrixPSF[max(0, i-shapeFSF[0]//2):min(self.shapeHR[0], i+shapeFSF[0]//2+1),
                               max(0, j-shapeFSF[1]//2):min(self.shapeHR[1],j+shapeFSF[1]//2+1),j+self.shapeHR[0]*i] \
                = self.PSFMuse[int(max(0, shapeFSF[0]//2-i)):int(min(shapeFSF[0], self.shapeHR[0]+shapeFSF[0]*1/2.-i)),
                             int(max(0, shapeFSF[1]//2-j)):int(min(shapeFSF[1], self.shapeHR[1]+shapeFSF[1]*1/2.-j))]
        self.matrixPSF = self.matrixPSF.reshape((self.shapeHR[0]*self.shapeHR[1], self.shapeHR[0]*self.shapeHR[1]))

    def generateSrc(self, src):
        self.src=src
        self.src.cubes['MUSE_CUBE'].data = self.CubeLR
        self.src.images['HST_F606W'].data = self.ImHR2
        self.src.images['HST_F775W'].data = self.ImHR2
        self.src.images['HST_F814W'].data = self.ImHR2
        self.src.images['HST_F850LP'].data = self.ImHR2
        self.src.header['FSF00FWA'] = 4*0.2*2*np.sqrt(2**(1/2.8)-1)
        self.src.header['FSF00FWB'] = 0
        self.src.header['FSF00BET'] = 2.8



def generateGaussianIm(center=(12,12), shape=(25,25), sig=3.):
    ind = np.indices(shape)
    res = np.exp(- ((ind[0] - center[0])**2 + (ind[1] - center[1])**2) / (2 * sig**2))
    res = res/np.sum(res)
    return res
