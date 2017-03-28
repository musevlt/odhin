# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 14:04:39 2016

@author: raphael
"""

import numpy as np
import scipy.signal as ssl
from scipy.stats import multivariate_normal
from downsampling import downsampling
from deblend_utils import generateMoffatIm


class SimuDeblending:

    def __init__(self, listCenter, listSpectra, listRadius, listIntens,
                 shapeLR=np.array([41,41]), shapeHR=np.array([161,161]),
                                 PSFMuse=None, FiltResp=None, listHiddenSources=[]):
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
            # self.PSFMuse=generateGaussianIm(self.shapeHR/2,self.shapeHR,sig=4.)
            # self.PSFMuse=generateGaussianIm(self.shapeLR/2,self.shapeLR,sig=5.)
            # self.PSFMuse=generateMoffatIm(self.shapeHR/2,self.shapeHR,3*self.d[0],2.7)
            self.PSFMuse = generateMoffatIm(shape=(15, 15), center=(7, 7), alpha=3, beta=2.8,dim=None)
        else:
            self.PSFMuse = PSFMuse  # f*f
        #self.generatePSFMatrixHR()
        self.generatePSFMatrixLR()
        self.generateImHR()  # N1*N2

        self.CubeHR = self.generateCubeHR()
        self.subMatrix = np.zeros((self.ImHR.size, self.shapeLR[0]*self.shapeLR[1]))

        self.CubeLR = np.zeros((self.LBDA,self.shapeLR[0],self.shapeLR[1]))
        for k in xrange(self.LBDA):
            #imConvol=np.dot(self.matrixPSF,self.CubeHR[k].flatten()).reshape(self.shapeHR)
            #self.CubeLR[k],self.downsamplingMatrix,self.Mh,self.Mv = downsampling(imConvol,self.shapeLR,returnMatrix=True)#downsampling
            self.CubeLR[k],self.downsamplingMatrix,self.Mh, self.Mv = \
                       downsampling(self.CubeHR[k], self.shapeLR, returnMatrix=True)#downsampling
        self.CubeLR = ssl.fftconvolve(self.CubeLR, self.PSFMuse[np.newaxis,:,:], mode='same') #l*n1*n2



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
        for i in xrange(self.shapeLR[0]):
            for j in xrange(self.shapeLR[1]):
                self.matrixPSF[max(0,i-shapeFSF[0]/2):min(self.shapeLR[0],i+shapeFSF[0]/2+1),
                               max(0,j-shapeFSF[1]/2):min(self.shapeLR[1],j+shapeFSF[1]/2+1),j+self.shapeLR[0]*i]\
                               = self.PSFMuse[int(max(0,shapeFSF[0]/2-i)):int(min(shapeFSF[0], self.shapeLR[0]+shapeFSF[0]*1/2.-i)),
                               int(max(0,shapeFSF[1]/2-j)):int(min(shapeFSF[1], self.shapeLR[1]+shapeFSF[1]*1/2.-j))]
        self.matrixPSF = self.matrixPSF.reshape((self.shapeLR[0]*self.shapeLR[1],\
                                                 self.shapeLR[0]*self.shapeLR[1]))

    def generatePSFMatrixHR(self):
        shapeFSF=self.PSFMuse.shape

        self.matrixPSF=np.zeros((self.shapeHR[0],self.shapeHR[1],self.shapeHR[0]*self.shapeHR[1]))
        for i in xrange(self.shapeHR[0]):
            for j in xrange(self.shapeHR[1]):
                self.matrixPSF[max(0, i-shapeFSF[0]/2):min(self.shapeHR[0], i+shapeFSF[0]/2+1),
                               max(0, j-shapeFSF[1]/2):min(self.shapeHR[1],j+shapeFSF[1]/2+1),j+self.shapeHR[0]*i] \
                = self.PSFMuse[int(max(0, shapeFSF[0]/2-i)):int(min(shapeFSF[0], self.shapeHR[0]+shapeFSF[0]*1/2.-i)),
                             int(max(0, shapeFSF[1]/2-j)):int(min(shapeFSF[1], self.shapeHR[1]+shapeFSF[1]*1/2.-j))]
        self.matrixPSF = self.matrixPSF.reshape((self.shapeHR[0]*self.shapeHR[1], self.shapeHR[0]*self.shapeHR[1]))

    def generateSrc(self, src):
        self.src=src
        self.src.cubes['MUSE_CUBE'].data = self.CubeLR
        self.src.images['HST_F606W'].data = self.ImHR
        self.src.images['HST_F775W'].data = self.ImHR
        self.src.images['HST_F814W'].data = self.ImHR
        self.src.images['HST_F850LP'].data = self.ImHR
        self.src.header['FSF00FWA'] = 3*2*np.sqrt(2**(1/2.8)-1)
        self.src.header['FSF00FWB'] = 0
        self.src.header['FSF00BET'] = 2.8



def generateGaussianIm(center=(12,12), shape=(25,25), sig=3.):
    ind = np.indices(shape)
    res = np.exp(- ((ind[0] - center[0])**2 + (ind[1] - center[1])**2) / (2 * sig**2))
    res = res/np.sum(res)
    return res
