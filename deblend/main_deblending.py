# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:31:40 2016

@author: raphael.bacher@gipsa-lab.fr
"""
import mpdaf
from mpdaf.obj import Cube,Image,Spectrum
import scipy.signal as ssl

import numpy as np
try:
    from skimage.measure import label
    from skimage.morphology import closing, square
    from skimage.measure import regionprops
except:
    pass
from scipy.interpolate import interp1d
import scipy.optimize as so
import os
from deblend_utils import convertFilt, calcFSF, apply_resampling_window, normalize,\
                        getSpatialShift, generateMoffatIm, approxNNLS,\
                        convertAbundanceMap, getMainSupport, ADMM_soft_neg, generatePSF_HST


DEFAULT_HSTFILTER606 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HST_ACS_WFC.F606W_81.dat')
DEFAULT_HSTFILTER775 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HST_ACS_WFC.F775W_81.dat')
DEFAULT_HSTFILTER814 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HST_ACS_WFC.F814W_81.dat')
DEFAULT_HSTFILTER850 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HST_ACS_WFC.F850LP_81.dat')

betaHST = 1.6
alphaHST = np.sqrt((0.085/0.2*15)**2/(4*(2**(1/betaHST)-1)))  # expressed in MUSE pixels


class Deblending():
    """
    Main class for deblending process


    Parameters
    ----------
    src : mpdaf Source
        The mpdaf Source object to be deblended
    listFiltName : list of filenames
        list of filenames of HST response filters.
    nbands : int
        The number of spectral bands to consider for computation
        MUSE FSF is assumed constant within a band.


    Attributes
    ----------

    cubeRebuilt : numpy.ndarray
        Estimated cube
    cubeRebuiltCont : numpy.ndarray
        Estimated cube continuum
    residus : numpy.ndarray
        The cube of residuals (datacube - estimated cube)
    spectraTot : list
        List of recovered spectra calibrated in flux
    sources: list
        list of spectra estimated during the process (non calibrated)

    listAbundanceMap (HR,LR,LRConvol) : list
    list of Abundance Map of each object detected, at each step :
        High resolution, after subsampling, and after convolution



    """
    def __init__(self, src, listFiltName=[DEFAULT_HSTFILTER606,
                                          DEFAULT_HSTFILTER775,
                                          DEFAULT_HSTFILTER814,
                                          DEFAULT_HSTFILTER850],
                                          nBands=10):

        self.src=src
        self.cubeLR = src.cubes['MUSE_CUBE'].data.filled(np.nanmedian(src.cubes['MUSE_CUBE'].data))
        self.cubeLRVar = src.cubes['MUSE_CUBE'].var.filled(np.nanmedian(src.cubes['MUSE_CUBE'].var))
        self.wcs = src.cubes['MUSE_CUBE'].wcs
        self.wave = src.cubes['MUSE_CUBE'].wave
        self.listImagesHR = [src.images['HST_F606W'].copy(),src.images['HST_F775W'].copy(),
                             src.images['HST_F814W'].copy(),src.images['HST_F850LP'].copy()]
        l_min, l_max = src.cubes['MUSE_CUBE'].wave.get_range()

        if 'FSF00FWA' in src.header.keys():
            self.fsf_a = src.header['FSF00FWA']
            self.fsf_b = src.header['FSF00FWB']
            self.betaFSF = src.header['FSF00BET']
        elif 'FSF99FWA' in src.header.keys():
            self.fsf_a = src.header['FSF99FWA']
            self.fsf_b = src.header['FSF99FWB']
            self.betaFSF = src.header['FSF99BET']
        elif 'FSF01FWA' in src.header.keys():
            self.fsf_a = src.header['FSF01FWA']
            self.fsf_b = src.header['FSF01FWB']
            self.betaFSF = src.header['FSF01BET']
        elif 'FSF02FWA' in src.header.keys():
            self.fsf_a = src.header['FSF02FWA']
            self.fsf_b = src.header['FSF02FWB']
            self.betaFSF = src.header['FSF02BET']
        elif 'FSF03FWA' in src.header.keys():
            self.fsf_a = src.header['FSF03FWA']
            self.fsf_b = src.header['FSF03FWB']
            self.betaFSF = src.header['FSF03BET']
        elif 'FSF04FWA' in src.header.keys():
            self.fsf_a = src.header['FSF04FWA']
            self.fsf_b = src.header['FSF04FWB']
            self.betaFSF = src.header['FSF04BET']
        elif 'FSF05FWA' in src.header.keys():
            self.fsf_a = src.header['FSF05FWA']
            self.fsf_b = src.header['FSF05FWB']
            self.betaFSF = src.header['FSF05BET']
        elif 'FSF06FWA' in src.header.keys():
            self.fsf_a = src.header['FSF06FWA']
            self.fsf_b = src.header['FSF06FWB']
            self.betaFSF = src.header['FSF06BET']
        elif 'FSF07FWA' in src.header.keys():
            self.fsf_a = src.header['FSF07FWA']
            self.fsf_b = src.header['FSF07FWB']
            self.betaFSF = src.header['FSF07BET']
        elif 'FSF08FWA' in src.header.keys():
            self.fsf_a = src.header['FSF08FWA']
            self.fsf_b = src.header['FSF08FWB']
            self.betaFSF = src.header['FSF08BET']
        elif 'FSF09FWA' in src.header.keys():
            self.fsf_a = src.header['FSF09FWA']
            self.fsf_b = src.header['FSF09FWB']
            self.betaFSF = src.header['FSF09BET']

        self.listFWHM = [self.fsf_a+self.fsf_b*(l+(l_min-l_max)/nBands)
                        for l in np.linspace(l_min, l_max, nBands)]

        self.nBands = nBands
        if listFiltName is not None:
            self.filtResp = [convertFilt(np.loadtxt(filtName), self.wave) for filtName in listFiltName]
        else:
            self.filtResp = [np.ones(self.cubeLR.shape[0])]*4
        for k, im in enumerate(self.listImagesHR):
            im.primary_header['FILTER'] = ['f606w', 'f775w', 'f814w', 'f850lp'][k]
        self.shapeHR = self.listImagesHR[0].shape
        self.shapeLR = self.cubeLR[0].shape
        self.residus = np.zeros((self.cubeLR.shape[0], self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        self.cubeRebuilt = np.zeros((self.cubeLR.shape[0], self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        self.PSF_HST = generatePSF_HST(alphaHST, betaHST)
        self.listShift = [[0, 0] for k in xrange(len(self.listImagesHR))]



    def corrAlign(self, PSF_HST=None,):
        """
        Align HST and MUSE images for each HST filter
        """
        self.listImagesLR = []
        for filt in xrange(len(self.filtResp)):
            # build muse image based on the spectral response filter
            imMUSE = (self.src.cubes['MUSE_CUBE']*self.filtResp[filt][:, np.newaxis, np.newaxis]).sum(axis=0)
            self.listImagesLR.append(imMUSE)

            #build muse average FSF for the filter spectral band
            fwhm_muse=self.fsf_a+self.fsf_b*[6060, 7750, 8140, 8500][filt]
            imHST = self.listImagesHR[filt]
            # get spatial shift (dx,dy)
            self.listShift[filt] = getSpatialShift(imMUSE, imHST, self.betaFSF, fwhm_muse, PSF_HST)

    def createAbundanceMap(self, thresh=None, segmap=None, background=False):
        # labelisation
        self.segmap = segmap
        if segmap is None:
            self.labelHR = self.getLabel(self.listImagesHR[0].data, thresh)
        else:
            self.labelHR = self.getLabel(segmap=segmap)
        self.background = background
        if self.background is False:
            self.nbSources = np.max(self.labelHR)
        else:
            self.nbSources = np.max(self.labelHR)+1

        self.listHST_ID = self.getHST_ID()

        self.listAbundanceMapHR = []

        # for each HST filter, create the high resolution intensity matrix (nbSources x Nb pixels )
        for j in xrange(len(self.listImagesHR)):
            abundanceMapHR = np.zeros((self.nbSources,self.listImagesHR[0].shape[0]*self.listImagesHR[0].shape[1]))
            mask = np.zeros(self.listImagesHR[0].shape)

            if self.background is True:  # if background is True, put abundanceMap of background in last position (estimated spectrum of background will also be last)
                mask[:] = 1.
                abundanceMapHR[np.max(self.labelHR)] = mask.copy().flatten()
                mask[:] = 0

            for k in xrange(1,np.max(self.labelHR)+1):
                mask[self.labelHR == k] = np.maximum(self.listImagesHR[j].data[self.labelHR==k],10**(-9))#avoid negative abundances


                #mask = mask/np.max(mask)
                abundanceMapHR[k-1] = mask.copy().flatten()  # k-1 to manage the background that have label 0 but is pushed to the last position
                mask[:] = 0

            self.listAbundanceMapHR.append(abundanceMapHR)

    def findSources(self, U=None, hstpsf=True, antialias=True, nonneg=True,alpha=2):
        """
        U: spatial support of abundances (for new estimations using OMP)
        hstpsf: apply hst psf on MUSE or not
        antialias: apply antialias filter or not
        nonneg: soft penalization for spectra estimation or not
        alpha: regularization parameter
        """
        self.convolvedCube = np.zeros_like(self.cubeLR)
        self.sources = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.varSources = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.listAbundanceMapLRConvol = []
        self.listAbundanceMapLRConvolClean = []
        self.tmp_sources = []
        self.tmp_var = []

        # If there are several HR images the process is applied on each image
        # and then the estimated spectra are combined using a mean weighted by the response filters
        for j in xrange(len(self.listImagesHR)):
            self.tmp_sources.append(np.zeros_like(self.sources))
            self.tmp_var.append(np.zeros_like(self.sources))

            self.listAbundanceMapLRConvol.append([])
            self.listAbundanceMapLRConvolClean.append([])


            delta = int(self.cubeLR.shape[0]/float(self.nBands))
            for i in xrange(self.nBands):
                abundanceMapLRConvol = convertAbundanceMap(self.listAbundanceMapHR[j],
                    self.src.cubes['MUSE_CUBE'][0,:,:], self.listImagesHR[j],self.listFWHM[i],
                    self.betaFSF, self.listShift[j], antialias=antialias)
                # abundanceMapLRConvol[abundanceMapLRConvol<0]=10**(-5) #Trunc negative abundances (from noisy negative HST pixels)

                ## TEST TO REMOVE CONVOLUTION WINGS
                U=getMainSupport(abundanceMapLRConvol[:-1], alpha=0.999)
                abundanceMapLRConvol[:-1][~U]=0

                if self.background is True:
                    abundanceMapLRConvol[-1] = 1.

                A = abundanceMapLRConvol.T
                B = self.cubeLR[i*delta:(i+1)*delta].reshape(
                        self.cubeLR[i*delta:(i+1)*delta].shape[0],
                        self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                var = self.cubeLRVar[i*delta:(i+1)*delta].reshape(
                            self.cubeLRVar[i*delta:(i+1)*delta].shape[0],
                            self.cubeLRVar.shape[1]*self.cubeLRVar.shape[2]).T

                if antialias:
                    if hstpsf:
                        B=np.hstack([apply_resampling_window(ssl.fftconvolve(B[:,l].reshape(self.shapeLR),self.PSF_HST,mode='same')).flatten()[:,np.newaxis] for l in xrange(B.shape[1]) ])
                        #B=np.hstack([apply_resampling_window(ssl.convolve2d(B[:,l].reshape(self.shapeLR),self.PSF_HST,mode='same',boundary='symm')).flatten()[:,np.newaxis] for l in xrange(B.shape[1]) ])
                        var=np.hstack([ssl.fftconvolve(var[:,l].reshape(self.shapeLR),self.PSF_HST**2,mode='same').flatten()[:,np.newaxis] for l in xrange(B.shape[1]) ])
                    else:
                        B=np.hstack([apply_resampling_window(B[:,l].reshape(self.shapeLR)).flatten()[:,np.newaxis] for l in xrange(B.shape[1]) ])
                else:
                    if hstpsf:
                        B=np.hstack([ssl.fftconvolve(B[:,l].reshape(self.shapeLR),self.PSF_HST,mode='same').flatten()[:,np.newaxis] for l in xrange(B.shape[1]) ])

                if j == 0:  # build a convolved cube for tests purposes
                    self.convolvedCube[i*delta:(i+1)*delta] = \
                                       B.T.reshape(B.shape[1], self.shapeLR[0], self.shapeLR[1])

                inv=np.linalg.pinv(A)
                if nonneg:
                    self.tmp_sources[j][:, i*delta:(i+1)*delta] = ADMM_soft_neg(A, B, alpha=alpha)
                    #self.tmp_sources[j][:, i*delta:(i+1)*delta] = ADMM_soft_neg(A, B, lmbda=[50./np.mean(var,axis=0)])
                else:
                    self.tmp_sources[j][:, i*delta:(i+1)*delta] = np.linalg.lstsq(A, B)[0]
                self.tmp_var[j][:, i*delta:(i+1)*delta] = \
                            np.array([np.sum(inv**2*col, axis=1) for col in var.T]).T

                # Estimation of abundances:
                # we have to recompute the abundance map by inversing again the system
                # save old abundanceMap
                self.listAbundanceMapLRConvolClean[j].append(abundanceMapLRConvol.copy())
                B = self.cubeLR[i*delta:(i+1)*delta].reshape(
                        self.cubeLR[i*delta:(i+1)*delta].shape[0],
                        self.cubeLR.shape[1]*self.cubeLR.shape[2])
                A = self.tmp_sources[j][:, i*delta:(i+1)*delta].T

                if self.background:
                    #U = np.array([(row/np.max(row)) > 0.0001 for row in abundanceMapLRConvol[:-1]])
                    #U = getMainSupport(abundanceMapLRConvol[:-1], alpha=0.99)
                    #U=np.ones_like(U).astype(int)
                    for it in xrange(abundanceMapLRConvol.shape[1]):
                        if np.sum(U[:, it]) > 0:
                            abundanceMapLRConvol[:-1, it][U[:, it]] = \
                                so.nnls(A[:,:-1][:,U[:, it]],B[:, it]-A[:,-1]*abundanceMapLRConvol[-1,it])[0]
                else:
                    #U=np.array([(row/np.max(row)) > 0.01 for row in abundanceMapLRConvol[:-1]])
                    U=getMainSupport(abundanceMapLRConvol, alpha=0.99)
                    for it in xrange(abundanceMapLRConvol.shape[1]):
                        abundanceMapLRConvol[:, it] = so.nnls(A, B[:, it])[0]

                self.listAbundanceMapLRConvol[j].append(abundanceMapLRConvol)

        self.combineSpectra()
        self.cubeRebuilt = self.rebuildCube(self.sources)
        self.getContinuumCube()
        self.getResidus()
        self.getTotSpectra()

    def combineSpectra(self):
        weigthTot = np.sum([self.filtResp[j] for j in xrange(len(self.filtResp))], axis=0)
        for i in xrange(self.nbSources):
            self.sources[i] = np.sum([self.filtResp[j]*self.tmp_sources[j][i]
                for j in xrange(len(self.filtResp))], axis=0)/weigthTot
            self.varSources[i] = np.sum([self.filtResp[j]*self.tmp_var[j][i]
                for j in xrange(len(self.filtResp))], axis=0)/weigthTot

    def getTotSpectra(self):
        self.spectraTot = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.varSpectraTot = np.zeros((self.nbSources, self.cubeLR.shape[0]))

        delta = self.cubeLR.shape[0]/float(self.nBands)
        tmp = []
        tmpVar = []
        for j in xrange(len(self.filtResp)):
            tmp.append(np.zeros_like(self.spectraTot))
            tmpVar.append(np.zeros_like(self.spectraTot))
            for i in xrange(self.nBands):
                for k in xrange(self.nbSources):
                    tmp[j][k, int(i*delta):int((i+1)*delta)] = \
                       np.sum(np.outer(self.tmp_sources[j][k, int(i*delta):int((i+1)*delta)],
                                       self.listAbundanceMapLRConvolClean[j][i][k]),axis=1)
                    tmpVar[j][k,int(i*delta):int((i+1)*delta)] = \
                          self.tmp_var[j][k,int(i*delta):int((i+1)*delta)]* \
                            np.sum(self.listAbundanceMapLRConvolClean[j][i][k])**2
        self.spectraTot = np.sum([self.filtResp[l]*tmp[l] for l in xrange(len(self.filtResp))], axis=0)/ \
                                np.sum([self.filtResp[l] for l in xrange(len(self.filtResp))],axis=0)

        self.varSpectraTot = np.sum([self.filtResp[l]*tmpVar[l] for l in xrange(len(self.filtResp))], axis=0)/ \
                                 np.sum([self.filtResp[l] for l in xrange(len(self.filtResp))], axis=0)


    def rebuildCube(self,sources,tmp_sources=None):
        if tmp_sources is None:
            tmp_sources = self.tmp_sources
        cubeRebuilt = np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        delta = self.cubeLR.shape[0]/float(self.nBands)
        for i in xrange(self.nBands):
            tmp = []
            weights = np.sum([self.filtResp[l][int(i*delta):int((i+1)*delta)]
                            for l in xrange(len(self.filtResp))], axis=0)
            for j in xrange(len(self.filtResp)):
                tmp.append(np.zeros_like(cubeRebuilt[int(i*delta):int((i+1)*delta),:]))
                tmp[j] = np.dot(tmp_sources[j][:,int(i*delta):int((i+1)*delta)].T,self.listAbundanceMapLRConvol[j][i])
            cubeRebuilt[int(i*delta):int((i+1)*delta),:] = np.sum(
                    [(self.filtResp[l][int(i*delta):int((i+1)*delta)]/weights)[:,np.newaxis]*tmp[l]
                    for l in xrange(len(self.filtResp))], axis=0)
        cubeRebuilt=cubeRebuilt.reshape(self.cubeLR.shape)
        return cubeRebuilt

    def getResidus(self):
        self.residus = self.cubeLR - self.cubeRebuilt

    def getContinuumCube(self, w=101):
        """
        Build continuum cube by median filtering (much faster here as it is done on objects spectra instead of all pixel spectra)
        """
        self.sourcesCont = ssl.medfilt(self.sources, kernel_size=(1, w))
        self.tmp_sourcesCont = [ssl.medfilt(tmp_source, kernel_size=(1, w)) for tmp_source in self.tmp_sources]
        self.cubeRebuiltCont = self.rebuildCube(self.sourcesCont, self.tmp_sourcesCont)

    def getLabel(self, image=None, thresh=None, segmap=None):
        if segmap is None:  # apply threshold
            bw = closing(image > thresh, square(2))
            label_image = label(bw)
            for region in regionprops(label_image):
                # skip small images
                if region.area < 3:
                    label_image[label_image == region.label] = 0

        else:  # exploit HST segmap
            label_image = np.zeros(segmap.shape, dtype='int')
            i = 0
            for k in sorted(set(segmap.flatten())):
                label_image[segmap == k] = i
                i = i+1

        return label_image

    def getHST_ID(self):
        if self.background is True:
            listHST_ID = [int(self.segmap[self.labelHR == k][0]) for k in xrange(1, self.nbSources)]
        else:
            listHST_ID = [int(self.segmap[self.labelHR == k][0]) for k in xrange(1, self.nbSources+1)]
        return listHST_ID

    def getsp(self):
        cat = {}
        for k, key in enumerate(self.listHST_ID):
            cat[key] = Spectrum(data=self.spectraTot[k], var=self.varSpectraTot[k], wave=self.src.spectra['MUSE_TOT'].wave)

        if self.background is True:
            cat['bg'] = Spectrum(data=self.spectraTot[k-1], wave=self.src.spectra['MUSE_TOT'].wave)
        return cat
