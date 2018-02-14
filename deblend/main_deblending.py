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
from .deblend_utils import convertFilt, calcFSF, apply_resampling_window, normalize,\
    generateMoffatIm,\
    convertIntensityMap, getMainSupport, generatePSF_HST,getBlurKernel


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
        The number of MUSE spectral bins to consider for computation.
        MUSE FSF is assumed constant within a bin.



    Attributes
    ----------

    estimatedCube : numpy.ndarray
        Estimated cube
    estimatedCubeCont : numpy.ndarray
        Estimated cube continuum
    residuals : numpy.ndarray
        The cube of residuals (datacube - estimated cube)
    sources: list
        list of estimated spectra
    varSources: list
        list of variances of estimated spectra
    listIntensityMap (HR, LRConvol) : list
        list of Abundance Map of each object detected, at high resolution, 
        and after convolution and subsampling



    """

    def __init__(self, src, 
                 params=None):

        self.src = src
        if params is None:
            params = Params()
        self.params = params
        
        self.cubeLR = src.cubes['MUSE_CUBE'].data.filled(
            np.nanmedian(src.cubes['MUSE_CUBE'].data))

        self.cubeLRVar = src.cubes['MUSE_CUBE'].var.filled(
            np.nanmedian(src.cubes['MUSE_CUBE'].var))
        self.wcs = src.cubes['MUSE_CUBE'].wcs
        self.wave = src.cubes['MUSE_CUBE'].wave
        self.listImagesHR = [
            src.images['HST_F606W'].copy(),
            src.images['HST_F775W'].copy(),
            src.images['HST_F814W'].copy(),
            src.images['HST_F850LP'].copy()]
        l_min, l_max = src.cubes['MUSE_CUBE'].wave.get_range()

        # get FSF parameters (the keys depends from the mosaic orignal field)
        if 'FSF00FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF00FWA']
            self.fsf_b = src.header['FSF00FWB']
            self.betaFSF = src.header['FSF00BET']
        elif 'FSF99FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF99FWA']
            self.fsf_b = src.header['FSF99FWB']
            self.betaFSF = src.header['FSF99BET']
        elif 'FSF01FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF01FWA']
            self.fsf_b = src.header['FSF01FWB']
            self.betaFSF = src.header['FSF01BET']
        elif 'FSF02FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF02FWA']
            self.fsf_b = src.header['FSF02FWB']
            self.betaFSF = src.header['FSF02BET']
        elif 'FSF03FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF03FWA']
            self.fsf_b = src.header['FSF03FWB']
            self.betaFSF = src.header['FSF03BET']
        elif 'FSF04FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF04FWA']
            self.fsf_b = src.header['FSF04FWB']
            self.betaFSF = src.header['FSF04BET']
        elif 'FSF05FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF05FWA']
            self.fsf_b = src.header['FSF05FWB']
            self.betaFSF = src.header['FSF05BET']
        elif 'FSF06FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF06FWA']
            self.fsf_b = src.header['FSF06FWB']
            self.betaFSF = src.header['FSF06BET']
        elif 'FSF07FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF07FWA']
            self.fsf_b = src.header['FSF07FWB']
            self.betaFSF = src.header['FSF07BET']
        elif 'FSF08FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF08FWA']
            self.fsf_b = src.header['FSF08FWB']
            self.betaFSF = src.header['FSF08BET']
        elif 'FSF09FWA' in list(src.header.keys()):
            self.fsf_a = src.header['FSF09FWA']
            self.fsf_b = src.header['FSF09FWB']
            self.betaFSF = src.header['FSF09BET']

        self.nBands = self.params.nBands
        self.listFWHM = [self.fsf_a +
                         self.fsf_b *
                         (l +
                          (l_min -
                           l_max) /
                             self.nBands) for l in np.linspace(l_min, l_max, self.nBands)]

        
        if self.params.listFiltName is not None:
            self.filtResp = [
                convertFilt(
                    np.loadtxt(filtName),
                    self.wave) for filtName in self.params.listFiltName]
        else:
            self.filtResp = [np.ones(self.cubeLR.shape[0])] * 4

        # needeed for muse_analysis functions
        for k, im in enumerate(self.listImagesHR):
            im.primary_header['FILTER'] = [
                'f606w', 'f775w', 'f814w', 'f850lp'][k]

        self.shapeHR = self.listImagesHR[0].shape
        self.shapeLR = self.cubeLR[0].shape
        self.residuals = np.zeros(
            (self.cubeLR.shape[0],
             self.cubeLR.shape[1] *
             self.cubeLR.shape[2]))
        self.estimatedCube = np.zeros(
            (self.cubeLR.shape[0],
             self.cubeLR.shape[1] *
             self.cubeLR.shape[2]))
        self.PSF_HST = generatePSF_HST(self.params.alphaHST, self.params.betaHST)

    def createIntensityMap(self, segmap=None, thresh=None):
        """
        Create intensity maps from HST images and segmentation map.
        To be called before calling findSources()

        Parameters
        ----------
        segmap : `ndarray`

        thres : (not used if segmap)
            threshold to use on HST image to segment

        """

        # labelisation
        self.segmap = segmap
        if segmap is None:
            self.labelHR = self._getLabel(self.listImagesHR[0].data, thresh)
        else:
            self.labelHR = self._getLabel(segmap=segmap)
        self.nbSources = np.max(self.labelHR) + 1  # add one for background

        self.listHST_ID = self._getHST_ID()

        self.listIntensityMapHR = []

        # for each HST filter, create the high resolution intensity matrix
        # (nbSources x Nb pixels )
        for j in range(len(self.listImagesHR)):
            intensityMapHR = np.zeros(
                (self.nbSources,
                 self.listImagesHR[0].shape[0] *
                 self.listImagesHR[0].shape[1]))
            mask = np.zeros(self.listImagesHR[0].shape)

            # put intensityMap of background in first position (estimated
            # spectrum of background will also be first)
            intensityMapHR[0] = 1.

            for k in range(1, np.max(self.labelHR) + 1):
                mask[self.labelHR == k] = np.maximum(
                    self.listImagesHR[j].data[self.labelHR == k], 10**(-9))  # avoid negative abundances
                # k-1 to manage the background that have label 0 but is pushed
                # to the last position
                intensityMapHR[k] = mask.copy().flatten()
                mask[:] = 0

            self.listIntensityMapHR.append(intensityMapHR)

    def findSources(self, regul=True, store=False, filt_w=101):
        """
        Main function : estimate sources spectra

        regul: bool
            add regularization or not
        store: bool
            store intermediate results
        filt_w : int
            size of median filter window
        """

        if regul:
            # precompute continuum
            shape = self.cubeLR.shape
            self.cubeLR_c = np.vstack([medfilt(y, filt_w) for y in self.cubeLR.reshape(
                shape[0], shape[1] * shape[2]).T]).T.reshape(shape)

        
        # compute HST-MUSE transfer functions for all MUSE FSF fwhm
        # considered
        self.listTransferKernel = self._generateHSTMUSE_transfer_PSF()

        self.sources = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.varSources = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.listIntensityMapLRConvol = []
        self.tmp_sources = []
        self.tmp_var = []
        self.listAlphas = []
        self.listRSS = []
        self.listCorrFlux = []
        if store:
            self.listMask = []
            self.listccoeff = []
            self.listlcoeff = []
            self.listY = []
            self.listYc = []
            self.listYl = []
            self.spatialMask = []

        # If there are several HR images the process is applied on each image
        # and then the estimated spectra are combined using a mean weighted by
        # the response filters
        for j in range(len(self.listImagesHR)):
            self.listAlphas.append([])
            self.listRSS.append([])
            self.listCorrFlux.append([])
            if store:
                self.listMask.append([])
                self.listccoeff.append([])
                self.listlcoeff.append([])
                self.listY.append([])
                self.listYc.append([])
                self.listYl.append([])
                self.spatialMask.append([])

            self.tmp_sources.append(np.zeros_like(self.sources))
            self.tmp_var.append(np.zeros_like(self.sources))

            self.listIntensityMapLRConvol.append([])

            delta = int(self.cubeLR.shape[0] / float(self.nBands))
            for i in range(self.nBands):
                self.listAlphas[j].append([])
                self.listRSS[j].append([])
                self.listCorrFlux[j].append([])
                if store:
                    self.listMask[j].append([])
                    self.listccoeff[j].append([])
                    self.listlcoeff[j].append([])
                    self.listY[j].append([])
                    self.listYc[j].append([])
                    self.listYl[j].append([])
                    self.spatialMask[j].append([])

                # Create intensity maps at MUSE resolution
                intensityMapLRConvol = convertIntensityMap(
                    self.listIntensityMapHR[j],
                    self.src.cubes['MUSE_CUBE'][0,:,:],
                    self.listImagesHR[j],
                    self.listFWHM[i],
                    self.betaFSF,
                    self.listTransferKernel[i])

                # truncate intensity map support after convolution
                supp = getMainSupport(intensityMapLRConvol[1:], alpha=0.999)
                intensityMapLRConvol[1:][~supp] = 0

                # put ones everywhere for background intensity map
                intensityMapLRConvol[0] = 1.

                
                # U : n x k (n number of pixels, k number of objects, lmbda number of wavelengths)
                # Y : n x lmbda
                # Yvar : n x lmbda
                
                U = intensityMapLRConvol.T
                Y = self.cubeLR[i * delta:(i + 1) * delta].reshape(
                    self.cubeLR[i * delta:(i + 1) * delta].shape[0],
                    self.cubeLR.shape[1] * self.cubeLR.shape[2]).T
                if regul:
                    Y_c = self.cubeLR_c[i * delta:(i + 1) * delta].reshape(
                        self.cubeLR[i * delta:(i + 1) * delta].shape[0],
                        self.cubeLR.shape[1] * self.cubeLR.shape[2]).T
                Yvar = self.cubeLRVar[i * delta:(i + 1) * delta].reshape(
                    self.cubeLRVar[i * delta:(i + 1) * delta].shape[0],
                    self.cubeLRVar.shape[1] * self.cubeLRVar.shape[2]).T

                # normalize intensity maps in flux to get flux-calibrated
                # estimated spectra
                for u in range(U.shape[1]):
                    U[:, u] = U[:, u] / np.sum(U[:, u])

                if regul:  # apply regularization

                    # remove background from intensity matrix as
                    # intercept is used instead
                    U_ = U[:, 1:]

                    # generate support
                    support = np.zeros(U.shape[0]).astype(bool)
                    for u in range(U_.shape[1]):
                        support[U_[:, u] > 0.1 * np.max(U_[:, u])] = True

                    Y_sig2 = np.var(Y[~support, :], axis=0)
                    res = regulDeblendFunc(
                        U_,
                        Y,
                        Y_c=Y_c,
                        ng=200,
                        l_method='gbic',
                        c_method='gridge_cv',
                        corrflux=True,
                        support=support,
                        Y_sig2=Y_sig2,
                        filt_w=filt_w,
                        oneSig=True)

                    # get spectra estimation

                    self.tmp_sources[j][1:, i * delta:(i + 1) * delta] = res[0]
                    # for background spectrum get intercept (multiply by number
                    # of pixels to get tot flux)
                    self.tmp_sources[j][0, i * \
                        delta:(i + 1) * delta] = res[1] * U.shape[0]

                    # store all elements for checking purposes
                    self.listAlphas[j][i] = res[8]
                    self.listRSS[j][i] = res[9]
                    self.listCorrFlux[j][i] = res[10]
                    if store:
                        self.spatialMask[j][i] = support
                        self.listMask[j][i] = res[2]
                        self.listccoeff[j][i] = res[3]
                        self.listlcoeff[j][i] = res[4]
                        self.listY[j][i] = res[5]
                        self.listYl[j][i] = res[6]
                        self.listYc[j][i] = res[7]

                else:  # use classical least squares solution
                    self.tmp_sources[j][:, i *
                                        delta:(i +
                                               1) *
                                        delta] = np.linalg.lstsq(U, Y)[0]

                # get spectra variance : as spectra is obtained by (U^T.U)^(-1).U^T.Y
                # variance of estimated spectra is obtained by
                # (U^T.U)^(-1).Yvar
                Uinv = np.linalg.pinv(U)
                self.tmp_var[j][:, i * delta:(i + 1) * delta] = np.array(
                    [np.sum(Uinv**2 * col, axis=1) for col in Yvar.T]).T

                self.listIntensityMapLRConvol[j].append(intensityMapLRConvol)

        self._combineSpectra()
        self.estimatedCube = self._rebuildCube(self.tmp_sources)
        self._getContinuumCube()
        self._getResiduals()

    def _generateHSTMUSE_transfer_PSF(self):
        """
        Generate HST-MUSE transfer PSF 
        """
        hst = self.listImagesHR[0]

        dy, dx = hst.get_step(unit=units.arcsec)

        shape = np.asarray(hst.shape).astype(int)

        # get odd shape
        shape_1 = shape // 2 * 2 + 1
        center = shape_1 // 2
        
        # Build "distances to center" matrix.
        ind = np.indices(shape_1)
        rsq = ((ind[0] - center[0]) * dx)**2 + (((ind[1] - center[1])) * dy)**2

        # Build HST FSF
        asq_hst = self.params.fwhmHST**2 / 4.0 / (2.0**(1.0 / self.params.betaHST) - 1.0)
        psf_hst = 1.0 / (1.0 + rsq / asq_hst)**self.params.betaHST
        psf_hst = psf_hst / np.sum(psf_hst)

        listTransferKernel = []
        for fwhm in self.listFWHM:
            # Build MUSE FSF
            asq = fwhm**2 / 4.0 / (2.0**(1.0 / self.betaFSF) - 1.0)
            im_muse = 1.0 / (1.0 + rsq / asq)**self.betaFSF
            im_muse = im_muse / np.sum(im_muse)
            listTransferKernel.append(getBlurKernel(imHR=psf_hst, imLR=im_muse, sizeKer=(21,21)))
        return listTransferKernel

            

    def _combineSpectra(self):
        """
        Combine spectra estimated on each HST image
        """
        weigthTot = np.sum([self.filtResp[j]
                            for j in range(len(self.filtResp))], axis=0)
        for i in range(self.nbSources):
            self.sources[i] = np.sum([self.filtResp[j] * self.tmp_sources[j][i]
                                      for j in range(len(self.filtResp))], axis=0) / weigthTot
            self.varSources[i] = np.sum([self.filtResp[j] * self.tmp_var[j][i]
                                         for j in range(len(self.filtResp))], axis=0) / weigthTot

    def _rebuildCube(self, tmp_sources):
        """
        Create estimated cube. We have to work on each MUSE spectral bin as
        the spatial distribution is different on each bin
        """
        estimatedCube = np.zeros(
            (self.cubeLR.shape[0],
             self.cubeLR.shape[1] *
             self.cubeLR.shape[2]))
        delta = self.cubeLR.shape[0] / float(self.nBands)

        for i in range(self.nBands):
            tmp = []
            weightTot = np.sum([self.filtResp[l][int(i * delta):int((i + 1) * delta)]
                                for l in range(len(self.filtResp))], axis=0)

            for j in range(len(self.filtResp)):
                tmp.append(np.zeros_like(
                    estimatedCube[int(i * delta):int((i + 1) * delta), :]))
                tmp[j] = np.dot(tmp_sources[j][:,
                                               int(i * delta):int((i + 1) * delta)].T,
                                self.listIntensityMapLRConvol[j][i])

            estimatedCube[int(i * delta):int((i + 1) * delta), :] = np.sum(
                [(self.filtResp[l][int(i * delta):int((i + 1) * delta)] / weightTot)[:, np.newaxis] * tmp[l]
                 for l in range(len(self.filtResp))], axis=0)

        estimatedCube = estimatedCube.reshape(self.cubeLR.shape)
        return estimatedCube

    def _getResiduals(self):
        self.residuals = self.cubeLR - self.estimatedCube

    def _getContinuumCube(self, w=101):
        """
        Build continuum cube by median filtering (much faster here as it is done on objects spectra instead of all pixel spectra)
        """
        self.sourcesCont = ssl.medfilt(self.sources, kernel_size=(1, w))
        self.tmp_sourcesCont = [
            ssl.medfilt(
                tmp_source, kernel_size=(
                    1, w)) for tmp_source in self.tmp_sources]
        self.estimatedCubeCont = self._rebuildCube(self.tmp_sourcesCont)

    def _getLabel(self, segmap=None):
        """
        Create a new segmap with contiguous indices
        """
        label_image = np.zeros(segmap.shape, dtype='int')
        i = 0
        for k in sorted(set(segmap.flatten())):
            label_image[segmap == k] = i
            i = i + 1

        return label_image

    def _getHST_ID(self):
        """
        Get the list of HST ids for each label of labelHR (first is background 'bg')
        """

        listHST_ID = ['bg'] + [int(self.segmap[self.labelHR == k][0])
                               for k in range(1, self.nbSources)]
        return listHST_ID

    def getsp(self):
        """
        Get estimated spectra as a dict (with the HST ids as keys) of mpdaf Spectra
        """
        cat = {}
        for k, key in enumerate(self.listHST_ID):
            cat[key] = Spectrum(
                data=self.sources[k],
                var=self.varSources[k],
                wave=self.src.spectra['MUSE_TOT'].wave)

        return cat
