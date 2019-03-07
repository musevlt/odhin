# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""

import astropy.units as units
import numpy as np
import scipy.signal as ssl

from astropy.table import Table
from mpdaf.obj import Spectrum
from mpdaf.sdetect import Source

from .regularization import regulDeblendFunc, medfilt
from .parameters import Params
from .deblend_utils import (convertFilt, _getLabel, convertIntensityMap,
                            getMainSupport, generatePSF_HST, getBlurKernel)


def deblendGroup(subcube, subhstimages, subsegmap, listObjInBlob,
                 listHSTObjInBlob, group, outfile):
    debl = Deblending(subcube, subhstimages)
    debl.createIntensityMap(subsegmap.data.filled(0.))
    debl.findSources()
    debl.write(outfile, listObjInBlob, listHSTObjInBlob, group)


class Deblending():
    """
    Main class for deblending process

    Parameters
    ----------
    cube : mpdaf Cube
        The mpdaf Cube object to be deblended
    HSTimages:
        HST mpdaf images
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

    def __init__(self, cube, HSTImages, params=None):

        self.cube = cube
        if params is None:
            params = Params()
        self.params = params
        self.HSTImages = HSTImages
        self.cubeLR = cube.data.filled(np.ma.median(cube.data))
        self.cubeLRVar = cube.var.filled(np.ma.median(cube.var))
        self.wcs = cube.wcs
        self.wave = cube.wave
        self.listImagesHR = [hst for hst in HSTImages]
        l_min, l_max = cube.wave.get_range()

        # get FSF parameters
        self.fsf_a = self.params.fsf_a_muse
        self.fsf_b = self.params.fsf_b_muse
        self.fsf_beta_muse = self.params.fsf_beta_muse

        self.nBands = self.params.nBands
        dl = (l_min - l_max) / self.nBands
        self.listFWHM = [self.fsf_a + self.fsf_b * (l + dl)
                         for l in np.linspace(l_min, l_max, self.nBands)]

        if self.params.listFiltName is not None:
            self.filtResp = [convertFilt(np.loadtxt(filtName), self.wave)
                             for filtName in self.params.listFiltName]
        else:
            self.filtResp = [np.ones(self.cubeLR.shape[0])] * 4

        # needeed for muse_analysis functions
        filters = ['f606w', 'f775w', 'f814w', 'f850lp']
        for k, im in enumerate(self.listImagesHR):
            im.primary_header['FILTER'] = filters[k]

        self.shapeHR = self.listImagesHR[0].shape
        self.shapeLR = self.cubeLR[0].shape
        self.residuals = np.zeros((self.cubeLR.shape[0],
                                   self.cubeLR.shape[1] *
                                   self.cubeLR.shape[2]))
        self.estimatedCube = self.cube.clone()
        self.PSF_HST = generatePSF_HST(self.params.alpha_hst,
                                       self.params.beta_hst)

        # for each HST band list all MUSE bands inside it
        self.listBands = self._getListBands(self.nBands, self.filtResp)

    def _getListBands(self, nBands, filtResp):
        """
        For each HST band, get the list of all MUSE bands inside it
        """
        listBands = []
        lmbda = len(filtResp[0])
        for i in range(len(filtResp)):
            listBands.append([])
            for j in range(nBands):
                if (filtResp[i][j * lmbda // nBands] != 0) or (
                        filtResp[i][np.minimum((j + 1) * lmbda // nBands, lmbda - 1)] != 0):
                    listBands[i].append(j)
        return listBands

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
            # FIXME: this will break with _getLabel ?
            self.labelHR = _getLabel(self.listImagesHR[0].data, thresh)
        else:
            self.labelHR = _getLabel(segmap=segmap)
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
                # avoid negative abundances
                mask[self.labelHR == k] = np.maximum(
                    self.listImagesHR[j].data[self.labelHR == k], 10**(-9))
                intensityMapHR[k] = mask.copy().flatten()
                mask[:] = 0

            self.listIntensityMapHR.append(intensityMapHR)

    def findSources(self, store=False):
        """
        Main function : estimate sources spectra
        store: bool
            store intermediate results
        """
        regul = self.params.regul
        filt_w = self.params.filt_w
        if regul:
            # precompute continuum
            shape = self.cubeLR.shape
            self.cubeLR_c = np.vstack([
                medfilt(y, filt_w)
                for y in self.cubeLR.reshape(
                    shape[0], shape[1] * shape[2]).T
            ]).T.reshape(shape)

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

                imin, imax = i * delta, (i + 1) * delta

                # Do the estimation only if MUSE band is in HST band
                if i in self.listBands[j]:
                    # Create intensity maps at MUSE resolution
                    intensityMapLRConvol = convertIntensityMap(
                        self.listIntensityMapHR[j],
                        self.cube[0, :, :],
                        self.listImagesHR[j],
                        self.listFWHM[i],
                        self.fsf_beta_muse,
                        self.listTransferKernel[i])

                    # truncate intensity map support after convolution using
                    # alpha_cut
                    supp = getMainSupport(
                        intensityMapLRConvol[1:], alpha=self.params.alpha_cut)
                    intensityMapLRConvol[1:][~supp] = 0

                    # put ones everywhere for background intensity map
                    intensityMapLRConvol[0] = 1.

                    # U : n x k (n number of pixels, k number of objects,
                    #            lmbda number of wavelengths)
                    # Y : n x lmbda
                    # Yvar : n x lmbda

                    U = intensityMapLRConvol.T
                    Y = self.cubeLR[imin:imax].reshape(
                        self.cubeLR[imin:imax].shape[0],
                        self.cubeLR.shape[1] * self.cubeLR.shape[2]).T
                    if regul:
                        Y_c = self.cubeLR_c[imin:imax].reshape(
                            self.cubeLR[imin:imax].shape[0],
                            self.cubeLR.shape[1] * self.cubeLR.shape[2]).T
                    Yvar = self.cubeLRVar[imin:imax].reshape(
                        self.cubeLRVar[imin:imax].shape[0],
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

                        # Y_sig2 = np.var(Y[~support, :], axis=0)
                        Y_sig2 = np.mean(Yvar, axis=0)
                        res = regulDeblendFunc(U_, Y, Y_c=Y_c, support=support,
                                               Y_sig2=Y_sig2, filt_w=filt_w)

                        # get spectra estimation
                        self.tmp_sources[j][1:, imin:imax] = res[0]
                        # for background spectrum get intercept (multiply by
                        # number of pixels to get tot flux)
                        self.tmp_sources[j][0, imin:imax] = res[1] * U.shape[0]

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
                        self.tmp_sources[j][:, imin:imax] = np.linalg.lstsq(U, Y)[0]

                    # get spectra variance : as spectra is obtained by (U^T.U)^(-1).U^T.Y
                    # variance of estimated spectra is obtained by
                    # (U^T.U)^(-1).Yvar
                    Uinv = np.linalg.pinv(U)
                    self.tmp_var[j][:, imin:imax] = np.array(
                        [np.sum(Uinv**2 * col, axis=1) for col in Yvar.T]).T

                    self.listIntensityMapLRConvol[j].append(
                        intensityMapLRConvol)
                else:
                    self.tmp_sources[j][:, imin:imax] = 0
                    self.tmp_var[j][:, imin:imax] = 0
                    self.listAlphas[j][i] = None
                    self.listRSS[j][i] = None
                    self.listCorrFlux[j][i] = None
                    if store:
                        self.spatialMask[j][i] = None
                        self.listMask[j][i] = None
                        self.listccoeff[j][i] = None
                        self.listlcoeff[j][i] = None
                        self.listY[j][i] = None
                        self.listYl[j][i] = None
                        self.listYc[j][i] = None
                    self.listIntensityMapLRConvol[j].append(
                        np.zeros((self.nbSources,
                                  self.shapeLR[0] * self.shapeLR[1]))
                    )

        self._combineSpectra()
        self.estimatedCube.data = self._rebuildCube(self.tmp_sources)

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
        asq_hst = self.params.fwhm_hst**2 / 4.0 / \
            (2.0**(1.0 / self.params.beta_hst) - 1.0)
        psf_hst = 1.0 / (1.0 + rsq / asq_hst)**self.params.beta_hst
        psf_hst = psf_hst / np.sum(psf_hst)

        listTransferKernel = []
        for fwhm in self.listFWHM:
            # Build MUSE FSF
            asq = fwhm**2 / 4.0 / (2.0**(1.0 / self.fsf_beta_muse) - 1.0)
            im_muse = 1.0 / (1.0 + rsq / asq)**self.fsf_beta_muse
            im_muse = im_muse / np.sum(im_muse)
            listTransferKernel.append(getBlurKernel(
                imHR=psf_hst, imLR=im_muse, sizeKer=(21, 21)))
        return listTransferKernel

    def _combineSpectra(self):
        """
        Combine spectra estimated on each HST image
        """
        weigthTot = np.sum([self.filtResp[j]
                            for j in range(len(self.filtResp))], axis=0)
        for i in range(self.nbSources):
            self.sources[i] = np.sum([
                self.filtResp[j] * self.tmp_sources[j][i]
                for j in range(len(self.filtResp))], axis=0) / weigthTot
            self.varSources[i] = np.sum([
                self.filtResp[j] * self.tmp_var[j][i]
                for j in range(len(self.filtResp))], axis=0) / weigthTot

        # for background, get voxel mean instead of sum
        self.sources[0] = self.sources[0] / self.cubeLR.size
        self.varSources[0] = self.varSources[0] / self.cubeLR.size

    def _rebuildCube(self, tmp_sources):
        """
        Create estimated cube. We have to work on each MUSE spectral bin as
        the spatial distribution is different on each bin
        """
        estimatedCube = np.zeros((self.cubeLR.shape[0],
                                  self.cubeLR.shape[1] * self.cubeLR.shape[2]))
        delta = int(self.cubeLR.shape[0] / float(self.nBands))

        for i in range(self.nBands):
            imin, imax = i * delta, (i + 1) * delta
            weightTot = np.sum([resp[imin:imax] for resp in self.filtResp],
                               axis=0)

            estim = []
            for j, resp in enumerate(self.filtResp):
                tmp = np.dot(tmp_sources[j][:, imin:imax].T,
                             self.listIntensityMapLRConvol[j][i])
                arr = (resp[imin:imax] / weightTot)[:, np.newaxis] * tmp
                estim.append(arr)

            estimatedCube[imin:imax, :] = np.sum(estim, axis=0)

        estimatedCube = estimatedCube.reshape(self.cubeLR.shape)
        return estimatedCube

    def _getResiduals(self):
        self.residuals = self.cubeLR - self.estimatedCube.data

    def _getContinuumCube(self, w=101):
        """
        Build continuum cube by median filtering (much faster here as it is
        done on objects spectra instead of all pixel spectra)
        """
        self.sourcesCont = ssl.medfilt(self.sources, kernel_size=(1, w))
        self.tmp_sourcesCont = [ssl.medfilt(tmp_source, kernel_size=(1, w))
                                for tmp_source in self.tmp_sources]
        self.estimatedCubeCont = self._rebuildCube(self.tmp_sourcesCont)

    def _getHST_ID(self):
        """
        Get the list of HST ids for each label of labelHR (first is
        background 'bg')
        """
        return ['bg'] + [int(self.segmap[self.labelHR == k][0])
                         for k in range(1, self.nbSources)]

    @property
    def Xi2_tot(self):
        return (1 / (np.size(self.residuals) - 3) *
                np.sum(self.residuals**2 / self.cubeLRVar))

    def calcXi2_source(self, k):
        mask = self.listIntensityMapLRConvol[0][0][k].reshape(self.shapeLR) > 0
        return (1 / (np.size(self.residuals[:, mask]) - 3) *
                np.sum(self.residuals[:, mask]**2 / self.cubeLRVar[:, mask]))

    def calcCondNumber(self, listobj=None):
        """Compute condition number."""
        if listobj is None:
            mat = np.array(self.listIntensityMapLRConvol[0][0][1:])
        else:
            mat = np.array(self.listIntensityMapLRConvol[0][0][listobj][1:])

        mat /= mat.sum(axis=1)[:, None]
        return np.linalg.cond(mat)

    def write(self, outfile, listObjInBlob, listHSTObjInBlob, group):
        origin = ('Odhin', '1.0-beta2', self.cube.filename,
                  self.cube.primary_header.get('CUBE_V', ''))
        src = Source.from_data(group.GID, 0, 0, origin=origin)

        cond_number = self.calcCondNumber(listObjInBlob)
        src.header['GRP_ID'] = group.GID
        src.header['GRP_AREA'] = group.region.area
        src.header['GRP_NSRC'] = group.nbSources
        src.header['COND_NB'] = cond_number
        src.header['XI2_TOT'] = self.Xi2_tot

        # add spectra, but remove spectra from objects not in the blob
        for k, iden in enumerate(self.listHST_ID):
            if iden == 'bg' or iden in listHSTObjInBlob:
                sp = Spectrum(data=self.sources[k], var=self.varSources[k],
                              wave=self.cube.wave, copy=False)
                src.spectra[iden] = sp

        # build sources table
        rows = [(self.listHST_ID[k], group.GID, self.calcXi2_source(k))
                for k in listObjInBlob]
        t = Table(rows=rows, names=('ID', 'G_ID', 'Xi2'))
        t['Group Area'] = group.region.area
        t['Number Sources'] = group.nbSources
        t['Condition Number'] = cond_number
        t['Xi2 Group'] = self.Xi2_tot
        src.tables['sources'] = t

        # save cubes
        src.cubes['orig'] = self.cube
        src.cubes['estim'] = self.estimatedCube

        src.write(outfile)
