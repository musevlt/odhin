# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""

import astropy.units as units
import logging
import numpy as np
import scipy.signal as ssl

from astropy.table import Table
from mpdaf.obj import Spectrum
from mpdaf.sdetect import Source
from scipy.ndimage import median_filter

from .regularization import regulDeblendFunc
from .parameters import Params
from .deblend_utils import (convertFilt, convertIntensityMap,
                            getMainSupport, generatePSF_HST, getBlurKernel)
from .version import __version__


def deblendGroup(subcube, subhstimages, subsegmap, group, outfile):
    logger = logging.getLogger(__name__)
    logger.debug('group %d, start', group.GID)
    debl = Deblending(subcube, subhstimages)
    logger.debug('group %d, createIntensityMap', group.GID)
    debl.createIntensityMap(subsegmap.data.filled(0.))
    logger.debug('group %d, findSources', group.GID)
    debl.findSources()
    logger.debug('group %d, write', group.GID)
    debl.write(outfile, group)
    logger.debug('group %d, done', group.GID)


class Deblending:
    """
    Main class for deblending process

    Parameters
    ----------
    cube : mpdaf Cube
        The Cube object to be deblended
    imagesHR:
        HR images
    listFiltName : list of filenames
        List of filenames of HST response filters.
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

    def __init__(self, cube, imagesHR, params=None):

        self.cube = cube
        if params is None:
            params = Params()
        self.params = params
        self.cubeLR = cube.data.filled(np.ma.median(cube.data))
        self.cubeLRVar = cube.var.filled(np.ma.median(cube.var))
        # self.wcs = cube.wcs
        # self.wave = cube.wave
        self.listImagesHR = imagesHR

        # get FSF parameters
        fsf_a = self.params.fsf_a_muse
        fsf_b = self.params.fsf_b_muse
        self.fsf_beta_muse = self.params.fsf_beta_muse

        self.nBands = self.params.nBands
        l_min, l_max = cube.wave.get_range()
        self.bands_wl = lb = np.linspace(l_min, l_max, self.nBands + 1)
        self.bands_center = np.mean([lb[:-1], lb[1:]], axis=0)
        self.listFWHM = fsf_a + fsf_b * self.bands_wl

        # FIXME: listFiltName should match the input HR images
        if self.params.listFiltName is not None:
            self.filtResp = [convertFilt(np.loadtxt(filtName), self.cube.wave)
                             for filtName in self.params.listFiltName]
        else:
            self.filtResp = [np.ones(self.cube.shape[0])] * 4

        # needeed for muse_analysis functions
        # FIXME: remove harcoded list here, is it still needed ?
        filters = ['f606w', 'f775w', 'f814w', 'f850lp']
        for k, im in enumerate(self.listImagesHR):
            im.primary_header['FILTER'] = filters[k]

        # spatial shapes
        self.shapeLR = self.cube.shape[1:]

        self.residuals = np.zeros((self.cube.shape[0], np.prod(self.shapeLR)))
        self.estimatedCube = self.cube.clone()
        self.PSF_HST = generatePSF_HST(self.params.alpha_hst,
                                       self.params.beta_hst)

        # for each HST band list all MUSE bands inside it
        self.listBands = self._getListBands(self.nBands, self.filtResp)

    def _getListBands(self, nBands, filtResp):
        """For each HST band, get the list of all MUSE bands inside it (if one
        of the band limits has filter values > 0).
        """
        listBands = []
        nl = len(filtResp[0])
        lind = list(np.linspace(0, nl-1, nBands+1, dtype=int))
        for i in range(len(filtResp)):
            val = filtResp[i][lind]
            bands_idx = np.where((val[:-1] > 0) | (val[1:] > 0))
            listBands.append(list(bands_idx[0]))
        return listBands

    def createIntensityMap(self, segmap, thresh=None):
        """
        Create intensity maps from HST images and segmentation map.
        To be called before calling findSources()

        Parameters
        ----------
        segmap : `ndarray`
            Segmentation map.
        thres : (not used if segmap)
            Threshold to use on HST image to segment.

        """
        # List of all HST ids in the segmap
        hst_ids = np.unique(segmap)
        self.listHST_ID = ['bg'] + sorted(hst_ids[hst_ids > 0])
        self.nbSources = len(self.listHST_ID)  # include background

        # for each HST filter, create the high resolution intensity matrix
        # (nbSources x Nb pixels)
        self.listIntensityMapHR = []  # [bands, array(sources, im.shape)]

        for im in self.listImagesHR:
            # clip image data to avoid negative abundances
            data = np.maximum(im.data, 10**(-9))

            intensityMapHR = np.zeros((self.nbSources, np.prod(data.shape)))
            # put intensityMap of background in first position (estimated
            # spectrum of background will also be first)
            intensityMapHR[0] = 1

            for k, hst_id in enumerate(self.listHST_ID[1:], start=1):
                mask = np.where(segmap == hst_id, data, 0)
                intensityMapHR[k] = mask.ravel()

            self.listIntensityMapHR.append(intensityMapHR)

    def findSources(self, store=False):
        """Main function to estimate sources spectra.

        Parameters
        ----------
        store : bool
            store intermediate results
        """
        regul = self.params.regul
        filt_w = self.params.filt_w
        if regul:
            # precompute continuum
            self.cubeLR_c = median_filter(self.cubeLR, size=(filt_w, 1, 1),
                                          mode='reflect')

        # compute HST-MUSE transfer functions for all MUSE FSF fwhm considered
        self.listTransferKernel = self._generateHSTMUSE_transfer_PSF()

        shapeLR = (self.nbSources, self.cubeLR.shape[0])

        # Lists of [HR band, Spectral band]
        nbImagesHR = len(self.listImagesHR)

        def _create_result_list():
            return [[None] * self.nBands for _ in range(nbImagesHR)]

        tmp_sources = []
        tmp_var = []
        self.listIntensityMapLRConvol = _create_result_list()
        self.listAlphas = _create_result_list()
        self.listRSS = _create_result_list()
        self.listCorrFlux = _create_result_list()

        if store:
            self.listMask = _create_result_list()
            self.listccoeff = _create_result_list()
            self.listlcoeff = _create_result_list()
            self.listY = _create_result_list()
            self.listYc = _create_result_list()
            self.listYl = _create_result_list()
            self.spatialMask = _create_result_list()

        # If there are several HR images the process is applied on each image
        # and then the estimated spectra are combined using a mean weighted by
        # the response filters
        for j in range(nbImagesHR):
            tmp_sources.append(np.zeros(shapeLR))
            tmp_var.append(np.zeros(shapeLR))

            delta = int(self.cubeLR.shape[0] / float(self.nBands))

            for i in range(self.nBands):
                imin, imax = i * delta, (i + 1) * delta

                # Do the estimation only if MUSE band is in HST band
                if i in self.listBands[j]:
                    # Create intensity maps at MUSE resolution
                    intensityMapLRConvol = convertIntensityMap(
                        self.listIntensityMapHR[j],
                        self.cube[0],
                        self.listImagesHR[j],
                        self.listFWHM[i],
                        self.fsf_beta_muse,
                        self.listTransferKernel[i]
                    )

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
                    Y = self.cubeLR[imin:imax].reshape(delta, -1).T
                    if regul:
                        Y_c = self.cubeLR_c[imin:imax].reshape(delta, -1).T
                    Yvar = self.cubeLRVar[imin:imax].reshape(delta, -1).T

                    # normalize intensity maps in flux to get flux-calibrated
                    # estimated spectra
                    U /= np.sum(U, axis=0)

                    if regul:  # apply regularization

                        # remove background from intensity matrix as
                        # intercept is used instead
                        U_ = U[:, 1:]

                        # generate support: U.shape = (image size, nsources)
                        support = np.zeros(U.shape[0], dtype=bool)
                        for u in range(U_.shape[1]):
                            support[U_[:, u] > 0.1 * np.max(U_[:, u])] = True

                        # Y_sig2 = np.var(Y[~support, :], axis=0)
                        Y_sig2 = np.mean(Yvar, axis=0)

                        res = regulDeblendFunc(U_, Y, Y_c=Y_c, support=support,
                                               Y_sig2=Y_sig2, filt_w=filt_w)
                        # res -> (res, intercepts, listMask, c_coeff, l_coeff,
                        #         Y, Y_l, Y_c, c_alphas, listRSS, listA)

                        # get spectra estimation
                        tmp_sources[j][1:, imin:imax] = res[0]
                        # for background spectrum get intercept (multiply by
                        # number of pixels to get tot flux)
                        tmp_sources[j][0, imin:imax] = res[1] * U.shape[0]

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
                        tmp_sources[j][:, imin:imax] = np.linalg.lstsq(U, Y)[0]

                    # get spectra variance : as spectra is obtained by
                    # (U^T.U)^(-1).U^T.Y
                    # variance of estimated spectra is obtained by
                    # (U^T.U)^(-1).Yvar
                    Uinv = np.linalg.pinv(U)
                    tmp_var[j][:, imin:imax] = np.dot(Uinv**2, Yvar)

                    self.listIntensityMapLRConvol[j][i] = intensityMapLRConvol
                else:
                    self.listIntensityMapLRConvol[j][i] = np.zeros(
                        (self.nbSources, self.shapeLR[0] * self.shapeLR[1]))

        self._combineSpectra(tmp_sources, tmp_var)
        self.estimatedCube.data = self._rebuildCube(tmp_sources)
        self._getContinuumCube(tmp_sources)
        self._getResiduals()

    def _generateHSTMUSE_transfer_PSF(self):
        """Generate HST to MUSE transfer PSF, for each spectral band."""
        hst = self.listImagesHR[0]
        dy, dx = hst.get_step(unit=units.arcsec)

        # get odd shape
        shape_1 = np.array(hst.shape) // 2 * 2 + 1
        center = shape_1 // 2

        # Build "distances to center" matrix.
        ind = np.indices(shape_1)
        rsq = ((ind[0] - center[0]) * dx)**2 + (((ind[1] - center[1])) * dy)**2

        # Build HST FSF
        asq_hst = self.params.fwhm_hst**2 / 4.0 / \
            (2.0**(1.0 / self.params.beta_hst) - 1.0)
        psf_hst = 1.0 / (1.0 + rsq / asq_hst)**self.params.beta_hst
        psf_hst = psf_hst / np.sum(psf_hst)
        # FIXME: use Moffat(rsq, asq_hst, self.params.beta_hst) ?

        listTransferKernel = []
        for fwhm in self.listFWHM:
            # Build MUSE FSF
            asq = fwhm**2 / 4.0 / (2.0**(1.0 / self.fsf_beta_muse) - 1.0)
            im_muse = 1.0 / (1.0 + rsq / asq) ** self.fsf_beta_muse
            im_muse /= im_muse.sum()
            listTransferKernel.append(getBlurKernel(
                imHR=psf_hst, imLR=im_muse, sizeKer=(21, 21)))
        return listTransferKernel

    def _combineSpectra(self, tmp_sources, tmp_var):
        """Combine spectra estimated on each HST image."""
        filtResp = np.array(self.filtResp)
        weigthTot = np.sum(filtResp, axis=0)
        self.sources = np.sum(filtResp[:, None, :] * tmp_sources,
                              axis=0) / weigthTot
        self.varSources = np.sum(filtResp[:, None, :] * tmp_var,
                                 axis=0) / weigthTot

        # for background, get voxel mean instead of sum
        self.sources[0] /= self.cubeLR.size
        self.varSources[0] /= self.cubeLR.size

    def _rebuildCube(self, tmp_sources):
        """
        Create estimated cube. We have to work on each MUSE spectral bin as
        the spatial distribution is different on each bin
        """
        estimatedCube = np.zeros((self.cubeLR.shape[0], np.prod(self.shapeLR)))
        delta = int(self.cubeLR.shape[0] / float(self.nBands))
        filtResp = np.array(self.filtResp)
        filtResp /= filtResp.sum(axis=0)

        for i in range(self.nBands):
            imin, imax = i * delta, (i + 1) * delta

            estim = []
            for j, resp in enumerate(filtResp):
                arr = np.dot(tmp_sources[j][:, imin:imax].T,
                             self.listIntensityMapLRConvol[j][i])
                arr *= resp[imin:imax][:, np.newaxis]
                estim.append(arr)

            estimatedCube[imin:imax, :] = np.sum(estim, axis=0)

        estimatedCube = estimatedCube.reshape(self.cubeLR.shape)
        return estimatedCube

    def _getResiduals(self):
        self.residuals = self.cubeLR - self.estimatedCube.data

    def _getContinuumCube(self, tmp_sources, w=101):
        """
        Build continuum cube by median filtering (much faster here as it is
        done on objects spectra instead of all pixel spectra)
        """
        self.sourcesCont = ssl.medfilt(self.sources, kernel_size=(1, w))
        self.tmp_sourcesCont = [ssl.medfilt(tmp_source, kernel_size=(1, w))
                                for tmp_source in tmp_sources]
        self.estimatedCubeCont = self._rebuildCube(self.tmp_sourcesCont)

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

    def write(self, outfile, group):
        origin = ('Odhin', __version__, self.cube.filename,
                  self.cube.primary_header.get('CUBE_V', ''))
        src = Source.from_data(group.GID, 0, 0, origin=origin)

        cond_number = self.calcCondNumber(group.idxSources)
        src.header['GRP_ID'] = group.GID
        src.header['GRP_AREA'] = group.region.area
        src.header['GRP_NSRC'] = group.nbSources
        src.header['COND_NB'] = cond_number
        src.header['XI2_TOT'] = self.Xi2_tot

        # add spectra, but remove spectra from objects not in the blob
        for k, iden in enumerate(self.listHST_ID):
            if iden == 'bg' or iden in group.listSources:
                sp = Spectrum(data=self.sources[k], var=self.varSources[k],
                              wave=self.cube.wave, copy=False)
                src.spectra[iden] = sp

        # build sources table
        ids = [f'bg_{group.GID}' if id_ == 'bg' else id_
               for id_ in self.listHST_ID]
        rows = [(ids[k], group.GID, self.calcXi2_source(k))
                for k in group.idxSources]
        t = Table(rows=rows, names=('ID', 'G_ID', 'Xi2'))
        t['Group Area'] = group.region.area
        t['Number Sources'] = group.nbSources
        t['Condition Number'] = cond_number
        t['Xi2 Group'] = self.Xi2_tot
        src.tables['sources'] = t

        # save cubes
        src.cubes['MUSE'] = self.cube
        src.cubes['FITTED'] = self.estimatedCube
        src.images['MUSE_WHITE'] = self.cube.mean(axis=0)

        src.write(outfile)
