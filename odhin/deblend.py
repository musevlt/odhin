# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""

import astropy.units as units
import logging
import numpy as np
import scipy.signal as ssl
import yaml

from astropy.table import Table
from mpdaf.obj import Spectrum, Image, Cube
from mpdaf.sdetect import Source
from scipy.ndimage import median_filter

from .utils import (load_filter, convertIntensityMap, extractHST,
                    getMainSupport, generatePSF_HST, getBlurKernel)
from .parameters import Params
from .regularization import regulDeblendFunc
from .version import __version__

__all__ = ('Deblending', 'deblendGroup')


def deblendGroup(group, outfile, conf):
    """Deblend a given group."""
    logger = logging.getLogger(__name__)
    logger.debug('group %d, start, %d sources', group.ID, group.nbSources)
    debl = Deblending(group, conf)
    logger.debug('group %d, createIntensityMap', group.ID)
    debl.createIntensityMap()
    logger.debug('group %d, findSources', group.ID)
    debl.findSources()
    logger.debug('group %d, write', group.ID)
    debl.write(outfile, conf)
    logger.debug('group %d, done', group.ID)


class Deblending:
    """
    Main class for deblending process

    Parameters
    ----------
    cube : mpdaf Cube
        The Cube object to be deblended
    nbands : int
        The number of MUSE spectral bins to consider for computation.
        MUSE FSF is assumed constant within a bin.
    conf : dict
        The settings dict.

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

    def __init__(self, group, conf):
        self.params = Params(**conf.get('params', {}))
        self.group = group

        cube = Cube(conf['cube'])
        self.cube = cube = cube[:, group.region.sy, group.region.sx]
        im = cube[0]

        self.segmap = (extractHST(Image(conf['segmap']), im, integer_mode=True)
                       .data.filled(0))

        # List of all HST ids in the segmap
        listHST_ID = np.unique(self.segmap)
        listHST_ID = listHST_ID[listHST_ID > 0]
        self.listHST_ID = ['bg'] + list(listHST_ID)
        self.nbSources = len(self.listHST_ID)  # include background

        # spatial shapes
        self.nlbda = cube.shape[0]
        self.shapeLR = cube.shape[1:]

        # load the HR images and filters
        self.listImagesHR = []
        filtResp = []
        lbda = cube.wave.coord()
        for band, d in conf['hr_bands'].items():
            imhr = extractHST(Image(d['file']), im)
            # store the flux conversion factor in the header
            imhr.primary_header['photflam'] = d.get('photflam', 1)
            self.listImagesHR.append(imhr)

            if 'filter' in d:
                filt = load_filter(d['filter'], lbda)
            else:
                filt = np.ones(self.nlbda)
            filtResp.append(filt)
        self.filtResp = np.array(filtResp)

        self.nBands = self.params.nBands
        # compute bands limit indices
        idx = np.linspace(0, self.nlbda, self.nBands + 1, dtype=int)
        self.idxBands = np.array([idx[:-1], idx[1:]]).T

        # compute FWHM at the center of bands
        bands_center = self.cube.wave.coord(self.idxBands.mean(axis=1))
        self.listFWHM = (self.params.fsf_a_muse +
                         self.params.fsf_b_muse * bands_center)

        self.PSF_HST = generatePSF_HST(self.params.alpha_hst,
                                       self.params.beta_hst)

        # for each HST band list all MUSE bands inside it
        self.listBands = self._getListBands(self.nBands, self.filtResp)

    def _getListBands(self, nBands, filtResp):
        """For each HST band, get the list of all MUSE bands inside it (if one
        of the band limits has filter values > 0).
        """
        listBands = []
        nl = filtResp.shape[1]
        lind = list(np.linspace(0, nl - 1, nBands + 1, dtype=int))
        for filt in filtResp:
            mask = filt[lind] > 0
            # check if band limits have non-zero values
            bands_idx = np.where(mask[:-1] | mask[1:])
            listBands.append(list(bands_idx[0]))
        return listBands

    def createIntensityMap(self):
        """Create intensity maps from HST images and segmentation map.
        To be called before calling findSources().
        """
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
                mask = self.segmap == hst_id
                arr = np.where(mask, data, 0)
                intensityMapHR[k] = arr.ravel()

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
        cubeLR = self.cube.data.filled(np.ma.median(self.cube.data))
        cubeLRVar = self.cube.var.filled(np.ma.median(self.cube.var))

        if regul:
            # precompute continuum
            cubeLR_c = median_filter(cubeLR, size=(filt_w, 1, 1),
                                     mode='reflect')

        # compute HST-MUSE transfer functions for all MUSE FSF fwhm considered
        self.listTransferKernel = self._generateHSTMUSE_transfer_PSF()

        shapeLR = (self.nbSources, self.nlbda)

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

            for i, (imin, imax) in enumerate(self.idxBands):

                # Do the estimation only if MUSE band is in HST band
                if i in self.listBands[j]:
                    # Create intensity maps at MUSE resolution
                    intensityMapLRConvol = convertIntensityMap(
                        self.listIntensityMapHR[j],
                        self.cube[0],
                        self.listImagesHR[j],
                        self.listFWHM[i],
                        self.params.fsf_beta_muse,
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

                    delta = imax - imin
                    U = intensityMapLRConvol.T
                    Y = cubeLR[imin:imax].reshape(delta, -1).T
                    if regul:
                        Y_c = cubeLR_c[imin:imax].reshape(delta, -1).T
                    Yvar = cubeLRVar[imin:imax].reshape(delta, -1).T

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
        self._rebuildCube(tmp_sources)
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
        psf_hst /= psf_hst.sum()
        # FIXME: use Moffat(rsq, asq_hst, self.params.beta_hst) ?

        listTransferKernel = []
        for fwhm in self.listFWHM:
            # Build MUSE FSF
            asq = fwhm ** 2 / 4.0 / (
                2.0 ** (1.0 / self.params.fsf_beta_muse) - 1.0)
            im_muse = 1.0 / (1.0 + rsq / asq) ** self.params.fsf_beta_muse
            im_muse /= im_muse.sum()
            listTransferKernel.append(getBlurKernel(
                imHR=psf_hst, imLR=im_muse, sizeKer=(21, 21)))
        return listTransferKernel

    def _combineSpectra(self, tmp_sources, tmp_var):
        """Combine spectra estimated on each HST image."""
        weigthTot = np.ma.masked_values(self.filtResp.sum(axis=0), 0)
        self.sources = np.sum(self.filtResp[:, None, :] * tmp_sources,
                              axis=0) / weigthTot
        self.varSources = np.sum(self.filtResp[:, None, :] * tmp_var,
                                 axis=0) / weigthTot

        # for background, get voxel mean instead of sum
        self.sources[0] /= self.cube.data.size
        self.varSources[0] /= self.cube.data.size

    def _rebuildCube(self, tmp_sources):
        """Create the estimated cube.

        We have to work on each MUSE spectral bin as the spatial
        distribution is different on each bin.

        """
        estimatedCube = np.zeros((self.nlbda, np.prod(self.shapeLR)))
        weigthTot = np.ma.masked_values(self.filtResp.sum(axis=0), 0)
        filtResp = self.filtResp / weigthTot

        for i, (imin, imax) in enumerate(self.idxBands):
            estim = []
            for j, resp in enumerate(filtResp):
                arr = np.dot(tmp_sources[j][:, imin:imax].T,
                             self.listIntensityMapLRConvol[j][i])
                arr *= resp[imin:imax][:, np.newaxis]
                estim.append(arr)

            estimatedCube[imin:imax, :] = np.sum(estim, axis=0)

        self.estimatedCube = self.cube.clone()
        self.estimatedCube.data = estimatedCube.reshape(self.cube.shape)

    def _getResiduals(self):
        self.residuals = self.cube.data - self.estimatedCube.data

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
        return (1 / (self.residuals.size - 3) *
                np.sum(self.residuals**2 / self.cube.var))

    def calcXi2_source(self, k):
        mask = self.listIntensityMapLRConvol[0][0][k].reshape(self.shapeLR) > 0
        return (1 / (self.residuals[:, mask].size - 3) *
                np.sum(self.residuals[:, mask]**2 / self.cube.var[:, mask]))

    def calcCondNumber(self, listobj=None):
        """Compute condition number."""
        if listobj is None:
            mat = np.array(self.listIntensityMapLRConvol[0][0][1:])
        else:
            mat = np.array(self.listIntensityMapLRConvol[0][0][listobj][1:])

        mat /= mat.sum(axis=1)[:, None]
        return np.linalg.cond(mat)

    def write(self, outfile, conf):
        group = self.group
        origin = ('Odhin', __version__, self.cube.filename,
                  self.cube.primary_header.get('CUBE_V', ''))
        src = Source.from_data(group.ID, group.region.ra, group.region.dec,
                               origin=origin)

        idxSources = [k for k, iden in enumerate(self.listHST_ID)
                      if iden in group.listSources]
        cond_number = self.calcCondNumber(idxSources)
        src.header['GRP_ID'] = group.ID
        src.header['GRP_AREA'] = group.region.area
        src.header['GRP_NSRC'] = group.nbSources
        src.header['COND_NB'] = cond_number
        src.header['XI2_TOT'] = self.Xi2_tot

        # add spectra from objects in the blob
        for k, iden in enumerate(self.listHST_ID):
            if iden in group.listSources:
                sp = Spectrum(data=self.sources[k], var=self.varSources[k],
                              wave=self.cube.wave, copy=False)
                src.spectra[iden] = sp

        # build sources table
        ids = [f'bg_{group.ID}' if id_ == 'bg' else id_
               for id_ in self.listHST_ID]
        rows = [(ids[k], group.ID, self.calcXi2_source(k))
                for k in idxSources]
        t = Table(rows=rows, names=('id', 'group_id', 'xi2'))
        t['group_area'] = group.region.area
        t['nb_sources'] = group.nbSources
        t['condition_number'] = cond_number
        t['xi2_group'] = self.Xi2_tot
        src.tables['sources'] = t

        # save cubes
        src.cubes['MUSE'] = self.cube
        src.cubes['FITTED'] = self.estimatedCube
        src.images['MUSE_WHITE'] = self.cube.mean(axis=0)
        src.images['FITTED'] = self.estimatedCube.mean(axis=0)

        # save params
        src.header.add_comment('')
        src.header.add_comment('ODHIN PARAMETERS:')
        src.header.add_comment('')
        for line in yaml.dump(conf).splitlines():
            src.header.add_comment(line)

        src.write(outfile)
