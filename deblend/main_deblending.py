# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:31:40 2016

@author: raphael.bacher@gipsa-lab.fr
"""
import mpdaf
from mpdaf.obj import Cube,Image,Spectrum
import scipy.signal as ssl
import scipy.sparse.linalg as sla
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
import astropy.units as units
import astropy.io.fits as pyfits
from regularization import regulDeblendFunc
from deblend_utils import convertFilt, calcFSF, apply_resampling_window, normalize,\
                        getSpatialShift, generateMoffatIm, approxNNLS,\
                        convertIntensityMap, getMainSupport, ADMM_soft_neg, generatePSF_HST,GCV


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
        The number of MUSE spectral bins to consider for computation.
        MUSE FSF is assumed constant within a bin.


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

    listIntensityMap (HR,LR,LRConvol) : list
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


        # get FSF parameters (the keys depends from the mosaic orignal field)
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
        self.generateHSTMUSE_transfert_PSF()

    def generateHSTMUSE_transfert_PSF(self):
        muse=self.src.cubes['MUSE_CUBE'][0,:,:]
        margin=2
        dy, dx = muse.get_step(unit=units.arcsec)
        shape = np.asarray(muse.shape) + np.ceil(np.abs(
            margin / muse.get_step(unit=units.arcsec) ) ).astype(int)
        # Round the image dimensions up to integer powers of two, to
        # ensure that an efficient FFT implementation is used.

        shape = (2**np.ceil(np.log(shape)/np.log(2.0))).astype(int)

        # Extract the dimensions of the expanded Y and X axes.
        ny,nx = shape
        # Calculate the frequency interval of the FFTs along the
        # X and Y axes.

        dfx = 1.0 / (nx * dx)
        dfy = 1.0 / (ny * dy)

        # Get a 2D array of the radius of each image pixel center relative
        # to pixel [0,0] (the spatial origin of the FFT algorithm).

        rsq = np.fft.fftfreq(nx, dfx)**2 + \
              np.fft.fftfreq(ny, dfy)[np.newaxis,:].T**2

        for fwhm in self.listFWHM:
            asq = fwhm**2 / 4.0 / (2.0**(1.0 / self.betaFSF) - 1.0)
            dy, dx = muse.get_step(unit=units.arcsec)


            # Compute an image of a Moffat function centered at pixel 0,0.
            betaHST = 1.6
            fwhm_hst=0.085
            im = 1.0 / (1.0 + rsq / asq)**self.betaFSF

            asq_hst = fwhm_hst**2 / 4.0 / (2.0**(1.0 / betaHST) - 1.0)
            psf_hst = 1.0 / (1.0 + rsq / asq_hst)**betaHST

            tmp_dir='./tmp/'
            pyfits.writeto(tmp_dir+'wider.fits',im,overwrite=True)
            pyfits.writeto(tmp_dir+'sharper.fits',psf_hst,overwrite=True)
            os.system('export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH')
            os.system('astconvolve --kernel=%ssharper.fits --makekernel=%s %swider.fits --output=%skernel_%s.fits'%(tmp_dir,np.maximum(im.shape[0],im.shape[1])/2+1,tmp_dir,tmp_dir,fwhm))

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

    def createIntensityMap(self, segmap=None, thresh=None,  background=False):
        """
        Create intensity maps from HST images and segmentation map.


        """

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

        self.listIntensityMapHR = []

        # for each HST filter, create the high resolution intensity matrix (nbSources x Nb pixels )
        for j in xrange(len(self.listImagesHR)):
            intensityMapHR = np.zeros((self.nbSources,self.listImagesHR[0].shape[0]*self.listImagesHR[0].shape[1]))
            mask = np.zeros(self.listImagesHR[0].shape)

            if self.background is True:  # if background is True, put intensityMap of background in last position (estimated spectrum of background will also be last)
                mask[:] = 1.
                intensityMapHR[np.max(self.labelHR)] = mask.copy().flatten()
                mask[:] = 0

            for k in xrange(1,np.max(self.labelHR)+1):
                mask[self.labelHR == k] = np.maximum(self.listImagesHR[j].data[self.labelHR==k],10**(-9))#avoid negative abundances

                intensityMapHR[k-1] = mask.copy().flatten()  # k-1 to manage the background that have label 0 but is pushed to the last position
                mask[:] = 0

            self.listIntensityMapHR.append(intensityMapHR)

    def findSources(self,  hstpsf=True, antialias=True, nonneg=True,
                    alpha=2, regul=False):
        """
        hstpsf: apply hst psf on MUSE or not (if not the transfert function from HST to MUSE
                                              is estimated and used on HST images instead of MUSE PSF)
        antialias: apply antialias filter or not
        regul: add regularization or not
        alpha: regularization parameter
        """
        self.sources = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.varSources = np.zeros((self.nbSources, self.cubeLR.shape[0]))
        self.listIntensityMapLRConvol = []
        self.tmp_sources = []
        self.tmp_var = []
        self.listMask = []
        self.listAlphas = []
        self.listAlphasMin = []
        self.listccoeff = []
        self.listlcoeff = []
        self.listY = []
        self.listYc = []
        self.listYl = []
        self.spatialMask = []
        self.listRSS = []
        self.listSig2 = []


        # If there are several HR images the process is applied on each image
        # and then the estimated spectra are combined using a mean weighted by the response filters
        for j in xrange(len(self.listImagesHR)):
            self.listMask.append([])
            self.listAlphas.append([])
            self.listAlphasMin.append([])
            self.listccoeff.append([])
            self.listlcoeff.append([])
            self.listY.append([])
            self.listYc.append([])
            self.listYl.append([])
            self.spatialMask.append([])
            self.listRSS.append([])
            self.listSig2.append([])
            self.tmp_sources.append(np.zeros_like(self.sources))
            self.tmp_var.append(np.zeros_like(self.sources))

            self.listIntensityMapLRConvol.append([])
            self.listIntensityMapLRConvolClean.append([])

            delta = int(self.cubeLR.shape[0]/float(self.nBands))
            for i in xrange(self.nBands):
                self.listMask[j].append([])
                self.listAlphas[j].append([])
                self.listAlphasMin[j].append([])
                self.listccoeff[j].append([])
                self.listlcoeff[j].append([])
                self.listY[j].append([])
                self.listYc[j].append([])
                self.listYl[j].append([])
                self.spatialMask[j].append([])
                self.listRSS[j].append([])
                self.listSig2[j].append([])


                #Create intensity maps at MUSE resolution
                intensityMapLRConvol = convertIntensityMap(self.listIntensityMapHR[j],
                    self.src.cubes['MUSE_CUBE'][0,:,:], self.listImagesHR[j],self.listFWHM[i],
                    self.betaFSF, self.listShift[j], antialias=antialias,psf_hst=None)

                ## truncate intensity maps support after onvolution
                U = getMainSupport(intensityMapLRConvol[:-1], alpha=0.999)
                intensityMapLRConvol[:-1][~U] = 0

                if self.background is True: #put ones everywhere for background intensity map
                    intensityMapLRConvol[-1] = 1.

                A = intensityMapLRConvol.T
                Y = self.cubeLR[i*delta:(i+1)*delta].reshape(
                        self.cubeLR[i*delta:(i+1)*delta].shape[0],
                        self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                var = self.cubeLRVar[i*delta:(i+1)*delta].reshape(
                            self.cubeLRVar[i*delta:(i+1)*delta].shape[0],
                            self.cubeLRVar.shape[1]*self.cubeLRVar.shape[2]).T

                if antialias:
                    if hstpsf:
                        Y=np.hstack([apply_resampling_window(ssl.fftconvolve(Y[:,l].reshape(self.shapeLR),self.PSF_HST,mode='same')).flatten()[:,np.newaxis] for l in xrange(Y.shape[1]) ])
                        var=np.hstack([ssl.fftconvolve(var[:,l].reshape(self.shapeLR),self.PSF_HST**2,mode='same').flatten()[:,np.newaxis] for l in xrange(Y.shape[1]) ])
                    else:
                        Y=np.hstack([apply_resampling_window(Y[:,l].reshape(self.shapeLR)).flatten()[:,np.newaxis] for l in xrange(Y.shape[1]) ])
                else:
                    if hstpsf:
                        Y=np.hstack([ssl.fftconvolve(Y[:,l].reshape(self.shapeLR),self.PSF_HST,mode='same').flatten()[:,np.newaxis] for l in xrange(Y.shape[1]) ])

                #normalize intensity maps in flux to get flux-calibrated estimated spectra
                for a in xrange(A.shape[1]):
                    A[:,a]=A[:,a]/np.sum(A[:,a])



                if regul==True: #apply regularization
                    if self.background:
                        A_=A[:,:-1]
                    else:
                        A_=A
                    support=np.zeros(A.shape[0]).astype(bool)
                    for a in xrange(A_.shape[1]):
                         support[A_[:,a]>0.1*np.max(A_[:,a])]=True

                    Y_sig2=np.var(B[~support,:],axis=0)
                    res = regulDeblendFunc(A_, Y,
                                    mask=True,ng=50,split=True,two_steps=True,l_method='glasso_bic',
                                    c_method='gridge_cv', cv=None,maskOnly=True, corrflux=True,
                                    support=support,#smooth=np.array([[0.5,1,0.5]])/2.,
                                    intercept=True,Y_sig2=Y_sig2)


                    self.spatialMask[j][i]=support

                    # get
                    self.tmp_sources[j][:-1, i*delta:(i+1)*delta]=res[0]
                    self.tmp_sources[j][-1, i*delta:(i+1)*delta]=res[1]*A.shape[0]

                    #store all elements for checking purposes
                    self.listMask[j][i]=res[2]
                    self.listccoeff[j][i]=res[3]
                    self.listlcoeff[j][i]=res[4]
                    self.listY[j][i]=res[5]
                    self.listYl[j][i]=res[6]
                    self.listYc[j][i]=res[7]
                    self.listAlphas[j][i]=res[8]
                    self.listAlphasMin[j][i]=res[9]
                    self.listRSS[j][i]=res[10]
                    self.listSig2[j][i]=res[11]

                else:
                    self.tmp_sources[j][:, i*delta:(i+1)*delta] = np.linalg.lstsq(A, Y)[0]


                # get spectra variance :
                inv=np.linalg.pinv(A)
                self.tmp_var[j][:, i*delta:(i+1)*delta] = \
                            np.array([np.sum(inv**2*col, axis=1) for col in var.T]).T

                self.listIntensityMapLRConvol[j].append(intensityMapLRConvol)

        self.combineSpectra()
        self.cubeRebuilt = self.rebuildCube(self.sources)
        self.getContinuumCube()
        self.getResidus()
        self.getTotSpectra()

    def combineSpectra(self):
        """
        Combine spectra estimated on each HST image
        """
        weigthTot = np.sum([self.filtResp[j] for j in xrange(len(self.filtResp))], axis=0)
        for i in xrange(self.nbSources):
            self.sources[i] = np.sum([self.filtResp[j]*self.tmp_sources[j][i]
                for j in xrange(len(self.filtResp))], axis=0)/weigthTot
            self.varSources[i] = np.sum([self.filtResp[j]*self.tmp_var[j][i]
                for j in xrange(len(self.filtResp))], axis=0)/weigthTot


    def rebuildCube(self):
        """
        Create estimated cube
        """
        tmp_sources = self.tmp_sources

        cubeRebuilt = np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        delta = self.cubeLR.shape[0]/float(self.nBands)

        for i in xrange(self.nBands):
            tmp = []
            weightTot = np.sum([self.filtResp[l][int(i*delta):int((i+1)*delta)]
                            for l in xrange(len(self.filtResp))], axis=0)

            for j in xrange(len(self.filtResp)):
                tmp.append(np.zeros_like(cubeRebuilt[int(i*delta):int((i+1)*delta),:]))
                tmp[j] = np.dot(tmp_sources[j][:,int(i*delta):int((i+1)*delta)].T,self.listIntensityMapLRConvol[j][i])

            cubeRebuilt[int(i*delta):int((i+1)*delta),:] = np.sum(
                    [(self.filtResp[l][int(i*delta):int((i+1)*delta)]/weightTot)[:,np.newaxis]*tmp[l]
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
        """
        If no segmentation map create one by thresholding image (ndarray) by thres (float).
        If given segmap, create a new segmap with contiguous indices
        """
        if segmap is None:  # apply threshold
            bw = closing(image > thresh, square(2))
            label_image = label(bw)
            for region in regionprops(label_image):
                # skip small regions
                if region.area < 3:
                    label_image[label_image == region.label] = 0

        else:  # exploit HST segmap

            label_image = np.zeros(segmap.shape, dtype='int')
            i = 0
            for k in sorted(set(segmap.flatten())):

                if (np.sum(segmap[:2] == k)== np.sum(segmap == k)) or \
                     (np.sum(segmap[:,:2] == k)== np.sum(segmap == k)) or \
                    (np.sum(segmap[-2:] == k)== np.sum(segmap == k)) or \
                    (np.sum(segmap[:,-2:] == k)== np.sum(segmap == k)):
                    # remove objects on the border of the image that are less than two pixels wide
                    pass
                else:
                    label_image[segmap == k] = i
                    i = i+1

        return label_image



    def getHST_ID(self):
        """
        Get the list of HST ids for each label of labelHR
        """

        if self.background is True:
            listHST_ID = [int(self.segmap[self.labelHR == k][0]) for k in xrange(1, self.nbSources)]
        else:
            listHST_ID = [int(self.segmap[self.labelHR == k][0]) for k in xrange(1, self.nbSources+1)]
        return listHST_ID

    def getsp(self):
        """
        Get estimated spectra as a dict (with the HST ids as keys) of mpdaf Spectra
        """
        cat = {}
        for k, key in enumerate(self.listHST_ID):
            cat[key] = Spectrum(data=self.sources[k], var=self.varSources[k], wave=self.src.spectra['MUSE_TOT'].wave)

        if self.background is True:
            cat['bg'] = Spectrum(data=self.sources[k-1], wave=self.src.spectra['MUSE_TOT'].wave)
        return cat
