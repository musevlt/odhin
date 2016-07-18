# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:31:40 2016

@author: raphael
"""

from mpdaf.obj import Cube,Image,Spectrum
from mpdaf.sdetect import Catalog
import scipy.signal as ssl
try:#only needed for test purposes
    from downsampling import downsampling
except:
    pass
import numpy as np
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
import scipy
from scipy.interpolate import interp1d
import os
from deblend_utils import convertFilt,calcFSF

DEFAULT_HSTFILTER606 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F606W_81.dat')
DEFAULT_HSTFILTER775 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F775W_81.dat')
DEFAULT_HSTFILTER814 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F814W_81.dat')
DEFAULT_HSTFILTER850 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F850LP_81.dat')


class Deblending():
    
    def __init__(self,src=None,cubeLR=None,listImagesHR=None,wcs=None,\
    listFiltName=[DEFAULT_HSTFILTER606,DEFAULT_HSTFILTER775,DEFAULT_HSTFILTER814,DEFAULT_HSTFILTER850],\
    listPSF=None,cat=None,simu=False):
        """
        src: MUSE Source
        if src these are optional:
            cubeLR: data from a MUSE cube (numpy array, lbda x n1 x n2)
            listImagesHR: list of HST images (mpdaf image) 
            wcs: wcs of MUSE data
        listFiltName: list of filenames of HST response filters
        listPSF: list of PSF images (numpy arrays) by ascendant lbdas
        cat: catalogue of source (mpdaf Catalog object) to make the link between between HST and MUSE objetcs ID
        simu: bool to indicate if process data is simulated (no real mpdaf source)
        
        ---------
        Attributes:
        ---------
        listAbundanceMap (HR,LR,LRConvol) : list of Abundance Map of each object detected, at each step : High resolution,
                                            after subsampling, and after convolution
        sources: list of spectra estimated during the process (non calibrated)
        spectraTot: list of recovered spectra calibrated in flux
        listHST_ID,listMUSE_ID: list of ID in the same order than the returned spectra
        cubeRebuilt: estimated cube
        residus: residus (data cube - estimated cube)
        
        """
        self.src=src
        if src:
            self.cubeLR=src.cubes['MUSE_CUBE'].data.filled(np.median(src.cubes['MUSE_CUBE'].data))
            self.wcs=src.cubes['MUSE_CUBE'].wcs
            self.listImagesHR=[src.images['HST_F606W'],src.images['HST_F775W'],src.images['HST_F814W'],src.images['HST_F850LP']]
            
        if cubeLR:
            self.cubeLR=cubeLR
        if wcs:
            self.wcs=wcs
        if listImagesHR:
            self.listImagesHR = listImagesHR

        self.filtResp=[convertFilt(np.loadtxt(filtName),src.cubes['MUSE_CUBE']) for filtName in listFiltName]
        
        self.listPSF=listPSF
        self.shapeHR=self.listImagesHR[0].shape
        self.shapeLR=self.cubeLR[0].shape
        self.residus=np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        self.cubeRebuilt=np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        self.simu=simu
        
            
    def createAbundanceMap(self,thresh=None,segmap=None,background=False):
        
        #labelisation
        
        self.labelHR=self.getLabel(self.listImagesHR[0].data,thresh,segmap)
        self.background=background
        if self.background==False:
            self.nbSources=np.max(self.labelHR)
        else:
            self.nbSources=np.max(self.labelHR)+1

        self.listAbundanceMapHR=[]
        self.listAbundanceMapLR=[]
        for j in xrange(len(self.listImagesHR)):
            abundanceMapHR=np.zeros((self.nbSources,self.listImagesHR[0].shape[0]*self.listImagesHR[0].shape[1]))
            mask=np.zeros(self.listImagesHR[0].shape)
        
            if self.background==True:#if background is True, put abundanceMap of background in last position (estimated spectrum of background will also be last)
                mask[:]=1.
                abundanceMapHR[np.max(self.labelHR)]=mask.copy().flatten()
                mask[:]=0            
            
            for k in xrange(1,np.max(self.labelHR)+1):
                mask[self.labelHR==k]=self.listImagesHR[j].data[self.labelHR==k]
                
                mask=mask/np.max(mask)
                abundanceMapHR[k-1]=mask.copy().flatten() # k-1 to manage the background that have label 0 but is pushed to the last position
                mask[:]=0            
            
            self.listAbundanceMapHR.append(abundanceMapHR)
            if self.simu==False:
                self.listAbundanceMapLR.append(self.abundanceDownsampling(abundanceMapHR,self.shapeHR,self.shapeLR))
            else:
                self.listAbundanceMapLR.append(self.abundanceDownsampling2(abundanceMapHR,self.shapeHR,self.shapeLR))
                
            self.listHST_ID = self.getMUSE_ID()
        
        
    def findSources(self,weight=None, regMatrix=None,lmbda=None,r=10):
        """
        weight: matrix of weights (inverse of variance )
        regMatrix,lmbda,r : used only for regularization tries
        
        """
        
        
        self.sources=np.zeros((self.nbSources,self.cubeLR.shape[0]))
        self.listAbundanceMapLRConvol=[]
        self.tmp_sources=[]

        #If there are several HR images the process is applied on each image and then the estimated spectra
        #are combined using a mean weighted by the response filters
        for j in xrange(len(self.listImagesHR)):
            self.tmp_sources.append(np.zeros_like(self.sources))
            self.listAbundanceMapLRConvol.append([])
            
            
            delta=self.cubeLR.shape[0]/float(len(self.listPSF))
            for i in xrange(len(self.listPSF)):
                abundanceMapLRConvol=np.zeros_like(self.listAbundanceMapLR[j])
                for k,im in enumerate(self.listAbundanceMapLR[j]):
                    
                    abundanceMapLRConvol[k] =ssl.fftconvolve(self.listAbundanceMapLR[j][k].reshape(self.shapeLR),self.listPSF[i],mode='same').flatten()
                self.listAbundanceMapLRConvol[j].append(abundanceMapLRConvol)
                
                if regMatrix is None:
                    if weight is not None:
                        A=np.sqrt(weight)[:,np.newaxis]*abundanceMapLRConvol.T
                        B=np.sqrt(weight)[:,np.newaxis]*self.cubeLR[int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].reshape(\
                            self.cubeLR[int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].shape[0],\
                            self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                    else:
                        A=abundanceMapLRConvol.T
                        B=self.cubeLR[int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].reshape(\
                            self.cubeLR[int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].shape[0],\
                            self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                    self.tmp_sources[j][:,int(max(0,i*delta-r/2.)):int((i+1)*delta+r)]=np.linalg.lstsq(A,B)[0]
                
                else: #regularize with regMatrix
                    self.tmp_sources[j][:,int(max(0,i*delta-r/2.)):int((i+1)*delta+r)]=\
                        scipy.linalg.solve_sylvester(\
                        np.dot(abundanceMapLRConvol,abundanceMapLRConvol.T),\
                        lmbda*np.dot(LSF[int(max(0,i*delta-r/2.)):int((i+1)*delta+r),int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].T,\
                        LSF[int(max(0,i*delta-r/2.)):int((i+1)*delta+r),int(max(0,i*delta-r/2.)):int((i+1)*delta+r)]),\
                        np.dot(abundanceMapLRConvol,self.cubeLR[int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].reshape(\
                        self.cubeLR[int(max(0,i*delta-r/2.)):int((i+1)*delta+r)].shape[0],\
                        self.cubeLR.shape[1]*self.cubeLR.shape[2]).T))
                    
                    
        self.combineSpectra()        
        
        self.rebuildCube()
        self.getResidus()
        self.getTotSpectra()

    
    def combineSpectra(self):
        if self.filtResp is not None:
            weigthTot=np.sum([self.filtResp[j] for j in xrange(len(self.filtResp))],axis=0)
            for i in xrange(self.nbSources):
                self.sources[i]=np.sum([self.filtResp[j]*self.tmp_sources[j][i] for j in xrange(len(self.filtResp))],axis=0)/weigthTot
        else:
            self.sources=self.tmp_sources[0]
            
    def getTotSpectra(self):
        self.spectraTot=np.zeros((self.nbSources,self.cubeLR.shape[0]))
        delta=self.cubeLR.shape[0]/float(len(self.listPSF))
        if self.filtResp is None:
            for i in xrange(len(self.listPSF)):
                for k in xrange(self.nbSources):
                    self.spectraTot[k,int(i*delta):int((i+1)*delta)]=np.sum(np.outer(self.sources[k,int(i*delta):int((i+1)*delta)].T,self.listAbundanceMapLRConvol[0][i][k]),axis=1)
        
        else:
            tmp=[]
            for j in xrange(len(self.filtResp)):
                tmp.append(np.zeros_like(self.spectraTot))
                for i in xrange(len(self.listPSF)):
                    for k in xrange(self.nbSources):
                        tmp[j][k,int(i*delta):int((i+1)*delta)]=np.sum(np.outer(self.tmp_sources[j][k,int(i*delta):int((i+1)*delta)],self.listAbundanceMapLRConvol[j][i][k]),axis=1)
            self.spectraTot=np.sum([self.filtResp[l]*tmp[l] for l in xrange(len(self.filtResp))],axis=0)/np.sum([self.filtResp[l] for l in xrange(len(self.filtResp))],axis=0)
                    
                            

    def rebuildCube(self):
        self.cubeRebuilt=np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        delta=self.cubeLR.shape[0]/float(len(self.listPSF))
        if self.filtResp is None:                    
            for i in xrange(len(self.listPSF)):
                self.cubeRebuilt[int(i*delta):int((i+1)*delta),:]=np.dot(self.sources[:,int(i*delta):int((i+1)*delta)].T,self.listAbundanceMapLRConvol[0][i])
            self.cubeRebuilt=self.cubeRebuilt.reshape(self.cubeLR.shape)

        else:
            
            for i in xrange(len(self.listPSF)):
                tmp=[]
                weights=np.sum([self.filtResp[l][int(i*delta):int((i+1)*delta)] for l in xrange(len(self.filtResp))],axis=0)
                for j in xrange(len(self.filtResp)):
                    tmp.append(np.zeros_like(self.cubeRebuilt[int(i*delta):int((i+1)*delta),:]))                    
                    tmp[j]=np.dot(self.tmp_sources[j][:,int(i*delta):int((i+1)*delta)].T,self.listAbundanceMapLRConvol[j][i])
                self.cubeRebuilt[int(i*delta):int((i+1)*delta),:]=np.sum([(self.filtResp[l][int(i*delta):int((i+1)*delta)]/weights)[:,np.newaxis]*tmp[l] for l in xrange(len(self.filtResp))],axis=0)
            self.cubeRebuilt=self.cubeRebuilt.reshape(self.cubeLR.shape)
    
    def getResidus(self):
        self.residus=self.cubeLR - self.cubeRebuilt
        
        
    def getLabel(self,image,thresh,segmap):
        
        
        if segmap is None:  # apply threshold      
            bw = closing(image > thresh, square(2))
            label_image=label(bw)
            for region in regionprops(label_image):
            
                # skip small images
                if region.area < 3:
                    label_image[label_image==region.label]=0
    
        else: #exploit HST segmap
            label_image=np.zeros(segmap.shape,dtype='int')
            i=0
            for k in xrange(int(np.max(segmap)+1)):
                if np.sum(segmap==k)>0:
                    label_image[segmap==k]=i
                    i=i+1
    
        return label_image
    

    def abundanceDownsampling(self,abundanceMap,shapeHR,shapeLR):
        abundanceMapLR=np.zeros((self.nbSources,shapeLR[0]*shapeLR[1]))
        imHST=self.listImagesHR[0].clone()
        if self.wcs is not None:
            start=self.wcs.get_start()
        else:
            start=imHST.get_start()
        for k,im in enumerate(abundanceMap):
            imHST.data=im.reshape(shapeHR).copy()
            abundanceMapLR[k]=imHST.resample(shapeLR,start , newstep=0.2,flux=True).data.flatten()
        return abundanceMapLR        
        
    def abundanceDownsampling2(self,abundanceMap,shapeHR,shapeLR):
        abundanceMapLR=np.zeros((self.nbSources,shapeLR[0]*shapeLR[1]))
        for k,im in enumerate(abundanceMap):
            abundanceMapLR[k]=downsampling(abundanceMap[k].reshape(shapeHR),shapeLR).flatten()
        return abundanceMapLR

    
    def getSamplingMatrix(self,shapeHR,shapeLR):        
        _,S,Mh,Mv=downsampling(np.ones(shapeHR),shapeLR,returnMatrix=True)
        return S,Mh,Mv


    def getMUSE_ID(self):
        listHST_ID = [int(self.src.images['HST_SEGMAP'].data[self.labelHR==k][0]) for k in xrange(1,self.nbSources)]
        return listHST_ID
    
    def getsp(self):
        cat = {}
        for k,key in enumerate(self.listHST_ID):
            cat[key] = Spectrum(data=self.spectraTot[k], wave=self.src.spectra['MUSE_TOT'].wave)
        cat['bg'] = Spectrum(data=self.spectraTot[k-1], wave=self.src.spectra['MUSE_TOT'].wave)
        return cat
        
if __name__ == '__main__':
    from muse_analysis.udf import UDFSource
    src = UDFSource.from_file('/Users/rolandbacon/Dropbox/MUSE/GTO/UDF/Sources/udf-10/deblend_test/udf_udf10_00062.fits')
    fullcat = Catalog.read('/Users/rolandbacon/UDF/Sources/udf-10/0.31/udf10_c031_e021.vot') 
    beta=2.8
    a=0.885
    b=-3.39e-5
    listPSF = calcFSF(a, b, beta,np.linspace(4800,9300,10))
    debl = Deblending(src, listPSF=listPSF)
    debl.createAbundanceMap(segmap=src.images['HST_SEGMAP'].data, background=True)
    debl.findSources()
    cat = debl.getsp()
    print len(cat)    
    print 'end'