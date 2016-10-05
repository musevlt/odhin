# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 16:31:40 2016

@author: raphael
"""
import matplotlib.pyplot as plt
import mpdaf
import logging
from mpdaf.obj import Cube,Image,Spectrum
import scipy.signal as ssl
try:#only needed for test purposes
    from downsampling import downsampling
    from OMPv2 import orthogonal_mp
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
from deblend_utils import convertFilt,calcFSF,apply_resampling_window,normalize

DEFAULT_HSTFILTER606 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F606W_81.dat')
DEFAULT_HSTFILTER775 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F775W_81.dat')
DEFAULT_HSTFILTER814 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F814W_81.dat')
DEFAULT_HSTFILTER850 = os.path.join(os.path.dirname(os.path.abspath(__file__)),'HST_ACS_WFC.F850LP_81.dat')


class Deblending():
    
    def __init__(self,src=None,cubeLR=None,listImagesHR=None,wcs=None,wave=None,\
    listFiltName=[DEFAULT_HSTFILTER606,DEFAULT_HSTFILTER775,DEFAULT_HSTFILTER814,DEFAULT_HSTFILTER850],\
    listPSF=None,simu=False,cubeLRVar=None):
        """
        src: MUSE Source
        if src these are optional:
            cubeLR: data from a MUSE cube (numpy array, lbda x n1 x n2)
            listImagesHR: list of HST images (mpdaf image) 
            wcs: wcs of MUSE data
        listFiltName: list of filenames of HST response filters
        listPSF: list of PSF images (numpy arrays) by ascendant lbdas
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
        if src is not None:
            self.cubeLR=src.cubes['MUSE_CUBE'].data.filled(np.nanmedian(src.cubes['MUSE_CUBE'].data))
            self.cubeLRVar=src.cubes['MUSE_CUBE'].var.filled(np.nanmedian(src.cubes['MUSE_CUBE'].var))
            self.wcs=src.cubes['MUSE_CUBE'].wcs
            self.wave=src.cubes['MUSE_CUBE'].wave
            self.listImagesHR=[src.images['HST_F606W'],src.images['HST_F775W'],src.images['HST_F814W'],src.images['HST_F850LP']]
            
        if cubeLR is not None:
            self.cubeLR=cubeLR
        if cubeLRVar is not None:
            self.cubeLRVar=cubeLRVar
        if wcs is not None:
            self.wcs=wcs
        if wave is not None:
            self.wave=wave
        if listImagesHR is not None:
            self.listImagesHR = listImagesHR
        if listFiltName is not None:
            
            self.filtResp=[convertFilt(np.loadtxt(filtName),self.wave) for filtName in listFiltName]
            
        else:
            self.filtResp=None
        self.listPSF=listPSF
        self.shapeHR=self.listImagesHR[0].shape
        self.shapeLR=self.cubeLR[0].shape
        self.residus=np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))
        self.cubeRebuilt=np.zeros((self.cubeLR.shape[0],self.cubeLR.shape[1]*self.cubeLR.shape[2]))            
            
        self.simu=simu
        
    
    def createAbundanceMap(self,thresh=None,segmap=None,background=False,antialias=True):
        
        #labelisation
        self.segmap=segmap
        self.antialias=antialias
        self.labelHR=self.getLabel(self.listImagesHR[0].data,thresh,segmap)
        self.background=background
        if self.background==False:
            self.nbSources=np.max(self.labelHR)
        else:
            self.nbSources=np.max(self.labelHR)+1
        if self.simu==False:
            self.listHST_ID=self.getHST_ID()
        
        self.listAbundanceMapHR=[]
        if self.antialias==True or self.simu==True:
            self.listAbundanceMapLR=[]
        else:
            self.listAbundanceMapHRConvol=[]
        for j in xrange(len(self.listImagesHR)):
            abundanceMapHR=np.zeros((self.nbSources,self.listImagesHR[0].shape[0]*self.listImagesHR[0].shape[1]))
            mask=np.zeros(self.listImagesHR[0].shape)
        
            if self.background==True:#if background is True, put abundanceMap of background in last position (estimated spectrum of background will also be last)
                mask[:]=1.
                abundanceMapHR[np.max(self.labelHR)]=mask.copy().flatten()
                mask[:]=0            
            
            for k in xrange(1,np.max(self.labelHR)+1):
                mask[self.labelHR==k]=np.maximum(self.listImagesHR[j].data[self.labelHR==k],0)#avoid negative abundances
                
                mask=mask/np.max(mask)
                abundanceMapHR[k-1]=mask.copy().flatten() # k-1 to manage the background that have label 0 but is pushed to the last position
                mask[:]=0            
            
            self.listAbundanceMapHR.append(abundanceMapHR)
            if self.simu==False:
                if antialias==True:
                    self.listAbundanceMapLR.append(self.abundanceDownsampling(abundanceMapHR,self.shapeHR,self.shapeLR,antialias))
                    
            else:
                self.listAbundanceMapLR.append(self.abundanceDownsampling2(abundanceMapHR,self.shapeHR,self.shapeLR))
        
        
    def findSources(self,getVar=True,nbIter=0,U=None,newEstim=False):
        """
        getVar: bool, get variance of spectra estimates
        nbIter: number of iteration of estimations
        U: spatial support of abundances (for new estimations using OMP)
        newEstim: if antialias is True,estimate again spectra using newly estimated abundances
        """
        
        self.getVar=getVar
        self.sources=np.zeros((self.nbSources,self.cubeLR.shape[0]))
        self.varSources=np.zeros((self.nbSources,self.cubeLR.shape[0]))
        self.listAbundanceMapLRConvol=[]
        self.tmp_sources=[]
        self.tmp_var=[]

        
        if self.antialias==False:
            self.listAbundanceMapHRConvol=[]
        niter=0
        #If there are several HR images the process is applied on each image and then the estimated spectra
        #are combined using a mean weighted by the response filters
        for j in xrange(len(self.listImagesHR)):
            self.tmp_sources.append(np.zeros_like(self.sources))
            self.tmp_var.append(np.zeros_like(self.sources))
            
            self.listAbundanceMapLRConvol.append([])
            if self.antialias==False:
                self.listAbundanceMapHRConvol.append([])
                
            delta=int(self.cubeLR.shape[0]/float(len(self.listPSF)))
            for i in xrange(len(self.listPSF)):
                if self.antialias==True or self.simu==True:
                    abundanceMapLRConvol=np.zeros_like(self.listAbundanceMapLR[j])
                    for k,im in enumerate(self.listAbundanceMapLR[j]):
                        abundanceMapLRConvol[k] =ssl.fftconvolve(self.listAbundanceMapLR[j][k].reshape(self.shapeLR),self.listPSF[i],mode='same').flatten()
                else:
                    abundanceMapHRConvol=np.zeros_like(self.listAbundanceMapHR[j])
                    for k,im in enumerate(self.listAbundanceMapHR[j]):
                        abundanceMapHRConvol[k]=ssl.fftconvolve(self.listAbundanceMapHR[j][k].reshape(self.shapeHR),self.listPSF[i],mode='same').flatten()
                    self.listAbundanceMapHRConvol[j].append(abundanceMapHRConvol)
                    abundanceMapLRConvol=self.abundanceDownsampling(abundanceMapHRConvol,self.shapeHR,self.shapeLR,self.antialias)
                
                
                A=abundanceMapLRConvol.T
                B=self.cubeLR[i*delta:(i+1)*delta].reshape(\
                    self.cubeLR[i*delta:(i+1)*delta].shape[0],\
                    self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                if getVar==True:
                    var=self.cubeLRVar[i*delta:(i+1)*delta].reshape(\
                    self.cubeLRVar[i*delta:(i+1)*delta].shape[0],\
                    self.cubeLRVar.shape[1]*self.cubeLRVar.shape[2]).T
                
                if self.antialias==True:#apply low pass filter
                    B=np.hstack([apply_resampling_window(B[:,l].reshape(self.shapeLR)).flatten()[:,np.newaxis] for l in xrange(B.shape[1]) ])
                
                if getVar==False:
                    self.tmp_sources[j][:,i*delta:(i+1)*delta]=np.linalg.lstsq(A,B)[0]

                else:
                    inv=np.linalg.pinv(A)                    
                    self.tmp_sources[j][:,i*delta:(i+1)*delta]=np.dot(inv,B)
                    self.tmp_var[j][:,i*delta:(i+1)*delta]=np.array([np.sum(inv**2*col,axis=1) for col in var.T]).T
                
                
                if self.antialias==True:#we have to recompute the abundance map by inversing again the system
                    
                    B=self.cubeLR[i*delta:(i+1)*delta].reshape(\
                        self.cubeLR[i*delta:(i+1)*delta].shape[0],\
                        self.cubeLR.shape[1]*self.cubeLR.shape[2])
                    A=self.tmp_sources[j][:,i*delta:(i+1)*delta].T

                    abundanceMapLRConvol=np.linalg.lstsq(A,B)[0]                        
                                    
                    #alternative : OMP step
                    #if U is None:
                    #    U=self.listAbundanceMapLR[j]>0.001
                    #A=normalize(A,axis=0)
                    #abundanceMapLRConvol=orthogonal_mp(A, B, n_nonzero_coefs=3,U=U)
                                    
                    if newEstim==True:
                        #new estimation step
                        A=abundanceMapLRConvol.T
                        B=self.cubeLR[i*delta:(i+1)*delta].reshape(\
                            self.cubeLR[i*delta:(i+1)*delta].shape[0],\
                            self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                        if getVar==True:
                            var=self.cubeLRVar[i*delta:(i+1)*delta].reshape(\
                            self.cubeLRVar[i*delta:(i+1)*delta].shape[0],\
                            self.cubeLRVar.shape[1]*self.cubeLRVar.shape[2]).T
                        if getVar==False:
                            self.tmp_sources[j][:,i*delta:(i+1)*delta]=np.linalg.lstsq(A,B)[0]
                                
                        else:
                            inv=np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)
                            self.tmp_sources[j][:,i*delta:(i+1)*delta]=np.dot(inv,B)
                            self.tmp_var[j][:,i*delta:(i+1)*delta]=np.array([np.sum(inv**2*col,axis=1) for col in var.T]).T


                while niter<nbIter:
                    B=self.cubeLR[i*delta:(i+1)*delta].reshape(\
                        self.cubeLR[i*delta:(i+1)*delta].shape[0],\
                        self.cubeLR.shape[1]*self.cubeLR.shape[2])
                    A=self.tmp_sources[j][:,i*delta:(i+1)*delta].T
                    abundanceMapLRConvol=np.linalg.lstsq(A,B)[0]
                    #alternative : OMP step
                    #if U is None:
                    #    try:
                    #        U=self.listAbundanceMapLR[j]>0.001
                    #    except:
                    #        U=abundanceMapLRConvol>0.001
                    #A=normalize(A,axis=0)
                    #abundanceMapLRConvol=orthogonal_mp(A, B, n_nonzero_coefs=3,U=U)
                    
                    #new estimation step
                    A=abundanceMapLRConvol.T
                    B=self.cubeLR[i*delta:(i+1)*delta].reshape(\
                        self.cubeLR[i*delta:(i+1)*delta].shape[0],\
                        self.cubeLR.shape[1]*self.cubeLR.shape[2]).T
                    if getVar==True:
                        var=self.cubeLRVar[i*delta:(i+1)*delta].reshape(\
                        self.cubeLRVar[i*delta:(i+1)*delta].shape[0],\
                        self.cubeLRVar.shape[1]*self.cubeLRVar.shape[2]).T
                    if getVar==False:
                        self.tmp_sources[j][:,i*delta:(i+1)*delta]=np.linalg.lstsq(A,B)[0]
                            
                    else:
                        inv=np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)
                        self.tmp_sources[j][:,i*delta:(i+1)*delta]=np.dot(inv,B)
                        self.tmp_var[j][:,i*delta:(i+1)*delta]=np.array([np.sum(inv**2*col,axis=1) for col in var.T]).T
                    
                    niter=niter+1

                self.listAbundanceMapLRConvol[j].append(abundanceMapLRConvol)
                    
        self.combineSpectra()        
        
        self.rebuildCube()
        self.getResidus()
        self.getTotSpectra()

    
    def combineSpectra(self):
        if self.filtResp is not None:
            weigthTot=np.sum([self.filtResp[j] for j in xrange(len(self.filtResp))],axis=0)
            for i in xrange(self.nbSources):
                self.sources[i]=np.sum([self.filtResp[j]*self.tmp_sources[j][i] for j in xrange(len(self.filtResp))],axis=0)/weigthTot
                if self.getVar==True:
                    self.varSources[i]=np.sum([self.filtResp[j]*self.tmp_var[j][i] for j in xrange(len(self.filtResp))],axis=0)/weigthTot
        else:
            self.sources=self.tmp_sources[0]
            if self.getVar==True:
                self.varSources=self.tmp_var[0]
                
            
    def getTotSpectra(self):
        self.spectraTot=np.zeros((self.nbSources,self.cubeLR.shape[0]))
        self.varSpectraTot=np.zeros((self.nbSources,self.cubeLR.shape[0]))
        
        delta=self.cubeLR.shape[0]/float(len(self.listPSF))
        if self.filtResp is None:
            for i in xrange(len(self.listPSF)):
                for k in xrange(self.nbSources):
                    self.spectraTot[k,int(i*delta):int((i+1)*delta)]=self.sources[k,int(i*delta):int((i+1)*delta)].T*np.sum(self.listAbundanceMapLRConvol[0][i][k])
                    if self.getVar==True:                        
                        self.varSpectraTot[k,int(i*delta):int((i+1)*delta)]=self.varSources[k,int(i*delta):int((i+1)*delta)].T*np.sum(self.listAbundanceMapLRConvol[0][i][k])**2
        
        else:
            tmp=[]
            tmpVar=[]
            for j in xrange(len(self.filtResp)):
                tmp.append(np.zeros_like(self.spectraTot))
                tmpVar.append(np.zeros_like(self.spectraTot))
                for i in xrange(len(self.listPSF)):
                    for k in xrange(self.nbSources):
                        tmp[j][k,int(i*delta):int((i+1)*delta)]=np.sum(np.outer(self.tmp_sources[j][k,int(i*delta):int((i+1)*delta)],self.listAbundanceMapLRConvol[j][i][k]),axis=1)
                        if self.getVar==True:
                            tmpVar[j][k,int(i*delta):int((i+1)*delta)]=self.tmp_var[j][k,int(i*delta):int((i+1)*delta)]*np.sum(self.listAbundanceMapLRConvol[j][i][k])**2
            self.spectraTot=np.sum([self.filtResp[l]*tmp[l] for l in xrange(len(self.filtResp))],axis=0)/np.sum([self.filtResp[l] for l in xrange(len(self.filtResp))],axis=0)
            if self.getVar==True:
                self.varSpectraTot=np.sum([self.filtResp[l]*tmpVar[l] for l in xrange(len(self.filtResp))],axis=0)/np.sum([self.filtResp[l] for l in xrange(len(self.filtResp))],axis=0)
                    
                            

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
            for k in sorted(set(segmap.flatten().astype(int))):
                label_image[segmap==k]=i
                i=i+1
    
        return label_image
    

    def abundanceDownsampling(self,abundanceMap,shapeHR,shapeLR,antialias):
        abundanceMapLR=np.zeros((self.nbSources,shapeLR[0]*shapeLR[1]))
        imHST=self.listImagesHR[0].clone()
        if self.wcs is not None:
            start=self.wcs.get_start()
        else:
            start=imHST.get_start()
        for k,im in enumerate(abundanceMap):
            imHST.data=im.reshape(shapeHR).copy()
            abundanceMapLR[k]=imHST.resample(shapeLR,start , newstep=0.2,flux=True,antialias=antialias).data.flatten()
        
        return abundanceMapLR        
        
    def abundanceDownsampling2(self,abundanceMap,shapeHR,shapeLR):
        abundanceMapLR=np.zeros((self.nbSources,shapeLR[0]*shapeLR[1]))
        for k,im in enumerate(abundanceMap):
            abundanceMapLR[k]=downsampling(abundanceMap[k].reshape(shapeHR),shapeLR).flatten()
        return abundanceMapLR

    
    def getSamplingMatrix(self,shapeHR,shapeLR):        
        _,S,Mh,Mv=downsampling(np.ones(shapeHR),shapeLR,returnMatrix=True)
        return S,Mh,Mv


    def getHST_ID(self):
        if self.background==True:
            listHST_ID = [int(self.segmap[self.labelHR==k][0]) for k in xrange(1,self.nbSources)]
        else:
            listHST_ID = [int(self.segmap[self.labelHR==k][0]) for k in xrange(1,self.nbSources+1)]
        return listHST_ID
        
    def getsp(self):
        cat = {}
        for k,key in enumerate(self.listHST_ID):
            if self.getVar==True:
                cat[key] = Spectrum(data=self.spectraTot[k], var=self.varSpectraTot[k],wave=self.src.spectra['MUSE_TOT'].wave)
            else:
                cat[key] = Spectrum(data=self.spectraTot[k],wave=self.src.spectra['MUSE_TOT'].wave)
        if self.background==True:
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
