# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 07:06:46 2016

@author: raphael
"""

from astropy.io import fits as pyfits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io
plt.rcParams['image.cmap']="coolwarm"
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
from matplotlib.colors import LogNorm
plt.rcParams['figure.figsize']=(15,15)
from mpdaf.obj import Cube,Image
from mpdaf.sdetect import Source
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot,iplot_mpl

init_notebook_mode()
from skimage.measure import regionprops
from plotly import *
from mpdaf.sdetect import Catalog
from astropy.table import Table
import scipy.signal as ssl
import os, sys
sys.path.append('../../lib/deblending/')
import main_deblending
import simuDeblending
import math
from math import pi
from plotly.graph_objs import *

from scipy.interpolate import interp1d
from deblend_utils import *

def convertFilt(filt,cube=None,x=None):
    """
    Resample response of HST filter on the spectral grid of MUSE
    
    filt: 2d array (wavelength,amplitude) of response filter
    cube: mpdaf cube, provides the wave sampling grid
    """
    if cube is not None:
        x = cube.wave.coord()
    f=interp1d(filt[:,0],filt[:,1],fill_value=0,bounds_error=False)
    return np.array(f(x))





def getSourceIso():
    #load sources
    listSources_iso=['00015','00016','00017','00030','00231','00246','00340']
    listSrc_iso=[Source.from_file('/home/data/MUSE/UDF/UDF-10/udf10_c031_e020/udf_udf10_%s.fits'%listSources_iso[k]) for k in xrange(len(listSources_iso))]
    
    
    beta=2.8 
    a=0.885
    b=-3.39*10**(-5)

    listPSF=calcFSF(a,b,beta,[4800, 5300, 5800,6300,6800,7300,7800,8300,8800,9300])
    
    filterdir = '../../lib/fsf_estimation/'
    hst606_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F606W_81.dat')
    hst606_resp = convertFilt(hst606_resp_orig, listSrc_iso[0].cubes['MUSE_CUBE'])
    hst775_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F775W_81.dat')
    hst775_resp = convertFilt(hst775_resp_orig, listSrc_iso[0].cubes['MUSE_CUBE'])
    hst814_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F814W_81.dat')
    hst814_resp = convertFilt(hst814_resp_orig, listSrc_iso[0].cubes['MUSE_CUBE'])
    hst850lp_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F850LP_81.dat')
    hst850lp_resp = convertFilt(hst850lp_resp_orig, listSrc_iso[0].cubes['MUSE_CUBE'])
    listResp=[hst606_resp,hst775_resp,hst814_resp,hst850lp_resp]
    print "Doing deblending..."
    
    listDebl_iso=[]    
    for src in listSrc_iso:
        listImHR=[src.images['HST_F606W'],src.images['HST_F775W'],src.images['HST_F814W'],src.images['HST_F850LP']]
        cubeLR=src.cubes['MUSE_CUBE']
        #debl=main_deblending.Deblending(cubeLR.data.filled(0)/np.sqrt(cubeLR.var),[imHR],None,listPSF)
        debl=main_deblending.Deblending(cubeLR.data.filled(0),listImHR,listResp,listPSF)
        debl.createAbundanceMap(thresh=None,segmap=src.images['HST_SEGMAP'].data,background=True)
        debl.findSources()
        listDebl_iso.append(debl)
    return listSrc_iso,listDebl_iso
        
def plotSourcesIso(listSrc_iso,listDebl_iso2):
    listCenter=[1,4,2,4,3,2,6]
    for k,debl in enumerate(listDebl_iso2):
        print "Objet %s"%listSrc_iso[k].ID
        plt.imshow(debl.labelHR)
        for region in regionprops(debl.labelHR):
            plt.annotate(
                region.label-1, 
                xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()
        trace0 = Scatter(
      Scatter(y=listSrc_iso[k].spectra['MUSE_WHITE'].data.data, name='Spectre source MUSE_WHITE')
        )
        trace1 = Scatter(
      Scatter(y=debl.spectraTot[listCenter[k]], name='Spectre reconstruit')
        )
        trace2 = Scatter(
      Scatter(y=listSrc_iso[k].spectra['MUSE_WHITE_SKYSUB'].data.data, name='Spectre source MUSE_WHITE_SKYSUB')
        )
        data = Data([trace0, trace1,trace2])
        iplot(data)
    
    
def getMergedSources():
    listSources=['00262','00255','00287','00467','00299','00542','00422','00405','00033','00018','00208']
    listSrc=[Source.from_file('/home/data/MUSE/UDF/UDF-10/udf10_c031_e020/udf_udf10_%s.fits'%listSources[k]) for k in xrange(len(listSources))]

    cat=Catalog.read('/home/data/MUSE/UDF/musecat_udf_10_c021_e017_full.vot',format='votable')
    listSources_periph=[]
    for src in listSrc:
        cat_tmp=cat.select(src.images['MUSE_WHITE'].wcs)
        for ID in cat_tmp['ID'].data.data:
            listSources_periph.append(ID)
    listSrc_periph=[Source.from_file('/home/data/MUSE/UDF/UDF-10/udf10_c031_e020/udf_udf10_%s.fits'%str(listSources_periph[k]).zfill(5)) for k in xrange(len(listSources_periph))]
    
    for src in listSrc:
        src.cubes['MUSE_CUBE_RC']=src.cubes['MUSE_CUBE'].clone()
        src.cubes['MUSE_CUBE_RC'].data=src.cubes['MUSE_CUBE'].data-ssl.medfilt(src.cubes['MUSE_CUBE'].data.filled(np.median(src.cubes['MUSE_CUBE'].data)),(101,1,1))
    

    #do deblending
    beta=2.8 
    a=0.885
    b=-3.39*10**(-5)
    listPSF=calcFSF(a,b,beta,np.arange(4800,9300,100))

    filterdir = '../../lib/fsf_estimation/'
    hst606_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F606W_81.dat')
    hst606_resp = convertFilt(hst606_resp_orig, listSrc[0].cubes['MUSE_CUBE'])
    hst775_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F775W_81.dat')
    hst775_resp = convertFilt(hst775_resp_orig, listSrc[0].cubes['MUSE_CUBE'])
    hst814_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F814W_81.dat')
    hst814_resp = convertFilt(hst814_resp_orig, listSrc[0].cubes['MUSE_CUBE'])
    hst850lp_resp_orig = np.loadtxt(filterdir+'/'+ 'HST_ACS_WFC.F850LP_81.dat')
    hst850lp_resp = convertFilt(hst850lp_resp_orig, listSrc[0].cubes['MUSE_CUBE'])

    listDebl=[]
    listResp=[hst606_resp,hst775_resp,hst814_resp,hst850lp_resp]
    for src in listSrc:
        print src.ID
        listImHR=[src.images['HST_F606W'],src.images['HST_F775W'],src.images['HST_F814W'],src.images['HST_F850LP']]
        cubeLR=src.cubes['MUSE_CUBE']
        
        debl=main_deblending.Deblending(cubeLR.data.filled(0),listImHR,filtResp=listResp,listPSF=listPSF,wcs=cubeLR.wcs)
        debl.createAbundanceMap(thresh=None,segmap=src.images['HST_SEGMAP'].data,background=True)
        debl.findSources(r,lmbda=0)
        listDebl.append(debl)


    return listSrc,listDebl

def plotInfoMerged(listSrc):
    for src in listSrc:
        fig,ax=plt.subplots(1,4)
        src.images['MUSE_WHITE'].plot(ax=ax[0])
        src.images['HST_F606W'].plot(ax=ax[1],zscale=True)
        ax[1].contour(src.images['HST_SEGMAP'].data>0,1)
        src.images['HST_F775W'].plot(ax=ax[2],zscale=True)
        ax[2].contour(src.images['HST_SEGMAP'].data>0,1)
        src.images['HST_F814W'].plot(ax=ax[3],zscale=True)
        #src.images['HST_SEGMAP'].plot(ax=ax[3],colorbar='v')
        ax[3].contour(src.images['HST_SEGMAP'].data>0,1)

def plotMerged(listSrc,listDebl):
    for k in xrange(len(listSrc)):
        print k
        plt.subplot(121)
        listSrc[k].images['MUSE_WHITE'].plot(zscale=True)
        plt.subplot(122)
        plt.imshow(listDebl[k].labelHR)
        for region in regionprops(listDebl[k].labelHR):
            plt.annotate(
                region.label-1, 
                xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()
        l=np.max(listDebl[k].labelHR)
        fig = tools.make_subplots(rows=l+1,shared_xaxes=True,print_grid=False)
        fig['layout'].update(height=1500, width=1000)
        fig.append_trace(Scatter(y=listSrc[k].spectra['MUSE_TOT'].data, name='Spectre source MUSE'), 1, 1)
        for i in xrange(0,l):
            fig.append_trace(Scatter(y=listDebl[k].sources[i], name='Objet %s'%i), i+2, 1)
        iplot(fig)    

def plotDetails(debl,listI,l1,l2,t=0.1,usePlotly=False):
    src=debl.src
    plt.subplot(131)
    plt.imshow(src.images['MUSE_WHITE'].data)
    plt.title('Muse white')
    plt.subplot(132)
    plt.title('HST F775')
    listHST_ID,listMUSE_ID=debl.getMUSE_ID()
    src.images['HST_F775W'].plot(zscale=True)
    for region in regionprops(debl.labelHR):
        plt.annotate(
        listHST_ID[region.label-1], 
        xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.subplot(133)
    plt.title('HST Segmap')
    plt.imshow(debl.labelHR)
    
    for region in regionprops(debl.labelHR):
        plt.annotate(
            listMUSE_ID[region.label-1], 
            xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()
    
    listHST_ID,listMUSE_ID=src.getMUSE_ID()
    if usePlotly==True:
        layout = {
		
		'shapes': [
		    # 1st highlight
		    {
		        'type': 'rect',
		        # x-reference is assigned to the x-values
		        'xref': 'x',
		        # y-reference is assigned to the plot paper [0,1]
		        'yref': 'paper',
		        'x0': l1-8,
		        'y0': 0,
		        'x1': l1+8,
		        'y1': 1,
		        'fillcolor': '#000000',
		        'opacity': 0.3,
		        'line': {
		            'width': 0,
		        }
		    },
		    # 2nd highlight during Feb 20 - Feb 23
		    {
		        'type': 'rect',
		        'xref': 'x',
		        'yref': 'paper',
		        'x0': l2-8,
		        'y0': 0,
		        'x1': l2+8,
		        'y1': 1,
		        'fillcolor': '#000000',
		        'opacity': 0.3,
		        'line': {
		            'width': 0,
		        }
		    }
		]
	}
        fig = tools.make_subplots(rows=len(listI)+1,shared_xaxes=True,shared_yaxes=True,print_grid=False)
        fig['layout'].update(height=1500, width=1000)
        fig['layout']['shapes']=layout['shapes']
        fig.append_trace(Scatter(x=np.arange(4750,9351,1.25),y=src.spectra['MUSE_TOT_SKYSUB'].data, name='Spectre source MUSE'), 1, 1)
        for i in xrange(0,len(listI)):
            fig.append_trace(Scatter(x=np.arange(4750,9351,1.25),y=debl.spectraTot[listI[i]], name='Source %s'%listMUSE_ID[listI[i]]), i+2, 1)
            iplot(fig)
    else:
        l=len(listI)+1
        f, axarr = plt.subplots(l, sharex=True)
        axarr[0].set_title('Spectre source MUSE')
        axarr[0].plot(np.arange(4750,9351,1.25),src.spectra['MUSE_TOT_SKYSUB'].data)
        for i in xrange(0,l):
            axarr[i+1].set_title('Objet %s'%listMUSE_ID[listI(i)])
            axarr[i+1].plot(np.arange(4750,9350,1.25),debl.spectraTot[listI(i)])

    src.add_narrow_band_image_lbdaobs(src.cubes['MUSE_CUBE'],tag='NB0',lbda=l1)
    src.add_narrow_band_image_lbdaobs(src.cubes['MUSE_CUBE'],tag='NB1',lbda=l2)

    plt.subplot(131)
    plt.title("White Image")
    plt.imshow(src.images['MUSE_WHITE'].data)
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.subplot(132)
    plt.title("NB 1")
    plt.imshow(src.images['NB0'].data)
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.subplot(133)
    plt.title("NB 2 ?")
    plt.imshow(src.images['NB1'].data)
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.show()
    plt.subplot(131)
    plt.title(u"White Image estimée")
    plt.imshow(np.sum(debl.cubeRebuilt,axis=0))
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.subplot(132)
    plt.title(u"NB 1 estimée")
    plt.imshow(np.sum(debl.cubeRebuilt[wave.pixel(l1-8,nearest=True):wave.pixel(l1+8,nearest=True)],axis=0))
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.subplot(133)
    plt.title(u"NB 2 estimée")
    plt.imshow(np.sum(debl.cubeRebuilt[wave.pixel(l2-8,nearest=True):wave.pixel(l2+8,nearest=True)],axis=0))
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')

    plt.show()
    plt.subplot(131)
    plt.title(u"White Image estimée")
    plt.imshow(np.sum(debl.cubeRebuilt,axis=0))
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.subplot(132)
    plt.title(u"NB 1 residus")
    plt.imshow(np.sum(np.abs(debl.residus[wave.pixel(l1-8,nearest=True):wave.pixel(l1+8,nearest=True)]),axis=0))
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')
    plt.subplot(133)
    plt.title(u"NB 2 residus")
    plt.imshow(np.sum(np.abs(debl.residus[wave.pixel(l2-8,nearest=True):wave.pixel(l2+8,nearest=True)]),axis=0))
    for i in listI:
        plt.contour((debl.listAbundanceMapLRConvol[0][0][i].reshape(25,25))>t,1,colors='r')

def plotAll(debl,usePlotly=False):   
    src=debl.src

    plt.subplot(131)
    plt.imshow(src.images['MUSE_WHITE'].data)
    plt.title('Muse white')
    plt.subplot(132)
    plt.title('HST F775')
    listHST_ID,listMUSE_ID=debl.getMUSE_ID()
    src.images['HST_F775W'].plot(zscale=True)
    for region in regionprops(debl.labelHR):
        plt.annotate(
        listHST_ID[region.label-1], 
        xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.subplot(133)
    plt.title('HST Segmap')
    plt.imshow(debl.labelHR)
    
    for region in regionprops(debl.labelHR):
        plt.annotate(
            listMUSE_ID[region.label-1], 
            xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()
    
    
    l=np.max(debl.labelHR)
    if usePlotly==True:
        fig = tools.make_subplots(rows=l+1,shared_xaxes=True,print_grid=False)
        fig['layout'].update(height=2000, width=1000)
        fig.append_trace(Scatter(x=np.arange(4750,9351,1.25),y=src.spectra['MUSE_TOT_SKYSUB'].data, name='Spectre source MUSE'), 1, 1)
        for i in xrange(0,l):
            fig.append_trace(Scatter(x=np.arange(4750,9351,1.25),y=debl.spectraTot[i], name='Objet %s'%listMUSE_ID[i]), i+2, 1)
        iplot(fig)
    else:
        f, axarr = plt.subplots(l+1, sharex=True)
        axarr[0].set_title('Spectre source MUSE')
        axarr[0].plot(np.arange(4750,9351,1.25),src.spectra['MUSE_TOT_SKYSUB'].data)
        for i in xrange(0,l):
            axarr[i+1].set_title('Objet %s'%listMUSE_ID[i])
            axarr[i+1].plot(np.arange(4750,9351,1.25),debl.spectraTot[i])
