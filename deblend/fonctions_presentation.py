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
from deblend import main_deblending, deblend_utils
try:
    from plotly.offline import download_plotlyjs, init_notebook_mode, iplot,iplot_mpl
    from plotly import *
    init_notebook_mode()
    from plotly.graph_objs import *
    from deblend_utils import *
except:
    pass

from skimage.measure import regionprops
from mpdaf.sdetect import Catalog
from astropy.table import Table
import math
from math import pi


from scipy.interpolate import interp1d
import scipy.signal as ssl
import matplotlib.patches as patches


def plotMainDeblend(debl,imMUSE,tab=None):
    
    ID=debl.src.ID
    print '-'*40
    print "\n Objet %s \n"%ID
    if tab is not None:
        tab[tab['ID']==ID].pprint(max_width=-1)

    if debl.background==True:
        listHST_ID= [int(debl.segmap[debl.labelHR==k][0]) for k in xrange(1,debl.nbSources)]
    else:
        listHST_ID= [int(debl.segmap[debl.labelHR==k][0]) for k in xrange(1,debl.nbSources+1)]
    
    listNeighbors=debl.src.RAF_ID.split(",")
    listNeighbors=[int(c) for c in listNeighbors]
    
    
    #get central source
    if debl.segmap[debl.shapeHR[0]/2,debl.shapeHR[0]/2]==0:
        HST_ID_center=listNeighbors[0]
    else:
        HST_ID_center=debl.segmap[debl.shapeHR[0]/2,debl.shapeHR[0]/2]
    
    #remove source if not close to central source 
    otherNeighbors=[]
    for HST_ID in listHST_ID:     
        if (np.sum(debl.segmap[30:-30,30:-30]==HST_ID)>0) and HST_ID not in listNeighbors:
            otherNeighbors.append(HST_ID)
    
    fig,ax=plt.subplots(1,3)
    imMUSE.plot(zscale=True,ax=ax[0])
    start=imMUSE.wcs.sky2pix(debl.src.images['MASK_OBJ'].wcs.get_start())[0]
    rect = patches.Rectangle((start[1],start[0]),debl.src.images['MASK_OBJ'].shape[0],\
                             debl.src.images['MASK_OBJ'].shape[1],linewidth=1,edgecolor='r',facecolor='none')
    ax[0].add_patch(rect)
    
    ax[1].imshow(debl.labelHR)
    for region in regionprops(debl.labelHR):
        ax[1].annotate(
            listHST_ID[region.label-1],
            xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    debl.src.images['HST_F775W'].plot(zscale=True,ax=ax[2])
    plt.show()
    
#    plt.figure(figsize=(10,5))
#    plt.subplot(121)
#    plt.title("Mask obj")
#    debl.src.images['MASK_OBJ'].plot()
#    plt.subplot(122)
#    plt.title("Abundance map")
#    plt.imshow(debl.listAbundanceMapLRConvol[0][0][listHST_ID.index(HST_ID_center)].reshape(debl.shapeLR))
#    plt.show()
    plt.subplot(131)
    plt.title("White image MUSE")
    plt.imshow(np.sum(debl.cubeLR,axis=0))
    plt.colorbar(fraction=0.046)
    plt.subplot(132)
    plt.title("White image estimated")
    plt.imshow(np.sum(debl.cubeRebuilt,axis=0))
    plt.colorbar(fraction=0.046)
    plt.subplot(133)
    plt.title("Residuals")
    plt.imshow(np.sum(debl.residus,axis=0),vmax=500,vmin=-500)
    plt.colorbar(fraction=0.046)
    plt.show()
    
    
    listNB=[im.replace('NB_','') for im in debl.src.images.keys() if "NB" in im]
    listNB=[im for im in listNB if (im in debl.src.lines['LINE']) or (im=='OII3727')]
    listNB=listNB[:5]#limit number of NB images
    listLines=[]
    for line in listNB:
        if line=='OII3727':#inconsistency between NB and lines
            listLines.append(debl.src.lines[debl.src.lines['LINE']=="OII3726"]['LBDA_OBS'][0])
        else:
            listLines.append(debl.src.lines[debl.src.lines['LINE']==line]['LBDA_OBS'][0])
    listNB=[x for (y, x) in sorted(zip(listLines, listNB))]
    listLines=sorted(listLines)
    debl.src.lines['LBDA_OBS']
    
    l_main=[x for (y, x) in sorted([(row['SNR'],row['LBDA_OBS']) for row in debl.src.lines])][-1]

    fig,ax=plt.subplots(1,3, gridspec_kw = {'width_ratios':[3, 1,1]},figsize=(15,4))
    debl.src.spectra['MUSE_PSF_SKYSUB'].plot(label='MUSE_PSF_SKYSUB',ax=ax[0])
    for line in listLines:
        ax[0].axvline(line,linestyle='--',c='r')
    
    debl.src.spectra['MUSE_PSF_SKYSUB'].subspec(l_main-10,l_main+10).plot(ax=ax[1],label='MUSE_PSF_SKYSUB')
    
    spectraTot=debl.getsp()
    for i in listNeighbors:
        spectraTot[i].plot(ax=ax[0],label=i)
        ax[0].legend()
        for line in listLines:
            ax[0].axvline(line,linestyle='--',c='r')
        spectraTot[i].subspec(l_main-10,l_main+10).plot(ax=ax[1],label=i)
    ax[2].imshow(debl.labelHR)
    for region in regionprops(debl.labelHR):
        if listHST_ID[region.label-1] in listNeighbors:
            ax[2].annotate(
                listHST_ID[region.label-1],
                xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()
    
    if len(otherNeighbors)>0:
        fig,ax=plt.subplots(len(otherNeighbors),3, gridspec_kw = {'width_ratios':[3, 1,1]},figsize=(15,4*len(otherNeighbors)))
        for _,i in enumerate(otherNeighbors):
            if len(otherNeighbors)>1:
                spectraTot[i].plot(ax=ax[_,0],label=i)
                for line in listLines:
                    ax[_,0].axvline(line,linestyle='--',c='r')
                    ax[_,0].legend()
                spectraTot[i].subspec(l_main-10,l_main+10).plot(ax=ax[_,1],label=i)
                ax[_,2].imshow(debl.labelHR)
                for region in regionprops(debl.labelHR):
                    if listHST_ID[region.label-1]==i:
                        ax[_,2].annotate(
                            listHST_ID[region.label-1],
                            xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            else:
                spectraTot[i].plot(ax=ax[0],label=i)
                for line in listLines:
                    ax[0].axvline(line,linestyle='--',c='r')
                ax[0].legend()
                ax[2].imshow(debl.labelHR)
                for region in regionprops(debl.labelHR):
                    if listHST_ID[region.label-1]==i:
                        ax[2].annotate(
                            listHST_ID[region.label-1],
                            xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.show()
            
    if len(listNB)>0:
        if len(listNB)<3:
            plt.figure(figsize=(5,5))
        else:
            plt.figure(figsize=(10,5))
        for i in xrange(len(listNB)):
            l=debl.wave.pixel(listLines[i],nearest=True)
            plt.subplot(2,len(listNB),i+1)
            plt.title('NB_'+listNB[i])
            debl.src.images['NB_'+listNB[i]].plot(zscale=True)
            plt.subplot(2,len(listNB),1+i+len(listNB))
            plt.title('NB_'+listNB[i])
            plt.imshow(np.sum(debl.cubeRebuilt[l-5:l+5],axis=0)-np.sum(ssl.medfilt(debl.cubeRebuilt[l-100:l+101],kernel_size=(101,1,1))[95:105],axis=0))
            
        plt.tight_layout()
        plt.show()    





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